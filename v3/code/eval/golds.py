"""Gold loaders. load(name) -> Gold with .works {id: {title, abstract, venue, ...}}, .labels {id: {...}}, .split {id: train|val|test}, .cands (candidate id space note).
Label conventions: every candidate/gold id is already in the gold's comparison space (canon string for keyword golds, MeSH UI, GND code, Porter-stemmed phrase for KP20k/Inspec).
  judge      : labels = {chosen, relevant, none_fits, judged}   (keyword gold v1.1, #1285; keys canon()ed)
  mesh       : labels = {all, noct, major}, work.indexing_method (4,728 human-indexed PubMed works, #1306)
  akw_*      : labels = {akw} canon author keywords; work.today = today's keywords (canon)   (#1306 § 8 slices, held out)
  tibkat     : labels = {all} GND codes (2,000 English TIBKAT records, #1306 § 9)
  kp20k/inspec: labels = {present, absent, all} stemmed phrases (fetched by fetch_ext.py into data/)"""
import json, os, re, collections
from common import P, DATA, HERE, canon, jl
class Gold:
    def __init__(self, name): self.name=name; self.works={}; self.labels={}; self.split={}; self.cands=''
    def ids(self, split=None): return [i for i in self.works if split in (None,'all') or self.split.get(i)==split]
    def load_split(self):
        p=f'{HERE}/splits/{self.name}.json'
        if os.path.exists(p): self.split=json.load(open(p))
        return self
_mesh_cache={}
def mesh_desc():
    if not _mesh_cache:
        D={json.loads(l)['ui']: json.loads(l) for l in open(f"{P['j1306']}/mesh/descriptors.jsonl")}
        n2u={d['name']:u for u,d in D.items()}
        CT=['Humans','Animals','Male','Female','Pregnancy','Infant, Newborn','Infant','Child, Preschool','Child','Adolescent','Adult','Young Adult','Middle Aged','Aged','Aged, 80 and over','Mice','Rats','Rabbits','Dogs','Cats','Cattle','Swine','Guinea Pigs','Cricetinae','Chick Embryo','Sheep','Horses','Goats','Chickens','Haplorhini','Primates','Mice, Inbred C57BL','Mice, Inbred BALB C','Rats, Sprague-Dawley','Rats, Wistar','Mice, Knockout','Mice, Transgenic','Disease Models, Animal','Cell Line, Tumor','Cells, Cultured']
        _mesh_cache.update(D=D, CHECKTAGS={n2u[n] for n in CT if n in n2u}, PUBTYPE={u for u,d in D.items() if d['class']=='2'}, NAME2UI=n2u)
    return _mesh_cache
def _stem():
    from nltk.stem import PorterStemmer; ps=PorterStemmer()
    def f(s): return ' '.join(ps.stem(t) for t in re.sub(r'[^a-z0-9 ]+',' ', s.lower()).split())
    return f
def load(name):
    g=Gold(name)
    if name=='judge':
        for w in jl(f"{P['j1292']}/gold/works.jsonl"): g.works[w['id']]={'title':w['title'],'abstract':w.get('abstract'),'venue':w.get('journal'),'set':w['set'],'language':w.get('language'),'today':[canon(k) for k in (w.get('keywords') or [])]}
        pools=json.load(open(f"{P['j1292']}/judge/pools.json")); delta=json.load(open(f"{P['j1292']}/judge/delta_cands.json"))
        v12=f'{HERE}/../delta/judge/gold_delta.jsonl'; delta2=f'{HERE}/../delta/judge/delta_cands.json'   # gold v1.2 (#1322 delta-judge of the vocab_v1 arms) when built
        gold_path=v12 if os.path.exists(v12) else f"{P['j1292']}/judge/gold_delta.jsonl"; g.version='v1.2' if os.path.exists(v12) else 'v1.1'
        delta2=json.load(open(delta2)) if os.path.exists(v12) else {}
        for r in jl(gold_path):
            judged={canon(x['key']) for x in pools.get(r['id'],{}).get('pool',[])}|{canon(x['key']) for x in delta.get(r['id'],[])}|{canon(x['key']) for x in delta2.get(r['id'],[])}|{canon(k) for k in r['relevant']}
            g.labels[r['id']]={'chosen':{canon(k) for k in r['chosen']},'relevant':{canon(k) for k in r['relevant']},'none_fits':bool(r.get('none_fits')),'judged':judged}
        g.works={i:w for i,w in g.works.items() if i in g.labels}; g.cands='canon keyword (any vocabulary); unjudged share reported'
    elif name=='mesh':
        M=mesh_desc()
        for w in jl(f"{P['j1306']}/sample/works.jsonl"):
            G={h['ui'] for h in w['mesh']}
            g.works[w['id']]={'title':w['title'],'abstract':w.get('abstract'),'venue':w.get('journal'),'indexing_method':w['indexing_method'],'langs':w.get('langs'),'akw':[canon(k) for k in (w.get('keywords') or [])]}
            g.labels[w['id']]={'all':G,'noct':G-M['CHECKTAGS'],'major':{h['ui'] for h in w['mesh'] if h['major']=='Y' or any(q['major']=='Y' for q in h['qualifiers'])}}
        g.cands='MeSH descriptor UI'
    elif name in ('akw_godeval','akw_godtrain'):
        GOD=f'{HERE}/../god'; ev=set(open(f'{GOD}/data/god_eval_ids.txt').read().split()); want=ev if name=='akw_godeval' else None
        lab={}
        for r in map(json.loads, open(f'{GOD}/results/god_god55.jsonl')):
            if 'answer' in r and r['answer']['keywords'] and ((r['id'] in ev) == (name=='akw_godeval')): lab[r['id']]={canon(k) for k in r['answer']['keywords'] if canon(k)}
        n=0
        for l in open(f'{GOD}/data/god_sample.jsonl'):
            w=json.loads(l)
            if w['id'] not in lab: continue
            if name=='akw_godtrain' and n>=30000: break
            n+=1; g.works[w['id']]={'title':w['title'],'abstract':w.get('abstract'),'venue':w.get('journal'),'language':w.get('language'),'today':[canon(k) for k in (w.get('current') or [])]}; g.labels[w['id']]={'akw':lab[w['id']]}
        g.cands='canon keyword (Opus vocabulary)'
    elif name in ('akw_field','akw_train'):
        for w in jl(f'{DATA}/{name}.jsonl'):
            g.works[w['id']]={'title':w['title'],'abstract':w.get('abstract'),'venue':w.get('journal'),'language':w.get('language'),'field':w.get('field'),'source':w['source'],'pmid':w.get('pmid'),'today':[canon(k['name']) for k in (w.get('current') or [])]}
            g.labels[w['id']]={'akw':{canon(k) for k in w['author_kw'] if canon(k)}}
        g.cands='canon keyword'
    elif name=='akw_kp20k':
        for w in jl(f'{DATA}/kp20k.jsonl'):
            g.works[w['id']]={'title':w['title'],'abstract':w.get('abstract'),'venue':None,'today':[]}
            g.labels[w['id']]={'akw':{canon(k) for k in w['present']+w['absent'] if canon(k)}}
        g.cands='canon keyword'
    elif name.startswith('akw_'):
        for w in jl(f"{P['j1306']}/slices/{name[4:]}.jsonl"):
            g.works[w['id']]={'title':w['title'],'abstract':w.get('abstract'),'venue':w.get('journal'),'language':w.get('language'),'today':[canon(k['name']) for k in (w.get('current') or [])]}
            g.labels[w['id']]={'akw':{canon(k) for k in w['author_kw'] if canon(k)}}
        g.cands='canon keyword'
    elif name=='tibkat':
        for w in jl(f"{P['j1306']}/ext/tibkat/sample/works.jsonl"):
            g.works[w['id']]={'title':w['title'],'abstract':w.get('abstract'),'venue':w.get('journal'),'kind':w.get('indexing_method')}
            g.labels[w['id']]={'all':{h['ui'] for h in w['mesh']}}
        g.cands='GND code'
    elif name in ('kp20k','inspec'):
        st=_stem()
        for w in jl(f'{DATA}/{name}.jsonl'):
            g.works[w['id']]={'title':w['title'],'abstract':w.get('abstract'),'venue':None}
            pres={st(k) for k in w['present']}; abs_={st(k) for k in w['absent']}-pres
            g.labels[w['id']]={'present':pres,'absent':abs_,'all':pres|abs_}
        g.cands='Porter-stemmed phrase'
    else: raise KeyError(name)
    return g.load_split()
_ST=None
def cand_id(g, key):
    """Map a vocabulary key (or MeSH UI / GND code) into the gold's comparison space."""
    global _ST
    if g.name in ('kp20k','inspec'):
        if _ST is None: _ST=_stem()
        return _ST(key)
    if g.name in ('mesh','tibkat'): return key
    return canon(key)
def work_text(w, cap=3000):
    parts=[f"Title: {w.get('title') or ''}"]
    if w.get('abstract'): parts.append(f"Abstract: {w['abstract'][:cap]}")
    if w.get('venue'): parts.append(f"Venue: {w['venue']}")
    return '\n'.join(parts)
