"""Normalize + dedupe free-text keyword proposals against the vocabulary and each other.
python normalize_kw.py gen/pilot_gold.jsonl [--min_works 2] [--vocab vocab_1292/vocab.jsonl] [--out gen/candidates.jsonl]
Steps: (1) canonical form: lower, unicode NFKC, hyphen/underscore -> space, collapse spaces, strip trailing 's' on the last token when the singular also occurs or the token is > 4 chars (crude plural fold; keeps 'physics', 'series' via exceptions);
(2) exact match to a vocabulary key -> 'existing'; (3) group proposals by canonical form, count distinct works; (4) keep groups with >= min_works, pick the most frequent surface form as display;
(5) near-duplicate report among survivors and vs vocabulary by bge-m3 cosine >= 0.93 (written for review, not auto-merged)."""
import json, argparse, collections, re, unicodedata, numpy as np
KEEP_S={'physics','series','species','analysis','diabetes','rabies','herpes','mathematics','economics','genetics','statistics','ethics','politics','linguistics','dynamics','electronics','robotics','optics','acoustics','thermodynamics','logistics','graphics','informatics','bioinformatics','genomics','proteomics','metabolomics','hydraulics','mechanics','kinetics','aesthetics','phonetics','semantics','pragmatics','stylistics','forensics','ceramics','cosmetics','athletics','gymnastics','tectonics','hydroponics','sepsis','psoriasis','tuberculosis','mitosis','meiosis','osmosis','diagnosis','prognosis','thesis','hypothesis','synthesis','emphasis','crisis','basis','cannabis','pancreas','atlas','gas','glass','stress','process','access','class','mass','bias','virus','fungus','census','campus','corpus','nucleus','radius','status','consensus','lupus','bus','plus','arthritis','hepatitis','bronchitis','gastritis','dermatitis','texas','kansas','wales','lens','news','means'}
GENERIC={'pilot study','case report','case series','conference proceedings','research guide','editorial','commentary','letter to the editor','book chapter','thesis','dissertation','proceedings','abstract','poster','erratum','corrigendum','retraction','preface','foreword','bibliography','annotated bibliography','process','analysis','study','data','method','methods','system','systems','model','models','approach','research','review','article','paper','dataset','results','effect','effects','role','application','applications','development','evaluation','assessment','design','performance','theory','practice','case study','overview','introduction','survey','framework','technique','techniques','tool','tools','issue','issues','challenge','challenges','trend','trends','impact','factor','factors','relationship','comparison','characteristics','properties','structure','function','behavior','behaviour','change','changes','use','usage','type','types','level','levels','quality','management','education','science','engineering','medicine','biology','chemistry','physics','mathematics','sociology','psychology','economics','history','philosophy','law','computer science','political science','environmental science','materials science','geography','geology','art','business','agriculture','health','technology','information','knowledge','learning','communication','society','culture','policy','industry','market','work','life','human','nature','world','country','region','area','time','space','field','case','problem','solution','value','measure','measurement','test','testing','simulation','optimization','modeling','modelling','prediction','classification','detection','identification','characterization','estimation','experiment','experimental study','literature review','systematic review','clinical trial','randomized controlled trial'}
def canon(s):
    s=unicodedata.normalize('NFKC', s).lower().strip(); s=re.sub(r'[‐-―_/\-]', ' ', s); s=re.sub(r'\s+', ' ', s); s=s.strip(' .,;:"\'')
    toks=s.split(' ')
    if toks and toks[-1].endswith('s') and not toks[-1].endswith('ss') and toks[-1] not in KEEP_S and len(toks[-1])>4: toks[-1]=toks[-1][:-1]
    return ' '.join(toks)

def main():
    ap=argparse.ArgumentParser()    ; ap.add_argument('src'); ap.add_argument('--min_works', type=int, default=2); ap.add_argument('--vocab', default='vocab_1292/vocab.jsonl'); ap.add_argument('--out', default='gen/candidates.jsonl'); ap.add_argument('--maxwords', type=int, default=4); ap.add_argument('--near', action='store_true'); a=ap.parse_args()
    V={}
    for l in open(a.vocab): v=json.loads(l); V.setdefault(canon(v['key']), v['key'])
    groups=collections.defaultdict(lambda: {'works':set(),'forms':collections.Counter()}); stats=collections.Counter()
    for l in open(a.src):
        r=json.loads(l)
        for k in r.get('keywords',[]):
            stats['proposals']+=1; c=canon(k)
            if not c or len(c.split())>a.maxwords or c in GENERIC or len(c)<3: stats['dropped_generic_or_long']+=1; continue
            if c in V: stats['existing']+=1; continue
            groups[c]['works'].add(r['id']); groups[c]['forms'][k.strip()]+=1
    surv={c:g for c,g in groups.items() if len(g['works'])>=a.min_works}
    with open(a.out,'w') as f:
        for c,g in sorted(surv.items(), key=lambda x:-len(x[1]['works'])):
            f.write(json.dumps({'key':c,'display':g['forms'].most_common(1)[0][0],'n_works':len(g['works']),'forms':dict(g['forms'])}, ensure_ascii=False)+'\n')
    print(dict(stats), 'new canonical forms', len(groups), f'kept (>= {a.min_works} works)', len(surv))
    print('top kept:', [(c, len(g['works'])) for c,g in sorted(surv.items(), key=lambda x:-len(x[1]['works']))[:25]])
    if a.near and surv:
        from sentence_transformers import SentenceTransformer; import torch
        m=SentenceTransformer('BAAI/bge-m3', device='mps' if torch.backends.mps.is_available() else 'cpu'); m.max_seq_length=64
        keys=list(surv); E=m.encode(keys, batch_size=256, normalize_embeddings=True); S=E@E.T; np.fill_diagonal(S,0)
        pairs=[(keys[i],keys[j],float(S[i,j])) for i,j in zip(*np.where(np.triu(S)>=0.93))]
        print('near-duplicate pairs among new (cos>=0.93):', len(pairs), pairs[:15])


if __name__=='__main__':
    main()
