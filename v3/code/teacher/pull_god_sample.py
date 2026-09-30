"""The God-eval sample: ~100K works with an abstract, non-xpac, 2010+, stratified by primary field to the corpus share (floor per field), natural language/type mix,
from hash buckets disjoint from every eval slice (akw_field 77; akw_train 101,203,307,409,511; #1306 slices used other keyspaces). Excludes ids of the eval slices and the judge gold.
Writes data/god_sample.jsonl {id, doi, pmid, field, subfield, title, abstract, journal, language, year, type, current}. python pull_god_sample.py [--n 100000] [--floor 800] [--buckets 13,29,43,61,83,97,113,131,149,167,181,199]"""
import argparse, json, random, sys, os, collections
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'common')); from dbx import sql
sys.path.insert(0, '../bench'); from common import P, jl, wjl
ap=argparse.ArgumentParser(); ap.add_argument('--n', type=int, default=100000); ap.add_argument('--floor', type=int, default=800); ap.add_argument('--buckets', default='13,29,43,61,83,97,113,131,149,167,181,199'); a=ap.parse_args()
rows=sql(f"""
SELECT w.id, w.doi, w.ids['pmid'] AS pmid_url, w.title, w.abstract, w.primary_location.source.display_name AS journal, w.language, w.publication_year AS year, w.type,
       w.primary_topic.field.display_name AS field, w.primary_topic.subfield.display_name AS subfield, w.keywords
FROM openalex.works.openalex_works w
WHERE w.abstract IS NOT NULL AND length(w.abstract) > 200 AND w.title IS NOT NULL AND COALESCE(w.is_xpac, false) = false AND w.publication_year >= 2010 AND w.primary_topic IS NOT NULL
  AND pmod(hash(w.id), 1000) IN ({a.buckets})""")
print('rows', len(rows), flush=True)
excl=set()
for f in ('akw_field','akw_train'): excl|={r['id'] for r in jl(f'../bench/data/{f}.jsonl')}
for sl in ('pubmed','landing','repo'): excl|={r['id'] for r in jl(f"{P['j1306']}/slices/{sl}.jsonl")}
excl|={r['id'] for r in jl(f"{P['j1292']}/gold/works.jsonl")}
def kws(x): return json.loads(x) if isinstance(x,str) else (x or [])
recs=[]
for r in rows:
    wid=f"W{r['id']}" if not str(r['id']).startswith('W') else r['id']
    if wid in excl: continue
    recs.append({'id':wid,'doi':r['doi'],'pmid':(r['pmid_url'] or '').rsplit('/',1)[-1] or None,'field':r['field'],'subfield':r['subfield'],'title':r['title'],'abstract':r['abstract'],'journal':r['journal'],'language':r['language'],'year':r['year'],'type':r['type'],
                 'current':[c['display_name'] for c in kws(r['keywords'])]})
byf=collections.defaultdict(list)
for x in recs: byf[x['field']].append(x)
tot=len(recs); out=[]
for f,xs in sorted(byf.items(), key=lambda kv:-len(kv[1])):
    random.Random(1322).shuffle(xs); n=max(a.floor, round(a.n*len(xs)/tot)); out+=xs[:n]; print(f'{f:45s} available {len(xs):7d} taken {min(n,len(xs)):6d}')
random.Random(7).shuffle(out); os.makedirs('data', exist_ok=True); wjl('data/god_sample.jsonl', out)
print('wrote', len(out), 'works;', collections.Counter(x['language'] for x in out).most_common(6), collections.Counter(x['type'] for x in out).most_common(5))
