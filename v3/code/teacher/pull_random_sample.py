"""The God-eval sample, take 2 (Jason: a simple random sample of the population, no stratification): random works from openalex_works with no filter except is_xpac = false
(the works the product serves), any year, with or without abstract or title. Two hash buckets of 1/5000 (≈ 190K works) → random 100K, seed 1322. Excludes eval-slice and judge ids.
Writes data/god_sample.jsonl with the same fields as the stratified pull. python pull_random_sample.py [--n 100000] [--buckets 313,2711]"""
import argparse, json, random, sys, os, collections
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'common')); from dbx import sql
sys.path.insert(0, '../bench'); from common import P, jl, wjl
ap=argparse.ArgumentParser(); ap.add_argument('--n', type=int, default=100000); ap.add_argument('--buckets', default='313,2711'); ap.add_argument('--out', default='data/god_sample.jsonl'); ap.add_argument('--seed', type=int, default=1322); a=ap.parse_args()
rows=sql(f"""
SELECT w.id, w.doi, w.ids['pmid'] AS pmid_url, w.title, w.abstract, w.primary_location.source.display_name AS journal, w.language, w.publication_year AS year, w.type,
       w.primary_topic.field.display_name AS field, w.primary_topic.subfield.display_name AS subfield, w.keywords
FROM openalex.works.openalex_works w
WHERE COALESCE(w.is_xpac, false) = false AND pmod(hash(w.id), 5000) IN ({a.buckets})""")
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
random.Random(a.seed).shuffle(recs); out=recs[:a.n]; os.makedirs('data', exist_ok=True); wjl(a.out, out)
print('wrote', len(out), 'works; with abstract', sum(1 for x in out if x['abstract']), '; with title', sum(1 for x in out if x['title']))
print('years', sorted(collections.Counter((x['year']//10*10 if x['year'] else None) for x in out).items(), key=lambda kv: str(kv[0])))
print('langs', collections.Counter(x['language'] for x in out).most_common(6)); print('types', collections.Counter(x['type'] for x in out).most_common(6)); print('fields', collections.Counter(x['field'] for x in out).most_common(6))
