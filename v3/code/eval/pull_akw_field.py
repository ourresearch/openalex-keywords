"""Field-stratified held-out author-keyword slice for the composite benchmark (EXPLORE § 8): author-supplied keywords only (PubMed KeywordList, landing-page citation_keywords;
no repository dc:subject), joined to works with an abstract, non-xpac, 2015+, one hash bucket of 1/1000, then stratified by primary field: n_field = max(floor, share × N).
Excludes the existing slices' and the judge gold's works. Writes data/akw_field.jsonl {id, pmid, doi, field, source, title, abstract, journal, language, year, current, author_kw}.
python pull_akw_field.py [--n 6000] [--floor 60] [--bucket 77]"""
import argparse, json, random, sys, os, collections
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'common')); from dbx import sql
from common import P, DATA, jl, wjl
ap=argparse.ArgumentParser(); ap.add_argument('--n', type=int, default=6000); ap.add_argument('--floor', type=int, default=60); ap.add_argument('--bucket', type=int, default=77); ap.add_argument('--buckets', default=None, help='comma list of 1/1000 buckets; with --all no stratification'); ap.add_argument('--all', action='store_true'); ap.add_argument('--out', default='akw_field'); a=ap.parse_args()
rows=sql(f"""
WITH pm AS (SELECT concat('https://pubmed.ncbi.nlm.nih.gov/', CAST(pmid AS STRING)) AS pmid_url, collect_set(kw) AS kws FROM openalex.transient.author_kw_pubmed GROUP BY pmid),
lp AS (SELECT lower(concat('https://doi.org/', native_id)) AS doi, collect_set(kw) AS kws FROM openalex.transient.author_kw_landing GROUP BY native_id),
w AS (SELECT w.id, w.title, w.abstract, w.primary_location.source.display_name AS journal, w.language, w.publication_year AS year, w.primary_topic.field.display_name AS field,
             w.ids['pmid'] AS pmid_url, lower(w.doi) AS doi, w.keywords
      FROM openalex.works.openalex_works w
      WHERE w.abstract IS NOT NULL AND length(w.abstract) > 200 AND COALESCE(w.is_xpac, false) = false AND w.publication_year >= 2015 AND w.primary_topic IS NOT NULL AND pmod(hash(w.id), 1000) IN ({a.buckets or a.bucket}))
SELECT w.*, pm.kws AS kw_pubmed, lp.kws AS kw_landing FROM w LEFT JOIN pm ON pm.pmid_url = w.pmid_url LEFT JOIN lp ON lp.doi = w.doi
WHERE pm.kws IS NOT NULL OR lp.kws IS NOT NULL""")
print('rows with author keywords', len(rows), flush=True)
excl=set()
for sl in ('pubmed','landing','repo'): excl|={r['id'] for r in jl(f"{P['j1306']}/slices/{sl}.jsonl")}
excl|={r['id'] for r in jl(f"{P['j1292']}/gold/works.jsonl")}
def kws(x): return json.loads(x) if isinstance(x,str) else (x or [])
recs=[]
for r in rows:
    wid=f"W{r['id']}" if not str(r['id']).startswith('W') else r['id']
    if wid in excl: continue
    k=[s.strip() for s in kws(r['kw_pubmed']) if s and s.strip()]; src='pubmed'
    if not k: k=[s.strip() for s in kws(r['kw_landing']) if s and s.strip()]; src='landing'
    k=list(dict.fromkeys(k))
    if len(k)<2 or len(k)>25: continue
    recs.append({'id':wid,'pmid':(r['pmid_url'] or '').rsplit('/',1)[-1] or None,'doi':r['doi'],'field':r['field'],'source':src,'title':r['title'],'abstract':r['abstract'],'journal':r['journal'],'language':r['language'],'year':r['year'],
                 'current':[{'id':c['id'],'name':c['display_name'],'score':c.get('score')} for c in kws(r['keywords'])],'author_kw':k})
if a.all:
    out=recs; print('no stratification')
else:
    byf=collections.defaultdict(list)
    for x in recs: byf[x['field']].append(x)
    tot=len(recs); out=[]
    for f,xs in sorted(byf.items(), key=lambda kv:-len(kv[1])):
        random.Random(1322).shuffle(xs); n=max(a.floor, round(a.n*len(xs)/tot)); out+=xs[:n]; print(f'{f:45s} available {len(xs):6d} taken {min(n,len(xs)):5d}')
wjl(f'{DATA}/{a.out}.jsonl', out); print('wrote', len(out), 'works;', collections.Counter(x['source'] for x in out))
