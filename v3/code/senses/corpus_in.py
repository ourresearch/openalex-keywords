"""#1476: build the corpus input for the Jev sense run from CURRENT work_keywords_v2 (post synmap10/10b; read-only) ->
openalex.transient.kw1476_corpus_in (work_id, heading, title, abstract[:1500], venue, bucket 0..319). One row per (work, split keyword).
python corpus_in.py"""
import os, json, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'common')); from dbx import sql
S = json.load(open('data/senses_summary.json'))
ids = ",".join("'https://openalex.org/keywords/" + s['heading'] + "'" for s in S)
sql(f"""CREATE OR REPLACE TABLE openalex.transient.kw1476_corpus_in AS
WITH e AS (SELECT DISTINCT k.work_id, substring_index(x.id, '/', -1) AS heading
           FROM openalex.works.work_keywords_v2 k LATERAL VIEW explode(k.keywords) t AS x WHERE x.id IN ({ids}))
SELECT e.work_id, e.heading, w.title, substr(w.abstract, 1, 1500) AS abstract, w.primary_location.source.display_name AS venue,
       pmod(xxhash64(e.work_id, e.heading), 320) AS bucket
FROM e JOIN openalex.works.openalex_works w ON w.id = e.work_id""", timeout_s=3600)
print(sql("SELECT count(*) n, count(DISTINCT work_id) works, count(DISTINCT heading) kws, sum(CASE WHEN abstract IS NULL THEN 1 ELSE 0 END) no_abs FROM openalex.transient.kw1476_corpus_in"))
