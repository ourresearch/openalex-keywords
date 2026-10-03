"""Sample 40 works per source string merged into keywords/long-covid (titles and abstracts, from the build tables). -> lc_samples.jsonl (not released)."""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'common')); from dbx import sql
T='openalex.transient'
SRC = ['post-covid-19-era','post-covid-19-pandemic','post-covid-19-period','post-covid','post-covid-19','post-covid-19-syndrome','post-covid-19-symptoms','post-covid-19-condition','long-covid']
r = sql(f"""WITH m AS (SELECT DISTINCT mf.work_id, st.kid0 AS src FROM {T}.kw1322v3n_mentions_final mf JOIN {T}.kw1322v3n_strings st ON st.raw = mf.raw
          WHERE mf.kid = 'long-covid' AND st.kid0 IN ({','.join("'"+s+"'" for s in SRC)})),
  s AS (SELECT *, row_number() OVER (PARTITION BY src ORDER BY xxhash64(work_id, 1465)) AS rn FROM m)
  SELECT s.src, s.work_id, w.title, w.publication_year AS year, w.type, w.primary_location.source.display_name AS venue, substr(w.abstract, 1, 2500) AS abstract
  FROM s JOIN openalex.works.openalex_works w ON w.id = CAST(substr(s.work_id, 2) AS BIGINT) WHERE s.rn <= 40""")
with open('lc_samples.jsonl', 'w') as f:
    for x in r: f.write(json.dumps(x, ensure_ascii=False)+'\n')
