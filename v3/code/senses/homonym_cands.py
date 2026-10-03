"""Sense-label sizing (#1465, 2026-09-30 CT): per keyword, the field mix of its works (primary topic field). Candidates = keywords on >= 300
works whose works split across >= 2 domains with the second domain >= 10%. -> data/homonym_fields.jsonl (kid, works, fields with shares + 3 titles each)
(Release note: the file it actually writes is data/homonym_domain_cands.jsonl, one row per candidate: kid, works, top two domains and shares;
the per-field counts stay in the table openalex.transient.kw1465_kw_field, which homonym_judge.py reads.)
python homonym_cands.py"""
import json, os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'common')); from dbx import sql
T = 'openalex.transient'
sql(f"""CREATE OR REPLACE TABLE {T}.kw1465_kw_field AS
WITH e AS (SELECT k.work_id, x.id AS kid FROM openalex.works.work_keywords_v2 k LATERAL VIEW explode(k.keywords) t AS x),
w AS (SELECT id, primary_topic.field.display_name AS field, primary_topic.domain.display_name AS domain FROM openalex.works.openalex_works WHERE primary_topic IS NOT NULL)
SELECT e.kid, w.domain, w.field, count(*) AS n FROM e JOIN w ON w.id = e.work_id GROUP BY 1, 2, 3""", verbose=False, timeout_s=7200)
rows = sql(f"""WITH d AS (SELECT kid, domain, sum(n) AS n FROM {T}.kw1465_kw_field GROUP BY 1, 2),
t AS (SELECT kid, sum(n) AS tot, max(n) AS top FROM d GROUP BY 1),
s AS (SELECT d.kid, t.tot, d.domain, d.n / t.tot AS share, row_number() OVER (PARTITION BY d.kid ORDER BY d.n DESC) AS r FROM d JOIN t ON t.kid = d.kid WHERE t.tot >= 300)
SELECT kid, max(tot) AS works, max(CASE WHEN r = 1 THEN domain END) AS d1, max(CASE WHEN r = 1 THEN share END) AS s1,
       max(CASE WHEN r = 2 THEN domain END) AS d2, max(CASE WHEN r = 2 THEN share END) AS s2 FROM s GROUP BY kid HAVING max(CASE WHEN r = 2 THEN share END) >= 0.10""")
print(len(rows), 'candidates;', sum(int(r['works']) for r in rows), 'works-assignments')
with open('data/homonym_domain_cands.jsonl', 'w') as f:
    for r in rows: f.write(json.dumps(r) + '\n')
