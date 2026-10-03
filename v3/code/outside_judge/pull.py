"""Outside-judge sample: draw ~400 works with BOTH old (production keywords as of the
2026-09-26 snapshot in openalex.transient.kwreview_works_sample.prod_kw, i.e. before the 2026-09-29
launch) and new keywords (openalex.works.work_keywords_v2 joined to openalex.common.keywords_v2).
Frame: the peer review's 1-in-1000 hash sample of openalex_works (475,072 works). Excludes the
peer review's own 399-work draw (hash(work_id,'rand1') mod 1000 < 1). Fresh seed 'gpt1322'.
Read-only. Written 2026-09-30. Internal tables: kwreview_works_sample is a frozen copy of the 2026-09-26
production keywords for the review sample; outside OpenAlex, use the released sample_a.jsonl instead."""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'common'))
from dbx import sql

Q = r"""
WITH s AS (
  SELECT id, work_id, title, substr(abstract,1,2500) abstract, coalesce(length(abstract),0) abs_len, type,
         publication_year, language, is_xpac, source_name, transform(prod_kw, x -> x.display_name) old_kw,
         pmod(hash(work_id, 'gpt1322'), 1000000) r
  FROM openalex.transient.kwreview_works_sample
  WHERE size(prod_kw) > 0 AND pmod(hash(work_id, 'rand1'), 1000) >= 1
),
cand AS (SELECT * FROM s WHERE r < 3000),
e AS (
  SELECT c.work_id, x.pos, x.kw
  FROM cand c JOIN openalex.works.work_keywords_v2 w ON w.work_id = c.id
  LATERAL VIEW posexplode(w.keywords) x AS pos, kw
),
k AS (
  SELECT e.work_id, e.pos, coalesce(v.display_name, e.kw.display_name) dn
  FROM e LEFT JOIN openalex.common.keywords_v2 v ON v.id = e.kw.id
),
ka AS (SELECT work_id, transform(sort_array(collect_list(struct(pos, dn))), x -> x.dn) new_kw FROM k GROUP BY work_id)
SELECT c.*, ka.new_kw FROM cand c JOIN ka USING (work_id) WHERE size(ka.new_kw) > 0 ORDER BY c.r
"""
rows = sql(Q)
print(len(rows))
out = []
for r in rows[:420]:
    for f in ('old_kw', 'new_kw'):
        if isinstance(r[f], str):
            r[f] = json.loads(r[f])
    out.append(r)
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'eval', 'outside_judge', 'sample_a_text.jsonl'), 'w') as f:
    for r in out:
        f.write(json.dumps(r) + '\n')
import collections
print('n', len(out), 'no_abs', sum(1 for r in out if not r['abs_len'] or int(r['abs_len']) == 0),
      'xpac', sum(1 for r in out if r['is_xpac'] in (True, 'true')))
