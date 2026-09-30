"""Acceptance checks for the vocabulary build (build_vocab.py). Writes data/check_vocab.json and prints a summary:
fates of all mentions; vocabulary size + works_count spread; keywords per work on a 1/1000 sample (all works, works with an abstract) vs EXPLORE § 18;
Jason's ten papers before/after; random samples of each removal fate for eyeballing; top 100 keywords.
uv run --with httpx python check_vocab.py"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'common')); import dbx
HERE = os.path.dirname(os.path.abspath(__file__)); T = 'openalex.transient'
S = lambda q: dbx.sql(q, warehouse=os.environ.get('DATABRICKS_WAREHOUSE_ID'), timeout_s=3 * 3600, verbose=False)
out = {}
out['fates'] = S(f"SELECT fate, count(*) AS mentions, count(DISTINCT kid) AS kids FROM {T}.kw1322_mentions_kept GROUP BY fate ORDER BY mentions DESC")
out['vocab'] = S(f"""SELECT count(*) AS keywords, sum(works_count) AS mentions, percentile(works_count, array(0.5, 0.9, 0.99)) AS pcts, max(works_count) AS mx,
                   sum(CASE WHEN kid RLIKE '[^\\\\x00-\\\\x7F]' THEN 1 ELSE 0 END) AS non_ascii FROM {T}.kw1322_vocab""")
out['per_work'] = S(f"""WITH s AS (SELECT concat('W', CAST(id AS STRING)) AS work_id, length(COALESCE(abstract, '')) >= 50 AS has_abs FROM openalex.works.openalex_works WHERE pmod(hash(id), 1000) = 7 AND NOT COALESCE(is_xpac, false)),
    n AS (SELECT s.work_id, s.has_abs, count(wk.keyword_id) AS n FROM s JOIN {T}.keywords_student_v1 st ON st.id = s.work_id LEFT JOIN {T}.kw1322_work_keywords wk ON wk.work_id = s.work_id GROUP BY s.work_id, s.has_abs)
    SELECT has_abs, least(n, 10) AS k, count(*) AS works FROM n GROUP BY 1, 2 ORDER BY 1, 2""")
priem = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'priem_keywords.json')))
ids = ", ".join(f"'{p['id']}'" for p in priem)
out['priem'] = S(f"""SELECT m.work_id, m.pos, m.raw, m.fate, v.works_count, wk.keyword_id FROM {T}.kw1322_mentions_kept m LEFT JOIN {T}.kw1322_vocab v ON v.kid = m.kid
                   LEFT JOIN {T}.kw1322_work_keywords wk ON wk.work_id = m.work_id AND wk.keyword_id = concat('keywords/', m.kid) WHERE m.work_id IN ({ids}) ORDER BY m.work_id, m.pos""")
out['samples'] = S(f"""SELECT fate, raw, work_id FROM (SELECT fate, raw, work_id, row_number() OVER (PARTITION BY fate ORDER BY rand(7)) AS r FROM {T}.kw1322_mentions_kept
                     WHERE fate <> 'keep' OR kid IN (SELECT kid FROM {T}.kw1322_country_kids)) WHERE r <= 25""")
out['country_kept_share'] = S(f"""SELECT m.kid, sum(CASE WHEN fate = 'keep' THEN 1 ELSE 0 END) AS kept, count(*) AS total FROM {T}.kw1322_mentions_kept m
                                WHERE m.kid IN (SELECT kid FROM {T}.kw1322_country_kids) GROUP BY m.kid ORDER BY total DESC LIMIT 25""")
out['top100'] = S(f"SELECT display_name, works_count FROM {T}.kw1322_vocab ORDER BY works_count DESC LIMIT 100")
os.makedirs(f'{HERE}/data', exist_ok=True); json.dump(out, open(f'{HERE}/data/check_vocab.json', 'w'), ensure_ascii=False, indent=1)
print(json.dumps({k: out[k] for k in ('fates', 'vocab', 'country_kept_share')}, ensure_ascii=False)[:4000])
