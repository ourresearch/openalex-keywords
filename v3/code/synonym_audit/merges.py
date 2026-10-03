"""All synmap9 merges that moved works in the shipped v3.0 build (kw1322v3n): raw kid0 (after plural fold) -> heading, with works moved.
Reads the vocabulary build tables (code/vocabulary/build_vocab.py, prefix kw1322v3n_). -> data/synonyms/audit_v31/merges.jsonl (released gzipped)
    python3 merges.py"""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'common')); from dbx import sql
AUD = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'synonyms', 'audit_v31')
T='openalex.transient'
os.makedirs(AUD, exist_ok=True)
r = sql(f"""WITH s AS (SELECT DISTINCT st.raw, COALESCE(f.kid, st.kid0) AS fk, st.kid FROM {T}.kw1322v3n_strings st LEFT JOIN {T}.kw1322v3n_fold f ON f.kid0 = st.kid0
                 WHERE st.kid IS NOT NULL AND st.kid <> COALESCE(f.kid, st.kid0)),
  m AS (SELECT s.fk, s.kid, count(DISTINCT mf.work_id) AS works, max_by(mf.raw, 1) AS example_raw FROM {T}.kw1322v3n_mentions_final mf JOIN s ON s.raw = mf.raw GROUP BY 1, 2)
  SELECT m.fk AS src, m.kid AS heading, m.works, m.example_raw, v.display_name AS heading_name, v.works_count AS heading_works FROM m JOIN {T}.kw1322v3n_vocab v ON v.kid = m.kid""")
with open(os.path.join(AUD, 'merges.jsonl'), 'w') as f:
    for x in r: f.write(json.dumps(x, ensure_ascii=False) + '\n')
tot = sum(int(x['works']) for x in r); print(len(r), 'merges', tot, 'work-assignments moved')
