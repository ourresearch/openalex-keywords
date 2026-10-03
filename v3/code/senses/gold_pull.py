"""#1476 gold sample: 30 random works per split keyword (from #1465 senses_summary.json), any field. -> data/gold_works.jsonl
python gold_pull.py"""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'common')); from dbx import sql
HERE = os.path.dirname(os.path.abspath(__file__)); os.makedirs(f'{HERE}/data', exist_ok=True)
S = json.load(open('data/senses_summary.json'))
ids = ",".join("'https://openalex.org/keywords/" + s['heading'] + "'" for s in S)
rows = sql(f"""WITH e AS (SELECT k.work_id, x.id AS kid FROM openalex.transient.kw1465_sense_in k LATERAL VIEW explode(k.keywords) t AS x WHERE x.id IN ({ids})),
s AS (SELECT *, row_number() OVER (PARTITION BY kid ORDER BY xxhash64(work_id, 1476)) AS rn FROM e)
SELECT s.kid, s.work_id, w.title, substr(w.abstract, 1, 1500) AS abstract, w.primary_location.source.display_name AS venue, w.primary_topic.field.display_name AS field
FROM s JOIN openalex.works.openalex_works w ON w.id = s.work_id WHERE s.rn <= 30""", timeout_s=3600)
with open(f'{HERE}/data/gold_works.jsonl', 'w') as f:
    for r in rows: f.write(json.dumps(r, ensure_ascii=False) + '\n')
print(len(rows), 'works for', len(S), 'keywords')
