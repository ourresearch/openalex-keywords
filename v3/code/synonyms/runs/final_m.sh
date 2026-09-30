#!/bin/zsh
# #1322 final synonym map (synmap8: keep-apart constraints + ambiguous-acronym isolation) -> vocab mw1322v3l_* -> split audits + dry run
set -e
: "${CLAUDE_API_KEY:?}"  # env: CLAUDE_API_KEY (or ANTHROPIC_API_KEY), DATABRICKS_HOST / DATABRICKS_TOKEN (CLI auth), DATABRICKS_WAREHOUSE_ID
S=${KW_SCRATCH:?set KW_SCRATCH to a working copy laid out as the original scratch dir, see README.md here}
cd $S/synonyms
uv run -q python - <<'PY'
import json, random
wc = {json.loads(l)['kid']: json.loads(l)['works_count'] for l in open('data/vocab_all.jsonl')}
M = [json.loads(l) for l in open('data/synmap8.jsonl')]; M = [m for m in M if m['kid'] in wc and m['canon'] in wc]
rng = random.Random(9); pick = []; seen = set()
while len(pick) < 400:
    m = rng.choices(M, weights=[wc.get(x['kid'], 1) for x in M])[0]
    if m['kid'] not in seen: seen.add(m['kid']); pick.append(m)
json.dump([[m['kid'], m['canon'], 1.0] for m in pick], open('data/cand_precision4.json', 'w'))
PY
uv run -q --with anthropic python judge_pairs2.py --cand data/cand_precision4.json --vocab data/vocab_all.jsonl --out data/judged_precision4.jsonl --workers 8 | tail -1
python3 -c "import json; R=[json.loads(l) for l in open('data/judged_precision4.jsonl')]; print('precision audit synmap8:', round(1-sum(r['merge'] is False for r in R)/len(R), 3))"
cd $S/god/corpus_v3
uv run -q --with httpx python - <<'PY'
import json, sys; sys.path.insert(0, '.'); import dbxr
rows = [json.loads(l) for l in open('../../synonyms/data/synmap8.jsonl')]
esc = lambda s: s.replace('\\', '\\\\').replace("'", "\\'")
dbxr.sql("CREATE OR REPLACE TABLE openalex.transient.kw1322v3_synmap8 (kid STRING, canon STRING)", warehouse=__import__('os').environ['DATABRICKS_WAREHOUSE_ID'])
for i in range(0, len(rows), 5000):
    dbxr.sql("INSERT INTO openalex.transient.kw1322v3_synmap8 VALUES " + ",".join(f"('{esc(r['kid'])}','{esc(r['canon'])}')" for r in rows[i:i+5000]), warehouse=__import__('os').environ['DATABRICKS_WAREHOUSE_ID'])
dbxr.sql("CREATE OR REPLACE TABLE openalex.transient.kw1322v3m_strings0 AS SELECT * FROM openalex.transient.kw1322v3f_strings0", warehouse=__import__('os').environ['DATABRICKS_WAREHOUSE_ID'])
print('synmap8 rows', len(rows))
PY
echo "$(date '+%T') vocab m"; cd $S/vocab && uv run -q --with httpx python build_vocab.py --source keywords_student_v3fix --prefix kw1322v3m_ --boilerplate --synmap kw1322v3_synmap8 --min-lp -1.2 --floor 100 --from-step 2 | tail -1
cd $S/synonyms
echo "$(date '+%T') audit m seed 29"; uv run -q --with anthropic --with httpx python audit.py --prefix kw1322v3m_ --n 1000 --seed 29 | head -1; cp data/audit_kw1322v3m_.json data/audit_m_seed29.json
echo "$(date '+%T') audit m seed 31"; uv run -q --with anthropic --with httpx python audit.py --prefix kw1322v3m_ --n 1000 --seed 31 | head -1
cd $S/writepath && python3 gen_build.py --wk openalex.transient.kw1322v3m_work_keywords --vocab openalex.transient.kw1322v3m_vocab --cutover 2026-09-29 --tagger-version kw1322-student-v3fix --vocab-version kw1322-v3m > /dev/null
cd $S/god/corpus_v3 && uv run -q --with httpx python - <<'PY'
import sys; sys.path.insert(0, '.'); import dbxr
sql = open('../../writepath/build_tables_dryrun.sql').read()
body = "\n".join(l for l in sql.splitlines() if not l.strip().startswith('--'))
for s in [s.strip() for s in body.split(';') if s.strip()]:
    r = dbxr.sql(s, warehouse=__import__('os').environ['DATABRICKS_WAREHOUSE_ID'])
    if r: print('dryrun', str(r)[:300])
PY
echo "$(date '+%T') FINAL M DONE"
