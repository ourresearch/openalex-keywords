#!/bin/zsh
# #1322 final synonym map (synmap9: keep-apart constraints + ambiguous-acronym isolation) -> vocab nw1322v3l_* -> split audits + dry run
set -e
: "${CLAUDE_API_KEY:?}"  # env: CLAUDE_API_KEY (or ANTHROPIC_API_KEY), DATABRICKS_HOST / DATABRICKS_TOKEN (CLI auth), DATABRICKS_WAREHOUSE_ID
S=${KW_SCRATCH:?set KW_SCRATCH to a working copy laid out as the original scratch dir, see README.md here}
cd $S/synonyms
cd $S/god/corpus_v3
uv run -q --with httpx python - <<'PY'
import json, sys; sys.path.insert(0, '.'); import dbxr
rows = [json.loads(l) for l in open('../../synonyms/data/synmap9.jsonl')]
esc = lambda s: s.replace('\\', '\\\\').replace("'", "\\'")
dbxr.sql("CREATE OR REPLACE TABLE openalex.transient.kw1322v3_synmap9 (kid STRING, canon STRING)", warehouse=__import__('os').environ['DATABRICKS_WAREHOUSE_ID'])
for i in range(0, len(rows), 5000):
    dbxr.sql("INSERT INTO openalex.transient.kw1322v3_synmap9 VALUES " + ",".join(f"('{esc(r['kid'])}','{esc(r['canon'])}')" for r in rows[i:i+5000]), warehouse=__import__('os').environ['DATABRICKS_WAREHOUSE_ID'])
dbxr.sql("CREATE OR REPLACE TABLE openalex.transient.kw1322v3n_strings0 AS SELECT * FROM openalex.transient.kw1322v3f_strings0", warehouse=__import__('os').environ['DATABRICKS_WAREHOUSE_ID'])
print('synmap9 rows', len(rows))
PY
echo "$(date '+%T') vocab n"; cd $S/vocab && uv run -q --with httpx python build_vocab.py --source keywords_student_v3fix --prefix kw1322v3n_ --boilerplate --synmap kw1322v3_synmap9 --min-lp -1.2 --floor 100 --from-step 2 | tail -1
cd $S/synonyms
echo "$(date '+%T') audits n"; uv run -q --with anthropic --with httpx python audit.py --prefix kw1322v3n_ --n 1000 --seed 29 > data/audit_n29.log 2>&1 &
sleep 90; uv run -q --with anthropic --with httpx python audit.py --prefix kw1322v3n_ --n 1000 --seed 31 > data/audit_n31.log 2>&1 &
wait; head -1 data/audit_n29.log data/audit_n31.log
cd $S/writepath && python3 gen_build.py --wk openalex.transient.kw1322v3n_work_keywords --vocab openalex.transient.kw1322v3n_vocab --cutover 2026-09-30 --tagger-version kw1322-student-v3fix --vocab-version kw1322-v3n > /dev/null
cd $S/god/corpus_v3 && uv run -q --with httpx python - <<'PY'
import sys; sys.path.insert(0, '.'); import dbxr
sql = open('../../writepath/build_tables_dryrun.sql').read()
body = "\n".join(l for l in sql.splitlines() if not l.strip().startswith('--'))
for s in [s.strip() for s in body.split(';') if s.strip()]:
    r = dbxr.sql(s, warehouse=__import__('os').environ['DATABRICKS_WAREHOUSE_ID'])
    if r: print('dryrun', str(r)[:300])
PY
echo "$(date '+%T') FINAL N DONE"
