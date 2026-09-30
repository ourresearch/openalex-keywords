#!/bin/zsh
# #1322 round-3 rebuild (synonyms stop-ship, 2026-09-28 evening): synmap3 -> vocab kw1322v3h_*
set -e
: "${CLAUDE_API_KEY:?}"  # env: CLAUDE_API_KEY (or ANTHROPIC_API_KEY), DATABRICKS_HOST / DATABRICKS_TOKEN (CLI auth), DATABRICKS_WAREHOUSE_ID
S=${KW_SCRATCH:?set KW_SCRATCH to a working copy laid out as the original scratch dir, see README.md here}
cd $S/god/corpus_v3
uv run -q --with httpx python - <<'PY'
import json, sys; sys.path.insert(0, '.'); import dbxr
rows = [json.loads(l) for l in open('../../synonyms/data/synmap3.jsonl')]
esc = lambda s: s.replace('\\', '\\\\').replace("'", "\\'")
dbxr.sql("CREATE OR REPLACE TABLE openalex.transient.kw1322v3_synmap3 (kid STRING, canon STRING)", warehouse=__import__('os').environ['DATABRICKS_WAREHOUSE_ID'])
for i in range(0, len(rows), 5000):
    dbxr.sql("INSERT INTO openalex.transient.kw1322v3_synmap3 VALUES " + ",".join(f"('{esc(r['kid'])}','{esc(r['canon'])}')" for r in rows[i:i+5000]), warehouse=__import__('os').environ['DATABRICKS_WAREHOUSE_ID'])
dbxr.sql("CREATE OR REPLACE TABLE openalex.transient.kw1322v3h_strings0 AS SELECT * FROM openalex.transient.kw1322v3f_strings0", warehouse=__import__('os').environ['DATABRICKS_WAREHOUSE_ID'])
print('synmap3 rows', dbxr.sql("SELECT count(*) n FROM openalex.transient.kw1322v3_synmap3", warehouse=__import__('os').environ['DATABRICKS_WAREHOUSE_ID']))
PY
echo "$(date '+%T') vocab h"; cd $S/vocab && uv run -q --with httpx python build_vocab.py --source keywords_student_v3fix --prefix kw1322v3h_ --boilerplate --synmap kw1322v3_synmap3 --min-lp -1.2 --floor 100 --from-step 2
echo "$(date '+%T') VOCAB DONE"
