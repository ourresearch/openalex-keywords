#!/bin/zsh
# #1322 synonym round 5 (2026-09-29 early): word-order pass, alt names for ranks 50K-200K, re-verify context-rejected merges with the v2 rubric,
# synmap5 -> rebuild kw1322v3j_* -> audits (seed 29 = round-4 fresh sample, seed 30 = new)
set -e
: "${CLAUDE_API_KEY:?}"  # env: CLAUDE_API_KEY (or ANTHROPIC_API_KEY), DATABRICKS_HOST / DATABRICKS_TOKEN (CLI auth), DATABRICKS_WAREHOUSE_ID
S=${KW_SCRATCH:?set KW_SCRATCH to a working copy laid out as the original scratch dir, see README.md here}
cd $S/synonyms
echo "$(date '+%T') word-order judge"; uv run -q --with anthropic python judge_pairs2.py --cand data/cand_wordorder.json --vocab data/vocab_i.jsonl --out data/judged_wordorder.jsonl --workers 24 | tail -1
echo "$(date '+%T') alt names 50K-200K"; uv run -q --with anthropic --with httpx python altnames.py --vocab data/vocab_i.jsonl --start 50000 --end 200000 --tag 2 | tail -1
uv run -q --with anthropic python judge_pairs2.py --cand data/cand_alt2.json --vocab data/vocab_i.jsonl --out data/judged_alt2.jsonl --workers 24 | tail -1
echo "$(date '+%T') re-verify"; uv run -q --with anthropic python verify_v2.py | head -1
python3 build_synmap3.py --extra-judged data/judged_twins.jsonl data/judged_wordorder.jsonl data/judged_alt2.jsonl --extra-maps data/synmap_acronyms_small_ok.jsonl --rejected data/hi_reverified.jsonl --out data/synmap5.jsonl
cd $S/god/corpus_v3
uv run -q --with httpx python - <<'PY'
import json, sys; sys.path.insert(0, '.'); import dbxr
rows = [json.loads(l) for l in open('../../synonyms/data/synmap5.jsonl')]
esc = lambda s: s.replace('\\', '\\\\').replace("'", "\\'")
dbxr.sql("CREATE OR REPLACE TABLE openalex.transient.kw1322v3_synmap5 (kid STRING, canon STRING)", warehouse=__import__('os').environ['DATABRICKS_WAREHOUSE_ID'])
for i in range(0, len(rows), 5000):
    dbxr.sql("INSERT INTO openalex.transient.kw1322v3_synmap5 VALUES " + ",".join(f"('{esc(r['kid'])}','{esc(r['canon'])}')" for r in rows[i:i+5000]), warehouse=__import__('os').environ['DATABRICKS_WAREHOUSE_ID'])
dbxr.sql("CREATE OR REPLACE TABLE openalex.transient.kw1322v3j_strings0 AS SELECT * FROM openalex.transient.kw1322v3f_strings0", warehouse=__import__('os').environ['DATABRICKS_WAREHOUSE_ID'])
print('synmap5 rows', len(rows))
PY
echo "$(date '+%T') vocab j"; cd $S/vocab && uv run -q --with httpx python build_vocab.py --source keywords_student_v3fix --prefix kw1322v3j_ --boilerplate --synmap kw1322v3_synmap5 --min-lp -1.2 --floor 100 --from-step 2 | tail -1
cd $S/synonyms
echo "$(date '+%T') audit j seed 29"; uv run -q --with anthropic --with httpx python audit.py --prefix kw1322v3j_ --n 1000 --seed 29 | head -1; cp data/audit_kw1322v3j_.json data/audit_j_seed29.json
echo "$(date '+%T') audit j seed 30"; uv run -q --with anthropic --with httpx python audit.py --prefix kw1322v3j_ --n 1000 --seed 30 | head -1
echo "$(date '+%T') ROUND5 DONE"
