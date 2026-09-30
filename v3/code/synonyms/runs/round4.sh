#!/bin/zsh
# #1322 synonym round 4 (2026-09-29 overnight): twins judge -> context check -> synmap4 -> rebuild kw1322v3i_* -> audits
set -e
: "${CLAUDE_API_KEY:?}"  # env: CLAUDE_API_KEY (or ANTHROPIC_API_KEY), DATABRICKS_HOST / DATABRICKS_TOKEN (CLI auth), DATABRICKS_WAREHOUSE_ID
S=${KW_SCRATCH:?set KW_SCRATCH to a working copy laid out as the original scratch dir, see README.md here}
cd $S/synonyms
echo "$(date '+%T') twins judge"; uv run -q --with anthropic python judge_pairs2.py --cand data/cand_twins_new.json --vocab data/vocab_h.jsonl --out data/judged_twins.jsonl --workers 40 | tail -1
uv run -q python - <<'PY'
import json
V = {json.loads(l)['kid']: json.loads(l)['works_count'] for l in open('data/vocab_h.jsonl')}
hi = []
for l in open('data/judged_twins.jsonl'):
    r = json.loads(l)
    if not r['merge']: continue
    other = r['b'] if r['keep'] == r['a'] else r['a']
    if V.get(other, 0) >= 10000: hi.append({'kid': other, 'canon': r['keep']})
json.dump(hi, open('data/hi_merges_v4.json', 'w')); print(len(hi), 'high-impact twin merges')
PY
sed -e "s#data/hi_merges.json#data/hi_merges_v4.json#; s#data/hi_examples.json#data/hi_examples_v4.json#; s#data/hi_verified.jsonl#data/hi_verified_v4.jsonl#g; s#data/vocab.jsonl#data/vocab_h.jsonl#" verify_hi.py > verify_hi_v4.py
cd $S/god/corpus_v3 && uv run -q --with httpx python - <<'PY'
import json, sys
sys.path.insert(0, '.'); import dbxr
hi = json.load(open('../../synonyms/data/hi_merges_v4.json')); kids = sorted({r['kid'] for r in hi} | {r['canon'] for r in hi})
ex = {}
if kids:
    vals = ",".join("('keywords/" + k.replace("'", "\\'") + "')" for k in kids)
    dbxr.sql(f"CREATE OR REPLACE TABLE openalex.transient.kw1322v3h_syncheck_kids AS SELECT * FROM VALUES {vals} AS t(keyword_id)", warehouse=__import__('os').environ['DATABRICKS_WAREHOUSE_ID'])
    rows = dbxr.sql("""WITH s AS (SELECT wk.keyword_id, wk.work_id, row_number() OVER (PARTITION BY wk.keyword_id ORDER BY xxhash64(wk.work_id)) r
                   FROM openalex.transient.kw1322v3h_work_keywords wk JOIN openalex.transient.kw1322v3h_syncheck_kids k ON k.keyword_id = wk.keyword_id)
                   SELECT s.keyword_id, w.title, w.primary_location.source.display_name AS venue FROM s JOIN openalex.works.openalex_works w ON w.id = CAST(substr(s.work_id, 2) AS BIGINT) WHERE s.r <= 3""", warehouse=__import__('os').environ['DATABRICKS_WAREHOUSE_ID'])
    for r in rows: ex.setdefault(r['keyword_id'][9:], []).append(f"{(r['title'] or '')[:140]} [{(r['venue'] or '')[:40]}]")
json.dump(ex, open('../../synonyms/data/hi_examples_v4.json', 'w'), ensure_ascii=False); print(len(ex), 'kids with examples')
PY
cd $S/synonyms
echo "$(date '+%T') context check"; uv run -q --with anthropic python verify_hi_v4.py | head -1
python3 build_synmap3.py --extra-judged data/judged_twins.jsonl --extra-maps data/synmap_acronyms_small_ok.jsonl --rejected data/hi_verified_v3.jsonl data/hi_verified_v4.jsonl --out data/synmap4.jsonl
cd $S/god/corpus_v3
uv run -q --with httpx python - <<'PY'
import json, sys; sys.path.insert(0, '.'); import dbxr
rows = [json.loads(l) for l in open('../../synonyms/data/synmap4.jsonl')]
esc = lambda s: s.replace('\\', '\\\\').replace("'", "\\'")
dbxr.sql("CREATE OR REPLACE TABLE openalex.transient.kw1322v3_synmap4 (kid STRING, canon STRING)", warehouse=__import__('os').environ['DATABRICKS_WAREHOUSE_ID'])
for i in range(0, len(rows), 5000):
    dbxr.sql("INSERT INTO openalex.transient.kw1322v3_synmap4 VALUES " + ",".join(f"('{esc(r['kid'])}','{esc(r['canon'])}')" for r in rows[i:i+5000]), warehouse=__import__('os').environ['DATABRICKS_WAREHOUSE_ID'])
dbxr.sql("CREATE OR REPLACE TABLE openalex.transient.kw1322v3i_strings0 AS SELECT * FROM openalex.transient.kw1322v3f_strings0", warehouse=__import__('os').environ['DATABRICKS_WAREHOUSE_ID'])
print('synmap4 rows', len(rows))
PY
echo "$(date '+%T') vocab i"; cd $S/vocab && uv run -q --with httpx python build_vocab.py --source keywords_student_v3fix --prefix kw1322v3i_ --boilerplate --synmap kw1322v3_synmap4 --min-lp -1.2 --floor 100 --from-step 2 | tail -2
cd $S/synonyms
echo "$(date '+%T') audit i (same sample seed 28)"; uv run -q --with anthropic --with httpx python audit.py --prefix kw1322v3i_ --n 1000 --seed 28 | head -1
cp data/audit_kw1322v3i_.json data/audit_i_seed28.json
echo "$(date '+%T') audit i (fresh sample seed 29)"; uv run -q --with anthropic --with httpx python audit.py --prefix kw1322v3i_ --n 1000 --seed 29 | head -1
echo "$(date '+%T') ROUND4 DONE"
