#!/bin/zsh
# Rebuild the vocabulary under synmap10 (2026-09-30). Same rules as v3.0 (code/vocabulary/build_vocab.py, unchanged); only the
# synonym map and the source differ. Source = the corpus tags plus the tags of works added after the corpus run (catch-up), so one
# build covers every tagged work. The plural fold is v3.0's, unchanged. Writes <catalog>.transient.kw1322v3o_* only.
# Production then took only the works whose keyword list changed (8,462,750 works for synmap10, 163,429 more for the four hand
# merges), added the new headings and fixed display names; no heading that unchanged works still carry was dropped.
#   DATABRICKS_WAREHOUSE_ID=... zsh rebuild_vocab.sh
set -e
cd "$(dirname "$0")"
uv run -q --with httpx python - <<'PY'
import gzip, json, os, sys, zlib
sys.path.insert(0, '../common'); from dbx import sql
T = 'openalex.transient'
rows = [json.loads(l) for l in gzip.open('../../data/synonyms/synmap10_final.jsonl.gz', 'rt')]
esc = lambda s: s.replace('\\', '\\\\').replace("'", "\\'")
# upload only if the table differs from the file (a full upload is ~72 INSERTs, several minutes)
have = sql(f"SELECT count(*) n, sum(crc32(concat(kid, '|', canon))) h FROM {T}.kw1322v3_synmap10", verbose=False) if sql(f"SHOW TABLES IN {T} LIKE 'kw1322v3_synmap10'", verbose=False) else [{'n': '0', 'h': None}]
want = sum(zlib.crc32(f"{r['kid']}|{r['canon']}".encode()) for r in rows)
if int(have[0]['n']) != len(rows) or str(have[0]['h']) != str(want):
    sql(f"CREATE OR REPLACE TABLE {T}.kw1322v3_synmap10 (kid STRING, canon STRING) COMMENT 'synmap9 minus audited bad merges'", verbose=False)
    for i in range(0, len(rows), 5000):
        sql(f"INSERT INTO {T}.kw1322v3_synmap10 VALUES " + ",".join(f"('{esc(r['kid'])}','{esc(r['canon'])}')" for r in rows[i:i+5000]), verbose=False)
print('synmap10 rows', sql(f"SELECT count(*) n FROM {T}.kw1322v3_synmap10", verbose=False), len(rows))
sql(f"""CREATE OR REPLACE TABLE {T}.kw1465_student_all AS SELECT * FROM {T}.keywords_student_v3fix
        UNION ALL SELECT n.* FROM {T}.keywords_student_v3new n LEFT ANTI JOIN {T}.keywords_student_v3fix f ON f.id = n.id""", verbose=False)
sql(f"""CREATE OR REPLACE TABLE {T}.kw1322v3o_strings0 AS SELECT raw, sum(mentions) AS mentions, kid0 FROM
        (SELECT raw, mentions, kid0 FROM {T}.kw1322v3n_strings0 UNION ALL SELECT raw, mentions, kid0 FROM {T}.kw1322v3new_strings0) GROUP BY raw, kid0""", verbose=False)
sql(f"CREATE OR REPLACE TABLE {T}.kw1322v3o_fold AS SELECT * FROM {T}.kw1322v3n_fold", verbose=False)
print('source / strings0 / fold ready')
PY
echo "$(date '+%T') build o"
uv run -q --with httpx python ../vocabulary/build_vocab.py --source kw1465_student_all --prefix kw1322v3o_ --boilerplate --synmap kw1322v3_synmap10 --min-lp -1.2 --floor 100 --from-step 3
echo "$(date '+%T') BUILD O DONE"
