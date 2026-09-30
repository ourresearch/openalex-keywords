"""COPY INTO openalex.transient.keywords_student_v3 from the shard files the retag workers PUT on the Databricks Volume (#1322). Only shards the ledger marks finished are
listed (a file being rewritten or cut off by a stopped container breaks the whole COPY, 2026-09-26). Idempotent: COPY INTO skips loaded files; data/loaded_v3.txt records them.
uv run --with httpx python load_v3.py"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); import dbxr as dbx
import argparse; ap = argparse.ArgumentParser(); ap.add_argument('--table', default='keywords_student_v3'); ap.add_argument('--vol', default='/Volumes/openalex/authors/oxjob_tmp/oxjob1322_god/corpus_v3'); ap.add_argument('--ledger', default='ledger_v3.jsonl'); ap.add_argument('--loaded', default='loaded_v3.txt'); a = ap.parse_args()
T = f'openalex.transient.{a.table}'; VOL = a.vol; WH = os.environ.get('DATABRICKS_WAREHOUSE_ID')
dbx.sql(f"CREATE TABLE IF NOT EXISTS {T} (id STRING, keywords ARRAY<STRING>, lp ARRAY<DOUBLE>, model STRING, shard BIGINT)", warehouse=WH, verbose=False)
done = sorted({json.loads(l)['shard'] for l in open(f'{HERE}/data/{a.ledger}')})
lp = f'{HERE}/data/{a.loaded}'; loaded = set(int(x) for x in open(lp).read().split()) if os.path.exists(lp) else set()
todo = [s for s in done if s not in loaded]
for i in range(0, len(todo), 500):
    part = todo[i:i + 500]
    dbx.sql(f"COPY INTO {T} FROM '{VOL}/' FILEFORMAT = JSON FILES = ({', '.join(repr(f'{s:04d}.jsonl.gz') for s in part)}) "
            "FORMAT_OPTIONS ('multiLine' = 'false') COPY_OPTIONS ('mergeSchema' = 'false', 'force' = 'false')", warehouse=WH, timeout_s=3 * 3600, verbose=False)
    with open(lp, 'a') as f: f.write('\n'.join(map(str, part)) + '\n')
    print(f'loaded {len(part)} shards', flush=True)
print(dbx.sql(f"SELECT count(*) n, count(DISTINCT id) ids, count(DISTINCT shard) shards FROM {T}", warehouse=WH, verbose=False)[0])
