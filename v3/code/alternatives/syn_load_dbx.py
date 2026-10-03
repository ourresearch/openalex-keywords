"""Load syn_final.jsonl into openalex.common.keywords_v2_display_name_alternatives
(id = https://openalex.org/keywords/<kid>, display_name_alternatives ARRAY<STRING>, in each keyword's synonym order)
via a Unity Catalog Volume + COPY INTO (bulk load: one local JSON file, `databricks fs cp`, then COPY INTO). Streams; prints counts.
The API's keywords build (CreateKeywordsAPI) LEFT JOINs this table; the keywords Elasticsearch index then carries the field.
    DATABRICKS_WAREHOUSE_ID=... DATABRICKS_VOLUME=/Volumes/<catalog>/<schema>/<volume> python3 syn_load_dbx.py [--table <table>]"""
import json, os, subprocess, sys, time, uuid
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'common')); from dbx import sql
TABLE = sys.argv[sys.argv.index("--table") + 1] if "--table" in sys.argv else "openalex.common.keywords_v2_display_name_alternatives"
WH = os.environ["DATABRICKS_WAREHOUSE_ID"]; VOL = os.environ["DATABRICKS_VOLUME"]; path = f"{VOL}/kw_alts_{uuid.uuid4().hex[:8]}"
local = "syn_final.dbx.json"; n = 0; t0 = time.time()
with open(local, "w") as f:
    for l in open("syn_final.jsonl"):
        r = json.loads(l)
        if r["synonyms"]:
            f.write(json.dumps({"id": "https://openalex.org/keywords/" + r["kid"], "display_name_alternatives": [s["text"] for s in r["synonyms"]]}, ensure_ascii=False) + "\n"); n += 1
print(f"{n:,} keywords written to {local}", flush=True)
subprocess.run(["databricks", "fs", "mkdir", f"dbfs:{path}"], capture_output=True, text=True)
out = subprocess.run(["databricks", "fs", "cp", local, f"dbfs:{path}/data.json"], capture_output=True, text=True)
if out.returncode: sys.exit(out.stderr[:500])
try:
    sql(f"CREATE OR REPLACE TABLE {TABLE} (id STRING, display_name_alternatives ARRAY<STRING>) COMMENT 'keyword synonyms (GPT-6.1 Sol + Wikidata hints, one owner keyword per synonym); joined by notebooks/keywords/CreateKeywordsAPI'", warehouse=WH, verbose=False)
    sql(f"COPY INTO {TABLE} FROM (SELECT CAST(id AS STRING) AS id, CAST(display_name_alternatives AS ARRAY<STRING>) AS display_name_alternatives FROM '{path}') "
        "FILEFORMAT = JSON FORMAT_OPTIONS ('inferSchema'='true') COPY_OPTIONS ('force'='true')", warehouse=WH, verbose=False)
    r = sql(f"SELECT COUNT(*) n, COUNT(DISTINCT id) ids, SUM(SIZE(display_name_alternatives)) alts FROM {TABLE}", warehouse=WH, verbose=False)[0]
    print(f"{TABLE}: {r} ({time.time() - t0:.0f}s)")
finally:
    subprocess.run(["databricks", "fs", "rm", "-r", f"dbfs:{path}"], capture_output=True, text=True)
