"""#1476: upload finished corpus buckets ($SENSES_OUT/b*.jsonl) to Volume openalex.transient.kw1476_vol/out/ and
rebuild openalex.transient.kw1476_jev_raw (work_id, heading, p map<string,double>, model, ok, err) from all uploaded files. Idempotent: skips
files already in the Volume. uv run --with httpx python load_out.py"""
import glob, os, subprocess, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'common')); from dbx import sql
OUT = os.environ.get('SENSES_OUT', 'out'); VOL = 'dbfs:/Volumes/openalex/transient/kw1476_vol/out'
subprocess.run(['databricks', 'volumes', 'create', 'openalex', 'transient', 'kw1476_vol', 'MANAGED'], capture_output=True)
subprocess.run(['databricks', 'fs', 'mkdir', VOL], check=True)
have = set(subprocess.run(['databricks', 'fs', 'ls', VOL], capture_output=True, text=True).stdout.split())
new = [f for f in sorted(glob.glob(f'{OUT}/b*.jsonl')) if os.path.basename(f) not in have]
for f in new: subprocess.run(['databricks', 'fs', 'cp', f, f'{VOL}/{os.path.basename(f)}'], check=True, capture_output=True)
print(len(new), 'uploaded;', len(have) + len(new), 'in the Volume')
sql("""CREATE OR REPLACE TABLE openalex.transient.kw1476_jev_raw AS
SELECT CAST(work_id AS BIGINT) AS work_id, heading, p, model, COALESCE(ok, true) AS ok, err, _metadata.file_name AS file
FROM read_files('/Volumes/openalex/transient/kw1476_vol/out/', format => 'json', schemaHints => 'work_id string, heading string, p map<string,double>, model string, ok boolean, err string')""")
print(sql("SELECT count(*) n, count(DISTINCT file) files, count_if(NOT ok) failed FROM openalex.transient.kw1476_jev_raw"))
