"""Pull vocab10.jsonl: every keywords_v2 keyword with its works count from openalex.works.work_keywords_v2 ({kid, name, n}).
Pulled in 8 hash chunks to vocab10.new.jsonl, then renamed over vocab10.jsonl (the old copy is kept as vocab10.prev.jsonl).
  uv run --with httpx python vocab_pull.py"""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'common')); from dbx import sql
n = 0
with open("vocab10.new.jsonl", "w") as f:
    for i in range(8):
        rows = sql(f"""WITH c AS (SELECT k.id, count(DISTINCT w.work_id) n FROM openalex.works.work_keywords_v2 w LATERAL VIEW explode(w.keywords) t AS k GROUP BY k.id)
                       SELECT v.kid, v.display_name AS name, coalesce(c.n, 0) AS n FROM openalex.common.keywords_v2 v LEFT JOIN c ON c.id = v.id
                       WHERE pmod(hash(v.id), 8) = {i}""", verbose=False)
        for r in rows: f.write(json.dumps({"kid": r["kid"], "name": r["name"], "n": int(r["n"])}, ensure_ascii=False) + "\n")
        n += len(rows); print(f"chunk {i}: {n:,}", flush=True); del rows
if os.path.exists("vocab10.jsonl"): os.replace("vocab10.jsonl", "vocab10.prev.jsonl")
os.replace("vocab10.new.jsonl", "vocab10.jsonl"); print("done", n)
