"""Run SQL on Databricks and pull the whole result to desk. Uses the CLI for auth
(`databricks api ...`), EXTERNAL_LINKS disposition so large results stream via presigned URLs.

    from dbx import sql
    rows = sql("SELECT ...")            # list[dict]
"""
import json, os, subprocess, time, httpx

WAREHOUSE = os.environ.get("DATABRICKS_WAREHOUSE_ID")  # any SQL warehouse id in your workspace


def _api(method, path, body=None):
    cmd = ["databricks", "api", method, path]
    if body is not None:
        cmd += ["--json", json.dumps(body)]
    out = subprocess.run(cmd, capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(out.stderr[:800])
    return json.loads(out.stdout) if out.stdout.strip() else {}


def sql(query, warehouse=WAREHOUSE, timeout_s=3600, verbose=True):
    st = _api("post", "/api/2.0/sql/statements", {
        "warehouse_id": warehouse, "statement": query, "wait_timeout": "30s",
        "disposition": "EXTERNAL_LINKS", "format": "JSON_ARRAY",
    })
    sid = st["statement_id"]
    t0 = time.time()
    while st["status"]["state"] in ("PENDING", "RUNNING"):
        if time.time() - t0 > timeout_s:
            raise TimeoutError(sid)
        time.sleep(3)
        st = _api("get", f"/api/2.0/sql/statements/{sid}")
    if st["status"]["state"] != "SUCCEEDED":
        raise RuntimeError(json.dumps(st["status"])[:1500])
    man = st.get("manifest") or {}
    cols = [c["name"] for c in (man.get("schema") or {}).get("columns", [])]
    n_chunks = man.get("total_chunk_count", 0)
    rows = []
    with httpx.Client(timeout=120) as h:
        for i in range(n_chunks):
            ch = _api("get", f"/api/2.0/sql/statements/{sid}/result/chunks/{i}")
            for link in ch.get("external_links", []):
                data = h.get(link["external_link"]).json()
                rows += [dict(zip(cols, r)) for r in data]
    if verbose:
        print(f"dbx: {len(rows)} rows, {n_chunks} chunks, {time.time()-t0:.0f}s", flush=True)
    return rows


if __name__ == "__main__":
    print(sql("SELECT 1 AS x, 'a' AS y"))
