"""Robust Databricks SQL for #1322: submit a statement ONCE, then poll; a poll that times out (the databricks CLI behind dbx._api gives up after 60 s of
inactivity; seen 3x on 2026-09-26) is retried, never the statement. Results via EXTERNAL_LINKS like dbx.sql."""
import os, sys, time, json, httpx
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'common')); import dbx
def _api(method, path, body=None, tries=8):
    for t in range(tries):
        try: return dbx._api(method, path, body)
        except RuntimeError as e:
            if t == tries - 1 or (method != 'get' and 'timed out' not in str(e)): raise   # GETs (polls, chunk links) are always safe to repeat
            time.sleep(min(60, 5 * (t + 1)))
def sql(query, warehouse=os.environ.get('DATABRICKS_WAREHOUSE_ID'), timeout_s=4 * 3600, **_):
    st = _api('post', '/api/2.0/sql/statements', {'warehouse_id': warehouse, 'statement': query, 'wait_timeout': '30s', 'disposition': 'EXTERNAL_LINKS', 'format': 'JSON_ARRAY'}, tries=1)
    sid = st['statement_id']; t0 = time.time()
    while st['status']['state'] in ('PENDING', 'RUNNING'):
        if time.time() - t0 > timeout_s: raise TimeoutError(sid)
        time.sleep(5); st = _api('get', f'/api/2.0/sql/statements/{sid}')
    if st['status']['state'] != 'SUCCEEDED': raise RuntimeError(json.dumps(st['status'])[:1500])
    man = st.get('manifest') or {}; cols = [c['name'] for c in (man.get('schema') or {}).get('columns', [])]; rows = []
    with httpx.Client(timeout=300) as h:
        for i in range(man.get('total_chunk_count', 0)):
            ch = _api('get', f'/api/2.0/sql/statements/{sid}/result/chunks/{i}')
            for link in ch.get('external_links', []): rows += [dict(zip(cols, r)) for r in h.get(link['external_link']).json()]
    return rows
