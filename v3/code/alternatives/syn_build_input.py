"""Inputs for the full synonym sweep: every keywords_v2 keyword with its description (openalex.common.keywords_v2_wikidata_description)
and, where it has a Wikidata id, the English label + aliases from a local Wikidata index (WIKIDATA_SQLITE: an SQLite file with table
items(qid, label, aliases JSON), built from the Wikidata JSON dump of 2026-09-03). Pulled from Databricks in 8 hash chunks and written
straight to syn_input.jsonl {kid, name, description, qid, wd_label, wd_aliases}, so memory stays small.
  WIKIDATA_SQLITE=/path/to/wikidata.sqlite uv run --with httpx python syn_build_input.py"""
import json, os, sqlite3, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'common')); from dbx import sql
wd = sqlite3.connect(os.environ.get("WIKIDATA_SQLITE", "wikidata.sqlite"))
n = nq = na = 0
with open("syn_input.jsonl", "w") as f:
    for i in range(8):
        rows = sql(f"""SELECT v.kid, v.display_name AS name, d.description, d.wikidata FROM openalex.common.keywords_v2 v
                       LEFT JOIN openalex.common.keywords_v2_wikidata_description d ON d.id = v.id WHERE pmod(hash(v.id), 8) = {i}""", verbose=False)
        qids = [r["wikidata"].rsplit("/", 1)[1] for r in rows if r.get("wikidata")]
        W = {}
        for j in range(0, len(qids), 900):
            ch = qids[j:j + 900]
            for q, lab, al in wd.execute(f"SELECT qid, label, aliases FROM items WHERE qid IN ({','.join('?' * len(ch))})", ch):
                W[q] = (lab, json.loads(al) if al else [])
        for r in rows:
            q = r["wikidata"].rsplit("/", 1)[1] if r.get("wikidata") else None
            lab, al = W.get(q, (None, []))
            nq += bool(q); na += bool(al)
            f.write(json.dumps({"kid": r["kid"], "name": r["name"], "description": r["description"] or "", "qid": q, "wd_label": lab, "wd_aliases": al}, ensure_ascii=False) + "\n")
        n += len(rows); print(f"chunk {i}: {n:,} keywords so far, {nq:,} with QID, {na:,} with aliases", flush=True)
        del rows, W
