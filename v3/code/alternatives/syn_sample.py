"""Synonym bake-off sample. 400 keywords stratified by works_count (150 head >= 20K works, 150 torso 1K-20K, 100 tail < 1K;
seed 1464) + the gold keywords of the 289-query paraphrase eval (eval_all.json). Pulls each keyword's description + Wikidata id
from openalex.common.keywords_v2_wikidata_description, then English label + aliases from the Wikidata API.
Writes syn_sample.jsonl {kid, name, n, description, qid, wd_label, wd_aliases, stratum}.  uv run --with httpx python syn_sample.py"""
import json, os, random, sys, time, httpx
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'common')); from dbx import sql
random.seed(1464)
V = {}
for l in open("vocab10.jsonl"):
    r = json.loads(l); V[r["kid"]] = r
head = [k for k, r in V.items() if r["n"] >= 20000]; torso = [k for k, r in V.items() if 1000 <= r["n"] < 20000]; tail = [k for k, r in V.items() if 0 < r["n"] < 1000]
pick = {k: "head" for k in random.sample(head, 150)} | {k: "torso" for k in random.sample(torso, 150)} | {k: "tail" for k in random.sample(tail, 100)}
for e in json.load(open("eval_all.json")):
    for g in e["gold"]:
        if g in V: pick.setdefault(g, "eval_gold")
ids = ",".join(f"'https://openalex.org/keywords/{k}'" for k in pick)
rows = sql(f"SELECT id, wikidata, description FROM openalex.common.keywords_v2_wikidata_description WHERE id IN ({ids})", verbose=False)
meta = {r["id"].rsplit("/", 1)[1]: r for r in rows}
qids = {k: (meta.get(k, {}).get("wikidata") or "").rsplit("/", 1)[-1] for k in pick}
wd = {}
todo = sorted({q for q in qids.values() if q.startswith("Q")})
with httpx.Client(timeout=60, headers={"User-Agent": "OpenAlex keyword alternatives (https://github.com/ourresearch/openalex-keywords)"}) as h:
    for i in range(0, len(todo), 50):
        d = h.get("https://www.wikidata.org/w/api.php", params={"action": "wbgetentities", "ids": "|".join(todo[i:i + 50]), "props": "labels|aliases", "languages": "en", "format": "json"}).json()
        for q, ent in d.get("entities", {}).items():
            wd[q] = ((ent.get("labels", {}).get("en") or {}).get("value"), [a["value"] for a in ent.get("aliases", {}).get("en", [])])
        time.sleep(0.5)
with open("syn_sample.jsonl", "w") as f:
    for k, s in pick.items():
        q = qids[k]; lab, al = wd.get(q, (None, []))
        f.write(json.dumps({"kid": k, "name": V[k]["name"], "n": V[k]["n"], "description": meta.get(k, {}).get("description") or "",
                            "qid": q or None, "wd_label": lab, "wd_aliases": al, "stratum": s}, ensure_ascii=False) + "\n")
import collections
print(len(pick), collections.Counter(pick.values()), "| with QID:", sum(1 for q in qids.values() if q), "| with Wikidata aliases:", sum(1 for q in qids.values() if wd.get(q, (0, []))[1]))
