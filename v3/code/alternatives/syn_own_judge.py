"""One owner per synonym, step 2: settle collisions.jsonl (a synonym claimed by several keywords) with GPT-6.1 Sol, 10 groups per
call (same model, endpoint and settings as the sweep: syn_sweep.oa_json). Rule: no synonym on several keywords; pick the ONE keyword a
person typing the string most likely means. Only a genuine homonym (an abbreviation or term naming clearly different concepts in
different fields, e.g. CRP = C-reactive protein / cAMP receptor protein) keeps every real meaning; near-duplicate keywords never share.
Resumable: own_judged.jsonl {norm, owner, homonym_owners}. Then merges into syn_owned.jsonl -> syn_final.jsonl.
  uv run --with httpx python syn_own_judge.py [--merge-only]"""
import asyncio, json, os, subprocess, sys, time
from syn_sweep import oa_json
SYS = """You assign search synonyms to keywords in OpenAlex, an index of scholarly works. Each item is a string that several keywords claimed as a synonym. A person types the string into a keyword search box: decide which keyword they mean.
- owner: the single keyword they most likely mean (the most standard, exact match for the string). If several candidates are near-duplicates of each other (the same concept worded differently), pick the one with the most works.
- homonym_owners: ONLY if the string is a genuine homonym, i.e. it names clearly different concepts (typically an abbreviation used in different fields), list every other keyword that is a real meaning of it; otherwise an empty list. Near-duplicates, variants and related concepts are never homonyms.
- If no candidate is actually meant by the string, set owner to "none"."""
SCHEMA = {"type": "object", "additionalProperties": False, "required": ["items"], "properties": {"items": {"type": "array", "items": {
    "type": "object", "additionalProperties": False, "required": ["id", "owner", "homonym_owners"],
    "properties": {"id": {"type": "string"}, "owner": {"type": "string"}, "homonym_owners": {"type": "array", "items": {"type": "string"}}}}}}}
def load_desc(kids):
    d = {}
    for l in open("syn_input.jsonl"):
        r = json.loads(l)
        if r["kid"] in kids: d[r["kid"]] = r["description"]
    return d
def fmt(i, g, desc):
    return f"id: {i}\nstring: {g['claims'][0]['text']}\ncandidates:\n" + "\n".join(f"- {c['kid']}: {c['name']} ({c['works']:,} works). {desc.get(c['kid'], '')}" for c in g["claims"])
def mem_free():
    try: return int(subprocess.run(["memory_pressure"], capture_output=True, text=True).stdout.strip().rsplit(" ", 1)[-1].rstrip("%"))
    except Exception: return 100
COST_CAP = float(os.environ.get("OWN_COST_CAP", "450"))
async def judge():
    """Streams collisions.jsonl in chunks of 20K groups (only one chunk + its descriptions in memory); resumable via own_judged.jsonl."""
    done = {json.loads(l)["norm"] for l in open("own_judged.jsonl")} if os.path.exists("own_judged.jsonl") else set()
    spent = [float(open("own_judged.cost").read()) if os.path.exists("own_judged.cost") else 0.0]; bad = [0]
    out = open("own_judged.jsonl", "a"); sem = asyncio.Semaphore(100); n_total = 0
    async def one(b, desc):
        async with sem:
            res, cost, st, _ = await oa_json(SYS, "\n\n".join(fmt(str(i), g, desc) for i, g in enumerate(b)), schema=SCHEMA)
        spent[0] += cost
        if not res: bad[0] += len(b); return
        for it in res["items"]:
            try: g = b[int(it["id"])]
            except Exception: continue
            ok = {c["kid"] for c in g["claims"]}
            out.write(json.dumps({"norm": g["norm"], "owner": it["owner"] if it["owner"] in ok else None,
                                  "homonym_owners": [k for k in it["homonym_owners"] if k in ok and k != it["owner"]]}, ensure_ascii=False) + "\n")
        out.flush()
    async def run(part):
        desc = load_desc({c["kid"] for g in part for c in g["claims"]})
        await asyncio.gather(*(one(part[i:i + 10], desc) for i in range(0, len(part), 10)))
        open("own_judged.cost", "w").write(f"{spent[0]:.4f}")
        print(f"{time.strftime('%H:%M:%S')}  +{len(part):,} groups ({n_total:,} seen), ${spent[0]:.2f}, {bad[0]:,} failed", flush=True)
    part = []; limit = int(os.environ.get("OWN_LIMIT", "0"))
    for l in open("collisions.jsonl"):
        g = json.loads(l); n_total += 1
        if g["norm"] not in done: part.append(g)
        if limit and n_total >= limit: break
        if len(part) >= 20000:
            while mem_free() < 50: print("memory under 50% free: pausing 60 s", flush=True); await asyncio.sleep(60)
            if spent[0] >= COST_CAP: print(f"cost cap ${COST_CAP:.0f} reached: stopping", flush=True); return
            await run(part); part = []
    if part: await run(part)
def merge():
    """Streams syn_owned.jsonl (1.6M keywords) and appends each keyword's won collisions; only the collisions sit in memory."""
    J = {}
    for l in open("own_judged.jsonl"):
        r = json.loads(l); J[r["norm"]] = r
    add = {}
    n_own = n_hom = n_none = 0
    for l in open("collisions.jsonl"):
        g = json.loads(l); j = J.get(g["norm"])
        if not j or not j["owner"]: n_none += 1; continue
        keep = {j["owner"], *j["homonym_owners"]}; n_own += 1; n_hom += bool(j["homonym_owners"])
        for c in g["claims"]:
            if c["kid"] in keep: add.setdefault(c["kid"], []).append({"text": c["text"], "source": c["source"]})
    n_kw = 0
    with open("syn_final.jsonl", "w") as f:
        for l in open("syn_owned.jsonl"):
            r = json.loads(l); r["synonyms"] += add.pop(r["kid"], [])
            f.write(json.dumps(r, ensure_ascii=False) + "\n"); n_kw += 1
        for k, v in add.items(): f.write(json.dumps({"kid": k, "synonyms": v}, ensure_ascii=False) + "\n"); n_kw += 1
    print(f"collisions: {n_own:,} given an owner ({n_hom:,} homonyms kept on several keywords), {n_none:,} dropped (no owner); {n_kw:,} keywords in syn_final.jsonl")
if __name__ == "__main__":
    if "--merge-only" not in sys.argv: asyncio.run(judge())
    merge()
