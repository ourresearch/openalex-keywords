"""Synonym bake-off, judging. Per keyword, pool every candidate from every arm (8 model arms + Wikidata English aliases/label),
dedupe case-insensitively, shuffle, and have two judges from different families label each one blind (no source shown):
SAME = someone typing it means exactly this keyword (synonym, abbreviation, spelling variant, lay term; same specificity)
CLOSE = broader, narrower or related: a misleading synonym; WRONG = different concept or nonsense.
5 keywords per call, reasoning medium, strict JSON. Resumable per judge: syn_judge_<judge>.jsonl {kid, labels:{cand: label}}.
  uv run --with httpx python syn_judge.py"""
import asyncio, json, os, random, sys, time
from or_util import or_json
JUDGES = {"opus": "anthropic/claude-opus-5.5", "astra": "openai/gpt-6-astra"}
ARMS = ["luna", "luna_pro", "gemini_flash", "sol", "sonnet", "opus", "sol_wd", "sol6_wd"]
SYS = """You check proposed search synonyms for keywords in OpenAlex, an index of scholarly works. A researcher types a string into a keyword search box; it should find a keyword only if they mean exactly that keyword's concept. For each proposed synonym, answer:
- SAME: someone typing it means exactly this keyword's concept at the same specificity. Synonyms, alternative names, abbreviations or acronyms in real use for this concept (even if they also mean other things elsewhere), spelling/hyphenation/plural variants, lay terms, reworded phrases all count.
- CLOSE: a broader, narrower or merely related concept. Useful context, but a misleading synonym.
- WRONG: a different concept, a wrong sense, a non-English translation, or nonsense.
Use the description to pin the sense. Label every proposed string exactly as written."""
SCHEMA = {"type": "object", "additionalProperties": False, "required": ["items"], "properties": {"items": {"type": "array", "items": {
    "type": "object", "additionalProperties": False, "required": ["kid", "labels"], "properties": {"kid": {"type": "string"}, "labels": {"type": "array", "items": {
        "type": "object", "additionalProperties": False, "required": ["synonym", "label"],
        "properties": {"synonym": {"type": "string"}, "label": {"type": "string", "enum": ["SAME", "CLOSE", "WRONG"]}}}}}}}}}
def pool():
    S = {json.loads(l)["kid"]: json.loads(l) for l in open("syn_sample.jsonl")}
    G = {a: {json.loads(l)["kid"]: json.loads(l)["synonyms"] for l in open(f"syn_gen_{a}.jsonl")} for a in ARMS}
    P = {}
    for k, s in S.items():
        c = {}
        for a in ARMS:
            for x in G[a].get(k, []): c.setdefault(x.strip().lower(), x.strip())
        for x in s["wd_aliases"] + ([s["wd_label"]] if s["wd_label"] else []): c.setdefault(x.strip().lower(), x.strip())
        c.pop(s["name"].strip().lower(), None)
        v = list(c.values()); random.Random(k).shuffle(v); P[k] = v
    return S, G, P
def fmt(s, cands): return f"id: {s['kid']}\nkeyword: {s['name']}\ndescription: {s['description'] or '(none)'}\nproposed synonyms:\n" + "\n".join(f"- {c}" for c in cands)
async def run(j, S, P, conc=10):
    path = f"syn_judge_{j}.jsonl"; have = {}
    if os.path.exists(path):
        for l in open(path):
            r = json.loads(l); have.setdefault(r["kid"], set()).update(x.strip().lower() for x in r["labels"])
    P = {k: [c for c in v if c.strip().lower() not in have.get(k, set())] for k, v in P.items()}   # only unlabelled candidates
    todo = [k for k in P if P[k]]; batches = [todo[i:i + 5] for i in range(0, len(todo), 5)]
    sem = asyncio.Semaphore(conc); out = open(path, "a"); st = {"cost": 0.0, "ok": 0, "fail": 0, "tokens_out": 0}; t0 = time.time()
    async def one(b):
        async with sem:
            res, cost, status, served = await or_json(JUDGES[j], SYS, "\n\n".join(fmt(S[k], P[k]) for k in b), SCHEMA, reasoning="medium", timeout=600)
        st["cost"] += cost
        if not res: st["fail"] += len(b); print(j, "fail", status, flush=True); return
        for it in res.get("items", []):
            if it["kid"] in b:
                out.write(json.dumps({"kid": it["kid"], "labels": {x["synonym"]: x["label"] for x in it["labels"]}, "judge": served or JUDGES[j]}, ensure_ascii=False) + "\n"); st["ok"] += 1
        out.flush()
    await asyncio.gather(*(one(b) for b in batches))
    print(f"{j}: {st['ok']} ok, {st['fail']} failed, ${st['cost']:.2f}, {time.time() - t0:.0f}s", flush=True)
if __name__ == "__main__":
    S, G, P = pool()
    print(len(P), "keywords,", sum(len(v) for v in P.values()), "unique candidates", flush=True)
    async def main(): await asyncio.gather(*(run(j, S, P) for j in (sys.argv[1:] or JUDGES)))
    asyncio.run(main())
