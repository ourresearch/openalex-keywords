"""Synonym bake-off, generation. Every model gets the same prompt, 10 keywords per call (name + description), reasoning
effort low, strict JSON schema, via OpenRouter. Arms ending in _wd also get the keyword's Wikidata aliases as hints (WD_HINT).
Resumable: skips keywords already in syn_gen_<tag>.jsonl. SYS, WD_HINT, SCHEMA and fmt are reused by the full sweep (syn_sweep.py).
  uv run --with httpx python syn_gen.py [tag ...]     (default: all MODELS)"""
import asyncio, json, os, sys, time
from or_util import or_json
MODELS = {"luna": "openai/gpt-6-luna", "luna_pro": "openai/gpt-6-luna-pro", "gemini_flash": "google/gemini-3.8-flash",
          "sol": "openai/gpt-6.1-sol", "sonnet": "anthropic/claude-sonnet-5.5", "opus": "anthropic/claude-opus-5.5", "sol_wd": "openai/gpt-6.1-sol", "sol6_wd": "openai/gpt-6-sol"}   # *_wd: Wikidata aliases given as hints
SYS = """You write search synonyms for keywords in OpenAlex, an index of scholarly works. People type into a keyword search and autocomplete box; a synonym is any string a researcher would plausibly type when they mean exactly this keyword's concept, so the keyword can be found by it.

Include:
- exact synonyms and alternative names for the same concept;
- abbreviations and acronyms in common use for this concept, and expansions of the keyword if it is itself an abbreviation;
- spelling variants (British/American, hyphenation, spacing, Greek letters spelled out, singular/plural where people type both);
- everyday or lay terms for a technical concept ("heart attack" for myocardial infarction);
- common alternative wordings and word orders; former or alternative names.

Exclude: broader or narrower concepts; related but different concepts; translations into other languages; the keyword itself or a case-only change of it; anything invented. Keep the same specificity as the keyword. Use the description to pick the right sense.

Give 0 to 12 synonyms per keyword, the most likely to be typed first. Give none rather than pad."""
SCHEMA = {"type": "object", "additionalProperties": False, "required": ["items"], "properties": {"items": {"type": "array", "items": {
    "type": "object", "additionalProperties": False, "required": ["kid", "synonyms"],
    "properties": {"kid": {"type": "string"}, "synonyms": {"type": "array", "items": {"type": "string"}}}}}}}
WD_HINT = "\n\nSome keywords list Wikidata aliases. They are human-written but noisy (some are broader, narrower, other-language or wrong): keep the ones that qualify as synonyms under the rules above and drop the rest."
def fmt(r, wd=False):
    s = f"id: {r['kid']}\nkeyword: {r['name']}\ndescription: {r['description'] or '(none)'}"
    al = [a for a in r.get("wd_aliases", []) + ([r["wd_label"]] if r.get("wd_label") else []) if a.strip().lower() != r["name"].strip().lower()]
    return s + (f"\nWikidata aliases: {'; '.join(al)}" if wd and al else "")
async def run(tag, rows, conc=12):
    path = f"syn_gen_{tag}.jsonl"; done = set()
    if os.path.exists(path): done = {json.loads(l)["kid"] for l in open(path)}
    todo = [r for r in rows if r["kid"] not in done]; batches = [todo[i:i + 10] for i in range(0, len(todo), 10)]
    sem = asyncio.Semaphore(conc); out = open(path, "a"); stats = {"cost": 0.0, "ok": 0, "fail": 0}; t0 = time.time()
    async def one(b):
        async with sem:
            wd = tag.endswith("_wd")
            res, cost, st, served = await or_json(MODELS[tag], SYS + (WD_HINT if wd else ""), "\n\n".join(fmt(r, wd) for r in b), SCHEMA, reasoning="low", timeout=300)
        stats["cost"] += cost
        if not res: stats["fail"] += len(b); print(tag, "fail", st, flush=True); return
        got = {it["kid"]: it["synonyms"] for it in res.get("items", [])}
        for r in b:
            if r["kid"] in got:
                out.write(json.dumps({"kid": r["kid"], "synonyms": got[r["kid"]], "model": served or MODELS[tag]}, ensure_ascii=False) + "\n"); stats["ok"] += 1
            else: stats["fail"] += 1
        out.flush()
    await asyncio.gather(*(one(b) for b in batches))
    print(f"{tag}: {stats['ok']} ok, {stats['fail']} failed, ${stats['cost']:.2f}, {time.time() - t0:.0f}s", flush=True)
    json.dump(stats, open(f"syn_gen_{tag}.cost.json", "w"))
async def main():
    rows = [json.loads(l) for l in open("syn_sample.jsonl")]
    await asyncio.gather(*(run(t, rows) for t in (sys.argv[1:] or MODELS)))
if __name__ == "__main__": asyncio.run(main())
