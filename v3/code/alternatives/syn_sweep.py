"""Full synonym sweep: GPT-6.1 Sol (OpenRouter), prompt = syn_gen.py SYS + WD_HINT (the bake-off winner "sol_wd": 94% of its
synonyms judged exact by both Opus 5.5 and GPT-6 Astra), 10 keywords per call, reasoning low, over syn_input.jsonl (1.88M keywords).
- Checkpoint: syn_full.jsonl rows carry prompt_version; a rerun skips (kid, PROMPT_VERSION) already written, never overwrites.
- Streams the input in chunks of 20,000 keywords; per-chunk progress (count, failures, cost, rate, ETA) to stdout + syn_full.log;
  failed batches to syn_full.errors.jsonl (rerun = retry them).
- Stops before spending more than COST_CAP. Pauses while the machine has under 50% of its memory free (local memory guard, macOS
  `memory_pressure`; returns 100 elsewhere, so the guard is a no-op).
  nohup uv run --with httpx python syn_sweep.py > syn_full.out 2>&1 &"""
import asyncio, json, os, re, subprocess, sys, time
import httpx
from syn_gen import SYS, WD_HINT, SCHEMA, fmt
# OpenRouter reserves each request's worst-case cost (max_tokens x price) against the account balance, so max_tokens is capped and
# 402 (balance top-up pending) is retried after 30 s. During one 24-minute stretch of the production run, 264,162 keywords went
# to the OpenAI API directly (same model, same prompt and settings); their rows carry model "gpt-6.1-sol" instead of "openai/gpt-6.1-sol".
MODEL, PROMPT_VERSION, COST_CAP, CONC, CHUNK = "openai/gpt-6.1-sol", "syn_v1_sol_wd", 1200.0, 120, 20000
MAX_TOKENS = 4000   # outputs run ~450 tokens + ~70 reasoning for 10 keywords
_client = None
async def oa_json(system, user, timeout=300, schema=None):
    """OpenRouter call -> (parsed json or None, cost $, status, served model). Retries 402 (top-up pending), 429, 5xx, transport errors."""
    global _client
    if _client is None: _client = httpx.AsyncClient(timeout=timeout, limits=httpx.Limits(max_connections=CONC + 20, max_keepalive_connections=CONC + 20))
    body = {"model": MODEL, "max_tokens": MAX_TOKENS, "reasoning": {"effort": "low"}, "usage": {"include": True},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "response_format": {"type": "json_schema", "json_schema": {"name": "out", "strict": True, "schema": schema or SCHEMA}}}
    for t in range(8):
        try:
            r = await _client.post("https://openrouter.ai/api/v1/chat/completions", headers={"Authorization": "Bearer " + os.environ["OPENROUTER_API_KEY"]}, json=body)
        except httpx.TransportError:
            await asyncio.sleep(2 ** t); continue
        if r.status_code == 402: await asyncio.sleep(30); continue
        if r.status_code in (429, 500, 502, 503, 504): await asyncio.sleep(2 ** t); continue
        try: d = r.json()
        except Exception: await asyncio.sleep(2 ** t); continue
        if r.status_code != 200 or "choices" not in d: return None, 0, f"err {r.status_code} {str(d)[:200]}", MODEL
        cost = (d.get("usage") or {}).get("cost", 0) or 0
        try: return json.loads(d["choices"][0]["message"]["content"]), cost, "ok", d.get("model")
        except Exception: return None, cost, "bad_json", d.get("model")
    return None, 0, "retries", MODEL
OUT, LOG, ERR = "syn_full.jsonl", "syn_full.log", "syn_full.errors.jsonl"
def log(msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}"; print(line, flush=True); open(LOG, "a").write(line + "\n")
def mem_free():
    try: return int(re.search(r"(\d+)%", subprocess.run(["memory_pressure"], capture_output=True, text=True).stdout.splitlines()[-1]).group(1))
    except Exception: return 100
def load_done():
    done = set()
    if os.path.exists(OUT):
        for l in open(OUT):
            r = json.loads(l)
            if r.get("prompt_version") == PROMPT_VERSION: done.add(r["kid"])
    return done
async def main():
    done = load_done(); total = sum(1 for _ in open("syn_input.jsonl")); spent = 0.0
    if os.path.exists("syn_full.cost"): spent = float(open("syn_full.cost").read())
    log(f"start: {len(done):,} done of {total:,}; spent so far ${spent:.2f}; model {MODEL}; version {PROMPT_VERSION}")
    out = open(OUT, "a"); err = open(ERR, "a"); seen = set(); t0 = time.time(); n_new = 0; sem = asyncio.Semaphore(CONC)
    async def one(b):
        nonlocal spent
        async with sem:
            res, cost, st, served = await oa_json(SYS + WD_HINT, "\n\n".join(fmt(r, True) for r in b))
        spent += cost
        if not res:
            err.write(json.dumps({"kids": [r["kid"] for r in b], "status": st}) + "\n"); return 0, len(b)
        got = {it["kid"]: it["synonyms"] for it in res.get("items", [])}; ok = 0
        for r in b:
            if r["kid"] in got:
                out.write(json.dumps({"kid": r["kid"], "synonyms": got[r["kid"]], "model": served or MODEL, "prompt_version": PROMPT_VERSION}, ensure_ascii=False) + "\n"); ok += 1
            else:
                err.write(json.dumps({"kids": [r["kid"]], "status": "missing_in_response"}) + "\n")
        return ok, len(b) - ok
    f = open("syn_input.jsonl"); chunk_no = 0
    while True:
        rows = []
        for l in f:
            r = json.loads(l)
            if r["kid"] in done or r["kid"] in seen: continue
            seen.add(r["kid"]); rows.append(r)
            if len(rows) >= CHUNK: break
        if not rows: break
        while mem_free() < 50:
            log(f"memory under 50% free ({mem_free()}%): pausing 60 s"); await asyncio.sleep(60)
        if spent >= COST_CAP:
            log(f"cost cap ${COST_CAP:.0f} reached (${spent:.2f}): stopping"); break
        c0 = time.time()
        res = await asyncio.gather(*(one(rows[i:i + 10]) for i in range(0, len(rows), 10)))
        out.flush(); err.flush(); open("syn_full.cost", "w").write(f"{spent:.4f}")
        ok = sum(r[0] for r in res); bad = sum(r[1] for r in res); n_new += ok; chunk_no += 1
        rate = n_new / (time.time() - t0); left = total - len(done) - n_new
        log(f"chunk {chunk_no}: +{ok:,} ok, {bad:,} failed, {time.time() - c0:.0f}s | total {len(done) + n_new:,}/{total:,} | ${spent:.2f} | {rate:.0f}/s | ETA {left / max(rate, 1e-9) / 3600:.1f} h")
    log(f"done: {len(done) + n_new:,} keywords with synonyms; ${spent:.2f}")
if __name__ == "__main__": asyncio.run(main())
