"""Post-swap on-topic rates for the launch post's two "keyword search finds what text search misses" examples.
Written 2026-10-02, after the API began serving the new keywords. Read-only: API GETs only.

Samples: OQL `<query> sample N` with seed=1322 on api.openalex.org (default, no xpac = what readers see).
Judge: openai/gpt-6-astra via OpenRouter, reasoning effort high, strict json_schema.
Prompt: identical system prompt and message to judge_gpt.py task_amr.
Abstract truncation: AMR 2,500 chars, remote work 4,000.
The released samples.jsonl has no titles or abstracts: `pull` refetches the same seeded samples from the API
(the sets drift as the corpus changes, so a fresh pull will not match exactly; judge then works on the refetched rows).
Checkpoints: samples.jsonl (per set), judgments.jsonl keyed set|work_id; resumable, skips done keys.
Usage: OPENALEX_API_KEY=... OPENROUTER_API_KEY=... python3 postswap_judge.py pull | judge | report
"""
import asyncio, json, math, os, sys, time, urllib.parse, urllib.request

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "eval", "outside_judge", "postswap")
KEY = os.environ.get("OPENALEX_API_KEY", "")
AMR, REMOTE = "antimicrobial resistance", "how remote work affects employee wellbeing"
SEED = 1322
COUNTS = json.load(open(os.path.join(HERE, "counts.json")))
# set -> (topic, n to judge, abstract truncation)
SETS = {"amr_kw_only": (AMR, 150, 2500), "amr_kw_only_noabs": (AMR, 100, 2500), "amr_text": (AMR, 50, 2500),
        "remote_kw_only": (REMOTE, 100, 4000), "remote_phrase": (REMOTE, 50, 4000)}
SAMPLES, OUT = os.path.join(HERE, "samples_text.jsonl"), os.path.join(HERE, "judgments.jsonl")

def oql(q, **p):
    url = "https://api.openalex.org/?" + urllib.parse.urlencode({"oql": q, **p})
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {KEY}"})
    for i in range(5):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.load(r)
        except Exception as e:
            err = e; time.sleep(2 + 3 * i)
    raise err

def abstract(inv, n):
    if not inv: return None
    pos = sorted((i, w) for w, ii in inv.items() for i in ii)
    return " ".join(w for _, w in pos)[:n]

def pull():
    have = {json.loads(l)["set"] for l in open(SAMPLES)} if os.path.exists(SAMPLES) else set()
    with open(SAMPLES, "a") as f:
        for s, (topic, n, trunc) in SETS.items():
            if s in have: continue
            q = f'{COUNTS[s]["oql"]} sample {n}'
            rows = []
            for page in range(1, math.ceil(n / 100) + 1):
                d = oql(q, **{"per-page": min(n, 100), "page": page, "seed": SEED,
                              "select": "id,display_name,abstract_inverted_index,type,publication_year,primary_location"})
                rows += d["results"]
            assert len({w["id"] for w in rows}) == n, (s, len(rows))
            for w in rows:
                src = ((w.get("primary_location") or {}).get("source") or {}).get("display_name")
                f.write(json.dumps({"set": s, "topic": topic, "id": w["id"], "title": w["display_name"], "type": w["type"],
                    "year": w["publication_year"], "venue": src, "abstract": abstract(w["abstract_inverted_index"], trunc)}) + "\n")
            print(s, len(rows), flush=True)

SYSTEM = ("You judge whether a scholarly work is about a given subject, for a precision audit of a subject index. "
          "Answer 'yes' if the subject is a main topic of the work (a reader searching for the subject would want it), "
          "'partly' if the subject is a clear secondary topic or substantial component, 'no' if it is absent, only "
          "mentioned in passing, or the work is about a different sense of the words. Judge only from the record "
          "given; if there is no abstract, judge from title and venue and say so when that makes you unsure.")
SCHEMA = {"type": "object", "additionalProperties": False, "properties": {
    "verdict": {"type": "string", "enum": ["yes", "partly", "no"]},
    "reason": {"type": "string"}}, "required": ["verdict", "reason"]}
MODEL, EFFORT = "openai/gpt-6-astra", "high"
PRICE_IN, PRICE_OUT = 10.0, 50.0  # $/MTok, openrouter.ai/api/v1/models 2026-10-02; fallback if usage.cost absent

def message(r):
    subj = r["topic"]
    rec = f'Title: {r["title"] or "(none)"}\nVenue: {r["venue"] or "(unknown)"}\nType: {r["type"]}; year: {r["year"]}\n'
    rec += f'Abstract: {r["abstract"]}' if r["abstract"] else "Abstract: (none available)"
    return f'Subject: "{subj}"\n\nWork:\n{rec}\n\nIs this work substantially about "{subj}"? Give verdict and a one-line reason.'

async def judge_all(limit=None):
    from openai import AsyncOpenAI
    client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"],
                         max_retries=0, timeout=300)
    done = {json.loads(l)["key"] for l in open(OUT) if '"verdict"' in l} if os.path.exists(OUT) else set()
    items = [json.loads(l) for l in open(SAMPLES)]
    todo = [r for r in items if f'{r["set"]}|{r["id"]}' not in done][:limit]
    print(f"{len(items)} samples, {len(done)} done, {len(todo)} to do", flush=True)
    sem, cost, out = asyncio.Semaphore(8), [0.0], open(OUT, "a")

    async def one(r):
        key = f'{r["set"]}|{r["id"]}'
        row = {"key": key, "set": r["set"], "id": r["id"], "model": MODEL, "effort": EFFORT,
               "has_abstract": bool(r["abstract"]), "ts": time.strftime("%Y-%m-%d %H:%M:%S CT")}
        async with sem:
            for attempt in range(4):
                try:
                    # shortcut: hard wall-clock cap; OpenRouter keep-alives defeat httpx read timeouts
                    resp = await asyncio.wait_for(client.chat.completions.create(
                        model=MODEL, messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": message(r)}],
                        response_format={"type": "json_schema", "json_schema": {"name": "verdict", "strict": True, "schema": SCHEMA}},
                        extra_body={"reasoning": {"effort": EFFORT}, "provider": {"require_parameters": True},
                                    "usage": {"include": True}}), timeout=300)
                    u = resp.usage
                    extra = u.model_extra or {}
                    row["cost"] = extra.get("cost") or (u.prompt_tokens * PRICE_IN + u.completion_tokens * PRICE_OUT) / 1e6
                    row.update(in_tok=u.prompt_tokens, out_tok=u.completion_tokens, served_model=resp.model,
                               provider=(resp.model_extra or {}).get("provider"))
                    row.update(json.loads(resp.choices[0].message.content))
                    row.pop("error", None)
                    break
                except Exception as e:
                    row["error"] = repr(e)[:300]
                    await asyncio.sleep(5 * (attempt + 1))
        cost[0] += row.get("cost", 0)
        out.write(json.dumps(row) + "\n"); out.flush()
        return row

    rows = []
    for i, coro in enumerate(asyncio.as_completed([one(r) for r in todo])):
        rows.append(await coro)
        if i % 25 == 0 or i == len(todo) - 1:
            errs = sum(1 for x in rows if "error" in x)
            print(f"{i + 1}/{len(todo)} ${cost[0]:.2f} errors={errs}", flush=True)
    print(f"done; this run ${cost[0]:.2f}", flush=True)

def wilson(k, n, z=1.96):
    if n == 0: return (0, 0)
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * (c - h), 100 * (c + h))

def report():
    last = {}
    for l in open(OUT):
        x = json.loads(l)
        if "verdict" in x: last[x["key"]] = x
    summ = {}
    for s in SETS:
        v = [x["verdict"] for x in last.values() if x["set"] == s]
        n, y, p = len(v), v.count("yes"), v.count("partly")
        summ[s] = {"n": n, "yes": y, "partly": p, "no": v.count("no"), "count": COUNTS[s]["count"],
                   "yes_pct": round(100 * y / n, 1), "yes_ci": [round(a, 1) for a in wilson(y, n)],
                   "yp_pct": round(100 * (y + p) / n, 1), "yp_ci": [round(a, 1) for a in wilson(y + p, n)]}
        print(s, summ[s])
    json.dump({"sets": summ}, open(os.path.join(HERE, "summary.json"), "w"), indent=1)

if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "pull": pull()
    elif cmd == "judge": asyncio.run(judge_all(int(sys.argv[2]) if len(sys.argv) > 2 else None))
    elif cmd == "report": report()
