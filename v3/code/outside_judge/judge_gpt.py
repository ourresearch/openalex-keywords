"""Outside judge for the launch accuracy numbers (independent of the Claude teacher). Written 2026-09-30.
Model: gpt-6-astra (OpenAI's flagship per developers.openai.com/api/docs/models, "Our most capable model
for the most demanding work"), Responses API, reasoning effort high, strict json_schema output.

Tasks (checkpoint judgments.jsonl keyed by key, judge_version JV; resumable, skips done keys, retries errors):
  single  key = A|<work_id>|old or A|<work_id>|new : one keyword list, per-keyword accurate/inaccurate
  pref    key = P|<work_id>  : both lists as A/B (random order seeded by work id), which describes the work better
  amr     key = R|<kid>|<id> : "Is this work substantially about antimicrobial resistance?" yes/partly/no.
          This pre-swap AMR sample is not released (its records carry work text); the launch post uses the
          post-swap measurement in postswap_judge.py instead. Its verdicts are in judgments.jsonl (R| keys).
Input: data/eval/outside_judge/sample_a.jsonl ships without titles and abstracts; run fetch_text.py first.
Needs OPENAI_API_KEY (or point the OpenAI client at OpenRouter with model "openai/gpt-6-astra").
Usage: judge_gpt.py single|pref|amr [limit]
"""
import json, os, sys, random, threading, time
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI

JV = 1
MODEL = "gpt-6-astra"
EFFORT = "high"
PRICE_IN, PRICE_OUT = 10.0, 50.0  # $/MTok, developers.openai.com/api/docs/models, 2026-09-30
client = OpenAI(max_retries=6, timeout=600)
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "..", "data", "eval", "outside_judge")
OUT = os.path.join(DATA, "judgments.jsonl")

RUBRIC = """Keywords in this index are meant to name what a work is ABOUT: its subjects, phenomena, the methods it is about, organisms, materials, diseases, places studied, people or texts it studies. They are used to aggregate what authors, journals, institutions and fields work on, and to help people find the work.

For each keyword, decide:
- accurate: it correctly and meaningfully describes what this work is about (a central or clearly present subject, method, entity, place, etc.). A broad-but-correct field term is accurate only if it is still informative for this work.
- inaccurate: any of
  - wrong_or_unsupported: the work is not about this, or the record gives no basis for it (plausible guesses beyond the text count as unsupported);
  - wrong_sense: the term is from the wrong field or word sense (e.g. "Transcription (linguistics)" on a gene transcription paper);
  - too_generic: so broad it tells a reader nothing about this work (e.g. "Biology", "Business", "Science" on a specific study);
  - not_aboutness: document type, venue, publisher boilerplate, or metadata rather than subject (e.g. "journal article", "peer review" on an ordinary paper, a bare year);
  - malformed: garbled or broken text, a stopword, or not a usable term.
Separately, mark redundant=true if the keyword is a near-duplicate of an earlier keyword in the same list (it can still be accurate).

Judge only from the record shown. If there is no abstract, judge from title, venue and type, and give the benefit of the doubt only where the title makes the subject clear."""

SINGLE_SCHEMA = {"type": "object", "additionalProperties": False, "properties": {
    "keywords": {"type": "array", "items": {"type": "object", "additionalProperties": False, "properties": {
        "n": {"type": "integer"}, "keyword": {"type": "string"},
        "verdict": {"type": "string", "enum": ["accurate", "inaccurate"]},
        "problem": {"type": "string", "enum": ["none", "wrong_or_unsupported", "wrong_sense", "too_generic", "not_aboutness", "malformed"]},
        "redundant": {"type": "boolean"}},
        "required": ["n", "keyword", "verdict", "problem", "redundant"]}},
    "note": {"type": "string"}}, "required": ["keywords", "note"]}

PREF_SCHEMA = {"type": "object", "additionalProperties": False, "properties": {
    "preferred": {"type": "string", "enum": ["A", "B", "tie", "neither_useful"]},
    "reason": {"type": "string"}}, "required": ["preferred", "reason"]}

AMR_SCHEMA = {"type": "object", "additionalProperties": False, "properties": {
    "verdict": {"type": "string", "enum": ["yes", "partly", "no"]},
    "reason": {"type": "string"}}, "required": ["verdict", "reason"]}

def record(r):
    return (f"Type: {r['type']}; Year: {r['publication_year']}; Language: {r['language']}\n"
            f"Venue/repository: {r['source_name'] or '(unknown)'}\n"
            f"Title: {(r['title'] or '(none)')[:400]}\n"
            f"Abstract/description: {r['abstract'] or '(none available)'}")

def call(instructions, prompt, schema, name):
    resp = client.responses.create(model=MODEL, reasoning={"effort": EFFORT}, instructions=instructions, input=prompt,
        text={"format": {"type": "json_schema", "name": name, "strict": True, "schema": schema}})
    u = resp.usage
    cost = (u.input_tokens * PRICE_IN + u.output_tokens * PRICE_OUT) / 1e6
    meta = {"status": resp.status, "in_tok": u.input_tokens, "out_tok": u.output_tokens,
            "reasoning_tok": u.output_tokens_details.reasoning_tokens, "cost": cost}
    txt = resp.output_text
    # refusals surface as a refusal content item, not text
    for item in resp.output:
        for c in getattr(item, "content", None) or []:
            if getattr(c, "type", "") == "refusal":
                meta["refusal"] = c.refusal
    return (json.loads(txt) if txt else None), meta

def task_single(r, which):
    kws = r[f"{which}_kw"]
    lst = "\n".join(f"{i+1}. {k}" for i, k in enumerate(kws))
    prompt = (f"Record:\n{record(r)}\n\nKeyword list ({len(kws)} keywords):\n{lst}\n\n"
              f"Return one entry per keyword, in order, with n = its number.")
    j, meta = call(RUBRIC, prompt, SINGLE_SCHEMA, "keyword_judgments")
    if j is not None:
        got = sorted(x["n"] for x in j["keywords"])
        if got != list(range(1, len(kws) + 1)):
            raise ValueError(f"parse: expected {len(kws)} items, got {got[:5]}.. ({len(got)})")
    return j, meta

PREF_INSTR = ("You compare two keyword lists attached to the same scholarly record in a large bibliographic index. "
    "Keywords should name what the work is about (subjects, phenomena, methods studied, organisms, materials, places, "
    "entities). Bad keywords are wrong or unsupported, the wrong word sense, too generic to be informative, document "
    "type / venue / boilerplate, malformed, or near-duplicates. Judge only from the record shown.")

def task_pref(r):
    a_is_new = random.Random(r["work_id"] + "gpt1322").random() < 0.5
    A, B = (r["new_kw"], r["old_kw"]) if a_is_new else (r["old_kw"], r["new_kw"])
    prompt = (f"Record:\n{record(r)}\n\nList A: {'; '.join(A)}\n\nList B: {'; '.join(B)}\n\n"
              "Which list better describes what this work is about? Answer A, B, tie (about equally good), "
              "or neither_useful (both are poor). reason: one sentence.")
    j, meta = call(PREF_INSTR, prompt, PREF_SCHEMA, "preference")
    meta["a_is_new"] = a_is_new
    return j, meta

AMR_SYSTEM = ("You judge whether a scholarly work is about a given subject, for a precision audit of a subject index. "
          "Answer 'yes' if the subject is a main topic of the work (a reader searching for the subject would want it), "
          "'partly' if the subject is a clear secondary topic or substantial component, 'no' if it is absent, only "
          "mentioned in passing, or the work is about a different sense of the words. Judge only from the record "
          "given; if there is no abstract, judge from title and venue and say so when that makes you unsure.")

def task_amr(r):
    subj = "antimicrobial resistance"
    rec = f'Title: {r["title"] or "(none)"}\nVenue: {r["venue"] or "(unknown)"}\nType: {r["type"]}; year: {r["year"]}\n'
    rec += f'Abstract: {r["abstract"]}' if r["abstract"] else "Abstract: (none available)"
    msg = f'Subject: "{subj}"\n\nWork:\n{rec}\n\nIs this work substantially about "{subj}"? Give verdict and a one-line reason.'
    return call(AMR_SYSTEM, msg, AMR_SCHEMA, "verdict")

def build(mode):
    items = []
    if mode in ("single", "pref"):
        rows = [json.loads(l) for l in open(os.path.join(DATA, "sample_a_text.jsonl"))][:400]
        for r in rows:
            if mode == "single":
                for which in ("old", "new"):
                    items.append((f"A|{r['work_id']}|{which}", lambda r=r, w=which: task_single(r, w), {"work_id": r["work_id"], "list": which}))
            else:
                items.append((f"P|{r['work_id']}", lambda r=r: task_pref(r), {"work_id": r["work_id"]}))
        # interleave old/new so a partial run is balanced; shuffle deterministically
        random.Random(7).shuffle(items)
    else:
        seen = set()
        for fn in ("samples.jsonl", "samples_ext.jsonl"):
            for l in open(os.path.join(HERE, "..", "recall_precision", fn)):
                r = json.loads(l)
                if r["kid"] != "https://openalex.org/keywords/antimicrobial-resistance": continue
                k = f"R|{r['kid']}|{r['id']}"
                if k in seen: continue
                seen.add(k)
                items.append((k, lambda r=r: task_amr(r), {"work_id": r["id"], "stratum": r["stratum"], "noabs": r["noabs"]}))
    return items

if __name__ == "__main__":
    mode = sys.argv[1]; limit = int(sys.argv[2]) if len(sys.argv) > 2 else None
    done = set()
    if os.path.exists(OUT):
        for l in open(OUT):
            x = json.loads(l)
            if x.get("judge_version") == JV and x.get("result") is not None:
                done.add(x["key"])
    items = build(mode)
    if limit: items = items[:limit]
    todo = [it for it in items if it[0] not in done]
    print(f"{mode}: {len(items)} items, {len(todo)} to do", flush=True)
    lock = threading.Lock(); cost = [0.0]; out = open(OUT, "a")
    def one(it):
        key, fn, info = it
        row = {"key": key, "judge_version": JV, "model": MODEL, "effort": EFFORT, **info,
               "ts": time.strftime("%Y-%m-%d %H:%M:%S CT")}
        for attempt in range(3):
            try:
                j, meta = fn()
                row.update(meta); row["result"] = j; row.pop("error", None)
                if j is None: row["error"] = f"no output ({meta.get('status')}, refusal={meta.get('refusal')})"
                break
            except Exception as e:
                row["error"] = repr(e)[:400]; row["result"] = None
        with lock:
            cost[0] += row.get("cost", 0)
            out.write(json.dumps(row) + "\n"); out.flush()
    with ThreadPoolExecutor(16) as ex:
        for i, _ in enumerate(ex.map(one, todo)):
            if i % 50 == 0: print(i, f"${cost[0]:.2f}", flush=True)
    print(f"done; this run ${cost[0]:.2f}", flush=True)
