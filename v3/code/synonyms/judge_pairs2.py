"""v2 (2026-09-28 evening, Jason: synonyms are stop-ship, the taxonomy must be usable): judges synonym-merge candidates with a searcher-oriented rubric; --model opus or sonnet.
Original: Opus 5.5 judges synonym-merge candidates for the v3 keyword vocabulary (#1322, peer review 2026-09-26). 100 pairs per call; for each pair: merge (same
subject heading for an indexer) and which name is the better heading. Resumable: data/judged.jsonl keyed by pair. Not applied anywhere: the merge map is
built from this by build_synmap.py and applied only with Jason's yes.
uv run --with anthropic python judge_pairs.py [--limit-calls N] [--workers 16]"""
import argparse, json, os, time, random, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import anthropic
HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser(); ap.add_argument('--limit-calls', type=int); ap.add_argument('--workers', type=int, default=16); ap.add_argument('--per-call', type=int, default=100); ap.add_argument('--cand', default='data/cand.json'); ap.add_argument('--vocab', default='data/vocab.jsonl'); ap.add_argument('--out', default='data/judged.jsonl'); ap.add_argument('--model', default='claude-opus-5-5'); a = ap.parse_args()
SYSTEM = """You maintain the subject vocabulary of a scholarly index; a usable vocabulary has ONE heading per concept. Each line gives two keyword headings. Merge (true) when a searcher looking for either would expect the other's works under the same heading, because they name the same subject:
- synonyms and near-synonyms used interchangeably in the literature (older adults / elderly, college students / university students, vitamin C / ascorbic acid, mice / mouse),
- an acronym or abbreviation and its expansion, ONLY when the acronym unambiguously means that expansion across research (IL-6 / interleukin-6, MRI / magnetic resonance imaging, NAFLD / non-alcoholic fatty liver disease); NOT when it has other common meanings (ECE, PD / Pd, DM, PET, ATM) or is a chemical symbol or formula that also reads as something else,
- a renamed concept and its old name (mpox / monkeypox, MASLD / NAFLD),
- spelling, transliteration, hyphenation, word-order or grammatical variants (tumor / tumour, nudging / nudges, CAR-T cells / CAR T cells),
- a Latin taxon and its common name for the same group (Tardigrada / tardigrades).
Do NOT merge (false): different things (World War I / World War II, food safety / food security), broader vs narrower (animals / dogs, deep learning / deep neural networks, monetary policy / monetary policy transmission), a thing vs a therapy or process using it (CAR-T cells / CAR-T cell therapy), a quantity vs its measurement (electron density / electron density measurement), a disease vs the pandemic as a societal event (COVID-19 / COVID-19 pandemic), related but distinct concepts (gender differences / sex differences), proper names that must keep their own form (UK Labour Party, a Latin genus name).
For merges, `keep` is the better heading: the standard, most widely used form in current literature (1 = first, 2 = second).
Return JSON only."""
SCHEMA = {"type": "object", "properties": {"items": {"type": "array", "items": {"type": "object", "properties": {"n": {"type": "integer"}, "merge": {"type": "boolean"}, "keep": {"type": "integer"}},
          "required": ["n", "merge", "keep"], "additionalProperties": False}}}, "required": ["items"], "additionalProperties": False}
name = {}; wc = {}
for l in open(f'{HERE}/{a.vocab}'):
    r = json.loads(l); name[r['kid']] = r['display_name']; wc[r['kid']] = r['works_count']
cand = json.load(open(f'{HERE}/{a.cand}'))
out_path = f'{HERE}/{a.out}'
done = {(r['a'], r['b']) for r in map(json.loads, open(out_path))} if os.path.exists(out_path) else set()
todo = [p for p in cand if (p[0], p[1]) not in done]; random.Random(1322).shuffle(todo)
calls = [todo[i:i + a.per_call] for i in range(0, len(todo), a.per_call)][:a.limit_calls]
client = anthropic.Anthropic(api_key=os.environ.get('CLAUDE_API_KEY') or os.environ.get('ANTHROPIC_API_KEY'), max_retries=4, timeout=600)
lock = threading.Lock(); cost = 0.0
def one(batch):
    txt = "\n".join(f"{n + 1}. {name[p[0]]}  |  {name[p[1]]}" for n, p in enumerate(batch))
    for attempt in range(6):
        try:
            r = client.messages.create(model=a.model, max_tokens=16000, system=SYSTEM, messages=[{'role': 'user', 'content': txt}],
                                       extra_body={'output_config': {'effort': 'low', 'format': {'type': 'json_schema', 'schema': SCHEMA}}})
            text = "".join(b.text for b in r.content if b.type == 'text')
            if not text: raise ValueError(f'empty, {r.stop_reason}')
            got = {it['n']: it for it in json.loads(text)['items']}
            rows = [{'a': p[0], 'b': p[1], 'cos': p[2], 'merge': bool(got[n + 1]['merge']), 'keep': p[0] if got[n + 1]['keep'] == 1 else p[1]} for n, p in enumerate(batch) if n + 1 in got]
            return rows, r.usage.input_tokens * (4e-6 if 'opus' in a.model else 2e-6) + r.usage.output_tokens * (20e-6 if 'opus' in a.model else 10e-6)
        except Exception as e:
            err = repr(e)[:200]; time.sleep(120 if 'credit balance' in err else 5 * (attempt + 1))
    print('failed batch:', err, flush=True); return [], 0.0
t0 = time.time(); n = 0
with open(out_path, 'a') as f, ThreadPoolExecutor(a.workers) as ex:
    for fut in as_completed([ex.submit(one, b) for b in calls]):
        rows, c = fut.result()
        with lock:
            for r in rows: f.write(json.dumps(r, ensure_ascii=False) + '\n')
            f.flush(); cost += c; n += 1
            if n % 50 == 0: print(f'{n}/{len(calls)} calls, ${cost:.2f}, {time.time() - t0:.0f}s', flush=True)
print(f'done {n} calls, ${cost:.2f} (${cost / max(1, n):.4f}/call)', flush=True)
