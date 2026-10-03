"""Bad-merge audit of synmap9 as shipped in v3.0 (2026-09-30). Unit = one merge (source keyword -> heading) with the works it moved.
Frame: every merge moving >= 3,000 works (exhaustive) + a works-weighted random sample of 1,500 of the rest. String-level Opus 5.5 pass (effort low,
50 pairs/call). Checkpoint audit_v31/audit_judged.jsonl keyed src|heading. `--tail` adds every other non-trivial merge (run the night of 09-30).
Prompt: prompts/17_merge_audit.md.
    CLAUDE_API_KEY=... uv run --with anthropic python audit_merges.py [--tail]"""
import json, os, random, threading, time
import gzip, os, sys
AUD = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'synonyms', 'audit_v31')
def _open(fn, mode='rt'):
    p = os.path.join(AUD, fn)
    if 'r' in mode and not os.path.exists(p) and os.path.exists(p + '.gz'): return gzip.open(p + '.gz', mode)
    return open(p, mode)
from concurrent.futures import ThreadPoolExecutor
import anthropic
M = [json.loads(l) for l in _open('merges.jsonl')]
for x in M: x['works'] = int(x['works'])
norm = lambda k: k.replace('-', '')
trivial = lambda s, h: norm(s) == norm(h) or norm(s).rstrip('s') == norm(h).rstrip('s')
M = [x for x in M if not trivial(x['src'], x['heading'])]
big = [x for x in M if x['works'] >= 3000]; rest = [x for x in M if x['works'] < 3000]
rng = random.Random(1465); samp = []; seen = set()
wts = [x['works'] for x in rest]
while len(samp) < 1500:
    x = rng.choices(rest, weights=wts)[0]
    if x['src'] not in seen: seen.add(x['src']); samp.append(x)
for x in big: x['frame'] = 'big'
for x in samp: x['frame'] = 'sample'
todo = big + samp
import sys
if '--tail' in sys.argv:
    for x in rest: x['frame'] = x.get('frame', 'tail')
    todo = big + samp + [x for x in rest if x['frame'] == 'tail']
json.dump({'big': len(big), 'big_works': sum(x['works'] for x in big), 'rest': len(rest), 'rest_works': sum(wts), 'sample': [x['src'] for x in samp]}, _open('audit_frame.json', 'w'))
OUT = os.path.join(AUD, 'audit_judged.jsonl')
done = {json.loads(l)['key'] for l in _open('audit_judged.jsonl')} if os.path.exists(OUT) or os.path.exists(OUT + '.gz') else set()
todo = [x for x in todo if f"{x['src']}|{x['heading']}" not in done]
print(len(big), 'big +', len(samp), 'sample;', len(todo), 'to judge', flush=True)
SYSTEM = """You audit the synonym map of a scholarly subject index. Each line says: every work tagged with keyword A now appears under heading B instead (A was merged into B). Judge each merge:
- "same": A and B name the same subject (synonym, spelling/transliteration/word-order variant, acronym and its expansion when unambiguous, old and new name, Latin taxon and common name).
- "close": not strictly the same (A is narrower or broader than B, or closely related), but nearly all works tagged A would still be relevant to someone searching B.
- "wrong": many works tagged A would NOT be about B: a different thing, a different sense of the words, an ambiguous acronym read the wrong way, a period of time or event vs a disease or condition (e.g. "post-COVID era" is a time period, not long COVID), a place or group that is not B.
Judge from the strings; think about how authors actually use keyword A in the literature. Return JSON only; `why` only for "wrong" (a few words)."""
SCHEMA = {"type": "object", "properties": {"items": {"type": "array", "items": {"type": "object", "properties": {"n": {"type": "integer"}, "v": {"type": "string", "enum": ["same", "close", "wrong"]}, "why": {"type": "string"}}, "required": ["n", "v", "why"], "additionalProperties": False}}}, "required": ["items"], "additionalProperties": False}
client = anthropic.Anthropic(api_key=os.environ['CLAUDE_API_KEY'], max_retries=4, timeout=600)
lock = threading.Lock(); out = open(OUT, 'a'); cost = [0.0]
def one(b):
    txt = "\n".join(f"{n + 1}. A: {x['example_raw']}  ->  B: {x['heading_name']}" for n, x in enumerate(b))
    for attempt in range(5):
        try:
            r = client.messages.create(model='claude-opus-5-5', max_tokens=16000, system=SYSTEM, messages=[{'role': 'user', 'content': txt}],
                                       extra_body={'output_config': {'effort': 'low', 'format': {'type': 'json_schema', 'schema': SCHEMA}}})
            got = {it['n']: it for it in json.loads("".join(c.text for c in r.content if c.type == 'text'))['items']}
            c = r.usage.input_tokens * 4e-6 + r.usage.output_tokens * 20e-6
            with lock:
                cost[0] += c
                for n, x in enumerate(b):
                    if n + 1 in got: out.write(json.dumps({'key': f"{x['src']}|{x['heading']}", **x, 'v': got[n + 1]['v'], 'why': got[n + 1]['why']}, ensure_ascii=False) + '\n')
                out.flush()
            return
        except Exception as e:
            err = repr(e)[:200]; time.sleep(5 * (attempt + 1))
    print('failed batch', err, flush=True)
batches = [todo[i:i + 50] for i in range(0, len(todo), 50)]
with ThreadPoolExecutor(24) as ex:
    for i, _ in enumerate(ex.map(one, batches)):
        if i % 25 == 0: print(i, '/', len(batches), f'${cost[0]:.2f}', flush=True)
print(f'done ${cost[0]:.2f}')
