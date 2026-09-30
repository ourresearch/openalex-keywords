"""Prompt pilot arms on data/works.jsonl, Opus 5.5 effort medium (as the God labels), synchronous, resumable -> data/arm_<arm>.jsonl.
v2 = the God-label prompt (control; god/prompt.py). A = minimal. B = A + one aboutness sentence + English. C = B on title + abstract + venue only. D = C + abstain line. E = A + English + abstain on lean input (no aboutness prohibitions; Jason 2026-09-25).
uv run --with anthropic python run_arms.py [--arms v2,A,B,C,D] [--limit N]"""
import argparse, json, os, sys, time, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import anthropic
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
import prompt as v2
A = "Assign keywords that describe what this work is about."
B = A + " Describe its subject matter: not the kind of document it is, and not a country or place unless the work is about that place. Write the keywords in English."
D = B + " If the record says nothing about its subject, return an empty list."
E = A + " Write the keywords in English. If the record says nothing about its subject, return an empty list."
F = ("Assign keywords that describe what this work is about. Describe its subject matter: not the kind of document or record it is, not its general field, not standard methods it merely uses, "
     "and not a country or place unless the work is about that place. Write the keywords in English. If the record says nothing about its subject, return an empty list.")
KW = {"type": "object", "properties": {"keywords": {"type": "array", "items": {"type": "string"}}}, "required": ["keywords"], "additionalProperties": False}
def lean_text(w): return "\n".join([f"Title: {w.get('title') or '(none)'}", f"Abstract: {(w.get('abstract') or '(none)')[:6000]}", f"Venue: {w.get('journal') or w.get('venue') or '(unknown)'}"])
ARMS = {'v2': (v2.SYSTEM, v2.SCHEMA, v2.user_text), 'A': (A, KW, v2.user_text), 'B': (B, KW, v2.user_text), 'C': (B, KW, lean_text), 'D': (D, KW, lean_text), 'E': (E, KW, lean_text), 'F': (F, KW, lean_text)}
ap = argparse.ArgumentParser(); ap.add_argument('--arms', default='v2,A,B,C,D'); ap.add_argument('--limit', type=int); ap.add_argument('--workers', type=int, default=16); ap.add_argument('--works', default='data/works.jsonl'); ap.add_argument('--prefix', default='arm'); a = ap.parse_args()
works = [json.loads(l) for l in open(f'{HERE}/{a.works}')][:a.limit]
client = anthropic.Anthropic(api_key=os.environ.get('CLAUDE_API_KEY'), max_retries=4, timeout=600)
def one(arm, w):
    system, schema, ut = ARMS[arm]
    kw = dict(model='claude-opus-5-5', max_tokens=4000, system=system, messages=[{'role': 'user', 'content': ut(w)}],
              output_config={'effort': 'medium', 'format': {'type': 'json_schema', 'schema': schema}})
    for attempt in range(6):
        try:
            r = client.beta.messages.create(**kw, betas=['server-side-fallback-2026-07-01'], fallbacks='default')
            rec = {'id': w['id'], 'arm': arm, 'model': r.model, 'stop_reason': r.stop_reason, 'usage': [r.usage.input_tokens, r.usage.output_tokens]}
            if r.stop_reason == 'refusal': return {**rec, 'error': 'refusal'}
            text = "".join(b.text for b in r.content if b.type == 'text')
            if not text: raise ValueError(f'empty text, {r.stop_reason}')
            return {**rec, 'keywords': json.loads(text)['keywords']}
        except Exception as e:
            err = repr(e)[:200]; time.sleep(120 if 'credit balance' in err else 5 * (attempt + 1))
    return {'id': w['id'], 'arm': arm, 'error': err}
lock = threading.Lock(); cost = 0.0
for arm in a.arms.split(','):
    path = f'{HERE}/data/{a.prefix}_{arm}.jsonl'
    done = {json.loads(l)['id'] for l in open(path) if 'keywords' in json.loads(l)} if os.path.exists(path) else set()
    todo = [w for w in works if w['id'] not in done]; t0 = time.time(); n_err = 0
    with open(path, 'a') as f, ThreadPoolExecutor(a.workers) as ex:
        for fut in as_completed([ex.submit(one, arm, w) for w in todo]):
            r = fut.result()
            with lock:
                f.write(json.dumps(r, ensure_ascii=False) + '\n'); f.flush(); n_err += 'error' in r
                if 'usage' in r: cost += r['usage'][0] * 4e-6 + r['usage'][1] * 20e-6
    print(f'{arm}: {len(todo)} run, {n_err} errors, {time.time() - t0:.0f}s; running cost ${cost:.2f}', flush=True)
