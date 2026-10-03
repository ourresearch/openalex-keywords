"""Members of the 122 headings whose display name is wrong: re-judge each merge against the heading ID (words), not its display name.
Same prompt as audit_merges.py (prompts/17_merge_audit.md), effort medium, B shown as the id words. -> audit_v31/rejudge_judged.jsonl
    CLAUDE_API_KEY=... uv run --with anthropic python rejudge.py"""
import json, os, threading, time
import gzip, os, sys
AUD = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'synonyms', 'audit_v31')
def _open(fn, mode='rt'):
    p = os.path.join(AUD, fn)
    if 'r' in mode and not os.path.exists(p) and os.path.exists(p + '.gz'): return gzip.open(p + '.gz', mode)
    return open(p, mode)
from concurrent.futures import ThreadPoolExecutor
import anthropic
H = {r['heading'] for r in json.load(_open('display_wrong.json'))}
M = [json.loads(l) for l in _open('merges.jsonl')]
todo = [dict(x, works=int(x['works'])) for x in M if x['heading'] in H]
print(len(todo), 'members', flush=True)
SYSTEM = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'audit_merges.py')).read().split('SYSTEM = """')[1].split('"""')[0]
SCHEMA = {"type": "object", "properties": {"items": {"type": "array", "items": {"type": "object", "properties": {"n": {"type": "integer"}, "v": {"type": "string", "enum": ["same", "close", "wrong"]}, "why": {"type": "string"}}, "required": ["n", "v", "why"], "additionalProperties": False}}}, "required": ["items"], "additionalProperties": False}
client = anthropic.Anthropic(api_key=os.environ['CLAUDE_API_KEY'], max_retries=4, timeout=600)
out = _open('rejudge_judged.jsonl', 'w'); lock = threading.Lock()
def one(b):
    txt = "\n".join(f"{n + 1}. A: {x['example_raw']}  ->  B: {x['heading'].replace('-', ' ')}" for n, x in enumerate(b))
    for attempt in range(5):
        try:
            r = client.messages.create(model='claude-opus-5-5', max_tokens=16000, system=SYSTEM, messages=[{'role': 'user', 'content': txt}],
                                       extra_body={'output_config': {'effort': 'medium', 'format': {'type': 'json_schema', 'schema': SCHEMA}}})
            got = {it['n']: it for it in json.loads("".join(c.text for c in r.content if c.type == 'text'))['items']}
            with lock:
                for n, x in enumerate(b):
                    if n + 1 in got: out.write(json.dumps({**x, 'v': got[n + 1]['v'], 'why': got[n + 1]['why'], 'frame': 'rejudge'}, ensure_ascii=False) + '\n')
                out.flush()
            return
        except Exception: time.sleep(5 * (attempt + 1))
with ThreadPoolExecutor(12) as ex: list(ex.map(one, [todo[i:i + 40] for i in range(0, len(todo), 40)]))
print('done')
