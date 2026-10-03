"""Headings whose display name comes from a merged member (39,221 judged): is the display name a correct name for the heading id? A wrong one hides a bad merge
from the string audit (cardiac-magnetic-resonance-imaging displays as 'civil-military relations').
Input audit_v31/display_from_member.json: [works, heading id, display name] for every heading whose display name came from a member.
-> audit_v31/display_judged.jsonl; the 'wrong' ones -> display_wrong.json. Prompt: prompts/18_display_name_check.md.
    CLAUDE_API_KEY=... uv run --with anthropic python judge_display.py"""
import json, os, threading, time
import gzip, os, sys
AUD = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'synonyms', 'audit_v31')
def _open(fn, mode='rt'):
    p = os.path.join(AUD, fn)
    if 'r' in mode and not os.path.exists(p) and os.path.exists(p + '.gz'): return gzip.open(p + '.gz', mode)
    return open(p, mode)
from concurrent.futures import ThreadPoolExecutor
import anthropic
B = json.load(_open('display_from_member.json'))
SYSTEM = """Each line gives a subject heading's identifier (a slug of its canonical name) and the display name the index shows for it. Answer "ok" if the display name names the same subject (synonym, variant, acronym of it, spelling), "wrong" if it names a different subject (e.g. an acronym clash: cardiac-magnetic-resonance-imaging shown as "civil-military relations"). Return JSON only."""
SCHEMA = {"type": "object", "properties": {"items": {"type": "array", "items": {"type": "object", "properties": {"n": {"type": "integer"}, "v": {"type": "string", "enum": ["ok", "wrong"]}}, "required": ["n", "v"], "additionalProperties": False}}}, "required": ["items"], "additionalProperties": False}
client = anthropic.Anthropic(api_key=os.environ['CLAUDE_API_KEY'], max_retries=4, timeout=600)
out = _open('display_judged.jsonl', 'w'); lock = threading.Lock(); cost = [0.0]
def one(b):
    txt = "\n".join(f"{n + 1}. id: {h}  |  display: {name}" for n, (w, h, name) in enumerate(b))
    for attempt in range(5):
        try:
            r = client.messages.create(model='claude-opus-5-5', max_tokens=16000, system=SYSTEM, messages=[{'role': 'user', 'content': txt}],
                                       extra_body={'output_config': {'effort': 'low', 'format': {'type': 'json_schema', 'schema': SCHEMA}}})
            got = {it['n']: it['v'] for it in json.loads("".join(c.text for c in r.content if c.type == 'text'))['items']}
            with lock:
                cost[0] += r.usage.input_tokens * 4e-6 + r.usage.output_tokens * 20e-6
                for n, (w, h, name) in enumerate(b):
                    if n + 1 in got: out.write(json.dumps({'heading': h, 'name': name, 'works': w, 'v': got[n + 1]}) + '\n')
                out.flush()
            return
        except Exception as e: time.sleep(5 * (attempt + 1))
bs = [B[i:i + 100] for i in range(0, len(B), 100)]
with ThreadPoolExecutor(12) as ex: list(ex.map(one, bs))
print(f'done ${cost[0]:.2f}')
