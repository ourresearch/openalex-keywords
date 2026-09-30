"""Second look at high-impact synonym merges (moved keyword on >= 10K works) with context: 3 example work titles + venues per keyword, Opus 5.5,
20 merges per call. A merge is kept only if confirmed. -> data/hi_verified.jsonl. uv run --with anthropic python verify_hi.py"""
import json, os, time
from concurrent.futures import ThreadPoolExecutor
import anthropic
HERE = os.path.dirname(os.path.abspath(__file__))
name = {json.loads(l)['kid']: json.loads(l)['display_name'] for l in open(f'{HERE}/data/vocab.jsonl')}
hi = json.load(open(f'{HERE}/data/hi_merges.json')); ex = json.load(open(f'{HERE}/data/hi_examples.json'))
SYSTEM = """You check proposed merges in the subject vocabulary of a scholarly index. For each numbered item you see two keyword headings, each with three example works tagged with it (title [venue]). Merge only if, as used on these works, the two headings mean the same subject, so works under either belong under one heading. Different senses (plasma fueling vs vehicle refueling), broader vs narrower, or related-but-distinct concepts: do not merge. Return JSON only."""
SCHEMA = {"type": "object", "properties": {"items": {"type": "array", "items": {"type": "object", "properties": {"n": {"type": "integer"}, "merge": {"type": "boolean"}}, "required": ["n", "merge"], "additionalProperties": False}}}, "required": ["items"], "additionalProperties": False}
client = anthropic.Anthropic(api_key=os.environ.get('CLAUDE_API_KEY') or os.environ.get('ANTHROPIC_API_KEY'), max_retries=4, timeout=600)
def block(k): return f"\"{name[k]}\"\n" + "\n".join(f"    - {t}" for t in ex.get(k, []))
def one(batch):
    txt = "\n\n".join(f"{n + 1}. A = {block(r['kid'])}\n   B = {block(r['canon'])}" for n, r in enumerate(batch))
    for attempt in range(6):
        try:
            m = client.messages.create(model='claude-opus-5-5', max_tokens=8000, system=SYSTEM, messages=[{'role': 'user', 'content': txt}],
                                       extra_body={'output_config': {'effort': 'medium', 'format': {'type': 'json_schema', 'schema': SCHEMA}}})
            got = {it['n']: it['merge'] for it in json.loads("".join(b.text for b in m.content if b.type == 'text'))['items']}
            return [{**r, 'verified': got.get(n + 1)} for n, r in enumerate(batch)], m.usage.input_tokens * 4e-6 + m.usage.output_tokens * 20e-6
        except Exception as e:
            err = repr(e)[:200]; time.sleep(5 * (attempt + 1))
    return [{**r, 'verified': None} for r in batch], 0.0
batches = [hi[i:i + 20] for i in range(0, len(hi), 20)]; cost = 0.0
with open(f'{HERE}/data/hi_verified.jsonl', 'w') as f, ThreadPoolExecutor(16) as ex_:
    for rows, c in ex_.map(one, batches):
        cost += c
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + '\n')
R = [json.loads(l) for l in open(f'{HERE}/data/hi_verified.jsonl')]
print(f"{len(R)} checked, confirmed {sum(r['verified'] is True for r in R)}, rejected {sum(r['verified'] is False for r in R)}, unknown {sum(r['verified'] is None for r in R)}, ${cost:.2f}")
for r in [r for r in R if r['verified'] is False][:25]: print('REJECT', name[r['kid']], '->', name[r['canon']])
