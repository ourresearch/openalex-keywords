"""Opus confirms UK/US spelling-variant merges (data/ukus_cand.json) for the final vocabulary (#1322, 2026-09-28). Merge only when the two are the same
concept spelled two ways; never respell a proper name (Labour Party, a Latin taxon such as Haemophilus, an organisation, a title). Canonical = the more-used
form. -> data/ukus_judged.jsonl, data/synmap_ukus.jsonl {kid, canon}. uv run --with anthropic --with httpx python judge_ukus.py"""
import json, os, sys, time
from concurrent.futures import ThreadPoolExecutor
import anthropic
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'common')); import dbxr
rows = dbxr.sql("SELECT kid, display_name, works_count FROM openalex.transient.kw1322v3f_vocab", warehouse=os.environ.get('DATABRICKS_WAREHOUSE_ID'))
name = {r['kid']: r['display_name'] for r in rows}; wc = {r['kid']: int(r['works_count']) for r in rows}
cand = json.load(open(f'{HERE}/data/ukus_cand.json'))
SYSTEM = """Each line gives two keyword headings from a scholarly subject vocabulary that differ by British vs American spelling. Merge (true) only if they are the same concept spelled two ways, so works under either belong under one heading. Do NOT merge (false) when either is a proper name that must keep its own spelling (a political party such as the UK Labour Party, a Latin taxon name such as Haemophilus or Haemonchus, an organisation, a place, a title of a work) or when the words mean different things. Return JSON only."""
SCHEMA = {"type": "object", "properties": {"items": {"type": "array", "items": {"type": "object", "properties": {"n": {"type": "integer"}, "merge": {"type": "boolean"}}, "required": ["n", "merge"], "additionalProperties": False}}}, "required": ["items"], "additionalProperties": False}
client = anthropic.Anthropic(api_key=os.environ.get('CLAUDE_API_KEY') or os.environ.get('ANTHROPIC_API_KEY'), max_retries=4, timeout=600)
def one(batch):
    txt = "\n".join(f"{n + 1}. {name[a]}  |  {name[b]}" for n, (a, b) in enumerate(batch))
    for attempt in range(6):
        try:
            m = client.messages.create(model='claude-opus-5-5', max_tokens=8000, system=SYSTEM, messages=[{'role': 'user', 'content': txt}],
                                       extra_body={'output_config': {'effort': 'low', 'format': {'type': 'json_schema', 'schema': SCHEMA}}})
            got = {it['n']: it['merge'] for it in json.loads("".join(b.text for b in m.content if b.type == 'text'))['items']}
            return [{'a': a, 'b': b, 'merge': got.get(n + 1)} for n, (a, b) in enumerate(batch)], m.usage.input_tokens * 4e-6 + m.usage.output_tokens * 20e-6
        except Exception as e: time.sleep(5 * (attempt + 1))
    return [{'a': a, 'b': b, 'merge': None} for a, b in batch], 0.0
batches = [cand[i:i + 100] for i in range(0, len(cand), 100)]; out = []; cost = 0.0
with ThreadPoolExecutor(16) as ex:
    for rows_, c in ex.map(one, batches): out += rows_; cost += c
with open(f'{HERE}/data/ukus_judged.jsonl', 'w') as f:
    for r in out: f.write(json.dumps(r) + '\n')
m = [r for r in out if r['merge']]
with open(f'{HERE}/data/synmap_ukus.jsonl', 'w') as f:
    for r in m:
        a, b = r['a'], r['b']; canon, other = (a, b) if wc[a] > wc[b] else (b, a)
        f.write(json.dumps({'kid': other, 'canon': canon}) + '\n')
print(f"{len(out)} judged, {len(m)} merge, {sum(r['merge'] is False for r in out)} keep apart, {sum(r['merge'] is None for r in out)} failed, ${cost:.2f}")
for r in [r for r in out if r['merge'] is False][:15]: print('  apart:', name[r['a']], '|', name[r['b']])
