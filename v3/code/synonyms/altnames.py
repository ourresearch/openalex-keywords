"""Synonym candidates by generation (#1322, 2026-09-28 evening): embedding neighbours miss lexically different names for one concept (older adults / elderly,
mpox / monkeypox, vitamin C / ascorbic acid). Opus 5.5 lists other names each of the top 50K keywords goes by in research; names found in the vocabulary
become candidate pairs for judge_pairs2.py. -> data/altnames.jsonl, data/cand_alt.json. uv run --with anthropic python altnames.py"""
import json, os, sys, time
from concurrent.futures import ThreadPoolExecutor
import anthropic
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'common')); from kid import kid0
import argparse; _ap = argparse.ArgumentParser(); _ap.add_argument('--vocab', default='data/vocab_g.jsonl'); _ap.add_argument('--start', type=int, default=0); _ap.add_argument('--end', type=int, default=50000); _ap.add_argument('--tag', default=''); A = _ap.parse_args()
V = {json.loads(l)['kid']: json.loads(l) for l in open(f'{HERE}/{A.vocab}')}
syn = {} if A.tag else {json.loads(l)['kid']: json.loads(l)['canon'] for l in open(f'{HERE}/data/synmap2.jsonl')}
top = sorted(V.values(), key=lambda v: -v['works_count'])[A.start:A.end]
SYSTEM = """For each numbered keyword from a scholarly subject vocabulary, list up to 6 other names researchers use for EXACTLY the same concept: synonyms, near-synonyms used interchangeably, older or newer names (monkeypox for mpox), common vs scientific names (ascorbic acid for vitamin C), abbreviations and expansions, British/American spellings. Not broader or narrower terms, not related concepts. Empty list if none. Return JSON only."""
SCHEMA = {"type": "object", "properties": {"items": {"type": "array", "items": {"type": "object", "properties": {"n": {"type": "integer"}, "names": {"type": "array", "items": {"type": "string"}}}, "required": ["n", "names"], "additionalProperties": False}}}, "required": ["items"], "additionalProperties": False}
client = anthropic.Anthropic(api_key=os.environ.get('CLAUDE_API_KEY') or os.environ.get('ANTHROPIC_API_KEY'), max_retries=4, timeout=600)
def one(batch):
    txt = "\n".join(f"{n + 1}. {v['display_name']}" for n, v in enumerate(batch))
    for attempt in range(6):
        try:
            m = client.messages.create(model='claude-opus-5-5', max_tokens=16000, system=SYSTEM, messages=[{'role': 'user', 'content': txt}], extra_body={'output_config': {'effort': 'low', 'format': {'type': 'json_schema', 'schema': SCHEMA}}})
            got = {it['n']: it['names'] for it in json.loads("".join(b.text for b in m.content if b.type == 'text'))['items']}
            return [{'kid': v['kid'], 'names': got.get(n + 1, [])} for n, v in enumerate(batch)], m.usage.input_tokens * 4e-6 + m.usage.output_tokens * 20e-6
        except Exception: time.sleep(5 * (attempt + 1))
    return [{'kid': v['kid'], 'names': []} for v in batch], 0.0
out, cost = [], 0.0
with ThreadPoolExecutor(24) as ex:
    for r_, c in ex.map(one, [top[i:i + 100] for i in range(0, len(top), 100)]): out += r_; cost += c
def lookup(s):
    k = kid0(s) if s else None
    if not k: return None
    for c in (k, k + 's', k[:-1] if k.endswith('s') else None):
        if c and (c in V or c in syn): return syn.get(c, c)
cand = set()
with open(f'{HERE}/data/altnames{A.tag}.jsonl', 'w') as f:
    for r in out:
        f.write(json.dumps(r, ensure_ascii=False) + '\n')
        for nm in r['names']:
            b = lookup(nm)
            if b and b != r['kid'] and b in V: cand.add(tuple(sorted((r['kid'], b))))
json.dump([[a, b, 1.0] for a, b in sorted(cand)], open(f'{HERE}/data/cand_alt{A.tag}.json', 'w'))
print(f'{len(out)} keywords, {sum(len(r["names"]) for r in out)} names, {len(cand)} candidate pairs in the vocabulary, ${cost:.2f}')
