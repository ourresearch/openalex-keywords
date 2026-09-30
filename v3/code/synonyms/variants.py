"""Variant candidates the kNN pass can miss (#1322 round-2 review, 2026-09-28): (a) string key = kid without hyphens, each word UK->US and de-pluralised,
two vocabulary kids sharing a key; (b) the hygiene reviewer's acronym <-> expansion pairs (openalex.transient.kwreview2_hygiene_pairs). Opus confirms each pair
(same concept, not a proper name / taxon / broader-narrower). -> data/variants_judged.jsonl, data/synmap_variants.jsonl {kid, canon} (canon = more works).
uv run --with anthropic --with httpx python variants.py"""
import json, os, sys, time, collections
from concurrent.futures import ThreadPoolExecutor
import anthropic
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'common')); import dbxr
sys.path.insert(0, HERE); from uk_us import us_word  # noqa (module runs its own candidate build on import: guarded below)
def singular(w):
    if len(w) <= 3 or w.endswith('ss') or w.endswith('us') or w.endswith('is'): return w
    if w.endswith('ies'): return w[:-3] + 'y'
    if w.endswith('es') and w[-3:-2] in ('s', 'x', 'z', 'h'): return w[:-2]
    if w.endswith('s'): return w[:-1]
    return w
def key(k): return ''.join(singular(us_word(p)) for p in k.split('-'))
rows = dbxr.sql("SELECT kid, display_name, works_count FROM openalex.transient.kw1322v3f_vocab", warehouse=os.environ.get('DATABRICKS_WAREHOUSE_ID'))
name = {r['kid']: r['display_name'] for r in rows}; wc = {r['kid']: int(r['works_count']) for r in rows}
groups = collections.defaultdict(list)
for k in wc: groups[key(k)].append(k)
cand = set()
for g in groups.values():
    if len(g) < 2: continue
    top = max(g, key=lambda k: wc[k])
    for k in g:
        if k != top: cand.add((k, top))
acr = dbxr.sql("SELECT substr(ida, 10) a, substr(idb, 10) b FROM openalex.transient.kwreview2_hygiene_pairs", warehouse=os.environ.get('DATABRICKS_WAREHOUSE_ID'))
for r in acr:
    if r['a'] in wc and r['b'] in wc: cand.add(tuple(sorted((r['a'], r['b']), key=lambda k: wc[k])))
cand = sorted(cand); print(len(cand), 'candidate pairs', flush=True)
SYSTEM = """Each line gives two keyword headings from a scholarly subject vocabulary. Merge (true) only if they name the same concept: spelling, hyphenation or spacing variants, singular vs plural, British vs American spelling, an acronym and its expansion when the acronym, as used in research, means that expansion. Do NOT merge (false) when one is broader or narrower, when they are different things that happen to look alike, when the acronym commonly means something else, or when either is a proper name that must keep its form (a party, an organisation, a Latin taxon, a place, a title). Return JSON only."""
SCHEMA = {"type": "object", "properties": {"items": {"type": "array", "items": {"type": "object", "properties": {"n": {"type": "integer"}, "merge": {"type": "boolean"}}, "required": ["n", "merge"], "additionalProperties": False}}}, "required": ["items"], "additionalProperties": False}
client = anthropic.Anthropic(api_key=os.environ.get('CLAUDE_API_KEY') or os.environ.get('ANTHROPIC_API_KEY'), max_retries=4, timeout=600)
def one(batch):
    txt = "\n".join(f"{n + 1}. {name[a]}  |  {name[b]}" for n, (a, b) in enumerate(batch))
    for attempt in range(6):
        try:
            m = client.messages.create(model='claude-opus-5-5', max_tokens=8000, system=SYSTEM, messages=[{'role': 'user', 'content': txt}], extra_body={'output_config': {'effort': 'low', 'format': {'type': 'json_schema', 'schema': SCHEMA}}})
            got = {it['n']: it['merge'] for it in json.loads("".join(b.text for b in m.content if b.type == 'text'))['items']}
            return [{'a': a, 'b': b, 'merge': got.get(n + 1)} for n, (a, b) in enumerate(batch)], m.usage.input_tokens * 4e-6 + m.usage.output_tokens * 20e-6
        except Exception: time.sleep(5 * (attempt + 1))
    return [{'a': a, 'b': b, 'merge': None} for a, b in batch], 0.0
out, cost = [], 0.0
with ThreadPoolExecutor(16) as ex:
    for r_, c in ex.map(one, [cand[i:i + 100] for i in range(0, len(cand), 100)]): out += r_; cost += c
with open(f'{HERE}/data/variants_judged.jsonl', 'w') as f:
    for r in out: f.write(json.dumps(r) + '\n')
with open(f'{HERE}/data/synmap_variants.jsonl', 'w') as f:
    for r in out:
        if r['merge']:
            a, b = r['a'], r['b']; canon, other = (a, b) if wc[a] >= wc[b] else (b, a); f.write(json.dumps({'kid': other, 'canon': canon}) + '\n')
print(f"{len(out)} judged, {sum(bool(r['merge']) for r in out)} merge, ${cost:.2f}")
