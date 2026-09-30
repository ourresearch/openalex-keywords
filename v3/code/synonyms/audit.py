"""Residual synonym audit, the stop rule (#1322, 2026-09-28): on the rebuilt vocabulary, sample N keywords weighted by works_count (what users see), find
candidate synonyms that are still SEPARATE headings through two independent channels (Opus lists other names -> vocabulary lookup; embedding neighbours
from data/pairs_audit.jsonl if present), and have the v2 judge (judge_pairs2 rubric) confirm. Reports the share of sampled keywords with a surviving
synonym, the works-weighted share, and the examples. uv run --with anthropic --with httpx python audit.py --prefix kw1322v3h_ [--n 1000]"""
import argparse, json, os, sys, random, time, collections
from concurrent.futures import ThreadPoolExecutor
import anthropic
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'common')); from kid import kid0
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'common')); import dbxr
ap = argparse.ArgumentParser(); ap.add_argument('--prefix', default='kw1322v3h_'); ap.add_argument('--n', type=int, default=1000); ap.add_argument('--seed', type=int, default=28); a = ap.parse_args()
rows = dbxr.sql(f"SELECT kid, display_name, works_count FROM openalex.transient.{a.prefix}vocab", warehouse=os.environ.get('DATABRICKS_WAREHOUSE_ID'))
V = {r['kid']: {'kid': r['kid'], 'name': r['display_name'], 'n': int(r['works_count'])} for r in rows}
json.dump(list(V.values()), open(f'{HERE}/data/vocab_{a.prefix}.json', 'w'), ensure_ascii=False)
rng = random.Random(a.seed); ks = list(V); wts = [V[k]['n'] for k in ks]
sample = []
seen = set()
while len(sample) < a.n:
    k = rng.choices(ks, weights=wts)[0]
    if k not in seen: seen.add(k); sample.append(k)
client = anthropic.Anthropic(api_key=os.environ.get('CLAUDE_API_KEY') or os.environ.get('ANTHROPIC_API_KEY'), max_retries=4, timeout=600)
def call(system, schema, txt, effort='low'):
    for attempt in range(6):
        try:
            m = client.messages.create(model='claude-opus-5-5', max_tokens=16000, system=system, messages=[{'role': 'user', 'content': txt}], extra_body={'output_config': {'effort': effort, 'format': {'type': 'json_schema', 'schema': schema}}})
            return json.loads("".join(b.text for b in m.content if b.type == 'text'))['items'], m.usage.input_tokens * 4e-6 + m.usage.output_tokens * 20e-6
        except Exception: time.sleep(5 * (attempt + 1))
    return [], 0.0
ALT_SYS = open(f'{HERE}/altnames.py').read().split('SYSTEM = """')[1].split('"""')[0]
ALT_SCHEMA = {"type": "object", "properties": {"items": {"type": "array", "items": {"type": "object", "properties": {"n": {"type": "integer"}, "names": {"type": "array", "items": {"type": "string"}}}, "required": ["n", "names"], "additionalProperties": False}}}, "required": ["items"], "additionalProperties": False}
def lookup(s):
    k = kid0(s) if s else None
    if not k: return None
    for c in (k, k + 's', k[:-1] if k.endswith('s') else None):
        if c and c in V: return c
batches = [sample[i:i + 100] for i in range(0, len(sample), 100)]
cand = set(); cost = 0.0
def alt(b):
    items, c = call(ALT_SYS, ALT_SCHEMA, "\n".join(f"{n + 1}. {V[k]['name']}" for n, k in enumerate(b)))
    out = []
    for it in items:
        if 1 <= it['n'] <= len(b):
            for nm in it['names']:
                o = lookup(nm)
                if o and o != b[it['n'] - 1]: out.append((b[it['n'] - 1], o))
    return out, c
with ThreadPoolExecutor(10) as ex:
    for pairs, c in ex.map(alt, batches): cand |= set(pairs); cost += c
if os.path.exists(f'{HERE}/data/pairs_audit.jsonl'):
    S = set(sample)
    for l in open(f'{HERE}/data/pairs_audit.jsonl'):
        r = json.loads(l)
        if r['a'] in S and r['b'] in V and r['cos'] >= 0.85: cand.add((r['a'], r['b']))
cand = sorted(cand)
J_SYS = open(f'{HERE}/judge_pairs2.py').read().split('SYSTEM = """')[1].split('"""')[0]
J_SCHEMA = {"type": "object", "properties": {"items": {"type": "array", "items": {"type": "object", "properties": {"n": {"type": "integer"}, "merge": {"type": "boolean"}, "keep": {"type": "integer"}}, "required": ["n", "merge", "keep"], "additionalProperties": False}}}, "required": ["items"], "additionalProperties": False}
res = []
def judge(b):
    items, c = call(J_SYS, J_SCHEMA, "\n".join(f"{n + 1}. {V[x]['name']}  |  {V[y]['name']}" for n, (x, y) in enumerate(b)))
    got = {it['n']: it['merge'] for it in items}
    return [(x, y, got.get(n + 1)) for n, (x, y) in enumerate(b)], c
with ThreadPoolExecutor(10) as ex:
    for r_, c in ex.map(judge, [cand[i:i + 100] for i in range(0, len(cand), 100)]): res += r_; cost += c
split = collections.defaultdict(list)
for x, y, mg in res:
    if mg: split[x].append(y)
share = len(split) / len(sample); wshare = sum(V[k]['n'] for k in split) / sum(V[k]['n'] for k in sample)
json.dump({'prefix': a.prefix, 'n': len(sample), 'candidates': len(cand), 'with_split': len(split), 'share': share, 'works_weighted_share': wshare,
           'examples': [[V[k]['name'], V[k]['n'], [V[o]['name'] for o in split[k]]] for k in sorted(split, key=lambda k: -V[k]['n'])]},
          open(f'{HERE}/data/audit_{a.prefix}.json', 'w'), ensure_ascii=False, indent=1)
print(f'{len(sample)} sampled keywords (works-weighted), {len(cand)} candidates, {len(split)} still have a separate synonym: {share:.1%} of keywords, '
      f'{wshare:.1%} works-weighted; ${cost:.2f}')
for k in sorted(split, key=lambda k: -V[k]['n'])[:25]: print('  ', V[k]['name'], V[k]['n'], '<->', '; '.join(V[o]['name'] for o in split[k]))
