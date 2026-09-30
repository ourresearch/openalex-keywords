"""Score the LR-schedule-fixed student (qwen4b_v3fix) against the first v3 student on the same 1,000 judged works (#1322, 2026-09-27): verdicts from
judge_union.py are reused by kid0; only keywords the fixed model adds go to the same judge (pooled per work). Prints the same table for both.
uv run --with anthropic python judge_fix.py"""
import json, os, sys, random, time
from concurrent.futures import ThreadPoolExecutor
import anthropic
HERE = os.path.dirname(os.path.abspath(__file__)); G = HERE; sys.path.insert(0, os.path.join(HERE, '..', 'teacher'))  # released layout: data/, results/ under code/student/

sys.path.insert(0, G); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'common')); from kid import kid0; from prompt import user_text
src = open(f'{HERE}/judge_union.py').read(); SYSTEM = src.split('SYSTEM = """')[1].split('"""')[0]
SCHEMA = {"type": "object", "properties": {"sufficient": {"type": "boolean"}, "accepted": {"type": "array", "items": {"type": "integer"}}, "wrong": {"type": "array", "items": {"type": "integer"}}}, "required": ["sufficient", "accepted", "wrong"], "additionalProperties": False}
rd = lambda p: [json.loads(l) for l in open(p)]
prev = {r['id']: r for r in rd(f'{HERE}/data/judged_v3.jsonl') if 'verdict' in r}
W = {w['id']: w for w in rd(f'{HERE}/data/sample.jsonl') if w['id'] in prev}
new = {r['id']: r['keywords'] for r in rd(f'{G}/results/arm_qwen4b_v3.jsonl')}; fix = {r['id']: r['keywords'] for r in rd(f'{G}/results/arm_qwen4b_v3fix.jsonl')}
opus = {r['id']: r['answer']['keywords'] for r in rd(f'{G}/results/god_v3.jsonl') if 'answer' in r}
client = anthropic.Anthropic(api_key=os.environ.get('CLAUDE_API_KEY') or os.environ.get('ANTHROPIC_API_KEY'), max_retries=4, timeout=600)
def one(i):
    v = dict(prev[i]['verdict']); items = sorted({kid0(k): k.strip() for k in fix.get(i, []) if kid0(k) and kid0(k) not in v}.items())
    if not items: return i, v, 0.0
    random.Random(i).shuffle(items)
    txt = user_text(W[i]) + "\n\nPROPOSED KEYWORDS:\n" + "\n".join(f"{n + 1}. {s}" for n, (_, s) in enumerate(items))
    for attempt in range(6):
        try:
            r = client.messages.create(model='claude-opus-5-5', max_tokens=6000, system=SYSTEM, messages=[{'role': 'user', 'content': txt}], extra_body={'output_config': {'effort': 'medium', 'format': {'type': 'json_schema', 'schema': SCHEMA}}})
            acc = set(json.loads("".join(b.text for b in r.content if b.type == 'text'))['accepted'])
            for n, (k0, _) in enumerate(items): v[k0] = (n + 1) in acc
            return i, v, r.usage.input_tokens * 4e-6 + r.usage.output_tokens * 20e-6
        except Exception as e: time.sleep(5 * (attempt + 1))
    return i, v, 0.0
V = {}; cost = 0.0
with ThreadPoolExecutor(16) as ex:
    for i, v, c in ex.map(one, list(prev)): V[i] = v; cost += c
json.dump(V, open(f'{HERE}/data/judged_fix.json', 'w'))
def grp(i):
    w = W[i]
    if w.get('xpac'): return 'xpac'
    a = len((w.get('abstract') or '').strip()); return 'abstract' if a >= 300 else ('short' if a >= 50 else 'title_only')
print(f'judge ${cost:.2f}\n| list | group | works | kw/work | empty | judged P | good/work | F1 vs Opus |\n|---|---|---|---|---|---|---|---|')
for nm, L in (('v3 (first)', new), ('v3 LR-fixed', fix)):
    for g in ['abstract', 'short', 'title_only', 'xpac', 'all_nonxpac']:
        ids = [i for i in V if i in L and (grp(i) == g or (g == 'all_nonxpac' and grp(i) != 'xpac'))]
        ks = [(i, kid0(k)) for i in ids for k in L[i] if kid0(k)]; jd = [V[i][k] for i, k in ks if k in V[i]]
        f1 = []
        for i in ids:
            go = {kid0(k) for k in opus.get(i, []) if kid0(k)}; p = {kid0(k) for k in L[i] if kid0(k)}
            f1.append(1.0 if not go and not p else 2 * len(go & p) / (len(go) + len(p)))
        print(f"| {nm} | {g} | {len(ids)} | {len(ks)/len(ids):.2f} | {sum(not L[i] for i in ids)/len(ids):.3f} | {sum(jd)/len(jd):.3f} | {sum(jd)/len(ids):.2f} | {sum(f1)/len(f1):.3f} |")
