"""Opus verdicts on the retag's own keywords (B200 FP8 output, with confidences) for the 1,000 works judged in relabel/judge_union.py, so the review page can show judged
precision under any threshold. Verdicts already given there are reused (same kid0); only unseen keywords go to the judge, pooled per work, same strict rubric.
-> data/judged_retag.jsonl. uv run --with anthropic --with httpx python judge_retag.py"""
import json, os, random, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
import anthropic
HERE = os.path.dirname(os.path.abspath(__file__)); G = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'common')); import dbx
sys.path.insert(0, os.path.join(G, 'teacher')); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'common')); from kid import kid0; from prompt import user_text
import argparse; _ap = argparse.ArgumentParser(); _ap.add_argument('--table', default='keywords_student_v3'); _ap.add_argument('--out', default='judged_retag.jsonl'); A = _ap.parse_args()
src = open(f'{G}/student/judge_union.py').read(); SYSTEM = src.split('SYSTEM = """')[1].split('"""')[0]
SCHEMA = {"type": "object", "properties": {"sufficient": {"type": "boolean"}, "accepted": {"type": "array", "items": {"type": "integer"}}, "wrong": {"type": "array", "items": {"type": "integer"}}},
          "required": ["sufficient", "accepted", "wrong"], "additionalProperties": False}
prev = {r['id']: r for r in map(json.loads, open(f'{G}/student/data/judged_v3.jsonl')) if 'verdict' in r}
W = {w['id']: w for w in map(json.loads, open(f'{G}/student/data/sample.jsonl')) if w['id'] in prev}
ids = sorted(prev)
q = f"SELECT id, keywords FROM openalex.transient.{A.table} WHERE id IN (" + ",".join(f"'{i}'" for i in ids) + ")"
sys.path.insert(0, HERE); import dbxr
retag = {r['id']: (json.loads(r['keywords']) if isinstance(r['keywords'], str) else r['keywords']) for r in dbxr.sql(q)}
print(f'{len(retag)} of {len(ids)} judged works have retag output', flush=True)
client = anthropic.Anthropic(api_key=os.environ.get('CLAUDE_API_KEY') or os.environ.get('ANTHROPIC_API_KEY'), max_retries=4, timeout=600)
def one(i):
    v = dict(prev[i]['verdict']); new = {}
    for k in retag[i]:
        k0 = kid0(k)
        if k0 and k0 not in v: new.setdefault(k0, k.strip())
    if not new: return {'id': i, 'verdict': v, 'judged_new': 0}
    items = sorted(new.items()); random.Random(i).shuffle(items)
    txt = user_text(W[i]) + "\n\nPROPOSED KEYWORDS:\n" + "\n".join(f"{n + 1}. {s}" for n, (_, s) in enumerate(items))
    for attempt in range(6):
        try:
            r = client.messages.create(model='claude-opus-5-5', max_tokens=6000, system=SYSTEM, messages=[{'role': 'user', 'content': txt}],
                                       extra_body={'output_config': {'effort': 'medium', 'format': {'type': 'json_schema', 'schema': SCHEMA}}})
            text = "".join(b.text for b in r.content if b.type == 'text')
            if not text: raise ValueError(f'empty, {r.stop_reason}')
            acc = set(json.loads(text)['accepted'])
            for n, (k0, _) in enumerate(items): v[k0] = (n + 1) in acc
            return {'id': i, 'verdict': v, 'judged_new': len(items), 'usage': [r.usage.input_tokens, r.usage.output_tokens]}
        except Exception as e:
            err = repr(e)[:200]; time.sleep(120 if 'credit balance' in err else 5 * (attempt + 1))
    return {'id': i, 'verdict': v, 'error': err}
out = []; cost = 0.0
with ThreadPoolExecutor(16) as ex:
    for f in as_completed([ex.submit(one, i) for i in ids if i in retag]):
        r = f.result(); out.append(r); cost += (r['usage'][0] * 4e-6 + r['usage'][1] * 20e-6) if 'usage' in r else 0
with open(f'{HERE}/data/{A.out}', 'w') as f:
    for r in out: f.write(json.dumps(r, ensure_ascii=False) + '\n')
print(f'{len(out)} works, {sum(r.get("judged_new", 0) for r in out)} new keywords judged, errors {sum("error" in r for r in out)}, ${cost:.2f}')
