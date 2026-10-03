"""#1476 gold labels: Opus 5.5 (medium) picks which sense of the keyword a work uses, from the keyword's senses (#1465 homonym_judged) + 'none'.
Same prompt as #1465 senses_validate.py, so the two label sets combine. Refusals / errors kept as their own class. -> data/gold_labels.jsonl
uv run --with anthropic python gold_label.py [--in data/gold_works.jsonl]"""
import argparse, json, os, threading, time
from concurrent.futures import ThreadPoolExecutor
import anthropic
ap = argparse.ArgumentParser(); ap.add_argument('--in', dest='inp', default='data/gold_works.jsonl'); ap.add_argument('--out', default='data/gold_labels.jsonl'); a = ap.parse_args()
J = {r['kid'].rsplit('/', 1)[-1]: r for r in map(json.loads, open('data/homonym_judged.jsonl'))}
SCHEMA = {"type": "object", "properties": {"choice": {"type": "integer"}, "why": {"type": "string"}}, "required": ["choice", "why"], "additionalProperties": False}
client = anthropic.Anthropic(api_key=os.environ['CLAUDE_API_KEY'], max_retries=4, timeout=600)
done = {(r['kid'], r['work_id']) for r in map(json.loads, open(a.out))} if os.path.exists(a.out) else set()
out = open(a.out, 'a'); lock = threading.Lock(); cost = [0.0]
def one(r):
    h = r['kid'].rsplit('/', 1)[-1]
    opts = [s['label'] for s in sorted(J[h]['senses'], key=lambda s: -s['share'])] + ['none of these / not about this keyword']
    msg = (f"Keyword: {h.replace('-', ' ')}\nWork title: {r['title']}\nAbstract: {r['abstract'] or '(none)'}\n\nWhich meaning of the keyword does this work use?\n"
           + "\n".join(f"{i + 1}. {o}" for i, o in enumerate(opts)) + "\nAnswer with the number.")
    row = {'kid': r['kid'], 'work_id': r['work_id'], 'field': r['field'], 'opts': opts, 'model': 'claude-opus-5-5', 'effort': 'medium', 'prompt': 'senses_v1'}
    for att in range(4):
        try:
            m = client.messages.create(model='claude-opus-5-5', max_tokens=4000, messages=[{'role': 'user', 'content': msg}],
                                       extra_body={'output_config': {'effort': 'medium', 'format': {'type': 'json_schema', 'schema': SCHEMA}}})
            t = "".join(b.text for b in m.content if b.type == 'text')
            if not t: row.update(choice=None, error=f'no text ({m.stop_reason})')
            else:
                v = json.loads(t); c = v['choice'] - 1
                row.update(choice=c if 0 <= c < len(opts) else None, why=v['why'])
            with lock:
                cost[0] += m.usage.input_tokens * 4e-6 + m.usage.output_tokens * 20e-6
                out.write(json.dumps(row, ensure_ascii=False) + '\n'); out.flush()
            return
        except Exception as e: err = repr(e)[:200]; time.sleep(5 * (att + 1))
    with lock: out.write(json.dumps({**row, 'choice': None, 'error': err}) + '\n'); out.flush()
todo = [r for r in map(json.loads, open(a.inp)) if (r['kid'], r['work_id']) not in done]
print(len(todo), 'to label', flush=True)
with ThreadPoolExecutor(16) as ex:
    for i, _ in enumerate(ex.map(one, todo)):
        if i % 500 == 0: print(i, f'${cost[0]:.2f}', flush=True)
print(f'done ${cost[0]:.2f}')
