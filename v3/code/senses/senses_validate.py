"""Homonym senses, validation (#1465): for each minority sense, sample up to 20 works the field rule moved there; Opus 5.5 (medium) reads title +
abstract and picks which meaning of the keyword the work uses, from all the keyword's senses (or 'none'), blind to the rule's choice.
Precision = share where the judge picks the assigned sense. -> data/senses_valid.jsonl
uv run --with anthropic --with httpx python senses_validate.py"""
import json, os, threading, time, collections
from concurrent.futures import ThreadPoolExecutor
import anthropic
import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'common'))
from dbx import sql
T = 'openalex.transient.'
summ = {s['heading']: s for s in json.load(open('data/senses_summary.json'))}
J = {r['kid'].rsplit('/', 1)[-1]: r for r in map(json.loads, open('data/homonym_judged.jsonl'))}
rows = sql(f"""WITH m AS (SELECT c.work_id, x.id AS sid FROM {T}kw1465_sense_changes c LATERAL VIEW explode(c.keywords) t AS x
  WHERE x.id IN (SELECT DISTINCT concat('https://openalex.org/keywords/', sense_id) FROM {T}kw1465_senses)
    AND NOT array_contains(transform(c.old_keywords, y -> y.id), x.id)),
s AS (SELECT m.*, row_number() OVER (PARTITION BY sid ORDER BY xxhash64(work_id, 99)) AS rn FROM m)
SELECT s.sid, s.work_id, w.title, substr(w.abstract, 1, 1500) AS abstract, w.primary_topic.field.display_name AS field
FROM s JOIN openalex.works.openalex_works w ON w.id = s.work_id WHERE s.rn <= 20""", timeout_s=3600)
sense_of = {}
for _, s in summ.items():
    for x in s['split']: sense_of['https://openalex.org/keywords/' + x['id']] = s['heading']
SCHEMA = {"type": "object", "properties": {"choice": {"type": "integer"}, "why": {"type": "string"}}, "required": ["choice", "why"], "additionalProperties": False}
client = anthropic.Anthropic(api_key=os.environ['CLAUDE_API_KEY'], max_retries=4, timeout=600)
out = open('data/senses_valid.jsonl', 'w'); lock = threading.Lock(); cost = [0.0]
def one(r):
    h = sense_of.get(r['sid'])
    if not h: return
    opts = [s['label'] for s in sorted(J[h]['senses'], key=lambda s: -s['share'])] + ['none of these / not about this keyword']
    msg = (f"Keyword: {h.replace('-', ' ')}\nWork title: {r['title']}\nAbstract: {r['abstract'] or '(none)'}\n\nWhich meaning of the keyword does this work use?\n"
           + "\n".join(f"{i + 1}. {o}" for i, o in enumerate(opts)) + "\nAnswer with the number.")
    for a in range(4):
        try:
            m = client.messages.create(model='claude-opus-5-5', max_tokens=4000, messages=[{'role': 'user', 'content': msg}],
                                       extra_body={'output_config': {'effort': 'medium', 'format': {'type': 'json_schema', 'schema': SCHEMA}}})
            t = "".join(b.text for b in m.content if b.type == 'text')
            if not t: return
            v = json.loads(t); c = v['choice'] - 1
            assigned = next((i for i, o in enumerate(opts) if 'https://openalex.org/keywords/' + __import__('sys').modules['__main__'].kid0(o) == r['sid']), None)
            with lock:
                cost[0] += m.usage.input_tokens * 4e-6 + m.usage.output_tokens * 20e-6
                out.write(json.dumps({'sid': r['sid'], 'heading': h, 'work_id': r['work_id'], 'field': r['field'], 'title': r['title'], 'opts': opts,
                                      'choice': c, 'assigned': assigned, 'ok': c == assigned, 'why': v['why']}, ensure_ascii=False) + '\n'); out.flush()
            return
        except Exception: time.sleep(5 * (a + 1))
import sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'common')); from kid import kid0
print(len(rows), 'works to check', flush=True)
with ThreadPoolExecutor(16) as ex: list(ex.map(one, rows))
R = [json.loads(l) for l in open('data/senses_valid.jsonl')]
by = collections.defaultdict(list)
for r in R: by[r['sid']].append(r['ok'])
print(f"done ${cost[0]:.2f}; {len(R)} works; precision {sum(r['ok'] for r in R) / max(1, len(R)):.3f}; senses with precision < 0.7: "
      + str(sorted((k.split('/')[-1], round(sum(v) / len(v), 2)) for k, v in by.items() if len(v) >= 5 and sum(v) / len(v) < 0.7)))
