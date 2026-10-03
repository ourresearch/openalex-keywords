"""Sense-label sizing, step 2 (#1465): for the top candidates by works, pull 3 titles from each of the keyword's top 3 fields and ask Opus 5.5
(effort medium) whether the keyword is one subject or several distinct senses; if several, name each sense as a parenthetical label and say which
fields carry it. -> data/homonym_judged.jsonl. uv run --with anthropic --with httpx python homonym_judge.py [--n 3000]"""
import argparse, json, os, threading, time
from concurrent.futures import ThreadPoolExecutor
import anthropic
import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'common'))
from dbx import sql
ap = argparse.ArgumentParser(); ap.add_argument('--n', type=int, default=3000); ap.add_argument('--cands', default='data/homonym_domain_cands.jsonl'); ap.add_argument('--out', default='data/homonym_judged.jsonl'); ap.add_argument('--tag', default=''); a = ap.parse_args()
C = sorted((json.loads(l) for l in open(a.cands)), key=lambda r: -int(r['works']))[:a.n]
T = 'openalex.transient'
sql(f"CREATE OR REPLACE TABLE {T}.kw1465_hom_kids{a.tag} (kid STRING)", verbose=False)
for i in range(0, len(C), 2000):
    sql(f"INSERT INTO {T}.kw1465_hom_kids{a.tag} VALUES " + ",".join("('" + r['kid'].replace("'", "\\'") + "')" for r in C[i:i+2000]), verbose=False)
rows = sql(f"""WITH f AS (SELECT kid, field, n, row_number() OVER (PARTITION BY kid ORDER BY n DESC) AS r FROM {T}.kw1465_kw_field WHERE kid IN (SELECT kid FROM {T}.kw1465_hom_kids{a.tag})),
top AS (SELECT kid, field FROM f WHERE r <= 3),
e AS (SELECT x.id AS kid, k.work_id FROM openalex.works.work_keywords_v2 k LATERAL VIEW explode(k.keywords) t AS x WHERE x.id IN (SELECT kid FROM {T}.kw1465_hom_kids{a.tag})),
s AS (SELECT e.kid, w.primary_topic.field.display_name AS field, w.title, row_number() OVER (PARTITION BY e.kid, w.primary_topic.field.display_name ORDER BY xxhash64(e.work_id, 1465)) AS rn
      FROM e JOIN openalex.works.openalex_works w ON w.id = e.work_id JOIN top ON top.kid = e.kid AND top.field = w.primary_topic.field.display_name WHERE w.title IS NOT NULL)
SELECT kid, field, collect_list(substr(title, 1, 160)) AS titles FROM s WHERE rn <= 3 GROUP BY 1, 2""", timeout_s=7200)
fs = {}
for r in sql(f"SELECT kid, field, n FROM {T}.kw1465_kw_field WHERE kid IN (SELECT kid FROM {T}.kw1465_hom_kids{a.tag})", verbose=False):
    fs.setdefault(r['kid'], []).append((r['field'], int(r['n'])))
ex = {}
for r in rows: ex.setdefault(r['kid'], {})[r['field']] = json.loads(r['titles']) if isinstance(r['titles'], str) else r['titles']
SYSTEM = """You curate a scholarly subject index. A keyword is attached to works in several fields. Decide whether the keyword names ONE subject
(possibly studied in several fields: "oxidative stress" in medicine and biochemistry is one subject) or DISTINCT SENSES that a searcher would never
want mixed ("transformer": the neural-network architecture vs the electrical device; "mercury": the element vs the planet; "depression": the mood
disorder vs an economic depression vs a topographic depression). If distinct senses, give each a short parenthetical label in Wikipedia style
("transformer (machine learning)"), the fields that carry it, and roughly what share of the keyword's works it covers. Judge from the field mix and
the example titles. Return JSON only."""
SCHEMA = {"type": "object", "properties": {"verdict": {"type": "string", "enum": ["one_subject", "distinct_senses"]},
  "senses": {"type": "array", "items": {"type": "object", "properties": {"label": {"type": "string"}, "fields": {"type": "array", "items": {"type": "string"}}, "share": {"type": "number"}},
  "required": ["label", "fields", "share"], "additionalProperties": False}}, "why": {"type": "string"}}, "required": ["verdict", "senses", "why"], "additionalProperties": False}
client = anthropic.Anthropic(api_key=os.environ['CLAUDE_API_KEY'], max_retries=4, timeout=600)
OUT = a.out
done = {json.loads(l)['kid'] for l in open(OUT)} if os.path.exists(OUT) else set()
out = open(OUT, 'a'); lock = threading.Lock(); cost = [0.0]
def one(c):
    kid = c['kid']; tot = sum(n for _, n in fs.get(kid, []))
    mix = "\n".join(f"- {f}: {n / tot:.0%}" for f, n in sorted(fs.get(kid, []), key=lambda x: -x[1])[:8])
    titles = "\n".join(f"[{f}] " + " | ".join(ts) for f, ts in ex.get(kid, {}).items())
    msg = f"Keyword: {kid.split('/')[-1].replace('-', ' ')} ({kid})\nWorks: {c['works']}\nField mix:\n{mix}\nExample titles by field:\n{titles}"
    for attempt in range(4):
        try:
            r = client.messages.create(model='claude-opus-5-5', max_tokens=8000, system=SYSTEM, messages=[{'role': 'user', 'content': msg}],
                                       extra_body={'output_config': {'effort': 'medium', 'format': {'type': 'json_schema', 'schema': SCHEMA}}})
            t = "".join(b.text for b in r.content if b.type == 'text')
            if not t: return
            v = json.loads(t)
            with lock:
                cost[0] += r.usage.input_tokens * 4e-6 + r.usage.output_tokens * 20e-6
                out.write(json.dumps({'kid': kid, 'works': int(c['works']), **v}, ensure_ascii=False) + '\n'); out.flush()
            return
        except Exception: time.sleep(5 * (attempt + 1))
todo = [c for c in C if c['kid'] not in done]
print(len(todo), 'to judge', flush=True)
with ThreadPoolExecutor(16) as ex_:
    for i, _ in enumerate(ex_.map(one, todo)):
        if i % 250 == 0: print(i, f'${cost[0]:.2f}', flush=True)
print(f'done ${cost[0]:.2f}')
