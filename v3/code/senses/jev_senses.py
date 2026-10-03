"""#1476 Jev sense classifier: one Jev choice per (work, keyword); options = the keyword's senses (#1465 labels) + none; state = keyword, title,
abstract, venue. Option text version OPT_V (only this is tuned, on train keywords). -> data/jev_<tag>.jsonl (probabilities per option)
uv run --with httpx python jev_senses.py --in data/gold_works.jsonl [--in2 ...] --tag v1"""
import argparse, asyncio, json, os, sys, hashlib
sys.path.insert(0, os.environ.get('JEVKIT_DIR', 'jevkit')); from jevx import JevClient   # OpenAlex's Jev client (not released; see README)
ap = argparse.ArgumentParser(); ap.add_argument('--in', dest='inp', nargs='+', required=True); ap.add_argument('--tag', default='v1'); ap.add_argument('--limit', type=int); a = ap.parse_args()
J = {r['kid'].rsplit('/', 1)[-1]: r for r in map(json.loads, open('data/homonym_judged.jsonl'))}
def senses(h): return [s['label'] for s in sorted(J[h]['senses'], key=lambda s: -s['share'])]
INSTR = "Which meaning of the keyword does this scholarly work use? Judge from the title and abstract; the venue is a hint only."
def question(h):
    crit = {f"s{i}": lab for i, lab in enumerate(senses(h))}
    crit["none"] = "none of these meanings / the work is not about this keyword"
    return {"sense": {"type": "choice", "instructions": INSTR, "criteria": crit}}
rows = []
for fn in a.inp:
    for r in map(json.loads, open(fn)):
        kid = r.get('kid') or r.get('sid'); rows.append({**r, 'kid': kid})
seen = set(); uniq = []
for r in rows:
    k = (r['kid'], r['work_id'])
    if k not in seen: seen.add(k); uniq.append(r)
rows = uniq[:a.limit] if a.limit else uniq
OUT = f'data/jev_{a.tag}.jsonl'
done = {(r['kid'], r['work_id']) for r in map(json.loads, open(OUT))} if os.path.exists(OUT) else set()
rows = [r for r in rows if (r['kid'], r['work_id']) not in done]
print(len(rows), 'to classify', flush=True)
async def main():
    out = open(OUT, 'a')
    async with JevClient(concurrency=60, cls='experiment', want_rps=150, session='1476: keyword homonym senses', job='1476') as c:
        async def one(r):
            h = r['kid'].rsplit('/', 1)[-1]
            state = {"keyword": h.replace('-', ' '), "title": r.get('title') or '', "abstract": (r.get('abstract') or '')[:1500], "venue": r.get('venue') or ''}
            res = await c.decide(state, question(h), tag=a.tag)
            out.write(json.dumps({'kid': r['kid'], 'work_id': r['work_id'], 'ok': res.ok, 'answers': res.answers, 'model': res.model}) + '\n'); out.flush()
        await asyncio.gather(*(one(r) for r in rows))
        print(f'cost ${c.total_cost:.2f}, {c.n_requests} requests', flush=True)
asyncio.run(main())
