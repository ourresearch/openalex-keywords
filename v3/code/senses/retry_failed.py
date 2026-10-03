"""#1476: re-run the `"ok": false` rows of finished corpus buckets with the same question as corpus_run.py and patch the bucket files in place
(failed line replaced by the new result; still-failing rows stay failed). Run before load_out.py (it skips files already in the Volume).
uv run --with httpx python retry_failed.py"""
import asyncio, glob, json, os, sys
sys.path.insert(0, os.environ.get('JEVKIT_DIR', 'jevkit')); from jevx import JevClient   # OpenAlex's Jev client (not released; see README)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'common')); from dbx import sql
OUT = os.environ.get('SENSES_OUT', 'out')
# same question as corpus_run.py (kept identical; corpus_run.py runs on import)
J = {r['kid'].rsplit('/', 1)[-1]: r for r in map(json.loads, open('data/homonym_judged.jsonl'))}
def senses(h): return [s['label'] for s in sorted(J[h]['senses'], key=lambda s: -s['share'])]
INSTR = "Which meaning of the keyword does this scholarly work use? Judge from the title and abstract; the venue is a hint only."
def question(h):
    crit = {f"s{i}": lab for i, lab in enumerate(senses(h))}
    crit["none"] = "none of these meanings / the work is not about this keyword"
    return {"sense": {"type": "choice", "instructions": INSTR, "criteria": crit}}

async def main():
    files = [f for f in sorted(glob.glob(f'{OUT}/b*.jsonl')) if '"ok": false' in open(f).read()]
    async with JevClient(concurrency=10, cls='backfill', want_rps=20, session='1476: homonym corpus run', job='1476') as c:
        for f in files:
            lines = open(f).read().splitlines()
            bad = {i: json.loads(l) for i, l in enumerate(lines) if '"ok": false' in l}
            ids = ','.join(sorted({str(r['work_id']) for r in bad.values()}))
            b = int(os.path.basename(f)[1:4])
            rows = {(str(r['work_id']), r['heading']): r for r in sql(
                f"SELECT work_id, heading, title, abstract, venue FROM openalex.transient.kw1476_corpus_in WHERE bucket = {b} AND work_id IN ({ids})", verbose=False)}
            async def one(i, r0):
                r = rows[(str(r0['work_id']), r0['heading'])]; h = r['heading']
                state = {"keyword": h.replace('-', ' '), "title": r.get('title') or '', "abstract": (r.get('abstract') or '')[:1500], "venue": r.get('venue') or ''}
                try: res = await c.decide(state, question(h), tag='corpus_v1')
                except Exception: return
                if res.ok:
                    p = res.answers['sense']['probabilities']
                    lines[i] = json.dumps({'work_id': r0['work_id'], 'heading': h, 'p': {k: round(v, 4) for k, v in p.items()}, 'model': res.model})
            await asyncio.gather(*(one(i, r) for i, r in bad.items()))
            open(f + '.tmp', 'w').write('\n'.join(lines) + '\n'); os.replace(f + '.tmp', f)
            print(os.path.basename(f), len(bad), 'retried;', sum('"ok": false' in l for l in lines), 'still failed', flush=True)
asyncio.run(main())
