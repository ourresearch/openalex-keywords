"""#1476 corpus run: Jev sense classifier v1 (identical question to jev_senses.py --tag v1) over openalex.transient.kw1476_corpus_in,
one bucket (~50K pairs) at a time -> $SENSES_OUT/b<NNN>.jsonl (default out/) (written to .part, renamed when the bucket is done;
resumable: done buckets are skipped). Memory guard (shared machine): bounded worker pool, one bucket in memory, exits if RSS > 1 GB or free < 50%.
uv run --with httpx python corpus_run.py [--buckets 0-319] [--limit N] [--rps 300]"""
import argparse, asyncio, json, os, resource, subprocess, sys, time
sys.path.insert(0, os.environ.get('JEVKIT_DIR', 'jevkit')); from jevx import JevClient   # OpenAlex's Jev client (not released; see README)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'common')); from dbx import sql
ap = argparse.ArgumentParser(); ap.add_argument('--buckets', default='0-319'); ap.add_argument('--limit', type=int)
ap.add_argument('--rps', type=float, default=300); ap.add_argument('--conc', type=int, default=120); a = ap.parse_args()
lo, hi = map(int, a.buckets.split('-')) if '-' in a.buckets else (int(a.buckets),) * 2
OUT = os.environ.get('SENSES_OUT', 'out'); os.makedirs(OUT, exist_ok=True)
J = {r['kid'].rsplit('/', 1)[-1]: r for r in map(json.loads, open('data/homonym_judged.jsonl'))}
def senses(h): return [s['label'] for s in sorted(J[h]['senses'], key=lambda s: -s['share'])]
INSTR = "Which meaning of the keyword does this scholarly work use? Judge from the title and abstract; the venue is a hint only."
QS = {}
def question(h):
    if h not in QS:
        crit = {f"s{i}": lab for i, lab in enumerate(senses(h))}
        crit["none"] = "none of these meanings / the work is not about this keyword"
        QS[h] = {"sense": {"type": "choice", "instructions": INSTR, "criteria": crit}}
    return QS[h]
def mem_ok(max_gb=1.0, min_free=50):
    rss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9
    free = int(subprocess.run(['memory_pressure'], capture_output=True, text=True).stdout.strip().rsplit(' ', 1)[-1].rstrip('%'))
    return rss_gb < max_gb and free >= min_free, rss_gb, free

def fetch(b):
    """Retries for up to ~1 h: a network drop killed the run once (1 Oct)."""
    for att in range(60):
        try: return _fetch(b)
        except Exception as e:
            print(f'{time.strftime("%H:%M:%S")} fetch b{b:03d} failed ({repr(e)[:120]}); retry in 60 s', flush=True); time.sleep(60)
    return _fetch(b)

def _fetch(b):
    return sql(f"SELECT work_id, heading, title, abstract, venue FROM openalex.transient.kw1476_corpus_in WHERE bucket = {b}"
               + (f" LIMIT {a.limit}" if a.limit else ""), verbose=False)

async def main():
    """Pipelined: one shared worker pool; the next bucket downloads while the current one runs, so the pool never drains between buckets.
    Per bucket: .part file, renamed when its last row is written. At most two buckets in memory."""
    t0 = time.time(); stats = {'pairs': 0}
    todo = [b for b in range(lo, hi + 1) if not os.path.exists(f'{OUT}/b{b:03d}.jsonl')]
    q = asyncio.Queue(maxsize=2 * a.conc); open_b = {}   # b -> [file, remaining, fails, n, t_start, producer_done]
    def finish(b):
        f, rem, nf, n, tb, done = open_b[b]
        if rem == 0 and done:
            f.close(); os.rename(f'{OUT}/b{b:03d}.jsonl.part', f'{OUT}/b{b:03d}.jsonl'); del open_b[b]; stats['pairs'] += n
            ok, rss, free = mem_ok()
            print(f'{time.strftime("%H:%M:%S")} b{b:03d} {n} pairs, {nf} failed | total {stats["pairs"]} {stats["pairs"]/(time.time()-t0):.0f}/s '
                  f'${c.total_cost:.2f} | rss {rss:.2f} GB free {free}%', flush=True)
    async def worker():
        while True:
            item = await q.get()
            if item is None: return
            b, r = item; h = r['heading']; st = open_b[b]
            state = {"keyword": h.replace('-', ' '), "title": r.get('title') or '', "abstract": (r.get('abstract') or '')[:1500], "venue": r.get('venue') or ''}
            try: res = await c.decide(state, question(h), tag='corpus_v1'); err = None
            except Exception as e: res = None; err = repr(e)[:200]
            if res is not None and res.ok:
                p = res.answers['sense']['probabilities']
                st[0].write(json.dumps({'work_id': r['work_id'], 'heading': h, 'p': {k: round(v, 4) for k, v in p.items()}, 'model': res.model}) + '\n')
            else:
                st[2] += 1
                st[0].write(json.dumps({'work_id': r['work_id'], 'heading': h, 'ok': False, 'err': (res.error if res else err)}) + '\n')
            st[1] -= 1; finish(b)
    global c
    async with JevClient(concurrency=a.conc, cls='backfill', want_rps=a.rps, session='1476: keyword homonym senses', job='1476') as c:
        ws = [asyncio.create_task(worker()) for _ in range(a.conc)]
        nxt = asyncio.create_task(asyncio.to_thread(fetch, todo[0])) if todo else None
        for k, b in enumerate(todo):
            rows = await nxt
            nxt = asyncio.create_task(asyncio.to_thread(fetch, todo[k + 1])) if k + 1 < len(todo) else None
            ok, rss, free = mem_ok()
            if rss >= 1.0: print(f'STOP memory guard: own rss {rss:.2f} GB', flush=True); break
            while free < 50:   # machine below 50% free: pause (stop feeding work) until it is back to 55% (a dip to 49% once ended the run)
                print(f'{time.strftime("%H:%M:%S")} PAUSE memory guard: free {free}%', flush=True)
                await asyncio.sleep(120); ok, rss, free = mem_ok(min_free=55)
                if free >= 55: break
                free = min(free, 49)
            open_b[b] = [open(f'{OUT}/b{b:03d}.jsonl.part', 'w'), len(rows), 0, len(rows), time.time(), False]
            for r in rows: await q.put((b, r))
            open_b[b][5] = True; finish(b) if b in open_b else None
            del rows
        for _ in ws: await q.put(None)
        await asyncio.gather(*ws)
        if nxt: nxt.cancel()
    print(f'DONE {stats["pairs"]} pairs in {(time.time()-t0)/3600:.1f} h, ${c.total_cost:.2f}', flush=True)
asyncio.run(main())
