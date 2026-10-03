"""#1476: run ship_1476.sql one statement at a time (PRODUCTION; signed off 2026-10-02). Stops on the first error; prints each result.
Resume after a failure with --from N (1-based statement number).
python run_ship.py [--file ship2_1476.sql] [--from N]"""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'common')); from dbx import sql
F = sys.argv[sys.argv.index('--file') + 1] if '--file' in sys.argv else 'ship_1476.sql'
st = [s.strip() for s in open(F).read().split(';\n\n') if s.strip()]
start = int(sys.argv[sys.argv.index('--from') + 1]) if '--from' in sys.argv else 1
for i, s in enumerate(st, 1):
    if i < start: continue
    body = s.rstrip(';'); head = [l for l in body.splitlines() if not l.startswith('--')][0][:110]
    t0 = time.time(); print(f'[{i}/{len(st)}] {time.strftime("%H:%M:%S")} {head}', flush=True)
    r = sql(body, verbose=False)
    print(f'   ok {time.time()-t0:.0f}s {str(r)[:400]}', flush=True)
