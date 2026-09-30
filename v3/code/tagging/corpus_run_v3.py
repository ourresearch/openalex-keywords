"""Desk driver for the v3 retag (#1322): Tagger.run_shard.map over every shard of kw_corpus_input_v3 not yet in the ledger; Modal fans out to max_containers.
python corpus_run_v3.py [--shards 0-3999] [--gpu B200] [--quant fp8] [--model /vol/models/student_qwen4b_v3] [--budget-usd 5500]
Ledger data/ledger_v3.jsonl (one line per finished shard; rerun = resume). Progress line every 25 shards: works/s, ETA, spend. Then load with load_v3.py (COPY INTO)."""
import argparse, json, os, time
import modal
HERE = os.path.dirname(os.path.abspath(__file__)); os.makedirs(f'{HERE}/data', exist_ok=True)
ap = argparse.ArgumentParser(); ap.add_argument('--shards', default='0-3999'); ap.add_argument('--gpu', default='B200'); ap.add_argument('--quant', default='')
ap.add_argument('--model', default='/vol/models/student_qwen4b_v3'); ap.add_argument('--budget-usd', type=float, default=5500); ap.add_argument('--first', help='json list of shards to run first'); ap.add_argument('--ledger', default='ledger_v3.jsonl'); ap.add_argument('--out-dir', default='/Volumes/openalex/authors/oxjob_tmp/oxjob1322_god/corpus_v3'); a = ap.parse_args()
PRICE = {'B200': (6.25 + 0.9) / 3600, 'H100': (3.95 + 0.9) / 3600, 'H200': (4.54 + 0.9) / 3600}[a.gpu]   # $/container-second incl. cpu/mem
LEDGER = f'{HERE}/data/{a.ledger}'
done = {json.loads(l)['shard'] for l in open(LEDGER)} if os.path.exists(LEDGER) else set()
spent = sum(json.loads(l).get('usd', 0) for l in open(LEDGER)) if os.path.exists(LEDGER) else 0.0
lo, hi = map(int, a.shards.split('-')); todo = [s for s in range(lo, hi + 1) if s not in done]
if a.first: f = [s for s in json.load(open(a.first)) if s in set(todo)]; todo = f + [s for s in todo if s not in set(f)]
print(f'{len(todo)} shards to run ({len(done)} done, ${spent:.0f} spent); gpu {a.gpu} quant {a.quant or "bf16"}', flush=True)
T = modal.Cls.from_name('oxjob1322-corpus-v3', 'Tagger').with_options(gpu=a.gpu)(model=a.model, quant=a.quant, out_dir=a.out_dir)
t0 = time.time(); n = works = 0; errs = 0
with open(LEDGER, 'a') as led:
    for r in T.run_shard.map(todo, order_outputs=False, return_exceptions=True):
        if isinstance(r, Exception): errs += 1; print('  shard error:', repr(r)[:200], flush=True); continue
        r['usd'] = round(r['total_s'] * PRICE, 3); r['gpu'] = a.gpu; r['quant'] = a.quant; r['ts'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        led.write(json.dumps(r) + '\n'); led.flush(); n += 1; works += r['works']; spent += r['usd']
        if n % 25 == 0 or n == len(todo):
            el = time.time() - t0; rate = works / el
            print(f'{time.strftime("%H:%M")} {n}/{len(todo)} shards, {works:,} works, {rate:,.0f} works/s, ETA {(len(todo) - n) * el / n / 3600:.1f} h, spent ${spent:,.0f}, errors {errs}', flush=True)
        if spent >= a.budget_usd: print(f'budget ${a.budget_usd} reached; stopping (rerun to resume)', flush=True); break
print(f'finished: {n} shards, {works:,} works, {errs} errors, ${spent:,.0f}', flush=True)
