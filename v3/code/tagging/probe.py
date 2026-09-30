"""Throughput / parity probe for the v3 retag: tag the same eval prompts on (gpu, quant) configs; prints works/s (warm) and agreement with the first config.
python probe.py --model /vol/models/student_qwen4ball8 --n 20000 --configs H100:,B200:,B200:fp8 [--eval ../data/student_eval.jsonl] [--out data/probe_<tag>.jsonl]"""
import argparse, json, os, time
import modal
ap = argparse.ArgumentParser(); ap.add_argument('--model', required=True); ap.add_argument('--n', type=int, default=20000); ap.add_argument('--configs', default='H100:,B200:,B200:fp8')
ap.add_argument('--eval', default='../data/student_eval.jsonl'); ap.add_argument('--out'); a = ap.parse_args()
rows = [json.loads(l) for l in open(a.eval)][:a.n]; texts = [r['prompt'] for r in rows]
res = {}
def run(cfg):
    gpu, quant = cfg.split(':'); T = modal.Cls.from_name('oxjob1322-corpus-v3', 'Tagger').with_options(gpu=gpu)(model=a.model, quant=quant)
    T.tag_texts.remote(texts[:512])                     # warm: load + graph capture
    out, s = T.tag_texts.remote(texts); return cfg, out, s
from concurrent.futures import ThreadPoolExecutor
with ThreadPoolExecutor(len(a.configs.split(','))) as ex:
    for cfg, out, s in ex.map(run, a.configs.split(',')): res[cfg] = (out, s)
base = a.configs.split(',')[0]
for cfg, (out, s) in res.items():
    same = sum(o[0] == b[0] for o, b in zip(out, res[base][0])); nk = sum(len(o[0]) for o in out) / len(out)
    print(f'{cfg:10s} {len(texts)/s:6.0f} works/s  ({s:.0f}s)  kw/work {nk:.2f}  identical lists vs {base}: {same/len(out):.3f}', flush=True)
if a.out:
    with open(a.out, 'w') as f:
        for i, r in enumerate(rows): f.write(json.dumps({'id': r['id'], **{cfg: res[cfg][0][i] for cfg in res}}, ensure_ascii=False) + '\n')
