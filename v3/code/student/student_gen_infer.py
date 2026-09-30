"""Generate keywords with a trained student for the eval works -> results/arm_<tag>.jsonl {id, keywords}. python student_gen_infer.py --model models/student_qwen06_N90k --tag qwen06_N90k [--limit N] [--bs 16]"""
import argparse, json, os, sys, time, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
ap=argparse.ArgumentParser(); ap.add_argument('--model', required=True); ap.add_argument('--tag', required=True); ap.add_argument('--eval', default='data/student_eval.jsonl'); ap.add_argument('--limit', type=int); ap.add_argument('--bs', type=int, default=16); ap.add_argument('--maxnew', type=int, default=192); a=ap.parse_args()
dev='cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
tok=AutoTokenizer.from_pretrained(a.model); tok.padding_side='left'; model=AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16 if dev=='cuda' else torch.float32).to(dev).eval()
rows=[json.loads(l) for l in open(a.eval)]; rows=rows[:a.limit] if a.limit else rows; outdir=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(a.eval))),'results') if ('/Volumes/' in os.path.abspath(a.eval) or os.path.abspath(a.eval).startswith('/vol/')) else 'results'; os.makedirs(outdir, exist_ok=True); out=open(f'{outdir}/arm_{a.tag}.jsonl','w'); t0=time.time()
for i in range(0, len(rows), a.bs):
    b=rows[i:i+a.bs]; enc=tok([r['prompt'][:4000]+"\nKeywords:" for r in b], return_tensors='pt', padding=True, truncation=True, max_length=560, add_special_tokens=False).to(dev)
    with torch.no_grad(): gen=model.generate(**enc, max_new_tokens=a.maxnew, do_sample=False, pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
    for r,g in zip(b, gen):
        text=tok.decode(g[enc['input_ids'].shape[1]:], skip_special_tokens=True); kws=[k.strip() for k in text.split(';') if k.strip()]
        out.write(json.dumps({'id':r['id'],'keywords':kws[:10]}, ensure_ascii=False)+'\n')
    if (i//a.bs)%20==0: print(f'{i+len(b)} {time.time()-t0:.0f}s', flush=True)
print('done', len(rows), f'{time.time()-t0:.0f}s')
