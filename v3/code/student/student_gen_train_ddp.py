"""Multi-GPU (DDP via accelerate) version of student_gen_train.py for the 940K run on a 4x A10G node. Same data/format/loss; bf16 weights, optional 8-bit AdamW.
accelerate launch --multi_gpu --num_processes 4 student_gen_train_ddp.py --train ... --out ... [--limit N] [--epochs 1] [--bs 4] [--grad-accum 2] [--maxlen 640] [--base Qwen/Qwen3-1.7B] [--optim8bit]"""
import argparse, json, os, sys, time, torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_schedule_with_warmup
from accelerate import Accelerator
ap=argparse.ArgumentParser(); ap.add_argument('--train', required=True); ap.add_argument('--out', required=True); ap.add_argument('--base', default='Qwen/Qwen3-1.7B'); ap.add_argument('--epochs', type=float, default=1); ap.add_argument('--bs', type=int, default=4); ap.add_argument('--lr', type=float, default=5e-5)
ap.add_argument('--maxlen', type=int, default=640); ap.add_argument('--limit', type=int); ap.add_argument('--grad-accum', type=int, default=2); ap.add_argument('--optim8bit', action='store_true'); ap.add_argument('--no-grad-ckpt', action='store_true'); a=ap.parse_args()
acc=Accelerator(gradient_accumulation_steps=a.grad_accum, mixed_precision='no')
tok=AutoTokenizer.from_pretrained(a.base); tok.padding_side='right'
model=AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.bfloat16); model.config.use_cache=False
if not a.no_grad_ckpt: model.gradient_checkpointing_enable()
rows=[json.loads(l) for l in open(a.train)]
if a.limit: rows=rows[:a.limit]
class DS(Dataset):
    def __len__(self): return len(rows)
    def __getitem__(self, i):
        r=rows[i]; p=tok(r['prompt'][:4000]+"\nKeywords:", add_special_tokens=False)['input_ids']; t=tok(" "+r['target']+tok.eos_token, add_special_tokens=False)['input_ids']
        p=p[:a.maxlen-len(t)-1] if len(p)+len(t)>a.maxlen else p; ids=p+t; labels=[-100]*len(p)+t
        return ids[:a.maxlen], labels[:a.maxlen]
def collate(b):
    m=max(len(x[0]) for x in b); ids=torch.full((len(b),m), tok.pad_token_id); lab=torch.full((len(b),m), -100); att=torch.zeros((len(b),m), dtype=torch.long)
    for i,(x,y) in enumerate(b): ids[i,:len(x)]=torch.tensor(x); lab[i,:len(y)]=torch.tensor(y); att[i,:len(x)]=1
    return ids, att, lab
dl=DataLoader(DS(), batch_size=a.bs, shuffle=True, collate_fn=collate, num_workers=2)
if a.optim8bit:
    import bitsandbytes as bnb; opt=bnb.optim.AdamW8bit(model.parameters(), lr=a.lr, weight_decay=0.01)
else: opt=torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
steps=int(len(dl)*a.epochs/(a.grad_accum*acc.num_processes))
# the prepared scheduler steps num_processes times per optimizer step (accelerate, split_batches=False), so size it in process-steps;
# sizing it in optimizer steps ran warmup 8x short and cycled the cosine ~4 times on 8 GPUs (found by #1363, 2026-09-27)
sch=get_cosine_schedule_with_warmup(opt, int(0.03*steps)*acc.num_processes, steps*acc.num_processes)
model, opt, dl, sch = acc.prepare(model, opt, dl, sch)
if acc.is_main_process: print(f'{acc.num_processes} procs, {len(rows):,} rows, {steps} optimizer steps', flush=True)
model.train(); t0=time.time(); step=0; run=0.0; k=0
for ep in range(int(a.epochs+0.999)):
    for ids,att,lab in dl:
        if step>=steps: break
        with acc.accumulate(model):
            out=model(input_ids=ids, attention_mask=att, labels=lab); acc.backward(out.loss); run+=out.loss.item(); k+=1
            if acc.sync_gradients: acc.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sch.step(); opt.zero_grad()
            if acc.sync_gradients:
                step+=1
                if step%50==0 and acc.is_main_process: print(f'step {step}/{steps} loss {run/max(1,k):.3f} lr {opt.param_groups[0]["lr"]:.2e} {time.time()-t0:.0f}s', flush=True); run=0.0; k=0
acc.wait_for_everyone()
if acc.is_main_process:
    os.makedirs(a.out, exist_ok=True); acc.unwrap_model(model).save_pretrained(a.out, save_function=acc.save); tok.save_pretrained(a.out); print('saved', a.out, f'{time.time()-t0:.0f}s', flush=True)
