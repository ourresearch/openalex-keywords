"""Generative student: fine-tune Qwen3-0.6B (causal LM, loss on the target only) to emit Opus's keyword list from the work text. Runs on a GPU (Databricks A10G) or, for smoke tests, MPS.
python student_gen_train.py --train data/student_train.jsonl --out models/student_qwen06_N90k [--epochs 2] [--bs 8] [--lr 5e-5] [--maxlen 640] [--limit N]
Output format the model learns: '<work text>\\nKeywords: kw1; kw2; ...<eos>'."""
import argparse, json, os, sys, time, torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_schedule_with_warmup
ap=argparse.ArgumentParser(); ap.add_argument('--train', required=True); ap.add_argument('--out', required=True); ap.add_argument('--base', default='Qwen/Qwen3-0.6B'); ap.add_argument('--epochs', type=float, default=2); ap.add_argument('--bs', type=int, default=8); ap.add_argument('--lr', type=float, default=5e-5)
ap.add_argument('--maxlen', type=int, default=640); ap.add_argument('--limit', type=int); ap.add_argument('--grad-accum', type=int, default=2); ap.add_argument('--optim8bit', action='store_true', help='bitsandbytes AdamW8bit (fits 1.7B on a 24 GB GPU)'); a=ap.parse_args()
dev='cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'); dtype=torch.bfloat16 if dev=='cuda' else torch.float32
tok=AutoTokenizer.from_pretrained(a.base); tok.padding_side='right'
model=AutoModelForCausalLM.from_pretrained(a.base, torch_dtype=dtype).to(dev); model.gradient_checkpointing_enable(); model.config.use_cache=False
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
dl=DataLoader(DS(), batch_size=a.bs, shuffle=True, collate_fn=collate, num_workers=0)
if a.optim8bit:
    import bitsandbytes as bnb; opt=bnb.optim.AdamW8bit(model.parameters(), lr=a.lr, weight_decay=0.01)
else: opt=torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
steps=int(len(dl)*a.epochs/a.grad_accum); sch=get_cosine_schedule_with_warmup(opt, int(0.03*steps), steps)
print(f'{dev} {len(rows):,} rows, {steps} optimizer steps', flush=True); model.train(); t0=time.time(); step=0; k=0; run=0.0
for ep in range(int(a.epochs+0.999)):
    for ids,att,lab in dl:
        if step>=steps: break
        out=model(input_ids=ids.to(dev), attention_mask=att.to(dev), labels=lab.to(dev)); (out.loss/a.grad_accum).backward(); run+=out.loss.item(); k+=1
        if k%a.grad_accum==0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sch.step(); opt.zero_grad(); step+=1
            if step%50==0: print(f'step {step}/{steps} loss {run/k:.3f} {time.time()-t0:.0f}s', flush=True); run=0.0; k=0
os.makedirs(a.out, exist_ok=True); model.save_pretrained(a.out); tok.save_pretrained(a.out); print('saved', a.out, f'{time.time()-t0:.0f}s')
