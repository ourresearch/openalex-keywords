"""The God eval at scale: Messages Batches API, one request per work, structured JSON {keywords, confidence}. Resumable: skips ids already in results/god_<tag>.jsonl.
python judge_batch.py submit --model claude-opus-5-5 --effort medium [--tag god55] [--chunk 10000] [--limit N]   -> batches/<tag>_<k>.json (batch ids)
python judge_batch.py poll --tag god55            -> collects ended batches into results/god_<tag>.jsonl (keyed by custom_id = work id; order is never positional)
python judge_batch.py retry --tag god55           -> re-runs refusals / errors synchronously with server-side fallbacks (Batches reject the fallbacks parameter)"""
import argparse, json, os, sys, time, glob, collections
import anthropic
import importlib
ap=argparse.ArgumentParser(); ap.add_argument('cmd', choices=['submit','poll','retry','status']); ap.add_argument('--model', default='claude-opus-5-5'); ap.add_argument('--effort', default='medium'); ap.add_argument('--tag', default='god55')
ap.add_argument('--chunk', type=int, default=10000); ap.add_argument('--limit', type=int); ap.add_argument('--sample', default='data/god_sample.jsonl'); ap.add_argument('--workers', type=int, default=8); ap.add_argument('--prompt', default='prompt', help='prompt module (prompt = God v2, prompt_v3 = relabel)'); a=ap.parse_args()
_pm=importlib.import_module(a.prompt); SYSTEM, SCHEMA, user_text = _pm.SYSTEM, _pm.SCHEMA, _pm.user_text
os.makedirs('batches', exist_ok=True); os.makedirs('results', exist_ok=True); out_path=f'results/god_{a.tag}.jsonl'; client=anthropic.Anthropic(max_retries=4, timeout=600)
def done_ids(): return set(json.loads(l)['id'] for l in open(out_path)) if os.path.exists(out_path) else set()
def params(w): return {'model':a.model,'max_tokens':4000,'system':[{'type':'text','text':SYSTEM,'cache_control':{'type':'ephemeral'}}],'messages':[{'role':'user','content':user_text(w)}],'output_config':{'effort':a.effort,'format':{'type':'json_schema','schema':SCHEMA}}}
if a.cmd=='submit':
    requeue=set()
    if os.path.exists(out_path):   # rows that failed on 'credit balance too low' go back into the next batch wave (cheaper than the synchronous retry)
        keep=[]
        for l in open(out_path):
            r=json.loads(l)
            if r.get('error')=='errored' and 'credit balance' in r.get('detail',''): requeue.add(r['id'])
            else: keep.append(l)
        if requeue:
            open(out_path+'.tmp','w').writelines(keep); os.replace(out_path+'.tmp', out_path); print(f'requeued {len(requeue)} credit-balance errors', flush=True)
    works=[json.loads(l) for l in open(a.sample)]; done=done_ids(); submitted=set()
    for f in glob.glob(f'batches/{a.tag}_*.json'): submitted|=set(json.load(open(f))['ids'])
    submitted-=requeue; works.sort(key=lambda w: w['id'] not in requeue)   # requeued first
    works=[w for w in works if w['id'] not in done and w['id'] not in submitted]
    if a.limit: works=works[:a.limit]
    print(f'{len(works)} works to submit in chunks of {a.chunk}', flush=True)
    for k in range(0, len(works), a.chunk):
        ch=works[k:k+a.chunk]; b=client.messages.batches.create(requests=[{'custom_id':w['id'],'params':params(w)} for w in ch])
        n=len(glob.glob(f'batches/{a.tag}_*.json')); json.dump({'batch_id':b.id,'ids':[w['id'] for w in ch],'model':a.model,'effort':a.effort,'created':str(b.created_at)}, open(f'batches/{a.tag}_{n:03d}.json','w'))
        print(f'  batch {b.id}: {len(ch)} requests', flush=True)
elif a.cmd in ('poll','status'):
    done=done_ids(); out=open(out_path,'a'); tot=collections.Counter()
    for f in sorted(glob.glob(f'batches/{a.tag}_*.json')):
        m=json.load(open(f)); b=client.messages.batches.retrieve(m['batch_id']); c=b.request_counts
        print(f"{os.path.basename(f)} {b.id} {b.processing_status} ok={c.succeeded} err={c.errored} exp={c.expired} canc={c.canceled} pending={c.processing}", flush=True)
        if a.cmd=='status' or b.processing_status!='ended' or m.get('collected'): continue
        for r in client.messages.batches.results(b.id):
            if r.custom_id in done: continue
            rec={'id':r.custom_id,'model':m['model'],'effort':m['effort']}
            if r.result.type!='succeeded': rec['error']=r.result.type; rec['detail']=str(getattr(r.result,'error',''))[:200]
            else:
                msg=r.result.message; rec.update(stop_reason=msg.stop_reason, usage={'in':msg.usage.input_tokens,'out':msg.usage.output_tokens})
                if msg.stop_reason=='refusal': rec['error']='refusal'
                else:
                    text=next((bl.text for bl in msg.content if bl.type=='text'), None)
                    try: rec['answer']=json.loads(text)
                    except Exception: rec['error']='bad_json'; rec['raw']=(text or '')[:300]
            out.write(json.dumps(rec, ensure_ascii=False)+'\n'); done.add(r.custom_id); tot['n']+=1; tot['err']+='error' in rec; tot['in']+=rec.get('usage',{}).get('in',0); tot['out']+=rec.get('usage',{}).get('out',0)
        out.flush(); m['collected']=True; json.dump(m, open(f,'w'))
    if tot['n']: print(f"collected {tot['n']} ({tot['err']} errors); mean tokens in {tot['in']/tot['n']:.0f} out {tot['out']/tot['n']:.0f}; total so far {len(done)}", flush=True)
elif a.cmd=='retry':
    works={json.loads(l)['id']: json.loads(l) for l in open(a.sample)}; rows=[json.loads(l) for l in open(out_path)]; bad=[r['id'] for r in rows if 'error' in r and r['id'] in works]; print(f'{len(bad)} to retry synchronously', flush=True)
    byid={r['id']:r for r in rows}; fixed=0; credit_waits=0; t0=time.time()
    def call(kw):
        try: return client.beta.messages.create(**kw, betas=['server-side-fallback-2026-07-01'], fallbacks='default')
        except anthropic.BadRequestError as e:
            if 'credit balance' in str(e): raise
            kw=dict(kw); kw['model']='claude-sonnet-5'; return client.messages.create(**kw)
    import threading; from concurrent.futures import ThreadPoolExecutor, as_completed
    lock=threading.Lock(); state={'fixed':0,'credit_waits':0,'n':0}
    def one(i):
        w=works[i]; kw=params(w); r=None
        for attempt in range(8):
            try: r=call(kw); break
            except Exception as e:
                if 'credit balance' in str(e):
                    with lock: state['credit_waits']+=1
                    time.sleep(120); continue
                return i, {**byid[i], 'error':'retry_failed', 'detail':repr(e)[:200]}
        if r is None: return i, byid[i]
        rec={'id':i,'model':r.model,'effort':a.effort,'stop_reason':r.stop_reason,'usage':{'in':r.usage.input_tokens,'out':r.usage.output_tokens},'retried':True}
        text=next((bl.text for bl in r.content if bl.type=='text'), None)
        if r.stop_reason=='refusal': rec['error']='refusal'
        else:
            try: rec['answer']=json.loads(text)
            except Exception: rec['error']='bad_json'
        return i, rec
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for fut in as_completed([ex.submit(one,i) for i in bad]):
            i,rec=fut.result()
            with lock:
                byid[i]=rec; state['n']+=1; state['fixed']+='answer' in rec
                if state['n']%200==0:
                    with open(out_path,'w') as f:
                        for x in byid.values(): f.write(json.dumps(x, ensure_ascii=False)+'\n')
                    print(f"  {state['n']}/{len(bad)} fixed {state['fixed']} credit_waits {state['credit_waits']} {time.time()-t0:.0f}s", flush=True)
    fixed=state['fixed']; credit_waits=state['credit_waits']
    with open(out_path,'w') as f:
        for x in byid.values(): f.write(json.dumps(x, ensure_ascii=False)+'\n')
    print(f'fixed {fixed} of {len(bad)}; credit waits {credit_waits}', flush=True)
