"""Score one arm on one gold. Arm rows: {"id", "scored": [[cand, score], ...] ranked desc, optional "tokens","cost","latency_ms"}.
python score.py --gold mesh --arm arms/mesh__jev_record.jsonl [--t 0.5] [--top 5] [--json results/x.json] [--label name]
Cuts: keyword golds (judge, akw_*, kp20k, inspec) use a product cut (score >= t, top n; defaults t=0.5 top=5, plus top 8 for akw);
mesh/tibkat choose the F1-best threshold on train+val and apply it to test ('t_fit'); rank metrics (top-5, recall@k) need no threshold.
Every result carries per-split numbers (train/val/test/all) so a campaign can select on val and report test once."""
import argparse, json, os, collections, numpy as np
from common import HERE, SPLITS, prf, jl
import golds
def cut(sc, t=None, top=None):
    P=[c for c,s in sc if t is None or s>=t]; return P[:top] if top else P
def micro(ids, pred, gold):
    tp=np_=ng=0
    for i in ids:
        G=gold(i); Pp=set(pred(i)); tp+=len(G&Pp); np_+=len(Pp); ng+=len(G)
    return prf(tp,np_,ng)
def best_t(ids, arm, gold, grid, top=None):
    best=None
    for t in grid:
        p,r,f=micro(ids, lambda i: cut(arm[i],t,top), gold)
        if best is None or f>best[1]: best=(float(t),f)
    return best[0] if best else 0.5
def ece(ids, arm, gold, skip=frozenset()):
    bins=[[0,0.0,0] for _ in range(10)]
    for i in ids:
        G=gold(i)
        for c,s in arm[i]:
            if c in skip or s>1 or s<0: continue
            b=min(9,int(s*10)); bins[b][0]+=1; bins[b][1]+=s; bins[b][2]+=c in G
    n=sum(b[0] for b in bins); return round(sum(b[0]/n*abs(b[1]/b[0]-b[2]/b[0]) for b in bins if b[0]),4) if n else None
def ceiling(ids, arm, gold):
    hit=tot=0
    for i in ids: G=gold(i); tot+=len(G); hit+=len(G&{c for c,_ in arm[i]})
    return round(hit/max(1,tot),4)
def splits_of(g, ids):
    out={s:[i for i in ids if g.split.get(i)==s] for s in SPLITS}; out['all']=ids; return out
def score(g, rows, t=0.5, top=5):
    arm={r['id']: r.get('scored',[]) for r in rows if 'scored' in r}; ids=[i for i in g.works if i in arm]
    n_err=sum(1 for r in rows if 'error' in r); S=splits_of(g, ids); res={'n':len(ids),'n_missing':len(g.works)-len(ids),'n_err':n_err,'cut':{},'splits':{}}
    ok=[r for r in rows if 'scored' in r and r.get('tokens')]
    res['tokens_per_work']=round(sum(r['tokens'] for r in ok)/len(ok)) if ok else 0
    res['cost_per_work']=round(sum(r.get('cost',0) for r in ok)/len(ok),7) if ok else 0
    res['latency_ms']=round(sum(r.get('latency_ms',0) for r in ok)/len(ok)) if ok else 0
    res['mean_pool']=round(sum(len(arm[i]) for i in ids)/max(1,len(ids)),1)
    if g.name=='judge':
        # threshold fit on train+val (random set) for top-5 F1, so chooser-scored arms (scores not on Jev's scale) are readable; the product cut t stays as 'fixed'
        fitids=[i for i in ids if g.split.get(i) in ('train','val') and g.works[i]['set']=='random' and not g.labels[i]['none_fits']] or ids
        def jf1(tt):
            tp=na=rh=nc=0
            for i in fitids:
                K=cut(arm[i],tt,top); L=g.labels[i]; na+=len(K); tp+=sum(1 for x in K if x in L['relevant']); nc+=len(L['chosen']); rh+=sum(1 for x in L['chosen'] if x in K)
            p=tp/max(1,na); r=rh/max(1,nc); return 2*p*r/max(1e-9,p+r)
        tfit=max(np.arange(0.05,1.0,0.05), key=jf1); res['cut']={'t':t,'top':top,'t_fit':float(tfit),'fit_on':'train+val random'}
        for sp,sids in S.items():
            out={}
            for st,tt in (('random',t),('cited',t),('all',t),('random_fit',tfit),('cited_fit',tfit)):
                G=[i for i in sids if (st in ('all',) or g.works[i]['set']==st.replace('_fit','')) and not g.labels[i]['none_fits']]
                tp=na=rh=nc=zero=empty=unj=0; nC=0
                for i in G:
                    K=cut(arm[i],tt,top); L=g.labels[i]; R=L['relevant']; C=L['chosen']
                    na+=len(K); tp+=sum(1 for x in K if x in R); empty+=not K; unj+=sum(1 for x in K if x not in L['judged'])
                    if C: nC+=1; nc+=len(C); rh+=sum(1 for x in C if x in K); zero+=not any(x in R for x in K)
                p=tp/max(1,na); r=rh/max(1,nc)
                out[st]={'n':len(G),'precision':round(p,4),'recall':round(r,4),'f1':round(2*p*r/max(1e-9,p+r),4),'zero_good':round(zero/max(1,nC),4),'empty':round(empty/max(1,len(G)),4),'mean_n':round(na/max(1,len(G)),2),'unjudged':round(unj/max(1,na),4)}
            res['splits'][sp]=out
        res['headline']=('random','f1'); res['ece']=ece(S['all'], arm, lambda i: g.labels[i]['relevant'])
    elif g.name in ('mesh','tibkat'):
        M=golds.mesh_desc() if g.name=='mesh' else None; skip=(M['PUBTYPE'] if M else set()); ct=(M['CHECKTAGS'] if M else set())
        views=[('a','all',set()),('b','noct',ct),('c','major',set())] if g.name=='mesh' else [('a','all',set())]
        fit=[i for i in ids if g.split.get(i) in ('train','val')] or ids
        grid=np.arange(0.05, max(1.0, max((s for i in ids for _,s in arm[i]), default=1))+0.05, 0.05)
        tfit={}
        for v,lab,excl in views:
            if v=='c': continue
            gold=lambda i,lab=lab: g.labels[i][lab]; arm_v={i:[(c,s) for c,s in arm[i] if c not in skip and c not in excl] for i in ids}
            tfit[v]=best_t(fit, arm_v, gold, grid)
        res['cut']={'t_fit':tfit,'fit_on':'train+val'}
        for sp,sids in S.items():
            out={}
            for v,lab,excl in views:
                gold=lambda i,lab=lab: g.labels[i][lab]; arm_v={i:[(c,s) for c,s in arm[i] if c not in skip and c not in excl] for i in sids}
                if v=='c':
                    p,r,f=micro(sids, lambda i: cut(arm_v[i],None,5), gold); out[v]={'top5':[p,r,f]}
                else:
                    tt=tfit[v]; p,r,f=micro(sids, lambda i: cut(arm_v[i],tt), gold)
                    out[v]={'precision':p,'recall':r,'f1':f,'t':tt,'mean_n':round(sum(len(cut(arm_v[i],tt)) for i in sids)/max(1,len(sids)),1),'ceiling':ceiling(sids, arm_v, gold),'ece':ece(sids, arm_v, gold)}
                    if v=='a':
                        p5,r5,f5=micro(sids, lambda i: cut(arm_v[i],None,5), gold); out[v]['top5']=[p5,r5,f5]
                        if g.name=='tibkat':  # LLMs4Subjects primary metric: average recall@k, k = 5..50, per record, plus P@5 / R@5
                            rk={}
                            for k in range(5,55,5): rk[k]=round(float(np.mean([len(gold(i)&{c for c,_ in arm_v[i][:k]})/max(1,len(gold(i))) for i in sids])),4)
                            out[v]['recall_at']=rk; out[v]['avg_recall']=round(float(np.mean(list(rk.values()))),4)
            if g.name=='mesh':
                for nm,sub in [('Manual',lambda w:w['indexing_method']=='Manual'),('Curated',lambda w:w['indexing_method']=='Curated')]:
                    ss=[i for i in sids if sub(g.works[i])]
                    if ss: out.setdefault('slices',{})[nm]=list(micro(ss, lambda i: cut([(c,s) for c,s in arm[i] if c not in skip],tfit['a']), lambda i: g.labels[i]['all']))
            res['splits'][sp]=out
        res['headline']=('a','f1')
    elif g.name.startswith('akw_'):
        fit=[i for i in ids if g.split.get(i) in ('train','val')] or ids; gold=lambda i: g.labels[i]['akw']
        tfit={n: best_t(fit, arm, gold, np.arange(0.05,1.0,0.05), top=n) for n in (5,8)}   # threshold fit on train+val per cap (chooser scores are not on Jev's scale)
        res['cut']={'t':t,'tops':[5,8],'t_fit':tfit,'fit_on':'train+val'}
        for sp,sids in S.items():
            out={'ceiling':ceiling(sids, arm, gold)}
            for n in (5,8):
                p,r,f=micro(sids, lambda i: cut(arm[i],t,n), gold); out[f'top{n}']={'precision':p,'recall':r,'f1':f,'mean_n':round(sum(len(cut(arm[i],t,n)) for i in sids)/max(1,len(sids)),2)}
                tt=tfit[n]; p,r,f=micro(sids, lambda i: cut(arm[i],tt,n), gold); out[f'top{n}_fit']={'precision':p,'recall':r,'f1':f,'t':tt,'mean_n':round(sum(len(cut(arm[i],tt,n)) for i in sids)/max(1,len(sids)),2)}
            p,r,f=micro(sids, lambda i: cut(arm[i],t), gold); out['uncapped']={'precision':p,'recall':r,'f1':f,'mean_n':round(sum(len(cut(arm[i],t)) for i in sids)/max(1,len(sids)),2)}
            res['splits'][sp]=out
        res['headline']=('top8','f1')
    elif g.name in ('kp20k','inspec'):
        fit=[i for i in ids if g.split.get(i) in ('train','val')] or ids
        tM=best_t(fit, arm, lambda i: g.labels[i]['all'], np.arange(0.05,1.0,0.05)); res['cut']={'t_M':tM,'fit_on':'train+val','tops':[5,10]}
        for sp,sids in S.items():
            out={}
            for lab in ('all','present','absent'):
                gold=lambda i,lab=lab: g.labels[i][lab]; o={'ceiling':ceiling(sids, arm, gold)}
                for n in (5,10): o[f'f1@{n}']=list(micro(sids, lambda i: cut(arm[i],None,n), gold))
                o['f1@M']=list(micro(sids, lambda i: cut(arm[i],tM), gold)); o['mean_M']=round(sum(len(cut(arm[i],tM)) for i in sids)/max(1,len(sids)),2)
                out[lab]=o
            res['splits'][sp]=out
        res['headline']=('all','f1@5')
    return res
if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--gold', required=True); ap.add_argument('--arm', required=True); ap.add_argument('--t', type=float, default=0.5); ap.add_argument('--top', type=int, default=5); ap.add_argument('--json'); ap.add_argument('--label'); a=ap.parse_args()
    g=golds.load(a.gold); rows=jl(a.arm); res=score(g, rows, a.t, a.top)
    label=a.label or os.path.basename(a.arm).replace('.jsonl','').split('__',1)[-1]; res.update(arm=label, gold=a.gold, arm_file=os.path.relpath(a.arm, HERE))
    print(json.dumps({k:v for k,v in res.items() if k!='splits'}), '\n test:', json.dumps(res['splits'].get('test')), '\n all:', json.dumps(res['splits'].get('all')))
    out=a.json or f"{HERE}/results/{a.gold}__{label}.json"; os.makedirs(os.path.dirname(out), exist_ok=True); json.dump(res, open(out,'w'), indent=1)
