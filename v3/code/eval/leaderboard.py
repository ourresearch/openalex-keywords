"""Write leaderboard.md from results/*.json: one table per gold, test split first (the number to quote), 'all' beside it (comparable to the retired jobs' numbers).
python leaderboard.py [--out leaderboard.md]"""
import argparse, json, glob, os
from common import HERE, GOLDS, AKW
ap=argparse.ArgumentParser(); ap.add_argument('--out', default=f'{HERE}/leaderboard.md'); a=ap.parse_args()
R=[json.load(open(p)) for p in sorted(glob.glob(f'{HERE}/results/*.json'))]
def f(x): return '–' if x is None else (f'{x:.3f}' if isinstance(x,float) else str(x))
def cost(r): return f"{r.get('tokens_per_work',0):,} / ${r.get('cost_per_work',0)*1e6:.0f}/M" if r.get('tokens_per_work') else '0 / $0'
L=['# Keyword tagger leaderboard (#1322)','', 'One row per arm per gold. **test** = the 30% split never used for fitting (quote this); **all** = every work (comparable to #1306/#1292 numbers). Cost = mean Jev input tokens per work / $ per million works. Splits: `splits/*.json` (seed 1322, 50/20/30).','']
# composite first: arms present on every AKW slice (same arm name), mean of test top-8 F1 at the fitted threshold
def composite():
    by={}
    for r in R:
        if r['gold'] in AKW: by.setdefault(r['arm'],{})[r['gold']]=r
    rows=[(a,d) for a,d in by.items() if all(g in d for g in AKW)]
    if not rows: return
    L.append('## Author-keyword composite (headline): mean test top-8 F1 over '+', '.join(AKW)); L.append('')
    L.append('The product\'s own definition (what a well-informed author would tag). Exact match after canon at the threshold fit on train+val; MeSH and the judge are diagnostics below, not part of the headline.'); L.append('')
    L.append('| arm | **composite** | '+' | '.join(g[4:] for g in AKW)+' | tokens/work |'); L.append('|---|---|'+'---|'*len(AKW)+'---|')
    def f8(r): return r['splits']['test'].get('top8_fit', r['splits']['test']['top8'])['f1']
    for a,d in sorted(rows, key=lambda x: -sum(f8(x[1][g]) for g in AKW)/len(AKW)):
        L.append(f"| {a} | **{sum(f8(d[g]) for g in AKW)/len(AKW):.3f}** | "+' | '.join(f"{f8(d[g]):.3f}" for g in AKW)+f" | {d[AKW[0]].get('tokens_per_work',0):,} |")
    L.append('')
composite()
def sect(gold, title, header, rowf, note=''):
    rs=[r for r in R if r['gold']==gold]
    if not rs: return
    L.append(f'## {title}'); L.append('')
    if note: L.append(note); L.append('')
    L.append('| arm | n | '+header+' | tokens / cost |'); L.append('|---|---|'+'---|'*(header.count('|')+1)+'---|')
    key=lambda r: -(rowf(r,'test',True) or 0)
    for r in sorted(rs, key=key): L.append(f"| {r['arm']} | {r['n']} | {rowf(r,'test')} | {cost(r)} |")
    L.append('')
def judge_row(r, sp, val=False):
    s=r['splits'][sp]; a=s.get('random_fit', s['random']); c=s.get('cited_fit', s['cited']); al=r['splits']['all']['random']; tf=r['cut'].get('t_fit',0.5)
    if val: return a['f1']
    return f"{f(a['precision'])} / {f(a['recall'])} / **{f(a['f1'])}** / {f(a['zero_good'])} ({tf:.2f}) | {f(c['precision'])} / {f(c['recall'])} / {f(c['zero_good'])} | {f(s['random']['precision'])} / {f(s['random']['recall'])} / {f(s['random']['f1'])} | {f(a['unjudged'])} | {f(a['mean_n'])}"
sect('judge','Judge keyword gold v1.2 (1,198 works; Fable judge; v1.1 + 4,478 relevant from the #1322 delta-judge; top 5 at the threshold fit on train+val)','random test P / R / **F1** / zero-good (t fit) | cited test P / R / zero-good | random test at p ≥ 0.5: P / R / F1 | unjudged share | kw/work', judge_row,
     'Precision counts only judged candidates as right; **unjudged share** = share of assigned keywords the judge never saw (delta-judge before reading precision when it is not ~0).')
def mesh_row(r, sp, val=False):
    s=r['splits'][sp]; al=r['splits']['all']
    if val: return s['a']['f1']
    return f"{f(s['a']['precision'])} / {f(s['a']['recall'])} / **{f(s['a']['f1'])}** (t {s['a']['t']:.2f}) | {f(s['b']['precision'])} / {f(s['b']['recall'])} / **{f(s['b']['f1'])}** | {f(s['c']['top5'][2])} | {f(al['a']['f1'])} / {f(al['b']['f1'])} | {f(s['a']['ceiling'])} | {f(s['a']['ece'])} | {f(s['a']['mean_n'])}"
sect('mesh','Human MeSH indexing (4,728 PubMed works 2017–2022; micro P / R / F1; threshold fit on train+val)','view a test P / R / **F1** | view b (no check tags) test | view c top-5 F1 | all: a / b F1 | pool ceiling | ECE | headings/work', mesh_row,
     'References: NLM MTIX 0.74 (supervised, in-domain), two human indexers ≈ 0.48, the restated bar 0.50 (view a or b).')
def akw_row(r, sp, val=False):
    s=r['splits'][sp]; al=r['splits']['all']; a5=s.get('top5_fit', s['top5']); a8=s.get('top8_fit', s['top8'])
    if val: return a8['f1']
    return f"{f(a5['precision'])} / {f(a5['recall'])} / **{f(a5['f1'])}** ({a5.get('t',0.5):.2f}) | {f(a8['precision'])} / {f(a8['recall'])} / **{f(a8['f1'])}** ({a8.get('t',0.5):.2f}) | {f(s['top5']['f1'])} / {f(s['top8']['f1'])} | {f(al.get('top8_fit', al['top8'])['f1'])} | {f(s['ceiling'])} | {f(a8['mean_n'])}"
for sl in ('pubmed','landing','repo','kp20k'):
    sect(f'akw_{sl}',f"Authors' own keywords, held out: {sl} slice (exact match after canon; threshold fit on train+val per cap; product cut p ≥ 0.5 beside it)",'top-5 test P / R / **F1** (t fit) | top-8 test P / R / **F1** (t fit) | at p ≥ 0.5: F1 top-5 / top-8 | all top-8 F1 | pool ceiling | kw/work', akw_row)
def tib_row(r, sp, val=False):
    s=r['splits'][sp]['a']; al=r['splits']['all']['a']
    if val: return s['avg_recall']
    return f"**{f(s['avg_recall'])}** | {f(s['recall_at']['5'] if '5' in s['recall_at'] else s['recall_at'][5])} | {f(s['top5'][0])} / {f(s['top5'][1])} / {f(s['top5'][2])} | {f(s['precision'])} / {f(s['recall'])} / {f(s['f1'])} (t {s['t']:.2f}) | {f(al['avg_recall'])} | {f(s['ceiling'])}"
sect('tibkat','TIBKAT / GND (LLMs4Subjects gold; 2,000 English records, 2.7 subjects each; official metric = average recall@5…50)','**avg recall@5…50** test | R@5 | top-5 P / R / F1 | best-t P / R / F1 | all avg recall | pool ceiling', tib_row,
     'References (all-subjects leaderboard, full test set): Annif 0.63 avg recall, P@5 0.26, R@5 0.49; DUTIR831 0.60; RUC 0.59. Our subset is English-with-abstract only (easier) and pool-capped.')
def kp_row(r, sp, val=False):
    s=r['splits'][sp]
    if val: return s['all']['f1@5'][2]
    return f"**{f(s['all']['f1@5'][2])}** / {f(s['all']['f1@10'][2])} / {f(s['all']['f1@M'][2])} | {f(s['present']['f1@5'][2])} / {f(s['present']['f1@10'][2])} / {f(s['present']['f1@M'][2])} | {f(s['absent']['f1@5'][2])} / {f(s['absent']['f1@M'][2])} | {f(s['all']['ceiling'])} | {f(r['splits']['all']['all']['f1@5'][2])}"
for nm,ttl,ref in [('kp20k','KP20k (Meng et al. 2017; author keyphrases, CS; Porter-stemmed exact match)','Published: supervised generators F1@5 ≈ 0.31–0.38, zero-shot LLMs ≈ 0.25–0.30, unsupervised extractors ≈ 0.15–0.20; absent F1 ≈ 0.03–0.06 (re-verify before quoting).'),
               ('inspec','Inspec (Hulth 2003; indexer keyphrases on physics/CS abstracts)','Published: generators F1@5 ≈ 0.30–0.35, unsupervised ≈ 0.25–0.30 (re-verify).')]:
    sect(nm, ttl, 'all F1@5 / @10 / @M test | present F1@5 / @10 / @M | absent F1@5 / @M | pool ceiling | all-split F1@5', kp_row, ref)
open(a.out,'w').write('\n'.join(L)); print('wrote', a.out, len(R), 'results')
