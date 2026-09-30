"""Fetch published author-keyphrase benchmarks into data/<name>.jsonl {id,title,abstract,present,absent} (gitignored; re-run to rebuild).
KP20k test (20,000; taln-ls2n/kp20k keeps title/abstract separate and tags each keyphrase P/R/M/U) -> random 2,000 (seed 1322) for Jev cost; Inspec test (500) whole.
present = PRMU 'P' (verbatim in title+abstract after stemming, the Meng et al. definition); absent = R, M, U.
python fetch_ext.py [--kp20k-n 2000]"""
import argparse, json, random, os
from huggingface_hub import hf_hub_download
from common import DATA, wjl
ap=argparse.ArgumentParser(); ap.add_argument('--kp20k-n', type=int, default=2000); a=ap.parse_args()
os.makedirs(DATA, exist_ok=True)
def rows(repo, fn):
    p=hf_hub_download(repo, fn, repo_type='dataset'); out=[]
    for l in open(p):
        l=l.strip()
        if not l: continue
        r=json.loads(l); kp=r['keyphrases']; pr=r['prmu']
        out.append({'id':f"{repo.split('/')[1]}:{r['id']}", 'title':r['title'], 'abstract':r['abstract'], 'present':[k for k,t in zip(kp,pr) if t=='P'], 'absent':[k for k,t in zip(kp,pr) if t!='P'], 'prmu':pr})
    return out
kp=rows('taln-ls2n/kp20k','test.json'); random.Random(1322).shuffle(kp); kp=kp[:a.kp20k_n]; wjl(f'{DATA}/kp20k.jsonl', kp)
ins=rows('taln-ls2n/inspec','test.jsonl'); wjl(f'{DATA}/inspec.jsonl', ins)
for nm,rs in (('kp20k',kp),('inspec',ins)):
    n=sum(len(r['present'])+len(r['absent']) for r in rs); pres=sum(len(r['present']) for r in rs)
    print(f"{nm}: {len(rs)} docs, {n/len(rs):.1f} keyphrases/doc, present {pres/n:.2f}, absent {(n-pres)/n:.2f}")
