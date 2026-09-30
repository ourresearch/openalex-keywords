"""Vocabulary coverage of a gold's labels (measure before any run: coverage caps recall). Exact canon match against vocab_v1 keys, keys ∪ aliases (#1319), and the #1292 vocabulary.
python coverage.py kp20k inspec akw_pubmed ..."""
import json, sys, collections
from common import P, canon, jl
import golds
V1={canon(v['key']) for v in map(json.loads, open(f"{P['j1295']}/vocab_v1/vocab_merged.jsonl"))}
AL=set()
for r in map(json.loads, open(f"{P['j1319']}/aliases_v1.jsonl")):
    for a in r.get('aliases',[]): AL.add(canon(a['form']))
V92={canon(v['key']) for v in map(json.loads, open(f"{P['j1292']}/vocab/vocab.jsonl"))}
print(f'vocab_v1 {len(V1):,} keys; aliases {len(AL):,} forms; #1292 {len(V92):,}')
for name in sys.argv[1:]:
    g=golds.load(name); lab='akw' if name.startswith('akw') else ('relevant' if name=='judge' else 'all')
    if name in ('kp20k','inspec'):  # stemmed gold: compare on stemmed forms
        st=golds._stem(); V1s={st(k) for k in V1}; ALs={st(k) for k in AL}; V92s={st(k) for k in V92}
        for lab in ('present','absent','all'):
            L=[k for i in g.works for k in g.labels[i][lab]]; n=len(L)
            print(f"{name:11s} {lab:8s} n={n:6d}  v1 {sum(k in V1s for k in L)/n:.3f}  v1+aliases {sum(k in V1s or k in ALs for k in L)/n:.3f}  #1292 {sum(k in V92s for k in L)/n:.3f}")
    else:
        L=[k for i in g.works for k in g.labels[i][lab]]; n=len(L)
        print(f"{name:11s} {lab:8s} n={n:6d}  v1 {sum(k in V1 for k in L)/n:.3f}  v1+aliases {sum(k in V1 or k in AL for k in L)/n:.3f}  #1292 {sum(k in V92 for k in L)/n:.3f}")
