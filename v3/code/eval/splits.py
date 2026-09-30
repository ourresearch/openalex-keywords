"""Fixed splits, written once and committed: splits/<gold>.json {id: train|val|test}, 50/20/30, seed 1322, stratified by judge set / MeSH indexing method / TIBKAT kind.
python splits.py [gold ...]   (never overwrites an existing split file)"""
import json, os, random, sys, collections
from common import HERE, GOLDS
import golds
def build(name):
    p=f'{HERE}/splits/{name}.json'
    if os.path.exists(p): print(name, 'exists, kept'); return
    g=golds.load(name); strata=collections.defaultdict(list)
    for i,w in g.works.items(): strata[w.get('set') or w.get('indexing_method') or w.get('kind') or w.get('field') or 'x'].append(i)
    out={}
    for s,ids in sorted(strata.items()):
        ids=sorted(ids); random.Random(1322).shuffle(ids); n=len(ids); a=int(n*0.5); b=int(n*0.7)
        for k,i in enumerate(ids): out[i]='train' if k<a else ('val' if k<b else 'test')
    json.dump(out, open(p,'w'), indent=0, sort_keys=True); c=collections.Counter(out.values()); print(name, dict(c))
if __name__=='__main__':
    for name in (sys.argv[1:] or GOLDS):
        try: build(name)
        except FileNotFoundError as e: print(name, 'skipped:', e)
