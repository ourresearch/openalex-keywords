"""Shared paths and helpers for the #1322 benchmark. Every path is absolute so the harness runs from anywhere.
Inputs live in the retired jobs' scratch dirs (never edited from here): #1292 (judge gold, vocab, ft-kw2b), #1306 (MeSH, slices, TIBKAT, priors), #1295 (vocab_v1), #1319 (aliases)."""
import json, os, sys
HERE=os.path.dirname(os.path.abspath(__file__))
# Inputs of the retired OpenAlex jobs (#1292 judge gold, #1306 MeSH / author-keyword slices / TIBKAT, #1295 vocabulary, #1319 aliases). In the release they are
# expected under KW_GOLDS_DIR (default v3/data/eval/golds/), one folder per job; see docs/evaluation.md for what each holds and which are published.
_G=os.environ.get('KW_GOLDS_DIR', os.path.join(HERE, '..', '..', 'data', 'eval', 'golds'))
P={'j1322': f'{_G}/j1322', 'j1292': f'{_G}/j1292', 'j1306': f'{_G}/j1306', 'j1295': f'{_G}/j1295', 'j1319': f'{_G}/j1319', 'j1300': f'{_G}/j1300'}
DATA=f'{HERE}/data'  # gitignored: fetched golds, pools, embeddings
sys.path.insert(0, HERE); from normalize_kw import canon  # canonical keyword form (lower, NFKC, hyphen->space, crude plural fold)
GOLDS=['judge','mesh','akw_field','akw_pubmed','akw_landing','akw_repo','akw_kp20k','tibkat','kp20k','inspec']
AKW=['akw_field','akw_pubmed','akw_landing','akw_repo','akw_kp20k']  # the author-keyword composite (headline benchmark, Jason 2026-09-22): mean test top-8 F1 over these slices
SPLITS=('train','val','test')
def opn(p):  # released gold files over 5 MB are gzipped: fall back to <path>.gz
    import gzip
    return open(p) if os.path.exists(p) or not os.path.exists(p + '.gz') else gzip.open(p + '.gz', 'rt')
def jl(p): return [json.loads(l) for l in opn(p)]
def wjl(p, rows):
    os.makedirs(os.path.dirname(p) or '.', exist_ok=True)
    with open(p,'w') as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False)+'\n')
def prf(tp, npred, ngold):
    p=tp/max(1,npred); r=tp/max(1,ngold); return round(p,4), round(r,4), round(2*p*r/max(1e-9,p+r),4)
