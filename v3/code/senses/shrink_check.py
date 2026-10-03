"""#1476 Next Action 2: would classifying only works in the field rule's minority fields miss Opus-minority works?
Random gold only (30 per keyword). Weighted by keyword works (gold is 30/keyword, corpus is not).
python shrink_check.py"""
import json, collections
D = 'data'
S = {s['heading']: s for s in json.load(open(f'{D}/senses_summary.json'))}
minf = collections.defaultdict(set)
for l in open(f'{D}/senses.jsonl'):
    r = json.loads(l); minf[r['heading']].add(r['field'])
rows = [json.loads(l) for l in open('data/gold_labels.jsonl')]
tot = collections.Counter(); kw = collections.defaultdict(collections.Counter)
for r in rows:
    h = r['kid'].rsplit('/', 1)[-1]
    if h not in S or r.get('choice') is None: continue
    nopt = len(r['opts']); c = r['choice']
    lab = 'main' if c == 0 else ('none' if c == nopt - 1 else 'minor')
    inmin = (r.get('field') in minf[h]) or r.get('field') is None
    key = (lab, 'minfield' if inmin else 'mainfield')
    tot[key] += 1; kw[h][key] += 1
print('unweighted', dict(tot))
mi = tot[('minor','minfield')] + tot[('minor','mainfield')]
print(f"Opus-minority works in main fields (missed by shrink): {tot[('minor','mainfield')]}/{mi} = {tot[('minor','mainfield')]/mi:.3f}")
# works-weighted
W = 0; Wm = 0; Wmiss = 0; Wcls = 0
for h, c in kw.items():
    n = sum(c.values()); w = S[h]['works'] / n
    Wm += w * (c[('minor','minfield')] + c[('minor','mainfield')]); Wmiss += w * c[('minor','mainfield')]
    Wcls += w * sum(v for k, v in c.items() if k[1] == 'minfield'); W += w * n
print(f"works-weighted: minority share {Wm/W:.3f}; minority missed {Wmiss/Wm:.3f}; share of works to classify {Wcls/W:.3f}")
print('keywords with most missed:', sorted(((c[('minor','mainfield')], h) for h, c in kw.items()), reverse=True)[:15])
