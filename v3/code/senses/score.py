"""#1476 score Jev senses against Opus gold. Keywords hashed into train/test (md5 of kid, even = train). Gold = data/gold_labels.jsonl (random
works) + #1465 senses_valid.jsonl (field-rule works). A work 'moves' when Jev's top sense is a minority sense with probability >= t.
t is chosen on TRAIN only (smallest t with train move-precision >= 0.90), then reported on TEST: move precision (Wilson 95% LB), per-sense
precision, main-sense recall (Opus says main and the work does not move), coverage (share of Opus-minority works that move).
uv run python score.py --tag v1"""
import argparse, collections, hashlib, json, math
ap = argparse.ArgumentParser(); ap.add_argument('--tag', default='v1'); a = ap.parse_args()
def split(kid): return 'train' if int(hashlib.md5(kid.encode()).hexdigest(), 16) % 2 == 0 else 'test'
def wilson(k, n, z=1.96):
    if n == 0: return float('nan')
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)); return (c - r) / d
gold = {}
for r in map(json.loads, open('data/gold_labels.jsonl')):
    if r.get('choice') is not None: gold[(r['kid'], str(r['work_id']))] = (r['choice'], len(r['opts']), 'random')
for r in map(json.loads, open('data/senses_valid.jsonl')):
    k = ('https://openalex.org/keywords/' + r['heading'], str(r['work_id']))
    if k not in gold and r.get('choice') is not None: gold[k] = (r['choice'], len(r['opts']), 'fieldrule')
pred = {}
for r in map(json.loads, open(f'data/jev_{a.tag}.jsonl')):
    if not r['ok']: continue
    p = r['answers']['sense']['probabilities']; n_s = len([k for k in p if k != 'none'])
    probs = [p.get(f's{i}', 0.0) for i in range(n_s)] + [p.get('none', 0.0)]
    pred[(r['kid'], str(r['work_id']))] = probs
rows = []
for k, (g, n, src) in gold.items():
    if k in pred and len(pred[k]) == n: rows.append((k[0], g, pred[k], src))
print(len(gold), 'gold;', len(rows), 'with Jev;', collections.Counter(s for *_, s in rows))
def evaluate(R, t):
    mv = [(kid, g, max(range(len(p)), key=lambda i: p[i]), p) for kid, g, p, _ in R]
    moves = [(kid, g, j) for kid, g, j, p in mv if 0 < j < len(p) - 1 and p[j] >= t]
    ok = sum(g == j for _, g, j in moves)
    main = [(g, j, p) for _, g, j, p in mv if g == 0]
    main_kept = sum(not (0 < j < len(p) - 1 and p[j] >= t) for g, j, p in main)
    minority = [(g, j, p) for _, g, j, p in mv if 0 < g < len(p) - 1]
    cov = sum(j == g and p[j] >= t for g, j, p in minority)
    return dict(moves=len(moves), move_prec=ok / max(1, len(moves)), move_prec_lb=wilson(ok, len(moves)), main_recall=main_kept / max(1, len(main)),
                n_main=len(main), coverage=cov / max(1, len(minority)), n_minority=len(minority)), moves
tr = [r for r in rows if split(r[0]) == 'train']; te = [r for r in rows if split(r[0]) == 'test']
ts = [x / 100 for x in range(50, 100)]
t = next((t for t in ts if evaluate(tr, t)[0]['move_prec'] >= 0.90), 0.99)
print('t chosen on train:', t, '| train', {k: round(v, 3) for k, v in evaluate(tr, t)[0].items()})
res, moves = evaluate(te, t)
print('TEST', {k: round(v, 3) for k, v in res.items()})
by = collections.defaultdict(lambda: [0, 0])
for kid, g, j in moves: by[(kid.rsplit('/', 1)[-1], j)][0] += g == j; by[(kid.rsplit('/', 1)[-1], j)][1] += 1
low = sorted(((k, v[0], v[1]) for k, v in by.items() if v[1] >= 5 and v[0] / v[1] < 0.9), key=lambda x: x[1] / x[2])
print('test senses under 0.9 (>= 5 moves):', low[:25])
json.dump({'t': t, 'test': res, 'train': evaluate(tr, t)[0]}, open(f'data/score_{a.tag}.json', 'w'), indent=1)
