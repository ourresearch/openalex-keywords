"""Apply the 2026-09-29 hand review on top of synmap8 (#1322): Jason's 180 decisions + Claude's 320 on the rest (scratch/review_tool/hand_decisions.jsonl,
A = merge into a, B = merge into b, X = keep apart) and the Opus-confirmed spacing/hyphen variants (data/judged_spacing.jsonl, e.g. schoolchildren /
school children). Both sides of every pair are headings in kw1322v3m, so joins work on headings: the two whole groups merge under the chosen heading
(hand choice first, then the judge's pick, then works). Hand merges bypass the ambiguous-acronym isolation on purpose (TNF-α was isolated).
Checks every keep-apart pair ends in different groups. -> data/synmap9.jsonl. python3 apply_hand.py"""
import json, os, collections
HERE = os.path.dirname(os.path.abspath(__file__))
m = {r['kid']: r['canon'] for r in map(json.loads, open(f'{HERE}/data/synmap8.jsonl'))}
wc = {r['kid']: r['n'] for r in json.load(open(f'{HERE}/data/vocab_kw1322v3m_.json'))}
name = {r['kid']: r['name'] for r in json.load(open(f'{HERE}/data/vocab_kw1322v3m_.json'))}
hand = [json.loads(l) for l in open(f'{HERE}/../../data/synonyms/hand_decisions.jsonl')]
edges = []   # (loser, winner, priority): 2 = hand, 1 = spacing judge
for r in hand:
    if r['d'] == 'A': edges.append((r['b'], r['a'], 2))
    elif r['d'] == 'B': edges.append((r['a'], r['b'], 2))
for r in map(json.loads, open(f'{HERE}/data/judged_spacing.jsonl')):
    if r['merge']: edges.append((r['b'] if r['keep'] == r['a'] else r['a'], r['keep'], 1))
root = lambda k: m.get(k, k)
par = {}
def find(x):
    par.setdefault(x, x)
    while par[x] != x: par[x] = par[par[x]]; x = par[x]
    return x
for l, w, _ in edges: par[find(root(l))] = find(root(w))
groups = collections.defaultdict(set)
for x in list(par): groups[find(x)].add(x)
losers = {(root(l), p) for l, w, p in edges}
def heading(G):
    for p in (2, 1):
        sinks = {root(w) for l, w, q in edges if q == p and root(w) in G} - {x for x, q in losers if q >= p}
        if sinks: return max(sinks, key=lambda k: wc.get(k, 0))
    return max(G, key=lambda k: wc.get(k, 0))
H = {}
for G in groups.values():
    if len(G) > 1:
        h = heading(G)
        for x in G: H[x] = h
n0 = len(m)
for k in list(m):
    if m[k] in H: m[k] = H[m[k]]
for x, h in H.items():
    if x != h: m[x] = h
for k in [k for k, v in m.items() if k == v]: del m[k]
bad = [(r['an'], r['bn']) for r in hand if r['d'] == 'X' and root(r['a']) == root(r['b'])]
big = sorted((G for G in groups.values() if len(G) > 1), key=len, reverse=True)[:5]
with open(f'{HERE}/data/synmap9.jsonl', 'w') as f:
    for k, c in m.items(): f.write(json.dumps({'kid': k, 'canon': c}, ensure_ascii=False) + '\n')
print(f'{len(edges)} forced joins ({sum(p == 2 for *_, p in edges)} hand); {sum(len(G) > 1 for G in groups.values())} groups of headings joined; map {n0:,} -> {len(m):,} kids; keep-apart violations: {bad}')
print('largest joined groups:', [(name.get(heading(G)), len(G), [name.get(x) for x in list(G)[:6]]) for G in big])
chk = ['sex-differences', 'covid-19-pandemic', 'online-learning', 'rice', 'high-school-students', 'school-children', 'tnf-α', 'finite-element-modeling']
print({k: name.get(root(k), root(k)) for k in chk})
