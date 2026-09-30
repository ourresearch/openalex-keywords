"""Combined merge map for the final build (#1322, 2026-09-28): edges = first pass (synmap_final, already applied once in kw1322v3f_*) + UK/US spelling
(synmap_ukus) + wider pass (judged_f merges, minus merges a context check rejected: data/hi_verified_f.jsonl if present). Union-find over all edges;
components of <= 3 keep the judge's preferred heading (ties: most works), larger ones are stars around the most-used keyword (only direct neighbours
merge). Output data/synmap2.jsonl {kid, canon}, fully resolved (no chains), over kids of the ORIGINAL strings (first-pass kids included).
python3 build_synmap2.py"""
import json, os, collections
HERE = os.path.dirname(os.path.abspath(__file__))
rd = lambda p: [json.loads(l) for l in open(f'{HERE}/{p}')] if os.path.exists(f'{HERE}/{p}') else []
wc = collections.Counter()
for p in ('data/vocab.jsonl', 'data/vocab_f.jsonl'):
    for r in rd(p): wc[r['kid']] = max(wc[r['kid']], r['works_count'])
edges, keep = [], collections.Counter()
for r in rd('data/synmap_final.jsonl'): edges.append((r['kid'], r['canon'])); keep[r['canon']] += 1
for r in rd('data/synmap_ukus.jsonl'): edges.append((r['kid'], r['canon'])); keep[r['canon']] += 1
for r in rd('data/synmap_variants.jsonl'): edges.append((r['kid'], r['canon'])); keep[r['canon']] += 1
rejected = {(r['kid'], r['canon']) for r in rd('data/hi_verified_f.jsonl') if r.get('verified') is False}
for r in rd('data/judged_f.jsonl'):
    if not r['merge']: continue
    a, b = r['a'], r['b']; other = b if r['keep'] == a else a
    if (other, r['keep']) in rejected: continue
    edges.append((other, r['keep'])); keep[r['keep']] += 1
par = {}
def find(x):
    par.setdefault(x, x)
    while par[x] != x: par[x] = par[par[x]]; x = par[x]
    return x
for a, b in edges: par[find(a)] = find(b)
comp = collections.defaultdict(set); adj = collections.defaultdict(set)
for a, b in edges: comp[find(a)] |= {a, b}; adj[a].add(b); adj[b].add(a)
m = {}; big = 0
for members in comp.values():
    if len(members) <= 3: hub = max(members, key=lambda k: (keep[k], wc[k])); group = members
    else: big += 1; hub = max(members, key=lambda k: wc[k]); group = {hub} | adj[hub]
    for k in group:
        if k != hub: m[k] = hub
with open(f'{HERE}/data/synmap2.jsonl', 'w') as f:
    for k, c in m.items(): f.write(json.dumps({'kid': k, 'canon': c}, ensure_ascii=False) + '\n')
print(f'{len(edges):,} edges -> {len(m):,} kids merged into {len(set(m.values())):,} headings ({big:,} star components)')
