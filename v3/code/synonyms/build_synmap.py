"""Merge map from the judged synonym pairs (#1322): data/judged.jsonl -> data/synmap.jsonl {kid, canon} (only kids that change).
Components of judged merges; to stop chaining (A~B, B~C pulling A and C together), a component of more than 3 keywords is a star around its hub (most
works): only keywords judged equal to the hub itself merge into it; the rest stay separate. A 2- or 3-keyword component merges whole into the heading the
judge preferred most often (ties: most works). Also prints the biggest merges for review. python3 build_synmap.py"""
import json, os, collections
HERE = os.path.dirname(os.path.abspath(__file__))
name, wc = {}, {}
for l in open(f'{HERE}/data/vocab.jsonl'):
    r = json.loads(l); name[r['kid']] = r['display_name']; wc[r['kid']] = r['works_count']
J = [r for r in map(json.loads, open(f'{HERE}/data/judged.jsonl')) if r['merge']]
edges = [(r['a'], r['b']) for r in J]
keepvotes = collections.Counter(r['keep'] for r in J)   # the judge's preferred heading per merge
par = {}
def find(x):
    par.setdefault(x, x)
    while par[x] != x: par[x] = par[par[x]]; x = par[x]
    return x
for a, b in edges: par[find(a)] = find(b)
comp = collections.defaultdict(set)
for a, b in edges: comp[find(a)] |= {a, b}
adj = collections.defaultdict(set)
for a, b in edges: adj[a].add(b); adj[b].add(a)
m = {}; sizes = collections.Counter()
for members in comp.values():
    sizes[min(len(members), 10)] += 1
    if len(members) <= 3: hub = max(members, key=lambda k: (keepvotes[k], wc[k])); group = members   # judge's preferred heading
    else: hub = max(members, key=lambda k: wc[k]); group = {hub} | adj[hub]                                # star around the most-used
    for k in group:
        if k != hub: m[k] = hub
with open(f'{HERE}/data/synmap.jsonl', 'w') as f:
    for k, c in m.items(): f.write(json.dumps({'kid': k, 'canon': c}, ensure_ascii=False) + '\n')
moved = sum(wc[k] for k in m)
print(f'{len(edges):,} judged merges -> {len(m):,} keywords merged into {len(set(m.values())):,} headings; component sizes {dict(sorted(sizes.items()))}; '
      f'keyword-assignments moved (upper bound) {moved:,}')
by = collections.defaultdict(list)
for k, c in m.items(): by[c].append(k)
for c, ks in sorted(by.items(), key=lambda kv: -sum(wc[k] for k in kv[1]))[:30]:
    print(f'{name[c]} ({wc[c]:,})  <=  ' + '; '.join(f'{name[k]} ({wc[k]:,})' for k in sorted(ks, key=lambda k: -wc[k])))
