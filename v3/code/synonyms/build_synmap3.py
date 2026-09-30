"""Round-3 combined merge map (#1322, 2026-09-28 evening). Edges: everything in synmap2 (already applied in kw1322v3g_*) plus round-3 merges on g headings:
judged_v2_opus / judged_v2_tail (v2 rubric), synmap_acronyms_ok, judged_alt (generated alternative names, confirmed by the v2 judge), minus merges a context
check rejected (data/hi_verified_v3.jsonl). Components of <= 3 keep the judge's preferred heading; bigger ones: every member within 2 judged hops of the hub
merges (round 2 used 1 hop and dropped real synonyms). Reports what the chaining guard dropped. -> data/synmap3.jsonl. python3 build_synmap3.py [--hops 2]"""
import json, os, collections, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser(); ap.add_argument('--hops', type=int, default=2); ap.add_argument('--extra-judged', nargs='*', default=[]); ap.add_argument('--extra-maps', nargs='*', default=[]); ap.add_argument('--rejected', nargs='*', default=['data/hi_verified_v3.jsonl']); ap.add_argument('--out', default='data/synmap3.jsonl'); a = ap.parse_args()
rd = lambda p: [json.loads(l) for l in open(f'{HERE}/{p}')] if os.path.exists(f'{HERE}/{p}') else []
wc = collections.Counter()
for p in ('data/vocab.jsonl', 'data/vocab_f.jsonl', 'data/vocab_g.jsonl', 'data/vocab_h.jsonl'):
    for r in rd(p): wc[r['kid']] = max(wc[r['kid']], r['works_count'])
edges, keep = [], collections.Counter()
for r in rd('data/synmap2.jsonl'): edges.append((r['kid'], r['canon'])); keep[r['canon']] += 1
for p in ['data/synmap_acronyms_ok.jsonl'] + a.extra_maps:
    for r in rd(p): edges.append((r['kid'], r['canon'])); keep[r['canon']] += 1
rejected = {(r['kid'], r['canon']) for p in a.rejected for r in rd(p) if r.get('verified') is False}
for p in ['data/judged_v2_opus.jsonl', 'data/judged_v2_tail.jsonl', 'data/judged_alt.jsonl'] + a.extra_judged:
    for r in rd(p):
        if not r.get('merge'): continue
        other = r['b'] if r['keep'] == r['a'] else r['a']
        if (other, r['keep']) in rejected: continue
        edges.append((other, r['keep'])); keep[r['keep']] += 1
# ambiguous acronyms (2026-09-29 precision audit: TAM joined tumor-associated macrophages and the technology acceptance model): an acronym-like keyword
# linked to >= 2 distinct non-acronym keywords is dropped from every edge, so it can neither merge nor bridge two concepts
import re
_disp = {}
for _p in ('data/vocab_all.jsonl',):
    for _r in rd(_p): _disp.setdefault(_r['kid'], _r['display_name'])
def _acr(k):   # a real acronym: display name with >= 2 capitals, no spaces, <= 12 chars (the id alone made 'elderly' an acronym, 2026-09-29)
    d = _disp.get(k, '')
    return bool(d) and ' ' not in d and len(d) <= 12 and sum(ch.isupper() for ch in d) >= 2
nbr = collections.defaultdict(set)
for x, y in edges: nbr[x].add(y); nbr[y].add(x)
_p = {}
def _f(x):
    _p.setdefault(x, x)
    while _p[x] != x: _p[x] = _p[_p[x]]; x = _p[x]
    return x
for x, y in edges:
    if not _acr(x) and not _acr(y): _p[_f(x)] = _f(y)
amb = {k for k, ns in nbr.items() if _acr(k) and len({_f(n) for n in ns if not _acr(n)}) >= 2}   # expansions in >= 2 different concept groups
edges = [(x, y) for x, y in edges if x not in amb and y not in amb]
print(f'{len(amb):,} ambiguous acronym keywords isolated')
par = {}
def find(x):
    par.setdefault(x, x)
    while par[x] != x: par[x] = par[par[x]]; x = par[x]
    return x
for x, y in edges: par[find(x)] = find(y)
comp = collections.defaultdict(set); adj = collections.defaultdict(set)
for x, y in edges: comp[find(x)] |= {x, y}; adj[x].add(y); adj[y].add(x)
# keep-apart = pairs whose FINAL verdict says different; sources in increasing authority (later overrides earlier): round-1/2 name judges, v2 name judges,
# context checks, v2 context re-check. (2026-09-29: using any 'no' from any round undid good later merges such as telemedicine / telehealth.)
verdict = {}
def _put(a_, b_, v): verdict[frozenset((a_, b_))] = v
for p in ['data/judged.jsonl', 'data/judged_f.jsonl', 'data/ukus_judged.jsonl', 'data/variants_judged.jsonl', 'data/judged_acr.jsonl', 'data/judged_acr_small.jsonl',
          'data/judged_v2_opus.jsonl', 'data/judged_v2_tail.jsonl', 'data/judged_alt.jsonl', 'data/judged_twins.jsonl', 'data/judged_wordorder.jsonl', 'data/judged_alt2.jsonl',
          'data/judged_precision.jsonl', 'data/judged_precision2.jsonl']:   # precision audits judged the final (kid, heading) pairs directly
    for r in rd(p):
        if r.get('merge') is not None: _put(r['a'], r['b'], bool(r['merge']))
for p in ['data/hi_verified.jsonl', 'data/hi_verified_f.jsonl', 'data/hi_verified_v3.jsonl', 'data/hi_verified_v4.jsonl', 'data/hi_reverified.jsonl']:
    for r in rd(p):
        if r.get('verified') is not None: _put(r['kid'], r['canon'], bool(r['verified']))
apart = {k for k, v in verdict.items() if v is False}
m = {}; dropped = 0; big = 0; blocked = 0
for members in comp.values():
    if len(members) <= 3: hub = max(members, key=lambda k: (keep[k], wc[k])); group = members
    else:
        big += 1; hub = max(members, key=lambda k: wc[k]); group = {hub}; frontier = {hub}
        for _ in range(a.hops):
            frontier = {n for f in frontier for n in adj[f]} - group; group |= frontier
        dropped += len(members) - len(group)
    for k in group:
        if k == hub: continue
        if frozenset((k, hub)) in apart: blocked += 1; continue
        m[k] = hub
with open(f'{HERE}/{a.out}', 'w') as f:
    for k, c in m.items(): f.write(json.dumps({'kid': k, 'canon': c}, ensure_ascii=False) + '\n')
print(f'{len(edges):,} edges -> {len(m):,} kids merged into {len(set(m.values())):,} headings; {big:,} components > 3; chaining guard left {dropped:,} keywords unmerged; {blocked:,} blocked by keep-apart verdicts')
