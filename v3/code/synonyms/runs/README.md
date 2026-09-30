# Run records for the synonym rounds

These are the command sequences that built the shipped synonym map, kept as a record. Two kinds:

- `*.sh`: the chain scripts as run (`rebuild_g.sh` round 2, `rebuild_h.sh` round 3, `round4.sh`, `round5.sh`, `final_m.sh` synmap8, `final_n.sh` synmap9 and the shipped vocabulary). They expect `KW_SCRATCH` to point at a working directory laid out like the original: `$KW_SCRATCH/synonyms/` (these scripts plus `data/`), `$KW_SCRATCH/vocab/` (`build_vocab.py`, `kid.py`, `data/countries.json`), `$KW_SCRATCH/god/corpus_v3/` (`dbxr.py`) and `$KW_SCRATCH/writepath/` (`gen_build.py`). The dry-run step reads `writepath/build_tables_dryrun.sql`, which `gen_build.py` writes.
- The one-off glue steps below. They were run inline in a shell, so they were not saved as files; they are recovered verbatim from the session log. Paths are relative to `synonyms/` in that layout.

Order of maps: synmap (round 1) → synmap_final → synmap2 (`build_synmap2.py`, round 2) → synmap3 (`build_synmap3.py`) → synmap4 (round4.sh) → synmap5 (round5.sh) → synmap6 → synmap7 → synmap8 (`build_synmap3.py` with the extra arguments shown in docs/pipeline.md § 5) → synmap9 (`apply_hand.py`).

## Round 1: candidate pairs from the bge-m3 kNN pairs (data/pairs_u.json = deduplicated pairs.jsonl from modal_embed_knn.py, K=8, cos >= 0.88)

```bash
cd $KW_SCRATCH/synonyms && python3 - <<'EOF'
import json
wc = {json.loads(l)['kid']: json.loads(l)['works_count'] for l in open('data/vocab.jsonl')}
P = json.load(open('data/pairs_u.json'))
sel = [p for p in P if (p[2] >= 0.90 and min(wc[p[0]], wc[p[1]]) >= 1000) or (p[2] >= 0.93 and min(wc[p[0]], wc[p[1]]) >= 300)]
print(len(sel), 'candidate pairs; works touched by the smaller side (upper bound):', sum(min(wc[a], wc[b]) for a, b, _ in sel))
json.dump(sel, open('data/cand.json', 'w'))
EOF
```

## Round 1: drop merges the context check rejected -> synmap_final.jsonl

```bash
cd $KW_SCRATCH/synonyms && python3 - <<'EOF'
import json
name = {json.loads(l)['kid']: json.loads(l)['display_name'] for l in open('data/vocab.jsonl')}
R = {json.loads(l)['kid']: json.loads(l) for l in open('data/hi_verified.jsonl')}
for k in ('hydrogen-fueling', 'covid-19-pandemic', 'university-students'):
    r = R.get(k); print(k, '->', r and name[r['canon']], r and r['verified'])
m = [json.loads(l) for l in open('data/synmap.jsonl')]
final = [x for x in m if R.get(x['kid'], {}).get('verified', True) is not False]
with open('data/synmap_final.jsonl', 'w') as f:
    for x in final: f.write(json.dumps(x, ensure_ascii=False) + '\n')
print(len(m), '->', len(final), 'merges after dropping rejected high-impact ones')
EOF
```

## Round 2: wider-pass candidates. openalex.transient.kw1322v3f_syncand_co held the K=30, cos >= 0.85 pairs with their co-occurrence counts (co = works carrying both, na / nb = works per side); the SQL that built it was run ad hoc and is not preserved. Also shows the in-place edit of the v1 rubric before judged_f.

```bash
cd $KW_SCRATCH/god/corpus_v3 && uv run -q --with httpx python - <<'EOF'
import sys, json; sys.path.insert(0, '.'); import dbxr
rows = dbxr.sql("""SELECT a, b, cos FROM openalex.transient.kw1322v3f_syncand_co c
                   JOIN openalex.transient.kw1322v3f_vocab va ON va.kid = c.a JOIN openalex.transient.kw1322v3f_vocab vb ON vb.kid = c.b
                   WHERE co / greatest(1, least(na, nb)) < 0.1 ORDER BY least(va.works_count, vb.works_count) * cos DESC LIMIT 200000""", warehouse=__import__('os').environ['DATABRICKS_WAREHOUSE_ID'])
json.dump([[r['a'], r['b'], float(r['cos'])] for r in rows], open('../../synonyms/data/cand_f.json', 'w')); print(len(rows))
EOF
cd ../../synonyms && python3 - <<'EOF'
p='judge_pairs.py'; s=open(p).read()
s=s.replace("ap.add_argument('--per-call', type=int, default=100); a = ap.parse_args()","ap.add_argument('--per-call', type=int, default=100); ap.add_argument('--cand', default='data/cand.json'); ap.add_argument('--vocab', default='data/vocab.jsonl'); ap.add_argument('--out', default='data/judged.jsonl'); a = ap.parse_args()")
s=s.replace("open(f'{HERE}/data/vocab.jsonl')","open(f'{HERE}/{a.vocab}')").replace("json.load(open(f'{HERE}/data/cand.json'))","json.load(open(f'{HERE}/{a.cand}'))").replace("out_path = f'{HERE}/data/judged.jsonl'","out_path = f'{HERE}/{a.out}'")
old_rules = s[s.index('- merge: synonyms'):s.index('When unsure, do not merge.')]
new_rules = """- merge: synonyms and near-synonyms used interchangeably in the literature (college students / university students), spelling or transliteration variants (tumor / tumour, Neanderthals / Neandertals), abbreviation or acronym vs full form (NAFLD / non-alcoholic fatty liver disease, CAR-T cells / chimeric antigen receptor T cells), a Latin taxon and its common name for the same group (Tardigrada / tardigrades, Salmo salar / Atlantic salmon), grammatical variants of one concept (nudging / nudges, climate migration / climate-induced migration), trivial word-order, hyphenation or wording differences (edge plasma diagnostics / plasma edge diagnostics, CAR T-cell therapy / CAR-T cell therapy), a disease and a renamed or equivalent name for the same disease (monkeypox / mpox), full vs short official names (type 2 diabetes / type 2 diabetes mellitus).
- do NOT merge: different things (World War I / World War II, food safety / food security), broader vs narrower (animals / dogs, antimicrobial activity / antibacterial activity, deep learning / deep neural network, monetary policy / monetary policy transmission, Arabidopsis / Arabidopsis thaliana), a quantity vs its measurement or profile or imaging (electron density / electron density measurement), related but distinct concepts (gender differences / sex differences, dairy cows / dairy cattle is fine to merge only if you judge them the same population as used), a disease vs the pandemic or societal event (COVID-19 / COVID-19 pandemic), proper names that must keep their own form (UK Labour Party).
"""
s=s.replace(old_rules, new_rules)
open(p,'w').write(s)
EOF
grep -c "Tardigrada" judge_pairs.py; nohup uv run -q --with anthropic python judge_pairs.py --cand data/cand_f.json --vocab data/vocab_f.jsonl --out data/judged_f.jsonl --workers 16 > data/judge_f.log 2>&1 & disown; echo launched
```

## Round 3: head / tail candidates for the v2 judge (rem_pairs / rej_pairs = pairs from rounds 1-2 left unmerged or rejected, written by build_synmap2.py runs)

```bash
cd $KW_SCRATCH/god/corpus_v3 && uv run -q --with httpx python - <<'EOF'
import sys, json; sys.path.insert(0, '.'); import dbxr
rows = dbxr.sql("SELECT kid, display_name, works_count FROM openalex.transient.kw1322v3g_vocab", warehouse=__import__('os').environ['DATABRICKS_WAREHOUSE_ID'])
with open('../../synonyms/data/vocab_g.jsonl', 'w') as f:
    for r in rows: f.write(json.dumps({'kid': r['kid'], 'display_name': r['display_name'], 'works_count': int(r['works_count'])}, ensure_ascii=False) + '\n')
print(len(rows))
EOF
cd ../../synonyms && python3 - <<'EOF'
import json
g = {json.loads(l)['kid']: json.loads(l)['works_count'] for l in open('data/vocab_g.jsonl')}
rem = json.load(open('data/rem_pairs.json')); rej = json.load(open('data/rej_pairs.json'))
ok = lambda a, b: a in g and b in g
opus = [[a, b, c] for a, b, c, w in rej if c >= 0.9 and ok(a, b) and min(g[a], g[b]) >= 1000] + [[a, b, c] for a, b, c, w in rem if ok(a, b) and min(g[a], g[b]) >= 1000]
son = [[a, b, c] for a, b, c, w in rem if ok(a, b) and 200 <= min(g[a], g[b]) < 1000]
json.dump(opus, open('data/cand_v2_opus.json', 'w')); json.dump(son, open('data/cand_v2_sonnet.json', 'w'))
print('opus pairs', len(opus), 'sonnet pairs', len(son))
EOF
```

## Round 3: tail filter (cos >= 0.88) and launch of the two v2 judge runs

```bash
cd $KW_SCRATCH/synonyms && python3 -c "
import json; son=json.load(open('data/cand_v2_sonnet.json')); t=[p for p in son if p[2]>=0.88]; json.dump(t, open('data/cand_v2_tail.json','w')); print(len(t))"; rm -f data/judged_v2_sonnet.jsonl; nohup uv run -q --with anthropic python judge_pairs2.py --cand data/cand_v2_opus.json --vocab data/vocab_g.jsonl --out data/judged_v2_opus.jsonl --workers 24 > data/judge_v2_opus.log 2>&1 & disown; nohup uv run -q --with anthropic python judge_pairs2.py --cand data/cand_v2_tail.json --vocab data/vocab_g.jsonl --out data/judged_v2_tail.jsonl --workers 40 > data/judge_v2_tail.log 2>&1 & disown; echo launched
```

## Round 3: acronym-like keywords (>= 200 works)

```bash
cd $KW_SCRATCH/synonyms && python3 - <<'EOF'
import json, re
V = [json.loads(l) for l in open('data/vocab_g.jsonl')]
acr = [v for v in V if re.fullmatch(r"[A-Za-z0-9α-ωΑ-Ω\-/+]{2,12}", v['display_name']) and sum(ch.isupper() for ch in v['display_name']) >= 2 and v['works_count'] >= 200]
print(len(acr), 'acronym-like keywords with >= 200 works; examples', [a['display_name'] for a in sorted(acr, key=lambda v: -v['works_count'])[:25]])
json.dump(acr, open('data/acronyms.json', 'w'))
EOF
```

## Round 3: v2 judge confirms acronym merges

```bash
cd $KW_SCRATCH/synonyms && python3 -c "
import json; m=[json.loads(l) for l in open('data/synmap_acronyms.jsonl')]; json.dump([[x['kid'], x['canon'], 1.0] for x in m], open('data/cand_acr.json','w')); print(len(m))"; uv run -q --with anthropic python judge_pairs2.py --cand data/cand_acr.json --vocab data/vocab_g.jsonl --out data/judged_acr.jsonl --workers 16 2>&1 | tail -1; python3 - <<'EOF'
import json
name = {json.loads(l)['kid']: json.loads(l)['display_name'] for l in open('data/vocab_g.jsonl')}
R = [json.loads(l) for l in open('data/judged_acr.jsonl')]
print(len(R), 'confirmed', sum(r['merge'] for r in R))
for r in [r for r in R if not r['merge']][:15]: print('  rejected:', name[r['a']], '|', name[r['b']])
EOF
```

## Round 3: confirmed acronym merges -> synmap_acronyms_ok.jsonl

```bash
cd $KW_SCRATCH/synonyms && python3 -c "
import json; x=[json.loads(l) for l in open('data/judged_acr.jsonl')]
open('data/synmap_acronyms_ok.jsonl','w').writelines(json.dumps({'kid': (r['b'] if r['keep']==r['a'] else r['a']), 'canon': r['keep']})+'\n' for r in x if r['merge'])"; tail -1 data/judge_v2_opus.log; tail -1 data/judge_v2_tail.log
```

## Round 3: context check of high-impact merges (examples pulled from the vocabulary build)

```bash
cd $KW_SCRATCH/synonyms && python3 - <<'EOF'
import json
V = {json.loads(l)['kid']: json.loads(l) for l in open('data/vocab_g.jsonl')}
hi = []
for p in ('data/judged_v2_opus.jsonl', 'data/judged_v2_tail.jsonl', 'data/judged_alt.jsonl'):
    for l in open(p):
        r = json.loads(l)
        if not r['merge']: continue
        other = r['b'] if r['keep'] == r['a'] else r['a']
        if V.get(other, {}).get('works_count', 0) >= 10000: hi.append({'kid': other, 'canon': r['keep']})
for l in open('data/synmap_acronyms_ok.jsonl'):
    r = json.loads(l)
    if V.get(r['kid'], {}).get('works_count', 0) >= 10000: hi.append(r)
seen = set(); hi = [h for h in hi if (h['kid'], h['canon']) not in seen and not seen.add((h['kid'], h['canon']))]
json.dump(hi, open('data/hi_merges_v3.json', 'w')); print(len(hi), 'high-impact round-3 merges')
EOF
sed -e "s#data/hi_merges.json#data/hi_merges_v3.json#; s#data/hi_examples.json#data/hi_examples_v3.json#; s#data/hi_verified.jsonl#data/hi_verified_v3.jsonl#g; s#data/vocab.jsonl#data/vocab_g.jsonl#" verify_hi.py > verify_hi_v3.py && cd ../god/corpus_v3 && uv run -q --with httpx python - <<'EOF'
import json, sys
sys.path.insert(0, '.'); import dbxr
hi = json.load(open('../../synonyms/data/hi_merges_v3.json')); kids = sorted({r['kid'] for r in hi} | {r['canon'] for r in hi})
vals = ",".join("('keywords/" + k.replace("'", "\\'") + "')" for k in kids)
dbxr.sql(f"CREATE OR REPLACE TABLE openalex.transient.kw1322v3g_syncheck_kids AS SELECT * FROM VALUES {vals} AS t(keyword_id)", warehouse=__import__('os').environ['DATABRICKS_WAREHOUSE_ID'])
rows = dbxr.sql("""WITH s AS (SELECT wk.keyword_id, wk.work_id, row_number() OVER (PARTITION BY wk.keyword_id ORDER BY xxhash64(wk.work_id)) r
                   FROM openalex.transient.kw1322v3g_work_keywords wk JOIN openalex.transient.kw1322v3g_syncheck_kids k ON k.keyword_id = wk.keyword_id)
                   SELECT s.keyword_id, w.title, w.primary_location.source.display_name AS venue FROM s JOIN openalex.works.openalex_works w ON w.id = CAST(substr(s.work_id, 2) AS BIGINT) WHERE s.r <= 3""", warehouse=__import__('os').environ['DATABRICKS_WAREHOUSE_ID'])
ex = {}
for r in rows: ex.setdefault(r['keyword_id'][9:], []).append(f"{(r['title'] or '')[:140]} [{(r['venue'] or '')[:40]}]")
json.dump(ex, open('../../synonyms/data/hi_examples_v3.json', 'w'), ensure_ascii=False); print(len(kids), 'kids with examples:', len(ex))
EOF
cd ../../synonyms && uv run -q --with anthropic python verify_hi_v3.py 2>&1 | tail -20
```

## Round 4: twins (K=30, cos >= 0.80 on vocabulary h; small side <= 1/10 of a >= 1,000-work heading)

```bash
cd $KW_SCRATCH/synonyms && python3 - <<'EOF'
import json
V = json.load(open('data/vocab_kw1322v3h_.json'))
with open('data/vocab_h.jsonl', 'w') as f:
    for v in sorted(V, key=lambda v: -v['n']): f.write(json.dumps({'kid': v['kid'], 'display_name': v['name'], 'works_count': v['n']}, ensure_ascii=False) + '\n')
print(len(V))
EOF
modal volume put -f oxjob1322 data/vocab_h.jsonl synonyms/vocab_h.jsonl 2>&1 | tail -1; modal run modal_embed_knn.py --vocab /vol/synonyms/vocab_h.jsonl --out /vol/synonyms/pairs_h.jsonl --min-cos 0.80 --k 30 2>&1 | grep -E "embedded|pairs" | tail -2; modal volume get --force oxjob1322 synonyms/pairs_h.jsonl data/pairs_h.jsonl >/dev/null 2>&1; python3 - <<'EOF'
import json
wc = {json.loads(l)['kid']: json.loads(l)['works_count'] for l in open('data/vocab_h.jsonl')}
P = {}
for l in open('data/pairs_h.jsonl'):
    r = json.loads(l); a, b = r['a'], r['b']
    small, big = (a, b) if wc[a] <= wc[b] else (b, a)
    if wc[big] >= 1000 and wc[big] >= 10 * wc[small]: P[(small, big)] = max(P.get((small, big), 0), r['cos'])
for c in (0.8, 0.82, 0.85, 0.88): print(f'twin candidates cos >= {c}: {sum(1 for v in P.values() if v >= c):,}')
json.dump([[a, b, c] for (a, b), c in P.items()], open('data/cand_twins.json', 'w'))
EOF
```

## Round 4: drop twins already judged; keep cos >= 0.82

```bash
cd $KW_SCRATCH/synonyms && python3 - <<'EOF'
import json
syn = {json.loads(l)['kid']: json.loads(l)['canon'] for l in open('data/synmap3.jsonl')}
m = lambda k: syn.get(k, k)
done = set()
for p in ('data/judged.jsonl', 'data/judged_f.jsonl', 'data/judged_v2_opus.jsonl', 'data/judged_v2_tail.jsonl', 'data/judged_alt.jsonl', 'data/judged_acr.jsonl', 'data/judged_acr_small.jsonl', 'data/variants_judged.jsonl', 'data/ukus_judged.jsonl'):
    try:
        for l in open(p):
            r = json.loads(l); done.add(tuple(sorted((m(r['a']), m(r['b'])))))
    except FileNotFoundError: pass
T = json.load(open('data/cand_twins.json'))
new = [t for t in T if tuple(sorted((t[0], t[1]))) not in done]
for c in (0.8, 0.82, 0.85, 0.88): print(f'new twins cos >= {c}: {sum(1 for t in new if t[2] >= c):,}')
json.dump([t for t in new if t[2] >= 0.82], open('data/cand_twins_new.json', 'w'))
EOF
```

## Round 4: small acronyms (100-199 works) and the tightened acronym bullet in the v2 rubric

```bash
cd $KW_SCRATCH/synonyms && python3 - <<'EOF'
p='judge_pairs2.py'; s=open(p).read()
a="- an acronym or abbreviation and its expansion (IL-6 / interleukin-6, MRI / magnetic resonance imaging, NAFLD / non-alcoholic fatty liver disease),"
assert a in s
s=s.replace(a,"- an acronym or abbreviation and its expansion, ONLY when the acronym unambiguously means that expansion across research (IL-6 / interleukin-6, MRI / magnetic resonance imaging, NAFLD / non-alcoholic fatty liver disease); NOT when it has other common meanings (ECE, PD / Pd, DM, PET, ATM) or is a chemical symbol or formula that also reads as something else,")
open(p,'w').write(s)
p='acronyms.py'; s=open(p).read()
s=s.replace("acr = json.load(open(f'{HERE}/data/acronyms.json'))","acr = json.load(open(f'{HERE}/data/' + (sys.argv[1] if len(sys.argv) > 1 else 'acronyms.json')))")
s=s.replace("open(f'{HERE}/data/acronyms_judged.jsonl', 'w')","open(f'{HERE}/data/acronyms_judged' + ('_small' if len(sys.argv) > 1 else '') + '.jsonl', 'w')")
s=s.replace("open(f'{HERE}/data/synmap_acronyms.jsonl', 'w')","open(f'{HERE}/data/synmap_acronyms' + ('_small' if len(sys.argv) > 1 else '') + '.jsonl', 'w')")
open(p,'w').write(s)
import json, re
V = [json.loads(l) for l in open('data/vocab_g.jsonl')]
small = [v for v in V if re.fullmatch(r"[A-Za-z0-9α-ωΑ-Ω\-/+]{2,12}", v['display_name']) and sum(ch.isupper() for ch in v['display_name']) >= 2 and 100 <= v['works_count'] < 200]
json.dump(small, open('data/acronyms_small.json', 'w')); print(len(small), 'small acronyms (100-199 works)')
EOF
uv run -q --with anthropic python acronyms.py acronyms_small.json 2>&1 | tail -12
```

## Round 5: word-order candidates

```bash
cd $KW_SCRATCH/synonyms && python3 - <<'EOF'
import json, re, collections, sys
sys.path.insert(0, '../vocab')
V = json.load(open('data/vocab_kw1322v3i_.json'))
with open('data/vocab_i.jsonl', 'w') as f:
    for v in sorted(V, key=lambda v: -v['n']): f.write(json.dumps({'kid': v['kid'], 'display_name': v['name'], 'works_count': v['n']}, ensure_ascii=False) + '\n')
W = {v['kid']: v['n'] for v in V}
sys.path.insert(0, '.'); from uk_us import us_word
def sing(w):
    if len(w) <= 3 or w.endswith(('ss', 'us', 'is')): return w
    if w.endswith('ies'): return w[:-3] + 'y'
    if w.endswith('s'): return w[:-1]
    return w
STOP = {'of', 'the', 'and', 'in', 'for', 'to', 'a', 'an', 'on'}
g = collections.defaultdict(list)
for k in W: g[' '.join(sorted(sing(us_word(p)) for p in k.split('-') if p not in STOP))].append(k)
cand = set()
for ks in g.values():
    if len(ks) < 2: continue
    top = max(ks, key=lambda k: W[k])
    for k in ks:
        if k != top: cand.add((k, top))
json.dump([[a, b, 1.0] for a, b in sorted(cand)], open('data/cand_wordorder.json', 'w'))
print(len(V), 'keywords in vocab i;', len(cand), 'word-order candidates')
EOF
```

## Round 5: vocab_all.jsonl (union of intermediate vocabularies) and the first 400-merge precision audit

```bash
cd $KW_SCRATCH/synonyms && uv run -q python - <<'EOF'
import json, random
wc, name = {}, {}
for p in ('data/vocab.jsonl', 'data/vocab_f.jsonl', 'data/vocab_g.jsonl', 'data/vocab_h.jsonl', 'data/vocab_i.jsonl'):
    for l in open(p):
        r = json.loads(l); wc[r['kid']] = max(wc.get(r['kid'], 0), r['works_count']); name.setdefault(r['kid'], r['display_name'])
M = [json.loads(l) for l in open('data/synmap6.jsonl')]
M = [m for m in M if m['kid'] in name and m['canon'] in name]
rng = random.Random(6); pick = []; seen = set()
while len(pick) < 400:
    m = rng.choices(M, weights=[wc.get(x['kid'], 1) for x in M])[0]
    if m['kid'] not in seen: seen.add(m['kid']); pick.append(m)
json.dump([[m['kid'], m['canon'], 1.0] for m in pick], open('data/cand_precision.json', 'w'))
with open('data/vocab_all.jsonl', 'w') as f:
    for k in name: f.write(json.dumps({'kid': k, 'display_name': name[k], 'works_count': wc[k]}, ensure_ascii=False) + '\n')
print(len(pick))
EOF
uv run -q --with anthropic python judge_pairs2.py --cand data/cand_precision.json --vocab data/vocab_all.jsonl --out data/judged_precision.jsonl --workers 8 2>&1 | tail -1; python3 - <<'EOF'
import json
name = {json.loads(l)['kid']: json.loads(l)['display_name'] for l in open('data/vocab_all.jsonl')}
R = [json.loads(l) for l in open('data/judged_precision.jsonl')]
bad = [r for r in R if r['merge'] is False]
print(f'{len(R)} sampled merges (works-weighted): {len(R)-len(bad)} judged same concept, {len(bad)} judged different -> precision {1-len(bad)/len(R):.1%}')
for r in bad[:25]: print('  ', name[r['a']], '->', name[r['b']])
EOF
```

## Round 5: irregular plurals, and adding the precision audits to the keep-apart sources; synmap7

```bash
cd $KW_SCRATCH/synonyms && python3 - <<'EOF'
import json
wc = {}
for l in open('data/vocab_i.jsonl'): r = json.loads(l); wc[r['kid']] = r['works_count']
IRR = {'child': 'children', 'woman': 'women', 'man': 'men', 'mouse': 'mice', 'tooth': 'teeth', 'foot': 'feet', 'goose': 'geese', 'louse': 'lice', 'criterion': 'criteria',
       'phenomenon': 'phenomena', 'bacterium': 'bacteria', 'fungus': 'fungi', 'nucleus': 'nuclei', 'stimulus': 'stimuli', 'alga': 'algae', 'larva': 'larvae', 'analysis': 'analyses',
       'hypothesis': 'hypotheses', 'thesis': 'theses', 'crisis': 'crises', 'diagnosis': 'diagnoses', 'medium': 'media', 'curriculum': 'curricula', 'vertebra': 'vertebrae', 'matrix': 'matrices',
       'vortex': 'vortices', 'apex': 'apices', 'appendix': 'appendices', 'cactus': 'cacti', 'radius': 'radii', 'genus': 'genera', 'corpus': 'corpora', 'datum': 'data'}
edges = []
for k in wc:
    parts = k.split('-'); last = parts[-1]
    if last in IRR:
        pl = '-'.join(parts[:-1] + [IRR[last]])
        if pl in wc: a, b = (k, pl) if wc[k] < wc[pl] else (pl, k); edges.append({'kid': a, 'canon': b})
# datum/data and medium/media are risky ('media' = news media): drop bare ones
edges = [e for e in edges if {e['kid'], e['canon']} not in ({'datum', 'data'}, {'medium', 'media'})]
with open('data/synmap_irregular.jsonl', 'w') as f:
    for e in edges: f.write(json.dumps(e) + '\n')
print(len(edges), 'irregular-plural merges, e.g.', [ (e['kid'], e['canon']) for e in edges[:8]])
p='build_synmap3.py'; s=open(p).read()
s=s.replace("          'data/judged_v2_opus.jsonl', 'data/judged_v2_tail.jsonl', 'data/judged_alt.jsonl', 'data/judged_twins.jsonl', 'data/judged_wordorder.jsonl', 'data/judged_alt2.jsonl']:",
            "          'data/judged_v2_opus.jsonl', 'data/judged_v2_tail.jsonl', 'data/judged_alt.jsonl', 'data/judged_twins.jsonl', 'data/judged_wordorder.jsonl', 'data/judged_alt2.jsonl',\n          'data/judged_precision.jsonl', 'data/judged_precision2.jsonl']:   # precision audits judged the final (kid, heading) pairs directly")
open(p,'w').write(s)
EOF
python3 build_synmap3.py --extra-judged data/judged_twins.jsonl data/judged_wordorder.jsonl data/judged_alt2.jsonl --extra-maps data/synmap_acronyms_small_ok.jsonl data/synmap_irregular.jsonl --rejected data/hi_reverified.jsonl --out data/synmap7.jsonl; python3 -c "
import json; M={json.loads(l)['kid']: json.loads(l)['canon'] for l in open('data/synmap7.jsonl')}
for k in ('tumor-associated-macrophages','tam','child','mouse','telehealth','eeg','sex-differences','dna-damage-response'): print(k,'->',M.get(k))"
```

## Hand review: the 500 contested pairs

```bash
cd $KW_SCRATCH/synonyms && python3 - <<'EOF'
import json, glob
V = {v['kid']: v for v in json.load(open('data/vocab_kw1322v3m_.json'))}
M = {json.loads(l)['kid']: json.loads(l)['canon'] for l in open('data/synmap8.jsonl')}
m = lambda k: M.get(k, k)
pairs = {}
def add(a, b, why):
    a, b = m(a), m(b)
    if a == b or a not in V or b not in V: return
    k = tuple(sorted((a, b)))
    pairs.setdefault(k, set()).add(why)
for p in glob.glob('data/judged*.jsonl'):
    tag = p.split('/')[-1].replace('.jsonl', '')
    for l in open(p):
        r = json.loads(l)
        if r.get('merge') is not None and ('precision' not in tag): add(r['a'], r['b'], ('judge said same' if r['merge'] else 'judge said different') + f' ({tag})')
for p in glob.glob('data/hi_*verified*.jsonl'):
    for l in open(p):
        r = json.loads(l); add(r['kid'], r['canon'], ('context check: same' if r.get('verified') else 'context check: different'))
for f in ('data/audit_m_seed29.json', 'data/audit_kw1322v3m_.json', 'data/audit_l_seed29.json', 'data/audit_kw1322v3l_.json', 'data/audit_kw1322v3k_.json'):
    try: x = json.load(open(f))
    except FileNotFoundError: continue
    name2kid = {v['name']: v['kid'] for v in V.values()}
    for nm, n, others in x['examples']:
        for o in others:
            if nm in name2kid and o in name2kid: add(name2kid[nm], name2kid[o], 'audit: still split')
ranked = sorted(pairs, key=lambda k: -min(V[k[0]]['n'], V[k[1]]['n']))
top = ranked[:500]
print(len(pairs), 'still-split pairs with any verdict;', 'top 500 smaller-side works from', min(V[top[0][0]]['n'], V[top[0][1]]['n']), 'down to', min(V[top[-1][0]]['n'], V[top[-1][1]]['n']))
json.dump([{'a': a, 'b': b, 'why': sorted(pairs[(a, b)])} for a, b in top], open('data/review500.json', 'w'))
for a, b in top[:12]: print('  ', V[a]['name'], V[a]['n'], '|', V[b]['name'], V[b]['n'])
EOF
```

## Hand review: spacing / hyphen variants (judged with judge_pairs2.py -> judged_spacing.jsonl)

```bash
cd $KW_SCRATCH/synonyms && sed -n 25,27p judge_pairs2.py; python3 - <<'EOF'
import json, collections
V=json.load(open('data/vocab_kw1322v3m_.json'))
g=collections.defaultdict(list)
for r in V: g[r['kid'].replace('-','')].append(r)
cand=[]; vj=open('data/vocab_spacing.jsonl','w')
for v in g.values():
    if len(v)<2: continue
    v=sorted(v,key=lambda r:-r['n'])
    for r in v: vj.write(json.dumps({'kid':r['kid'],'display_name':r['name'],'works_count':r['n']},ensure_ascii=False)+'\n')
    for r in v[1:]: cand.append([v[0]['kid'], r['kid']])
json.dump(cand,open('data/cand_spacing.json','w')); print(len(cand))
EOF
```
