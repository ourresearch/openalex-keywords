"""Score the pilot arms -> data/scores.json + markdown tables on stdout.
Per arm x group: keywords per work (mean, median, % empty, % >= 10); judged precision; good (accepted) keywords per work; pooled recall = accepted from this arm /
accepted across all arms (by kid0, per work, micro); keywords on records the judge calls insufficient; mechanical bad rates per 100 keywords from the vocab-build
lists (hard purge = document-type labels; dual-use redundant with the work's type / study design; country with no evidence in title/abstract or author
institutions); non-English keyword rate; author-keyword overlap (author_kw group: exact kid0 and token-overlap soft match).
uv run --with regex python score.py"""
import ast, json, os, statistics, sys, collections
import regex
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'common'))
from kid import kid0
ARMS = sys.argv[1].split(',') if len(sys.argv) > 1 else ['v2', 'A', 'B', 'C', 'D', 'E']
works = {json.loads(l)['id']: json.loads(l) for l in open(f'{HERE}/data/works.jsonl')}
meta = json.load(open(f'{HERE}/data/meta.json'))
judged = {r['id']: r for r in map(json.loads, open(f'{HERE}/data/judged.jsonl')) if 'verdict' in r} if os.path.exists(f'{HERE}/data/judged.jsonl') else {}
def variants(k):
    out = {k}
    if k.endswith('ies'): out.add(k[:-3] + 'y')
    if regex.search(r'(ss|sh|ch|x|z)es$', k): out.add(k[:-2])
    if k.endswith('s'): out.add(k[:-1])
    return out
purge = collections.defaultdict(list)
for p in json.load(open(f'{HERE}/data/purge.json')):
    p['dual_use'] = p['dual_use'] in (True, 'true')
    for key in {p['kid'], kid0(p['keyword'])}:
        if key: purge[key].append(p)
def java_rx(rx):
    rx = regex.sub(r'\\Q(.*?)\\E', lambda m: regex.escape(m.group(1)), rx); return regex.compile(rx.replace('(?iu)', '(?i)'))
country = collections.defaultdict(list)
for c in json.load(open(f'{HERE}/data/country_kids.json')): country[c['kid']].append((c['cc'], java_rx(c['rx'])))
def fate(k, w):
    ks = variants(kid0(k) or '')
    ps = [p for x in ks for p in purge.get(x, [])]
    if any(not p['dual_use'] for p in ps): return 'purge'
    m = meta.get(w['id'], {}); sd = m.get('study_designs') or []
    if isinstance(sd, str): sd = json.loads(sd)
    if any((p['kind'] == 'work_type' and p['maps_to'] == m.get('type')) or (p['kind'] == 'study_design' and p['maps_to'] in sd) for p in ps): return 'dual_redundant'
    cs = [c for x in ks for c in country.get(x, [])]
    if cs:
        cc = m.get('inst_cc') or []
        if isinstance(cc, str): cc = json.loads(cc)
        text = f"{w.get('title') or ''} {w.get('abstract') or ''}"
        if not any(c in cc or rx.search(text) for c, rx in cs): return 'country_no_evidence'
    return 'keep'
nonlatin = lambda k: any(ch.isalpha() and ord(ch) > 0x24F and not (0x370 <= ord(ch) <= 0x3FF) for ch in k)
def toks(k): return set((kid0(k) or '').split('-')) - {'', 'of', 'and', 'the', 'in', 'for'}
def arm_rows(arm):
    rows = {}
    for l in open(f'{HERE}/data/arm_{arm}.jsonl'):
        r = json.loads(l)
        if 'keywords' in r: rows[r['id']] = r['keywords']
    return rows
A = {arm: arm_rows(arm) for arm in ARMS}
common = [i for i in works if all(i in A[arm] for arm in ARMS)]
groups = ['abstract', 'short_abstract', 'title_only', 'non_english', 'author_kw', 'xpac', 'all']
def good_union(i):
    v = judged.get(i, {}).get('verdict', {}); return {k for arm in ARMS for kw in A[arm][i] for k in [kid0(kw)] if k and v.get(k)}
S = {}
for arm in ARMS:
    for g in groups:
        ids = [i for i in common if g == 'all' or works[i]['group'] == g]
        if not ids: continue
        n = [len(A[arm][i]) for i in ids]; ks = [(i, k) for i in ids for k in A[arm][i]]
        fates = collections.Counter(fate(k, works[i]) for i, k in ks)
        jd = [(i, k) for i, k in ks if i in judged and kid0(k) in judged[i].get('verdict', {})]
        acc = sum(judged[i]['verdict'][kid0(k)] for i, k in jd)
        gu = sum(len(good_union(i)) for i in ids if i in judged)
        good_arm = sum(len({kid0(k) for k in A[arm][i] if judged.get(i, {}).get('verdict', {}).get(kid0(k))}) for i in ids)
        insuff = [len(A[arm][i]) for i in ids if judged.get(i, {}).get('sufficient') is False]
        s = {'works': len(ids), 'kw_mean': round(statistics.mean(n), 2), 'kw_median': statistics.median(n), 'empty': round(sum(x == 0 for x in n) / len(n), 3),
             'ge10': round(sum(x >= 10 for x in n) / len(n), 3), 'judged_precision': round(acc / len(jd), 3) if jd else None, 'good_per_work': round(good_arm / len(ids), 2),
             'pooled_recall': round(good_arm / gu, 3) if gu else None, 'insufficient_works': len(insuff), 'kw_on_insufficient': round(statistics.mean(insuff), 2) if insuff else None,
             'per100_purge': round(100 * fates['purge'] / max(1, len(ks)), 2), 'per100_dual_redundant': round(100 * fates['dual_redundant'] / max(1, len(ks)), 2),
             'per100_country_no_evidence': round(100 * fates['country_no_evidence'] / max(1, len(ks)), 2),
             'works_with_flag': round(sum(any(fate(k, works[i]) != 'keep' for k in A[arm][i]) for i in ids) / len(ids), 3),
             'per100_nonlatin': round(100 * sum(nonlatin(k) for _, k in ks) / max(1, len(ks)), 2)}
        if g == 'author_kw':
            tp = sp = tr = sr = na = nk = 0
            for i in ids:
                akl = works[i]['author_kw']; akl = ast.literal_eval(akl) if isinstance(akl, str) else akl
                ak = [x for x in akl if kid0(x)]; mine = A[arm][i]
                ak0 = {kid0(x) for x in ak}; my0 = {kid0(x) for x in mine}
                tr += len(ak0 & my0); tp += len(my0 & ak0); na += len(ak0); nk += len(my0)
                soft = lambda x, ys: any(len(toks(x) & toks(y)) / max(1, len(toks(x) | toks(y))) >= 0.5 for y in ys)
                sr += sum(soft(x, mine) for x in ak); sp += sum(soft(y, ak) for y in mine)
            s.update(akw_exact_recall=round(tr / na, 3), akw_exact_precision=round(tp / nk, 3), akw_soft_recall=round(sr / na, 3), akw_soft_precision=round(sp / max(1, sum(len(A[arm][i]) for i in ids)), 3))
        S[f'{arm}|{g}'] = s
    # precision by rank (is the list ordered by importance?)
    for g in ('abstract', 'title_only', 'xpac'):
        pr = collections.defaultdict(lambda: [0, 0])
        for i in common:
            if works[i]['group'] != g or i not in judged: continue
            for r, k in enumerate(A[arm][i][:12]):
                v = judged[i]['verdict'].get(kid0(k))
                if v is not None: pr[r + 1][0] += v; pr[r + 1][1] += 1
        S[f'{arm}|rank|{g}'] = {r: round(a / b, 2) for r, (a, b) in sorted(pr.items()) if b >= 10}
json.dump(S, open(f'{HERE}/data/scores.json', 'w'), indent=1)
print(f'{len(common)} works in every arm; judged {len(judged)}\n')
cols = [('kw_mean', 'kw/work'), ('empty', 'empty'), ('judged_precision', 'judged P'), ('good_per_work', 'good/work'), ('pooled_recall', 'pooled R'),
        ('per100_purge', 'doc-type /100'), ('per100_country_no_evidence', 'country-no-evid /100'), ('kw_on_insufficient', 'kw on insufficient')]
for g in groups:
    print(f'### {g} ({S[f"{ARMS[0]}|{g}"]["works"]} works)\n| arm | ' + ' | '.join(c[1] for c in cols) + ' |\n|' + '---|' * (len(cols) + 1))
    for arm in ARMS: print(f'| {arm} | ' + ' | '.join(str(S[f'{arm}|{g}'].get(c[0])) for c in cols) + ' |')
    print()
print('### author keywords (author_kw group)\n| arm | exact R | exact P | soft R | soft P |\n|---|---|---|---|---|')
for arm in ARMS: s = S[f'{arm}|author_kw']; print(f"| {arm} | {s['akw_exact_recall']} | {s['akw_exact_precision']} | {s['akw_soft_recall']} | {s['akw_soft_precision']} |")
print('\n### judged precision by rank')
for g in ('abstract', 'title_only', 'xpac'):
    for arm in ARMS: print(f'{g:10s} {arm:3s} ' + ' '.join(f'{v:.2f}' for v in S[f'{arm}|rank|{g}'].values()))
print('\nnon-Latin keywords per 100 (non_english group): ' + ', '.join(f"{arm} {S[f'{arm}|non_english']['per100_nonlatin']}" for arm in ARMS))
