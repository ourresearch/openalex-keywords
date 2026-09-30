"""Confidence cut analysis on the 10K eval (#1322 Next Action 2). Input: results/conf_eval.jsonl (modal_confidence.py), the Opus gold (results/god_god55.jsonl),
Opus's accept/wrong verdicts on the 4B student's keywords for 1,000 works (results/judge_qwen4ball8.jsonl), corpus frequencies of the eval keywords (data/conf_kw_freq.json, pulled here once).
Per keyword: token log-probs (mean, min, first, sum) and the probability that the list continues after it. Reports AUC of each signal vs Opus's verdict, then, per cut,
keywords per work (mean, median, distribution), exact P/R/F1 vs Opus, and judged precision, with and without the 100-work floor; rank cuts at equal mean count as the baseline.
python conf_analyze.py"""
import json, sys, os, collections, numpy as np
sys.path.insert(0, '../bench'); from common import canon
R = [json.loads(l) for l in open('results/conf_eval.jsonl')]
G = {}
for l in open('results/god_god55.jsonl'):
    r = json.loads(l)
    if 'answer' in r: G[r['id']] = {canon(k) for k in r['answer']['keywords'] if canon(k)}
J = {r['id']: r for r in map(json.loads, open('results/judge_qwen4ball8.jsonl'))}
ids_eval = set(open('data/god_eval_ids.txt').read().split())

def split_keywords(toks):
    """-> list of (keyword text, [token lps], P(list stops right after it))"""
    out, cur, lps = [], '', []
    for i, t in enumerate(toks):
        if ';' in t['t'] or '\n' in t['t'] or t['t'] in ('<|endoftext|>', '<|im_end|>'):
            pre = '' if t['t'] in ('<|endoftext|>', '<|im_end|>') else t['t'].split(';')[0].split('\n')[0]
            if pre.strip(): cur += pre; lps.append(t['lp'])
            stop = sum(np.exp(v) for k, v in t['top'].items() if '\n' in k or k in ('<|endoftext|>', '<|im_end|>'))
            if cur.strip(): out.append((cur.strip(), lps, float(stop)))
            cur, lps = t['t'].split(';', 1)[1] if ';' in t['t'] else '', []
            if ';' not in t['t']: break
        else:
            cur += t['t']; lps.append(t['lp'])
    if cur.strip(): out.append((cur.strip(), lps, 1.0))
    return out

W = []   # per work: list of dicts
for r in R:
    ks = split_keywords(r['tokens'])[:10]; rows = []
    for i, (k, lps, stop) in enumerate(ks):
        lps = lps or [0.0]
        rows.append({'k': k, 'c': canon(k), 'rank': i, 'n': len(ks), 'mean': float(np.mean(lps)), 'min': float(min(lps)), 'first': lps[0], 'sum': float(sum(lps)),
                     'p_seq': float(np.exp(sum(lps))), 'p_first': float(np.exp(lps[0])), 'p_cont': 1 - stop})
    # probability the model reaches keyword i at all = product of continue-probs of earlier keywords
    reach = 1.0
    for x in rows: x['reach'] = reach; reach *= x['p_cont']
    W.append({'id': r['id'], 'kws': rows, 'text': r['text'], 'prefill': r.get('prefill_lp', [])})

# prefill vs generation log-prob agreement
d = [abs(a['lp'] - b) for r in R for a, b in zip(r['tokens'], r.get('prefill_lp', []))]
if d: print(f"prefill vs generate token log-probs: {len(d):,} tokens, mean |diff| {np.mean(d):.4f}, p99 {np.percentile(d, 99):.4f}, max {max(d):.3f}")

freq_path = 'data/conf_kw_freq.json'
if not os.path.exists(freq_path):
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'common')); import dbx
    ks = sorted({x['k'].lower().strip() for w in W for x in w['kws']})
    dbx.sql("CREATE OR REPLACE TABLE openalex.transient.kw1322_conf_eval_kws (kc STRING)", verbose=False)
    for i in range(0, len(ks), 5000):
        dbx.sql("INSERT INTO openalex.transient.kw1322_conf_eval_kws VALUES " + ",".join("('" + k.replace("\\", "\\\\").replace("'", "''") + "')" for k in ks[i:i + 5000]), verbose=False)
    rows = dbx.sql("""SELECT e.kc, count(DISTINCT s.id) AS n FROM openalex.transient.kw1322_conf_eval_kws e
                      JOIN (SELECT id, lower(trim(k)) AS kc FROM openalex.transient.keywords_student_v1 LATERAL VIEW explode(keywords) x AS k) s ON s.kc = e.kc GROUP BY e.kc""", timeout_s=3600, verbose=False)
    json.dump({r['kc']: int(r['n']) for r in rows}, open(freq_path, 'w'))
F = json.load(open(freq_path))
for w in W:
    for x in w['kws']: x['freq'] = F.get(x['k'].lower().strip(), 0)

def auc(pos, neg):
    s = np.array(pos + neg); y = np.array([1] * len(pos) + [0] * len(neg)); o = s.argsort(); rk = np.empty(len(s)); rk[o] = np.arange(1, len(s) + 1)
    return (rk[y == 1].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
JV = {}   # (id, canon keyword) -> 1 accepted / 0 wrong, from Opus's verdicts on the same student's keywords
for j in J.values():
    for i, k in enumerate(j['keywords']):
        if i + 1 in j['accepted']: JV[(j['id'], canon(k))] = 1
        elif i + 1 in j['wrong']: JV[(j['id'], canon(k))] = 0
lab = [(x, JV[(w['id'], x['c'])]) for w in W for x in w['kws'] if (w['id'], x['c']) in JV]
print(f"\njudged keywords matched to this regeneration: {len(lab):,} ({sum(y for _, y in lab):,} accepted)")
feats = ['mean', 'min', 'first', 'sum', 'p_cont', 'reach', 'rank_score', 'log_freq']
for x, _ in lab: x['rank_score'] = -x['rank']; x['log_freq'] = np.log1p(x['freq'])
for x in (x for w in W for x in w['kws']): x['rank_score'] = -x['rank']; x['log_freq'] = np.log1p(x['freq'])
print("AUC vs Opus accept/wrong:", {f: round(auc([x[f] for x, y in lab if y], [x[f] for x, y in lab if not y]), 3) for f in feats})
# exact-match labels on the full eval (keyword in Opus's own list)
ex = [(x, int(x['c'] in G[w['id']])) for w in W if w['id'] in G and w['id'] in ids_eval for x in w['kws']]
print("AUC vs 'in Opus's list' (10K):", {f: round(auc([x[f] for x, y in ex if y], [x[f] for x, y in ex if not y]), 3) for f in feats})

def evaluate(keep, name):
    n, P, Rc, Fs, jacc, jn = [], [], [], [], 0, 0
    for w in W:
        kept = [x for x in w['kws'] if keep(x)]; n.append(len(kept))
        if w['id'] in G and w['id'] in ids_eval and G[w['id']]:
            p = {x['c'] for x in kept}; g = G[w['id']]; tp = len(p & g); pr = tp / max(1, len(p)); rc = tp / len(g); P.append(pr); Rc.append(rc); Fs.append(2 * pr * rc / max(1e-9, pr + rc))
        for x in kept:
            v = JV.get((w['id'], x['c']))
            if v is not None: jacc += v; jn += 1
    n = np.array(n); h = np.bincount(np.minimum(n, 10), minlength=11) / len(n)
    print(f"{name:34s} mean {n.mean():.2f} med {np.median(n):.0f} p10 {np.percentile(n,10):.0f} p90 {np.percentile(n,90):.0f} | P {np.mean(P):.3f} R {np.mean(Rc):.3f} F1 {np.mean(Fs):.3f} | judged P {jacc/max(1,jn):.3f} | "
          + " ".join(f"{k}:{h[k]*100:.0f}" for k in range(11)))
print("\n=== no floor ===")
evaluate(lambda x: True, 'all')
for N in (5, 6, 7): evaluate(lambda x, N=N: x['rank'] < N, f'top {N}')
for f, ts in (('reach', (0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3)), ('mean', (-0.3, -0.5, -0.7, -1.0, -1.3)), ('first', (-0.5, -1, -1.5, -2, -3))):
    for t in ts: evaluate(lambda x, f=f, t=t: x[f] >= t, f'{f} >= {t}')
print("\n=== floor 100 works (corpus frequency, lower/trim, 282M loaded) ===")
evaluate(lambda x: x['freq'] >= 100, 'floor only')
for f, ts in (('reach', (0.9, 0.8, 0.7, 0.6, 0.5, 0.4)), ('mean', (-0.5, -0.7, -1.0, -1.3)), ('first', (-1, -1.5, -2, -3))):
    for t in ts: evaluate(lambda x, f=f, t=t: x['freq'] >= 100 and x[f] >= t, f'floor + {f} >= {t}')
for N in (5, 6): evaluate(lambda x, N=N: x['freq'] >= 100 and x['rank'] < N, f'floor + top {N}')

# works with an abstract only (the fair comparison with authors' counts, which come from works that have abstracts)
ABS = {json.loads(l)['id'] for l in open('data/student_eval.jsonl') if 'Abstract: (none)' not in json.loads(l)['prompt']}
print(f"\n=== works with an abstract ({len(ABS):,}) ===")
W_all = W; W = [w for w in W_all if w['id'] in ABS]
evaluate(lambda x: True, 'all')
evaluate(lambda x: x['freq'] >= 100, 'floor only')
for t in (-0.7, -1.0, -1.3): evaluate(lambda x, t=t: x['freq'] >= 100 and x['mean'] >= t, f'floor + mean >= {t}')
W = W_all
# what the cut removes on top of the floor: judged precision of the removed keywords
for t in (-0.7, -1.0, -1.3):
    rem = [JV[(w['id'], x['c'])] for w in W for x in w['kws'] if x['freq'] >= 100 and x['mean'] < t and (w['id'], x['c']) in JV]
    kept = [JV[(w['id'], x['c'])] for w in W for x in w['kws'] if x['freq'] >= 100 and x['mean'] >= t and (w['id'], x['c']) in JV]
    print(f"floor + mean >= {t}: removes {len(rem)} judged keywords, {np.mean(rem):.3f} of them accepted by Opus; kept {len(kept)}, {np.mean(kept):.3f} accepted")
