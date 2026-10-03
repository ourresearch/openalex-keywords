"""#1476 rung-3: Opus vs GPT-6 Astra (different family + prompt) vs Jev on the same random works. Agreement overall and on 'moves'
(either judge says a minority sense).
python compare3.py"""
import json, collections
op = {(r['kid'], str(r['work_id'])): r['choice'] for r in map(json.loads, open('data/gold_labels.jsonl')) if r.get('choice') is not None}
n_opts = {(r['kid'], str(r['work_id'])): len(r['opts']) for r in map(json.loads, open('data/gold_labels.jsonl'))}
asr = {(r['kid'], str(r['work_id'])): r['choice'] for r in map(json.loads, open('data/astra_labels.jsonl')) if r.get('choice') is not None}
jv = {}
for r in map(json.loads, open('data/jev_v1.jsonl')):
    if r['ok']:
        p = r['answers']['sense']['probabilities']; n = len([k for k in p if k != 'none'])
        pr = [p.get(f's{i}', 0) for i in range(n)] + [p.get('none', 0)]; jv[(r['kid'], str(r['work_id']))] = max(range(len(pr)), key=lambda i: pr[i])
K = [k for k in asr if k in op and k in jv]
agree = lambda a, b: sum(a[k] == b[k] for k in K) / len(K)
print(len(K), 'works with all three; agreement opus-astra', round(agree(op, asr), 3), 'opus-jev', round(agree(op, jv), 3), 'astra-jev', round(agree(asr, jv), 3))
minor = lambda c, k: 0 < c < n_opts[k] - 1
mv = [k for k in K if minor(jv[k], k)]
print('Jev moves:', len(mv), '| right by Opus', sum(op[k] == jv[k] for k in mv), '| right by Astra', sum(asr[k] == jv[k] for k in mv), '| right by both', sum(op[k] == jv[k] == asr[k] for k in mv))
main_both = [k for k in K if op[k] == 0 and asr[k] == 0]
print('works both judges call main:', len(main_both), '| Jev moves them:', sum(minor(jv[k], k) for k in main_both))
dis = collections.Counter((op[k], asr[k]) for k in K if op[k] != asr[k]); print('opus/astra disagreements (opus, astra):', dis.most_common(8))
