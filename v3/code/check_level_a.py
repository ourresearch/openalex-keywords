"""Level A replication check: recompute the headline claims from the released files alone (no GPU, no API calls, standard library only).
Prints each claim with the expected value and what this copy of the repository gives, and exits non-zero if any check fails.
    cd v3 && python3 code/check_level_a.py"""
import collections, gzip, hashlib, json, os, subprocess, sys

V = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
os.chdir(V)
sys.path.insert(0, 'code/common')
from kid import kid0

ok = True


def check(name, got, want):
    global ok
    good = got == want if not isinstance(want, float) else abs(got - want) < 5e-4
    ok &= good
    print(f"{'PASS' if good else 'FAIL'}  {name}: {got}" + ('' if good else f"  (expected {want})"))


def lines(p):
    return sum(1 for _ in (gzip.open(p, 'rt') if p.endswith('.gz') else open(p)))


# 1. Student vs teacher, F1 over kid0 sets on the held-out works
T = {r['id']: r['keywords'] for r in map(json.loads, open('data/teacher/teacher_labels_eval.jsonl')) if r['keywords'] is not None}
S = {r['id']: r['keywords'] for r in map(json.loads, open('data/student/student_v3fix_eval_outputs.jsonl'))}
f = []
for i in S.keys() & T.keys():
    g = {kid0(k) for k in T[i] if kid0(k)}
    p = {kid0(k) for k in S[i] if kid0(k)}
    f.append(1.0 if not g and not p else 2 * len(g & p) / (len(g) + len(p)))
check('held-out works scored', len(f), 13338)
check('student vs teacher F1', round(sum(f) / len(f), 4), 0.5684)

# 2. Vocabularies: v3.0 as built, and as served on 2026-10-03
check('v3.0 vocabulary rows', lines('data/vocabulary/kw1322v3n_vocabulary.tsv.gz') - 1, 1874460)
check('2026-10-03 vocabulary rows', lines('data/vocabulary/vocabulary_20261003.tsv.gz') - 1, 1882088)

# 3. Synonym maps: synmap9 (v3.0) and synmap10 rebuilt from the audit verdicts
check('synmap9 rows', lines('data/synonyms/synmap9_final.jsonl.gz'), 362272)
out = subprocess.run([sys.executable, 'code/synonym_audit/make_synmap10.py'], capture_output=True, text=True).stdout
check('synmap10 rebuilt from released verdicts is identical to the served map', 'identical to data/synonyms/synmap10_final.jsonl.gz: True' in out, True)
check('synmap10 merges broken', lines('data/synonyms/synmap10_breaks.jsonl'), 5016)

# 4. Hand decisions
hd = collections.Counter((r['by'], r['d']) for r in map(json.loads, open('data/synonyms/hand_decisions.jsonl')))
check('hand-reviewed pairs', sum(hd.values()), 500)

# 4b. Senses and alternative names as served
import csv
sv = list(csv.DictReader(open('data/senses/senses.tsv'), delimiter='\t'))
check('split senses', sum(r['split'] == 'True' for r in sv), 89)
check('keywords with a split sense', len({r['heading'] for r in sv if r['split'] == 'True'}), 74)
check('per-work moves into split senses', lines('data/senses/work_senses_shipped.tsv.gz') - 1, 1698244)
na = nk = 0
for r in map(json.loads, gzip.open('data/alternatives/display_name_alternatives.jsonl.gz', 'rt')):
    nk += 1
    na += len(r['display_name_alternatives'])
check('keywords with alternative names / alternatives', (nk, na), (1701306, 5091086))

# 5. Outside judge: per-keyword accuracy and head-to-head from the released verdicts
acc = collections.Counter()
pref = collections.Counter()
for x in map(json.loads, open('data/eval/outside_judge/judgments.jsonl')):
    if x.get('result') is None:
        continue
    if x['key'].startswith('A|'):
        for k in x['result']['keywords']:
            acc[(x['list'], k['verdict'])] += 1
    elif x['key'].startswith('P|'):
        v = x['result']['preferred']
        pref['new' if (v == 'A') == x['a_is_new'] and v in 'AB' else ('old' if v in 'AB' else v)] += 1
for lst, want in (('new', 0.911), ('old', 0.435)):
    check(f'outside judge: {lst} keywords accurate', round(acc[(lst, 'accurate')] / (acc[(lst, 'accurate')] + acc[(lst, 'inaccurate')]), 3), want)
check('outside judge: head-to-head new / old / neither', (pref['new'], pref['old'], pref['neither_useful']), (395, 1, 4))
ps = collections.defaultdict(collections.Counter)
for x in map(json.loads, open('data/eval/outside_judge/postswap/judgments.jsonl')):
    if 'verdict' in x:
        ps[x['set']][x['verdict']] += 1
for s, want in (('amr_kw_only', 0.813), ('amr_kw_only_noabs', 0.74)):
    c = ps[s]
    check(f'outside judge: {s} on topic (yes or partly)', round((c['yes'] + c['partly']) / sum(c.values()), 3), want)

print('\nall checks pass' if ok else '\nSOME CHECKS FAILED')
sys.exit(0 if ok else 1)
