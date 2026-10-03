"""synmap10 = synmap9 minus the merges the 2026-09-30 audit judged wrong. This is the synonym map OpenAlex serves since 2026-10-01.
Break list: every merge the string audit (audit_v31/audit_judged.jsonl.gz) or the display-name re-judge (audit_v31/rejudge_judged.jsonl)
called "wrong", except merges decided by hand on 2026-09-29 (synonyms/hand_decisions.jsonl, by = jason). Four of those hand merges were
also judged wrong and are broken too (consumption, brain injury, ultrasound, industries; shipped a few hours later as "synmap10b", folded in
here). A broken member keeps its own id (it becomes its own heading if it has >= 100 works). Long COVID: the three post-COVID-era strings go
to the existing 'post-pandemic-period' heading; 'post-covid' and 'post-covid-19' (both senses, mostly the era) leave long-covid and stand alone.
Reads only released files; writes synmap10.jsonl + synmap10_breaks.jsonl to the output dir (default: a temp dir) and compares them with
data/synonyms/synmap10_final.jsonl.gz.
    python3 make_synmap10.py [out_dir]"""
import gzip, hashlib, json, os, sys, tempfile

SYN = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'synonyms')
AUD = os.path.join(SYN, 'audit_v31')
OUT = sys.argv[1] if len(sys.argv) > 1 else tempfile.mkdtemp(prefix='synmap10_')
os.makedirs(OUT, exist_ok=True)

m = {r['kid']: r['canon'] for r in map(json.loads, gzip.open(os.path.join(SYN, 'synmap9_final.jsonl.gz'), 'rt'))}
hand = {frozenset((r['a'], r['b'])) for r in map(json.loads, open(os.path.join(SYN, 'hand_decisions.jsonl'))) if r['by'] == 'jason' and r['d'] in 'AB'}
hand -= {frozenset(p) for p in [('consumption', 'tuberculosis'), ('brain-injury', 'traumatic-brain-injury'), ('ultrasound', 'ultrasonography'), ('industry', 'manufacturing-industry')]}
breaks, kept_hand = {}, 0
for fn, opener in (('audit_judged.jsonl.gz', gzip.open), ('rejudge_judged.jsonl', open)):
    for r in map(json.loads, opener(os.path.join(AUD, fn), 'rt')):
        if r['v'] != 'wrong' or m.get(r['src']) != r['heading']:
            continue
        if frozenset((r['src'], r['heading'])) in hand:
            kept_hand += 1
            continue
        breaks[r['src']] = {'src': r['src'], 'heading': r['heading'], 'works': r['works'], 'why': r.get('why', ''),
                            'from': 'data/' + fn.replace('.gz', '')}
for k in breaks:
    del m[k]
for k in ('post-covid-19-era', 'post-covid-19-pandemic', 'post-covid-19-period'):
    m[k] = 'post-pandemic-period'
    breaks[k] = {'src': k, 'heading': 'long-covid', 'to': 'post-pandemic-period', 'works': 0, 'why': 'time period, not the illness'}
for k in ('post-covid', 'post-covid-19'):
    m.pop(k, None)
    breaks[k] = {'src': k, 'heading': 'long-covid', 'works': 0, 'why': 'mostly the post-pandemic period (58-72% judged off topic)'}
with open(os.path.join(OUT, 'synmap10.jsonl'), 'w') as f:
    for k, c in m.items():
        f.write(json.dumps({'kid': k, 'canon': c}, ensure_ascii=False) + '\n')
with open(os.path.join(OUT, 'synmap10_breaks.jsonl'), 'w') as f:
    for b in sorted(breaks.values(), key=lambda b: -b['works']):
        f.write(json.dumps(b, ensure_ascii=False) + '\n')
print(f'synmap10: {len(m):,} rows, {len(set(m.values())):,} headings ({len(breaks):,} merges broken or moved; '
      f'{kept_hand} judged-wrong merges kept because they were chosen by hand)')
md5 = lambda p, o=open: hashlib.md5(o(p, 'rb').read()).hexdigest()
same = md5(os.path.join(OUT, 'synmap10.jsonl')) == md5(os.path.join(SYN, 'synmap10_final.jsonl.gz'), gzip.open)
print('identical to data/synonyms/synmap10_final.jsonl.gz:', same, '| output in', OUT)
