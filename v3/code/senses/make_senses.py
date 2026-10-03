"""Homonym senses (#1465, 2026-09-30): turn the judge's verdicts (data/homonym_judged.jsonl) into a sense table.
Rules: the largest sense keeps the keyword's id and name (most works don't change). Each other sense becomes a heading: the existing heading
if the sense label's kid0 is already in the vocabulary ('collective memory'), else a new id keywords/<kid0(label)> with the label as display
name ('inflation (cosmology)'). A work moves to a minority sense only when its primary-topic field is listed for that sense and for no other
sense; anything else stays with the main sense. A minority sense needs >= 3% of the keyword's works and >= 100 works.
Skipped: keywords whose main sense isn't the keyword itself; labels with ':' or > 50 chars. Excluded: keywords merged by hand in the synonym review that the judge splits along the merge line (tuberculosis: consumption), which go back to that reviewer instead.
-> data/senses.jsonl rows {heading, field, sense_id, sense_name, sense_new} + data/senses_summary.json
(Release note: the field rule itself failed its check and never shipped; senses_summary.json, the list of 115 keywords, is what the per-work
classifier used.) python make_senses.py"""
import os, json, sys, collections
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'common')); from kid import kid0
from dbx import sql
EXCLUDE = {'tuberculosis'}
J = [json.loads(l) for l in open('data/homonym_judged.jsonl')]
hk = lambda r: r['kid'].rsplit('/', 1)[-1]   # judged ids are full URLs
D = [r for r in J if r['verdict'] == 'distinct_senses' and hk(r) not in EXCLUDE and len(r['senses']) >= 2]
fields = collections.defaultdict(dict)
for r in sql("SELECT kid, field, n FROM openalex.transient.kw1465_kw_field WHERE kid IN (" + ",".join("'" + r['kid'] + "'" for r in D) + ")", verbose=False):
    fields[r['kid']][r['field']] = int(r['n'])
labels = {kid0(s['label']) for r in D for s in r['senses']}
vocab = {r['kid']: r['display_name'] for r in sql("SELECT kid, display_name FROM openalex.common.keywords_v2 WHERE kid IN (" + ",".join("'" + k.replace("'", "\\'") + "'" for k in labels if k) + ")", verbose=False)}
rows, summ = [], []
for r in D:
    kid = hk(r); fs = fields[r['kid']]; tot = sum(fs.values())
    senses = sorted(r['senses'], key=lambda s: -s['share']); main = senses[0]
    # the main sense must BE the keyword ('inflation (economics)' for inflation), else the bare id would end up meaning something else ('memory' -> collective memory)
    mk = kid0(main['label']) or ''
    if not (mk == kid or mk.startswith(kid + '-') or kid0(main['label'].split(' (')[0]) == kid): continue
    owner = collections.defaultdict(list)
    for i, s in enumerate(senses):
        for f in s['fields']:
            if f in fs: owner[f].append(i)
    moved = collections.Counter()
    for f, idx in owner.items():
        if len(idx) == 1 and idx[0] != 0: moved[idx[0]] += fs[f]
    kept = []
    for i, s in enumerate(senses[1:], 1):
        n = moved[i]
        if n < 100 or n < 0.03 * tot: continue
        k = kid0(s['label'])
        if not k or k == kid or ':' in s['label'] or len(s['label']) > 50: continue
        new = k not in vocab
        name = s['label'] if new else vocab[k]
        kept.append((i, k, name, new, n))
        for f, idx in owner.items():
            if idx == [i]: rows.append({'heading': kid, 'field': f, 'sense_id': k, 'sense_name': name, 'sense_new': new, 'works': fs[f]})
    if kept: summ.append({'heading': kid, 'works': tot, 'main': main['label'], 'split': [{'id': k, 'name': nm, 'new': nw, 'works': n} for _, k, nm, nw, n in kept]})
with open('data/senses.jsonl', 'w') as f:
    for x in rows: f.write(json.dumps(x, ensure_ascii=False) + '\n')
json.dump(summ, open('data/senses_summary.json', 'w'), indent=1, ensure_ascii=False)
print(f"{len(J)} judged, {len(D)} with distinct senses; {len(summ)} keywords split into {sum(len(s['split']) for s in summ)} minority senses "
      f"({sum(x['new'] for s in summ for x in s['split'])} new headings); {len(rows)} (heading, field) rules; ≈ {sum(x['works'] for x in rows):,} assignments move")
