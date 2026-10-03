"""#1476: Jev option index -> sense heading, for every keyword in the corpus run (the 115 of #1465 senses_summary.json).
Option order = jev_senses.py / corpus_run.py (senses by share; s0 = main). Option i >= 1 gets a heading when its label makes a usable id
(make_senses.py rules: kid0(label) non-empty, != the keyword, no ':', <= 50 chars); the display name is the vocabulary's when that id
already exists ('collective memory'), else the label. A qualified label whose bare term is an existing heading other than the
keyword uses that heading ('middle age (human life stage)' -> middle-age). Unlike make_senses.py there is no field-share cut: the per-sense gate on corpus
moves decides. -> data/sense_map.json {heading: [{opt, label, sense_id, sense_name, new}]}
python sense_map.py"""
import os, json, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'common')); from kid import kid0
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'common')); from dbx import sql
D = 'data'
J = {r['kid'].rsplit('/', 1)[-1]: r for r in map(json.loads, open(f'{D}/homonym_judged.jsonl'))}
H = [s['heading'] for s in json.load(open(f'{D}/senses_summary.json'))]
labels = {kid0(s['label']) for h in H for s in J[h]['senses']} | {kid0(s['label'].split(' (')[0]) for h in H for s in J[h]['senses']}
vocab = {r['kid']: r['display_name'] for r in sql("SELECT kid, display_name FROM openalex.common.keywords_v2 WHERE kid IN ("
         + ",".join("'" + k.replace("'", "\\'") + "'" for k in labels if k) + ")", verbose=False)}
M = {}
for h in H:
    out = []
    for i, s in enumerate(sorted(J[h]['senses'], key=lambda s: -s['share'])):
        k = kid0(s['label']) if i else None
        if i and (not k or k == h or ':' in s['label'] or len(s['label']) > 50): k = None
        bare = kid0(s['label'].split(' (')[0]) if i else None
        if k and k not in vocab and bare and bare != h and bare in vocab and bare not in H: k = bare   # 'middle age (human life stage)' -> existing 'middle age'
        out.append({'opt': f's{i}', 'label': s['label'], 'sense_id': k, 'sense_name': (vocab.get(k) or s['label']) if k else None, 'new': bool(k) and k not in vocab})
    M[h] = out
json.dump(M, open('data/sense_map.json', 'w'), indent=1, ensure_ascii=False)
n = [x for v in M.values() for x in v[1:]]
print(len(M), 'keywords;', len(n), 'minority options;', sum(bool(x['sense_id']) for x in n), 'with a heading;', sum(x['new'] for x in n), 'new headings')
print('no heading:', [(h, x['label']) for h, v in M.items() for x in v[1:] if not x['sense_id']][:20])
