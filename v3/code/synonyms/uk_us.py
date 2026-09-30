"""UK -> US spelling-variant candidates over the final vocabulary (#1322, round-2 review 2026-09-28: 416 UK/US pairs still split; embedding kNN missed
them in dense regions). Word-level rules; a candidate counts only if the US form is itself a vocabulary keyword. -> data/ukus_cand.json [[uk_kid, us_kid]].
uv run --with httpx python uk_us.py"""
import json, re, sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'common')); import dbxr
WORD = {'aluminium': 'aluminum', 'grey': 'gray', 'programme': 'program', 'programmes': 'programs', 'defence': 'defense', 'licence': 'license', 'offence': 'offense',
        'ageing': 'aging', 'sulphur': 'sulfur', 'sulphate': 'sulfate', 'sulphide': 'sulfide', 'mould': 'mold', 'moulds': 'molds', 'plough': 'plow', 'tyre': 'tire', 'tyres': 'tires',
        'kerb': 'curb', 'draught': 'draft', 'storey': 'story', 'manoeuvre': 'maneuver', 'jewellery': 'jewelry', 'cheque': 'check', 'judgement': 'judgment', 'acknowledgement': 'acknowledgment',
        'enrolment': 'enrollment', 'fulfil': 'fulfill', 'skilful': 'skillful', 'foetus': 'fetus', 'foetal': 'fetal', 'diarrhoea': 'diarrhea', 'gonorrhoea': 'gonorrhea', 'oedema': 'edema',
        'oestrogen': 'estrogen', 'coeliac': 'celiac', 'faeces': 'feces', 'faecal': 'fecal', 'caesarean': 'cesarean', 'anaemia': 'anemia', 'leukaemia': 'leukemia', 'ischaemia': 'ischemia',
        'ischaemic': 'ischemic', 'septicaemia': 'septicemia', 'haemorrhage': 'hemorrhage', 'orthopaedic': 'orthopedic', 'orthopaedics': 'orthopedics', 'encyclopaedia': 'encyclopedia',
        'archaeology': 'archeology', 'mediaeval': 'medieval', 'palaeontology': 'paleontology', 'palaeoclimate': 'paleoclimate', 'catalogue': 'catalog', 'analogue': 'analog', 'dialogue': 'dialog',
        'centre': 'center', 'centres': 'centers', 'fibre': 'fiber', 'fibres': 'fibers', 'litre': 'liter', 'metre': 'meter', 'theatre': 'theater', 'calibre': 'caliber', 'spectre': 'specter'}
def us_word(w):
    if w in WORD: return WORD[w]
    for a, b in (('haem', 'hem'), ('paed', 'ped'), ('anaes', 'anes'), ('oesophag', 'esophag'), ('oestr', 'estr'), ('palaeo', 'paleo'), ('haemat', 'hemat')):
        if w.startswith(a): return b + w[len(a):]
    for a, b in (('isation', 'ization'), ('isations', 'izations'), ('ising', 'izing'), ('ised', 'ized'), ('iser', 'izer'), ('isers', 'izers'), ('ise', 'ize'), ('ises', 'izes'),
                 ('yse', 'yze'), ('ysed', 'yzed'), ('ysing', 'yzing'), ('elling', 'eling'), ('elled', 'eled'), ('eller', 'eler'), ('our', 'or'), ('ours', 'ors'), ('ourable', 'orable'), ('ourite', 'orite')):
        if w.endswith(a) and len(w) > len(a) + 2: return w[:-len(a)] + b
    return w
if __name__ == '__main__':
    rows = dbxr.sql("SELECT kid, display_name, works_count FROM openalex.transient.kw1322v3f_vocab", warehouse=os.environ.get('DATABRICKS_WAREHOUSE_ID'))
    kids = {r['kid']: int(r['works_count']) for r in rows}
    cand = []
    for k in kids:
        parts = k.split('-'); us = '-'.join(us_word(p) for p in parts)
        if us != k and us in kids: cand.append([k, us])
    json.dump(cand, open(f'{HERE}/data/ukus_cand.json', 'w'))
    print(len(cand), 'candidates; works on the UK side', sum(kids[a] for a, _ in cand))
    for a, b in sorted(cand, key=lambda p: -kids[p[0]])[:25]: print(f'  {a} ({kids[a]}) -> {b} ({kids[b]})')
