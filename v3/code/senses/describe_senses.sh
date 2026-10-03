#!/bin/zsh
# #1476: description + Wikidata id for the new sense headings (openalex.transient.kw1476_vocab_new), with oxjob #1463's chain unchanged:
# enwiki/Wikidata candidates -> Jev link (confident p >= 0.7 kept) -> GPT-6 Luna description. Example titles come from the staged moves
# (top 6 by Jev prob from kw1476_work_senses), since the headings carry no production works yet. Output: transient kw1476_desc_new, same columns
# as openalex.common.keywords_v2_wikidata_description; the ship MERGEs it. Run after gen_ship.py stage.
# Ship 2 (near-miss recheck): DESC_DIR=desc2 VOCAB=openalex.transient.kw1476b_vocab_new WS=openalex.works.work_keyword_senses
#
# Release note: this script drives the keyword description + Wikidata chain built for the whole vocabulary (a separate job, not in this folder).
# Point DESC_CHAIN at that chain's scripts (pilot/: dbx_fetch.py, fetch_missing.py, make_cands.py, choose_jev.py; full/: describe_chunked.py),
# ALIAS_CHAIN at the alias job's match_enwiki.py / match_wikidata.py, and WIKIDATA_SQLITE at a local Wikidata items table (qid, label,
# description, aliases, p31). Needs TYPESAFE_API_KEY (Jev) and OPENROUTER_API_KEY (GPT-6 Luna). Only the shell glue below is ours to release.
# DESC_CHAIN=... ALIAS_CHAIN=... WIKIDATA_SQLITE=... zsh describe_senses.sh
set -e
D=${DESC_DIR:-desc}; VOCAB=${VOCAB:-openalex.transient.kw1476_vocab_new}; WS=${WS:-openalex.transient.kw1476_work_senses}; mkdir -p $D/data
K=${DESC_CHAIN:?set DESC_CHAIN}; H=$K/pilot; S=${ALIAS_CHAIN:?set ALIAS_CHAIN}
PY=${PY:-python3}; export PYTHONPATH=$H
cat > $D/titles.sql <<SQL
WITH r AS (SELECT n.kid, w.title, row_number() OVER (PARTITION BY n.kid ORDER BY ws.prob DESC, hash(ws.work_id)) rn
  FROM $VOCAB n JOIN $WS ws ON ws.sense_id = n.kid
  JOIN openalex.works.openalex_works w ON w.id = ws.work_id WHERE w.title IS NOT NULL)
SELECT n.kid, n.display_name, to_json(collect_list(r.title) FILTER (WHERE r.rn <= 6)) AS titles
FROM $VOCAB n LEFT JOIN r ON r.kid = n.kid GROUP BY n.kid, n.display_name
SQL
python3 $H/dbx_fetch.py $D/titles.sql $D/titles_raw.jsonl
python3 - $D <<'PY'
import json, sys; D = sys.argv[1]
with open(f'{D}/vocab.jsonl', 'w') as v, open(f'{D}/titles.jsonl', 'w') as t:
    for l in open(f'{D}/titles_raw.jsonl'):
        r = json.loads(l); ti = [x for x in json.loads(r['titles'] or '[]') if x][:6]
        v.write(json.dumps({'key': r['display_name'].lower(), 'display': r['display_name'], 'source': [], 'kid': r['kid']}) + '\n')
        t.write(json.dumps({'kid': r['kid'], 'display': r['display_name'], 'titles': ti}, ensure_ascii=False) + '\n')
PY
$PY $S/match_enwiki.py --vocab $D/vocab.jsonl --out $D/enwiki_hits.jsonl 2>&1 | tail -1
$PY $S/match_wikidata.py --vocab $D/vocab.jsonl --out $D/wd_hits.jsonl --enwiki $D/enwiki_hits.jsonl 2>&1 | tail -1
$PY - $D <<'PY'
import json, sqlite3, os, sys; D = sys.argv[1]
con = sqlite3.connect('file:' + os.path.expanduser(os.environ['WIKIDATA_SQLITE']) + '?mode=ro', uri=True)
q = {h['qid'] for f in ('enwiki_hits', 'wd_hits') for l in open(f'{D}/{f}.jsonl') for h in json.loads(l)['hits'] if h.get('qid')}
with open(f'{D}/wd_entities.jsonl', 'w') as f:
    for qid in q:
        r = con.execute('SELECT label,description,aliases,p31 FROM items WHERE qid=?', (qid,)).fetchone()
        f.write(json.dumps({'qid': qid, 'label': r[0], 'description': r[1], 'aliases': json.loads(r[2]) if r[2] else [], 'p31': json.loads(r[3]) if r[3] else []} if r else {'qid': qid, 'missing': True}) + '\n')
PY
python3 $H/fetch_missing.py $D/wd_entities.jsonl
(cd $D && $PY $H/make_cands.py --hits enwiki_hits.jsonl --wd wd_entities.jsonl --wd_hits wd_hits.jsonl --out cands.jsonl)
python3 - $D <<'PY'
import json, sys; D = sys.argv[1]
T = {json.loads(l)['kid']: json.loads(l)['titles'] for l in open(f'{D}/titles.jsonl')}
V = {json.loads(l)['key']: json.loads(l) for l in open(f'{D}/vocab.jsonl')}
with open(f'{D}/cands_t.jsonl', 'w') as f:
    for l in open(f'{D}/cands.jsonl'):
        r = json.loads(l)
        if not r['cands']: continue
        r['kid'] = V[r['key']]['kid']; r['titles'] = T.get(r['kid'], []); f.write(json.dumps(r, ensure_ascii=False) + '\n')
PY
JEV_SESSION=kw1476 $PY $H/choose_jev.py --cands $D/cands_t.jsonl --out $D/jev_choice.jsonl --concurrency 16 2>&1 | tail -1
python3 - $D <<'PY'
import json, sys; D = sys.argv[1]
K = {json.loads(l)['key']: json.loads(l)['kid'] for l in open(f'{D}/vocab.jsonl')}
wd = {}
for l in open(f'{D}/jev_choice.jsonl'):
    r = json.loads(l)
    if r.get('qid') and (r.get('p_choice') or 0) >= 0.7: wd[K.get(r['key'], r.get('kid', r['key']))] = (r['label'], r['description'])
with open(f'{D}/todo_desc.jsonl', 'w') as f:
    for l in open(f'{D}/titles.jsonl'):
        r = json.loads(l); w = wd.get(r['kid'])
        f.write(json.dumps({'key': r['kid'], 'display': r['display'], 'titles': r['titles'], 'wd': {'label': w[0], 'description': w[1]} if w else None}, ensure_ascii=False) + '\n')
print(len(wd), 'confident Wikidata links')
PY
rm -f $D/desc_luna.jsonl
$PY $K/full/describe_chunked.py --todo $D/todo_desc.jsonl --out $D/desc_luna.jsonl --conc 10
echo DESCRIBE_DONE
