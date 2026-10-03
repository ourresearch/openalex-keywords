"""#1476: pull title, abstract and venue for the #1465 field-rule gold (senses_valid.jsonl: works the failed field rule moved, with Opus's
blind sense choice), so the Jev classifier can be scored on them too. -> data/fieldrule_works.jsonl
python pull_fieldrule.py"""
import os, json, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'common')); from dbx import sql
V = [json.loads(l) for l in open('data/senses_valid.jsonl')]
ids = ",".join(sorted({str(r['work_id']) for r in V}))
W = {r['id']: r for r in sql(f"SELECT id, title, substr(abstract, 1, 1500) AS abstract, primary_location.source.display_name AS venue FROM openalex.works.openalex_works WHERE id IN ({ids})", verbose=False)}
with open('data/fieldrule_works.jsonl', 'w') as f:
    for r in V:
        w = W.get(str(r['work_id']))
        if w: f.write(json.dumps({'kid': 'https://openalex.org/keywords/' + r['heading'], 'work_id': r['work_id'], 'title': w['title'], 'abstract': w['abstract'], 'venue': w['venue'], 'field': r['field']}, ensure_ascii=False) + '\n')
print(len(V), len(W))
