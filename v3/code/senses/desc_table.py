"""#1476: $DESC_DIR (default desc/) (describe_senses.sh) -> openalex.transient.kw1476_desc_new, same columns as
openalex.common.keywords_v2_wikidata_description (Wikidata only when Jev p >= 0.7, as #1463's build_table.sql). The ship MERGEs it.
python desc_table.py   (ship 2: DESC_DIR=desc2 DESC_OUT=openalex.transient.kw1476b_desc_new VOCAB=openalex.transient.kw1476b_vocab_new)"""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'common')); from dbx import sql
D = os.environ.get('DESC_DIR', 'desc'); OUT = os.environ.get('DESC_OUT', 'openalex.transient.kw1476_desc_new'); VOCAB = os.environ.get('VOCAB', 'openalex.transient.kw1476_vocab_new')
K = {json.loads(l)['key']: json.loads(l)['kid'] for l in open(f'{D}/vocab.jsonl')}
J = {K[r['key']]: r for r in map(json.loads, open(f'{D}/jev_choice.jsonl'))}
q = lambda v: 'NULL' if v is None else "'" + str(v).replace("\\", "\\\\").replace("'", "\\'") + "'"
rows = []
for r in map(json.loads, open(f'{D}/desc_luna.jsonl')):
    j = J.get(r['key'], {}); ok = j.get('qid') and (j.get('p_choice') or 0) >= 0.7
    rows.append(f"({q('https://openalex.org/keywords/' + r['key'])}, {q('https://www.wikidata.org/wiki/' + j['qid']) if ok else 'NULL'}, "
                f"{float(j['p_choice']) if ok else 'NULL'}, {q(r['description'])}, {q(r['model'])}, {'true' if r['unclear'] else 'false'})")
cols = [c['col_name'] for c in sql("DESCRIBE openalex.common.keywords_v2_wikidata_description", verbose=False) if c['col_name'] and not c['col_name'].startswith('#')]
print(cols)
sql(f"""CREATE OR REPLACE TABLE {OUT} AS SELECT id, wikidata, CAST(wikidata_p_choice AS DOUBLE) AS wikidata_p_choice, description,
  description_model, description_unclear, current_timestamp() AS built_at FROM VALUES {','.join(rows)} AS t(id, wikidata, wikidata_p_choice, description, description_model, description_unclear)""", verbose=False)
print(sql(f"""SELECT count(*) n, count(wikidata) wd, count(description) d, count_if(n.id IN (SELECT id FROM {VOCAB})) in_vocab_new,
  count_if(n.id IN (SELECT id FROM openalex.common.keywords_v2_wikidata_description)) already_in_prod FROM {OUT} n""", verbose=False))
