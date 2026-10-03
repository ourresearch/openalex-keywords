"""#1476 stage + ship SQL for the homonym senses.
  python gen_ship.py stage [--test]   transient tables only (runs now): kw1476_senses, kw1476_work_senses, kw1476_rows_out, kw1476_wk_changes,
                                           kw1476_vocab_new + checks. --test: every minority sense counts as passed (SQL check before the gate).
  python gen_ship.py ship             writes ship_1476.sql (PRODUCTION: needs a person's sign-off). Recomputes the changed rows from the
                                           production rows at ship time with the nightly's own sense_statement (sense_sql.py), so a nightly in between can't be undone.
Gate: data/gate_v1.json (senses with pass = true). Decisions: openalex.transient.kw1476_jev_raw (load_out.py)."""
import os, importlib.util, json, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'common')); from dbx import sql
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import sense_sql as kn   # the nightly's sense SQL (copied into sense_sql.py)
mode = sys.argv[1]; TEST = '--test' in sys.argv; T = 'openalex.transient.'; DATE = '20261002'
M = json.load(open('data/sense_map.json'))
passed = None if TEST else {(r['heading'], r['opt']) for r in json.load(open('data/gate_v1.json'))['senses'] if r['pass']}
q = lambda v: 'NULL' if v is None else "'" + str(v).replace("\\", "\\\\").replace("'", "\\'") + "'"
vals = ",\n".join(f"({q(h)}, {q(o['opt'])}, {q(o['label'])}, {q(o['sense_id'])}, {q(o['sense_name'])}, "
                  f"{'true' if o['opt'] != 's0' and (TEST or (h, o['opt']) in passed) else 'false'}, {'true' if o['new'] else 'false'})"
                  for h, opts in M.items() for o in opts)
STAGE = [
 f"""CREATE OR REPLACE TABLE {T}kw1476_senses AS SELECT * FROM VALUES {vals} AS t(heading, opt, label, sense_id, sense_name, split, new)""",
 f"""CREATE OR REPLACE TABLE {T}kw1476_work_senses AS
WITH e AS (SELECT r.work_id, r.heading, r.model, m.key AS opt, m.value AS prob,
                  row_number() OVER (PARTITION BY r.work_id, r.heading ORDER BY m.value DESC, m.key) AS k
           FROM {T}kw1476_jev_raw r LATERAL VIEW explode(r.p) m AS key, value WHERE r.ok)
SELECT e.work_id, e.heading, e.opt, e.prob,
       CASE WHEN e.opt NOT IN ('s0', 'none') AND e.prob >= {kn.SENSE_T} THEN s.sense_id END AS sense_id, e.model, current_timestamp() AS decided_at
FROM e LEFT JOIN {T}kw1476_senses s ON s.heading = e.heading AND s.opt = e.opt WHERE e.k = 1""",
 f"""CREATE OR REPLACE TABLE {T}kw1476_rows_in AS SELECT * FROM openalex.works.work_keywords_v2 WHERE work_id IN
  (SELECT ws.work_id FROM {T}kw1476_work_senses ws JOIN {T}kw1476_senses s ON s.split AND s.heading = ws.heading AND s.sense_id = ws.sense_id)""",
 kn.sense_statement(f"{T}kw1476_rows_in", f"{T}kw1476_rows_out", f"{T}kw1476_work_senses", f"{T}kw1476_senses"),
 f"""CREATE OR REPLACE TABLE {T}kw1476_wk_changes AS SELECT o.work_id, o.keywords FROM {T}kw1476_rows_out o JOIN {T}kw1476_rows_in i ON i.work_id = o.work_id
  WHERE to_json(o.keywords) <> to_json(i.keywords)""",
 f"""CREATE OR REPLACE TABLE {T}kw1476_vocab_new AS SELECT DISTINCT concat('https://openalex.org/keywords/', s.sense_id) AS id, s.sense_id AS kid,
  s.sense_name AS display_name, current_timestamp() AS created_date, 'kw1476-senses' AS vocab_version
  FROM {T}kw1476_senses s LEFT ANTI JOIN openalex.common.keywords_v2 v ON v.kid = s.sense_id WHERE s.split""",
]
CHECKS = [
 f"""SELECT (SELECT count(*) FROM {T}kw1476_work_senses) AS decisions, (SELECT count_if(sense_id IS NOT NULL) FROM {T}kw1476_work_senses) AS moves_any_sense,
  (SELECT count(*) FROM {T}kw1476_rows_in) AS works_in, (SELECT count(*) FROM {T}kw1476_rows_out) AS works_out, (SELECT count(*) FROM {T}kw1476_wk_changes) AS works_changing,
  (SELECT count(*) FROM {T}kw1476_vocab_new) AS new_headings,
  (SELECT typeof(keywords) FROM {T}kw1476_rows_out LIMIT 1) = (SELECT typeof(keywords) FROM openalex.works.work_keywords_v2 LIMIT 1) AS type_same""",
 f"""SELECT count(*) AS assignments, count_if(v.id IS NULL AND n.id IS NULL) AS ids_missing FROM (SELECT explode(keywords) AS k FROM {T}kw1476_wk_changes) e
  LEFT JOIN openalex.common.keywords_v2 v ON v.id = e.k.id LEFT JOIN {T}kw1476_vocab_new n ON n.id = e.k.id""",
 f"""SELECT count_if(size(o.keywords) > size(i.keywords)) AS grew, count_if(size(o.keywords) < size(i.keywords)) AS shrank_by_dedupe,
  count_if(size(array_distinct(transform(o.keywords, x -> x.id))) <> size(o.keywords)) AS dup_ids
  FROM {T}kw1476_rows_out o JOIN {T}kw1476_rows_in i ON i.work_id = o.work_id""",
]
if mode == 'stage':
    for s in STAGE: sql(s, verbose=False)
    for c in CHECKS: print(sql(c, verbose=False))
    print(sql(f"""SELECT o.work_id, slice(transform(i.keywords, x -> x.display_name), 1, 6) AS before, slice(transform(o.keywords, x -> x.display_name), 1, 6) AS after
      FROM {T}kw1476_wk_changes o JOIN {T}kw1476_rows_in i ON i.work_id = o.work_id ORDER BY xxhash64(o.work_id) LIMIT 5""", verbose=False))
else:
    W = 'openalex.works.work_keywords_v2'
    SHIP = [
     f"-- PRODUCTION WRITES (#1476, signed off). Backups first.\nCREATE OR REPLACE TABLE {T}bak1476_work_keywords_v2_{DATE} DEEP CLONE {W}",
     f"CREATE OR REPLACE TABLE {T}bak1476_keywords_v2_{DATE} DEEP CLONE openalex.common.keywords_v2",
     f"CREATE OR REPLACE TABLE {T}bak1476_keywords_v2_wikidata_description_{DATE} DEEP CLONE openalex.common.keywords_v2_wikidata_description",
     f"-- new sense headings into the served vocabulary\nINSERT INTO openalex.common.keywords_v2 (id, kid, display_name, created_date, vocab_version)\n"
     f"SELECT n.id, n.kid, n.display_name, n.created_date, n.vocab_version FROM {T}kw1476_vocab_new n LEFT ANTI JOIN openalex.common.keywords_v2 v ON v.id = n.id",
     f"-- description + Wikidata id for the new headings (describe_senses.sh, desc_table.py: oxjob #1463's chain); served after the next Sync Keywords\n"
     f"MERGE INTO openalex.common.keywords_v2_wikidata_description t USING {T}kw1476_desc_new s ON t.id = s.id\n"
     f"WHEN MATCHED THEN UPDATE SET * WHEN NOT MATCHED THEN INSERT *",
     f"-- the two tables the Keywords nightly's sense step reads\nCREATE TABLE {kn.SENSES} AS SELECT heading, opt, label, sense_id, sense_name, split FROM {T}kw1476_senses",
     f"CREATE TABLE {kn.WORK_SENSES} AS SELECT work_id, heading, opt, prob, sense_id, model, decided_at FROM {T}kw1476_work_senses",
     f"-- recompute from the production rows now (the nightly's own statement), then write only works that change\n"
     f"CREATE OR REPLACE TABLE {T}kw1476_ship_rows_in AS SELECT * FROM {W} WHERE work_id IN (SELECT ws.work_id FROM {kn.WORK_SENSES} ws JOIN {kn.SENSES} s ON s.split AND s.heading = ws.heading AND s.sense_id = ws.sense_id)",
     kn.sense_statement(f"{T}kw1476_ship_rows_in", f"{T}kw1476_ship_rows_out"),
     f"CREATE OR REPLACE TABLE {T}kw1476_ship_changes AS SELECT o.work_id, o.keywords FROM {T}kw1476_ship_rows_out o JOIN {T}kw1476_ship_rows_in i ON i.work_id = o.work_id WHERE to_json(o.keywords) <> to_json(i.keywords)",
     f"MERGE INTO {W} t USING {T}kw1476_ship_changes s ON t.work_id = s.work_id WHEN MATCHED THEN UPDATE SET t.keywords = s.keywords, t.updated_at = CURRENT_TIMESTAMP()",
     f"-- after: every id on a changed work is in the vocabulary\nSELECT count(*) AS assignments, count_if(v.id IS NULL) AS ids_missing FROM (SELECT explode(keywords) AS k FROM {W} WHERE work_id IN (SELECT work_id FROM {T}kw1476_ship_changes)) e LEFT JOIN openalex.common.keywords_v2 v ON v.id = e.k.id",
    ]
    open('ship_1476.sql', 'w').write(";\n\n".join(SHIP) + ";\n"); print('wrote ship_1476.sql:', len(SHIP), 'statements')
