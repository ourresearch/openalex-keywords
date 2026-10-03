"""#1476 ship 2: the 9 near-miss senses that passed the fresh 150-move recheck (recheck.py -> data/recheck.json; pre-registration in EXPLORE.md).
Their decisions are already in production (openalex.works.work_keyword_senses; split = false in keywords_v2_senses), so ship 2 only adds the 9 headings
+ descriptions, flips split, and rewrites the works whose decisions name them.
  python gen_ship2.py stage   transient only: kw1476b_vocab_new, kw1476b_senses (production senses with the 9 flipped), kw1476b_rows_in/out,
                                   kw1476b_wk_changes + checks. Then describe_senses.sh / desc_table.py with DESC_DIR=.../desc2 VOCAB=...kw1476b_vocab_new
                                   WS=openalex.works.work_keyword_senses DESC_OUT=openalex.transient.kw1476b_desc_new
  python gen_ship2.py ship    writes ship2_1476.sql (PRODUCTION: needs a person's sign-off); run with run_ship.py --file ship2_1476.sql"""
import os, importlib.util, json, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'common')); from dbx import sql
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import sense_sql as kn   # the nightly's sense SQL (copied into sense_sql.py)
mode = sys.argv[1]; T = 'openalex.transient.'; DATE = '20261003'; W = 'openalex.works.work_keywords_v2'
P = [r for r in json.load(open('data/recheck.json')) if r['pass']]; assert len(P) == 9, len(P)
q = lambda v: "'" + str(v).replace("\\", "\\\\").replace("'", "\\'") + "'"
PASS = ", ".join(f"({q(r['heading'])}, {q(r['opt'])})" for r in P)
VOCAB = (f"SELECT DISTINCT concat('https://openalex.org/keywords/', s.sense_id) AS id, s.sense_id AS kid, s.sense_name AS display_name, "
         f"current_timestamp() AS created_date, 'kw1476-senses-2' AS vocab_version FROM {kn.SENSES} s "
         f"LEFT ANTI JOIN openalex.common.keywords_v2 v ON v.kid = s.sense_id WHERE (s.heading, s.opt) IN ({PASS})")
def rows_in(senses, out): return (f"CREATE OR REPLACE TABLE {out} AS SELECT * FROM {W} WHERE work_id IN (SELECT ws.work_id FROM {kn.WORK_SENSES} ws "
                                  f"JOIN {senses} s ON s.heading = ws.heading AND s.sense_id = ws.sense_id WHERE (s.heading, s.opt) IN ({PASS}))")
if mode == 'stage':
    for s in [f"CREATE OR REPLACE TABLE {T}kw1476b_vocab_new AS {VOCAB}",
              f"CREATE OR REPLACE TABLE {T}kw1476b_senses AS SELECT heading, opt, label, sense_id, sense_name, split OR (heading, opt) IN ({PASS}) AS split FROM {kn.SENSES}",
              rows_in(f'{T}kw1476b_senses', f'{T}kw1476b_rows_in'),
              kn.sense_statement(f'{T}kw1476b_rows_in', f'{T}kw1476b_rows_out', kn.WORK_SENSES, f'{T}kw1476b_senses'),
              f"CREATE OR REPLACE TABLE {T}kw1476b_wk_changes AS SELECT o.work_id, o.keywords FROM {T}kw1476b_rows_out o JOIN {T}kw1476b_rows_in i ON i.work_id = o.work_id WHERE to_json(o.keywords) <> to_json(i.keywords)"]:
        sql(s, verbose=False)
    for c in [f"""SELECT (SELECT count(*) FROM {T}kw1476b_vocab_new) new_headings, (SELECT count_if(split) FROM {T}kw1476b_senses) split_after,
                 (SELECT count_if(split) FROM {kn.SENSES}) split_before, (SELECT count(*) FROM {T}kw1476b_rows_in) works_in, (SELECT count(*) FROM {T}kw1476b_wk_changes) works_changing""",
              f"""SELECT count(*) AS assignments, count_if(v.id IS NULL AND n.id IS NULL) AS ids_missing FROM (SELECT explode(keywords) AS k FROM {T}kw1476b_wk_changes) e
                 LEFT JOIN openalex.common.keywords_v2 v ON v.id = e.k.id LEFT JOIN {T}kw1476b_vocab_new n ON n.id = e.k.id""",
              f"""SELECT count_if(size(o.keywords) > size(i.keywords)) AS grew, count_if(size(o.keywords) < size(i.keywords)) AS shrank_by_dedupe,
                 count_if(size(array_distinct(transform(o.keywords, x -> x.id))) <> size(o.keywords)) AS dup_ids FROM {T}kw1476b_rows_out o JOIN {T}kw1476b_rows_in i ON i.work_id = o.work_id""",
              f"""SELECT n.kid, count(*) works FROM (SELECT explode(keywords) AS k FROM {T}kw1476b_wk_changes) e JOIN {T}kw1476b_vocab_new n ON n.id = e.k.id GROUP BY 1 ORDER BY 2 DESC"""]:
        print(sql(c, verbose=False))
    print(sql(f"""SELECT o.work_id, slice(transform(i.keywords, x -> x.display_name), 1, 6) AS before, slice(transform(o.keywords, x -> x.display_name), 1, 6) AS after
      FROM {T}kw1476b_wk_changes o JOIN {T}kw1476b_rows_in i ON i.work_id = o.work_id ORDER BY xxhash64(o.work_id) LIMIT 5""", verbose=False))
else:
    SHIP = [
     f"-- PRODUCTION WRITES (#1476 ship 2, signed off). Backups first.\nCREATE OR REPLACE TABLE {T}bak1476b_work_keywords_v2_{DATE} DEEP CLONE {W}",
     f"CREATE OR REPLACE TABLE {T}bak1476b_keywords_v2_{DATE} DEEP CLONE openalex.common.keywords_v2",
     f"CREATE OR REPLACE TABLE {T}bak1476b_keywords_v2_wikidata_description_{DATE} DEEP CLONE openalex.common.keywords_v2_wikidata_description",
     f"CREATE OR REPLACE TABLE {T}bak1476b_keywords_v2_senses_{DATE} DEEP CLONE {kn.SENSES}",
     f"-- the 9 new sense headings into the served vocabulary\nINSERT INTO openalex.common.keywords_v2 (id, kid, display_name, created_date, vocab_version) {VOCAB}",
     f"-- description + Wikidata id (describe_senses.sh + desc_table.py into kw1476b_desc_new); served after the next Sync Keywords\n"
     f"MERGE INTO openalex.common.keywords_v2_wikidata_description t USING {T}kw1476b_desc_new s ON t.id = s.id WHEN MATCHED THEN UPDATE SET * WHEN NOT MATCHED THEN INSERT *",
     f"-- the 9 senses now split (the Keywords nightly applies them to new works from tonight)\nUPDATE {kn.SENSES} SET split = true WHERE (heading, opt) IN ({PASS})",
     f"-- recompute from the production rows now (the nightly's own statement), then write only works that change\n" + rows_in(kn.SENSES, f'{T}kw1476b_ship_rows_in'),
     kn.sense_statement(f'{T}kw1476b_ship_rows_in', f'{T}kw1476b_ship_rows_out'),
     f"CREATE OR REPLACE TABLE {T}kw1476b_ship_changes AS SELECT o.work_id, o.keywords FROM {T}kw1476b_ship_rows_out o JOIN {T}kw1476b_ship_rows_in i ON i.work_id = o.work_id WHERE to_json(o.keywords) <> to_json(i.keywords)",
     f"MERGE INTO {W} t USING {T}kw1476b_ship_changes s ON t.work_id = s.work_id WHEN MATCHED THEN UPDATE SET t.keywords = s.keywords, t.updated_at = CURRENT_TIMESTAMP()",
     f"-- after: every id on a changed work is in the vocabulary\nSELECT count(*) AS assignments, count_if(v.id IS NULL) AS ids_missing FROM (SELECT explode(keywords) AS k FROM {W} WHERE work_id IN (SELECT work_id FROM {T}kw1476b_ship_changes)) e LEFT JOIN openalex.common.keywords_v2 v ON v.id = e.k.id",
    ]
    open('ship2_1476.sql', 'w').write(";\n\n".join(SHIP) + ";\n"); print('wrote ship2_1476.sql:', len(SHIP), 'statements')
