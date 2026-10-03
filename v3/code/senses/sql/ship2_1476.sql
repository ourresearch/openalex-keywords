-- PRODUCTION WRITES (#1476 ship 2, signed off). Backups first.
CREATE OR REPLACE TABLE openalex.transient.bak1476b_work_keywords_v2_20261003 DEEP CLONE openalex.works.work_keywords_v2;

CREATE OR REPLACE TABLE openalex.transient.bak1476b_keywords_v2_20261003 DEEP CLONE openalex.common.keywords_v2;

CREATE OR REPLACE TABLE openalex.transient.bak1476b_keywords_v2_wikidata_description_20261003 DEEP CLONE openalex.common.keywords_v2_wikidata_description;

CREATE OR REPLACE TABLE openalex.transient.bak1476b_keywords_v2_senses_20261003 DEEP CLONE openalex.common.keywords_v2_senses;

-- the 9 new sense headings into the served vocabulary
INSERT INTO openalex.common.keywords_v2 (id, kid, display_name, created_date, vocab_version) SELECT DISTINCT concat('https://openalex.org/keywords/', s.sense_id) AS id, s.sense_id AS kid, s.sense_name AS display_name, current_timestamp() AS created_date, 'kw1476-senses-2' AS vocab_version FROM openalex.common.keywords_v2_senses s LEFT ANTI JOIN openalex.common.keywords_v2 v ON v.kid = s.sense_id WHERE (s.heading, s.opt) IN (('accessibility', 's1'), ('social-welfare', 's1'), ('population-structure', 's1'), ('entropy', 's1'), ('resource-utilization', 's1'), ('taxonomy', 's1'), ('decentralization', 's1'), ('housing', 's1'), ('inflation', 's2'));

-- description + Wikidata id (describe_senses.sh + desc_table.py into kw1476b_desc_new); served after the next Sync Keywords
MERGE INTO openalex.common.keywords_v2_wikidata_description t USING openalex.transient.kw1476b_desc_new s ON t.id = s.id WHEN MATCHED THEN UPDATE SET * WHEN NOT MATCHED THEN INSERT *;

-- the 9 senses now split (the Keywords nightly applies them to new works from tonight)
UPDATE openalex.common.keywords_v2_senses SET split = true WHERE (heading, opt) IN (('accessibility', 's1'), ('social-welfare', 's1'), ('population-structure', 's1'), ('entropy', 's1'), ('resource-utilization', 's1'), ('taxonomy', 's1'), ('decentralization', 's1'), ('housing', 's1'), ('inflation', 's2'));

-- recompute from the production rows now (the nightly's own statement), then write only works that change
CREATE OR REPLACE TABLE openalex.transient.kw1476b_ship_rows_in AS SELECT * FROM openalex.works.work_keywords_v2 WHERE work_id IN (SELECT ws.work_id FROM openalex.works.work_keyword_senses ws JOIN openalex.common.keywords_v2_senses s ON s.heading = ws.heading AND s.sense_id = ws.sense_id WHERE (s.heading, s.opt) IN (('accessibility', 's1'), ('social-welfare', 's1'), ('population-structure', 's1'), ('entropy', 's1'), ('resource-utilization', 's1'), ('taxonomy', 's1'), ('decentralization', 's1'), ('housing', 's1'), ('inflation', 's2')));

CREATE OR REPLACE TABLE openalex.transient.kw1476b_ship_rows_out AS
WITH e AS (SELECT r.work_id, r.tagger_version, r.updated_at, p.pos, p.x FROM openalex.transient.kw1476b_ship_rows_in r LATERAL VIEW posexplode(r.keywords) p AS pos, x),
d0 AS (SELECT ws.work_id, concat('https://openalex.org/keywords/', ws.heading) AS hid, s.sense_id, s.sense_name
       FROM openalex.works.work_keyword_senses ws JOIN openalex.common.keywords_v2_senses s ON s.split AND s.heading = ws.heading AND s.sense_id = ws.sense_id
       WHERE ws.work_id IN (SELECT work_id FROM openalex.transient.kw1476b_ship_rows_in)),
m AS (SELECT e.work_id, e.tagger_version, e.updated_at, e.pos, e.x.score AS score,
             COALESCE(concat('https://openalex.org/keywords/', d0.sense_id), e.x.id) AS id, COALESCE(d0.sense_name, e.x.display_name) AS display_name
      FROM e LEFT JOIN d0 ON d0.work_id = e.work_id AND d0.hid = e.x.id),
d AS (SELECT *, row_number() OVER (PARTITION BY work_id, id ORDER BY pos) AS k FROM m)
SELECT work_id,
       TRANSFORM(ARRAY_SORT(COLLECT_LIST(NAMED_STRUCT('pos', pos, 'id', id, 'display_name', display_name, 'score', score))),
                 y -> NAMED_STRUCT('id', y.id, 'display_name', y.display_name, 'score', y.score)) AS keywords,
       FIRST(tagger_version) AS tagger_version, FIRST(updated_at) AS updated_at
FROM d WHERE k = 1 GROUP BY work_id;

CREATE OR REPLACE TABLE openalex.transient.kw1476b_ship_changes AS SELECT o.work_id, o.keywords FROM openalex.transient.kw1476b_ship_rows_out o JOIN openalex.transient.kw1476b_ship_rows_in i ON i.work_id = o.work_id WHERE to_json(o.keywords) <> to_json(i.keywords);

MERGE INTO openalex.works.work_keywords_v2 t USING openalex.transient.kw1476b_ship_changes s ON t.work_id = s.work_id WHEN MATCHED THEN UPDATE SET t.keywords = s.keywords, t.updated_at = CURRENT_TIMESTAMP();

-- after: every id on a changed work is in the vocabulary
SELECT count(*) AS assignments, count_if(v.id IS NULL) AS ids_missing FROM (SELECT explode(keywords) AS k FROM openalex.works.work_keywords_v2 WHERE work_id IN (SELECT work_id FROM openalex.transient.kw1476b_ship_changes)) e LEFT JOIN openalex.common.keywords_v2 v ON v.id = e.k.id;
