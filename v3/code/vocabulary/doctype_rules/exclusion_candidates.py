"""Candidates for the keyword exclusion list: keywords on >= 25 works (lower/trim over keywords_student_v1) whose wording touches a document type
(OpenAlex `type`); with the top 3 work types each keyword sits on or a study design (#1312 study_designs schema). Broad regex on purpose; Opus judges each (judge_exclusions.py). Writes data/exclusion_candidates.json"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'common')); from dbx import sql
HERE = os.path.dirname(os.path.abspath(__file__))
RX = (r"(trial|\\brcts?\\b|randomi[sz]|review|meta.?analy|case report|case series|observational|cohort|case.control|cross.sectional|protocol|systematic|"
      r"evidence synthesis|letter|editorial|errat|corrig|retract|dissertation|thesis|theses|preprint|pre-print|dataset|data set|data paper|chapter|monograph|"
      r"\\bbooks?\\b|proceedings|conference|abstract|poster|report|obituar|biograph|commentary|guide|standard|reference entry|encyclop|supplement|"
      r"software|peer.review|paratext|front matter|table of contents|masthead|issue information|\\bnews\\b|interview|correspondence|\\breply\\b|comment|"
      r"study design|clinical study|phase (i|ii|iii|iv|1|2|3|4)\\b|twin stud|research article|original article|original research|short communication|"
      r"working paper|white paper|technical note|errata|addendum|in memoriam|festschrift|catalog)")
Q = f"""
WITH x AS (SELECT DISTINCT id, lower(trim(k)) AS kc FROM openalex.transient.keywords_student_v1 LATERAL VIEW explode(keywords) e AS k WHERE trim(k) <> ''),
kf AS (SELECT kc, count(*) AS n FROM x GROUP BY kc),
cand AS (SELECT kc, n FROM kf WHERE n >= 25 AND kc RLIKE '{RX}'),
w AS (SELECT concat('W', CAST(id AS STRING)) AS id, type FROM openalex.works.openalex_works),
ty AS (SELECT x.kc, w.type, count(*) AS m FROM x JOIN cand ON cand.kc = x.kc JOIN w ON w.id = x.id GROUP BY x.kc, w.type),
top2 AS (SELECT kc, collect_list(concat(type, ' ', CAST(m AS STRING))) AS types FROM (SELECT kc, type, m, row_number() OVER (PARTITION BY kc ORDER BY m DESC) AS r FROM ty) WHERE r <= 3 GROUP BY kc)
SELECT cand.kc, cand.n, top2.types FROM cand LEFT JOIN top2 ON top2.kc = cand.kc ORDER BY cand.n DESC
"""
rows = sql(Q, warehouse=os.environ.get('DATABRICKS_WAREHOUSE_ID'), timeout_s=5400, verbose=False)
json.dump(rows, open(f'{HERE}/data/exclusion_candidates.json', 'w')); print('done', len(rows))
