"""The sense constants and SQL the OpenAlex Keywords nightly uses (copied from the pipeline repo's keywords nightly module, verbatim apart from
this docstring). gen_ship.py and gen_ship2.py import it so the one-off ship and the nightly rewrite works with the same statement.
Not a script: import sense_sql as kn."""

SENSES = "openalex.common.keywords_v2_senses"     # homonym senses (oxjob #1476): one row per Jev option of a split keyword:
#   heading, opt ('s0' = main, 's1'..), label (the Jev option text), sense_id / sense_name (the sense heading; NULL for s0), split (passed the gate)
WORK_SENSES = "openalex.works.work_keyword_senses"  # one row per classified (work, heading): opt, prob, sense_id (NULL = stays), model, decided_at

# The Jev question exactly as validated in oxjob #1476 (jev_senses.py / corpus_run.py, jev-1.13.0): options in `opt` order + none; a work
# moves when the top option is a minority sense with probability >= SENSE_T (chosen on train keywords).
SENSE_INSTR = "Which meaning of the keyword does this scholarly work use? Judge from the title and abstract; the venue is a hint only."
SENSE_NONE = "none of these meanings / the work is not about this keyword"
SENSE_T = 0.5


def sense_question(labels):
    """labels: the keyword's option texts in opt order (s0 first) -> Jev questions dict."""
    crit = {f"s{i}": lab for i, lab in enumerate(labels)}
    crit["none"] = SENSE_NONE
    return {"sense": {"type": "choice", "instructions": SENSE_INSTR, "criteria": crit}}


def sense_state(heading, title, abstract, venue):
    return {"keyword": heading.replace("-", " "), "title": title or "", "abstract": (abstract or "")[:1500], "venue": venue or ""}


def sense_decision(probs):
    """Jev probabilities {'s0': .., 's1': .., 'none': ..} -> (opt, prob) of the top option; the caller moves the work only when opt != 's0',
    opt != 'none' and prob >= SENSE_T."""
    opt = max(probs, key=probs.get)
    return opt, float(probs[opt])


def sense_pairs_statement(rows, queue, out, work_senses=WORK_SENSES, senses=SENSES):
    """(work, split keyword) pairs in tonight's rows with no decision yet, with the text Jev reads (from the tagger queue)."""
    return f"""CREATE OR REPLACE TABLE {out} AS
WITH e AS (SELECT r.work_id, substring_index(x.id, '/', -1) AS heading FROM {rows} r LATERAL VIEW explode(r.keywords) t AS x),
p AS (SELECT DISTINCT e.work_id, e.heading FROM e JOIN (SELECT DISTINCT heading FROM {senses}) s ON s.heading = e.heading
      LEFT ANTI JOIN {work_senses} ws ON ws.work_id = e.work_id AND ws.heading = e.heading)
SELECT p.work_id, p.heading, q.title, substr(q.abstract, 1, 1500) AS abstract, q.journal AS venue FROM p JOIN {queue} q ON q.id = p.work_id"""


def sense_statement(rows_in, rows_out, work_senses=WORK_SENSES, senses=SENSES):
    """Homonym senses (oxjob #1476): rows in the work_keywords_v2 shape -> the same rows with each keyword whose per-work decision names a
    sense that passed the gate (senses.split) replaced by that sense heading (id + display name). Everything else is unchanged: order,
    scores, tagger_version, updated_at. A work with no such decision keeps the main sense. If a work ends up with the same id twice (it
    already carried the sense heading), the first position wins."""
    return f"""CREATE OR REPLACE TABLE {rows_out} AS
WITH e AS (SELECT r.work_id, r.tagger_version, r.updated_at, p.pos, p.x FROM {rows_in} r LATERAL VIEW posexplode(r.keywords) p AS pos, x),
d0 AS (SELECT ws.work_id, concat('https://openalex.org/keywords/', ws.heading) AS hid, s.sense_id, s.sense_name
       FROM {work_senses} ws JOIN {senses} s ON s.split AND s.heading = ws.heading AND s.sense_id = ws.sense_id
       WHERE ws.work_id IN (SELECT work_id FROM {rows_in})),
m AS (SELECT e.work_id, e.tagger_version, e.updated_at, e.pos, e.x.score AS score,
             COALESCE(concat('https://openalex.org/keywords/', d0.sense_id), e.x.id) AS id, COALESCE(d0.sense_name, e.x.display_name) AS display_name
      FROM e LEFT JOIN d0 ON d0.work_id = e.work_id AND d0.hid = e.x.id),
d AS (SELECT *, row_number() OVER (PARTITION BY work_id, id ORDER BY pos) AS k FROM m)
SELECT work_id,
       TRANSFORM(ARRAY_SORT(COLLECT_LIST(NAMED_STRUCT('pos', pos, 'id', id, 'display_name', display_name, 'score', score))),
                 y -> NAMED_STRUCT('id', y.id, 'display_name', y.display_name, 'score', y.score)) AS keywords,
       FIRST(tagger_version) AS tagger_version, FIRST(updated_at) AS updated_at
FROM d WHERE k = 1 GROUP BY work_id"""
