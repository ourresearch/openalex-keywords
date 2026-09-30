"""Build the keyword vocabulary + per-work keywords under Jason's 2026-09-25 rules (#1359 decisions, #1322 PLAN):
(a) keyword id = keywords/<kid>, kid = kid.py (canon + slug), display name = the most common surface form;
(b) purge every keyword in keyword_doctype_verdicts_v1 with kind <> 'keep', except the approved dual-use strings (keyword_dual_use_v1.proposed_dual_use), which are dropped
    only on works whose own type (work_type) or study design (openalex.works.works_study_design, #1335) already says it;
(c) country keywords kept only with evidence: the country (name, adjective, native name) in the title/abstract, or an author institution in that country;
(d) floor: a keyword survives only on >= 100 works (counted after (b) and (c)).
(e) cleanups (peer review 2026-09-26): 'malformed' (U+FFFD, non-letter first character, stopwords, bare years, centuries > 21st), 'loop' (>= 6 keywords
    on a work sharing a 2-word prefix: keep the first 3), 'venue_name' (the keyword is the work's own source name).
Tables (openalex.transient): kw1322_strings0, kw1322_fold, kw1322_strings, kw1322_purge, kw1322_country_kids, kw1322_mentions_kept, kw1322_vocab, kw1322_work_keywords.
uv run --with httpx python build_vocab.py [--from-step N]
v3: build_vocab.py --source keywords_student_v3 --prefix kw1322v3_ [--min-lp -1.0] [--floor 100]  (tables openalex.transient.kw1322v3_*; score = exp(mean token log-prob))"""
import sys, os, json, re, time, argparse
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'common')); import dbxr as dbx   # submit once, retry polls
HERE = os.path.dirname(os.path.abspath(__file__)); T = 'openalex.transient'
ap = argparse.ArgumentParser(); ap.add_argument('--from-step', type=int, default=0); ap.add_argument('--floor', type=int, default=100); ap.add_argument('--source', default='keywords_student_v1'); ap.add_argument('--prefix', default='kw1322_'); ap.add_argument('--boilerplate', action='store_true', help='drop keywords on >= 90%% of the works of a source with >= 10K works (stamped sets)'); ap.add_argument('--boiler-keep', default='large-helical-device', help='comma list of kids exempt from --boilerplate'); ap.add_argument('--taxon-evidence', action='store_true', help='title-only works whose title is a bare taxon name keep only keywords whose words appear in the title'); ap.add_argument('--synmap', default=None, help='openalex.transient table (kid, canon) of judged synonym merges, applied after the plural fold'); ap.add_argument('--min-lp', type=float, default=None, help='sources with lp (v3): drop mentions whose mean token log-prob is below this'); ap.add_argument('--also', type=int, action='append', default=[], help='run this step even if before --from-step'); a = ap.parse_args()
P = a.prefix; HAS_LP = a.source != 'keywords_student_v1'
# one row per mention: work_id, pos, raw, lp (NULL for v1); the confidence cut (if any) applies here, before fold / purge / floor
MSRC = (f"SELECT s.id AS work_id, e.pos, trim(e.z.keywords) AS raw, e.z.lp AS lp FROM {T}.{a.source} s LATERAL VIEW posexplode(arrays_zip(s.keywords, s.lp)) e AS pos, z"
        + (f" WHERE e.z.lp >= {a.min_lp}" if a.min_lp is not None else "")) if HAS_LP else \
       f"SELECT s.id AS work_id, e.pos, trim(e.k) AS raw, CAST(NULL AS DOUBLE) AS lp FROM {T}.{a.source} s LATERAL VIEW posexplode(s.keywords) e AS pos, k"
SYNJ = f" LEFT JOIN {T}.{a.synmap} sy ON sy.kid = COALESCE(f.kid, s.kid0)" if a.synmap else ""
KIDX = "COALESCE(sy.canon, f.kid, s.kid0)" if a.synmap else "COALESCE(f.kid, s.kid0)"
KEEPB = ", ".join("'" + k.strip() + "'" for k in a.boiler_keep.split(",") if k.strip()) or "''"
BOILER_CTE = (f"wsrc AS (SELECT concat('W', CAST(id AS STRING)) AS work_id, primary_location.source.id AS src FROM openalex.works.openalex_works WHERE primary_location.source.id IN (SELECT DISTINCT src FROM {T}.{P}boiler)),\n"
              f"    boil AS (SELECT mk.work_id, mk.pos FROM mk JOIN wsrc ON wsrc.work_id = mk.work_id JOIN {T}.{P}boiler b ON b.src = wsrc.src AND b.kid = mk.kid WHERE mk.kid NOT IN ({KEEPB})),") if a.boilerplate else ""
BOILER_CASE = "WHEN bo.pos IS NOT NULL THEN 'boilerplate'" if a.boilerplate else ""
BOILER_JOIN = "LEFT JOIN boil bo ON bo.work_id = f.work_id AND bo.pos = f.pos" if a.boilerplate else ""
# a bare taxon title: Genus [species] [(Author, year) | Author]; no abstract. Keep a keyword only if every word of it (>= 3 letters) is in the title.
TAXON_CTE = ("tw AS (SELECT concat('W', CAST(id AS STRING)) AS work_id, lower(title) AS lt FROM openalex.works.openalex_works WHERE COALESCE(length(abstract), 0) < 50 "
             "AND primary_location.source.display_name IN ('DiSSCo', 'Global Biodiversity Information Facility', 'Plutof.  Data Management and Publishing Platform', 'Atlas of Living Australia', 'E-Pics 4, Biosys', 'Zenodo (CERN European Organization for Nuclear Research)') "
             "AND title RLIKE '^[A-Z][a-z]+( [a-z]{3,}){0,2}( [(A-Z].*)?$' AND length(title) < 120),\n"
             "    taxon AS (SELECT mk.work_id, mk.pos FROM mk JOIN tw ON tw.work_id = mk.work_id "
             "WHERE NOT forall(filter(split(lower(mk.raw), '[^a-z]+'), x -> length(x) >= 3), x -> instr(tw.lt, x) > 0)),") if a.taxon_evidence else ""
TAXON_CASE = "WHEN tx.pos IS NOT NULL THEN 'taxon_unsupported'" if a.taxon_evidence else ""
TAXON_JOIN = "LEFT JOIN taxon tx ON tx.work_id = f.work_id AND tx.pos = f.pos" if a.taxon_evidence else ""
DISC = ", ".join("'" + k + "'" for k in """medicine biology physics chemistry computer-science engineering psychology economics history education literature literary-criticism epidemiology
theology geography public-health physiology sociology philosophy anthropology archaeology linguistics law political-science statistics ecology geology astronomy botany zoology
genetics immunology microbiology pharmacology neuroscience nursing agriculture management business art music religion humanities social-sciences natural-sciences life-sciences
earth-sciences environmental-science materials-science biochemistry molecular-biology cell-biology oncology cardiology psychiatry pediatrics surgery dentistry veterinary-medicine
architecture finance accounting marketing sciences science technology health research""".split())   # peer review 2026-09-28: discipline labels only on field-level works
SCORE = "round(exp(lp), 3)" if HAS_LP else "round(1 - rnk / n, 3)"   # v3: geometric-mean token probability of the keyword
kid_src = open(os.path.join(HERE, '..', 'common', 'kid.py')).read().split('"""', 2)[2]   # drop the module docstring
countries = json.load(open(f'{HERE}/data/countries.json'))
AMBIG_KW = {'us', 'roc', 'rok', 'gb', 'brd', 'frg'}          # not treated as country keywords (ROC curve, pronoun, ...)
def evidence_rx(c):
    names = [n for n in c['aliases'] if (len(n) >= 4 or '.' in n) and n != 'america'] + c['adjectives'] + c['native']
    latin = sorted({n for n in names if all(ord(ch) < 0x250 for ch in n)}, key=len, reverse=True)
    other = sorted({n for n in names if n not in latin}, key=len, reverse=True)
    parts = []
    if latin: parts.append(r'(?<![\p{L}\p{N}])(?:' + '|'.join(r'\Q' + n + r'\E' for n in latin) + r')(?![\p{L}\p{N}])')
    if other: parts.append('(?:' + '|'.join(r'\Q' + n + r'\E' for n in other) + ')')
    return '(?iu)' + '|'.join(parts)
q = lambda s: "'" + s.replace('\\', '\\\\').replace("'", "\\'") + "'"
STEPS = [
 ("udf", f"CREATE OR REPLACE FUNCTION {T}.kw1322_kid0(s STRING) RETURNS STRING LANGUAGE PYTHON DETERMINISTIC AS $$\n{kid_src}\nreturn kid0(s)\n$$"),
 ("strings0", f"""CREATE OR REPLACE TABLE {T}.{P}strings0 AS
    SELECT raw, mentions, {T}.kw1322_kid0(raw) AS kid0 FROM (SELECT raw, count(*) AS mentions FROM ({MSRC}) GROUP BY raw)"""),
 ("fold", f"""CREATE OR REPLACE TABLE {T}.{P}fold AS
    WITH b AS (SELECT kid0, sum(mentions) AS m FROM {T}.{P}strings0 WHERE kid0 IS NOT NULL GROUP BY kid0),
    caps AS (SELECT kid0 FROM {T}.{P}strings0 WHERE kid0 IS NOT NULL GROUP BY kid0 HAVING sum(CASE WHEN raw RLIKE '^[A-Z][A-Z0-9]*S$' THEN mentions ELSE 0 END) >= 0.5 * sum(mentions)),   -- ALS, NAS: acronyms, not plurals
    p AS (SELECT kid0, m FROM b WHERE kid0 RLIKE '[^s]s$' AND kid0 NOT IN (SELECT kid0 FROM caps)),
    c AS (SELECT p.kid0 AS plural, p.m AS pm, x.cand FROM p LATERAL VIEW explode(array(regexp_replace(p.kid0, 'ies$', 'y'), regexp_replace(p.kid0, '(ss|sh|ch|x|z)es$', '$1'), regexp_replace(p.kid0, 's$', ''))) x AS cand),
    pairs AS (SELECT c.plural, max_by(b.kid0, b.m) AS singular FROM c JOIN b ON b.kid0 = c.cand AND b.kid0 <> c.plural WHERE b.m >= 0.05 * c.pm GROUP BY c.plural),
    grp AS (SELECT singular AS g, plural AS member FROM pairs UNION SELECT singular, singular FROM pairs),
    rep AS (SELECT grp.g, max_by(grp.member, b.m) AS kid FROM grp JOIN b ON b.kid0 = grp.member GROUP BY grp.g)
    SELECT grp.member AS kid0, rep.kid FROM grp JOIN rep ON rep.g = grp.g"""),
 ("strings", f"""CREATE OR REPLACE TABLE {T}.{P}strings AS
    SELECT s.raw, s.mentions, s.kid0, {KIDX} AS kid FROM {T}.{P}strings0 s LEFT JOIN {T}.{P}fold f ON f.kid0 = s.kid0{SYNJ}"""),
 ("purge", f"""CREATE OR REPLACE TABLE {T}.{P}purge AS
    SELECT COALESCE(f.kid, {T}.kw1322_kid0(v.keyword)) AS kid, v.keyword, v.kind, v.maps_to, COALESCE(d.proposed_dual_use, false) AS dual_use
    FROM {T}.keyword_doctype_verdicts_v1 v LEFT JOIN {T}.keyword_dual_use_v1 d ON d.keyword = v.keyword
    LEFT JOIN {T}.{P}fold f ON f.kid0 = {T}.kw1322_kid0(v.keyword) WHERE v.kind <> 'keep'"""),
 ("countries", f"""CREATE OR REPLACE TABLE {T}.{P}country_kids AS SELECT * FROM VALUES """ + ", ".join(
     f"({q(k)}, {q(c['cc'])}, {q(evidence_rx(c))})" for c in countries for k in sorted({x for x in c['aliases'] + c['native'] if x not in AMBIG_KW})) + " AS t(alias, cc, rx)"),
 ("countries_kid", f"""CREATE OR REPLACE TABLE {T}.{P}country_kids AS SELECT DISTINCT COALESCE(f.kid, c.kid0) AS kid, c.cc, c.rx FROM (SELECT {T}.kw1322_kid0(alias) AS kid0, cc, rx FROM {T}.{P}country_kids) c LEFT JOIN {T}.{P}fold f ON f.kid0 = c.kid0"""),
 ("boiler", f"""CREATE OR REPLACE TABLE {T}.{P}boiler AS
    WITH ws AS (SELECT concat('W', CAST(id AS STRING)) AS work_id, primary_location.source.id AS src FROM openalex.works.openalex_works WHERE primary_location.source.id IS NOT NULL),
    big AS (SELECT src, count(*) AS n FROM ws GROUP BY src HAVING count(*) >= 10000),
    mk AS (SELECT DISTINCT m.work_id, st.kid FROM ({MSRC}) m JOIN {T}.{P}strings st ON st.raw = m.raw WHERE st.kid IS NOT NULL)
    SELECT ws.src, mk.kid, count(*) AS works, max(big.n) AS src_works FROM mk JOIN ws ON ws.work_id = mk.work_id JOIN big ON big.src = ws.src
    GROUP BY ws.src, mk.kid HAVING count(*) >= 0.9 * max(big.n)""" if a.boilerplate else "SELECT 1"),
 ("mentions_kept", f"""CREATE OR REPLACE TABLE {T}.{P}mentions_kept AS
    WITH m AS ({MSRC}),
    mk AS (SELECT m.work_id, m.pos, m.raw, m.lp, st.kid, st.kid0 FROM m JOIN {T}.{P}strings st ON st.raw = m.raw WHERE st.kid IS NOT NULL),
    pk AS (SELECT kid, max(CAST(NOT dual_use AS INT)) AS hard, collect_set(CASE WHEN dual_use THEN named_struct('kind', kind, 'maps_to', maps_to) END) AS dual FROM {T}.{P}purge GROUP BY kid),
    ck AS (SELECT kid, collect_set(named_struct('cc', cc, 'rx', rx)) AS cs FROM {T}.{P}country_kids GROUP BY kid),
    flagged AS (SELECT mk.*, pk.hard, pk.dual, ck.cs FROM mk LEFT JOIN pk ON pk.kid = mk.kid LEFT JOIN ck ON ck.kid = mk.kid),
    mal AS (SELECT work_id, pos FROM mk WHERE
                 raw RLIKE '\\\\x{{FFFD}}' OR NOT raw RLIKE '^[\\\\p{{L}}\\\\p{{N}}(\\\\["#.\\']'
                 OR lower(raw) IN ('as','in','the','a','an','of','on','and','for','to','with','by','at','from','or','is')
                 OR raw RLIKE '^[0-9]{{1,4}}$' OR NOT raw RLIKE '\\\\p{{L}}'
                 OR length(raw) - length(replace(raw, '(', '')) <> length(raw) - length(replace(raw, ')', ''))
                 OR (raw RLIKE '(?i)^[0-9]+(st|nd|rd|th)[ -]century' AND CAST(regexp_extract(raw, '^([0-9]+)', 1) AS INT) > 21)),
    lp2 AS (SELECT work_id, pos, concat_ws(' ', slice(split(lower(raw), ' '), 1, 2)) AS pfx FROM mk WHERE size(split(raw, ' ')) >= 3),
    loops AS (SELECT work_id, pos FROM (SELECT *, count(*) OVER (PARTITION BY work_id, pfx) AS c, row_number() OVER (PARTITION BY work_id, pfx ORDER BY pos) AS r FROM lp2) WHERE c >= 6 AND r > 3),
    src AS (SELECT DISTINCT {T}.kw1322_kid0(n) AS kid0 FROM openalex.sources.sources_api LATERAL VIEW explode(array(display_name, host_organization_name)) t AS n WHERE n IS NOT NULL),
    vcand AS (SELECT mk.work_id, mk.pos, mk.kid0 FROM mk JOIN src ON src.kid0 = mk.kid0),
    venue AS (SELECT v.work_id, v.pos FROM vcand v JOIN openalex.works.openalex_works w ON w.id = CAST(substr(v.work_id, 2) AS BIGINT)
              WHERE v.kid0 IN ({T}.kw1322_kid0(w.primary_location.source.display_name), {T}.kw1322_kid0(w.primary_location.source.host_organization_name))),
    {BOILER_CTE}
    {TAXON_CTE}
    disc AS (SELECT mk.work_id, mk.pos FROM mk JOIN openalex.works.openalex_works dw ON dw.id = CAST(substr(mk.work_id, 2) AS BIGINT)
             WHERE mk.kid IN ({DISC}) AND COALESCE(dw.type, '') NOT IN ('book', 'reference-entry', 'libguides')
               AND instr(lower(COALESCE(dw.title, '')), replace(mk.kid, '-', ' ')) = 0),
    need AS (SELECT DISTINCT work_id FROM flagged WHERE hard = 0 OR cs IS NOT NULL),
    w AS (SELECT concat('W', CAST(w.id AS STRING)) AS work_id, w.type, COALESCE(w.title, '') AS title, COALESCE(w.abstract, '') AS abstract,
                 flatten(transform(w.authorships, x -> COALESCE(x.countries, array()))) AS inst_cc, sd.study_designs
          FROM openalex.works.openalex_works w LEFT JOIN openalex.works.works_study_design sd ON sd.work_id = w.id
          WHERE concat('W', CAST(w.id AS STRING)) IN (SELECT work_id FROM need)),
    judged AS (SELECT f.work_id, f.pos, f.raw, f.lp, f.kid,
                 CASE WHEN ml.pos IS NOT NULL THEN 'malformed' {BOILER_CASE} {TAXON_CASE} WHEN lo.pos IS NOT NULL THEN 'loop' WHEN di.pos IS NOT NULL THEN 'discipline_lazy' WHEN ve.pos IS NOT NULL THEN 'venue_name'
                      WHEN f.hard = 1 THEN 'purge'
                      WHEN f.hard = 0 AND exists(f.dual, d -> (d.kind = 'work_type' AND d.maps_to = w.type) OR (d.kind = 'study_design' AND array_contains(COALESCE(w.study_designs, array()), d.maps_to))) THEN 'dual_use_redundant'
                      WHEN f.cs IS NOT NULL AND NOT exists(f.cs, c -> array_contains(w.inst_cc, c.cc) OR concat(w.title, ' ', w.abstract) RLIKE c.rx) THEN 'country_no_evidence'
                      ELSE 'keep' END AS fate
               FROM flagged f LEFT JOIN w ON w.work_id = f.work_id
               LEFT JOIN mal ml ON ml.work_id = f.work_id AND ml.pos = f.pos LEFT JOIN loops lo ON lo.work_id = f.work_id AND lo.pos = f.pos
               LEFT JOIN venue ve ON ve.work_id = f.work_id AND ve.pos = f.pos LEFT JOIN disc di ON di.work_id = f.work_id AND di.pos = f.pos {BOILER_JOIN} {TAXON_JOIN})
    SELECT work_id, pos, raw, lp, kid, fate FROM judged"""),
 ("templated", f"""CREATE OR REPLACE TABLE {T}.{P}templated AS
    WITH ws AS (SELECT concat('W', CAST(id AS STRING)) AS work_id, primary_location.source.id AS src FROM openalex.works.openalex_works WHERE primary_location.source.id IS NOT NULL),
    k AS (SELECT DISTINCT m.work_id, m.kid FROM {T}.{P}mentions_kept m WHERE m.fate = 'keep'),
    sk AS (SELECT ws.src, k.kid, count(*) AS n FROM k JOIN ws ON ws.work_id = k.work_id GROUP BY 1, 2),
    sn AS (SELECT src, count(*) AS n FROM ws GROUP BY src HAVING count(*) >= 10000)
    SELECT sn.src, sn.n AS works, max(sk.n) AS top_kw_works, max_by(sk.kid, sk.n) AS top_kid FROM sn JOIN sk ON sk.src = sn.src GROUP BY sn.src, sn.n HAVING max(sk.n) >= 0.5 * sn.n"""),
 ("final", f"""CREATE OR REPLACE TABLE {T}.{P}mentions_final AS
    WITH k AS (SELECT m.*, row_number() OVER (PARTITION BY m.work_id ORDER BY m.pos) AS rk FROM {T}.{P}mentions_kept m WHERE m.fate = 'keep'),
    tw AS (SELECT concat('W', CAST(w.id AS STRING)) AS work_id FROM openalex.works.openalex_works w JOIN {T}.{P}templated t ON t.src = w.primary_location.source.id)
    SELECT k.work_id, k.pos, k.raw, k.lp, k.kid FROM k LEFT JOIN tw ON tw.work_id = k.work_id WHERE tw.work_id IS NULL OR k.rk <= 2"""),
 ("vocab", f"""CREATE OR REPLACE TABLE {T}.{P}vocab AS
    WITH k AS (SELECT kid, count(DISTINCT work_id) AS works_count FROM {T}.{P}mentions_final GROUP BY kid),
    forms AS (SELECT kid, raw, count(*) AS n FROM {T}.{P}mentions_final GROUP BY kid, raw),
    disp AS (SELECT kid, max_by(raw, n) AS display_name, count(*) AS n_forms FROM forms GROUP BY kid)
    SELECT concat('keywords/', k.kid) AS keyword_id, k.kid, d.display_name, k.works_count, d.n_forms FROM k JOIN disp d ON d.kid = k.kid WHERE k.works_count >= {a.floor}"""),
 ("work_keywords", f"""CREATE OR REPLACE TABLE {T}.{P}work_keywords AS
    WITH m AS (SELECT work_id, kid, min(pos) AS pos, max(lp) AS lp FROM {T}.{P}mentions_final GROUP BY work_id, kid),
    v AS (SELECT m.work_id, m.kid, m.pos, m.lp, v.keyword_id, v.display_name FROM m JOIN {T}.{P}vocab v ON v.kid = m.kid),
    r AS (SELECT *, row_number() OVER (PARTITION BY work_id ORDER BY pos) - 1 AS rnk, count(*) OVER (PARTITION BY work_id) AS n FROM v)
    SELECT work_id, keyword_id, display_name, CAST(rnk AS INT) AS position, {SCORE} AS score FROM r"""),
]
for i, (name, sql) in enumerate(STEPS):
    if i < a.from_step and i not in a.also: continue
    t0 = time.time(); print(f'[{i}] {name} ...', flush=True)
    dbx.sql(sql, warehouse=os.environ.get('DATABRICKS_WAREHOUSE_ID'), timeout_s=4 * 3600, verbose=False)
    print(f'[{i}] {name} done in {time.time() - t0:.0f}s', flush=True)
