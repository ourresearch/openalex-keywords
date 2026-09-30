# Vocabulary build

`code/vocabulary/build_vocab.py` runs 14 SQL steps on Databricks and turns 3.52B raw keyword strings from the tagger into 1,874,460 keyword entities with per-work assignments. Run it with `--from-step N` to resume. The shipped build was `--source keywords_student_v3fix --prefix kw1322v3n_ --boilerplate --synmap kw1322v3_synmap9 --min-lp -1.2 --floor 100`.

## Identity

- **kid0** (`code/common/kid.py`, also registered as a Python UDF): a pure string function. It applies NFKC, lowercases, turns hyphens, underscores and slashes into spaces, collapses spaces and strips edge punctuation. It then builds a slug the same way the OpenAlex text API does: Greek letters spelled out, ASCII fold, and every run of non-alphanumerics becomes `-`. Strings with letters outside the Latin and Greek scripts keep their Unicode letters, because folding would erase them.
- **Plural fold** (step `fold`): a kid0 ending in *s* merges with its singular (-ies → -y, -(ss|sh|ch|x|z)es, -s) **only if** that singular exists with ≥ 5% of the plural's mentions. The group's id is its most common member. Acronyms written in capitals (ALS, NAS) are exempt. A blind "strip the final s" rule turned apoptosis into `apoptosi`, diabetes mellitus into `diabetes-mellitu` and Paris into `pari`.
- **Synonym map** (`--synmap`, [synonyms.md](synonyms.md)): applied after the fold. A keyword's final id is `synmap[fold[kid0]]`.
- **Display name:** the most common surface form among a keyword's mentions after all rules.
- **Keyword id:** `keywords/<kid>`; served as `https://openalex.org/keywords/<kid>`.

## Rules, in order

| rule | what it removes | mentions (share of 3.41B after the cut) |
|---|---|---|
| confidence cut `--min-lp -1.2` | mentions whose mean token log-prob is below −1.2 (score < 0.30), applied before everything else | 108.1M of 3,520.6M raw (3.1%) |
| `malformed` | U+FFFD, a first character that is not a letter or digit, bare stopwords, bare years, no letters at all, unbalanced parentheses, centuries above the 21st | 5.88M (0.17%) |
| `boilerplate` | a keyword stamped on ≥ 90% of the works of a source with ≥ 10K works (41 source-keyword pairs, e.g. UNITE species-hypothesis sets, BacDive form fields); Large Helical Device is exempt | 42.7M (1.25%) |
| `loop` | repetition loops: when ≥ 6 keywords on one work share a two-word prefix, keep the first 3 | 7.49M (0.22%) |
| `discipline_lazy` | bare discipline labels (medicine, biology, botany, ... 68 terms), except on books, reference entries and library guides, or when the title names the discipline | 3.00M (0.09%) |
| `venue_name` | the keyword is the work's own source or publisher name | 1.36M (0.04%) |
| `purge` | keywords that only label the kind of document or study (65,973 candidates judged; 2,812 purge keywords), e.g. "book review", "erratum", "randomized controlled trial" as a label | 5.29M (0.15%) |
| `dual_use_redundant` | purge keywords that are also real subjects ("peer review", "systematic review") are dropped only where the work's own type or study design already says it | 0.04M |
| `country_no_evidence` | a country keyword kept only if the country's name, adjective or native name appears in the title or abstract, or an author's institution is in that country | 2.61M (0.08%) |
| kept | | 3,344.1M (98.0%) |

Two more steps come after the fates:

- **Templated sources, capped at 2** (step `templated`): 64 sources with ≥ 10K works whose single most common keyword is on ≥ 50% of their works keep only each work's first 2 keywords. These are record families such as NIFS fusion shots (41.2M records), GBIF occurrences and plant genetic resources accessions. The keywords stay honest, but these families no longer flood corpus statistics.
- **Floor:** a keyword becomes an entity only if it is on ≥ 100 distinct works after all of the above. Assignments of keywords below the floor are dropped.

Output tables: `<prefix>vocab` (keyword_id, kid, display_name, works_count, n_forms) and `<prefix>work_keywords` (work_id, keyword_id, display_name, position, score = round(exp(lp), 3)).

## Rule tables (data/vocabulary/)

| file | rows | what |
|---|---|---|
| `doctype_verdicts.tsv` | 65,973 | Opus verdicts on document-type / study-design keyword candidates ([prompt](../prompts/13_doctype_exclusions.md), code `code/vocabulary/doctype_rules/`) |
| `dual_use.tsv` | 2,812 | which purge keywords are also subjects ([prompt](../prompts/14_dual_use.md)); approved by hand 2026-09-25 |
| `purge.tsv` | 2,812 | the purge list as applied (kid, keyword, kind, maps_to, dual_use) |
| `countries.json`, `country_kids.tsv` | 1,479 | country aliases and evidence regexes ([prompt](../prompts/15_countries.md)) |
| `boilerplate_source_keywords.tsv` | 41 | the stamped source-keyword pairs |
| `templated_sources.tsv` | 64 | sources capped at 2 keywords per work |
| `kw1322v3n_vocabulary.tsv.gz` | 1,874,460 | the shipped vocabulary: keyword_id, display_name, works_count (counted before served-table filtering) |

## What was tried and rejected

- **Taxon-evidence rule** (`--taxon-evidence`): on bare species-name records from biodiversity sources, keep a keyword only if its words are in the title. It mostly deleted correct higher taxa (moths / Lepidoptera / Crambidae for *Udea lutealis*), so it is off. A fix against the GBIF backbone is future work.
- **Confidence cut alone vs floor alone:** on the eval set the 100-work floor already brought list lengths down to the authors' own distribution (mean 5.2 vs 5.3). The cut adds about 1 point of precision overall and more on title-only records, so both are on.
