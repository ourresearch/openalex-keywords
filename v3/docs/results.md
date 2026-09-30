# Results

We measured three separate things, and it matters which is which:

1. **Agreement with human labels** (the independent check): the author-keyword slices, KP20k, Inspec and MeSH.
2. **Imitation of the teacher**: F1 against Opus labels and Opus-judged precision. These tell you how well the student copies Opus, not whether Opus is right.
3. **The output itself**: vocabulary size, coverage and synonym hygiene.

## 1. Human labels (held-out test split, `code/eval/`)

The harness was built before this model existed, to compare earlier designs. It uses fixed 50 / 20 / 30 train / val / test splits (seed 1322). A score threshold is fit on train + val and applied to test. The student tagged every harness work with the shipped settings (B200, FP8, title + abstract + venue), and a keyword's score is exp(mean token log-prob). Keywords are compared as exact strings after canonicalisation.

| gold (test) | v3 student | best earlier design (retrieval + LLM chooser, never shipped) | earlier design without the LLM call | v2 (production until 2026-09-29) |
|---|---|---|---|---|
| **Author-keyword composite**, mean top-8 F1 over 5 slices | **0.178** | 0.218 | 0.202 | 0.09 |
| akw_field (6,180 works stratified by field) | 0.195 | 0.241 | 0.225 | |
| akw_pubmed | 0.240 | 0.314 | 0.288 | |
| akw_landing (publisher landing pages) | 0.174 | 0.215 | 0.198 | |
| akw_repo (repositories; 416 works, noisy) | 0.076 | 0.103 | 0.102 | |
| akw_kp20k (KP20k keyphrases as author keywords) | 0.206 | 0.218 | 0.196 | |
| **KP20k** F1@5, all keyphrases (2,000-doc sample of the test set) | **0.223** | 0.217 | | |
| **Inspec** F1@5, all keyphrases (500) | **0.279** | 0.183 | | |
| MeSH view a (4,728 PubMed works; name / entry-term match) | 0.223 | 0.588 (a chooser over MeSH descriptors) | | not comparable |

How to read this:

- v3 is about **2× the previous production keywords** on authors' own keywords. It beats every earlier design on the two published keyphrase benchmarks.
- It sits 2 to 4 points below the designs trained directly on author keywords. Those designs picked from a vocabulary harvested from author keywords, which exact-string scoring favours. v3 is a free-text model that words concepts its own way, and it imitates an indexer, not an author.
- The MeSH row does not measure quality. The student emits free text, which maps to MeSH descriptors only by exact name or entry term (3.3 descriptors per work).
- For the judge-gold row (F1 0.268) the problem is coverage: 61% of the student's keywords were never shown to that judge.
- TIBKAT (German GND headings) is skipped.

Published reference points for KP20k F1@5 (all keyphrases): unsupervised extractors about 0.15 to 0.20, zero-shot LLMs about 0.25 to 0.30, supervised generators about 0.31 to 0.38. v3 was never trained on KP20k.

## 2. Imitation of the teacher (held-out works)

**F1 against the teacher's labels** (13,338 held-out works, kid0 sets): LR-fixed student **0.568**, first v3 student 0.560 (paired difference +0.0086 ± 0.0027, positive in every slice).

**Opus-judged precision** ([judge prompt](../prompts/04_judge_keywords.md)): 1,000 held-out works (700 non-xpac, 300 xpac). The judge sees the pooled, shuffled union of every list and does not know which list a keyword came from.

| list | group | keywords / work | empty | judged precision | accepted / work |
|---|---|---|---|---|---|
| **v3 student (shipped, LR-fixed)** | non-xpac (700) | 6.91 | 9.0% | **0.863** | 5.96 |
| | abstract ≥ 300 chars (341) | 9.32 | 0.9% | 0.910 | 8.48 |
| | short abstract (88) | 6.15 | 4.5% | 0.813 | 5.00 |
| | title only (271) | 4.12 | 20.7% | **0.754** | 3.11 |
| | xpac (300) | 5.62 | 16.3% | **0.849** | 4.77 |
| first v3 student (same data, LR-schedule bug) | non-xpac | 6.89 | 9.1% | 0.856 | 5.90 |
| | title only | 4.05 | 20.7% | 0.729 | 2.95 |
| | xpac | 5.73 | 15.3% | 0.828 | 4.74 |
| teacher (Opus 5.5, prompt v3) | non-xpac | 7.57 | 8.7% | 0.918 | 6.95 |
| | abstract ≥ 300 chars | 10.76 | 0.9% | 0.953 | 10.25 |
| | title only | 4.03 | 19.6% | 0.825 | 3.32 |
| | xpac | 6.01 | 13.7% | 0.885 | 5.32 |
| previous student (prompt v2 labels) | non-xpac | 5.88 | 9.3% | 0.840 | 4.94 |

These are raw model lists, before the confidence cut and vocabulary rules. Verdicts: `data/student/judged_v3.jsonl` (first pass) and `judged_fix.json` (adds the LR-fixed student's new keywords). Scripts: `code/student/judge_union.py`, `judge_fix.py`.

**Blind head-to-head with v2** (peer review, 2026-09-26, first v3 student, final-vocabulary rules not yet applied): an Opus judge preferred v3's keywords on 390 of 399 works and v2's on 5. On author profiles it preferred v3 on 394 of 400.

Caveat for this whole section: Opus judges output taught by Opus. Blinding keeps self-preference from favouring one list over another, but it can raise every list's absolute precision. Section 1 is the independent check.

## 3. The output (shipped 2026-09-29)

| | |
|---|---|
| Works tagged by the student | 474,293,283 (all of OpenAlex, xpac included) |
| Raw keyword mentions | 3.52B (7.4 per work); 10.0% of works got an empty list because the model abstained |
| After the confidence cut and vocabulary rules | 3.34B mentions kept (98.0% of those above the cut) |
| Keywords (entities on ≥ 100 works) | **1,874,460** |
| Served work-keyword table | 420,920,976 works |
| Works with ≥ 1 keyword, of those in OpenAlex at build time | 418,624,601 of 472,275,003 (**88.6%**); non-xpac 299,199,127 of 326,474,770 (**91.6%**) |
| Keywords per tagged work | 5.6 on average (2,371,713,202 assignments); at most 65 |
| Synonyms | 362,272 ids merged into 213,043 headings; residual split 4.4% works-weighted (excluding deliberately isolated acronyms); merge precision ≈ 86% |

Coverage is lower than v2's 95% on purpose. v2 always assigned 5 keywords from each work's topics, even to records with nothing to describe. v3 abstains on bare identifiers, file names and specimen numbers.

## Known weaknesses

- **Title-only records** are the weakest group (judged precision 0.754).
- **Record families.** The templated-source cap keeps any one source from flooding counts, but large families still sit at the head of the vocabulary. Large Helical Device, on about 9.9M NIFS fusion-shot records, is an honest keyword. About 12.5M NIFS records are not flagged as xpac.
- **Homonyms share one heading** ("transformer" in machine learning and in power engineering). Sense labels are future work.
- **Invented taxonomy** on specimen records that carry only a species name. A GBIF-backbone check is future work.
- **Long-tail synonyms** remain; see [synonyms.md](synonyms.md).
