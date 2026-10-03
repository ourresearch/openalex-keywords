# The synonym audit (v3.1)

A day after launch we found merges in the shipped synonym map that joined unrelated subjects: "dark matter" had been folded into "diabetes mellitus" through the acronym DM, so 28,452 works with "dark matter" in the title carried a diabetes keyword. We then audited every merge in the map, not a sample. The audit judged **5,012 merges wrong**, and four more chosen by hand on 2026-09-29 were judged wrong too. All **5,016** were broken. The result, synmap10 (`data/synonyms/synmap10_final.jsonl.gz`), folds **357,259 keyword ids into 211,451 headings** and is what OpenAlex serves since 2026-10-01.

## What went wrong

The map builder isolates ambiguous acronyms ([synonyms.md](synonyms.md), guard 4). It decided whether a keyword is an acronym from its most common display form: two or more capitals, no spaces. That rule was added on 2026-09-29 to stop "elderly" counting as an acronym. But many short ids are mostly written in lower case or as chemical symbols: `dm` displays as "dm", `ir` as "Ir" (iridium), `er` as "Er", `pd` as "Pd". They were never isolated, so each one bridged its two meanings.

| merge in synmap9 | works moved | why it is wrong |
|---|---:|---|
| neon → National Ecological Observatory Network | 242,384 | the element vs the observatory network (NEON) |
| information retrieval → insulin resistance | 127,201 | IR read the wrong way |
| dark matter → diabetes mellitus | 99,375 | DM |
| selenium → socioeconomic status | 70,934 | Se / SES |
| estrogen receptor → emergency department | 62,430 | ER |
| renin-angiotensin system → rheumatoid arthritis | 60,573 | RAS / RA |
| pea → Pisum sativum | 59,162 | PEA is usually phenylethylamine and other acronyms |
| peritoneal dialysis → Parkinson's disease | 53,454 | PD |

Not every bad merge ran through a short token: civil-military relations → cardiac MRI went through "CMR", and consumption → tuberculosis through an old name for the disease. A rule alone cannot catch those, which is why we audited every merge.

The launch audit had missed this because it sampled 400 merges by keyword, and its judge saw display names. The few high-volume acronym bridges were rarely drawn ("merge precision ≈ 86%" in [results.md](results.md) is that earlier measure).

## How the audit worked

1. **The unit is one merge**: a source keyword folded into a heading, with the works it moved (`code/synonym_audit/merges.py` → `data/synonyms/audit_v31/merges.jsonl.gz`). synmap9 made 361,126 merges in the shipped build, moving 354.0M work assignments. 333,938 of them are not trivial (not just a hyphen or plural variant).
2. **A string judge** (Opus 5.5, effort low, 50 merges per call, [prompt 17](../prompts/17_merge_audit.md)) calls each merge same, close or wrong. "Close" (narrower, broader or closely related, but nearly every work would still suit someone searching the heading) is kept. Order: every merge that moved ≥ 3,000 works, then a works-weighted sample of 1,500 of the rest to size the tail, then the whole tail the same night.
3. **Display names.** 39,221 headings take their display name from a merged member. A wrong one hides a bad merge from the string judge, because the judge compares the member with that display name (`cardiac-magnetic-resonance-imaging` was displayed as "civil-military relations"). A second pass checks each display name against the heading id ([prompt 18](../prompts/18_display_name_check.md)), and the members of the wrong ones are re-judged against the id words.
4. **Long COVID** was checked at the level of works ([prompt 19](../prompts/19_long_covid_check.md), 40 works per source string): "post-COVID-19 era", "-pandemic" and "-period" name a time, not the illness (39 or 40 of 40 off topic for the first two).
5. **synmap10** = synmap9 minus every merge judged wrong (`code/synonym_audit/make_synmap10.py`). A broken member keeps its own id and becomes its own heading if it has ≥ 100 works. The post-COVID era strings move to the existing `post-pandemic-period` heading, and bare "post-COVID" stands alone.

| pass | merges judged | judged wrong | work assignments in the wrong ones |
|---|---:|---:|---:|
| merges moving ≥ 3,000 works | 18,692 | 448 | 4,447,453 |
| works-weighted sample of the rest | 1,500 | 19 | 23,416 |
| the rest (tail) | 312,181 | 4,481 | 2,611,890 |
| display names (heading id vs display) | 39,221 | 122 | |
| members of the 122 re-judged against the id | 298 | 82 | |

332,373 of the 333,938 non-trivial merges got a verdict. Of the judged merges, 80% were "same", 19% "close" and 1.5% "wrong", but the wrong ones were concentrated among the biggest merges (2.5% of the assignments moved by merges of ≥ 3,000 works).

The breaks: 4,929 from the string audit, 82 from the re-judge and 5 long-COVID strings, together 5,016 (`data/synonyms/synmap10_breaks.jsonl`, with the judge's reason). Four of them overrule hand decisions from 2026-09-29: consumption → tuberculosis, brain injury → traumatic brain injury, ultrasound → ultrasonography and industries → manufacturing industry. Every other hand decision stands.

## What changed in OpenAlex

- The vocabulary was rebuilt under synmap10 with the v3.0 rules unchanged (`code/synonym_audit/rebuild_vocab.sh`), over the corpus tags plus the works tagged after the corpus run.
- Only works whose keyword list changed were rewritten: 8,462,750 works for synmap10 and 163,429 for the four hand merges. Dark matter is its own keyword again (91,452 works on 2026-10-02).
- 7,547 headings were added (broken members that have ≥ 100 works of their own) and 1,153 display names fixed. No heading that unchanged works still carry was removed.
- The map builder now has an opt-in guard for this failure (`build_synmap3.py --acronym-guard`: every hyphen-free id of ≤ 5 characters counts as an acronym). It is off by default so that the v3.0 map still rebuilds exactly. On its own it would fix only part of the problem: it fixes 1.6M of the 4.1M wrong assignments in the big merges and breaks 117 good merges, so the audited break list is the fix and the guard is a safety net for future builds.

## Limits

- The string judge is strict. Some breaks are judgment calls (humour → humor, labour → labor). Breaking a correct merge only leaves two headings for one concept, the state before the merge, so we preferred false alarms to misses.
- 1,565 non-trivial merges got no verdict and are kept.
- Merges the audit called "close" are kept. They are broader / narrower pairs that searchers rarely mind, but they are not exact synonyms.

## Files

| path | what |
|---|---|
| `data/synonyms/synmap10_final.jsonl.gz` | **the synonym map OpenAlex serves**: `{kid, canon}`, 357,259 rows |
| `data/synonyms/synmap10_breaks.jsonl` | the 5,016 broken merges: `{src, heading, works, why, from}` |
| `data/synonyms/audit_v31/merges.jsonl.gz` | every synmap9 merge with works moved, an example raw string, the heading's name and works |
| `data/synonyms/audit_v31/audit_judged.jsonl.gz` | string-judge verdicts: `{key, src, heading, works, frame, v, why}` |
| `data/synonyms/audit_v31/audit_frame.json` | the sampling frame (sizes and the 1,500 sampled merges) |
| `data/synonyms/audit_v31/display_from_member.json.gz`, `display_judged.jsonl.gz`, `display_wrong.json` | the display-name pass |
| `data/synonyms/audit_v31/rejudge_judged.jsonl` | the 298 members re-judged against the heading id |
| `data/synonyms/audit_v31/long_covid_judged.jsonl` | the work-level long-COVID verdicts (work ids, no text) |
| `code/synonym_audit/` | the scripts, in run order: `merges.py`, `audit_merges.py`, `judge_display.py`, `rejudge.py`, `lc_pull.py`, `lc_judge.py`, `make_synmap10.py`, `rebuild_vocab.sh` |

`python3 code/synonym_audit/make_synmap10.py` rebuilds synmap10 from released files alone and checks it byte for byte against the released map.
