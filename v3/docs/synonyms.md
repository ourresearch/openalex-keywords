# Synonym merging

A free-text tagger names one concept several ways: "elderly" and "older adults", "IL-6" and "interleukin-6", "tumour" and "tumor". A vocabulary is only usable with one heading per concept, so reviewers of the first build treated split synonyms as a stop-ship defect. The final map (`data/synonyms/synmap9_final.jsonl.gz`, `{kid, canon}`) folds **362,272 keyword ids into 213,043 headings**. It is applied in the vocabulary build after the plural fold.

## How a merge is decided

1. **Candidates** come from channels that each catch what the others miss.
2. **An Opus 5.5 judge** reads 100 pairs per call as bare names ([v1 rubric](../prompts/05_synonym_judge_v1.md), then the searcher-oriented [v2 rubric](../prompts/06_synonym_judge_v2.md)). For each pair it decides merge or not and names the better heading.
3. **Context checks.** Every merge that would move a keyword on ≥ 10K works is re-judged with 3 example titles and venues per side ([conservative](../prompts/11_context_check_v1.md), then [v2](../prompts/12_context_check_v2.md)). A rejected merge becomes a keep-apart pair.
4. **Map building** (`build_synmap3.py`) runs union-find over the merge edges with three guards:
   - components of ≤ 3 keywords take the judge's preferred heading;
   - larger components become a star around the most-used keyword, and only members within 2 judged hops of that hub join, so chains like A~B~C cannot pull A and C together;
   - **keep-apart:** each pair's final verdict comes from the most authoritative source that judged it (name judges < v2 name judges < precision audits < context checks < v2 context re-check), and a pair judged different never merges;
   - **ambiguous acronyms:** an acronym (display name with ≥ 2 capitals, no spaces) whose expansions fall in ≥ 2 different concept groups is isolated. TAM (tumor-associated macrophages vs technology acceptance model), NBI, AML, LDH and 1,354 others stay separate from every expansion.
5. **Hand review** (`apply_hand.py`) of the 500 most-used contested pairs, then a final rebuild and audits.

## The five rounds (2026-09-27 to 2026-09-29)

| round | candidates | judged | result |
|---|---|---|---|
| 1 | bge-m3 embeddings of all 2.35M keywords (`modal_embed_knn.py`, 1 H100, K = 8); pairs with cos ≥ 0.90 and both sides ≥ 1,000 works, or cos ≥ 0.93 and both ≥ 300 | 169,147 pairs, v1 judge; 1,575 high-impact merges context-checked (158 rejected, e.g. hydrogen fueling ≠ refueling, COVID-19 ≠ COVID-19 pandemic) | 74,898 merges |
| 2 | UK → US spelling rules (`uk_us.py`); string-key variants plus reviewer acronym pairs (`variants.py`); a wider kNN pass (K = 30, cos ≥ 0.85) filtered to pairs that rarely co-occur on a work | 8,673 UK/US (8,478 merged; Haemophilus and Labour Party kept); 44,948 variants; 200,000 wider-pass pairs; 1,018 context checks | 126,316 keywords into 104,330 headings |
| 3 | v2 judge on the remaining head pairs (62,218, both ≥ 1,000 works) and tail pairs (478,919, cos ≥ 0.88); acronym expansion (`acronyms.py`, 22,264 acronyms); generated alternative names for the top 50K keywords (`altnames.py`) | 4,465 + 108,244 merges; 6,123 acronym merges; 19,731 of 32,871 alt-name pairs; 3,446 context checks (862 rejected) | 247,442 into 167,891 |
| 4 | "twins": a small keyword within cos 0.82 of a ≥ 10× larger heading (bge-m3, K = 30); small acronyms (100 to 199 works) | 1,553,192 pairs; 2,665 acronym pairs; 134 context checks | 325,934 into 191,546 |
| 5 | word-order variants (38,702 pairs); alt names for ranks 50K to 200K (67,929 pairs); v2 re-check of the 917 context rejections (230 restored); irregular plurals (mouse / mice, 494); keep-apart and acronym guards (synmap6 to synmap8) | 400-merge precision audits after each map | 360,747 into 212,207 (synmap8) |
| hand | the 500 most-used contested pairs (conflicting or "still split" verdicts, ranked by the smaller side's works); spacing and hyphen variants (schoolchildren / school children) | Jason decided 180 (129 merge, 20 apart, 31 skipped as outside his expertise); Claude decided the other 351 by his rules (295 merge, 56 apart); Opus confirmed 1,112 of 1,780 spacing pairs | synmap9: 362,272 into 213,043 |

The hand rules were: the literature's usual term wins; otherwise the shorter, more conversational one. English species names win unless the Latin dominates. Broader / narrower pairs and acronym collisions stay apart. The decisions are in `data/synonyms/hand_decisions.jsonl` (`d`: A = merge into a, B = merge into b, X = keep apart; `by`: jason or claude), and the page used for them is in `code/synonyms/review_tool/`.

## Audits

- **Residual splits (the stop rule, `audit.py`).** Sample 1,000 keywords weighted by works. Find candidate synonyms that are still separate headings through two independent channels (generated alternative names plus embedding neighbours). Have the v2 judge confirm them. Result:

  | vocabulary | share of sampled keywords with a separate synonym | works-weighted |
  |---|---|---|
  | before round 3 (`g`) | 32.3% | 49.8% |
  | after round 3 (`h`) | 13.8% | 8.4% |
  | shipped (`n`), raw, seeds 29 / 31 | 10.5% / 11.7% | 11.2% / 11.6% |
  | shipped (`n`), seed 31, not counting deliberately isolated ambiguous acronyms and hand keep-aparts | | **4.4%** |

  Most of the raw residual is pairs we chose to keep apart. The biggest is "neutral beam injection" / NBI, where NBI also means narrow-band imaging. Clear misses left: China / People's Republic of China, 19th century / 1800s, some archaic or debatable aliases.
- **Merge precision.** Sample 400 merges weighted by works; the v2 judge says whether each pair is the same concept. Round 5 went from 91.5% down to **85.8%** on synmap8 (`data/synonyms/judgments/judged_precision*.jsonl.gz`). The fall came from merging more borderline pairs, and most flagged merges are judgment calls (Lower / Early Cretaceous). Judges disagree with each other on 10 to 15% of borderline pairs, so beyond this point the audit oscillated between split errors and wrong merges. That is where we stopped and handed the contested pairs to a person.

## Reproducing the map

Recomputing the verdicts costs API calls. Rebuilding the map from the released verdicts does not:

1. `data/synonyms/judgments/*.jsonl.gz` holds every verdict file `build_synmap3.py` reads. The exception is the 1.55M round-4 twin verdicts (`judged_twins.jsonl`, 36.6 MB gzipped), listed in [../data/LARGE_FILES.md](../data/LARGE_FILES.md).
2. `build_synmap3.py` also reads works counts from the intermediate vocabularies (`vocab*.jsonl`, `vocab_all.jsonl`), which decide hub choice. These are also listed in LARGE_FILES.
3. The exact command sequence, including one-off glue steps that built candidate files, is in `code/synonyms/runs/` (see its README).

We had no time to re-run the rebuild end to end from the released files. Treat `synmap9_final.jsonl.gz` as the authoritative map.
