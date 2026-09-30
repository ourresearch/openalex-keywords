# Prompts

Every prompt the pipeline sends to a closed model, verbatim, with model id and parameters. All calls use the Anthropic Messages API with structured outputs (`output_config.format` = strict JSON schema).

| file | prompt | used for |
|---|---|---|
| [01_teacher_v3.md](01_teacher_v3.md) | Teacher prompt v3 (shipped) | Labelling the ~1M-work teaching set (944,707 requests; 943,769 answered, 938 refusals) |
| [02_teacher_v2_superseded.md](02_teacher_v2_superseded.md) | Teacher prompt v2 ("God" labels, superseded) | The first ~940K labels (2026-09-23) and the first students |
| [03_prompt_pilot_arms.md](03_prompt_pilot_arms.md) | Prompt pilot arms | choosing the teacher prompt |
| [04_judge_keywords.md](04_judge_keywords.md) | Keyword judge (strict indexer rubric) | Judged precision of students vs teacher (1,000 held-out works: 700 non-xpac + 300 xpac), the prompt pilot (`teacher/pilot2/judge_arms |
| [05_synonym_judge_v1.md](05_synonym_judge_v1.md) | Synonym judge v1 (names only, rounds 1-2) | Rounds 1 and 2: kNN candidate pairs from bge-m3 (169,147 pairs, then 200,000 wider-pass pairs) |
| [06_synonym_judge_v2.md](06_synonym_judge_v2.md) | Synonym judge v2 (searcher rubric, rounds 3-5, audits) | Round 3 head (62,218 pairs) and tail (478,919), acronym confirmations, generated alternative names (32,871 + 67,929), twins (1,553,192), word order (38,702), spacing variants (1,780), the merge-precision audits (4 x 400) and the residual split audit |
| [07_alternative_names.md](07_alternative_names.md) | Alternative-name generation | Generating synonym candidates embeddings miss (older adults / elderly, mpox / monkeypox, vitamin C / ascorbic acid) for the top 200K keywords; names found in the vocabulary become pairs for the v2 judge |
| [08_acronym_expansion.md](08_acronym_expansion.md) | Acronym expansion | Expanding 22,264 acronym-like keywords (>= 200 works) and 17,501 small ones (100-199 works); expansions found in the vocabulary go to the v2 judge |
| [09_uk_us_spelling.md](09_uk_us_spelling.md) | UK / US spelling confirmation | Confirming 8,673 rule-generated UK/US pairs (8,478 merged) |
| [10_variants.md](10_variants.md) | Spelling / hyphenation / acronym variant confirmation | Confirming 44,948 string-key variant pairs and reviewer-found acronym pairs |
| [11_context_check_v1.md](11_context_check_v1.md) | Context check of high-impact merges (conservative) | Every merge that moves a keyword on >= 10K works (1,575 + 1,018 + 3,446 + 134 checks); rejected merges are kept apart |
| [12_context_check_v2.md](12_context_check_v2.md) | Context re-check with the v2 searcher rubric | Re-verifying the 917 merges the conservative context check had rejected (230 restored) |
| [13_doctype_exclusions.md](13_doctype_exclusions.md) | Document-type / study-design keyword exclusions | Building `keyword_doctype_verdicts_v1` (65,973 candidate keywords; the purge list) |
| [14_dual_use.md](14_dual_use.md) | Dual-use exceptions to the purge | Drafting `keyword_dual_use_v1` (2,812 rows; approved by hand 2026-09-25): purge keywords dropped only where the work's own type or design already says it |
| [15_countries.md](15_countries.md) | Country evidence table | Names, adjectives and native names per country (`data/vocabulary/countries |
| [16_keyword_types_audit.md](16_keyword_types_audit.md) | Keyword type audit (diagnostic) | Diagnosing non-aboutness keywords (discipline labels, routine methods, record kinds) that led to the v3 clauses |
