# Prompts

Every prompt the pipeline sends to a closed model, verbatim, with model id and parameters. Prompts 01 to 20 use the Anthropic Messages API (Claude Opus 5.5) with structured outputs (`output_config.format` = strict JSON schema). The v3.1 additions also use models from other providers, named in each file: Jev (21), OpenAI GPT-6 Astra (22, 26 to 28; directly or through OpenRouter) and GPT-6.1 Sol (23 to 25, through OpenRouter).

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
| [17_merge_audit.md](17_merge_audit.md) | Synonym merge audit (v3.1) | Judging every non-trivial synmap9 merge same / close / wrong (332,373 judged; 5,012 wrong broken in synmap10) |
| [18_display_name_check.md](18_display_name_check.md) | Heading display-name check (v3.1) | 39,221 headings named after a merged member (122 wrong) |
| [19_long_covid_check.md](19_long_covid_check.md) | Long COVID work-level check (v3.1) | Which source strings merged into long COVID name the illness and which the post-pandemic period |
| [20_sense_discovery.md](20_sense_discovery.md) | Sense discovery judge (v3.1) | Whether a keyword names one subject or distinct senses (3,000 biggest candidates + 996-keyword tail sample) |
| [21_sense_classifier_jev.md](21_sense_classifier_jev.md) | Per-work sense classifier (Jev, v3.1) | Which sense each work uses: 15.9M corpus pairs, the gold set, and new works every night |
| [22_sense_check_judge.md](22_sense_check_judge.md) | Per-work sense judge, with GPT-6 Astra as second judge (v3.1) | Gold labels, the field-rule check, the per-sense gate (7,877 moves) and the near-miss recheck (1,650) |
| [23_alternative_names_writer.md](23_alternative_names_writer.md) | Alternative-name writer (GPT-6.1 Sol with Wikidata hints, v3.1) | `display_name_alternatives` for every keyword: 5,091,086 alternatives on 1,701,306 keywords after the one-owner step |
| [24_alternatives_bakeoff_judge.md](24_alternatives_bakeoff_judge.md) | Alternative-name bake-off judge (Opus 5.5 and GPT-6 Astra, v3.1) | Choosing the writer: 9,103 pooled candidates for 713 keywords, judged blind |
| [25_alternatives_owner_judge.md](25_alternatives_owner_judge.md) | Alternative-name owner judge (GPT-6.1 Sol, v3.1) | One owner per contested alternative (573,931 strings; 17,030 homonyms keep several) |
| [26_outside_judge_keyword_accuracy.md](26_outside_judge_keyword_accuracy.md) | Outside judge: per-keyword accuracy (GPT-6 Astra, v3.1) | The launch numbers: 91.1% of new keywords accurate vs 43.5% of old on 400 random works |
| [27_outside_judge_preference.md](27_outside_judge_preference.md) | Outside judge: old vs new head-to-head (v3.1) | New preferred on 395 of 400 works |
| [28_outside_judge_on_topic.md](28_outside_judge_on_topic.md) | Outside judge: is this work about the subject? (v3.1) | The launch post's search examples (antimicrobial resistance 81% / 74% on topic; remote work) |
