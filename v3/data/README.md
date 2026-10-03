# Data

Works are identified by OpenAlex id (`W…`). Work text is not included; fetch it from the OpenAlex API or snapshot. Anything too large for git is listed in [LARGE_FILES.md](LARGE_FILES.md).

| path | what |
|---|---|
| `teacher/sample_ids.tsv.gz` | the 1M-work teaching sample: work_id, is_xpac, held_out |
| `teacher/eval_ids.txt` | the 14,500 held-out ids (10,000 non-xpac + 4,500 xpac) |
| `teacher/teacher_labels_eval.jsonl` | Opus 5.5 prompt-v3 labels for the held-out works: id, model, effort, keywords (null on refusal), error |
| `teacher/prompt_pilot/` | the 620-work prompt pilot: work groups, each arm's keywords (`arm_*.jsonl`, prompts in `prompts/03`), pooled judge verdicts (`judged.jsonl`), scores, keyword-type audits (`types*.jsonl`) |
| `student/student_v3fix_eval_outputs.jsonl` | the shipped student's keywords on the 13,338 held-out works (HF generation, bf16) |
| `student/student_v3_eval_outputs.jsonl` | the same for the first v3 student (LR-schedule bug), for comparison |
| `student/judged_v3.jsonl`, `judged_fix.json`, `score_v3.json` | Opus judge verdicts per work and keyword (kid0 → accepted) on 1,000 held-out works; per-group scores |
| `student/train_v3fix.log` | training log: loss and learning rate every 50 steps |
| `tagging/ledger_v3fix.jsonl` | one line per corpus shard: works, fetch / generate / total seconds, bytes, GPU, quantisation |
| `tagging/judged_retag_fix.jsonl` | judge verdicts on the corpus run's own output for the 1,000 judged works |
| `eval/splits/` | train / val / test assignment per gold (seed 1322) |
| `eval/results/` | the shipped student's and v2 production's score files per gold |
| `eval/leaderboard.md` | every arm ever scored on the harness, incl. the retired designs |
| `eval/golds/` | label files for the MeSH gold (`j1306/`: 4,728 PubMed works with their MeSH headings, plus the MeSH 2026 descriptor table) and the judge gold v1.2 (`j1292/` + `j1322/`: 1,198 works, relevant keywords, candidate pools); ids and labels only, no titles or abstracts; files over 5 MB gzipped |
| `eval/student_v3fix_harness_tags.json.gz` | the student's keywords and mean log-probs on all 18,706 harness works |
| `vocabulary/vocabulary_20261003.tsv.gz` | **the vocabulary as served on 2026-10-03 (v3.1)**: keyword_id, display_name, works_count over live works (1,882,088 headings; 1,881,484 carry at least one work). Includes the synonym-audit and sense-split headings. A heading can drift below 100 works as works are added, merged or deleted |
| `vocabulary/kw1322v3n_vocabulary.tsv.gz` | the vocabulary as built for v3.0: keyword_id, display_name, works_count (1,874,460 rows) |
| `vocabulary/*.tsv`, `countries.json` | rule tables, see docs/vocabulary.md |
| `synonyms/synmap10_final.jsonl.gz` | **the synonym map served since 2026-10-01 (v3.1)** `{kid, canon}`: synmap9 minus 5,016 merges the audit judged wrong ([docs/synonym_audit.md](../docs/synonym_audit.md)) |
| `synonyms/synmap10_breaks.jsonl` | the 5,016 broken merges with works moved and the judge's reason |
| `synonyms/audit_v31/` | the audit: every synmap9 merge, string-judge verdicts, display-name checks, re-judge, long-COVID work verdicts |
| `synonyms/synmap9_final.jsonl.gz` | the v3.0 synonym map `{kid, canon}` (applied after the plural fold) |
| `synonyms/hand_decisions.jsonl`, `review500.json` | the 500 hand-reviewed pairs with prior verdict counts, and the decisions |
| `synonyms/judgments/` | every judge verdict file the map builder reads (gzipped), plus the per-channel merge maps; twins verdicts in LARGE_FILES |
| `synonyms/audits/` | residual-split audits on successive vocabularies, incl. the shipped one (`audit_kw1322v3n_.json`, `audit_n29.log`, `audit_n31.log`) |

Formats of the verdict files: name judges `{a, b, cos, merge, keep}` (`keep` = preferred heading); context checks `{kid, canon, verified}`; UK/US and variants `{a, b, merge}`; acronyms `{kid, acronym, expansion, exp_kid}`; alternative names `{kid, names}`.
| `senses/` | keyword sense splits (v3.1): sense map, gold and gate labels, per-sense gate results, shipped per-work senses; see [senses/README.md](senses/README.md) and [docs/senses.md](../docs/senses.md) |
| `alternatives/` | keyword synonyms for search and autocomplete (v3.1): the served `display_name_alternatives`, bake-off, ownership and eval aggregates; see [alternatives/README.md](alternatives/README.md) and [docs/alternatives.md](../docs/alternatives.md) |
| `eval/outside_judge/` | the outside judge (GPT-6 Astra, v3.1): the 400-work sample (ids and keyword lists, no text), every verdict, tables; `postswap/` = the launch post's search examples. See [docs/outside_judge.md](../docs/outside_judge.md) |
