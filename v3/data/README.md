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
| `vocabulary/kw1322v3n_vocabulary.tsv.gz` | **the shipped vocabulary**: keyword_id, display_name, works_count (1,874,460 rows) |
| `vocabulary/*.tsv`, `countries.json` | rule tables, see docs/vocabulary.md |
| `synonyms/synmap9_final.jsonl.gz` | **the shipped synonym map** `{kid, canon}` (applied after the plural fold) |
| `synonyms/hand_decisions.jsonl`, `review500.json` | the 500 hand-reviewed pairs with prior verdict counts, and the decisions |
| `synonyms/judgments/` | every judge verdict file the map builder reads (gzipped), plus the per-channel merge maps; twins verdicts in LARGE_FILES |
| `synonyms/audits/` | residual-split audits on successive vocabularies, incl. the shipped one (`audit_kw1322v3n_.json`, `audit_n29.log`, `audit_n31.log`) |

Formats of the verdict files: name judges `{a, b, cos, merge, keep}` (`keep` = preferred heading); context checks `{kid, canon, verified}`; UK/US and variants `{a, b, merge}`; acronyms `{kid, acronym, expansion, exp_kid}`; alternative names `{kid, names}`.
