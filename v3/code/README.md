# Code

| folder | stage | main entry points |
|---|---|---|
| `common/` | shared helpers | `kid.py` (keyword ids), `dbx.py` / `dbxr.py` (Databricks SQL through the CLI) |
| `teacher/` | 1. teacher labels | `prompt_v3.py`, `build_sample.py`, `judge_batch.py`, `driver.sh`; `pilot2/` = the prompt pilot |
| `student/` | 2. student | `student_gen_data.py`, `student_gen_train_ddp.py`, `modal_student.py`, `judge_union.py`, `judge_fix.py` |
| `tagging/` | 3. corpus tagging | `build_input.py`, `modal_corpus_v3.py`, `corpus_run_v3.py`, `load_v3.py`; `probe.py`, `modal_confidence.py`, `conf_analyze.py` (studies) |
| `vocabulary/` | 4. vocabulary | `build_vocab.py`; `doctype_rules/` (purge and dual-use tables); `build_countries.py` |
| `synonyms/` | 5. synonyms | `modal_embed_knn.py`, `judge_pairs2.py`, `altnames.py`, `acronyms.py`, `uk_us.py` + `judge_ukus.py`, `variants.py`, `verify_hi.py`, `verify_v2.py`, `build_synmap3.py`, `apply_hand.py`, `audit.py`; `runs/` = the command records; `review_tool/` = the hand-review page |
| `synonym_audit/` | synonym audit (v3.1) | `merges.py`, `audit_merges.py`, `judge_display.py`, `rejudge.py`, `lc_judge.py`, `make_synmap10.py`, `rebuild_vocab.sh` |
| `senses/` | keyword senses (v3.1) | see [senses/README.md](senses/README.md) |
| `alternatives/` | alternative names (v3.1) | see [alternatives/README.md](alternatives/README.md) |
| `outside_judge/` | outside judge (v3.1) | `pull.py`, `fetch_text.py`, `judge_gpt.py`, `analyze.py`, `postswap_judge.py` |
| `eval/` | 7. human-label harness | `golds.py`, `splits.py`, `score.py`, `fetch_ext.py`, `student_arm.py`, `leaderboard.py` |
| `publish/` | 6. served tables | `gen_build.py` |

These are the scripts that produced the shipped keywords, copied from the working directory of OpenAlex job #1322. For the release we changed only import paths (everything shared now comes from `common/`), replaced hard-coded Databricks warehouse ids with `DATABRICKS_WAREHOUSE_ID`, and pointed data paths at a per-stage `data/` folder. Superseded variants (`build_synmap.py`, `build_synmap2.py`, `judge_pairs.py`, `student_gen_train.py`) are kept because the run records call them. Scripts that call the Anthropic API print cost estimates from hard-coded list prices; the logic is unchanged.

`check_level_a.py` recomputes the headline claims from the released files (no GPU, no API calls).

The v3.1 folders come from OpenAlex jobs #1465 (synonym audit), #1476 (senses), #1464 (alternative names) and #1322 (outside judge), with the same release edits: shared helpers from `common/`, environment variables for warehouse ids, keys and local paths, and data paths pointed at `data/`.
