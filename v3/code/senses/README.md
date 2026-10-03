# Keyword senses (homonym splits)

The scripts that produced the v3.1 sense splits, in run order. The method and results are in [docs/senses.md](../../docs/senses.md); the prompts are [20](../../prompts/20_sense_discovery.md), [21](../../prompts/21_sense_classifier_jev.md) and [22](../../prompts/22_sense_check_judge.md).

They are copied from the working directories of OpenAlex jobs #1465 (discovery and the failed field rule) and #1476 (everything after). For the release we changed only import paths (`dbx.py` and `kid.py` now come from `../common/`), replaced machine paths with environment variables or relative paths, renamed `gen_ship_1476.py` / `gen_ship2_1476.py` to `gen_ship.py` / `gen_ship2.py`, and added a command line to docstrings that lacked one. Scripts that call a paid API print cost estimates from hard-coded list prices; the logic is unchanged.

## Setup

- Run every script from one working directory. Scripts read and write `data/` relative to it; gunzip the files from [`data/senses/`](../../data/senses) into it to start from our results instead of recomputing them.
- `DATABRICKS_WAREHOUSE_ID` (and a configured `databricks` CLI) for every script that touches tables. The tables are OpenAlex's own (`openalex.works.work_keywords_v2`, `openalex.works.openalex_works`, `openalex.common.keywords_v2`); intermediate tables go to `openalex.transient.kw1465_*` / `kw1476_*`.
- `CLAUDE_API_KEY` for the Opus calls, `OPENROUTER_API_KEY` for the GPT-6 Astra check, `TYPESAFE_API_KEY` for Jev.
- `JEVKIT_DIR`: the Jev scripts import `JevClient` from OpenAlex's internal Jev client (`jevx.py`, not released: it also leases capacity from an internal broker). `JevClient.decide(state, questions)` is one `POST https://api.typesafe.ai/v1/systemone` with body `{"model": "jev-1.13.0", "state": ..., "questions": ...}` and a bearer key; its result exposes `.ok`, `.answers`, `.model`. A few lines of `httpx` replace it.
- `SENSES_OUT`: folder for the corpus run's bucket files (default `out/`, about 1.8 GB for 320 buckets).

## Run order

| step | script | reads | writes |
|---|---|---|---|
| 1. candidates | `homonym_cands.py` | `work_keywords_v2`, `openalex_works` | table `kw1465_kw_field` (keyword × field counts); `data/homonym_domain_cands.jsonl` (460K candidates) |
| 2. discovery judge | `homonym_judge.py [--n 3000]` (Opus, [prompt 20](../../prompts/20_sense_discovery.md)); the tail sample ran with `--cands` / `--out` / `--tag` | candidates, `kw1465_kw_field`, titles | `data/homonym_judged.jsonl` (`homonym_tail_judged.jsonl` for the 1,000-keyword tail sample) |
| 3. keyword list + field rule | `make_senses.py` | `data/homonym_judged.jsonl`, `kw1465_kw_field`, `keywords_v2` | `data/senses_summary.json` (the 115 keywords), `data/senses.jsonl` (the field rule) |
| 4. field-rule check (failed) | `senses_validate.py` (Opus, [prompt 22](../../prompts/22_sense_check_judge.md)); needs the field rule's staged tables `kw1465_senses` / `kw1465_sense_changes` (built by #1465's `senses_build.py`, not released because the rule never shipped) | staged field-rule moves | `data/senses_valid.jsonl` (2,775 works, 42% right) |
| 5. sense map | `sense_map.py` | `homonym_judged.jsonl`, `senses_summary.json`, `keywords_v2` | `data/sense_map.json`: every Jev option of the 115 keywords → heading id and name (161 minority options) |
| 6. gold works | `gold_pull.py`, `pull_fieldrule.py` | `senses_summary.json`, `senses_valid.jsonl`, `openalex_works` | `data/gold_works.jsonl` (3,437 random works), `data/fieldrule_works.jsonl` (2,775) |
| 7. gold labels | `gold_label.py` (Opus, prompt 22) | gold works | `data/gold_labels.jsonl` |
| 8. classifier on gold | `jev_senses.py --in data/gold_works.jsonl data/fieldrule_works.jsonl --tag v1` (Jev, [prompt 21](../../prompts/21_sense_classifier_jev.md)) | gold works | `data/jev_v1.jsonl` |
| 9. score | `score.py --tag v1` | gold labels, `senses_valid.jsonl`, `jev_v1.jsonl` | `data/score_v1.json` (threshold from train keywords, numbers on test keywords) |
| 10. second judge | `astra_senses.py` (GPT-6 Astra, prompt 22), then `compare3.py` | gold works, labels, Jev answers | `data/astra_labels.jsonl`; agreement printed |
| 11. shortcut check | `shrink_check.py` | gold labels, `senses.jsonl` | printed: share of minority works in main-sense fields (51%), so every pair is classified |
| 12. corpus input | `corpus_in.py` | `work_keywords_v2`, `openalex_works` | table `kw1476_corpus_in` (15,898,677 pairs in 320 hash buckets) |
| 13. corpus run | `corpus_run.py` (Jev; resumable, one bucket in memory, pauses when the machine is under 50% free memory), then `retry_failed.py` | `kw1476_corpus_in` | `$SENSES_OUT/b000.jsonl` … `b319.jsonl` (work_id, heading, probabilities) |
| 14. per-sense gate | `gate_chain.sh` = `gate.py sample` → `gold_label.py --in data/gate_works.jsonl --out data/gate_labels.jsonl` → `gate.py score`; later `gate_topup.py` + the same label and score steps | bucket files, `kw1476_corpus_in`, `sense_map.json` | `data/gate_works.jsonl`, `data/gate_labels.jsonl`, `data/moves_by_sense.json`, `data/gate_v1.json` (80 of 161 pass) |
| 15. load decisions | `load_out.py` | bucket files | Volume `kw1476_vol`, table `kw1476_jev_raw` |
| 16. stage ship 1 | `gen_ship.py stage` | `kw1476_jev_raw`, `sense_map.json`, `gate_v1.json` | tables `kw1476_senses`, `kw1476_work_senses`, `kw1476_rows_in/out`, `kw1476_wk_changes`, `kw1476_vocab_new` + integrity checks |
| 17. descriptions | `describe_senses.sh` (drives the vocabulary's description + Wikidata chain, not in this folder; see its header), then `desc_table.py` | `kw1476_vocab_new`, `kw1476_work_senses` | `$DESC_DIR/*.jsonl`; table `kw1476_desc_new` |
| 18. ship 1 | `gen_ship.py ship` → `run_ship.py` | staged tables | `ship_1476.sql` (copy in [`sql/`](sql)); production writes: backups, 72 headings, descriptions, `keywords_v2_senses`, `work_keyword_senses`, 1,527,116 works rewritten |
| 19. near-miss recheck | `recheck.py draw` → `gold_label.py --in data/recheck_works.jsonl --out data/recheck_labels.jsonl` → `recheck.py score` | bucket files, `gate_v1.json`, `gate_works.jsonl` | `data/recheck_works.jsonl`, `data/recheck_labels.jsonl`, `data/recheck.json` (9 of 11 pass) |
| 20. stage + ship 2 | `gen_ship2.py stage`, `describe_senses.sh` + `desc_table.py` with the ship-2 variables, `gen_ship2.py ship`, `run_ship.py --file ship2_1476.sql` | `recheck.json`, production sense tables | `ship2_1476.sql` (copy in `sql/`); 9 headings, descriptions, 162,774 works rewritten |

`sense_sql.py` is not a step: it holds the sense constants and the two SQL builders (`sense_statement`, `sense_pairs_statement`) that the Keywords nightly uses, copied so that `gen_ship.py` / `gen_ship2.py` rewrite works with exactly the nightly's statement. The nightly's own glue (Databricks notebook code that calls Jev for each night's new works and then runs `sense_statement`) lives in OpenAlex's pipeline repository and is not included here.

`ship_1476.sql` and `ship2_1476.sql` are the statements that ran, kept as a record. They write to OpenAlex production tables and back them up first; do not run them against a workspace you care about without changing the table names.
