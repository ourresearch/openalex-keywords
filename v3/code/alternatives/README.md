# Keyword alternatives (`display_name_alternatives`)

The scripts that wrote, owned, tested and loaded the keyword field `display_name_alternatives` (v3.1, shipped 2026-10-02). The method and results are in [docs/alternatives.md](../../docs/alternatives.md), and the prompts in [prompts/23](../../prompts/23_alternative_names_writer.md), [24](../../prompts/24_alternatives_bakeoff_judge.md) and [25](../../prompts/25_alternatives_owner_judge.md).

They are copied from the working directory of OpenAlex job #1464. For the release we changed only paths and settings: hard-coded file paths became environment variables or working-directory files, the Databricks warehouse id and Volume became `DATABRICKS_WAREHOUSE_ID` and `DATABRICKS_VOLUME`, the OpenRouter helper was copied in (`or_util.py`), Databricks SQL comes from `../common/dbx.py`, and the bake-off scorer lost its projected-cost column. Otherwise the logic is unchanged. Cost guards and the memory guard (pause while under 50% of memory is free) are kept.

## Running them

Every script reads and writes files in the current working directory, so run them from one work folder: `cd work && uv run --with httpx python ../code/alternatives/<script>.py`.

| variable | used by |
|---|---|
| `OPENROUTER_API_KEY` | `syn_gen.py`, `syn_judge.py`, `syn_sweep.py`, `syn_own_judge.py` |
| `CLAUDE_API_KEY` or `ANTHROPIC_API_KEY` | `gen_paraphrases.py`, `syn_eval.py` (Opus 5.5 judge) |
| `DATABRICKS_WAREHOUSE_ID` | everything that queries Databricks (through `../common/dbx.py` and the `databricks` CLI) |
| `DATABRICKS_VOLUME` | `syn_load_dbx.py` (a Unity Catalog Volume for the bulk load) |
| `WIKIDATA_SQLITE` | `syn_build_input.py`: a local SQLite index of the Wikidata JSON dump, table `items(qid, label, aliases)` (aliases = JSON list of English aliases) |
| `SYNMAP` | `syn_own.py`: the synonym map `{kid, canon}` (default `synmap10.jsonl`) |
| `OWN_COST_CAP`, `OWN_LIMIT` | `syn_own_judge.py`: spending cap, and a group limit for pilots |

## Run order

| step | script | reads | writes |
|---|---|---|---|
| 1. vocabulary | `vocab_pull.py` | `openalex.common.keywords_v2`, `openalex.works.work_keywords_v2` | `vocab10.jsonl` `{kid, name, n}` (n = works count) |
| 2. paraphrase eval set | `gen_paraphrases.py` | `vocab.jsonl` (same shape, the shipped vocabulary) | `eval_gen.json` (239 model-written queries); with the 50 hand-written queries this is `eval_all.json` |
| 3. bake-off sample | `syn_sample.py` | `vocab10.jsonl`, `eval_all.json`, the descriptions table, the Wikidata API | `syn_sample.jsonl` (713 keywords) |
| 4. bake-off generation | `syn_gen.py [arm ...]` | `syn_sample.jsonl` | `syn_gen_<arm>.jsonl` `{kid, synonyms, model}` |
| 5. bake-off judging | `syn_judge.py` | `syn_sample.jsonl`, every `syn_gen_<arm>.jsonl` | `syn_judge_opus.jsonl`, `syn_judge_astra.jsonl` `{kid, labels, judge}` |
| 6. bake-off scores | `syn_score.py` | the files of steps 3 to 5 | printed table, `syn_scores.json` |
| 7. sweep input | `syn_build_input.py` | `keywords_v2`, `keywords_v2_wikidata_description`, `WIKIDATA_SQLITE` | `syn_input.jsonl` `{kid, name, description, qid, wd_label, wd_aliases}` (1,882,007 keywords) |
| 8. full sweep | `syn_sweep.py` | `syn_input.jsonl` | `syn_full.jsonl` `{kid, synonyms, model, prompt_version}`, `syn_full.log`, `syn_full.errors.jsonl`. Resumable; run it a second time to retry failed batches. |
| 9. ownership rules | `syn_own.py` | `syn_full.jsonl`, `SYNMAP`, `vocab10.jsonl` | `syn_owned.jsonl`, `collisions.jsonl`, `name_clash.jsonl` |
| 10. owner judge | `syn_own_judge.py` (then once more for failed batches) | `collisions.jsonl`, `syn_input.jsonl` (descriptions), `syn_owned.jsonl` | `own_judged.jsonl` `{norm, owner, homonym_owners}`, then `syn_final.jsonl` `{kid, synonyms: [{text, source}]}` |
| 11. eval | `RANK=penalty PENALTY=100 syn_eval.py` | `syn_final.jsonl`, `vocab10.jsonl`, `eval_all.json`, `real_queries.json` | `syn_fts.sqlite`, printed aggregates, `syn_eval_real*.json` |
| 12. load | `syn_load_dbx.py` | `syn_final.jsonl` | table `openalex.common.keywords_v2_display_name_alternatives` (1,701,306 rows) |

Steps 2 to 6 choose the model; steps 7 to 12 are the production run. `syn_sweep.py` imports the prompt from `syn_gen.py`, and `syn_own_judge.py` imports the API call from `syn_sweep.py`, so keep the folder together.

After step 12 the API's keywords build joins the table, and the keywords Elasticsearch index gets `display_name_alternatives` as a text field with an `.autocomplete` (search-as-you-type) and a `.keyword` subfield, mapped like `display_name`. Those steps live in the OpenAlex API and pipeline repositories, not here.

## What is released, and what is not

`data/alternatives/` holds the shipped field, every bake-off file (sample, arm outputs, both judges' labels, scores), the owner-judge verdicts, the paraphrase eval set and the eval's aggregate output. See [data/alternatives/README.md](../../data/alternatives/README.md).

Not released:

- `real_queries.json` and the per-query eval output (`syn_eval_real*.json`). These are strings people typed into the openalex.org keyword box, which is user data. Only the aggregate numbers are published, so step 11's real-query half cannot be rerun from this repository.
- The intermediate files of steps 7 to 10 (`syn_input.jsonl` 519 MB, `syn_full.jsonl` 394 MB, `syn_owned.jsonl` 321 MB, `collisions.jsonl` 198 MB, `name_clash.jsonl` 90 MB). They can be rebuilt with the scripts, though step 8 and step 10 call a model.
- Two prompts sit only in the code: the paraphrase writer in `gen_paraphrases.py` and the real-query judge in `syn_eval.py` (both Opus 5.5, `claude-opus-5-5`, no effort setting, `max_tokens` 3000 and 6000).
