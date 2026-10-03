# Keyword alternatives: data

How these files were made is in [docs/alternatives.md](../../docs/alternatives.md); the scripts are in [code/alternatives/](../../code/alternatives/).

| path | what |
|---|---|
| `display_name_alternatives.jsonl.gz` | **the shipped field**: `{kid, display_name_alternatives}` for 1,701,306 keywords, 5,091,086 alternatives, in the order the API serves them (38 MB). Keywords with no alternatives are absent. The full id is `https://openalex.org/keywords/<kid>`. |
| `bakeoff/syn_sample.jsonl.gz` | the 713 bake-off keywords: `{kid, name, n, description, qid, wd_label, wd_aliases, stratum}` (stratum: head ≥ 20K works, torso 1K to 20K, tail < 1K, eval_gold) |
| `bakeoff/syn_gen_<arm>.jsonl.gz` | each arm's synonyms for the sample: `{kid, synonyms, model}`; arms and model ids in [prompts/23](../../prompts/23_alternative_names_writer.md) |
| `bakeoff/syn_judge_opus.jsonl.gz`, `syn_judge_astra.jsonl.gz` | both judges' blind labels of every pooled candidate: `{kid, labels: {candidate: SAME / CLOSE / WRONG}, judge}` |
| `bakeoff/bakeoff_scores.json` | per arm: synonyms per keyword, SAME share per judge and for both, CLOSE and WRONG shares (mean of the judges), recall of the pooled good synonyms, recall of the good Wikidata aliases. `code/alternatives/syn_score.py` rebuilds it from the files above. |
| `ownership/own_judged.jsonl.gz` | the owner judge's verdict for each contested string: `{norm, owner, homonym_owners}` (573,431 rows; `owner` null = no candidate meant) |
| `eval/eval_all.json` | the 289 paraphrase queries: `{query, gold, src}` (src `hand` = the 50 hand-written queries; `gen_head` / `gen_torso` = the 239 written by Opus 5.5, with the source `keyword` and its works count `n`) |
| `eval/eval_results_penalty100_shipped.txt` | `syn_eval.py` output for the shipped ranking (an alternative-only match needs 100x the works of a name match) |
| `eval/eval_results_level.txt`, `eval_results_tiered.txt` | the same for the two rankings we rejected (names and alternatives level; name matches always first) |

The real-query half of the eval used 400 strings people typed into the openalex.org keyword box. Those strings and the per-query results are user data and are not published; the `.txt` files hold only aggregate numbers.
