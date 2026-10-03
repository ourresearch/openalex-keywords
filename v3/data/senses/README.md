# Keyword senses: data

Results and labels behind [docs/senses.md](../../docs/senses.md). `work_id` is the numeric part of an OpenAlex work id (`W` + work_id). `kid` is a full keyword id URL; `heading` is the keyword id without the prefix. Option numbers follow the sense map: `s0` (or choice 0) is the main sense, then the minority senses, and the last choice is "none of these". No titles, abstracts or judge rationales are included.

| file | rows | what |
|---|---|---|
| `senses.tsv` | 276 | one row per option of the 115 keywords (115 main + 161 minority): label, heading id and name, new heading or existing, full-corpus moves, gate result (right, judged, unlabelled, precision, Wilson lower bound, pass), recheck result, `split`, `shipped` (1 or 2) |
| `sense_map.json` | 115 keywords | the option → heading map the code reads: `{heading: [{opt, label, sense_id, sense_name, new}]}` |
| `new_headings.tsv` | 81 | the new sense headings: id, display name, ship, Wikidata id (only when the link has p ≥ 0.7), description, description model, unclear flag |
| `homonym_judged.jsonl.gz` | 3,000 | discovery verdicts on the 3,000 biggest candidates: `{kid, works, verdict, senses: [{label, fields, share}]}` |
| `homonym_tail_judged.jsonl.gz` | 996 | the same for a works-weighted sample of smaller candidates |
| `senses_summary.json` | 115 | the keywords the classifier ran on, with the failed field rule's sense list and work counts |
| `gold_labels_random.jsonl.gz` | 3,437 | Opus 5.5 labels on random works: `{kid, work_id, field, opts, choice, model, effort, prompt}`; `choice` null = refusal or error |
| `gold_labels_fieldrule.jsonl.gz` | 2,775 | Opus 5.5 labels on works the field rule moved: `{heading, sid, work_id, field, opts, choice, assigned, ok}` (`assigned` = the rule's sense, `ok` = Opus agrees) |
| `jev_v1_gold.jsonl.gz` | 6,211 | Jev's answers on the gold works: `{kid, work_id, ok, answers, model}` |
| `astra_labels.jsonl.gz` | 500 | GPT-6 Astra's labels on 500 random gold works: `{kid, work_id, choice, model}` |
| `score_v1.json` | | the classifier against the gold: threshold, train and test numbers |
| `moves_by_sense.json` | | full-corpus move counts per sense (`heading|opt`) |
| `gate_labels.jsonl.gz` | 7,877 | the per-sense gate: `{kid, work_id, field, moved_to, opts, choice, right, model, effort, prompt}` (`moved_to` = Jev's sense, `choice` = Opus's) |
| `gate_v1.json` | 161 senses | gate result per sense |
| `recheck_labels.jsonl.gz`, `recheck.json` | 1,650; 11 senses | the near-miss recheck, same formats |
| `work_senses_shipped.tsv.gz` | 1,698,244 | every corpus decision that moves a work into one of the 89 split senses: work_id, keyword, sense_id, opt, prob |

`work_senses_shipped.tsv.gz` holds the corpus run's decisions (1,535,668 for the first ship's 80 senses, 162,576 for the second ship's 9). It does not include the decisions the nightly has made for new works since 2 October. Counts differ from the works rewritten (1,527,116 and 162,774) because a work can carry two split keywords, some works already carried the sense heading, and works changed between the corpus run and the ship.
