# Keyword senses (homonyms)

Some keywords hold unrelated meanings. "Inflation" carried both economics papers and cosmology papers, and "entropy" carried thermodynamics and information theory, so a filter on one of them mixed fields that a searcher never wants mixed. v3.1 splits **89 senses on 74 keywords** off into their own headings, such as "inflation (cosmology)" and "entropy (information theory)". Two ships rewrote **1,527,116 works** (2 October 2026) and then **162,774** more (3 October). A per-work classifier decides which sense each work uses, and every split sense first passed a pre-registered precision gate: pooled precision **0.987** on the 80 senses of the first ship and **0.953** on the 9 of the second.

The main sense keeps the keyword's existing id and name, so most works do not change. "Inflation" now means economic inflation; the 13K cosmology works carry `keywords/inflation-cosmology`.

## The problem

We sized it from the field mix of each keyword's works. A keyword is a candidate when it sits on ≥ 300 works whose primary topics span ≥ 2 domains, the second with ≥ 10% of works (460,000 candidates). Most candidates are one subject studied in several fields ("oxidative stress"). An Opus 5.5 judge ([prompt 20](../prompts/20_sense_discovery.md)) read each candidate's field mix and nine example titles and said whether it names one subject or distinct senses.

| sample | keywords judged | distinct senses | share |
|---|---|---|---|
| the 3,000 biggest candidates | 3,000 | 156 | 5.2% |
| works-weighted sample of the other 457K | 996 | 77 | 7.7% |

So about 5% of big keywords mix meanings, and the smaller ones are no better. Examples: inflation (economics, cosmology, mechanics), entropy, stress (psychological, mechanical, biological), inheritance (law, biology, programming), ontology (philosophy, information science), Apple (fruit, company).

v3.1 works on 115 of the 156 big ones: those whose main sense is the keyword itself (we skip "memory", whose largest sense is collective memory) and whose minority sense looked big enough. They have 161 minority senses between them, and each got a heading: 140 new, 21 that already existed in the vocabulary ("aging materials", "Apple Inc.").

## Why the field shortcut failed

The obvious rule is to move a work to the minority sense when its research field belongs only to that sense: cosmology papers sit in physics, economics papers do not. We built that rule end to end (115 keywords, 139 senses, 2,064,233 works would have changed) and checked it blind. Opus read title and abstract of 2,775 works the rule moved and picked the sense each one uses ([prompt 22](../prompts/22_sense_check_judge.md)).

**The rule was right on 1,178 of 2,775 works (42%).** Most misses (1,409 of 1,597) use the main sense from another field: a mechanics paper about economic inflation is still about economic inflation. Only 10 of the 139 senses reached 90%. It did not ship.

The reverse also fails. In a random gold sample, 313 of 609 works that Opus put in a minority sense (51%) sit in a field the rule gives to the main sense. Field is not a usable proxy in either direction, so every work had to be read.

## The per-work classifier

Each (work, keyword) pair gets one decision from Jev, TypeSafe's hosted decision model (`jev-1.13.0`). Jev takes the work's title, abstract (first 1,500 characters) and venue, and the keyword's senses as options plus "none", and returns a probability for every option ([prompt 21](../prompts/21_sense_classifier_jev.md)). A work moves when its top option is a minority sense with probability ≥ 0.5. The main sense and "none" leave the keyword as it is.

**Gold and pre-registration.** Before any tuning we hashed the keywords into train and test halves and wrote down the bar: on test keywords, move precision ≥ 0.90 with a Wilson 95% lower bound ≥ 0.85, and main-sense recall ≥ 0.95 (we must not move works that use the main sense). Gold labels come from Opus 5.5 on 3,437 random works (30 per keyword) plus the 2,775 field-rule works. The threshold was chosen on train keywords only. Option text was the only thing we were allowed to tune; the first version cleared the bar, so the labels from the discovery judge were used unchanged.

| gold (Opus 5.5 labels) | train keywords | test keywords |
|---|---|---|
| move precision | 0.963 | **0.957** |
| Wilson 95% lower bound | 0.949 | 0.941 |
| main-sense recall | 0.992 | 0.989 |
| coverage (Opus-minority works that move) | 0.965 | 0.947 |
| moves judged | 1,018 | 851 |

**A second judge.** GPT-6 Astra, from another model family and with a different prompt, labelled 500 random gold works. It agreed with Opus on 0.968 of them, with Jev on 0.952, and Opus agreed with Jev on 0.946. Of the 79 works Jev moved, 75 are right by Opus and 73 by both judges.

**The corpus run.** Jev decided all 15,898,677 (work, keyword) pairs on 15,013,573 works (30 September to 1 October). 51 failed calls were retried and none failed in the end. 2,266,847 pairs (14.3%) came out as moves.

## The per-sense gate

The pooled gold numbers say the classifier works in general. They do not say every sense is clean, and the gold has too few moves per sense for a per-sense bound (20 right of 20 still gives a lower bound of only 0.84). So, before reading any corpus output, we moved the per-sense check to what would actually be written:

- for each minority sense, draw 50 random corpus moves (all of them if fewer), by a fixed hash rank;
- Opus 5.5 labels each one with the gold prompt, blind to Jev's choice;
- a sense splits only if precision ≥ 0.90 **and** the Wilson 95% lower bound ≥ 0.85. 48 of 50 passes, 47 of 50 fails;
- nothing is tuned on this draw, and a failing sense stays unsplit.

The draw used the first 64 of 320 corpus buckets (a hash partition, so a random sample of what gets written), then topped up the senses with fewer than 50 moves there from all 320. 7,877 moves were labelled; 10 got no usable answer and were left out.

| gate (50 Opus-judged corpus moves per sense) | senses | corpus moves |
|---|---|---|
| pass | **80** | 1,535,668 |
| fail on about 50 judged moves | 74 | 731,002 |
| fail, fewer than 50 moves in the whole corpus | 7 | 177 |
| all minority senses | 161 | 2,266,847 |

| precision of judged moves | right | judged | precision |
|---|---|---|---|
| the 80 passing senses, pooled | 3,939 | 3,992 | **0.987** |
| all 161 senses, pooled | 7,023 | 7,867 | 0.893 |

The second row is why the gate exists: splitting every sense would have put about one move in nine in the wrong place. The biggest passing senses are performance evaluation (human resources), reliability statistics, mediation (conflict resolution), ontology (information science) and family history (medicine). Per-sense results are in `data/senses/gate_v1.json` and `senses.tsv`.

## The near-miss recheck (second ship)

Eleven senses failed at exactly 47 of 50, one judged move short of the bar, and together they held 214,756 moves. We gave exactly those 11 a bigger, fresh sample, with the rule written down before the draw: 150 new corpus moves per sense, none from the first draw, labelled the same way; a sense passes on the fresh sample alone (the first 50 are not pooled) with the same bar, which at 150 means ≥ 137 right. The 12 senses at 46 of 50 and everything lower were not rechecked.

| near-miss recheck (150 fresh moves each) | right | lower bound | result |
|---|---|---|---|
| housing (animal husbandry) | 149 | 0.963 | pass |
| inflation (mechanics) | 146 | 0.933 | pass |
| resource utilization (natural resources) | 145 | 0.924 | pass |
| accessibility (transport and spatial geography) | 142 | 0.898 | pass |
| entropy (information theory) | 142 | 0.898 | pass |
| taxonomy (general classification scheme) | 142 | 0.898 | pass |
| decentralization (distributed systems) | 142 | 0.898 | pass |
| social welfare (welfare economics) | 140 | 0.882 | pass |
| population structure (demography) | 138 | 0.865 | pass |
| coagulation (water treatment) | 135 | 0.842 | fail |
| cosmology (religion and anthropology) | 128 | 0.788 | fail |

9 of 11 pass, pooled 1,286 of 1,350 right (0.953); they carry 162,576 corpus moves. The two that failed stay unsplit for good.

## What shipped

| | first ship (2 Oct 2026) | second ship (3 Oct 2026) | total |
|---|---|---|---|
| senses split | 80 | 9 | 89 |
| keywords newly split | 67 | 7 | 74 |
| new headings | 72 | 9 | 81 |
| existing headings reused | 8 | 0 | 8 |
| works rewritten | 1,527,116 | 162,774 | 1,689,890 |
| judged precision of the senses (pooled) | 0.987 | 0.953 | |

The works total counts a work twice if both ships changed it. A rewrite replaces only the keyword whose sense moved; order, scores and the work's other keywords stay as they were. If a work already carried the sense heading, the first position wins (288 works lost a duplicate when the first ship was staged). After each ship every keyword id on a changed work was in the vocabulary (0 missing).

Each new heading got a one-sentence description and, where a confident match exists, a Wikidata id, from the same chain that describes the rest of the vocabulary: 56 of the 81 have a Wikidata id, and 1 description is flagged unclear. They are in `data/senses/new_headings.tsv`.

The sense headings reach the API through the normal nightly sync. Example: `keywords.id:inflation-cosmology` returns 12,827 works (including the expansion pack), and `/keywords/inflation-cosmology` shows its description and Wikidata id (Q273508).

## New works, every night

The Keywords nightly now asks Jev the same question for every new work that carries one of the 115 keywords and applies the split senses before the works are served. Every decision is stored, including those for senses that did not split, so a sense can be switched on later without re-running the classifier. On its first night it decided 31,844 pairs and moved 4,509.

## Limits and what is left

- **Errors in what shipped.** By Opus's reading, about 1 move in 77 in the first ship and 1 in 21 in the second is in the wrong sense. The other way round, about 5% of works that use a minority sense stay on the main keyword (coverage 0.947 on test keywords).
- **72 senses stay unsplit.** 65 failed on about 50 judged moves (2 of them again on the recheck) and 7 have fewer than 50 moves in the whole corpus. Their works stay on the main keyword. The biggest are weight loss (materials), resilience (ecology), energy balance (biology), hypoxia (environmental), cosmology (religion and anthropology), stress (biological) and stress (mechanics).
- **The tail is not done.** Discovery covered only the 3,000 biggest candidates, and 7.7% of the smaller keywords also mix meanings ("grace": theology and the GRACE satellite mission). A cheap keyword-level filter with Jev did not find them reliably (recall 0.867 with 34% of keywords flagged), so the tail needs an LLM judge first.
- **One judge family.** Opus wrote the sense labels and also judged the moves. The GPT-6 Astra check (0.968 agreement with Opus) is the guard against shared blind spots; its disagreements were not settled by hand.
- **Names.** The main sense keeps the bare name ("inflation", not "inflation (economics)"), so the bare name still reads as ambiguous even though its works are now one sense.

## Files

| what | where |
|---|---|
| scripts, in run order | [`code/senses/`](../code/senses) (see its README) |
| prompts | [20](../prompts/20_sense_discovery.md) discovery, [21](../prompts/21_sense_classifier_jev.md) Jev classifier, [22](../prompts/22_sense_check_judge.md) per-work judge |
| one row per sense option: label, heading, gate, recheck, shipped | `data/senses/senses.tsv` |
| the option-to-heading map the code reads | `data/senses/sense_map.json` |
| new headings with descriptions and Wikidata ids | `data/senses/new_headings.tsv` |
| discovery verdicts (rationales removed) | `data/senses/homonym_judged.jsonl.gz`, `homonym_tail_judged.jsonl.gz`, `senses_summary.json` |
| gold labels, Jev answers on gold, second judge, score | `data/senses/gold_labels_random.jsonl.gz`, `gold_labels_fieldrule.jsonl.gz`, `jev_v1_gold.jsonl.gz`, `astra_labels.jsonl.gz`, `score_v1.json` |
| gate and recheck labels and results | `data/senses/gate_labels.jsonl.gz`, `gate_v1.json`, `moves_by_sense.json`, `recheck_labels.jsonl.gz`, `recheck.json` |
| per-work moves into the 89 split senses | `data/senses/work_senses_shipped.tsv.gz` |

The data files hold work ids, options and verdicts only, no titles or abstracts. To rerun a label, fetch the work from the OpenAlex API by its id.
