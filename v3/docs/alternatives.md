# Keyword alternatives

v3.1 gives keywords a list of other names people type for them, served as **`display_name_alternatives`**. Keyword search and autocomplete now find a keyword by those names, so "heart attack" finds myocardial infarction. **1,701,306 keywords carry 5,091,086 alternatives.** GPT-6.1 Sol wrote them, with Wikidata aliases as hints, and each alternative belongs to one keyword.

On 400 strings people typed into the openalex.org keyword box, a blind judge rated the top suggestion right 26% of the time, up from 23%. On 50 hand-written paraphrases typed in full, the right keyword is now in the top 5 for 70% of queries, up from 32%.

## Why

A keyword's name is one wording of its concept. Before v3.1, `/autocomplete/keywords` and `/keywords?search=` matched only `display_name`. A keyword search for "heart attack" returned "heart attack prediction", and "antibacterial resistance" returned nothing.

The synonym map ([synonyms.md](synonyms.md)) does not solve this. It merges keywords that name the same concept, so it only knows wordings that were themselves keywords. Alternatives are strings attached to a keyword. They merge nothing and change no work's keywords. The wordings the synonym map folded into a heading are included among its alternatives (126,074 of them).

## How it was built

| step | what | code |
|---|---|---|
| 1. Bake-off | 8 model arms and raw Wikidata aliases write synonyms for 713 sampled keywords; two judges from different model families label every candidate blind | `syn_sample.py`, `syn_gen.py`, `syn_judge.py`, `syn_score.py` |
| 2. Sweep | the winner (GPT-6.1 Sol with Wikidata hints) writes up to 12 synonyms for every keyword | `syn_build_input.py`, `syn_sweep.py` |
| 3. Ownership | rules drop synonyms that are a keyword's own name or another keyword's name; a judge gives each contested string one owner | `syn_own.py`, `syn_own_judge.py` |
| 4. Eval | autocomplete simulated with and without alternatives, on paraphrases and on real queries | `syn_eval.py` |
| 5. Load | a Databricks table that the API's keywords build joins | `syn_load_dbx.py` |

The scripts are in [code/alternatives/](../code/alternatives/), the prompts in [prompts/23](../prompts/23_alternative_names_writer.md) (writer), [24](../prompts/24_alternatives_bakeoff_judge.md) (bake-off judge) and [25](../prompts/25_alternatives_owner_judge.md) (owner judge), and the outputs in [data/alternatives/](../data/alternatives/).

## The bake-off

**Sample.** 713 keywords: 400 drawn by works count (150 with ≥ 20K works, 150 with 1K to 20K, 100 with < 1K; seed 1464), plus the 313 gold keywords of the paraphrase eval.

**Generation.** Every arm got the same prompt: exact synonyms, abbreviations and expansions, spelling variants, lay terms and rewordings; no broader, narrower or related terms; no translations; 0 to 12 per keyword. Each call carried 10 keywords with their descriptions, at low reasoning effort. Two arms also saw the keyword's Wikidata English aliases, with an instruction to keep only those that qualify.

**Judging.** For each keyword, every candidate from every arm and from Wikidata was pooled, de-duplicated and shuffled, then labelled without its source by Claude Opus 5.5 and by GPT-6 Astra: SAME (someone typing it means exactly this keyword), CLOSE (broader, narrower or related) or WRONG. The judges agree on 88.9% of the 9,103 candidates. "Exact" below means both judges said SAME.

| source | synonyms per keyword | exact (both judges) | broader, narrower or related | share of all good synonyms found | share of good Wikidata aliases found |
|---|---|---|---|---|---|
| **GPT-6.1 Sol + Wikidata hints (chosen)** | 4.1 | **94%** | 4% | 45% | **80%** |
| GPT-6.1 Sol | 4.3 | 95% | 3% | 48% | 51% |
| GPT-6 Sol + Wikidata hints | 3.9 | 89% | 7% | 40% | 72% |
| GPT-6 Luna Pro | 3.6 | 84% | 12% | 35% | 39% |
| GPT-6 Luna | 3.7 | 83% | 13% | 36% | 39% |
| Claude Opus 5.5 | 4.3 | 73% | 22% | 37% | 49% |
| Gemini 3.8 Flash | 4.9 | 71% | 23% | 41% | 50% |
| Claude Sonnet 5.5 | 6.3 | 64% | 30% | 48% | 50% |
| Wikidata English aliases (human-written) | 1.7 | 65% | 24% | 13% | 100% |

- **Sol is the most exact writer.** The Opus judge alone also rates Sol far above Opus (95% vs 73% SAME), so this is not one model family favouring itself. Opus and Gemini drift to broader terms ("drug resistance" for antimicrobial resistance). Sonnet pads with related concepts ("coronary thrombosis" for myocardial infarction).
- **Precision is flat across popular, mid and rare keywords** for every arm (Sol + hints: 92%, 95%, 95%).
- **Hints recover the human-written aliases.** With them, Sol keeps 80% of the Wikidata aliases the judges accept, against 51% without, for one point of precision. That is why we chose it over plain Sol.
- **Raw Wikidata aliases are not usable alone.** Only 65% are exact; 24% are broader, narrower or related, and 6% are wrong.
- **No source is complete.** The "share of all good synonyms" column is relative to everything any source found, and the best arm reaches 48%. More good synonyms exist than any one model writes.

Every arm's output and both judges' labels are in `data/alternatives/bakeoff/`; `syn_score.py` rebuilds the table from them.

## The sweep

GPT-6.1 Sol, called through OpenRouter, wrote synonyms for every keyword in the keyword table on 2026-09-30: 1,882,007 keywords, 10 per call, low reasoning effort. Each keyword came with its description and, for the 225,141 keywords whose Wikidata item has English aliases, those aliases (414,167 keywords have a Wikidata id; dump of 2026-09-03).

| | count |
|---|---|
| keywords answered | 1,881,983 |
| keywords with no synonyms written | 81,668 |
| synonyms written | 6,973,242 |

## One owner per alternative

Autocomplete shows a short list. If "heart attack" were an alternative of five near-duplicate keywords, all five would crowd it. So each alternative string belongs to one keyword. Strings are compared after normalising: Unicode NFKC, lower case, hyphens, underscores and slashes to spaces, apostrophes and periods dropped.

**Rules first** (`syn_own.py`, no model calls). The 6.97M written synonyms and the synonym map's folded wordings were sorted by normalised form:

| outcome | synonyms |
|---|---|
| owned outright, model-written | 4,378,630 |
| owned outright, synonym-map wordings | 103,215 |
| dropped: same as the keyword's own name | 353,332 |
| dropped: same as another keyword's name (the name-clash drop) | 743,558 |
| contested: the same string claimed by 2 or more keywords | 1,319,905 claims on 573,931 strings |

**The name-clash drop.** A string that is another keyword's exact name already finds that keyword by name, so the rules dropped it as an alternative. Most of these pairs are near-duplicate keywords. They became merge candidates for the synonym map. The rule has a cost, described under Limits.

**Then a judge** (`syn_own_judge.py`, GPT-6.1 Sol, 10 strings per call). For each contested string it sees every claiming keyword with its works count and description, and picks the one keyword a person typing the string most likely means. Among near-duplicates, the keyword with the most works wins. Only a genuine homonym, mostly an abbreviation used in different fields (CRP = C-reactive protein / cAMP receptor protein), keeps every real meaning.

| contested strings | count |
|---|---|
| given one owner | 556,286 |
| homonyms kept on several keywords | 17,030 |
| dropped (no candidate meant, or the call failed twice) | 615 |

**Result:** 1,701,306 keywords with 5,091,086 alternatives (4,965,012 model-written, 126,074 synonym-map wordings), 3.0 per keyword on average. The remaining 180,701 keywords have none.

## Ranking and the eval

`/autocomplete/keywords` matches the typed text as a prefix against `display_name` and `display_name_alternatives`. The first ranking we tried treated a name match and an alternative match the same, ordered by works count. It made real queries worse (top suggestion right 23% → 22%): "scienc" suggested artificial intelligence and "social c" suggested social media, because big keywords had alternatives starting with those letters.

**The shipped rule:** an exact name match comes first, then an exact alternative match, then everything else by works count, where **a keyword that matches only through an alternative needs 100× the works of a keyword whose name matches to outrank it**. "heart att" still puts myocardial infarction first, because it has about 430× the works of the best name match ("heart attack prediction", 627 works).

### Real queries

400 strings sampled from what people typed into the openalex.org keyword box, July to September 2026 (final strings of 1 to 5 words, no DOIs). For each, Claude Opus 5.5 labelled the top 3 suggestions of both versions blind (pooled, no version shown): SAME (what they were looking for), CLOSE (related and useful) or WRONG.

| measure | names only | names + alternatives |
|---|---|---|
| top suggestion is what they wanted | 23% | **26%** |
| top suggestion useful (same or related) | 32% | 36% |
| wanted keyword in the top 3 | 26% | 30% |
| no suggestions at all | 66% | 61% |

14 queries gained a right top suggestion and 1 lost it. Absolute numbers are low because two thirds of the strings match no keyword name at all. The strings are user data, so neither they nor the per-query results are published; only these aggregates are.

The three rankings we compared, on the same 400 strings:

| ranking | top suggestion right | wanted keyword in top 3 |
|---|---|---|
| names and alternatives level | 22% | 27% |
| name matches always first | 26% | 29% |
| **alternative-only match needs 100× the works (shipped)** | **26%** | **30%** |

"Name matches always first" scores as well on the first result, but it loses "heart att" → myocardial infarction. The names-only baseline came out at 22% or 23% in different runs of the judge, so differences of one point are noise.

### Paraphrases

289 paraphrase queries (`data/alternatives/eval/eval_all.json`): 50 written by hand (lay terms, spelling variants, abbreviations: "heart attack", "tumour microenvironment", "LLMs"; 1 to 4 acceptable keywords each) and 239 written by Opus 5.5, which was asked how a researcher would type a sampled keyword in different words. The score is strict: is a gold keyword in the top 5, with 40%, 60%, 80% or 100% of the query typed? Shipped ranking:

| typed | hand-written: names only | hand-written: + alternatives | Opus-written: names only | Opus-written: + alternatives |
|---|---|---|---|---|
| 40% | 58% | 68% | 3% | 11% |
| 60% | 42% | 64% | 1% | 23% |
| 80% | 42% | 68% | 1% | 30% |
| 100% | 32% | **70%** | 0% | **30%** |

The Opus-written queries avoid the keyword's own words by design, and many of them are keywords in their own right ("macroalgae" for seaweed), so the strict score undercounts.

## What is live

| | |
|---|---|
| field | `display_name_alternatives` on keyword entities: a list of strings, the same field name as on institutions and authors |
| keywords with alternatives | 1,701,306 |
| alternatives | 5,091,086 |
| keyword documents in the API index carrying the field, at the first sync | 1,700,802 |
| shipped | 2026-10-02 |

`/autocomplete/keywords`, which the keyword pickers on openalex.org use, matches alternatives with the ranking above. When a keyword appears only because of an alternative, the suggestion shows the matched alternative as its hint (myocardial infarction with "heart attacks" for "heart att"). `/keywords?search=` also matches alternatives. Checked live after the load: "heart att" → myocardial infarction, "antibacterial res" → antibiotic resistance, "social c" → social change.

The full list is `data/alternatives/display_name_alternatives.jsonl.gz`.

We also built a meaning-based keyword search over embeddings of keyword names and their synonym-map wordings. It put the searched concept first for 57% of the 289 paraphrases against 30% for the old lexical search, but it is shelved, not live: prefix matching over alternatives handles half-typed queries, where meaning search fails ("self-dri" → self-regulation).

## Limits

- **The name-clash drop took abbreviations away from big keywords.** When a small keyword's name equals an abbreviation, the big keyword loses it. "ML" is the name of a 465-work keyword (the programming language), so machine learning (349K works) lost "ML". Its only other synonym normalised to its own name, so machine learning has no alternatives at all. MI, AD, PD, CVD and NLP are hit the same way: 2,429 short abbreviations on keywords with ≥ 10K works. A fix that restores such abbreviations as homonym alternatives is not in this release.
- **The 94% is measured on the bake-off.** The full run's output was not judged again, and it ran with an output cap (4,000 tokens per 10 keywords) the bake-off did not have.
- **Coverage is partial.** 180,701 keywords have no alternatives, and the bake-off shows no model writes more than about half of the good synonyms that exist.
- **The eval is not fully independent.** The writer prompt uses "heart attack" for myocardial infarction as its example, and that is also one of the 50 hand-written queries. The bake-off sample included the eval's gold keywords. The prompt was not tuned on the eval.
- **Synonym-map wordings come from keyword ids**, so they are lower case with punctuation turned into spaces ("0 18 micron cmos").
- **Works counts include the expansion pack**, which lifts a few keywords in the ranking (microwave reflectometry counts 1.3M works, nearly all of them plasma-shot datasets).
- **English only.** The prompt excludes translations, and the judges count other-language names as WRONG.
- **The real-query half of the eval cannot be rerun from this repository**, because its inputs are user data.
