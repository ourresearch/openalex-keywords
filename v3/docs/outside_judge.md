# The outside judge (v3.1)

The launch post's accuracy numbers come from a judge built by a different lab than the teacher. The keywords were taught by Claude Opus 5.5, and every judge in [results.md](results.md) is also Opus 5.5, which could favour its own style. So for the public numbers we asked OpenAI's GPT-6 Astra, the flagship on OpenAI's model page when we ran it (2026-09-30). It rates **91% of the new keywords accurate, against 44% of the old ones**, and prefers the new list on **395 of 400 works**.

## Old vs new keywords on 400 random works

**Sample** (`code/outside_judge/pull.py`, `data/eval/outside_judge/sample_a.jsonl`). 400 works drawn with a fresh seed from a 1-in-1000 hash sample of OpenAlex (475,072 works), leaving out the 399 works of the earlier blind review. "Old" is the production keywords as of 2026-09-26, before the launch (topic-derived, from the v2 pipeline). "New" is the served v3 keywords. Only works where both lists are non-empty qualify (1,217 of 1,425 candidates), so this measures accuracy where both taggers say something, not coverage: most works it drops are ones where the new tagger correctly returns nothing. 118 of the 400 have no abstract (30%, the natural share) and 109 are in the expansion pack (xpac). Lists average 6.8 old and 5.9 new keywords.

**Method** (`code/outside_judge/judge_gpt.py`, [prompt 26](../prompts/26_outside_judge_keyword_accuracy.md)). Each list is judged on its own: 800 calls, shuffled across works, never saying which list is which. The judge sees type, year, language, venue, title and abstract (first 2,500 characters) and rates every keyword accurate or inaccurate, with a problem type for the inaccurate ones. Near-duplicates are flagged but do not count against accuracy. The work-cluster bootstrap resamples works, so it accounts for keywords of one work being correlated.

| slice | list | works | keywords | accurate | 95% CI (Wilson) | 95% CI (work bootstrap) |
|---|---|---:|---:|---:|---|---|
| all | old | 400 | 2,708 | 43.5% | 41.7–45.4 | 41.1–45.8 |
| all | **new** | 400 | 2,357 | **91.1%** | 89.9–92.2 | 89.6–92.5 |
| with abstract | old | 282 | 2,211 | 46.9% | 44.8–49.0 | 44.3–49.3 |
| with abstract | **new** | 282 | 1,866 | **94.4%** | 93.3–95.4 | 93.1–95.7 |
| no abstract | old | 118 | 497 | 28.6% | 24.8–32.7 | 23.3–33.5 |
| no abstract | **new** | 118 | 491 | **78.4%** | 74.6–81.8 | 73.7–82.9 |
| non-xpac | old | 291 | 2,184 | 44.9% | 42.8–47.0 | 42.3–47.3 |
| non-xpac | new | 291 | 1,803 | 92.2% | 90.8–93.3 | 90.5–93.7 |
| xpac | old | 109 | 524 | 38.0% | 33.9–42.2 | 31.7–43.9 |
| xpac | new | 109 | 554 | 87.5% | 84.5–90.0 | 83.8–90.9 |

What the inaccurate keywords were (share of all keywords in the list):

| list | wrong, unsupported or wrong sense | too generic | not about the subject | malformed |
|---|---:|---:|---:|---:|
| old | 41.0% | 14.3% | 1.0% | 0.1% |
| new | 8.4% | 0.1% | 0.4% | 0.0% |

The new tagger's errors are mostly plausible inferences beyond the text, concentrated on records with no abstract: "rheumatoid arthritis" on a supplement about adiposity, the RAAS and diabetes, or "silicon solar cells" on a paper about defects in Czochralski silicon. The old keywords' errors are the topic-derived kind the launch post shows ("Black hole (networking)" on the first gravitational-wave paper): a term from the work's topic, in the wrong sense or too broad to say anything.

**Head-to-head** ([prompt 27](../prompts/27_outside_judge_preference.md)). Both lists shown as A and B, in random order per work, 400 calls.

| slice | works | new preferred | old preferred | tie | neither useful |
|---|---:|---:|---:|---:|---:|
| all | 400 | **395** | 1 | 0 | 4 |
| with abstract | 282 | 282 | 0 | 0 | 0 |
| no abstract | 118 | 113 | 1 | 0 | 4 |
| xpac | 109 | 109 | 0 | 0 | 0 |

Raw picks were A 207 and B 189, so position bias is small. The one old win is a title-only physics paper where the new list missed ferromagnetism. The four "neither useful" works are title-only entries in reference works.

**Agreement with the Opus judge.** The earlier blind Opus review flagged 17% of new keywords and 76% of old ones, against GPT-6 Astra's 9% and 56%, with slightly different rubrics. The direction and the size of the gap agree; on preference, Opus's 390 : 5 of 399 works becomes 395 : 1 of 400.

## The search examples in the launch post

The post shows what the keyword finds that a title and abstract phrase search misses. These sets were sampled from the live API on 2026-10-02, after the API began serving the new keywords, with the default view (no xpac) and seed 1322 (`code/outside_judge/postswap_judge.py`, `data/eval/outside_judge/postswap/`). Same judge, through OpenRouter, asked whether each work is substantially about the subject ([prompt 28](../prompts/28_outside_judge_on_topic.md)).

| set | works in the set | judged | yes | partly | no | yes or partly | 95% CI |
|---|---:|---:|---:|---:|---:|---:|---|
| antimicrobial-resistance keyword, phrase not in title or abstract | 122,550 | 150 | 100 | 22 | 28 | **81.3%** | 74.3–86.8 |
| the same, works with no abstract | 47,269 | 100 | 67 | 7 | 26 | **74.0%** | 64.6–81.6 |
| phrase "antimicrobial resistance" in title or abstract | 118,099 | 50 | 33 | 7 | 10 | 80.0% | 67.0–88.8 |
| remote-work keyword and a wellbeing keyword, phrases not in the text | 7,922 | 100 | 41 | 28 | 31 | 69.0% | 59.4–77.2 |
| phrases in title or abstract | 2,462 | 50 | 17 | 8 | 25 | 50.0% | 36.6–63.4 |

Scaled up, the antimicrobial-resistance keyword adds about 100,000 on-topic works the phrase search misses (81% of 122,550), and for remote work about 5,500 against the phrase search's 1,230. The exact queries are in `postswap/counts.json`. The post's other counts for the new combined search (118,920 works for the phrase in titles and abstracts, 241,167 with keywords) were read from the API on 2026-10-03 and are not part of this measurement.

An earlier run on a pre-launch sample (2026-09-30, `judge_gpt.py amr`) gave 77% and 90% for the first two sets. Those were different works from a different day's keyword assignments, and that sample is not released because its records carry work text; its verdicts are in `judgments.jsonl` under `R|` keys.

## Caveats

- **Style shows.** Old keywords are Title Case with Wikipedia-style disambiguators ("Sample (material)"), new ones mostly lower case. The judge was never told which list was which, but could tell them apart. This matters most for the head-to-head.
- **Both lists must be non-empty**, so the new tagger's correct abstentions and its gaps are outside the accuracy measure.
- **No refusals and no final parse failures.** The script retries up to 3 times on a wrong item count; retries that later succeeded are not logged.
- **Title-only verdicts are weak evidence** for both lists, and most of the new tagger's errors sit there.
- **Samples of 50** carry about ±13 points of uncertainty.

## Rerun it

The released samples carry ids and keyword lists only, not titles or abstracts. `code/outside_judge/fetch_text.py` refetches the text from the OpenAlex API (4 of the 420 works no longer resolve). `analyze.py` rebuilds `tables.md` and `summary.json` from the released verdicts exactly; `postswap_judge.py report` does the same for the search examples.

```
cd v3/code/outside_judge
python3 analyze.py                                   # Measurement A from released verdicts
python3 postswap_judge.py report                     # the search-example table
OPENALEX_API_KEY=... python3 fetch_text.py           # titles and abstracts for a fresh judge run
OPENAI_API_KEY=... uv run --with openai python judge_gpt.py single   # resumes; delete or move judgments.jsonl to start over
```
