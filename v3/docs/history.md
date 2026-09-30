# How we got here (2026-09-22 to 2026-09-29)

This page records what was tried and dropped, so the shipped design reads as a choice and not a default. Each item points to the evidence.

## 1. Retrieve-then-choose (dropped)

The first week improved a pipeline that retrieved candidates from a controlled vocabulary and then chose among them:

- a lexical channel over a reviewed vocabulary plus aliases;
- a dense channel (fine-tuned bge-m3);
- a neighbour prior built from the author keywords of similar works;
- an LLM decision model, or a gradient-boosted chooser trained on author keywords.

The best version reached an author-keyword composite of 0.218, and 0.202 without the LLM call (see [results.md](results.md)). GEPA prompt optimisation added 3 to 4 points to the LLM alone and nothing once the chooser was in place. It was dropped for a simpler goal: ask the most capable general model what the keywords are, then copy it cheaply. Its benchmark rows are kept in `data/eval/leaderboard.md`.

## 2. The teacher as the gold standard

Keywords exist to improve recall (search, expert finding, author disambiguation). So the target became the best available description of what a work is about: a frontier model acting as an indexer, reading title, abstract and venue. MeSH catalogers and authors were no longer the target. Human-label benchmarks stayed as the independent check.

## 3. Prompt v2 → v3

The first labels (about 940K works, [prompt v2](../prompts/02_teacher_v2_superseded.md)) asked for 3 to 8 keywords and showed the model Year / Type / Language. A 620-work pilot tested seven prompts with an arm-blind Opus judge ([arms](../prompts/03_prompt_pilot_arms.md), data in `data/teacher/prompt_pilot/`):

- **Dropping the count** roughly doubled the list on abstracts (7 → 12) and added about 4 accepted keywords per work for 3 points of precision. The order stays importance order, so the floor and the confidence cut can trim the tail.
- **The aboutness sentence** did the work of deny lists. Document-type keywords fell from 1.25 to 0.22 per 100, countries without evidence from 0.41 to 0, and non-Latin keywords on non-English works from 7.0 per 100 to 0.
- **Lean input** (no Type / Language / Year) cost nothing measurable, and it removed the inputs that caused doc-type and "Japan" keywords in the first place.
- **"May abstain"** is essential once there is no count. Without it, records with nothing to say still got 5 to 7 keywords each.
- **"Not its general field, not standard methods it merely uses"** (arm F) raised judged precision from 0.887 to 0.945, most on title-only records (0.736 → 0.858) and xpac (0.813 → 0.940). It removed discipline labels and routine methods, which a keyword-type audit showed were 6 to 9 and 4 to 5 per 100 keywords.

Arm F shipped as prompt v3. The teaching set was relabelled with it: about 1M works, now including the expansion pack, which the first sample had left out.

## 4. Student size and data

Learning curves on the v2 labels:

- Qwen3 0.6B / 1.7B / 4B / 8B at 25K to 940K labels: the 4B matched the 8B, and the 8B costs twice as much to run.
- Retrieve-then-choose over Opus's own vocabulary scored below generative distillation on F1 vs Opus.
- The 4B student on vLLM gave the same score as HF generation at about 300 works/s on one H100.

## 5. The LR-schedule bug

Another team's session found that Accelerate's prepared scheduler steps once per process on each optimizer step. Our schedule had been sized in optimizer steps, so on 8 GPUs warmup was 8× too short and the cosine cycled about 4 times. Both the v2 and first v3 students had it. After the fix, F1 vs teacher rose by 0.009 and judged precision rose 0.7 to 2.5 points by group. The corpus was retagged with the fixed model.

## 6. Confidence

Mean token log-prob per keyword predicts the judge's accept/reject with AUC 0.81 (0.80 within title-only records). Rank position alone gives 0.62 and corpus frequency 0.54. Capturing log-probs costs about 8% more GPU time. The shipped cut (log-prob ≥ −1.2, score ≥ 0.30) removes 3.1% of mentions. Most of what it removes is wrong on title-only records, and the cut barely touches works with abstracts.

## 7. FP8 on B200

FP8 changes 43% of lists token for token, but F1 vs teacher moves only 0.4553 → 0.4540 (noise), and throughput per GPU doubles relative to an H100 in bf16.

## 8. Peer reviews of the output

Two rounds of blind reviews by Opus agents looked at trends, author profiles, journals, individual works, horizon scanning, expert search, institutional fingerprints, vocabulary hygiene and the semantic landscape. They drove most of the vocabulary rules: boilerplate, loops, malformed labels, venue names, lazy discipline labels, the templated-source cap and the synonym work. The taxon-evidence rule they suggested was tested and rejected ([vocabulary.md](vocabulary.md)).

## 9. Synonyms

Five rounds, then a hand review ([synonyms.md](synonyms.md)). Sonnet 5 was tried as a cheaper pair judge and rejected: it merged 46% of tail pairs, including wrong ones. The loop stopped when the two audits, split synonyms and wrong merges, began to trade off against each other at the judges' own disagreement rate.
