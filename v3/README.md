# OpenAlex Keywords v3

v3 is the keyword tagger OpenAlex shipped on 2026-09-29. It replaces v2, which picked each work's keywords from 65,004 topic-derived phrases. v3 writes free-text keywords for each work and then folds them into a vocabulary of **1.88 million keywords**, which today cover **419.6M works**. An independent judge from another lab rates **91% of the new keywords accurate, against 44% of the old ones**.

The approach is distillation. A frontier model (Claude Opus 5.5) acts as an indexer on about 1M works. A small open model (Qwen3-4B) learns to copy it. The small model then tags all 474M works in OpenAlex for a small fraction of what the frontier model would cost. A rule-based vocabulary build and a synonym merge turn its free text into keyword entities.

This is release **v3.1** (2026-10-03). It brings the repository up to what OpenAlex serves today: the synonym audit, the keyword sense splits, the alternative names that search and autocomplete use, and the outside-judge evaluation behind the [launch post](https://blog.openalex.org/openalex-keywords-are-finally-good/). The tagger and its weights are unchanged from [v3.0](https://github.com/ourresearch/openalex-keywords/releases/tag/v3.0).

## Results

| accuracy | v3 | v2 (previous production) |
|---|---|---|
| Outside judge (OpenAI GPT-6 Astra): keywords rated accurate, 400 random works | **91.1%** | 43.5% |
| Outside judge, the two lists side by side on the same 400 works | **preferred on 395** | preferred on 1 (4: neither useful) |
| Author-keyword composite (mean top-8 F1 over 5 slices, human labels) | **0.178** | 0.09 |
| KP20k F1@5, all keyphrases | **0.223** | – |
| Inspec F1@5, all keyphrases | **0.279** | – |
| Blind Opus judge, v3 vs v2 on 399 works | preferred 390 : 5 | |

| finding what text search misses (launch post, 2026-10-02) | works | on topic (outside judge) |
|---|---:|---:|
| antimicrobial-resistance keyword, phrase not in the title or abstract | 122,550 | 81% |
| the same, works with no abstract | 47,269 | 74% |

| the output, as served on 2026-10-03 | |
|---|---|
| Keywords | 1,881,484 headings on at least one work (1,882,088 in the vocabulary) |
| Works with ≥ 1 keyword | 419,634,002 of 475,749,838 (88.2%); outside the expansion pack 300,391,034 of 330,132,640 (91.0%) |
| Keyword assignments | 2.36B; 5.6 per tagged work |
| Judged precision of the student's keywords (Opus judge, strict rubric, 1,000 held-out works) | 0.863 non-xpac, 0.754 title-only, 0.849 xpac (teacher: 0.918) |
| Synonym map | 357,259 keyword ids folded into 211,451 headings, after an audit broke 5,016 wrong merges |
| Keyword senses | 89 senses on 74 keywords split into their own headings; 1,689,890 work rewrites; judged precision of the moves 0.987 (80 senses) and 0.953 (9 rechecked) |
| Alternative names for search and autocomplete | 5,091,086 on 1,701,306 keywords (94% judged exact in the bake-off) |

Every number is explained, with its caveats, in [docs/results.md](docs/results.md). The outside judge and the author-keyword composite are the independent checks: one comes from a different lab than the teacher, the other scores against keywords authors chose. The Opus-judged numbers measure how well the student imitates its teacher. The launch post rounds coverage to 89%, the share at build time (88.6%).

## What changed in v3.1

- **Synonym audit** ([docs/synonym_audit.md](docs/synonym_audit.md)). An audit of every merge in the v3.0 synonym map found merges that joined unrelated subjects through acronyms (dark matter → diabetes mellitus through DM, information retrieval → insulin resistance through IR). 5,016 wrong merges were broken; the served map is now `data/synonyms/synmap10_final.jsonl.gz`, and the keyword lists of 8.46M works were corrected.
- **Keyword senses** ([docs/senses.md](docs/senses.md)). 89 senses on 74 keywords that mixed unrelated meanings now have their own headings, such as "inflation (cosmology)" and "entropy (information theory)". A per-work classifier reads each work's title and abstract; splitting by research field was right only 42% of the time. Every split sense passed a pre-registered gate of Opus-judged moves (pooled precision 0.987; 9 near-misses later passed a fresh 150-move recheck at 0.953), and the nightly update classifies new works.
- **Alternative names** ([docs/alternatives.md](docs/alternatives.md)). Keywords gain `display_name_alternatives`: 5,091,086 other names for 1,701,306 keywords, written by GPT-6.1 Sol with Wikidata hints, one owner keyword per name except genuine homonyms. Search and autocomplete match them, so "heart att" finds myocardial infarction.
- **Outside judge** ([docs/outside_judge.md](docs/outside_judge.md)). The accuracy numbers in the launch post, from OpenAI's GPT-6 Astra: code, prompts, the sample (ids and keyword lists) and every verdict.
- Also: the vocabulary as served today (`data/vocabulary/vocabulary_20261003.tsv.gz`), an opt-in acronym guard in the synonym-map builder, and a one-command check of the headline claims (`code/check_level_a.py`).

The second batch of sense splits (9 senses, 162,774 works) is in OpenAlex's served tables and reaches the API with the 2026-10-04 nightly update.

## Pipeline

| step | what | where it runs | code |
|---|---|---|---|
| 1. Teacher labels | Opus 5.5 labels about 1M random works (31% from the expansion pack) with [prompt v3](prompts/01_teacher_v3.md): aboutness only, title + abstract + venue as input, and it may abstain | Anthropic Batches API | [code/teacher](code/teacher) |
| 2. Student | Qwen3-4B, one epoch of full fine-tuning on 907,610 labels | 8× H200, 2.0 h | [code/student](code/student) |
| 3. Tagging | vLLM, FP8, greedy decoding over 474.3M works; confidence per keyword = exp(mean token log-prob) | ≤ 49× B200, 325 GPU-h | [code/tagging](code/tagging) |
| 4. Vocabulary | canonical ids, plural fold, confidence cut (log-prob ≥ −1.2), purge / dual-use / country / boilerplate / cleanup rules, templated sources capped at 2, floor of 100 works | Databricks SQL | [code/vocabulary](code/vocabulary) |
| 5. Synonyms | bge-m3 kNN plus generated alternative names, acronyms, UK/US spellings and variants → Opus judge → context checks → hand review of 500 pairs | 1× H100 + Anthropic API | [code/synonyms](code/synonyms) |
| 5b. Synonym audit (v3.1) | every merge judged same / close / wrong by Opus 5.5; wrong ones broken (synmap10); vocabulary rebuilt with the same rules | Anthropic API + Databricks SQL | [code/synonym_audit](code/synonym_audit) |
| 6. Evaluation | human-label harness (author keywords, KP20k, Inspec, MeSH), Opus-judged precision, and the outside judge (v3.1) | laptop + 1 GPU; OpenAI API | [code/eval](code/eval), [code/outside_judge](code/outside_judge) |
| 7. Senses (v3.1) | Opus finds keywords with distinct senses; Jev classifies each work's sense from title + abstract; a per-sense gate of Opus-judged moves decides which split | Anthropic API + Jev + Databricks SQL | [code/senses](code/senses) |
| 8. Alternative names (v3.1) | GPT-6.1 Sol writes other names per keyword with Wikidata hints; one owner per name; ranking checked on real queries | OpenRouter | [code/alternatives](code/alternatives) |

Step-by-step commands are in [docs/pipeline.md](docs/pipeline.md). The vocabulary rules are in [docs/vocabulary.md](docs/vocabulary.md), the synonym rounds in [docs/synonyms.md](docs/synonyms.md), and compute in [docs/compute.md](docs/compute.md). The experiments we dropped along the way are in [docs/history.md](docs/history.md).

## Closed and open parts

- **Closed:** the teacher and the v3.0 judges are Claude Opus 5.5 (`claude-opus-5-5`, Anthropic API, structured JSON outputs). v3.1 adds models from other providers: OpenAI's GPT-6 Astra as the outside judge, GPT-6.1 Sol as the alternative-name writer (both mostly through OpenRouter), and Jev for the per-work sense classifier. [prompts/](prompts/) has each prompt verbatim, with its model id, effort setting, token limit and output schema.
- **Open:** the student base model [Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B), the embedding model [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3), vLLM, PyTorch, Transformers and Accelerate. The fine-tuned student weights are on GitHub, in the [`v3.0` release](https://github.com/ourresearch/openalex-keywords/releases/tag/v3.0); see [`docs/weights.md`](docs/weights.md).

## Replicate it

[docs/replicate.md](docs/replicate.md) is written for a person or an AI agent. It covers the environment variables, the data layout each stage expects, the order of commands, and what to check after each step. The cheapest way to verify our claims needs no GPU, no API spend and no extra packages:

```
cd v3 && python3 code/check_level_a.py
```

It recomputes the student's agreement with its teacher, the vocabulary and synonym-map sizes, rebuilds the served synonym map byte for byte from the released audit verdicts, and recomputes the outside judge's accuracy, head-to-head and search-example numbers from the released verdicts.

## Layout

```
README.md       this file
docs/           results, pipeline, vocabulary, synonyms, synonym_audit, senses, alternatives, outside_judge, evaluation, compute, history, replicate, weights
prompts/        every closed-model prompt, verbatim, with model and parameters
code/           teacher/ student/ tagging/ vocabulary/ synonyms/ synonym_audit/ senses/ alternatives/ eval/ outside_judge/ publish/ common/
data/           gold and eval sets, teacher and student outputs, rule tables, the vocabulary, synonym maps and verdicts,
                sense splits, alternative names, outside-judge verdicts (data/LARGE_FILES.md lists what is too large for git)
```

## License

CC0 1.0, like the rest of this repository (see [LICENSE](../LICENSE)). Model licenses: Qwen3-4B is Apache-2.0 and bge-m3 is MIT.

## Problems

If you see a wrong keyword or two keywords that should be one, send a support request: https://openalex.org/help
