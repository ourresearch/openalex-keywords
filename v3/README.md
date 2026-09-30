# OpenAlex Keywords v3

v3 is the keyword tagger OpenAlex shipped on 2026-09-29. It replaces v2, which picked each work's keywords from 65,004 topic-derived phrases. v3 writes free-text keywords for each work and then folds them into a vocabulary of **1,874,460 keywords**. That vocabulary now covers **418.6M works**.

The approach is distillation. A frontier model (Claude Opus 5.5) acts as an indexer on about 1M works. A small open model (Qwen3-4B) learns to copy it. The small model then tags all 474M works in OpenAlex for a small fraction of what the frontier model would cost. A rule-based vocabulary build and a synonym merge turn its free text into keyword entities.

## Results

| benchmark (held-out test split) | v3 | v2 (previous production) |
|---|---|---|
| Author-keyword composite (mean top-8 F1 over 5 slices) | **0.178** | 0.09 |
| KP20k F1@5, all keyphrases | **0.223** | – |
| Inspec F1@5, all keyphrases | **0.279** | – |
| Blind Opus judge, v3 vs v2 on the same 399 works | preferred 390 : 5 | |

| the output | |
|---|---|
| Keywords (entities, each on ≥ 100 works) | 1,874,460 |
| Works with ≥ 1 keyword | 418.6M of 472.3M (88.6%); 91.6% outside the expansion pack |
| Keyword assignments | 2.37B; 5.6 per tagged work |
| Judged precision of the student's keywords (Opus judge, strict rubric, 1,000 held-out works) | 0.863 non-xpac, 0.754 title-only, 0.849 xpac (teacher: 0.918) |
| Synonym merges | 362,272 keyword ids folded into 213,043 headings; 500 contested pairs reviewed by hand |
| Merge precision (400 sampled merges, Opus judge) | ≈ 86% |
| Residual split synonyms (1,000 keywords sampled by works) | 4.4% works-weighted, not counting deliberately isolated ambiguous acronyms (11.6% counting them) |

Every number is explained, with its caveats, in [docs/results.md](docs/results.md). The author-keyword composite is the fairest comparison because it scores against human labels. The Opus-judged numbers measure how well the student imitates its teacher.

## Pipeline

| step | what | where it runs | code |
|---|---|---|---|
| 1. Teacher labels | Opus 5.5 labels about 1M random works (31% from the expansion pack) with [prompt v3](prompts/01_teacher_v3.md): aboutness only, title + abstract + venue as input, and it may abstain | Anthropic Batches API | [code/teacher](code/teacher) |
| 2. Student | Qwen3-4B, one epoch of full fine-tuning on 907,610 labels | 8× H200, 2.0 h | [code/student](code/student) |
| 3. Tagging | vLLM, FP8, greedy decoding over 474.3M works; confidence per keyword = exp(mean token log-prob) | ≤ 49× B200, 325 GPU-h | [code/tagging](code/tagging) |
| 4. Vocabulary | canonical ids, plural fold, confidence cut (log-prob ≥ −1.2), purge / dual-use / country / boilerplate / cleanup rules, templated sources capped at 2, floor of 100 works | Databricks SQL | [code/vocabulary](code/vocabulary) |
| 5. Synonyms | bge-m3 kNN plus generated alternative names, acronyms, UK/US spellings and variants → Opus judge → context checks → hand review of 500 pairs | 1× H100 + Anthropic API | [code/synonyms](code/synonyms) |
| 6. Evaluation | human-label harness (author keywords, KP20k, Inspec, MeSH) and Opus-judged precision | laptop + 1 GPU | [code/eval](code/eval) |

Step-by-step commands are in [docs/pipeline.md](docs/pipeline.md). The vocabulary rules are in [docs/vocabulary.md](docs/vocabulary.md), the synonym rounds in [docs/synonyms.md](docs/synonyms.md), and compute in [docs/compute.md](docs/compute.md). The experiments we dropped along the way are in [docs/history.md](docs/history.md).

## Closed and open parts

- **Closed:** every call to a closed model went to Claude Opus 5.5 (`claude-opus-5-5`, Anthropic API, structured JSON outputs). That covers the teacher and every judge. [prompts/](prompts/) has each prompt verbatim, with its model id, effort setting, token limit and output schema.
- **Open:** the student base model [Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B), the embedding model [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3), vLLM, PyTorch, Transformers and Accelerate. The fine-tuned student weights are not in this repository yet (see [data/LARGE_FILES.md](data/LARGE_FILES.md)).

## Replicate it

[docs/replicate.md](docs/replicate.md) is written for a person or an AI agent. It covers the environment variables, the data layout each stage expects, the order of commands, and what to check after each step. The cheapest way to verify our claims needs no GPU and no API spend:

1. Score the released student outputs against the released teacher labels: `data/student/student_v3fix_eval_outputs.jsonl` vs `data/teacher/teacher_labels_eval.jsonl`.
2. Rebuild the benchmark rows from `data/eval/` with `code/eval/score.py`.
3. Rebuild the synonym map from the released verdicts in `data/synonyms/judgments/` with `code/synonyms/build_synmap3.py` and `apply_hand.py`.

## Layout

```
README.md       this file
docs/           results, pipeline, vocabulary, synonyms, evaluation, compute, history, replicate
prompts/        every closed-model prompt, verbatim, with model and parameters
code/           teacher/ student/ tagging/ vocabulary/ synonyms/ eval/ publish/ common/
data/           gold and eval sets, teacher and student outputs, rule tables, the vocabulary, synonym map and verdicts
                (data/LARGE_FILES.md lists what is too large for git)
```

## License

CC0 1.0, like the rest of this repository (see [LICENSE](../LICENSE)). Model licenses: Qwen3-4B is Apache-2.0 and bge-m3 is MIT.

## Problems

If you see a wrong keyword or two keywords that should be one, send a support request: https://openalex.org/help
