# Compute

What it took to produce the shipped keywords. GPU numbers are container-hours from the run ledgers and logs. API numbers are request counts; token counts are given where we recorded them.

## GPU (Modal)

| step | hardware | time | notes |
|---|---|---|---|
| Student training (shipped `qwen4b_v3fix`) | 8× H200 (one node, DDP) | 2.0 h wall (7,177 s) ≈ 16 GPU-h | 907,610 examples, 1 epoch; an 80 GB H100 ran out of memory at batch 8 for the 4B model |
| Held-out inference after training | same container | minutes | HF `generate`, 13,338 works |
| Full-corpus tagging (shipped) | B200, FP8, up to 49 containers (later capped at 10 to leave GPUs for other work) | **325 B200-hours** over 4,000 shards | 474.3M works ≈ 405 works/s per GPU end to end (fetch + generate + upload) |
| Synonym embeddings (bge-m3) | 1× H100 per pass | ≈ 2 min per 2.3M keywords (+ kNN) | 3 passes (K = 8, K = 30 twice) |
| Harness tagging (18,706 works) | 1× B200 | minutes | |

Superseded runs are not needed to reproduce the result: the first v3 student (8× H200, 1 h 57 min) and its full retag (331 B200-hours), a first corpus run with the prompt-v2 student on 328M non-xpac works (H100), and model-size and label-count curves (Qwen3 0.6B, 1.7B, 4B and 8B on A10G, H100 and H200).

Throughput probe (20K eval prompts, one GPU, warm): H100 bf16 250 works/s, H100 FP8 302, B200 bf16 370, B200 FP8 500.

## Databricks

- Corpus input table: one CTAS over the works table, about 6 minutes on a serverless SQL warehouse.
- Vocabulary build: 14 SQL steps; the shipped build (steps 2 to 13) took 50 minutes on serverless SQL warehouses.
- Loading the tagger output: `COPY INTO` from a Databricks Volume, 4,000 gzipped JSONL files (≈ 44 GB compressed).

## Anthropic API (Claude Opus 5.5)

| use | requests | unit |
|---|---|---|
| Teacher labels (prompt v3) | 944,707 (Batches API) + synchronous retries of 7,287 refusals / errors | 1 work per request; mean 562 input / 73 output tokens; effort medium |
| Teacher labels (prompt v2, superseded) | ≈ 940K | 1 work per request |
| Prompt pilot | 7 arms × 620 works + 4,707 extra + judge calls | |
| Student judging | 1,000 works + deltas | 1 work (pooled lists) per call |
| Synonym judging, all rounds | 2.67M pairs | 100 pairs per call, effort low |
| Alternative names / acronym expansion | 200K keywords / 39,765 acronyms | 100 / 150 per call |
| Context checks | 7,090 merges | 20 per call, effort medium |
| Audits | 8 × 1,000 keywords (split audits), 4 × 400 merges (precision audits) | |
| Vocabulary rule tables (from OpenAlex job #1359) | 65,973 + 2,812 keywords, about 250 countries | 150 per call, effort medium |
