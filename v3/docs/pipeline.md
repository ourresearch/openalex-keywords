# The pipeline, step by step

Each step lists its inputs, the command we ran, and its outputs. Paths are relative to `v3/code/<stage>/` unless marked otherwise. Every script is the version that produced the shipped tables (2026-09-26 to 2026-09-29). We changed only import paths and hard-coded warehouse ids for the release, and the logic is untouched. [replicate.md](replicate.md) covers environment setup.

Conventions:

- **Databricks.** Tables live in the OpenAlex Unity Catalog (`openalex.works.openalex_works` is the works table; intermediate tables go in `openalex.transient`). SQL runs through `code/common/dbx.py` / `dbxr.py`, which call the `databricks` CLI (auth from your CLI profile) with `DATABRICKS_WAREHOUSE_ID`. If you are outside OpenAlex, substitute your own copy of the works table: the only columns used are `id`, `title`, `abstract`, `primary_location.source.{id, display_name, host_organization_name}`, `type`, `is_xpac`, `authorships[].countries` and `works_study_design.study_designs`.
- **Anthropic.** Scripts read `CLAUDE_API_KEY` (or `ANTHROPIC_API_KEY`). Every call uses `claude-opus-5-5` with structured JSON outputs; see [../prompts/](../prompts/).
- **Modal.** GPU steps run on [Modal](https://modal.com) with a Volume named `oxjob1322` (mounted at `/vol`) and, for tagging, a Modal Secret `oxjob-databricks` holding `DATABRICKS_HOST`, `DATABRICKS_TOKEN` and `DATABRICKS_WAREHOUSE_ID`.
- **Expansion pack ("xpac").** OpenAlex's expansion pack holds 147M records (datasets, specimens, repository items). They are flagged `is_xpac`, and xpac works are tagged like every other work.

## 1. Teacher labels (code/teacher)

**1a. Sample.** `pull_random_sample.py` draws a simple random sample of non-xpac works. That sample became the 1M-work "God" sample of the first label wave. `build_sample.py` then builds the v3 teaching set, 1,000,000 works in population proportion:

- 690K non-xpac works drawn from the God sample, always including the 10K old eval ids;
- ≈ 310K xpac works: every work in 11 random `pmod(hash(id), 5000)` buckets.

It holds out 14,500 works (10K non-xpac + 4.5K xpac) as the eval set.

```
uv run --with httpx python build_sample.py          # -> data/sample.jsonl, data/eval_ids.txt
```

Released: `data/teacher/sample_ids.tsv.gz` (every work id, xpac flag, held-out flag) and `data/teacher/eval_ids.txt`. Work text is not redistributed. Fetch it from the OpenAlex API or snapshot by id (title, abstract, `primary_location.source.display_name`).

**1b. Label with Opus 5.5** ([prompt v3](../prompts/01_teacher_v3.md), effort medium, Batches API). `driver.sh` submits 100K-request waves, collects each wave, and finally retries refusals and errors synchronously:

```
./driver.sh     # = judge_batch.py --prompt prompt_v3 --tag v3 --sample data/sample.jsonl submit|poll|retry
                # -> results/god_v3.jsonl  {id, model, effort, answer: {keywords: [...]}, usage}
```

The result was 944,707 requests, of which 943,769 were answered (87,494 of them with an empty list) and 938 refused. Mean usage was 562 input and 73 output tokens. Released: the teacher's labels for the held-out works (`data/teacher/teacher_labels_eval.jsonl`). The full label file (307 MB) is listed in [../data/LARGE_FILES.md](../data/LARGE_FILES.md).

**How the prompt was chosen.** `pilot2/` compared seven prompts on 620 fixed works using an arm-blind Opus judge. Data is in `data/teacher/prompt_pilot/`, and the story is in [history.md](history.md).

## 2. Student (code/student)

**2a. Training data.** Rows are `{id, prompt, target}`. `prompt` is the user text above and `target` is the teacher's keywords joined by `"; "` in the teacher's order. Held-out ids go to eval.

```
cd code/teacher && python ../student/student_gen_data.py --tags v3 --prompt prompt_v3 \
    --samples data/sample.jsonl --eval-ids data/eval_ids.txt --suffix _v3
# -> data/student_train_v3.jsonl (907,610 rows), data/student_eval_v3.jsonl (13,338 rows)
```

**2b. Fine-tune Qwen3-4B** on Modal, 8× H200, DDP through Accelerate (`student_gen_train_ddp.py`):

```
modal volume put oxjob1322 data/student_train_v3.jsonl data/student_train_v3.jsonl
modal volume put oxjob1322 data/student_eval_v3.jsonl  data/student_eval_v3.jsonl
for f in student_gen_train.py student_gen_train_ddp.py student_gen_infer.py; do modal volume put oxjob1322 $f $f; done
modal run --detach modal_student.py --tag qwen4b_v3fix --base Qwen/Qwen3-4B --n 907610 --epochs 1 \
    --bs 8 --acc 1 --maxlen 640 --gpus 8 --train-file student_train_v3.jsonl --eval-file student_eval_v3.jsonl
# -> /vol/models/student_qwen4b_v3fix, /vol/results/arm_qwen4b_v3fix.jsonl (eval outputs)
```

Recipe: full fine-tune in bf16 weights with no mixed precision and no gradient checkpointing. AdamW at lr 5e-5 with weight decay 0.01, cosine schedule with 3% warmup, gradient clipping at 1.0. Batch 8 per GPU × 8 GPUs = 64. One epoch = 14,181 optimizer steps in 7,177 s, final loss ≈ 0.73. The input is `prompt[:4000] + "\nKeywords:"`, the target is `" " + keywords + eos`, and sequences are capped at 640 tokens with the prompt truncated first. Loss is computed on target tokens only. The training log is `data/student/train_v3fix.log`.

**The LR-schedule fix.** Accelerate's prepared scheduler steps once per process on each optimizer step. Sizing the cosine schedule in optimizer steps therefore ran warmup 8× too short and cycled the cosine about 4 times on 8 GPUs. `student_gen_train_ddp.py` now sizes the schedule in process-steps. The fixed model (`v3fix`) beat the unfixed one (`v3`) on the same data: F1 vs teacher 0.5598 → 0.5684 (+0.0086 ± 0.0027) on 13,338 held-out works, and judged precision rose everywhere (see [results.md](results.md)).

**2c. Judge the student** against the teacher (strict indexer rubric, [prompt](../prompts/04_judge_keywords.md)):

```
uv run --with anthropic python judge_union.py   # 700 non-xpac + 300 xpac held-out works; new vs old student vs teacher
uv run --with anthropic python judge_fix.py     # adds the LR-fixed student; reuses verdicts by kid0
```

Released: `data/student/judged_v3.jsonl`, `judged_fix.json`, `score_v3.json`, and the eval outputs `student_v3fix_eval_outputs.jsonl` and `student_v3_eval_outputs.jsonl`.

## 3. Full-corpus tagging (code/tagging)

**3a. Input table.** Every work (474,293,283, xpac included) with title, abstract (≤ 6,000 chars) and venue, sharded by `pmod(hash(id), 4000)` and clustered by shard:

```
uv run --with httpx python build_input.py    # -> openalex.transient.kw_corpus_input_v3
```

**3b. Tag.** `modal_corpus_v3.py` is a Modal class `Tagger` with vLLM, FP8 weights (`quantization="fp8"`) on B200, greedy decoding (temperature 0, `max_tokens` 192, stop at newline) and `logprobs=1`. It uses the same prompt as training, truncated to 560 tokens. Each keyword gets the mean log-prob of its tokens; the separator token counts toward the keyword it closes, and special tokens are skipped. One call runs one shard: it reads the shard through the Databricks statement API and writes gzipped JSONL to a Databricks Volume.

```
modal deploy modal_corpus_v3.py
python corpus_run_v3.py --gpu B200 --quant fp8 --model /vol/models/student_qwen4b_v3fix \
    --ledger ledger_v3fix.jsonl --out-dir /Volumes/<catalog>/<schema>/<volume>/corpus_v3fix
uv run --with httpx python load_v3.py --table keywords_student_v3fix --vol /Volumes/<...>/corpus_v3fix \
    --ledger ledger_v3fix.jsonl --loaded loaded_v3fix.txt
# -> openalex.transient.keywords_student_v3fix {id, keywords[], lp[], model, shard}
```

The run covered 4,000 shards and 474,293,283 works with 0 failed shards. It produced 3,520,603,730 raw keyword mentions (7.4 per work); 47.5M works (10.0%) got an empty list, meaning the model abstained. The per-shard timings are in `data/tagging/ledger_v3fix.jsonl`. `probe.py` measured throughput by GPU and precision on 20K eval prompts: H100 bf16 250 works/s, B200 bf16 370, B200 FP8 500. FP8 changed F1 vs teacher from 0.4553 to 0.4540 (noise). `modal_confidence.py` and `conf_analyze.py` are the confidence study: mean token log-prob predicts the judge's accept/reject with AUC 0.81.

## 4. Vocabulary (code/vocabulary)

One SQL pipeline, `build_vocab.py`, turns raw mentions into keyword entities. The final build was:

```
uv run --with httpx python build_vocab.py --source keywords_student_v3fix --prefix kw1322v3n_ \
    --boilerplate --synmap kw1322v3_synmap9 --min-lp -1.2 --floor 100
# -> openalex.transient.kw1322v3n_{strings0,fold,strings,purge,country_kids,boiler,mentions_kept,templated,mentions_final,vocab,work_keywords}
```

The steps, rules and counts are in [vocabulary.md](vocabulary.md). The rule tables are in `data/vocabulary/`, and the resulting vocabulary is `data/vocabulary/kw1322v3n_vocabulary.tsv.gz`.

## 5. Synonyms (code/synonyms)

Candidate pairs come from several channels, an Opus judge decides each pair, and `build_synmap3.py` builds the merge map with a chaining guard, keep-apart constraints and ambiguous-acronym isolation. `apply_hand.py` then applies the hand review. The map is loaded as a two-column table `(kid, canon)` and passed to step 4 as `--synmap`. [synonyms.md](synonyms.md) walks through the five rounds and the hand review. The final map is `data/synonyms/synmap9_final.jsonl.gz`, and every released verdict is in `data/synonyms/judgments/`.

The final command chain, as run (`code/synonyms/runs/final_n.sh`):

```
python3 build_synmap3.py --extra-judged data/judged_twins.jsonl data/judged_wordorder.jsonl data/judged_alt2.jsonl \
    --extra-maps data/synmap_acronyms_small_ok.jsonl data/synmap_irregular.jsonl --rejected data/hi_reverified.jsonl --out data/synmap8.jsonl
python3 apply_hand.py                               # synmap8 + hand decisions + spacing variants -> data/synmap9.jsonl
# upload synmap9 as openalex.transient.kw1322v3_synmap9 (kid, canon); rebuild the vocabulary (step 4, --from-step 2)
uv run --with anthropic --with httpx python audit.py --prefix kw1322v3n_ --n 1000 --seed 29   # residual split audit
uv run --with anthropic --with httpx python audit.py --prefix kw1322v3n_ --n 1000 --seed 31
```

## 6. Publish (code/publish)

`gen_build.py` writes the SQL that builds the two served tables from the vocabulary build: `openalex.works.work_keywords_v2` (per work, `ARRAY<STRUCT<id, display_name, score FLOAT>>` in the model's order) and `openalex.common.keywords_v2` (the entity table). `score` is the keyword's confidence, exp(mean token log-prob), no longer a rank. The rest of the OpenAlex write path (walden notebooks, Elasticsearch and database sync) lives in OpenAlex's internal pipeline and is not part of this repository.

## 7. Evaluation (code/eval)

See [evaluation.md](evaluation.md). Short version:

```
python fetch_ext.py                                     # KP20k 2,000-doc sample (seed 1322) + Inspec 500 from Hugging Face
uv run --with modal --with numpy --with nltk python student_arm.py   # tag every harness work with the shipped student
python score.py --gold akw_field --arm arms/akw_field__student_v3fix.jsonl --top 8   # etc.; leaderboard.py collects rows
```
