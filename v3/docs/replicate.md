# Replicating v3

This page is for a person or an AI agent. Run the steps in order, and check each result against the numbers given before moving on.

## Levels of replication

| level | needs | reproduces |
|---|---|---|
| **A. Check the claims** | Python 3.11, this repository | teacher-imitation F1, synonym map statistics, vocabulary statistics, rule tables |
| **B. Re-score the benchmarks** | A + a GPU with the student weights (or `data/eval/student_v3fix_harness_tags.json.gz`) | KP20k / Inspec (fetched), MeSH and judge-gold rows (labels in `data/eval/golds/`); the akw labels are private |
| **C. Rebuild from teacher labels** | the released teacher labels (LARGE_FILES), 8 GPUs with ≥ 141 GB each for 2 h, a tagging cluster, a SQL engine over OpenAlex | student, corpus tags, vocabulary |
| **D. Full rerun** | C + an Anthropic API key with Opus 5.5 access | everything, including new teacher labels and synonym verdicts |

Opus outputs are not deterministic, so a level D rerun will not match our labels exactly. Compare distributions (keywords per work, empty share, judged precision), not individual lists.

## Environment

```
pip install uv            # scripts are run with `uv run --with <deps> python <script>`
pip install modal && modal setup          # GPU steps
databricks auth login     # SQL steps; code/common/dbx.py shells out to the databricks CLI
export CLAUDE_API_KEY=...                 # or ANTHROPIC_API_KEY
export DATABRICKS_WAREHOUSE_ID=...        # any SQL warehouse
modal volume create oxjob1322             # models, training data, HF cache at /vol
modal secret create oxjob-databricks DATABRICKS_HOST=... DATABRICKS_TOKEN=... DATABRICKS_WAREHOUSE_ID=...
```

Python dependencies by stage: teacher `anthropic`; student `torch==2.6.0 transformers>=4.51,<5 accelerate>=1.0` (pinned inside `modal_student.py`); tagging `vllm>=0.10` on CUDA 12.8 (pinned inside `modal_corpus_v3.py`); synonyms `sentence-transformers>=3.0`; eval `numpy nltk huggingface_hub`.

## Level A in five commands

```
cd v3
# 1. Student vs teacher, F1 over kid0 sets on the held-out works (expect ≈ 0.568)
python3 - <<'EOF'
import json, sys; sys.path.insert(0, 'code/common'); from kid import kid0
T = {r['id']: r['keywords'] for r in map(json.loads, open('data/teacher/teacher_labels_eval.jsonl')) if r['keywords'] is not None}
S = {r['id']: r['keywords'] for r in map(json.loads, open('data/student/student_v3fix_eval_outputs.jsonl'))}
f = []
for i in S.keys() & T.keys():
    g = {kid0(k) for k in T[i] if kid0(k)}; p = {kid0(k) for k in S[i] if kid0(k)}
    f.append(1.0 if not g and not p else 2 * len(g & p) / (len(g) + len(p)))
print(len(f), sum(f) / len(f))
EOF
# 2. Vocabulary size and head
gzip -dc data/vocabulary/kw1322v3n_vocabulary.tsv.gz | wc -l          # 1,874,461 incl. header
gzip -dc data/vocabulary/kw1322v3n_vocabulary.tsv.gz | head
# 3. Synonym map size
gzip -dc data/synonyms/synmap9_final.jsonl.gz | wc -l                 # 362,272
# 4. Hand decisions
python3 -c "import json,collections; print(collections.Counter((r['by'], r['d']) for r in map(json.loads, open('data/synonyms/hand_decisions.jsonl'))))"
# 5. Benchmark rows as scored
cat data/eval/results/akw_field__student_v3fix.json
```

## Level C / D order of operations

Follow [pipeline.md](pipeline.md) steps 1 to 7. Checks after each step:

| after | check | expect |
|---|---|---|
| 1b teacher | answered share, empty share, mean keywords per answered work | 99.9%, 9.3%, 7.2 |
| 2b student | `step 50` loss / lr in the training log; final lr near 0; held-out F1 vs teacher | loss ≈ 2.1 → 0.7; lr warms to 5e-5 in about 425 steps; ≈ 0.568 |
| 3b tagging | works tagged, empty share, mentions per work | 474.3M, 10.0%, 7.4 |
| 4 vocabulary | fates table (`<prefix>mentions_kept` grouped by fate), vocabulary size, coverage | see [vocabulary.md](vocabulary.md); ≈ 1.87M; 88.6% |
| 5 synonyms | `audit.py` works-weighted split share; 400-merge precision | ≈ 11% raw / ≈ 4% excluding isolated acronyms; ≈ 86% |

Things that bit us, in case they bite you:

- **Accelerate scheduler:** size the LR schedule in process-steps (already fixed in `student_gen_train_ddp.py`). With the bug, warmup is 8× too short on 8 GPUs.
- **vLLM keeps the stop token** in `token_ids`. Skip special tokens when splitting keywords (`keyword_lps` does), or every list's last keyword gets an extra token and a skewed log-prob.
- **Batch reads from a shared SQL warehouse:** 49 GPU workers reading shards from one small warehouse queued into timeouts while holding GPUs. Use a large warehouse and a per-statement deadline with server-side cancel (`run_shard` does).
- **Large prompt batches in one vLLM call** starved Modal's heartbeat. `_tag` chunks by 16,384.
- **Anthropic credit balance** can run out mid-batch. `judge_batch.py submit` requeues those rows.
- **Plural folding:** never strip a final *s* blindly (see [vocabulary.md](vocabulary.md)).
- **Acronym detection:** decide from the display name (≥ 2 capitals, no spaces), not from the lowercase id. The id alone made "elderly" an acronym.

## For AI agents

- Start from `README.md`, then this file, then [pipeline.md](pipeline.md). Every script has a docstring with its command line at the top.
- Scripts are resumable: they skip ids already in their output files. Rerunning after a failure is safe; deleting outputs is not.
- Scripts that call the Anthropic API print a running cost estimate from hard-coded list prices. Check current pricing before a level D run.
- `code/synonyms/runs/` records ad hoc steps exactly as run. Treat them as a log, not as a program to run blindly: they overwrite intermediate files.
