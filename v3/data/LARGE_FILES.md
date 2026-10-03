# Files too large for git

Everything hosted is on GitHub, as assets of the [`v3.0` release](https://github.com/ourresearch/openalex-keywords/releases/tag/v3.0): the **student weights** (split into files under 2 GB each; see [`docs/weights.md`](../docs/weights.md)), the **teacher labels**, the **twins verdicts** and the **intermediate vocabularies**. Download with `gh release download v3.0 -R ourresearch/openalex-keywords` (add `-p '<pattern>'` for a subset).

| artifact | size | where it is now | planned home | why it matters |
|---|---|---|---|---|
| **Student weights** `student_qwen4b_v3fix` (Qwen3-4B full fine-tune, bf16 safetensors + tokenizer) | 7.5 GiB | Modal Volume `oxjob1322`, `/models/student_qwen4b_v3fix` | release `v3.0`: `model-0000{1..5}-of-00005.safetensors` + config/tokenizer files | Lets anyone tag new text exactly as OpenAlex does, without retraining. Base model license Apache-2.0. |
| Teacher labels, all 944,707 requests (`god_v3.jsonl`: id, model, effort, keywords, usage) | 307 MB (77 MB gzipped) | job scratch (`god/results/god_v3.jsonl`) | release `v3.0`: `teacher_labels_v3.jsonl.gz` | Retrains the student without Opus calls (level C). Only the 13,677 held-out rows are in this repo. |
| Student training / eval files (`student_train_v3.jsonl`, `student_eval_v3.jsonl`: id, prompt text, target) | 815 MB + 12 MB | Modal Volume and job scratch | not needed if the labels are published: rebuild from labels + OpenAlex text with `student_gen_data.py` | Contains titles and abstracts. |
| Corpus tagger output (`keywords_student_v3fix`: id, keywords, per-keyword log-prob, 474.3M works) | ≈ 45 GB gzipped JSONL (4,000 files) | Databricks Volume + table `openalex.transient.keywords_student_v3fix` | not planned for now | Raw model output before the vocabulary rules; lets others build their own vocabulary. |
| Served work keywords (`openalex.works.work_keywords_v2`) | 420.9M works, 2.37B rows | Databricks | already public through the OpenAlex API and snapshot | Nothing to add. |
| Synonym round-4 "twins" verdicts (`judged_twins.jsonl`, 1,553,192 pairs) | 192 MB (36.6 MB gzipped) | job scratch | release `v3.0`: `judged_twins.jsonl.gz` | Needed to rebuild `synmap9` exactly from verdicts. Every other verdict file is in `data/synonyms/judgments/`. |
| Intermediate vocabularies used for hub choice in `build_synmap3.py` (`vocab.jsonl`, `vocab_f/g/h/i.jsonl`, `vocab_all.jsonl`) | 175 to 237 MB each | job scratch | release `v3.0`: `intermediate_vocab_*.jsonl.gz` | Works counts that decide which heading a merge group gets. |
| bge-m3 neighbour pairs (`pairs*.jsonl`) | 0.3 to 1.5 GB each | job scratch | optional | Candidate generation only; `modal_embed_knn.py` regenerates them in minutes. |

Also not published, on purpose: work text (titles, abstracts). The repository contains no abstract text; every file identifies works by OpenAlex id (or PMID for the MeSH gold). Fetch the text from the OpenAlex API or snapshot. The MeSH and judge golds are in git (`data/eval/golds/`), with ids and labels only.

## v3.1

Everything v3.1 adds fits in git, so the [`v3.1` release](https://github.com/ourresearch/openalex-keywords/releases/tag/v3.1) has no new assets; the weights and large files above stay on `v3.0`. Left out on purpose:

| artifact | size | why it is not here |
|---|---|---|
| Jev's sense decisions for all 15.9M (work, keyword) pairs, including senses that did not split | ≈ 1.8 GB | only the moves into the 89 split senses are released (`data/senses/work_senses_shipped.tsv.gz`); `code/senses/corpus_run.py` regenerates the rest |
| Alternative-name intermediates (`syn_input`, `syn_full`, `syn_owned`, `collisions`, `name_clash`) | 90 to 520 MB each | rebuilt by the scripts in `code/alternatives/` from the released vocabulary and synonym map |
| Real keyword-box queries used to rank alternative names | 400 queries | user data; only aggregate scores are released (`data/alternatives/eval/`) |
| The pre-launch antimicrobial-resistance sample of the outside judge | 256 works | its records carry titles and abstracts; the verdicts are in `data/eval/outside_judge/judgments.jsonl` (keys starting `R`) |
