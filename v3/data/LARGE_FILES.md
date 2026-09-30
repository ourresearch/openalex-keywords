# Files too large for git, and files kept private

Nothing below is uploaded yet; links will be added here when the uploads are live. Planned homes: the **student weights** go to the Hugging Face model repo `OpenAlex/keywords-v3-qwen3-4b`; the **teacher labels, the twins verdicts and the intermediate vocabularies** go to the Hugging Face dataset `OpenAlex/keywords-v3-data`.

| artifact | size | where it is now | planned home | why it matters |
|---|---|---|---|---|
| **Student weights** `student_qwen4b_v3fix` (Qwen3-4B full fine-tune, bf16 safetensors + tokenizer) | 7.5 GiB | Modal Volume `oxjob1322`, `/models/student_qwen4b_v3fix` | Hugging Face model `OpenAlex/keywords-v3-qwen3-4b` (planned, not uploaded yet) | Lets anyone tag new text exactly as OpenAlex does, without retraining. Base model license Apache-2.0. |
| Teacher labels, all 944,707 requests (`god_v3.jsonl`: id, model, effort, keywords, usage) | 307 MB (77 MB gzipped) | job scratch (`god/results/god_v3.jsonl`) | HF dataset `OpenAlex/keywords-v3-data` (planned) | Retrains the student without Opus calls (level C). Only the 13,677 held-out rows are in this repo. |
| Student training / eval files (`student_train_v3.jsonl`, `student_eval_v3.jsonl`: id, prompt text, target) | 815 MB + 12 MB | Modal Volume and job scratch | not needed if the labels are published: rebuild from labels + OpenAlex text with `student_gen_data.py` | Contains titles and abstracts. |
| Corpus tagger output (`keywords_student_v3fix`: id, keywords, per-keyword log-prob, 474.3M works) | ≈ 45 GB gzipped JSONL (4,000 files) | Databricks Volume + table `openalex.transient.keywords_student_v3fix` | not planned for now | Raw model output before the vocabulary rules; lets others build their own vocabulary. |
| Served work keywords (`openalex.works.work_keywords_v2`) | 420.9M works, 2.37B rows | Databricks | already public through the OpenAlex API and snapshot | Nothing to add. |
| Synonym round-4 "twins" verdicts (`judged_twins.jsonl`, 1,553,192 pairs) | 192 MB (36.6 MB gzipped) | job scratch | HF dataset `OpenAlex/keywords-v3-data` (planned) | Needed to rebuild `synmap9` exactly from verdicts. Every other verdict file is in `data/synonyms/judgments/`. |
| Intermediate vocabularies used for hub choice in `build_synmap3.py` (`vocab.jsonl`, `vocab_f/g/h/i.jsonl`, `vocab_all.jsonl`) | 175 to 237 MB each | job scratch | HF dataset `OpenAlex/keywords-v3-data` (planned) | Works counts that decide which heading a merge group gets. |
| bge-m3 neighbour pairs (`pairs*.jsonl`) | 0.3 to 1.5 GB each | job scratch | optional | Candidate generation only; `modal_embed_knn.py` regenerates them in minutes. |
| **Author-keyword gold slices** (akw_field, akw_pubmed, akw_landing, akw_repo: titles, abstracts, author keywords) | ≈ 40 MB | OpenAlex job scratch (#1306, #1322) | **kept private** | Author keywords stay internal (standing OpenAlex decision). Their ids, splits and scores are published; akw_field can be rebuilt with `pull_akw_field.py` by anyone with OpenAlex data access. |

Also not published, on purpose: work text (titles, abstracts). The repository contains no abstract text; every file identifies works by OpenAlex id (or PMID for the MeSH gold). Fetch the text from the OpenAlex API or snapshot. The MeSH and judge golds are in git (`data/eval/golds/`), with ids and labels only.
