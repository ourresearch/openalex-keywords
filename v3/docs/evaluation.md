# Evaluation

There are two harnesses. One scores against human labels, the other against the teacher.

## Human-label harness (code/eval)

It was built on 2026-09-22 to compare keyword designs on one benchmark, before the distillation route existed. The student entered it later as one more arm (`student_arm.py`).

| gold | works (all splits) | labels | metric | released here |
|---|---|---|---|---|
| akw_field | 6,180, stratified by field (non-xpac, 2015+, with abstract) | author-supplied keywords (PubMed KeywordList, landing-page `citation_keywords`) | top-8 F1, exact after canon | splits (ids) + scores |
| akw_pubmed | ≈ 2,000 | PubMed author keywords | top-8 F1 | splits + scores |
| akw_landing | 1,684 | landing-page author keywords | top-8 F1 | splits + scores |
| akw_repo | 416 | repository subject keywords | top-8 F1 | splits + scores |
| akw_kp20k | the KP20k sample | KP20k keyphrases treated as author keywords | top-8 F1 | via `fetch_ext.py` |
| kp20k | 2,000 random of the 20K test set (seed 1322) | present / absent keyphrases (PRMU), Porter-stemmed | F1@5 | via `fetch_ext.py` (Hugging Face `taln-ls2n/kp20k`) |
| inspec | 500 (test) | indexer phrases, Porter-stemmed | F1@5 | via `fetch_ext.py` (`taln-ls2n/inspec`) |
| mesh | 4,728 human-indexed PubMed works 2017 to 2022 | MeSH descriptors (views a / b / c) | micro F1, threshold fit on train+val | **labels** (`data/eval/golds/j1306/`) + splits + scores |
| judge | 1,198 | Fable-judged relevant keywords (gold v1.2) | P / R / F1, unjudged share | **labels** (`data/eval/golds/j1292/`, `j1322/`) + splits + scores |
| tibkat | 2,000 English TIBKAT records | GND codes | skipped for v3 (German headings) | splits |

The **author-keyword composite** is the mean test top-8 F1 over the five akw slices. It is the headline number because it is the only one built from human labels that describe what each work is about.

Splits: `data/eval/splits/<gold>.json` maps work id → train / val / test (50 / 20 / 30, seed 1322, stratified). Scores: `data/eval/results/<gold>__student_v3fix.json` and `__today.json` (v2 production), plus the full leaderboard of earlier arms in `data/eval/leaderboard.md`. The student's raw output on every harness work is in `data/eval/student_v3fix_harness_tags.json.gz` (`{work_id: [[keyword, mean log-prob], ...]}`).

**Released labels:** the MeSH gold (4,728 PubMed works identified by PMID, with their MeSH headings; MeSH and PubMed indexing are NLM public-domain data) and the judge gold v1.2 (1,198 works: relevant and chosen keywords, the candidate pools and delta candidates the judge saw), plus the MeSH 2026 descriptor table the loader needs. Ids and labels only: no titles or abstracts. Scoring the released student tags against them reproduces the published rows exactly (MeSH view a test F1 0.2227; judge random-set test F1 0.2682).

**Author-keyword golds (released):** all four akw slices, the headline benchmark: `akw_field` (6,180 works, `data/eval/golds/j1322/akw_field.jsonl.gz`) and `akw_pubmed`, `akw_landing`, `akw_repo` (2,000 works each, `data/eval/golds/j1306/slices/`). Each row has the OpenAlex work id, venue, language, year, source, the author keywords (from PubMed or the publisher's landing page / repository record) and the old OpenAlex keywords (`current`). Titles and abstracts are blank (`null`); fetch them from the OpenAlex API or snapshot by id.

Commands:

```
cd code/eval
python fetch_ext.py                                        # data/kp20k.jsonl, data/inspec.jsonl
uv run --with modal --with numpy --with nltk python student_arm.py   # needs the deployed tagging app and the model on the Volume
python score.py --gold kp20k --arm arms/kp20k__student_v3fix.jsonl --json results/kp20k__student_v3fix.json --label student_v3fix
python leaderboard.py                                      # -> leaderboard.md
```

`golds.py` loads each gold from `KW_GOLDS_DIR` (default `v3/data/eval/golds/`), one folder per originating OpenAlex job: `j1292` + `j1322` (judge gold), `j1306` (MeSH sample and descriptors; the private slices and TIBKAT would go here too). Files over 5 MB are gzipped and read transparently. Splits are read from `v3/data/eval/splits/`. `normalize_kw.py` is the shared `canon()` used for exact matching.

## Teacher harness (code/student, code/teacher/pilot2, code/tagging)

- **F1 vs teacher** on the 13,338 held-out works: compare `data/student/student_v3fix_eval_outputs.jsonl` with `data/teacher/teacher_labels_eval.jsonl` (sets of `kid0`; an empty-vs-empty work scores 1).
- **Judged precision:** the strict indexer rubric ([prompt](../prompts/04_judge_keywords.md)). Each judge call pools one work's lists so that verdicts are shared and blind to which list a keyword came from.
- **Prompt pilot:** 620 fixed works in six groups (abstract, short abstract, title only, non-English, author-keyword works, xpac over-drawn from hard sources). Seven prompts, arm-blind judge. Data in `data/teacher/prompt_pilot/`.
- **Retag check:** `tagging/judge_retag.py` judges the corpus run's own output on the same 1,000 works, so a confidence threshold can be read against judged precision (`data/tagging/judged_retag_fix.jsonl`).
