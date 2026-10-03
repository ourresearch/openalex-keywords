# Per-work sense classifier (Jev)

- **Used for:** Deciding which sense of a split keyword each work uses. It ran on every (work, keyword) pair for the 115 candidate keywords: 15,898,677 pairs on 15,013,573 works (2026-09-30 to 10-01), plus the 6,211 gold works it was scored on. The Keywords nightly asks the same question for new works that carry a split keyword. A work moves to a minority sense when the top option is a minority sense (`s1`, `s2`, ...) with probability ≥ 0.5; `s0` (the main sense) and `none` leave the keyword as it is.
- **Source:** `code/senses/jev_senses.py` (gold), `code/senses/corpus_run.py` (corpus), `code/senses/sense_sql.py` (`SENSE_INSTR`, `sense_question`, `sense_state`, used by the nightly); the text below is extracted verbatim from the code
- **Model:** Jev, TypeSafe's hosted decision model, pinned to `jev-1.13.0` (native API `POST https://api.typesafe.ai/v1/systemone`, body `{"model", "state", "questions"}`). Jev is not a chat model: it takes a structured state and a set of typed questions and returns a probability for every option.
- **Parameters:** one request per (work, keyword); question type `choice`; no temperature, effort or token settings (the API has none); abstract cut to 1,500 characters

This is the only model in the senses step that is not Claude. It was chosen because it returns a probability for every option and is cheap and fast enough for a 16M-pair corpus run. The move threshold (0.5) was chosen on train keywords only.

## Question (`questions`)

```json
{
 "sense": {
  "type": "choice",
  "instructions": "Which meaning of the keyword does this scholarly work use? Judge from the title and abstract; the venue is a hint only.",
  "criteria": {
   "s0": "{main sense label}",
   "s1": "{minority sense label}",
   "...": "(one entry per sense the discovery judge named, largest first)",
   "none": "none of these meanings / the work is not about this keyword"
  }
 }
}
```

Example options, from `data/senses/sense_map.json`: for "inflation", `s0` = "inflation (economics)", `s1` = "inflation (cosmology)", `s2` = "inflation (mechanics)". Option text was the one thing allowed to be tuned (on train keywords only); version 1 cleared the bar, so the discovery judge's labels were used unchanged.

## State (`state`)

```json
{
 "keyword": "{keyword id with hyphens as spaces}",
 "title": "{title or empty}",
 "abstract": "{abstract[:1500] or empty}",
 "venue": "{primary source display name or empty}"
}
```

## Output

```json
{
 "answers": {
  "sense": {
   "type": "choice",
   "choice": "s0",
   "confidence": 0.99,
   "probabilities": {"s0": 1.0, "s1": 0.0, "none": 0.0}
  }
 },
 "model": "jev-1.13.0"
}
```

The corpus run stores only `probabilities` (rounded to 4 places) and `model`. Released: the gold-set answers in `data/senses/jev_v1_gold.jsonl.gz` and the moves that shipped in `data/senses/work_senses_shipped.tsv.gz`.

## How well it does

Scored against Opus 5.5 on 6,196 labelled works ([22](22_sense_check_judge.md)), on test keywords only: move precision 0.957 (Wilson 95% lower bound 0.941), main-sense recall 0.989, coverage of Opus-minority works 0.947 (`data/senses/score_v1.json`). On 500 random works, a second judge from another model family agreed with Opus on 0.968 of works and with Jev on 0.952; Opus and Jev agreed on 0.946.
