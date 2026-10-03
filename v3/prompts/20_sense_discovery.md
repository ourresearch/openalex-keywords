# Sense discovery judge (one subject or distinct senses)

- **Used for:** Finding keywords that mix unrelated meanings (homonyms). It judged the 3,000 biggest candidate keywords (156 have distinct senses, 5.2%) and a works-weighted sample of 1,000 smaller ones (996 answered, 77 distinct, 7.7%). Candidates are keywords on ≥ 300 works whose works span ≥ 2 domains, the second with ≥ 10% (`homonym_cands.py`). The sense labels it wrote became the options of the per-work classifier ([21](21_sense_classifier_jev.md)) and of the per-work judge ([22](22_sense_check_judge.md)), via `make_senses.py` and `sense_map.py`. Verdicts: `data/senses/homonym_judged.jsonl.gz` and `homonym_tail_judged.jsonl.gz` (rationales removed).
- **Source:** `code/senses/homonym_judge.py` (variable `SYSTEM`; text below is extracted verbatim from the code)
- **Model:** `claude-opus-5-5`
- **Parameters:** effort `medium`, `max_tokens` 8000; one call per keyword

## System prompt

```text
You curate a scholarly subject index. A keyword is attached to works in several fields. Decide whether the keyword names ONE subject
(possibly studied in several fields: "oxidative stress" in medicine and biochemistry is one subject) or DISTINCT SENSES that a searcher would never
want mixed ("transformer": the neural-network architecture vs the electrical device; "mercury": the element vs the planet; "depression": the mood
disorder vs an economic depression vs a topographic depression). If distinct senses, give each a short parenthetical label in Wikipedia style
("transformer (machine learning)"), the fields that carry it, and roughly what share of the keyword's works it covers. Judge from the field mix and
the example titles. Return JSON only.
```

## User message

```text
Keyword: {keyword, hyphens as spaces} ({keyword id URL})
Works: {works carrying the keyword}
Field mix:
- {field}: {share, as a whole percent}
...  (top 8 primary-topic fields by works)
Example titles by field:
[{field}] {title} | {title} | {title}
...  (3 titles from each of the keyword's top 3 fields, picked by a fixed hash, each cut to 160 characters)
```

## Output schema (structured outputs, `output_config.format = json_schema`)

```json
{
 "type": "object",
 "properties": {
  "verdict": {
   "type": "string",
   "enum": [
    "one_subject",
    "distinct_senses"
   ]
  },
  "senses": {
   "type": "array",
   "items": {
    "type": "object",
    "properties": {
     "label": {
      "type": "string"
     },
     "fields": {
      "type": "array",
      "items": {
       "type": "string"
      }
     },
     "share": {
      "type": "number"
     }
    },
    "required": [
     "label",
     "fields",
     "share"
    ],
    "additionalProperties": false
   }
  },
  "why": {
   "type": "string"
  }
 },
 "required": [
  "verdict",
  "senses",
  "why"
 ],
 "additionalProperties": false
}
```

Caveat: the judge sees only field shares and nine titles, so it finds senses that show up as a field split, and its field lists are not a usable per-work rule: a sense split by field alone was 42% right on a blind check (see [docs/senses.md](../docs/senses.md)). Which sense each work uses is decided by the per-work classifier. Shares are the judge's rough estimates and only order the options (largest first = the main sense, which keeps the keyword's id).
