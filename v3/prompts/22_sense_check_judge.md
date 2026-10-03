# Per-work sense judge (gold labels, per-sense gate, recheck)

- **Used for:** Every per-work sense label in the senses step, with one prompt (`senses_v1`) so the label sets combine:
  - the blind check of the field-based shortcut (2,775 works the field rule moved; 42% right, so it never shipped);
  - the gold set for the Jev classifier (3,437 random works carrying a candidate keyword, 30 per keyword);
  - the per-sense gate (7,877 corpus moves, 50 per minority sense or all of them if fewer);
  - the near-miss recheck (1,650 fresh corpus moves, 150 for each of 11 senses).
- **Source:** `code/senses/gold_label.py` (gold, gate and recheck) and `code/senses/senses_validate.py` (field-rule check); the text below is extracted verbatim from the code
- **Model:** `claude-opus-5-5`
- **Parameters:** effort `medium`, `max_tokens` 4000; one call per (work, keyword); no system prompt. Refusals, errors and out-of-range answers are kept as unlabelled and left out of every denominator (14 of 12,964 calls in gold, gate and recheck).

## User message

```text
Keyword: {keyword id with hyphens as spaces}
Work title: {title}
Abstract: {abstract[:1500] or "(none)"}

Which meaning of the keyword does this work use?
1. {sense label}
2. {sense label}
...  (the keyword's senses from the discovery judge, largest first: 1 = the main sense)
{n}. none of these / not about this keyword
Answer with the number.
```

## Output schema (structured outputs, `output_config.format = json_schema`)

```json
{
 "type": "object",
 "properties": {
  "choice": {
   "type": "integer"
  },
  "why": {
   "type": "string"
  }
 },
 "required": [
  "choice",
  "why"
 ],
 "additionalProperties": false
}
```

`choice` is 1-based in the answer and stored 0-based (`choice` in the released files: 0 = main sense, last = none). The released label files (`data/senses/gold_labels_*.jsonl.gz`, `gate_labels.jsonl.gz`, `recheck_labels.jsonl.gz`) keep the options and the choice and drop `why`, which paraphrases the abstract.

Caveat: Opus wrote the sense labels (prompt 20) and also judges which label a work uses, so a label that is vague to Opus is vague in both places. The cross-family check below is the guard against that.

## Second-opinion judge (different model family and prompt)

- **Used for:** The outside check of the gold labels on 500 random gold works (hash order). Agreement: Opus with this judge 0.968, Opus with Jev 0.946, this judge with Jev 0.952. Of 79 Jev moves in the 500, 75 are right by Opus, 73 by this judge, 73 by both. Labels: `data/senses/astra_labels.jsonl.gz`.
- **Source:** `code/senses/astra_senses.py`
- **Model:** GPT-6 Astra. The released labels (2026-09-30) came from OpenAI's API as `gpt-6-astra`; the script now calls the same model through OpenRouter (`openai/gpt-6-astra`).
- **Parameters:** reasoning effort `medium`, `max_tokens` 16000; strict JSON schema; one call per work

```text
The keyword "{keyword id with hyphens as spaces}" has these distinct meanings in scholarly literature:
- "{sense label}": the sense used in works from {up to 4 fields the discovery judge listed for it, or "various fields"}
...
- "none": the work uses none of these meanings or is not about the keyword.

Read the work and say which meaning it uses. Base it on what the work is about, not on its field.

Title: {title}
Venue: {venue or "unknown"}
Abstract: {abstract[:1500] or "(none)"}
```

```json
{
 "type": "object",
 "properties": {
  "meaning": {
   "type": "string",
   "enum": ["{sense label}", "...", "none"]
  }
 },
 "required": [
  "meaning"
 ],
 "additionalProperties": false
}
```
