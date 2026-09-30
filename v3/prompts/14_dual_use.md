# Dual-use exceptions to the purge

- **Used for:** Drafting `keyword_dual_use_v1` (2,812 rows; approved by hand 2026-09-25): purge keywords dropped only where the work's own type or design already says it.
- **Source:** `code/vocabulary/doctype_rules/judge_dual_use.py` (variable `SYSTEM`; text below is extracted verbatim from the code)
- **Model:** `claude-opus-5-5`
- **Parameters:** effort `medium`, `max_tokens` 16000, 150 keywords per call

## System prompt

```text
OpenAlex is removing keywords that label what KIND of document or study a work is (book review, erratum, randomized controlled trial, obituary), because those belong in structured filters. Some of these strings are also genuine research SUBJECTS: a paper can be ABOUT peer review, about research software, about clinical trial methodology, about obituaries as a genre. For those, OpenAlex will keep the keyword on works where it is the subject and drop it only where it merely labels the work.

For each keyword, answer dual_use = true if a meaningful number of scholarly works are plausibly ABOUT this thing as their subject (research on it, its history, its practice, its quality), not just instances of it. Answer false if the string is essentially only ever a label for the document itself (e.g. "erratum", "book review", "corrigendum", "retraction notice", "letter to the editor", "doctoral thesis", "case report", "phase ii clinical trial").
Evidence per line: the keyword, how many works carry it, what it maps to, and the work types it most often sits on. Be conservative: dual_use only when subject use is common, not merely conceivable.
```

## User message

```text
index<TAB>keyword<TAB>works<TAB>maps to<TAB>top work types
{rows}
```

## Output schema (structured outputs, `output_config.format = json_schema`)

```json
{
 "type": "object",
 "properties": {
  "v": {
   "type": "array",
   "items": {
    "type": "object",
    "properties": {
     "i": {
      "type": "integer"
     },
     "d": {
      "type": "boolean"
     }
    },
    "required": [
     "i",
     "d"
    ],
    "additionalProperties": false
   }
  }
 },
 "required": [
  "v"
 ],
 "additionalProperties": false
}
```
