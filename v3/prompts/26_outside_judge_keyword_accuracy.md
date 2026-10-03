# Outside judge: per-keyword accuracy (launch numbers)

- **Used for:** Rating every keyword of the old (topic-derived) and new lists on 400 random works, each list judged on its own, shuffled, unlabelled: 91.1% of new keywords accurate vs 43.5% of old
- **Source:** `code/outside_judge/judge_gpt.py` (variables `RUBRIC`, `SINGLE_SCHEMA`; task `single`)
- **Model:** `gpt-6-astra` (OpenAI Responses API, called directly on 2026-09-30)
- **Parameters:** reasoning effort `high`, strict `json_schema`, one call per list (800 calls), up to 3 tries when the item count is wrong
- **Note:** The judge is from a different lab than the teacher (Claude Opus 5.5) and every earlier judge, so self-preference cannot lift the scores.

## System prompt

```text
Keywords in this index are meant to name what a work is ABOUT: its subjects, phenomena, the methods it is about, organisms, materials, diseases, places studied, people or texts it studies. They are used to aggregate what authors, journals, institutions and fields work on, and to help people find the work.

For each keyword, decide:
- accurate: it correctly and meaningfully describes what this work is about (a central or clearly present subject, method, entity, place, etc.). A broad-but-correct field term is accurate only if it is still informative for this work.
- inaccurate: any of
  - wrong_or_unsupported: the work is not about this, or the record gives no basis for it (plausible guesses beyond the text count as unsupported);
  - wrong_sense: the term is from the wrong field or word sense (e.g. "Transcription (linguistics)" on a gene transcription paper);
  - too_generic: so broad it tells a reader nothing about this work (e.g. "Biology", "Business", "Science" on a specific study);
  - not_aboutness: document type, venue, publisher boilerplate, or metadata rather than subject (e.g. "journal article", "peer review" on an ordinary paper, a bare year);
  - malformed: garbled or broken text, a stopword, or not a usable term.
Separately, mark redundant=true if the keyword is a near-duplicate of an earlier keyword in the same list (it can still be accurate).

Judge only from the record shown. If there is no abstract, judge from title, venue and type, and give the benefit of the doubt only where the title makes the subject clear.
```

## User message

```text
Record:
Type: {type}; Year: {publication_year}; Language: {language}
Venue/repository: {source_name or '(unknown)'}
Title: {title, first 400 chars, or '(none)'}
Abstract/description: {abstract, first 2,500 chars, or '(none available)'}

Keyword list ({n} keywords):
1. {keyword}
2. ...

Return one entry per keyword, in order, with n = its number.
```

## Output schema (structured outputs, strict JSON schema)

```json
{
 "type": "object",
 "additionalProperties": false,
 "properties": {
  "keywords": {
   "type": "array",
   "items": {
    "type": "object",
    "additionalProperties": false,
    "properties": {
     "n": {
      "type": "integer"
     },
     "keyword": {
      "type": "string"
     },
     "verdict": {
      "type": "string",
      "enum": [
       "accurate",
       "inaccurate"
      ]
     },
     "problem": {
      "type": "string",
      "enum": [
       "none",
       "wrong_or_unsupported",
       "wrong_sense",
       "too_generic",
       "not_aboutness",
       "malformed"
      ]
     },
     "redundant": {
      "type": "boolean"
     }
    },
    "required": [
     "n",
     "keyword",
     "verdict",
     "problem",
     "redundant"
    ]
   }
  },
  "note": {
   "type": "string"
  }
 },
 "required": [
  "keywords",
  "note"
 ]
}
```
