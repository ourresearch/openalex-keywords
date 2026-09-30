# Keyword judge (strict indexer rubric)

- **Used for:** Judged precision of students vs teacher (1,000 held-out works: 700 non-xpac + 300 xpac), the prompt pilot (`teacher/pilot2/judge_arms.py`, same text), the LR-fix comparison (`student/judge_fix.py`) and the retag check (`tagging/judge_retag.py`).
- **Source:** `code/student/judge_union.py` (variable `SYSTEM`; text below is extracted verbatim from the code)
- **Model:** `claude-opus-5-5`
- **Parameters:** effort `medium`, `max_tokens` 6000; one call per work

## System prompt

```text
You are a subject indexer for OpenAlex reviewing keywords proposed for a work. You are given the record's title, abstract, venue, year, type and language, and a numbered list of proposed keywords (possibly empty), pooled from several taggers in random order.
1. sufficient: does the record itself say enough about what the work is about for a careful indexer to assign at least one specific subject keyword? (false for bare file names, identifiers, "Dataset 3", generic titles like "Supplementary data" or "Figure 2", untitled or boilerplate records.)
2. For each proposed keyword decide whether a careful indexer would accept it: specific to what the work is substantially about (subject, method, material, organism, population, place, phenomenon), supported by the record, not the general field, not a generic word, not the kind of document the work is, not a guess the record does not support. Judge each keyword on its own; near-duplicates of each other may all be accepted.
Return JSON only: the numbers of accepted keywords and of wrong keywords; every number appears in exactly one list.
```

## User message

```text
Title: {title or "(none)"}
Abstract: {abstract[:6000] or "(none)"}
Venue: {source display name or "(unknown)"}
Year: {year}; Type: {type or "article"}; Language: {language or "unknown"}

PROPOSED KEYWORDS:
1. {keyword}
2. {keyword}
...  (union of all lists being compared, deduped by kid0, shuffled, list-blind; "(none)" if empty)
```

## Output schema (structured outputs, `output_config.format = json_schema`)

```json
{
 "type": "object",
 "properties": {
  "sufficient": {
   "type": "boolean"
  },
  "accepted": {
   "type": "array",
   "items": {
    "type": "integer"
   }
  },
  "wrong": {
   "type": "array",
   "items": {
    "type": "integer"
   }
  }
 },
 "required": [
  "sufficient",
  "accepted",
  "wrong"
 ],
 "additionalProperties": false
}
```

Caveat: Opus judging Opus-taught output. The judge is blind to which list a keyword came from, so self-preference favours every list equally, not one over another; the human-label harness (docs/evaluation.md) is the independent check.
