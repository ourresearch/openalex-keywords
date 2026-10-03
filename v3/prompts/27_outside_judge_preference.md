# Outside judge: old vs new head-to-head (launch numbers)

- **Used for:** Both lists on the same 400 works as A and B, order randomised per work: new preferred on 395, old on 1, neither useful on 4
- **Source:** `code/outside_judge/judge_gpt.py` (variables `PREF_INSTR`, `PREF_SCHEMA`; task `pref`)
- **Model:** `gpt-6-astra` (OpenAI Responses API)
- **Parameters:** reasoning effort `high`, strict `json_schema`, one call per work

## System prompt

```text
You compare two keyword lists attached to the same scholarly record in a large bibliographic index. Keywords should name what the work is about (subjects, phenomena, methods studied, organisms, materials, places, entities). Bad keywords are wrong or unsupported, the wrong word sense, too generic to be informative, document type / venue / boilerplate, malformed, or near-duplicates. Judge only from the record shown.
```

## User message

```text
Record:
Type: {type}; Year: {publication_year}; Language: {language}
Venue/repository: {source_name or '(unknown)'}
Title: {title, first 400 chars, or '(none)'}
Abstract/description: {abstract, first 2,500 chars, or '(none available)'}

List A: {keywords; separated}

List B: {keywords; separated}

Which list better describes what this work is about? Answer A, B, tie (about equally good), or neither_useful (both are poor). reason: one sentence.
```

## Output schema (structured outputs, strict JSON schema)

```json
{
 "type": "object",
 "additionalProperties": false,
 "properties": {
  "preferred": {
   "type": "string",
   "enum": [
    "A",
    "B",
    "tie",
    "neither_useful"
   ]
  },
  "reason": {
   "type": "string"
  }
 },
 "required": [
  "preferred",
  "reason"
 ]
}
```
