# Context check of high-impact merges (conservative)

- **Used for:** Every merge that moves a keyword on >= 10K works (1,575 + 1,018 + 3,446 + 134 checks); rejected merges are kept apart.
- **Source:** `code/synonyms/verify_hi.py` (variable `SYSTEM`; text below is extracted verbatim from the code)
- **Model:** `claude-opus-5-5`
- **Parameters:** effort `medium`, `max_tokens` 8000, 20 merges per call

## System prompt

```text
You check proposed merges in the subject vocabulary of a scholarly index. For each numbered item you see two keyword headings, each with three example works tagged with it (title [venue]). Merge only if, as used on these works, the two headings mean the same subject, so works under either belong under one heading. Different senses (plasma fueling vs vehicle refueling), broader vs narrower, or related-but-distinct concepts: do not merge. Return JSON only.
```

## User message

```text
1. A = "{heading}"
    - {example title} [{venue}]
    - ... (3 random works tagged with it)
   B = "{heading}"
    - ...
```

## Output schema (structured outputs, `output_config.format = json_schema`)

```json
{
 "type": "object",
 "properties": {
  "items": {
   "type": "array",
   "items": {
    "type": "object",
    "properties": {
     "n": {
      "type": "integer"
     },
     "merge": {
      "type": "boolean"
     }
    },
    "required": [
     "n",
     "merge"
    ],
    "additionalProperties": false
   }
  }
 },
 "required": [
  "items"
 ],
 "additionalProperties": false
}
```
