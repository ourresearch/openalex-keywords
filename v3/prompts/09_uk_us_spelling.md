# UK / US spelling confirmation

- **Used for:** Confirming 8,673 rule-generated UK/US pairs (8,478 merged).
- **Source:** `code/synonyms/judge_ukus.py` (variable `SYSTEM`; text below is extracted verbatim from the code)
- **Model:** `claude-opus-5-5`
- **Parameters:** effort `low`, `max_tokens` 8000, 100 pairs per call

## System prompt

```text
Each line gives two keyword headings from a scholarly subject vocabulary that differ by British vs American spelling. Merge (true) only if they are the same concept spelled two ways, so works under either belong under one heading. Do NOT merge (false) when either is a proper name that must keep its own spelling (a political party such as the UK Labour Party, a Latin taxon name such as Haemophilus or Haemonchus, an organisation, a place, a title of a work) or when the words mean different things. Return JSON only.
```

## User message

```text
1. {UK form}  |  {US form}
2. ...
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
