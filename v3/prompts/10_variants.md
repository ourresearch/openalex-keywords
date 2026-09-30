# Spelling / hyphenation / acronym variant confirmation

- **Used for:** Confirming 44,948 string-key variant pairs and reviewer-found acronym pairs.
- **Source:** `code/synonyms/variants.py` (variable `SYSTEM`; text below is extracted verbatim from the code)
- **Model:** `claude-opus-5-5`
- **Parameters:** effort `low`, `max_tokens` 8000, 100 pairs per call

## System prompt

```text
Each line gives two keyword headings from a scholarly subject vocabulary. Merge (true) only if they name the same concept: spelling, hyphenation or spacing variants, singular vs plural, British vs American spelling, an acronym and its expansion when the acronym, as used in research, means that expansion. Do NOT merge (false) when one is broader or narrower, when they are different things that happen to look alike, when the acronym commonly means something else, or when either is a proper name that must keep its form (a party, an organisation, a Latin taxon, a place, a title). Return JSON only.
```

## User message

```text
1. {heading A}  |  {heading B}
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
