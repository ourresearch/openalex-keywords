# Alternative-name generation

- **Used for:** Generating synonym candidates embeddings miss (older adults / elderly, mpox / monkeypox, vitamin C / ascorbic acid) for the top 200K keywords; names found in the vocabulary become pairs for the v2 judge. Also the first channel of the residual audit.
- **Source:** `code/synonyms/altnames.py` (variable `SYSTEM`; text below is extracted verbatim from the code)
- **Model:** `claude-opus-5-5`
- **Parameters:** effort `low`, `max_tokens` 16000, 100 keywords per call

## System prompt

```text
For each numbered keyword from a scholarly subject vocabulary, list up to 6 other names researchers use for EXACTLY the same concept: synonyms, near-synonyms used interchangeably, older or newer names (monkeypox for mpox), common vs scientific names (ascorbic acid for vitamin C), abbreviations and expansions, British/American spellings. Not broader or narrower terms, not related concepts. Empty list if none. Return JSON only.
```

## User message

```text
1. {heading}
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
     "names": {
      "type": "array",
      "items": {
       "type": "string"
      }
     }
    },
    "required": [
     "n",
     "names"
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
