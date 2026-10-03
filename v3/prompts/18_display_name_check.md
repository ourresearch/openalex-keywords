# Heading display-name check (v3.1)

- **Used for:** Checking the 39,221 headings whose display name came from a merged member; a wrong one hides a bad merge from the string audit (122 found)
- **Source:** `code/synonym_audit/judge_display.py` (variable `SYSTEM`)
- **Model:** `claude-opus-5-5` (Anthropic Messages API)
- **Parameters:** effort `low`, `max_tokens` 16000, 100 headings per call

## System prompt

```text
Each line gives a subject heading's identifier (a slug of its canonical name) and the display name the index shows for it. Answer "ok" if the display name names the same subject (synonym, variant, acronym of it, spelling), "wrong" if it names a different subject (e.g. an acronym clash: cardiac-magnetic-resonance-imaging shown as "civil-military relations"). Return JSON only.
```

## User message

```text
1. id: {heading id}  |  display: {display name}
2. ...
```

## Output schema (structured outputs, strict JSON schema)

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
     "v": {
      "type": "string",
      "enum": [
       "ok",
       "wrong"
      ]
     }
    },
    "required": [
     "n",
     "v"
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
