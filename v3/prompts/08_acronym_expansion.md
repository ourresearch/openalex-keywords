# Acronym expansion

- **Used for:** Expanding 22,264 acronym-like keywords (>= 200 works) and 17,501 small ones (100-199 works); expansions found in the vocabulary go to the v2 judge.
- **Source:** `code/synonyms/acronyms.py` (variable `SYSTEM`; text below is extracted verbatim from the code)
- **Model:** `claude-opus-5-5`
- **Parameters:** effort `low`, `max_tokens` 8000, 150 acronyms per call

## System prompt

```text
For each numbered keyword from a scholarly subject vocabulary that looks like an acronym or abbreviation, give its expansion as used in research literature, in standard English form (e.g. IL-6 -> interleukin-6, MRI -> magnetic resonance imaging, NDVI -> normalized difference vegetation index). Return null when: it has several common meanings across fields (PET, ATM, BIM, CAD); it is not an abbreviation; or the abbreviation itself is the standard name nobody expands (DNA, HIV, COVID-19, SARS-CoV-2, ChatGPT, XGBoost, microRNA). Return JSON only.
```

## User message

```text
1. {acronym-like keyword}
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
     "expansion": {
      "type": [
       "string",
       "null"
      ]
     }
    },
    "required": [
     "n",
     "expansion"
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
