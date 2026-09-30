# Country evidence table

- **Used for:** Names, adjectives and native names per country (`data/vocabulary/countries.json`), used by the country-without-evidence rule.
- **Source:** `code/vocabulary/build_countries.py` (variable `SYSTEM`; text below is extracted verbatim from the code)
- **Model:** `claude-opus-5-5`
- **Parameters:** effort `medium`, `max_tokens` 16000

## System prompt

```text
You build a lookup table used to check whether a scholarly work's title or abstract mentions a country. For each country (ISO 3166-1 code, name, official name) give:
- aliases: lower-case English strings that, used as a whole keyword on a paper, mean this country: the common name, official name, short forms and abbreviations in real use (e.g. united states: "united states", "usa", "us", "u.s.", "u.s.a.", "united states of america"; united kingdom: "uk", "u.k.", "britain", "great britain"; south korea: "korea", "republic of korea", "south korea"). Include former names still used in the literature only if they refer to the same country (e.g. "burma" for Myanmar, "swaziland" for Eswatini, "czech republic"/"czechia"). No cities, regions or languages.
- adjectives: lower-case English adjective / demonym forms (e.g. "japanese", "indonesian", "brazilian"). Omit an adjective if it is also a very common English word unrelated to the country (e.g. "turkish" is fine, but skip "french fries"-style phrases; single words only).
- native: the country's own name(s) in its official or main languages, in their native scripts, lower-case where the script has case (e.g. japan: "日本", "にほん", "nippon"; germany: "deutschland"; spain: "españa"; china: "中国", "中國"; south korea: "한국", "대한민국"; russia: "россия"; indonesia: "indonesia"; brazil: "brasil"), plus the native adjective forms when common in titles (e.g. "deutsch", "español", "española", "brasileiro", "brasileira", "français", "française", "русский", "российской").
Be complete for every country given; return every index.
```

## User message

```text
index<TAB>code<TAB>name<TAB>official name
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
     "aliases": {
      "type": "array",
      "items": {
       "type": "string"
      }
     },
     "adjectives": {
      "type": "array",
      "items": {
       "type": "string"
      }
     },
     "native": {
      "type": "array",
      "items": {
       "type": "string"
      }
     }
    },
    "required": [
     "i",
     "aliases",
     "adjectives",
     "native"
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
