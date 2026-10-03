# Alternative-name bake-off judge (exactness)

- **Used for:** Choosing the model that writes `display_name_alternatives`. For each of 713 sampled keywords, every candidate from the 8 bake-off arms plus the keyword's Wikidata English aliases and label was pooled, de-duplicated case-insensitively, shuffled, and labelled blind (no source shown): 9,103 candidates. Two judges from different model families labelled every candidate; they agree on 88.9%. A candidate counts as exact when both say SAME.
- **Source:** `code/alternatives/syn_judge.py` (variables `SYS`, `SCHEMA`, `fmt`; text below is extracted verbatim from the code); scores from `code/alternatives/syn_score.py`.
- **Models:** `anthropic/claude-opus-5.5` and `openai/gpt-6-astra`, both through OpenRouter (chat completions, `response_format` = strict JSON schema)
- **Parameters:** reasoning effort `medium`, no `max_tokens`, 5 keywords per call
- **Verdicts:** `data/alternatives/bakeoff/syn_judge_opus.jsonl.gz`, `syn_judge_astra.jsonl.gz` (`{kid, labels: {candidate: SAME | CLOSE | WRONG}, judge}`)

## System prompt

```text
You check proposed search synonyms for keywords in OpenAlex, an index of scholarly works. A researcher types a string into a keyword search box; it should find a keyword only if they mean exactly that keyword's concept. For each proposed synonym, answer:
- SAME: someone typing it means exactly this keyword's concept at the same specificity. Synonyms, alternative names, abbreviations or acronyms in real use for this concept (even if they also mean other things elsewhere), spelling/hyphenation/plural variants, lay terms, reworded phrases all count.
- CLOSE: a broader, narrower or merely related concept. Useful context, but a misleading synonym.
- WRONG: a different concept, a wrong sense, a non-English translation, or nonsense.
Use the description to pin the sense. Label every proposed string exactly as written.
```

## User message

Five keywords per call, separated by a blank line.

```text
id: {kid}
keyword: {display name}
description: {description, or "(none)"}
proposed synonyms:
- {candidate 1}
- {candidate 2}
...
```

## Output schema (strict JSON schema)

```json
{
 "type": "object",
 "additionalProperties": false,
 "required": [
  "items"
 ],
 "properties": {
  "items": {
   "type": "array",
   "items": {
    "type": "object",
    "additionalProperties": false,
    "required": [
     "kid",
     "labels"
    ],
    "properties": {
     "kid": {
      "type": "string"
     },
     "labels": {
      "type": "array",
      "items": {
       "type": "object",
       "additionalProperties": false,
       "required": [
        "synonym",
        "label"
       ],
       "properties": {
        "synonym": {
         "type": "string"
        },
        "label": {
         "type": "string",
         "enum": [
          "SAME",
          "CLOSE",
          "WRONG"
         ]
        }
       }
      }
     }
    }
   }
  }
 }
}
```
