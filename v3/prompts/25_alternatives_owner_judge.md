# Alternative-name owner judge

- **Used for:** Giving each contested alternative one owner. After the rule-based step (`syn_own.py`), 573,931 normalised strings were claimed by more than one keyword (1,319,905 claims). The judge picked one owner for each, or kept several owners when the string is a genuine homonym (CRP = C-reactive protein / cAMP receptor protein). Result: 573,316 strings given an owner, 17,030 of them homonyms kept on several keywords, 615 dropped (no owner, or still failing after one retry pass).
- **Source:** `code/alternatives/syn_own_judge.py` (variables `SYS`, `SCHEMA`, `fmt`; text below is extracted verbatim from the code)
- **Model:** `openai/gpt-6.1-sol` through OpenRouter (chat completions, `response_format` = strict JSON schema; the call is `oa_json` in `syn_sweep.py`)
- **Parameters:** reasoning effort `low`, `max_tokens` 4000, 10 contested strings per call
- **Checks:** an `owner` or `homonym_owners` id that is not among the string's candidates is discarded in code (`"none"` therefore means no owner).
- **Verdicts:** `data/alternatives/ownership/own_judged.jsonl.gz` (`{norm, owner, homonym_owners}`)

## System prompt

```text
You assign search synonyms to keywords in OpenAlex, an index of scholarly works. Each item is a string that several keywords claimed as a synonym. A person types the string into a keyword search box: decide which keyword they mean.
- owner: the single keyword they most likely mean (the most standard, exact match for the string). If several candidates are near-duplicates of each other (the same concept worded differently), pick the one with the most works.
- homonym_owners: ONLY if the string is a genuine homonym, i.e. it names clearly different concepts (typically an abbreviation used in different fields), list every other keyword that is a real meaning of it; otherwise an empty list. Near-duplicates, variants and related concepts are never homonyms.
- If no candidate is actually meant by the string, set owner to "none".
```

## User message

Ten strings per call, separated by a blank line. `{i}` is the item's position in the call. The string shown is the first claimant's spelling; the description is the keyword's OpenAlex description.

```text
id: {i}
string: {the contested alternative}
candidates:
- {kid}: {display name} ({works_count with thousands separators} works). {description}
- {kid}: ...
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
     "id",
     "owner",
     "homonym_owners"
    ],
    "properties": {
     "id": {
      "type": "string"
     },
     "owner": {
      "type": "string"
     },
     "homonym_owners": {
      "type": "array",
      "items": {
       "type": "string"
      }
     }
    }
   }
  }
 }
}
```
