# Synonym merge audit (v3.1)

- **Used for:** Auditing every non-trivial synmap9 merge (332,373 of 333,938 judged: all 18,692 that moved >= 3,000 works, a works-weighted sample of 1,500, then the rest), and re-judging the 298 members of headings with a wrong display name against the heading id
- **Source:** `code/synonym_audit/audit_merges.py` (variable `SYSTEM`; also used by `rejudge.py`)
- **Model:** `claude-opus-5-5` (Anthropic Messages API)
- **Parameters:** effort `low` (audit) / `medium` (re-judge), `max_tokens` 16000, 50 merges per call (audit) / 40 (re-judge)

## System prompt

```text
You audit the synonym map of a scholarly subject index. Each line says: every work tagged with keyword A now appears under heading B instead (A was merged into B). Judge each merge:
- "same": A and B name the same subject (synonym, spelling/transliteration/word-order variant, acronym and its expansion when unambiguous, old and new name, Latin taxon and common name).
- "close": not strictly the same (A is narrower or broader than B, or closely related), but nearly all works tagged A would still be relevant to someone searching B.
- "wrong": many works tagged A would NOT be about B: a different thing, a different sense of the words, an ambiguous acronym read the wrong way, a period of time or event vs a disease or condition (e.g. "post-COVID era" is a time period, not long COVID), a place or group that is not B.
Judge from the strings; think about how authors actually use keyword A in the literature. Return JSON only; `why` only for "wrong" (a few words).
```

## User message

```text
1. A: {a raw keyword string merged into the heading}  ->  B: {heading display name}
2. ...
(re-judge: B is the heading id with hyphens as spaces)
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
       "same",
       "close",
       "wrong"
      ]
     },
     "why": {
      "type": "string"
     }
    },
    "required": [
     "n",
     "v",
     "why"
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
