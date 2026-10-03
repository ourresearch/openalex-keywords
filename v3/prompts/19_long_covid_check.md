# Long COVID work-level check (v3.1)

- **Used for:** 40 works per source string merged into keywords/long-covid (359 judged): which strings name the illness and which the post-pandemic period
- **Source:** `code/synonym_audit/lc_judge.py` (variables `SYSTEM`, `subj`)
- **Model:** `claude-opus-5-5` (Anthropic Messages API)
- **Parameters:** effort `medium`, `max_tokens` 8000, one work per call

## System prompt

```text
You judge whether a scholarly work is about a given subject, for a precision audit of a subject index. 'yes' if the subject is a main topic, 'partly' if it is a clear secondary topic or substantial component, 'no' if absent, mentioned in passing, or a different sense of the words. Judge only from the record; if there is no abstract, judge from title and venue.
```

## User message

```text
Subject: "long COVID (post-COVID-19 condition: symptoms persisting after SARS-CoV-2 infection)"

Work:
Title: {title}
Venue: {venue}
Type: {type}; year: {year}
Abstract: {abstract, or "(none available)"}

Is this work substantially about the subject?
```

## Output schema (structured outputs, strict JSON schema)

```json
{
 "type": "object",
 "properties": {
  "verdict": {
   "type": "string",
   "enum": [
    "yes",
    "partly",
    "no"
   ]
  },
  "reason": {
   "type": "string"
  }
 },
 "required": [
  "verdict",
  "reason"
 ],
 "additionalProperties": false
}
```
