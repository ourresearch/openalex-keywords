# Outside judge: is this work about the subject? (search examples)

- **Used for:** The launch post's search examples: works the antimicrobial-resistance keyword finds that a title and abstract phrase search misses (150 sampled; 81% at least partly on topic), the same without an abstract (100; 74%), the phrase search itself (50), and the remote-work example (100 + 50). Also the pre-launch AMR run in `judge_gpt.py` (task `amr`)
- **Source:** `code/outside_judge/postswap_judge.py` (variables `SYSTEM`, `SCHEMA`; function `message`); identical to `judge_gpt.py` `AMR_SYSTEM`
- **Model:** `openai/gpt-6-astra` via OpenRouter (served by OpenAI), 2026-10-02
- **Parameters:** reasoning effort `high`, strict `json_schema`, `provider.require_parameters` true, one call per work, 300 s cap per call, 8 in flight

## System prompt

```text
You judge whether a scholarly work is about a given subject, for a precision audit of a subject index. Answer 'yes' if the subject is a main topic of the work (a reader searching for the subject would want it), 'partly' if the subject is a clear secondary topic or substantial component, 'no' if it is absent, only mentioned in passing, or the work is about a different sense of the words. Judge only from the record given; if there is no abstract, judge from title and venue and say so when that makes you unsure.
```

## User message

```text
Subject: "{subject: antimicrobial resistance | how remote work affects employee wellbeing}"

Work:
Title: {title or "(none)"}
Venue: {venue or "(unknown)"}
Type: {type}; year: {year}
Abstract: {abstract, first 2,500 chars (AMR) or 4,000 (remote work), or "(none available)"}

Is this work substantially about "{subject}"? Give verdict and a one-line reason.
```

## Output schema (structured outputs, strict JSON schema)

```json
{
 "type": "object",
 "additionalProperties": false,
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
 ]
}
```
