# Teacher prompt v3 (shipped)

- **Used for:** Labelling the ~1M-work teaching set (944,707 requests; 943,769 answered, 938 refusals). The student is trained on the labels, joined with "; " in the model's order.
- **Source:** `code/teacher/prompt_v3.py` (variable `SYSTEM`; text below is extracted verbatim from the code)
- **Model:** `claude-opus-5-5` via the Message Batches API; synchronous retries of refusals / errors used the `server-side-fallback-2026-07-01` beta with `fallbacks="default"` (6,225 of 943,769 labels came back from `claude-opus-5`, 6 from `claude-opus-4-8`)
- **Parameters:** `max_tokens` 4000; `output_config.effort` = `medium`; system prompt sent with `cache_control: ephemeral`; no temperature set (API default)

## System prompt

```text
Assign keywords that describe what this work is about. Describe its subject matter: not the kind of document or record it is, not its general field, not standard methods it merely uses, and not a country or place unless the work is about that place. Write the keywords in English. If the record says nothing about its subject, return an empty list.
```

## User message

```text
Title: {title or "(none)"}
Abstract: {abstract[:6000] or "(none)"}
Venue: {source display name or "(unknown)"}
```

## Output schema (structured outputs, `output_config.format = json_schema`)

```json
{
 "type": "object",
 "properties": {
  "keywords": {
   "type": "array",
   "items": {
    "type": "string"
   }
  }
 },
 "required": [
  "keywords"
 ],
 "additionalProperties": false
}
```

The same user template (title, abstract, venue; nothing else) is the student's input at training and at inference, followed by `\nKeywords:`.
