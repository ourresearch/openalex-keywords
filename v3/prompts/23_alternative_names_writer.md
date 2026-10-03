# Alternative-name writer (GPT-6.1 Sol with Wikidata hints)

- **Used for:** Writing `display_name_alternatives` for every keyword: 1,882,007 keywords sent, 1,881,983 answered, 6,973,242 synonyms written (81,668 keywords got none). After the one-owner step, 1,701,306 keywords carry 5,091,086 alternatives. The same prompt, without the Wikidata paragraph, was the generation prompt for every arm of the bake-off (see the table below and [docs/alternatives.md](../docs/alternatives.md)).
- **Source:** `code/alternatives/syn_gen.py` (variables `SYS`, `WD_HINT`, `SCHEMA`, `fmt`; text below is extracted verbatim from the code). The full run is `code/alternatives/syn_sweep.py`, which sends `SYS + WD_HINT`.
- **Model:** `openai/gpt-6.1-sol` through OpenRouter (chat completions, `response_format` = strict JSON schema). During one 24-minute stretch of the full run, 264,162 keywords (14%) went to the OpenAI API directly with the same model id (`gpt-6.1-sol`), prompt and settings.
- **Parameters:** reasoning effort `low`, `max_tokens` 4000 in the full run (the bake-off sent no `max_tokens`), 10 keywords per call, prompt version `syn_v1_sol_wd`.
- **Input per keyword:** the keyword's display name, its description (the OpenAlex keyword description, written by GPT-6 Luna), and, for the 225,141 keywords whose Wikidata item has English aliases (414,167 have a Wikidata id), those aliases plus the English label, minus any equal to the keyword's name. Wikidata dump of 2026-09-03.

## System prompt

The full run sends the first block followed by the second. Bake-off arms without Wikidata hints sent the first block alone.

```text
You write search synonyms for keywords in OpenAlex, an index of scholarly works. People type into a keyword search and autocomplete box; a synonym is any string a researcher would plausibly type when they mean exactly this keyword's concept, so the keyword can be found by it.

Include:
- exact synonyms and alternative names for the same concept;
- abbreviations and acronyms in common use for this concept, and expansions of the keyword if it is itself an abbreviation;
- spelling variants (British/American, hyphenation, spacing, Greek letters spelled out, singular/plural where people type both);
- everyday or lay terms for a technical concept ("heart attack" for myocardial infarction);
- common alternative wordings and word orders; former or alternative names.

Exclude: broader or narrower concepts; related but different concepts; translations into other languages; the keyword itself or a case-only change of it; anything invented. Keep the same specificity as the keyword. Use the description to pick the right sense.

Give 0 to 12 synonyms per keyword, the most likely to be typed first. Give none rather than pad.
```

Appended when Wikidata aliases are given (`WD_HINT`):

```text
Some keywords list Wikidata aliases. They are human-written but noisy (some are broader, narrower, other-language or wrong): keep the ones that qualify as synonyms under the rules above and drop the rest.
```

## User message

Ten keywords per call, separated by a blank line. The `Wikidata aliases` line appears only when the keyword has some.

```text
id: {kid}
keyword: {display name}
description: {description, or "(none)"}
Wikidata aliases: {alias 1}; {alias 2}; ...

id: ...
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
     "synonyms"
    ],
    "properties": {
     "kid": {
      "type": "string"
     },
     "synonyms": {
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

## Bake-off arms (same prompt, 713 sampled keywords)

| arm | model id (OpenRouter) | Wikidata hints |
|---|---|---|
| luna | `openai/gpt-6-luna` | no |
| luna_pro | `openai/gpt-6-luna-pro` | no |
| gemini_flash | `google/gemini-3.8-flash` | no |
| sol | `openai/gpt-6.1-sol` | no |
| sonnet | `anthropic/claude-sonnet-5.5` | no |
| opus | `anthropic/claude-opus-5.5` | no |
| sol_wd (**chosen**) | `openai/gpt-6.1-sol` | yes |
| sol6_wd | `openai/gpt-6-sol` | yes |

All arms: reasoning effort `low`, no `max_tokens`, 10 keywords per call. Outputs are in `data/alternatives/bakeoff/syn_gen_<arm>.jsonl.gz`.
