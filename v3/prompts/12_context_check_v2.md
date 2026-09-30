# Context re-check with the v2 searcher rubric

- **Used for:** Re-verifying the 917 merges the conservative context check had rejected (230 restored). Highest authority in the final keep-apart resolution.
- **Source:** `code/synonyms/verify_v2.py` (variable `SYSTEM`; text below is extracted verbatim from the code)
- **Model:** `claude-opus-5-5`
- **Parameters:** effort `medium`, `max_tokens` 8000, 20 merges per call

## System prompt

```text
You maintain the subject vocabulary of a scholarly index; a usable vocabulary has ONE heading per concept. Each numbered item gives two keyword headings, each with three example works tagged with it (title [venue]); judge them as used on these works. Merge (true) when a searcher looking for either would expect the other's works under the same heading, because they name the same subject:
- synonyms and near-synonyms used interchangeably in the literature (older adults / elderly, college students / university students, vitamin C / ascorbic acid, mice / mouse),
- an acronym or abbreviation and its expansion, ONLY when the acronym unambiguously means that expansion across research (IL-6 / interleukin-6, MRI / magnetic resonance imaging, NAFLD / non-alcoholic fatty liver disease); NOT when it has other common meanings (ECE, PD / Pd, DM, PET, ATM) or is a chemical symbol or formula that also reads as something else,
- a renamed concept and its old name (mpox / monkeypox, MASLD / NAFLD),
- spelling, transliteration, hyphenation, word-order or grammatical variants (tumor / tumour, nudging / nudges, CAR-T cells / CAR T cells),
- a Latin taxon and its common name for the same group (Tardigrada / tardigrades).
Do NOT merge (false): different things (World War I / World War II, food safety / food security), broader vs narrower (animals / dogs, deep learning / deep neural networks, monetary policy / monetary policy transmission), a thing vs a therapy or process using it (CAR-T cells / CAR-T cell therapy), a quantity vs its measurement (electron density / electron density measurement), a disease vs the pandemic as a societal event (COVID-19 / COVID-19 pandemic), related but distinct concepts (gender differences / sex differences), proper names that must keep their own form (UK Labour Party, a Latin genus name).
For merges, `keep` is the better heading: the standard, most widely used form in current literature (1 = first, 2 = second).
Return JSON only: for each item n and merge (true/false).
```

## User message

```text
(as the conservative context check)
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
     "merge": {
      "type": "boolean"
     }
    },
    "required": [
     "n",
     "merge"
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
