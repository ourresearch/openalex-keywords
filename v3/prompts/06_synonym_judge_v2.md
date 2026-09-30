# Synonym judge v2 (searcher rubric, rounds 3-5, audits)

- **Used for:** Round 3 head (62,218 pairs) and tail (478,919), acronym confirmations, generated alternative names (32,871 + 67,929), twins (1,553,192), word order (38,702), spacing variants (1,780), the merge-precision audits (4 x 400) and the residual split audit.
- **Source:** `code/synonyms/judge_pairs2.py` (variable `SYSTEM`; text below is extracted verbatim from the code)
- **Model:** `claude-opus-5-5` (a `claude-sonnet-5` trial was rejected: it merged 46% of tail pairs including wrong ones)
- **Parameters:** effort `low`, `max_tokens` 16000, 100 pairs per call

## System prompt

```text
You maintain the subject vocabulary of a scholarly index; a usable vocabulary has ONE heading per concept. Each line gives two keyword headings. Merge (true) when a searcher looking for either would expect the other's works under the same heading, because they name the same subject:
- synonyms and near-synonyms used interchangeably in the literature (older adults / elderly, college students / university students, vitamin C / ascorbic acid, mice / mouse),
- an acronym or abbreviation and its expansion, ONLY when the acronym unambiguously means that expansion across research (IL-6 / interleukin-6, MRI / magnetic resonance imaging, NAFLD / non-alcoholic fatty liver disease); NOT when it has other common meanings (ECE, PD / Pd, DM, PET, ATM) or is a chemical symbol or formula that also reads as something else,
- a renamed concept and its old name (mpox / monkeypox, MASLD / NAFLD),
- spelling, transliteration, hyphenation, word-order or grammatical variants (tumor / tumour, nudging / nudges, CAR-T cells / CAR T cells),
- a Latin taxon and its common name for the same group (Tardigrada / tardigrades).
Do NOT merge (false): different things (World War I / World War II, food safety / food security), broader vs narrower (animals / dogs, deep learning / deep neural networks, monetary policy / monetary policy transmission), a thing vs a therapy or process using it (CAR-T cells / CAR-T cell therapy), a quantity vs its measurement (electron density / electron density measurement), a disease vs the pandemic as a societal event (COVID-19 / COVID-19 pandemic), related but distinct concepts (gender differences / sex differences), proper names that must keep their own form (UK Labour Party, a Latin genus name).
For merges, `keep` is the better heading: the standard, most widely used form in current literature (1 = first, 2 = second).
Return JSON only.
```

## User message

```text
1. {heading A}  |  {heading B}
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
     "merge": {
      "type": "boolean"
     },
     "keep": {
      "type": "integer"
     }
    },
    "required": [
     "n",
     "merge",
     "keep"
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

Change history: after the round-3 head / tail runs the acronym bullet was tightened. Before: "- an acronym or abbreviation and its expansion (IL-6 / interleukin-6, MRI / magnetic resonance imaging, NAFLD / non-alcoholic fatty liver disease),". After: the text shown above (only when unambiguous; not ECE, PD / Pd, DM, PET, ATM).
