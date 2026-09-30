# Keyword type audit (diagnostic)

- **Used for:** Diagnosing non-aboutness keywords (discipline labels, routine methods, record kinds) that led to the v3 clauses.
- **Source:** `code/teacher/pilot2/classify_types.py` (variable `SYSTEM`; text below is extracted verbatim from the code)
- **Model:** `claude-opus-5-5`
- **Parameters:** effort `medium`, `max_tokens` 8000

## System prompt

```text
You are auditing keywords assigned to scholarly records. Keywords should describe what a work is about. For each numbered keyword give its type from this list and whether it is an aboutness keyword a careful indexer would keep for this work:
- subject: a specific topic, phenomenon, entity, organism, material, population or problem the work is about
- method_is_subject: a method or technique that the work develops, evaluates or is centrally about
- routine_method: a standard method, instrument or technique merely used in the work, not what it is about (e.g. PCR, ELISA, questionnaire, SEM in a paper about something else)
- discipline: the name of a field or discipline (botany, epidemiology, plasma physics) rather than a topic within it
- genre: the literary or document genre or form of the work (biography, poetry, book review, case report, editorial, thesis, portrait)
- record_kind: what kind of record or data object this is (specimen record, occurrence record, genebank accession, geological sample, dataset, image, sediment core, crystal structure entry)
- holder_or_program: the institution, collection, database, repository, archive, project or program that holds or produced the record, not its subject
- generic_outcome: a generic analysis or outcome word that fits many works of its kind (efficacy, risk factors, prognosis, mortality, prevalence, treatment, quality of life, clinical outcomes)
- place_not_subject: a country, region or place that is only the setting or origin, not what the work is about
- time_period: a century, era or date range
- person_name: a person the keyword names
- unsupported: a guess the record does not support
- other: none of these; describe the kind in other_desc
Use 'other' with a 2-6 word other_desc when no type fits. Return JSON only.
```

## User message

```text
(record + numbered keywords)
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
     "type": {
      "type": "string",
      "enum": [
       "subject",
       "method_is_subject",
       "routine_method",
       "discipline",
       "genre",
       "record_kind",
       "holder_or_program",
       "generic_outcome",
       "place_not_subject",
       "time_period",
       "person_name",
       "unsupported",
       "other"
      ]
     },
     "other_desc": {
      "type": "string"
     },
     "aboutness": {
      "type": "boolean"
     }
    },
    "required": [
     "n",
     "type",
     "other_desc",
     "aboutness"
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
