# Document-type / study-design keyword exclusions

- **Used for:** Building `keyword_doctype_verdicts_v1` (65,973 candidate keywords; the purge list). From OpenAlex job #1359, reused unchanged here.
- **Source:** `code/vocabulary/doctype_rules/judge_exclusions.py` (variable `SYSTEM`; text below is extracted verbatim from the code)
- **Model:** `claude-opus-5-5`
- **Parameters:** effort `medium`, `max_tokens` 16000, 150 keywords per call

## System prompt

```text
OpenAlex is removing keywords that duplicate two structured fields it already has, so a keyword never repeats what a filter already says.

Field 1, work type (what kind of document the work is): article, dataset, book-chapter, conference-paper, dissertation, preprint, book, paratext, conference-abstract, reference-entry, report, book-review, libguides, peer-review, editorial, review, erratum, letter, standard, supplementary-materials, software, retraction, data-paper, software-paper. ("review" means a review article; "libguides" means library research guides; "paratext" means front matter, covers, tables of contents, issue information.)
Field 2, study design (PubMed's study-characteristics schema): randomized controlled trial; clinical trial (incl. phase I-IV, controlled clinical trial, clinical study); observational study; case report; systematic review; meta-analysis (incl. network meta-analysis); study protocol (clinical trial protocol); other primary research.

For each keyword decide:
- type:<value> if the keyword, as used, labels the kind of document the work itself is, and that kind is the same as, a synonym of, or a close variant of a work type. Examples: "book review" -> type:book-review; "library research guide" -> type:libguides; "erratum", "corrigendum" -> type:erratum; "doctoral thesis", "master thesis" -> type:dissertation; "literature review", "narrative review", "review article" -> type:review; "editorial" -> type:editorial; "letter to the editor" -> type:letter.
- design:<value> if it is the same as, a synonym of, or a close variant of a study-design value. Examples: "RCT", "randomised controlled trial", "randomized clinical trial", "pragmatic randomized trial" -> design:randomized-controlled-trial; "phase II trial" -> design:clinical-trial; "systematic literature review" -> design:systematic-review; "network meta-analysis" -> design:meta-analysis; "trial protocol" -> design:study-protocol.
- genre-not-in-schema if it labels the kind of document but no work type covers it. Examples: "biography", "obituary", "interview", "conference report", "news".
- keep for everything else: subjects (a work ABOUT peer review, open access, clinical trial regulation, the history of dissertations), narrower or more detailed designs and methods that the schema does not name ("cohort study", "case-control study", "cross-sectional study", "scoping review", "umbrella review", "crossover trial" only if it is not simply a kind of RCT, "qualitative research", "Delphi study"), and anything unrelated that the candidate filter caught by accident ("standard deviation", "software engineering", "reporting guidelines").

Use the evidence: each line gives the keyword, how many works carry it, and the work types it most often sits on. If a document-type-like keyword sits mostly on the matching work type (e.g. "peer review" mostly on peer-review records), it labels the document; if it sits mostly on articles, it is probably a subject. Return one verdict per index.
```

## User message

```text
{index}\t{keyword}\t{works} works\t{top 3 work types}
```

## Output schema (structured outputs, `output_config.format = json_schema`)

```json
{
 "type": "object",
 "properties": {
  "v": {
   "type": "array",
   "items": {
    "type": "object",
    "properties": {
     "i": {
      "type": "integer"
     },
     "c": {
      "type": "string",
      "enum": [
       "keep",
       "type:article",
       "type:dataset",
       "type:book-chapter",
       "type:conference-paper",
       "type:dissertation",
       "type:preprint",
       "type:book",
       "type:paratext",
       "type:conference-abstract",
       "type:reference-entry",
       "type:report",
       "type:book-review",
       "type:libguides",
       "type:peer-review",
       "type:editorial",
       "type:review",
       "type:erratum",
       "type:letter",
       "type:standard",
       "type:supplementary-materials",
       "type:software",
       "type:retraction",
       "type:data-paper",
       "type:software-paper",
       "design:randomized-controlled-trial",
       "design:clinical-trial",
       "design:observational-study",
       "design:case-report",
       "design:systematic-review",
       "design:meta-analysis",
       "design:study-protocol",
       "design:other-primary-research",
       "genre-not-in-schema"
      ]
     }
    },
    "required": [
     "i",
     "c"
    ],
    "additionalProperties": false
   }
  }
 },
 "required": [
  "v"
 ],
 "additionalProperties": false
}
```
