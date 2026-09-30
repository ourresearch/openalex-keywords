"""Judge each exclusion candidate against the two schemas (Jason 2026-09-25): drop a keyword only if it exactly or approximately replicates an OpenAlex
work `type` or a #1312 `study_designs` value; keep narrower designs/methods and subject uses. Opus 5.5, 150 per call, fixed schema, resumable by rubric version.
Evidence per keyword: works count and the top 3 work types it sits on. Writes data/exclusion_verdicts.jsonl"""
import json, os, argparse, concurrent.futures as cf, time
import anthropic
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = f'{HERE}/data/exclusion_verdicts.jsonl'; RUBRIC = 'excl_v1'
TYPES = ['article', 'dataset', 'book-chapter', 'conference-paper', 'dissertation', 'preprint', 'book', 'paratext', 'conference-abstract', 'reference-entry', 'report',
         'book-review', 'libguides', 'peer-review', 'editorial', 'review', 'erratum', 'letter', 'standard', 'supplementary-materials', 'software', 'retraction', 'data-paper', 'software-paper']
DESIGNS = ['randomized-controlled-trial', 'clinical-trial', 'observational-study', 'case-report', 'systematic-review', 'meta-analysis', 'study-protocol', 'other-primary-research']
VERDICTS = ['keep'] + [f'type:{t}' for t in TYPES] + [f'design:{d}' for d in DESIGNS] + ['genre-not-in-schema']
SYSTEM = f"""OpenAlex is removing keywords that duplicate two structured fields it already has, so a keyword never repeats what a filter already says.

Field 1, work type (what kind of document the work is): {', '.join(TYPES)}. ("review" means a review article; "libguides" means library research guides; "paratext" means front matter, covers, tables of contents, issue information.)
Field 2, study design (PubMed's study-characteristics schema): randomized controlled trial; clinical trial (incl. phase I-IV, controlled clinical trial, clinical study); observational study; case report; systematic review; meta-analysis (incl. network meta-analysis); study protocol (clinical trial protocol); other primary research.

For each keyword decide:
- type:<value> if the keyword, as used, labels the kind of document the work itself is, and that kind is the same as, a synonym of, or a close variant of a work type. Examples: "book review" -> type:book-review; "library research guide" -> type:libguides; "erratum", "corrigendum" -> type:erratum; "doctoral thesis", "master thesis" -> type:dissertation; "literature review", "narrative review", "review article" -> type:review; "editorial" -> type:editorial; "letter to the editor" -> type:letter.
- design:<value> if it is the same as, a synonym of, or a close variant of a study-design value. Examples: "RCT", "randomised controlled trial", "randomized clinical trial", "pragmatic randomized trial" -> design:randomized-controlled-trial; "phase II trial" -> design:clinical-trial; "systematic literature review" -> design:systematic-review; "network meta-analysis" -> design:meta-analysis; "trial protocol" -> design:study-protocol.
- genre-not-in-schema if it labels the kind of document but no work type covers it. Examples: "biography", "obituary", "interview", "conference report", "news".
- keep for everything else: subjects (a work ABOUT peer review, open access, clinical trial regulation, the history of dissertations), narrower or more detailed designs and methods that the schema does not name ("cohort study", "case-control study", "cross-sectional study", "scoping review", "umbrella review", "crossover trial" only if it is not simply a kind of RCT, "qualitative research", "Delphi study"), and anything unrelated that the candidate filter caught by accident ("standard deviation", "software engineering", "reporting guidelines").

Use the evidence: each line gives the keyword, how many works carry it, and the work types it most often sits on. If a document-type-like keyword sits mostly on the matching work type (e.g. "peer review" mostly on peer-review records), it labels the document; if it sits mostly on articles, it is probably a subject. Return one verdict per index."""
SCHEMA = {"type": "object", "properties": {"v": {"type": "array", "items": {"type": "object", "properties": {"i": {"type": "integer"}, "c": {"type": "string", "enum": VERDICTS}}, "required": ["i", "c"], "additionalProperties": False}}},
          "required": ["v"], "additionalProperties": False}
ap = argparse.ArgumentParser(); ap.add_argument('--workers', type=int, default=16); ap.add_argument('--batch', type=int, default=150); ap.add_argument('--limit', type=int, default=None); a = ap.parse_args()
cand = json.load(open(f'{HERE}/data/exclusion_candidates.json'))[:a.limit]
def types_of(r):
    t = r.get('types');
    t = json.loads(t) if isinstance(t, str) else (t or [])
    return ', '.join(t)
done = set()
if os.path.exists(OUT):
    for l in open(OUT):
        r = json.loads(l)
        if r['rubric'] == RUBRIC: done.add(r['kc'])
todo = [r for r in cand if r['kc'] not in done]; print(f'{len(cand)} candidates, {len(done)} judged, {len(todo)} to do', flush=True)
client = anthropic.Anthropic(api_key=os.environ['CLAUDE_API_KEY'])
def call(batch):
    user = "\n".join(f"{i}\t{r['kc']}\t{int(r['n']):,} works\t{types_of(r)}" for i, r in enumerate(batch))
    for attempt in range(5):
        try:
            resp = client.messages.create(model='claude-opus-5-5', max_tokens=16000, system=SYSTEM,
                                          messages=[{'role': 'user', 'content': f"index<TAB>keyword<TAB>works<TAB>top work types\n{user}"}],
                                          extra_body={'output_config': {'effort': 'medium', 'format': {'type': 'json_schema', 'schema': SCHEMA}}})
            text = "".join(b.text for b in resp.content if b.type == 'text')
            if not text: raise ValueError(f'empty text, stop_reason {resp.stop_reason}')
            v = {x['i']: x['c'] for x in json.loads(text)['v'] if 0 <= x['i'] < len(batch)}
            return [(batch[i], v[i]) for i in v], resp.usage
        except Exception as e:
            print('retry', attempt, str(e)[:200], flush=True); time.sleep(10 * (attempt + 1))
    return [], None
batches = [todo[i:i + a.batch] for i in range(0, len(todo), a.batch)]; cost = 0.0; n = 0
with open(OUT, 'a') as f, cf.ThreadPoolExecutor(a.workers) as ex:
    for res, u in ex.map(call, batches):
        for r, c in res: f.write(json.dumps({'kc': r['kc'], 'n': int(r['n']), 'types': types_of(r), 'c': c, 'rubric': RUBRIC}, ensure_ascii=False) + '\n')
        f.flush(); n += len(res)
        if u: cost += u.input_tokens * 4e-6 + u.output_tokens * 20e-6
        print(f'{n}/{len(todo)} judged, ${cost:.2f}', flush=True)
