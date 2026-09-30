"""Draft the dual-use exception list (Jason 2026-09-25): which purge keywords (kind <> 'keep' in keyword_doctype_verdicts_v1) are ALSO commonly a genuine
research subject, so they should be dropped only on works whose own type / design already says it. Opus 5.5, 150 per call. Writes data/dual_use_verdicts.jsonl"""
import json, os, sys, concurrent.futures as cf, time
import anthropic
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'common')); from dbx import sql
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = f'{HERE}/data/dual_use_verdicts.jsonl'; RUBRIC = 'dual_v1'
rows = sql("SELECT keyword, works, kind, maps_to, top_work_types FROM openalex.transient.keyword_doctype_verdicts_v1 WHERE kind <> 'keep' ORDER BY works DESC", warehouse=os.environ.get('DATABRICKS_WAREHOUSE_ID'), verbose=False)
SYSTEM = """OpenAlex is removing keywords that label what KIND of document or study a work is (book review, erratum, randomized controlled trial, obituary), because those belong in structured filters. Some of these strings are also genuine research SUBJECTS: a paper can be ABOUT peer review, about research software, about clinical trial methodology, about obituaries as a genre. For those, OpenAlex will keep the keyword on works where it is the subject and drop it only where it merely labels the work.

For each keyword, answer dual_use = true if a meaningful number of scholarly works are plausibly ABOUT this thing as their subject (research on it, its history, its practice, its quality), not just instances of it. Answer false if the string is essentially only ever a label for the document itself (e.g. "erratum", "book review", "corrigendum", "retraction notice", "letter to the editor", "doctoral thesis", "case report", "phase ii clinical trial").
Evidence per line: the keyword, how many works carry it, what it maps to, and the work types it most often sits on. Be conservative: dual_use only when subject use is common, not merely conceivable."""
SCHEMA = {"type": "object", "properties": {"v": {"type": "array", "items": {"type": "object", "properties": {"i": {"type": "integer"}, "d": {"type": "boolean"}}, "required": ["i", "d"], "additionalProperties": False}}}, "required": ["v"], "additionalProperties": False}
client = anthropic.Anthropic(api_key=os.environ['CLAUDE_API_KEY'])
def call(batch):
    user = "\n".join(f"{i}\t{r['keyword']}\t{int(r['works']):,} works\t{r['kind']}:{r['maps_to']}\t{r['top_work_types']}" for i, r in enumerate(batch))
    for attempt in range(5):
        try:
            resp = client.messages.create(model='claude-opus-5-5', max_tokens=16000, system=SYSTEM, messages=[{'role': 'user', 'content': f"index<TAB>keyword<TAB>works<TAB>maps to<TAB>top work types\n{user}"}],
                                          extra_body={'output_config': {'effort': 'medium', 'format': {'type': 'json_schema', 'schema': SCHEMA}}})
            text = "".join(b.text for b in resp.content if b.type == 'text')
            if not text: raise ValueError(f'empty text, stop_reason {resp.stop_reason}')
            v = {x['i']: x['d'] for x in json.loads(text)['v'] if 0 <= x['i'] < len(batch)}
            return [(batch[i], v[i]) for i in v], resp.usage
        except Exception as e:
            print('retry', attempt, str(e)[:200], flush=True); time.sleep(10 * (attempt + 1))
    return [], None
batches = [rows[i:i + 150] for i in range(0, len(rows), 150)]; cost = 0.0; n = 0
with open(OUT, 'w') as f, cf.ThreadPoolExecutor(16) as ex:
    for res, u in ex.map(call, batches):
        for r, d in res: f.write(json.dumps({**r, 'works': int(r['works']), 'dual_use': d, 'rubric': RUBRIC}, ensure_ascii=False) + '\n')
        f.flush(); n += len(res)
        if u: cost += u.input_tokens * 4e-6 + u.output_tokens * 20e-6
print(f'done {n}/{len(rows)} ${cost:.2f}', flush=True)
