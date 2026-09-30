"""Hunt for non-aboutness keyword types (Jason, 2026-09-25): Opus 5.5 tags each keyword the recommended prompt (arm D) gave a work with a type from a seeded,
open taxonomy (an 'other' slot with a description surfaces new types) and whether it is an aboutness keyword. Sample: 600 non-xpac + 300 xpac works from data/big_D.jsonl.
-> data/types.jsonl. uv run --with anthropic python classify_types.py"""
import json, os, random, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
import anthropic
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
from prompt import user_text
TYPES = {
 'subject': 'a specific topic, phenomenon, entity, organism, material, population or problem the work is about',
 'method_is_subject': 'a method or technique that the work develops, evaluates or is centrally about',
 'routine_method': 'a standard method, instrument or technique merely used in the work, not what it is about (e.g. PCR, ELISA, questionnaire, SEM in a paper about something else)',
 'discipline': 'the name of a field or discipline (botany, epidemiology, plasma physics) rather than a topic within it',
 'genre': 'the literary or document genre or form of the work (biography, poetry, book review, case report, editorial, thesis, portrait)',
 'record_kind': 'what kind of record or data object this is (specimen record, occurrence record, genebank accession, geological sample, dataset, image, sediment core, crystal structure entry)',
 'holder_or_program': 'the institution, collection, database, repository, archive, project or program that holds or produced the record, not its subject',
 'generic_outcome': 'a generic analysis or outcome word that fits many works of its kind (efficacy, risk factors, prognosis, mortality, prevalence, treatment, quality of life, clinical outcomes)',
 'place_not_subject': 'a country, region or place that is only the setting or origin, not what the work is about',
 'time_period': 'a century, era or date range',
 'person_name': 'a person the keyword names',
 'unsupported': 'a guess the record does not support',
 'other': 'none of these; describe the kind in other_desc',
}
SYSTEM = ("You are auditing keywords assigned to scholarly records. Keywords should describe what a work is about. For each numbered keyword give its type from this list and whether it is "
          "an aboutness keyword a careful indexer would keep for this work:\n" + "\n".join(f"- {k}: {v}" for k, v in TYPES.items()) +
          "\nUse 'other' with a 2-6 word other_desc when no type fits. Return JSON only.")
SCHEMA = {"type": "object", "properties": {"items": {"type": "array", "items": {"type": "object", "properties": {
    "n": {"type": "integer"}, "type": {"type": "string", "enum": list(TYPES)}, "other_desc": {"type": "string"}, "aboutness": {"type": "boolean"}},
    "required": ["n", "type", "other_desc", "aboutness"], "additionalProperties": False}}}, "required": ["items"], "additionalProperties": False}
import argparse; ap = argparse.ArgumentParser(); ap.add_argument('--works', default='data/works5k.jsonl'); ap.add_argument('--rows', default='data/big_D.jsonl'); ap.add_argument('--out', default='data/types.jsonl'); ap.add_argument('--all', action='store_true'); a = ap.parse_args()
works = {json.loads(l)['id']: json.loads(l) for l in open(f'{HERE}/{a.works}')}
rows = [r for r in map(json.loads, open(f'{HERE}/{a.rows}')) if r.get('keywords')]
rng = random.Random(1322); non = [r for r in rows if works[r['id']]['group'] != 'xpac']; xp = [r for r in rows if works[r['id']]['group'] == 'xpac']
rng.shuffle(non); rng.shuffle(xp); sample = rows if a.all else non[:600] + xp[:300]
out_path = f'{HERE}/{a.out}'
done = {json.loads(l)['id'] for l in open(out_path) if 'items' in json.loads(l)} if os.path.exists(out_path) else set()
client = anthropic.Anthropic(api_key=os.environ.get('CLAUDE_API_KEY'), max_retries=4, timeout=600)
def one(r):
    w = works[r['id']]; txt = user_text(w) + "\n\nKEYWORDS:\n" + "\n".join(f"{n + 1}. {k}" for n, k in enumerate(r['keywords']))
    for attempt in range(5):
        try:
            m = client.messages.create(model='claude-opus-5-5', max_tokens=8000, system=SYSTEM, messages=[{'role': 'user', 'content': txt}],
                                       extra_body={'output_config': {'effort': 'medium', 'format': {'type': 'json_schema', 'schema': SCHEMA}}})
            text = "".join(b.text for b in m.content if b.type == 'text')
            if not text: raise ValueError(f'empty, {m.stop_reason}')
            items = json.loads(text)['items']
            for it in items: it['keyword'] = r['keywords'][it['n'] - 1] if 0 < it['n'] <= len(r['keywords']) else None
            return {'id': r['id'], 'group': w['group'], 'items': items, 'usage': [m.usage.input_tokens, m.usage.output_tokens]}
        except Exception as e:
            err = repr(e)[:200]; time.sleep(5 * (attempt + 1))
    return {'id': r['id'], 'error': err}
cost = 0.0; n_err = 0
with open(out_path, 'a') as f, ThreadPoolExecutor(16) as ex:
    for fut in as_completed([ex.submit(one, r) for r in sample if r['id'] not in done]):
        x = fut.result(); f.write(json.dumps(x, ensure_ascii=False) + '\n'); f.flush(); n_err += 'error' in x
        if 'usage' in x: cost += x['usage'][0] * 4e-6 + x['usage'][1] * 20e-6
print(f'classified {len(sample)} works, errors {n_err}, ${cost:.2f}')
