"""Judge the pilot arms: per work, the union of every arm's keywords (deduped by kid0, shuffled, arm-blind), one Opus 5.5 call with the xpac judge's strict rubric
(accept = specific to what the work is substantially about AND supported by the record, not a guess). The judge always sees the full record (title, abstract, venue,
year, type, language) whatever the arm saw. One verdict per keyword serves every arm, so arm differences are not judge noise. -> data/judged.jsonl (resumable).
uv run --with anthropic python judge_arms.py [--arms v2,A,B,C,D]"""
import argparse, json, os, random, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
import anthropic
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'common'))
from prompt import user_text
from kid import kid0
SYSTEM = """You are a subject indexer for OpenAlex reviewing keywords proposed for a work. You are given the record's title, abstract, venue, year, type and language, and a numbered list of proposed keywords (possibly empty), pooled from several taggers in random order.
1. sufficient: does the record itself say enough about what the work is about for a careful indexer to assign at least one specific subject keyword? (false for bare file names, identifiers, "Dataset 3", generic titles like "Supplementary data" or "Figure 2", untitled or boilerplate records.)
2. For each proposed keyword decide whether a careful indexer would accept it: specific to what the work is substantially about (subject, method, material, organism, population, place, phenomenon), supported by the record, not the general field, not a generic word, not the kind of document the work is, not a guess the record does not support. Judge each keyword on its own; near-duplicates of each other may all be accepted.
Return JSON only: the numbers of accepted keywords and of wrong keywords; every number appears in exactly one list."""
SCHEMA = {"type": "object", "properties": {"sufficient": {"type": "boolean"}, "accepted": {"type": "array", "items": {"type": "integer"}}, "wrong": {"type": "array", "items": {"type": "integer"}}},
          "required": ["sufficient", "accepted", "wrong"], "additionalProperties": False}
ap = argparse.ArgumentParser(); ap.add_argument('--arms', default='v2,A,B,C,D'); ap.add_argument('--workers', type=int, default=16); a = ap.parse_args()
works = {json.loads(l)['id']: json.loads(l) for l in open(f'{HERE}/data/works.jsonl')}
kws = {i: {} for i in works}          # id -> kid0 -> surface form (first seen)
for arm in a.arms.split(','):
    for l in open(f'{HERE}/data/arm_{arm}.jsonl'):
        r = json.loads(l)
        for k in r.get('keywords', []):
            k0 = kid0(k)
            if k0: kws[r['id']].setdefault(k0, k.strip())
out_path = f'{HERE}/data/judged.jsonl'
done = {}
if os.path.exists(out_path):
    for l in open(out_path):
        r = json.loads(l)
        if 'verdict' in r: done[r['id']] = r
todo = [i for i in works if i not in done or set(kws[i]) - set(done[i]['verdict'])]   # re-judge if a rerun added keywords
client = anthropic.Anthropic(api_key=os.environ.get('CLAUDE_API_KEY'), max_retries=4, timeout=600)
def one(i):
    items = sorted(kws[i].items()); random.Random(i).shuffle(items)
    txt = user_text(works[i]) + "\n\nPROPOSED KEYWORDS:\n" + ("\n".join(f"{n + 1}. {s}" for n, (_, s) in enumerate(items)) or "(none)")
    for attempt in range(6):
        try:
            r = client.messages.create(model='claude-opus-5-5', max_tokens=6000, system=SYSTEM, messages=[{'role': 'user', 'content': txt}],
                                       extra_body={'output_config': {'effort': 'medium', 'format': {'type': 'json_schema', 'schema': SCHEMA}}})
            text = "".join(b.text for b in r.content if b.type == 'text')
            if not text: raise ValueError(f'empty, {r.stop_reason}')
            j = json.loads(text); acc = set(j['accepted'])
            return {'id': i, 'sufficient': j['sufficient'], 'verdict': {k: (n + 1) in acc for n, (k, _) in enumerate(items)},
                    'unassigned': [n + 1 for n in range(len(items)) if (n + 1) not in acc and (n + 1) not in set(j['wrong'])], 'usage': [r.usage.input_tokens, r.usage.output_tokens]}
        except Exception as e:
            err = repr(e)[:200]; time.sleep(120 if 'credit balance' in err else 5 * (attempt + 1))
    return {'id': i, 'error': err}
cost = 0.0; n_err = 0
with ThreadPoolExecutor(a.workers) as ex:
    for fut in as_completed([ex.submit(one, i) for i in todo]):
        r = fut.result(); done[r['id']] = r if 'verdict' in r else done.get(r['id'], r); n_err += 'error' in r
        if 'usage' in r: cost += r['usage'][0] * 4e-6 + r['usage'][1] * 20e-6
with open(out_path, 'w') as f:
    for r in done.values(): f.write(json.dumps(r, ensure_ascii=False) + '\n')
print(f'judged {len(todo)} works ({sum(len(v) for v in kws.values())} keywords), errors {n_err}, ${cost:.2f}')
