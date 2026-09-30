"""Score the v3 student (#1322, 2026-09-26): Opus 5.5 judges, per work, the union of several keyword lists (deduped by kid0, shuffled, list-blind), with the pilot2 strict
rubric, on 700 old-eval non-xpac works (new student, old student qwen4ball8, Opus v3 labels) + 300 xpac eval works (new student, Opus v3 labels). Then prints, per list and
input group: kw/work, empty share, judged precision, accepted kw/work, F1 vs Opus v3 (kid0 sets). -> data/judged_v3.jsonl (resumable), data/score_v3.json.
uv run --with anthropic python judge_union.py"""
import json, os, random, sys, time, collections
from concurrent.futures import ThreadPoolExecutor, as_completed
import anthropic
HERE = os.path.dirname(os.path.abspath(__file__)); G = HERE; sys.path.insert(0, os.path.join(HERE, '..', 'teacher'))  # released layout: data/, results/ under code/student/
sys.path.insert(0, G); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'common'))
from kid import kid0
from prompt import user_text
SYSTEM = """You are a subject indexer for OpenAlex reviewing keywords proposed for a work. You are given the record's title, abstract, venue, year, type and language, and a numbered list of proposed keywords (possibly empty), pooled from several taggers in random order.
1. sufficient: does the record itself say enough about what the work is about for a careful indexer to assign at least one specific subject keyword? (false for bare file names, identifiers, "Dataset 3", generic titles like "Supplementary data" or "Figure 2", untitled or boilerplate records.)
2. For each proposed keyword decide whether a careful indexer would accept it: specific to what the work is substantially about (subject, method, material, organism, population, place, phenomenon), supported by the record, not the general field, not a generic word, not the kind of document the work is, not a guess the record does not support. Judge each keyword on its own; near-duplicates of each other may all be accepted.
Return JSON only: the numbers of accepted keywords and of wrong keywords; every number appears in exactly one list."""
SCHEMA = {"type": "object", "properties": {"sufficient": {"type": "boolean"}, "accepted": {"type": "array", "items": {"type": "integer"}}, "wrong": {"type": "array", "items": {"type": "integer"}}},
          "required": ["sufficient", "accepted", "wrong"], "additionalProperties": False}
def rd(p): return [json.loads(l) for l in open(p)]
W = {w['id']: w for w in rd(f'{HERE}/data/sample.jsonl')}
old_eval = set(open(f'{G}/data/god_eval_ids.txt').read().split())
new = {r['id']: r['keywords'] for r in rd(f'{G}/results/arm_qwen4b_v3.jsonl')}
old = {r['id']: r['keywords'] for r in rd(f'{G}/results/arm_qwen4ball8.jsonl') if 'keywords' in r}
opus = {r['id']: r['answer']['keywords'] for r in rd(f'{G}/results/god_v3.jsonl') if 'answer' in r}
rng = random.Random(1322)
non = sorted(i for i in new if i in old_eval and i in old and i in opus and i in W); rng.shuffle(non)
xp = sorted(i for i in new if W.get(i, {}).get('xpac') and i in opus); rng.shuffle(xp)
sample = non[:700] + xp[:300]
LISTS = {'new': new, 'old': old, 'opus': opus}
out_path = f'{HERE}/data/judged_v3.jsonl'
done = {r['id']: r for r in map(json.loads, open(out_path)) if 'verdict' in r} if os.path.exists(out_path) else {}
client = anthropic.Anthropic(api_key=os.environ.get('CLAUDE_API_KEY') or os.environ.get('ANTHROPIC_API_KEY'), max_retries=4, timeout=600)
def one(i):
    kws = {}
    for L in LISTS.values():
        for k in L.get(i, []):
            k0 = kid0(k)
            if k0: kws.setdefault(k0, k.strip())
    items = sorted(kws.items()); random.Random(i).shuffle(items)
    w = W[i]; txt = user_text({**w, 'type': w.get('type'), 'language': w.get('language')}) + "\n\nPROPOSED KEYWORDS:\n" + ("\n".join(f"{n + 1}. {s}" for n, (_, s) in enumerate(items)) or "(none)")
    for attempt in range(6):
        try:
            r = client.messages.create(model='claude-opus-5-5', max_tokens=6000, system=SYSTEM, messages=[{'role': 'user', 'content': txt}],
                                       extra_body={'output_config': {'effort': 'medium', 'format': {'type': 'json_schema', 'schema': SCHEMA}}})
            text = "".join(b.text for b in r.content if b.type == 'text')
            if not text: raise ValueError(f'empty, {r.stop_reason}')
            j = json.loads(text); acc = set(j['accepted'])
            return {'id': i, 'sufficient': j['sufficient'], 'verdict': {k: (n + 1) in acc for n, (k, _) in enumerate(items)}, 'usage': [r.usage.input_tokens, r.usage.output_tokens]}
        except Exception as e:
            err = repr(e)[:200]; time.sleep(120 if 'credit balance' in err else 5 * (attempt + 1))
    return {'id': i, 'error': err}
todo = [i for i in sample if i not in done]; cost = 0.0
with ThreadPoolExecutor(16) as ex:
    for f in as_completed([ex.submit(one, i) for i in todo]):
        r = f.result()
        if 'verdict' in r: done[r['id']] = r; cost += r['usage'][0] * 4e-6 + r['usage'][1] * 20e-6
with open(out_path, 'w') as f:
    for r in done.values(): f.write(json.dumps(r, ensure_ascii=False) + '\n')
print(f'judged {len(todo)} works, ${cost:.2f}; have {len(done)}')
def grp(i):
    w = W[i]
    if w.get('xpac'): return 'xpac'
    a = len((w.get('abstract') or '').strip())
    return 'abstract' if a >= 300 else ('short' if a >= 50 else 'title_only')
S = {}
for name, L in LISTS.items():
    for g in ['abstract', 'short', 'title_only', 'xpac', 'all_nonxpac']:
        ids = [i for i in sample if i in done and i in L and (grp(i) == g or (g == 'all_nonxpac' and grp(i) != 'xpac'))]
        if not ids: continue
        ks = [(i, kid0(k)) for i in ids for k in L[i] if kid0(k)]
        jd = [done[i]['verdict'][k] for i, k in ks if k in done[i]['verdict']]
        f1 = []
        for i in ids:
            if name == 'opus': continue
            g_ = {kid0(k) for k in opus.get(i, []) if kid0(k)}; p = {kid0(k) for k in L[i] if kid0(k)}
            f1.append(1.0 if not g_ and not p else 2 * len(g_ & p) / (len(g_) + len(p)))
        S[f'{name}|{g}'] = {'works': len(ids), 'kw': round(len(ks) / len(ids), 2), 'empty': round(sum(not L[i] for i in ids) / len(ids), 3),
                            'precision': round(sum(jd) / len(jd), 3) if jd else None, 'good_per_work': round(sum(jd) / len(ids), 2), 'f1_vs_opus': round(sum(f1) / len(f1), 3) if f1 else None}
json.dump(S, open(f'{HERE}/data/score_v3.json', 'w'), indent=1)
print('| list | group | works | kw/work | empty | judged P | good/work | F1 vs Opus v3 |\n|---|---|---|---|---|---|---|---|')
for k, s in S.items():
    n, g = k.split('|'); print(f"| {n} | {g} | {s['works']} | {s['kw']} | {s['empty']} | {s['precision']} | {s['good_per_work']} | {s['f1_vs_opus']} |")
