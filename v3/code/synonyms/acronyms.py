"""Acronym <-> expansion merges (#1322, 2026-09-28 evening): Opus 5.5 expands each acronym-like keyword (>= 200 works) as used in research (null when ambiguous
across fields, e.g. PET, or when it is itself the standard name, e.g. DNA); the expansion is looked up in the vocabulary (kid0 + plural fold); merge pairs go
into the heading with more works. -> data/acronyms_judged.jsonl, data/synmap_acronyms.jsonl. uv run --with anthropic python acronyms.py"""
import json, os, sys, time
from concurrent.futures import ThreadPoolExecutor
import anthropic
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'common')); from kid import kid0
V = {json.loads(l)['kid']: json.loads(l) for l in open(f'{HERE}/data/vocab_g.jsonl')}
syn = {json.loads(l)['kid']: json.loads(l)['canon'] for l in open(f'{HERE}/data/synmap2.jsonl')}
acr = json.load(open(f'{HERE}/data/' + (sys.argv[1] if len(sys.argv) > 1 else 'acronyms.json')))
SYSTEM = """For each numbered keyword from a scholarly subject vocabulary that looks like an acronym or abbreviation, give its expansion as used in research literature, in standard English form (e.g. IL-6 -> interleukin-6, MRI -> magnetic resonance imaging, NDVI -> normalized difference vegetation index). Return null when: it has several common meanings across fields (PET, ATM, BIM, CAD); it is not an abbreviation; or the abbreviation itself is the standard name nobody expands (DNA, HIV, COVID-19, SARS-CoV-2, ChatGPT, XGBoost, microRNA). Return JSON only."""
SCHEMA = {"type": "object", "properties": {"items": {"type": "array", "items": {"type": "object", "properties": {"n": {"type": "integer"}, "expansion": {"type": ["string", "null"]}}, "required": ["n", "expansion"], "additionalProperties": False}}}, "required": ["items"], "additionalProperties": False}
client = anthropic.Anthropic(api_key=os.environ.get('CLAUDE_API_KEY') or os.environ.get('ANTHROPIC_API_KEY'), max_retries=4, timeout=600)
def one(batch):
    txt = "\n".join(f"{n + 1}. {a['display_name']}" for n, a in enumerate(batch))
    for attempt in range(6):
        try:
            m = client.messages.create(model='claude-opus-5-5', max_tokens=8000, system=SYSTEM, messages=[{'role': 'user', 'content': txt}], extra_body={'output_config': {'effort': 'low', 'format': {'type': 'json_schema', 'schema': SCHEMA}}})
            got = {it['n']: it['expansion'] for it in json.loads("".join(b.text for b in m.content if b.type == 'text'))['items']}
            return [{'kid': a['kid'], 'acronym': a['display_name'], 'expansion': got.get(n + 1)} for n, a in enumerate(batch)], m.usage.input_tokens * 4e-6 + m.usage.output_tokens * 20e-6
        except Exception: time.sleep(5 * (attempt + 1))
    return [{'kid': a['kid'], 'acronym': a['display_name'], 'expansion': None} for a in batch], 0.0
out, cost = [], 0.0
with ThreadPoolExecutor(16) as ex:
    for r_, c in ex.map(one, [acr[i:i + 150] for i in range(0, len(acr), 150)]): out += r_; cost += c
def lookup(s):
    k = kid0(s) if s else None
    if not k: return None
    for c in (k, k + 's', k[:-1] if k.endswith('s') else None):
        if c and (c in V or c in syn): return syn.get(c, c)
    return None
merges = []
with open(f'{HERE}/data/acronyms_judged' + ('_small' if len(sys.argv) > 1 else '') + '.jsonl', 'w') as f:
    for r in out:
        r['exp_kid'] = lookup(r['expansion']); f.write(json.dumps(r, ensure_ascii=False) + '\n')
        a, b = r['kid'], r['exp_kid']
        if b and b != a and a in V and b in V:
            canon, other = (a, b) if V[a]['works_count'] >= V[b]['works_count'] else (b, a); merges.append({'kid': other, 'canon': canon})
with open(f'{HERE}/data/synmap_acronyms' + ('_small' if len(sys.argv) > 1 else '') + '.jsonl', 'w') as f:
    for m_ in merges: f.write(json.dumps(m_) + '\n')
print(f"{len(out)} acronyms, {sum(1 for r in out if r['expansion'])} expanded, {len(merges)} found in vocabulary -> merges, ${cost:.2f}")
for m_ in sorted(merges, key=lambda m_: -V[m_['kid']]['works_count'])[:20]: print('  ', V[m_['kid']]['display_name'], '->', V[m_['canon']]['display_name'])
