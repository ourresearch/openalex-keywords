"""Second look at high-impact synonym merges (moved keyword on >= 10K works) with context: 3 example work titles + venues per keyword, Opus 5.5,
20 merges per call. A merge is kept only if confirmed. -> data/hi_reverified.jsonl. uv run --with anthropic python verify_hi.py"""
import json, os, time
from concurrent.futures import ThreadPoolExecutor
import anthropic
HERE = os.path.dirname(os.path.abspath(__file__))
name = {}
for _p in ('data/vocab_g.jsonl', 'data/vocab_h.jsonl'):
    for l in open(f'{HERE}/{_p}'): _r = json.loads(l); name.setdefault(_r['kid'], _r['display_name'])
import sys as _s
src = [json.loads(l) for p in ('data/hi_verified_v3.jsonl', 'data/hi_verified_v4.jsonl') for l in open(f'{HERE}/{p}')]
hi = [r for r in src if r.get('verified') is False]
ex = {**json.load(open(f'{HERE}/data/hi_examples_v3.json')), **json.load(open(f'{HERE}/data/hi_examples_v4.json'))}
SYSTEM = """You maintain the subject vocabulary of a scholarly index; a usable vocabulary has ONE heading per concept. Each numbered item gives two keyword headings, each with three example works tagged with it (title [venue]); judge them as used on these works. Merge (true) when a searcher looking for either would expect the other's works under the same heading, because they name the same subject:
- synonyms and near-synonyms used interchangeably in the literature (older adults / elderly, college students / university students, vitamin C / ascorbic acid, mice / mouse),
- an acronym or abbreviation and its expansion, ONLY when the acronym unambiguously means that expansion across research (IL-6 / interleukin-6, MRI / magnetic resonance imaging, NAFLD / non-alcoholic fatty liver disease); NOT when it has other common meanings (ECE, PD / Pd, DM, PET, ATM) or is a chemical symbol or formula that also reads as something else,
- a renamed concept and its old name (mpox / monkeypox, MASLD / NAFLD),
- spelling, transliteration, hyphenation, word-order or grammatical variants (tumor / tumour, nudging / nudges, CAR-T cells / CAR T cells),
- a Latin taxon and its common name for the same group (Tardigrada / tardigrades).
Do NOT merge (false): different things (World War I / World War II, food safety / food security), broader vs narrower (animals / dogs, deep learning / deep neural networks, monetary policy / monetary policy transmission), a thing vs a therapy or process using it (CAR-T cells / CAR-T cell therapy), a quantity vs its measurement (electron density / electron density measurement), a disease vs the pandemic as a societal event (COVID-19 / COVID-19 pandemic), related but distinct concepts (gender differences / sex differences), proper names that must keep their own form (UK Labour Party, a Latin genus name).
For merges, `keep` is the better heading: the standard, most widely used form in current literature (1 = first, 2 = second).
Return JSON only: for each item n and merge (true/false)."""
SCHEMA = {"type": "object", "properties": {"items": {"type": "array", "items": {"type": "object", "properties": {"n": {"type": "integer"}, "merge": {"type": "boolean"}}, "required": ["n", "merge"], "additionalProperties": False}}}, "required": ["items"], "additionalProperties": False}
client = anthropic.Anthropic(api_key=os.environ.get('CLAUDE_API_KEY') or os.environ.get('ANTHROPIC_API_KEY'), max_retries=4, timeout=600)
def block(k): return f"\"{name[k]}\"\n" + "\n".join(f"    - {t}" for t in ex.get(k, []))
def one(batch):
    txt = "\n\n".join(f"{n + 1}. A = {block(r['kid'])}\n   B = {block(r['canon'])}" for n, r in enumerate(batch))
    for attempt in range(6):
        try:
            m = client.messages.create(model='claude-opus-5-5', max_tokens=8000, system=SYSTEM, messages=[{'role': 'user', 'content': txt}],
                                       extra_body={'output_config': {'effort': 'medium', 'format': {'type': 'json_schema', 'schema': SCHEMA}}})
            got = {it['n']: it['merge'] for it in json.loads("".join(b.text for b in m.content if b.type == 'text'))['items']}
            return [{**r, 'verified': got.get(n + 1)} for n, r in enumerate(batch)], m.usage.input_tokens * 4e-6 + m.usage.output_tokens * 20e-6
        except Exception as e:
            err = repr(e)[:200]; time.sleep(5 * (attempt + 1))
    return [{**r, 'verified': None} for r in batch], 0.0
batches = [hi[i:i + 20] for i in range(0, len(hi), 20)]; cost = 0.0
with open(f'{HERE}/data/hi_reverified.jsonl', 'w') as f, ThreadPoolExecutor(16) as ex_:
    for rows, c in ex_.map(one, batches):
        cost += c
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + '\n')
R = [json.loads(l) for l in open(f'{HERE}/data/hi_reverified.jsonl')]
print(f"{len(R)} checked, confirmed {sum(r['verified'] is True for r in R)}, rejected {sum(r['verified'] is False for r in R)}, unknown {sum(r['verified'] is None for r in R)}, ${cost:.2f}")
for r in [r for r in R if r['verified'] is False][:25]: print('REJECT', name[r['kid']], '->', name[r['canon']])
