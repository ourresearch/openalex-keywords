"""Country name table for the country-evidence rule (Jason 2026-09-25: keep a country keyword only when the country is named in the title/abstract or an author institution is there).
ISO 3166-1 list from pycountry (+ Kosovo XK); Opus 5.5 supplies per country: keyword aliases (strings that, as a whole keyword, mean the country), adjectives/demonyms, and native names in the official languages' scripts.
Writes data/countries.json [{cc, name, aliases, adjectives, native}] and openalex.transient.kw1322_countries.
uv run --with pycountry --with anthropic --with httpx python build_countries.py"""
import json, os, sys, time, concurrent.futures as cf
import pycountry, anthropic
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'common')); import dbx
HERE = os.path.dirname(os.path.abspath(__file__)); os.makedirs(f'{HERE}/data', exist_ok=True)
C = [(c.alpha_2, getattr(c, 'common_name', None) or c.name, getattr(c, 'official_name', c.name)) for c in pycountry.countries] + [('XK', 'Kosovo', 'Republic of Kosovo')]
SYSTEM = """You build a lookup table used to check whether a scholarly work's title or abstract mentions a country. For each country (ISO 3166-1 code, name, official name) give:
- aliases: lower-case English strings that, used as a whole keyword on a paper, mean this country: the common name, official name, short forms and abbreviations in real use (e.g. united states: "united states", "usa", "us", "u.s.", "u.s.a.", "united states of america"; united kingdom: "uk", "u.k.", "britain", "great britain"; south korea: "korea", "republic of korea", "south korea"). Include former names still used in the literature only if they refer to the same country (e.g. "burma" for Myanmar, "swaziland" for Eswatini, "czech republic"/"czechia"). No cities, regions or languages.
- adjectives: lower-case English adjective / demonym forms (e.g. "japanese", "indonesian", "brazilian"). Omit an adjective if it is also a very common English word unrelated to the country (e.g. "turkish" is fine, but skip "french fries"-style phrases; single words only).
- native: the country's own name(s) in its official or main languages, in their native scripts, lower-case where the script has case (e.g. japan: "日本", "にほん", "nippon"; germany: "deutschland"; spain: "españa"; china: "中国", "中國"; south korea: "한국", "대한민국"; russia: "россия"; indonesia: "indonesia"; brazil: "brasil"), plus the native adjective forms when common in titles (e.g. "deutsch", "español", "española", "brasileiro", "brasileira", "français", "française", "русский", "российской").
Be complete for every country given; return every index."""
SCHEMA = {"type": "object", "properties": {"v": {"type": "array", "items": {"type": "object", "properties": {"i": {"type": "integer"}, "aliases": {"type": "array", "items": {"type": "string"}},
          "adjectives": {"type": "array", "items": {"type": "string"}}, "native": {"type": "array", "items": {"type": "string"}}}, "required": ["i", "aliases", "adjectives", "native"], "additionalProperties": False}}},
          "required": ["v"], "additionalProperties": False}
client = anthropic.Anthropic(api_key=os.environ['CLAUDE_API_KEY'])
def call(batch):
    user = "\n".join(f"{i}\t{cc}\t{n}\t{o}" for i, (cc, n, o) in enumerate(batch))
    for attempt in range(5):
        try:
            r = client.messages.create(model='claude-opus-5-5', max_tokens=16000, system=SYSTEM, messages=[{'role': 'user', 'content': "index<TAB>code<TAB>name<TAB>official name\n" + user}],
                                       extra_body={'output_config': {'effort': 'medium', 'format': {'type': 'json_schema', 'schema': SCHEMA}}})
            text = "".join(b.text for b in r.content if b.type == 'text')
            if not text: raise ValueError(f'empty, {r.stop_reason}')
            v = {x['i']: x for x in json.loads(text)['v'] if 0 <= x['i'] < len(batch)}
            if len(v) < len(batch): raise ValueError(f'{len(v)}/{len(batch)} returned')
            return [{'cc': batch[i][0], 'name': batch[i][1], 'aliases': v[i]['aliases'], 'adjectives': v[i]['adjectives'], 'native': v[i]['native']} for i in range(len(batch))], r.usage
        except Exception as e:
            print('retry', attempt, str(e)[:200], flush=True); time.sleep(10 * (attempt + 1))
    raise RuntimeError('batch failed')
out, cost = [], 0.0
with cf.ThreadPoolExecutor(12) as ex:
    for res, u in ex.map(call, [C[i:i + 12] for i in range(0, len(C), 12)]):
        out += res; cost += u.input_tokens * 4e-6 + u.output_tokens * 20e-6
for x in out:
    x['aliases'] = sorted({a.lower().strip() for a in x['aliases'] + [x['name'].lower()] if a.strip()})
    x['adjectives'] = sorted({a.lower().strip() for a in x['adjectives'] if a.strip()}); x['native'] = sorted({a.lower().strip() for a in x['native'] if a.strip()})
json.dump(out, open(f'{HERE}/data/countries.json', 'w'), ensure_ascii=False, indent=0)
q = lambda s: "'" + s.replace("\\", "\\\\").replace("'", "\\'") + "'"
arr = lambda xs: "array(" + ", ".join(q(s) for s in xs) + ")" if xs else "CAST(array() AS ARRAY<STRING>)"
dbx.sql("CREATE OR REPLACE TABLE openalex.transient.kw1322_countries AS SELECT * FROM VALUES " + ", ".join(f"({q(x['cc'])}, {q(x['name'])}, {arr(x['aliases'])}, {arr(x['adjectives'])}, {arr(x['native'])})" for x in out)
        + " AS t(cc, name, aliases, adjectives, native)", verbose=False)
print(f'{len(out)} countries, ${cost:.2f}; table openalex.transient.kw1322_countries written', flush=True)
