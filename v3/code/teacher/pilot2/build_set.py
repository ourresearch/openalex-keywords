"""Prompt pilot work set (PLAN Next Action 1, 2026-09-25): ~620 fixed works, seeded, grouped by what the record carries.
Non-xpac from the 10K God eval (abstract >= 300 chars / short / title-only / non-English), 100 author-keyword works (bench akw_field, independent gold),
140 xpac from xpac/data/xpac_sample.jsonl (specimen and sample-registration sources over-drawn: DiSSCo, PlutoF, SESAR, IGSN, NIFS, GBIF).
Then pulls per-work metadata for the mechanical scores (type, study designs, author-institution countries) and the purge / country lists from the vocab build.
uv run --with httpx python build_set.py   -> data/works.jsonl, data/meta.json, data/purge.json, data/country_kids.json"""
import json, os, random, sys
HERE = os.path.dirname(os.path.abspath(__file__)); S = os.path.dirname(os.path.dirname(HERE)); os.makedirs(f'{HERE}/data', exist_ok=True)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'common')); import dbx
rng = random.Random(1322)
def rd(p): return [json.loads(l) for l in open(p)]
eval_ids = set(open(f'{S}/god/data/god_eval_ids.txt').read().split())
ev = [w for w in rd(f'{S}/god/data/god_sample.jsonl') if w['id'] in eval_ids]
alen = lambda w: len((w.get('abstract') or '').strip())
def pick(pool, n, group):
    pool = sorted(pool, key=lambda w: w['id']); rng.shuffle(pool); out = []
    for w in pool[:n]: out.append({**{k: w.get(k) for k in ('id', 'title', 'abstract', 'journal', 'language', 'year', 'type')}, 'group': group})
    return out
nonen = [w for w in ev if (w.get('language') or 'en') != 'en']
en = [w for w in ev if (w.get('language') or 'en') == 'en']
works = pick([w for w in en if alen(w) >= 300], 160, 'abstract') + pick([w for w in en if 50 <= alen(w) < 300], 40, 'short_abstract') \
      + pick([w for w in en if alen(w) < 50], 120, 'title_only') + pick(nonen, 60, 'non_english')
seen = {w['id'] for w in works}
akw = [w for w in rd(f'{S}/bench/data/akw_field.jsonl') if w['id'] not in seen and w.get('author_kw')]
for w in pick(akw, 100, 'author_kw'):
    src = next(x for x in akw if x['id'] == w['id']); w['author_kw'] = src['author_kw']; w['type'] = w.get('type') or 'article'; works.append(w)
xs = [w for w in rd(f'{S}/xpac/data/xpac_sample.jsonl') if w['id'] not in seen]
QUOTA = [('DiSSCo', 15), ('Plutof', 15), ('SESAR', 10), ('IGSN', 10), ('NIFS', 15), ('Global Biodiversity', 10)]
used = set()
for key, n in QUOTA:
    got = pick([w for w in xs if key in (w.get('journal') or '') and w['id'] not in used], n, 'xpac')
    for w in got: w['xsrc'] = key; used.add(w['id'])
    works += got
rest = pick([w for w in xs if w['id'] not in used and not any(k in (w.get('journal') or '') for k, _ in QUOTA)], 65, 'xpac')
for w in rest: w['xsrc'] = 'other'
works += rest
with open(f'{HERE}/data/works.jsonl', 'w') as f:
    for w in works: f.write(json.dumps(w, ensure_ascii=False) + '\n')
print(len(works), 'works;', {g: sum(w['group'] == g for w in works) for g in dict.fromkeys(w['group'] for w in works)})
ids = ','.join(w['id'][1:] for w in works)
T = 'openalex.transient'
meta = dbx.sql(f"""SELECT concat('W', CAST(w.id AS STRING)) AS work_id, w.type, sd.study_designs,
    array_distinct(flatten(transform(w.authorships, x -> COALESCE(x.countries, array())))) AS inst_cc
    FROM openalex.works.openalex_works w LEFT JOIN openalex.works.works_study_design sd ON sd.work_id = w.id WHERE w.id IN ({ids})""", verbose=False)
json.dump({r['work_id']: r for r in meta}, open(f'{HERE}/data/meta.json', 'w'))
purge = dbx.sql(f"SELECT kid, keyword, kind, maps_to, dual_use FROM {T}.kw1322_purge", verbose=False); json.dump(purge, open(f'{HERE}/data/purge.json', 'w'))
ck = dbx.sql(f"SELECT kid, cc, rx FROM {T}.kw1322_country_kids", verbose=False); json.dump(ck, open(f'{HERE}/data/country_kids.json', 'w'))
print(f'meta {len(meta)}, purge {len(purge)}, country kids {len(ck)}')
