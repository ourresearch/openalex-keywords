"""Relabel sample (2026-09-25; prompt v3 = pilot2 arm F). 1M works, proportional to the population (non-xpac 327M : xpac 147M ≈ 69 : 31),
so it is equivalent to a simple random sample:
  - non-xpac: 690K random of the 1M already sampled for the God labels (god/data/god_sample*.jsonl; text on disk; old and new labels comparable on identical works),
    always including the 10K old eval ids;
  - xpac: all works in 11 random 1/5000 hash buckets (≈ 29.5K each; bucket 313 = the xpac pilot sample excluded), pulled one bucket at a time, checkpointed.
Eval (held out from training): the old 10K non-xpac eval ids + 4.5K random xpac (the population ratio). Writes data/sample.jsonl, data/eval_ids.txt.
uv run --with httpx python build_sample.py"""
import json, os, random, sys, glob
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'common')); from dbx import sql
HERE = os.path.dirname(os.path.abspath(__file__)); G = HERE   # released layout: god_sample*.jsonl and god_eval_ids.txt live in code/teacher/data/ too; os.makedirs(f'{HERE}/data/xpac_buckets', exist_ok=True)
FIELDS = ('id', 'title', 'abstract', 'journal', 'language', 'year', 'type')
buckets = [b for b in random.Random(20260925).sample(range(5000), 12) if b != 313][:11]
for b in buckets:
    p = f'{HERE}/data/xpac_buckets/{b}.jsonl'
    if os.path.exists(p): continue
    rows = sql(f"""SELECT concat('W', CAST(w.id AS STRING)) AS id, w.title, w.abstract, w.primary_location.source.display_name AS journal, w.language,
                          w.publication_year AS year, w.type FROM openalex.works.openalex_works w WHERE w.is_xpac AND pmod(hash(w.id), 5000) = {b}""",
               warehouse=os.environ.get('DATABRICKS_WAREHOUSE_ID'), verbose=False)
    with open(p + '.tmp', 'w') as f:
        for r in rows: f.write(json.dumps({k: r.get(k) for k in FIELDS}, ensure_ascii=False) + '\n')
    os.replace(p + '.tmp', p); print(f'bucket {b}: {len(rows)} xpac works', flush=True)
xp = [json.loads(l) for b in buckets for l in open(f'{HERE}/data/xpac_buckets/{b}.jsonl')]
old_eval = set(open(f'{G}/data/god_eval_ids.txt').read().split())
non = [json.loads(l) for p in [f'{G}/data/god_sample.jsonl'] + sorted(glob.glob(f'{G}/data/god_sample_w*.jsonl')) for l in open(p)]
rng = random.Random(1322)
ev_non = [w for w in non if w['id'] in old_eval]; rest = [w for w in non if w['id'] not in old_eval]; rng.shuffle(rest)
non_s = ev_non + rest[:690_000 - len(ev_non)]
rng.shuffle(xp); n_x = 1_000_000 - len(non_s); xp_s = xp[:n_x]
ev_x = [w['id'] for w in xp_s[:4500]]
seen = set(); out = []
for w, grp in [(w, 'nonxpac') for w in non_s] + [(w, 'xpac') for w in xp_s]:
    if w['id'] in seen: continue
    seen.add(w['id']); out.append({**{k: w.get(k) for k in FIELDS}, 'xpac': grp == 'xpac'})
rng.shuffle(out)
with open(f'{HERE}/data/sample.jsonl', 'w') as f:
    for w in out: f.write(json.dumps(w, ensure_ascii=False) + '\n')
open(f'{HERE}/data/eval_ids.txt', 'w').write('\n'.join([w['id'] for w in ev_non] + ev_x) + '\n')
print(f'sample {len(out)} (non-xpac {len(non_s)}, xpac {len(xp_s)} of {len(xp)} pulled); eval {len(ev_non)} + {len(ev_x)}; buckets {buckets}')
