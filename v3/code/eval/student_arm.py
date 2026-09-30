"""The shipped keyword student on the human-gold harness (#1322, 2026-09-29; asked by the 'Model development practices review' session, Jason okayed, < $10):
tag every work of the harness golds with the v3fix student (Modal Tagger.tag_texts, B200 FP8, the corpus prompt: title / abstract <= 6000 / venue) and write
arms/<gold>__student_v3fix.jsonl in the harness format ({id, scored: [[cand_id, score]...]}, score = exp(mean token log-prob), model order).
MeSH: a keyword maps to a descriptor when its canon form equals a descriptor name or entry term. TIBKAT is skipped (German GND headings; English keywords
cannot match without translation). uv run --with modal --with numpy --with nltk python student_arm.py"""
import json, math, os, sys
import modal
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import golds
from common import canon, P
GOLDS = ['akw_field', 'akw_pubmed', 'akw_landing', 'akw_repo', 'akw_kp20k', 'kp20k', 'inspec', 'mesh', 'judge']
G = {n: golds.load(n).load_split() for n in GOLDS}
def text(w):
    return "\n".join([f"Title: {w.get('title') or '(none)'}", f"Abstract: {(w.get('abstract') or '(none)')[:6000]}", f"Venue: {w.get('venue') or w.get('journal') or '(unknown)'}"])
ids = sorted({i for g in G.values() for i in g.works}); works = {}
for g in G.values():
    for i, w in g.works.items(): works.setdefault(i, w)
cache = f'{HERE}/data/student_v3fix_tags.json'
tags = json.load(open(cache)) if os.path.exists(cache) else {}
todo = [i for i in ids if i not in tags]
print(f'{len(ids)} works across {len(GOLDS)} golds; {len(todo)} to tag', flush=True)
if todo:
    T = modal.Cls.from_name('oxjob1322-corpus-v3', 'Tagger').with_options(gpu='B200')(model='/vol/models/student_qwen4b_v3fix', quant='fp8')
    for s in range(0, len(todo), 8000):
        chunk = todo[s:s + 8000]; out, sec = T.tag_texts.remote([text(works[i]) for i in chunk])
        for i, (k, lp) in zip(chunk, out): tags[i] = [[kw, l] for kw, l in zip(k, lp)]
        json.dump(tags, open(cache, 'w')); print(f'  tagged {s + len(chunk)}/{len(todo)} ({sec:.0f}s)', flush=True)
M = golds.mesh_desc(); mesh_map = {}
for u, d in M['D'].items():
    for n in [d['name']] + d.get('entry_terms', []): mesh_map.setdefault(canon(n), u)
for n, g in G.items():
    out = []
    for i in g.works:
        sc, seen = [], set()
        for kw, lp in tags.get(i, []):
            cid = mesh_map.get(canon(kw)) if n == 'mesh' else golds.cand_id(g, kw)
            if not cid or cid in seen: continue
            seen.add(cid); sc.append([cid, round(math.exp(lp), 4)])
        out.append({'id': i, 'scored': sc})
    with open(f'{HERE}/arms/{n}__student_v3fix.jsonl', 'w') as f:
        for r in out: f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(n, len(out), 'rows; mean candidates', round(sum(len(r['scored']) for r in out) / max(1, len(out)), 2), flush=True)
