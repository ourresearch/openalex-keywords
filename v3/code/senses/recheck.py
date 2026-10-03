"""#1476 near-miss recheck (pre-registered in EXPLORE.md § Near-miss recheck). The 11 senses at 47/50 in gate_v1.json get 150 FRESH corpus moves each
(not in gate_works.jsonl; smallest md5(work|heading|recheck)), labelled by gold_label.py; pass on the fresh sample alone: prec >= 0.90 AND Wilson LB >= 0.85.
  python recheck.py draw      -> data/recheck_works.jsonl
  uv run --with anthropic python gold_label.py --in data/recheck_works.jsonl --out data/recheck_labels.jsonl
  python recheck.py score     -> data/recheck.json"""
import collections, glob, hashlib, heapq, json, os, sys
from math import sqrt
OUT = os.environ.get('SENSES_OUT', 'out'); N = 150; T = 0.5
def move(p):   # same as gate.py
    top = max(p, key=p.get)
    return int(top[1:]) if top != 'none' and top != 's0' and p[top] >= T else None
def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; return (p + z * z / (2 * n) - z * sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / d
G = json.load(open('data/gate_v1.json'))['senses']
target = {(s['heading'], int(s['opt'][1:])): s for s in G if not s['pass'] and s['judged'] == 50 and s['right'] == 47}
assert len(target) == 11, len(target)
if sys.argv[1] == 'draw':
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'common')); from dbx import sql
    seen = {(r['kid'].rsplit('/', 1)[-1], r['opt'], str(r['work_id'])) for r in map(json.loads, open('data/gate_works.jsonl'))}
    heaps = collections.defaultdict(list)   # max-heap via negation: keep the N smallest hashes
    for fn in sorted(glob.glob(f'{OUT}/b*.jsonl')):
        for l in open(fn):
            r = json.loads(l)
            if 'p' not in r: continue
            i = move(r['p']); key = (r['heading'], i)
            if i is None or key not in target or (r['heading'], i, str(r['work_id'])) in seen: continue
            hv = -int(hashlib.md5(f"{r['work_id']}|{r['heading']}|recheck".encode()).hexdigest(), 16)
            if len(heaps[key]) < N: heapq.heappush(heaps[key], (hv, r['work_id']))
            elif hv > heaps[key][0][0]: heapq.heapreplace(heaps[key], (hv, r['work_id']))
    want = [(h, i, w) for (h, i), hp in heaps.items() for _, w in hp]
    for k in sorted(target): print(k, len(heaps[k]))
    vals = ",".join(f"({w}, '{h}')" for h, i, w in want)
    info = {(str(r['work_id']), r['heading']): r for r in sql(f"""WITH s AS (SELECT * FROM VALUES {vals} AS t(work_id, heading))
        SELECT c.work_id, c.heading, c.title, c.abstract, c.venue, w.primary_topic.field.display_name AS field
        FROM s JOIN openalex.transient.kw1476_corpus_in c ON c.work_id = s.work_id AND c.heading = s.heading
        JOIN openalex.works.openalex_works w ON w.id = s.work_id""", verbose=False)}
    with open('data/recheck_works.jsonl', 'w') as f:
        for h, i, w in want:
            r = info[(str(w), h)]
            f.write(json.dumps({'kid': f'https://openalex.org/keywords/{h}', 'work_id': str(w), 'opt': i, 'title': r['title'], 'abstract': r['abstract'],
                                'venue': r['venue'], 'field': r['field']}, ensure_ascii=False) + '\n')
    print(len(want), 'works written')
else:
    W = {(r['kid'], r['work_id']): r['opt'] for r in map(json.loads, open('data/recheck_works.jsonl'))}
    mv = json.load(open('data/moves_by_sense.json'))['moves']; agg = collections.defaultdict(lambda: [0, 0, 0])
    for r in map(json.loads, open('data/recheck_labels.jsonl')):
        i = W.get((r['kid'], r['work_id']))
        if i is None: continue
        key = (r['kid'].rsplit('/', 1)[-1], i)
        if r.get('choice') is None: agg[key][2] += 1; continue
        agg[key][1] += 1; agg[key][0] += r['choice'] == i
    res = []
    for key in sorted(target, key=lambda k: -mv.get(f'{k[0]}|s{k[1]}', 0)):
        k, n, u = agg[key]; s = target[key]
        res.append({'heading': key[0], 'opt': f's{key[1]}', 'sense_id': s['sense_id'], 'sense_name': s['sense_name'], 'right': k, 'judged': n, 'unlabelled': u,
                    'prec': round(k / n, 3) if n else None, 'lb': round(wilson(k, n), 3) if n else None, 'moves': mv.get(f'{key[0]}|s{key[1]}', 0),
                    'pass': n > 0 and k / n >= 0.90 and wilson(k, n) >= 0.85})
    json.dump(res, open('data/recheck.json', 'w'), indent=1, ensure_ascii=False)
    for r in res: print(f"{'PASS' if r['pass'] else 'fail'} {r['sense_name']}: {r['right']}/{r['judged']} (+{r['unlabelled']} unlabelled) lb {r['lb']} moves {r['moves']:,}")
    P = [r for r in res if r['pass']]; print(f"{len(P)}/11 pass, carrying {sum(r['moves'] for r in P):,} moves")
