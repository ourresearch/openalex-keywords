"""#1476 per-sense gate (pre-registered 22:58 CT in PLAN.md): for each minority sense, 50 random corpus MOVES (top option is a minority sense
with p >= 0.5), drawn by md5 rank from the finished corpus buckets (buckets are a hash partition, so any set of them is a random sample of what
gets written). Labelled by Opus 5.5 medium with the gold prompt (gold_label.py senses_v1). Pass = precision >= 0.90 AND Wilson LB >= 0.85.
  python gate.py sample   -> data/gate_works.jsonl (+ data/moves_by_sense.json: moves counted so far)
  uv run --with anthropic python gold_label.py --in data/gate_works.jsonl --out data/gate_labels.jsonl
  python gate.py score    -> data/gate_v1.json"""
import collections, glob, hashlib, heapq, json, math, os, sys
OUT = os.environ.get('SENSES_OUT', 'out'); N = 50; T = 0.5
def wilson(k, n, z=1.96):
    if n == 0: return float('nan')
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)); return (c - r) / d
def move(p):
    """-> minority option index i >= 1 if the work moves, else None (p: {'s0':..,'s1':..,'none':..})"""
    top = max(p, key=p.get)
    return int(top[1:]) if top != 'none' and top != 's0' and p[top] >= T else None
if sys.argv[1] == 'sample':
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'common')); from dbx import sql
    heaps = collections.defaultdict(list); moves = collections.Counter(); pairs = 0; files = sorted(glob.glob(f'{OUT}/b*.jsonl'))
    for fn in files:
        for l in open(fn):
            r = json.loads(l)
            if 'p' not in r: continue
            pairs += 1; i = move(r['p'])
            if i is None: continue
            key = (r['heading'], i); moves[key] += 1
            hv = -int(hashlib.md5(f"{r['work_id']}|{r['heading']}|gate".encode()).hexdigest(), 16)   # keep the N smallest hashes
            if len(heaps[key]) < N: heapq.heappush(heaps[key], (hv, r['work_id']))
            elif hv > heaps[key][0][0]: heapq.heapreplace(heaps[key], (hv, r['work_id']))
    json.dump({'buckets': len(files), 'pairs': pairs, 'moves': {f'{h}|s{i}': n for (h, i), n in sorted(moves.items())}}, open('data/moves_by_sense.json', 'w'), indent=1)
    want = [(h, i, w) for (h, i), hp in heaps.items() for _, w in hp]
    vals = ",".join(f"({w}, '{h}')" for h, i, w in want)
    info = {(str(r['work_id']), r['heading']): r for r in sql(f"""WITH s AS (SELECT * FROM VALUES {vals} AS t(work_id, heading))
        SELECT c.work_id, c.heading, c.title, c.abstract, c.venue, w.primary_topic.field.display_name AS field
        FROM s JOIN openalex.transient.kw1476_corpus_in c ON c.work_id = s.work_id AND c.heading = s.heading
        JOIN openalex.works.openalex_works w ON w.id = s.work_id""", verbose=False)}
    with open('data/gate_works.jsonl', 'w') as f:
        for h, i, w in want:
            r = info[(str(w), h)]
            f.write(json.dumps({'kid': f'https://openalex.org/keywords/{h}', 'work_id': str(w), 'opt': i, 'title': r['title'], 'abstract': r['abstract'],
                                'venue': r['venue'], 'field': r['field']}, ensure_ascii=False) + '\n')
    print(f'{len(files)} buckets, {pairs} pairs, {sum(moves.values())} moves ({sum(moves.values())/max(1,pairs):.3f}); {len(heaps)} senses; {len(want)} works to judge')
else:
    M = json.load(open('data/sense_map.json')); W = {(r['kid'], r['work_id']): r['opt'] for r in map(json.loads, open('data/gate_works.jsonl'))}
    mv = json.load(open('data/moves_by_sense.json'))
    agg = collections.defaultdict(lambda: [0, 0, 0])   # right, judged, unlabelled
    for r in map(json.loads, open('data/gate_labels.jsonl')):
        i = W.get((r['kid'], r['work_id']))
        if i is None: continue
        key = (r['kid'].rsplit('/', 1)[-1], i)
        if r.get('choice') is None: agg[key][2] += 1; continue
        agg[key][1] += 1; agg[key][0] += r['choice'] == i
    res = []
    for (h, i), (k, n, u) in sorted(agg.items()):
        sm = M[h][i]; ok = n > 0 and k / n >= 0.90 and wilson(k, n) >= 0.85
        res.append({'heading': h, 'opt': f's{i}', 'sense_id': sm['sense_id'], 'sense_name': sm['sense_name'], 'label': sm['label'], 'right': k, 'judged': n,
                    'unlabelled': u, 'prec': round(k / n, 3) if n else None, 'lb': round(wilson(k, n), 3) if n else None,
                    'moves_so_far': mv['moves'].get(f'{h}|s{i}', 0), 'pass': ok})
    json.dump({'buckets': mv['buckets'], 'senses': res}, open('data/gate_v1.json', 'w'), indent=1, ensure_ascii=False)
    P = [r for r in res if r['pass']]; tm = sum(r['moves_so_far'] for r in res)
    print(f"{len(P)}/{len(res)} senses pass; they carry {sum(r['moves_so_far'] for r in P)}/{tm} moves so far "
          f"({sum(r['moves_so_far'] for r in P)/max(1,tm):.3f}); pooled precision of judged moves {sum(r['right'] for r in res)}/{sum(r['judged'] for r in res)}")
    for r in sorted(res, key=lambda r: (r['prec'] or 0))[:30]:
        print(f"  {'PASS' if r['pass'] else 'fail'} {r['heading']} -> {r['label']}: {r['right']}/{r['judged']} lb {r['lb']} moves {r['moves_so_far']}")
