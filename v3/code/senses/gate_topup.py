"""#1476 gate top-up on the full corpus (all 320 buckets). Senses with fewer than 50 drawn moves in the 64-bucket gate (gate_v1_64b.json) or none at all
get more moves from the full output, picked by the same md5 rank as gate.py (smallest hashes not already drawn), up to 50 drawn in all. Senses that
already had 50 drawn are not redrawn. Appends to data/gate_works.jsonl and rewrites data/moves_by_sense.json with full-corpus counts. Then:
  uv run --with anthropic python gold_label.py --in data/gate_works.jsonl --out data/gate_labels.jsonl   (skips labelled rows)
  python gate.py score"""
import collections, glob, hashlib, heapq, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'common')); from dbx import sql
OUT = os.environ.get('SENSES_OUT', 'out'); N = 50; T = 0.5
def move(p):   # same as gate.py (gate.py runs on import)
    top = max(p, key=p.get)
    return int(top[1:]) if top != 'none' and top != 's0' and p[top] >= T else None
M = json.load(open('data/sense_map.json'))
drawn = collections.defaultdict(set)
for r in map(json.loads, open('data/gate_works_64b.jsonl')): drawn[(r['kid'].rsplit('/', 1)[-1], r['opt'])].add(str(r['work_id']))
target = {(h, i) for h, opts in M.items() for i in range(1, len(opts)) if len(drawn[(h, i)]) < N}
heaps = collections.defaultdict(list); moves = collections.Counter(); pairs = 0; files = sorted(glob.glob(f'{OUT}/b*.jsonl'))
for fn in files:
    for l in open(fn):
        r = json.loads(l)
        if 'p' not in r: continue
        pairs += 1; i = move(r['p'])
        if i is None: continue
        key = (r['heading'], i); moves[key] += 1
        if key not in target or str(r['work_id']) in drawn[key]: continue
        need = N - len(drawn[key]); hv = -int(hashlib.md5(f"{r['work_id']}|{r['heading']}|gate".encode()).hexdigest(), 16)
        if len(heaps[key]) < need: heapq.heappush(heaps[key], (hv, r['work_id']))
        elif hv > heaps[key][0][0]: heapq.heapreplace(heaps[key], (hv, r['work_id']))
json.dump({'buckets': len(files), 'pairs': pairs, 'moves': {f'{h}|s{i}': n for (h, i), n in sorted(moves.items())}}, open('data/moves_by_sense.json', 'w'), indent=1)
want = [(h, i, w) for (h, i), hp in heaps.items() for _, w in hp]
print(f'{len(files)} buckets, {pairs} pairs, {sum(moves.values())} moves ({sum(moves.values())/max(1,pairs):.3f}); {len(target)} target senses; {len(want)} works to add')
for (h, i) in sorted(target): print(f'  {h} s{i}: {len(drawn[(h, i)])} drawn, {moves[(h, i)]} moves in full corpus, +{len(heaps[(h, i)])}')
if want:
    vals = ",".join(f"({w}, '{h}')" for h, i, w in want)
    info = {(str(r['work_id']), r['heading']): r for r in sql(f"""WITH s AS (SELECT * FROM VALUES {vals} AS t(work_id, heading))
        SELECT c.work_id, c.heading, c.title, c.abstract, c.venue, w.primary_topic.field.display_name AS field
        FROM s JOIN openalex.transient.kw1476_corpus_in c ON c.work_id = s.work_id AND c.heading = s.heading
        JOIN openalex.works.openalex_works w ON w.id = s.work_id""", verbose=False)}
    with open('data/gate_works.jsonl', 'a') as f:
        for h, i, w in want:
            r = info[(str(w), h)]
            f.write(json.dumps({'kid': f'https://openalex.org/keywords/{h}', 'work_id': str(w), 'opt': i, 'title': r['title'], 'abstract': r['abstract'],
                                'venue': r['venue'], 'field': r['field']}, ensure_ascii=False) + '\n')
