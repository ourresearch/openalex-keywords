"""Synonym bake-off, scoring (no model calls). Per arm: synonyms per keyword; precision = share labelled SAME by each judge and by both;
CLOSE and WRONG shares (both judges' mean); recall = share of the pooled "both judges SAME" synonyms the arm found (pool = all arms +
Wikidata, so recall is relative); Wikidata recall = share of human-written Wikidata aliases that both judges call SAME which the arm
also wrote (the outside rung). Judge agreement reported. Writes syn_scores.json. For the release the projected-cost column was
removed; the scores are unchanged. Runs on the released files in data/alternatives/bakeoff/ (gunzip them into the working directory).
  python3 syn_score.py"""
import json, collections
from syn_judge import pool, ARMS
S, G, P = pool()
J = {}
for j in ("opus", "astra"):
    J[j] = {}
    for l in open(f"syn_judge_{j}.jsonl"):
        r = json.loads(l); J[j].setdefault(r["kid"], {}).update({k.strip().lower(): v for k, v in r["labels"].items()})
lab = lambda j, k, c: J[j].get(k, {}).get(c.strip().lower())
both_same = {k: {c.lower() for c in P[k] if lab("opus", k, c) == "SAME" and lab("astra", k, c) == "SAME"} for k in P}
agree = [lab("opus", k, c) == lab("astra", k, c) for k in P for c in P[k] if lab("opus", k, c) and lab("astra", k, c)]
print(f"judges agree on {sum(agree) / len(agree):.1%} of {len(agree):,} candidates; missing labels: opus {sum(1 for k in P for c in P[k] if not lab('opus', k, c))}, astra {sum(1 for k in P for c in P[k] if not lab('astra', k, c))}")
wd = {k: {a.strip().lower() for a in S[k]["wd_aliases"] + ([S[k]["wd_label"]] if S[k]["wd_label"] else [])} & both_same[k] for k in S}
rows = []
arms = ARMS + ["wikidata"]
for a in arms:
    cand = {k: ({x.strip().lower() for x in G[a].get(k, [])} if a != "wikidata" else {x.strip().lower() for x in S[k]["wd_aliases"] + ([S[k]["wd_label"]] if S[k]["wd_label"] else [])}) - {S[k]["name"].lower()} for k in S}
    n = sum(len(v) for v in cand.values())
    c = collections.Counter();
    for j in ("opus", "astra"):
        for k, v in cand.items():
            for x in v: c[(j, lab(j, k, x))] += 1
    both = sum(1 for k, v in cand.items() for x in v if x in both_same[k])
    rec = sum(len(cand[k] & both_same[k]) for k in S) / max(1, sum(len(v) for v in both_same.values()))
    wdr = sum(len(cand[k] & wd[k]) for k in S) / max(1, sum(len(v) for v in wd.values()))
    close = (c[("opus", "CLOSE")] + c[("astra", "CLOSE")]) / 2 / max(n, 1); wrong = (c[("opus", "WRONG")] + c[("astra", "WRONG")]) / 2 / max(n, 1)
    rows.append((a, n / len(S), c[("opus", "SAME")] / max(n, 1), c[("astra", "SAME")] / max(n, 1), both / max(n, 1), close, wrong, rec, wdr))
print("\n| source | synonyms per keyword | SAME (Opus judge) | SAME (Astra judge) | SAME (both) | CLOSE | WRONG | recall of pooled good synonyms | recall of good Wikidata aliases |")
print("|---|---|---|---|---|---|---|---|---|")
for r in rows: print(f"| {r[0]} | {r[1]:.1f} | {r[2]:.0%} | {r[3]:.0%} | {r[4]:.0%} | {r[5]:.0%} | {r[6]:.0%} | {r[7]:.0%} | {r[8]:.0%} |")
for st in ("head", "torso", "tail"):
    ks = [k for k in S if S[k]["stratum"] == st]
    line = []
    for a in ARMS:
        cand = {k: {x.strip().lower() for x in G[a].get(k, [])} - {S[k]["name"].lower()} for k in ks}
        n = sum(len(v) for v in cand.values()); both = sum(1 for k, v in cand.items() for x in v if x in both_same[k])
        line.append(f"{a} {both / max(n, 1):.0%}")
    print(f"both-judge precision, {st}: " + ", ".join(line))
json.dump({"rows": rows}, open("syn_scores.json", "w"), indent=1)
