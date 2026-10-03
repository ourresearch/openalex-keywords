"""Analyze the outside-judge results (2026-09-30). Prints markdown tables (data/eval/outside_judge/tables.md); writes summary.json.
Measurement B (pre-swap AMR) needs the unreleased recall_precision sample and is skipped without it.
    python3 analyze.py > ../../data/eval/outside_judge/tables.md"""
import json, math, random, collections, os
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "..", "data", "eval", "outside_judge")
def wilson(k, n, z=1.96):
    if n == 0: return (float("nan"),) * 3
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0, c - h), min(1, c + h)
pc = lambda t: f"{100*t[0]:.1f}% ({100*t[1]:.1f}–{100*t[2]:.1f})"
J = {}; errors = collections.Counter(); cost = 0.0; refusals = []
for l in open(os.path.join(DATA, "judgments.jsonl")):
    x = json.loads(l); cost += x.get("cost", 0)
    if x.get("refusal"): refusals.append(x["key"])
    if x.get("result") is not None: J[x["key"]] = x
    else: errors[x["key"]] = x.get("error")
errors = collections.Counter(k.split("|")[0] for k in errors if k not in J)
rows = [json.loads(l) for l in open(os.path.join(DATA, "sample_a.jsonl"))][:400]
W = {r["work_id"]: r for r in rows}
noabs = lambda r: not r["abs_len"] or int(r["abs_len"]) == 0
xpac = lambda r: r["is_xpac"] in (True, "true")
S = {"summary": {}}

# ---- Measurement A: per-keyword accuracy
def boot_ci(per_work, B=4000):
    # cluster bootstrap over works for per-keyword accuracy (keywords within a work are correlated)
    rnd = random.Random(1); n = len(per_work); vals = []
    for _ in range(B):
        s = [per_work[rnd.randrange(n)] for _ in range(n)]
        a = sum(x[0] for x in s); t = sum(x[1] for x in s); vals.append(a / t if t else 0)
    vals.sort(); return vals[int(0.025 * B)], vals[int(0.975 * B)]
print("## Measurement A: per-keyword accuracy\n")
print("| Slice | List | Works | Keywords | Accurate (Wilson 95% CI) | Work-cluster bootstrap CI | Per-work mean accuracy | Wrong/unsupported/wrong-sense | Too generic | Not aboutness | Malformed | Redundant (flag) |")
print("|---|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|")
slices = [("All", lambda r: True), ("With abstract", lambda r: not noabs(r)), ("No abstract", noabs),
          ("Non-xpac", lambda r: not xpac(r)), ("xpac", xpac)]
both = [w for w in W if f"A|{w}|old" in J and f"A|{w}|new" in J]
S["A_works_both_judged"] = len(both)
for sname, f in slices:
    for which in ("old", "new"):
        pw = []; prob = collections.Counter(); red = 0
        for w in both:
            if not f(W[w]): continue
            ks = J[f"A|{w}|{which}"]["result"]["keywords"]
            a = sum(k["verdict"] == "accurate" for k in ks); pw.append((a, len(ks)))
            for k in ks:
                if k["verdict"] == "inaccurate": prob[k["problem"]] += 1
                red += k["redundant"]
        A = sum(a for a, _ in pw); T = sum(t for _, t in pw)
        wi = wilson(A, T); bl, bh = boot_ci(pw); pwm = sum(a / t for a, t in pw) / len(pw)
        wrong = prob["wrong_or_unsupported"] + prob["wrong_sense"] + prob["none"]
        print(f"| {sname} | {which} | {len(pw)} | {T} | {pc(wi)} | {100*bl:.1f}–{100*bh:.1f} | {100*pwm:.1f}% | "
              f"{100*wrong/T:.1f}% | {100*prob['too_generic']/T:.1f}% | {100*prob['not_aboutness']/T:.1f}% | {100*prob['malformed']/T:.1f}% | {100*red/T:.1f}% |")
        S["summary"][f"A|{sname}|{which}"] = {"works": len(pw), "keywords": T, "acc": wi, "boot": [bl, bh], "per_work_mean": pwm,
                                              "problems": dict(prob), "redundant": red}
# per-work head-to-head from single judgments
better = collections.Counter()
for w in both:
    ao = S and (lambda k: sum(x["verdict"] == "accurate" for x in k) / len(k))
    o = ao(J[f"A|{w}|old"]["result"]["keywords"]); n = ao(J[f"A|{w}|new"]["result"]["keywords"])
    better["new higher" if n > o else "old higher" if o > n else "equal"] += 1
print(f"\nPer-work accuracy, new vs old (from the single-list judgments): {dict(better)}\n")
S["A_per_work_compare"] = dict(better)

# ---- Preference
print("## Preference (head-to-head)\n")
print("| Slice | n | New preferred | Old preferred | Tie | Neither useful |")
print("|---|---:|---:|---:|---:|---:|")
for sname, f in slices:
    c = collections.Counter()
    for w, r in W.items():
        x = J.get(f"P|{w}")
        if not x or not f(r): continue
        p = x["result"]["preferred"]
        if p in ("A", "B"): p = "new" if (p == "A") == x["a_is_new"] else "old"
        c[p] += 1
    n = sum(c.values())
    print(f"| {sname} | {n} | {c['new']} | {c['old']} | {c['tie']} | {c['neither_useful']} |")
    S["summary"][f"P|{sname}"] = dict(c)
# position bias check
pos = collections.Counter(J[f"P|{w}"]["result"]["preferred"] for w in W if f"P|{w}" in J)
print(f"\nRaw A/B picks (position-bias check): {dict(pos)}\n")

RP = os.path.join(HERE, "..", "recall_precision")  # unreleased pre-swap AMR sample (records carry work text)
if os.path.exists(os.path.join(RP, "samples.jsonl")):
    # ---- Measurement B: AMR
    print("## Measurement B: antimicrobial resistance\n")
    kid = "https://openalex.org/keywords/antimicrobial-resistance"
    samp = [json.loads(l) for fn in ("samples.jsonl", "samples_ext.jsonl") for l in open(os.path.join(RP, fn))]
    claude = {}
    for l in open(os.path.join(RP, "judgments.jsonl")):
        x = json.loads(l)
        if x.get("verdict") and x["kid"] == kid: claude[x["work_id"]] = x["verdict"]
    agg = collections.defaultdict(collections.Counter); aggc = collections.defaultdict(collections.Counter); agree = collections.Counter()
    for r in samp:
        if r["kid"] != kid: continue
        x = J.get(f"R|{kid}|{r['id']}")
        if not x: continue
        v = x["result"]["verdict"]; agg[r["stratum"]][v] += 1
        if r["id"] in claude:
            aggc[r["stratum"]][claude[r["id"]]] += 1
            agree[(claude[r["id"]], v)] += 1
    print("| Stratum | n | GPT yes (95% CI) | GPT yes+partly (95% CI) | Claude Opus 5.5 yes | Claude yes+partly |")
    print("|---|---:|---|---|---|---|")
    for s in ("kw_only", "kw_only_noabs", "text"):
        c = agg[s]; n = sum(c.values()); cc = aggc[s]; nc = sum(cc.values())
        y = wilson(c["yes"], n); yp = wilson(c["yes"] + c["partly"], n)
        print(f"| {s} | {n} | {pc(y)} | {pc(yp)} | {100*cc['yes']/nc:.0f}% | {100*(cc['yes']+cc['partly'])/nc:.0f}% |")
        S["summary"][f"B|{s}"] = {"n": n, "yes": y, "yes_partly": yp, "counts": dict(c), "claude_counts": dict(cc)}
    print("\nClaude (rows) vs GPT (cols) verdicts, unique works:")
    for a in ("yes", "partly", "no"):
        print(f"  claude {a:6}: " + "  ".join(f"gpt {b}={agree[(a,b)]}" for b in ("yes", "partly", "no")))
    S["B_agreement"] = {f"{a}->{b}": agree[(a, b)] for a in ("yes", "partly", "no") for b in ("yes", "partly", "no")}
print(f"\nErrors unresolved by type: {dict(errors)}; refusals: {len(refusals)}")
S["errors"] = dict(errors); S["refusals"] = refusals
json.dump(S, open(os.path.join(DATA, "summary.json"), "w"), indent=1)
