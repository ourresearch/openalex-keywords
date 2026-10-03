"""Before/after test of keyword synonyms on autocomplete. Simulates /autocomplete/keywords: every typed word must match a word of the
keyword's name or of one of its display_name_alternatives, the last word as a prefix. SQLite FTS5 over syn_final.jsonl (+ names from
vocab10.jsonl), built once to syn_fts.sqlite in the working directory. Ranking, set by RANK:
  (unset) level: exact match first, then works_count; name and alternative matches rank equally;
  tiered: exact name, exact alternative, name matches, then alternative-only matches; works_count within each;
  penalty (shipped, with PENALTY=100): exact name, exact alternative, then log(1+works) where an alternative-only match pays
          log(PENALTY), i.e. it needs PENALTY x the works of a name match to outrank it.
 1. Paraphrase eval (eval_all.json, 289 queries; the hand-written 50 reported separately, since the other 239 were written by Opus):
    gold keyword in the top 5 at 40/60/80/100% of the query typed, names only vs names + alternatives.
 2. Real queries typed into the website's keyword box (real_queries.json: [{q, final}], Jul-Sep 2026; NOT released, it is user data):
    the 'final' strings of 1-5 words, no DOIs, 400 sampled (seed 1464); share with zero results, and a blind Opus 5.5 judge of the
    top 3 (SAME/CLOSE/WRONG: is it the keyword the person was looking for?) before vs after. Per-query output goes to
    syn_eval_real*.json (also not released); only the printed aggregates are published.
  RANK=penalty PENALTY=100 uv run --with anthropic python syn_eval.py"""
import json, os, random, re, sqlite3, concurrent.futures as cf
D = os.getcwd(); DB = os.path.join(D, "syn_fts.sqlite")
if not os.path.exists(DB):                       # built on disk (no 1.6M-keyword dict in memory)
    con = sqlite3.connect(DB)
    con.execute("CREATE VIRTUAL TABLE kw USING fts5(kid UNINDEXED, name, alts, n UNINDEXED, tokenize='unicode61')")
    con.execute("CREATE TEMP TABLE a (kid TEXT PRIMARY KEY, alts TEXT)")
    b = []
    for l in open(os.path.join(D, "syn_final.jsonl")):
        r = json.loads(l); b.append((r["kid"], " | ".join(s["text"] for s in r["synonyms"])))
        if len(b) >= 20000: con.executemany("INSERT OR REPLACE INTO a VALUES (?,?)", b); b = []
    con.executemany("INSERT OR REPLACE INTO a VALUES (?,?)", b); b = []
    con.execute("CREATE TEMP TABLE v (kid TEXT, name TEXT, n INTEGER)")
    for l in open(os.path.join(D, "vocab10.jsonl")):
        r = json.loads(l); b.append((r["kid"], r["name"], r["n"]))
        if len(b) >= 20000: con.executemany("INSERT INTO v VALUES (?,?,?)", b); b = []
    con.executemany("INSERT INTO v VALUES (?,?,?)", b); b = []
    con.execute("INSERT INTO kw SELECT v.kid, v.name, COALESCE(a.alts, ''), v.n FROM v LEFT JOIN a ON a.kid = v.kid")
    con.commit(); con.close()
con = sqlite3.connect(DB, check_same_thread=False)
def ac(q, alts=True, k=5):
    toks = re.findall(r"\w+", q.lower())
    if not toks: return []
    m = " AND ".join([f'"{t}"' for t in toks[:-1]] + [f'"{toks[-1]}"*']); cols = "{name alts}" if alts else "{name}"
    try:
        rows = con.execute(f"SELECT kid, name, alts, CAST(n AS INTEGER) FROM kw WHERE kw MATCH ? ORDER BY 4 DESC LIMIT 200", (f"{cols} : ({m})",)).fetchall()
        if alts and os.environ.get("RANK") in ("tiered", "penalty"):   # name matches get their own pool, so popular alternative matches can't crowd them out
            seen = {r[0] for r in rows}
            rows += [r for r in con.execute(f"SELECT kid, name, alts, CAST(n AS INTEGER) FROM kw WHERE kw MATCH ? ORDER BY 4 DESC LIMIT 200", (f"{{name}} : ({m})",)).fetchall() if r[0] not in seen]
    except sqlite3.OperationalError:
        return []
    ql = q.strip().lower()
    exact = lambda r: ql == r[1].lower() or (alts and ql in [a.strip().lower() for a in r[2].split("|")])
    if os.environ.get("RANK") == "tiered":       # exact name, exact alternative, then name matches, then alternative-only matches; works within each
        def name_hit(r):
            nt = re.findall(r"\w+", r[1].lower())
            return all(t in nt for t in toks[:-1]) and any(x.startswith(toks[-1]) for x in nt)
        return [(r[0], r[1]) for r in sorted(rows, key=lambda r: (ql != r[1].lower(), not exact(r), not name_hit(r), -r[3]))[:k]]
    if os.environ.get("RANK") == "penalty":     # like the ES tiers, but an alternative-only match pays ln(PENALTY) on the log1p(works) score
        import math
        def name_hit(r):
            nt = re.findall(r"\w+", r[1].lower())
            return all(t in nt for t in toks[:-1]) and any(x.startswith(toks[-1]) for x in nt)
        P = math.log(float(os.environ.get("PENALTY", "100")))
        return [(r[0], r[1]) for r in sorted(rows, key=lambda r: (ql != r[1].lower(), not exact(r), -(math.log1p(r[3]) - (0 if name_hit(r) else P))))[:k]]
    return [(r[0], r[1]) for r in sorted(rows, key=lambda r: (not exact(r), -r[3]))[:k]]
E = json.load(open(os.path.join(D, "eval_all.json")))
for name, ev in [("all 289", E), ("hand-written 50", [e for e in E if e["src"] == "hand"]), ("Opus-written 239", [e for e in E if e["src"] != "hand"])]:
    print(f"\n### paraphrase eval, {name}: gold keyword in top 5\n| typed | names only | names + alternatives |\n|---|---|---|")
    for f in (0.4, 0.6, 0.8, 1.0):
        a = b = 0
        for e in ev:
            p = e["query"][: max(3, round(len(e["query"]) * f))]; g = set(e["gold"])
            a += bool(g & {x[0] for x in ac(p, False)}); b += bool(g & {x[0] for x in ac(p, True)})
        print(f"| {int(f * 100)}% | {a / len(ev):.2f} | {b / len(ev):.2f} |")
R = [x for x in json.load(open(os.path.join(D, "real_queries.json"))) if x["final"] and 1 <= len(x["q"].split()) <= 5 and not re.search(r"10\.\d{4,}/", x["q"])]
random.seed(1464); R = random.sample(R, min(400, len(R)))
before = {x["q"]: ac(x["q"], False, 3) for x in R}; after = {x["q"]: ac(x["q"], True, 3) for x in R}
print(f"\n### real queries ({len(R)} sampled final strings of 1-5 words)\nzero results: names only {sum(not v for v in before.values()) / len(R):.0%}, names + alternatives {sum(not v for v in after.values()) / len(R):.0%}")
import anthropic
cl = anthropic.Anthropic(api_key=os.environ.get("CLAUDE_API_KEY") or os.environ.get("ANTHROPIC_API_KEY"))
PROMPT = """Someone typed this into the keyword search box of OpenAlex, an index of scholarly works:

  "{q}"

For each suggested keyword, label it: SAME = the concept they were looking for; CLOSE = related and useful but not quite it; WRONG = not useful. The text may be cut off mid-word or in another language; judge what they most plausibly meant.

{c}

Answer as JSON only: {{"labels": {{"<keyword>": "SAME|CLOSE|WRONG"}}}}"""
def judge(q):
    cands = sorted({n for _, n in before[q] + after[q]})
    if not cands: return q, {}
    for _ in range(3):
        try:
            m = cl.messages.create(model="claude-opus-5-5", max_tokens=6000, messages=[{"role": "user", "content": PROMPT.format(q=q, c="\n".join("- " + c for c in cands))}])
            t = next(b.text for b in m.content if b.type == "text"); return q, json.loads(t[t.find("{"): t.rfind("}") + 1])["labels"]
        except Exception:
            pass
    return q, None
with cf.ThreadPoolExecutor(12) as ex: L = dict(ex.map(judge, [x["q"] for x in R]))
json.dump({"before": before, "after": after, "labels": L}, open(os.path.join(D, "syn_eval_real" + ("_" + os.environ["RANK"] + os.environ.get("PENALTY", "") if os.environ.get("RANK") else "") + ".json"), "w"))
ok = [q for q in L if L[q] is not None]
for nm, S in [("names only", before), ("names + alternatives", after)]:
    t1 = sum(1 for q in ok if S[q] and L[q].get(S[q][0][1]) == "SAME") / len(ok)
    t1u = sum(1 for q in ok if S[q] and L[q].get(S[q][0][1]) in ("SAME", "CLOSE")) / len(ok)
    any3 = sum(1 for q in ok if any(L[q].get(n) == "SAME" for _, n in S[q])) / len(ok)
    print(f"{nm}: top-1 what they wanted {t1:.0%}, top-1 useful {t1u:.0%}, wanted keyword in top 3 {any3:.0%}  (n={len(ok)})")
