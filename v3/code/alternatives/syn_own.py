"""One owner per synonym, step 1 (no model calls). Inputs: syn_full.jsonl (LLM synonyms, prompt_version syn_v1_sol_wd), the
synonym map's merged wordings (SYNMAP, default synmap10.jsonl, {kid, canon}: each merged keyword id maps to exactly one heading),
vocab10.jsonl (names, works counts).
Normalised form = NFKC, lower case, hyphens/underscores/slashes -> space, apostrophes and periods dropped, spaces collapsed.
- drop a synonym whose normalised form equals its own keyword's name;
- drop a synonym that equals ANOTHER keyword's name (that keyword already matches it by name) and log the pair to name_clash.jsonl
  (a duplicate-keyword candidate for the synonym map when the two keywords really are the same concept);
- a normalised synonym claimed by more than one keyword -> collisions.jsonl for the owner judge (syn_own_judge.py);
- everything else -> owned by its keyword.
Writes syn_owned.jsonl {kid, synonyms: [{text, source}]} (source: llm | synonym_map) and prints counts.  python3 syn_own.py
Low-memory design: keyword names and synonym claims go through one on-disk `sort` keyed by normalised form instead of dicts
(7M synonyms ~ 3 GB in dicts; this peaks ~ 0.4 GB). Stops if its own RSS passes 1 GB."""
import collections, json, os, re, resource, subprocess, sys, unicodedata
SYNMAP = os.environ.get("SYNMAP", "synmap10.jsonl")
TMP = "syn_own_tmp"; os.makedirs(TMP, exist_ok=True)
def norm(s):
    s = unicodedata.normalize("NFKC", s).lower()
    s = re.sub(r"[-_/‐-―]", " ", s); s = re.sub(r"['’.]", "", s)
    return re.sub(r"\s+", " ", s).strip()
clean = lambda t: re.sub(r"[\t\r\n]+", " ", t)
def guard():
    if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9 > 1.0: sys.exit("RSS over 1 GB: stopping (memory guard)")
def sort(src, dst, keys):
    subprocess.run(["sort", "-t", "\t", *keys, "-S", "200M", "-T", TMP, "-o", dst, src], check=True, env={**os.environ, "LC_ALL": "C"})
state = {}                                              # kid -> 1 once its LLM row is taken (first row per kid wins)
with open(f"{TMP}/claims.tsv", "w") as f:              # norm, seq, kid, text, source; seq 0 = the keyword's own name
    for l in open("vocab10.jsonl"):
        r = json.loads(l); state[sys.intern(r["kid"])] = 0
        f.write(f"{norm(r['name'])}\t0\t{r['kid']}\t\tNAME\n")
    seq = 0
    for i, l in enumerate(open("syn_full.jsonl")):
        r = json.loads(l); k = r["kid"]
        if r.get("prompt_version") != "syn_v1_sol_wd" or state.get(k, 1) != 0: continue
        state[k] = 1
        for t in r["synonyms"]:
            t = clean(t).strip()
            if t and norm(t): seq += 1; f.write(f"{norm(t)}\t{seq}\t{k}\t{t}\tllm\n")
        if i % 200000 == 0: guard()
    for l in open(SYNMAP):
        s = json.loads(l)
        if s["canon"] in state and s["kid"] not in state:
            t = s["kid"].replace("-", " "); seq += 1; f.write(f"{norm(t)}\t{seq}\t{s['canon']}\t{t}\tsynonym_map\n")
n_seen = sum(state.values()); del state
sort(f"{TMP}/claims.tsv", f"{TMP}/claims.sorted.tsv", ["-k1,1", "-k2,2n"])
c = collections.Counter()
def groups():
    cur, names, who = None, set(), {}
    for l in open(f"{TMP}/claims.sorted.tsv"):
        n, sq, k, t, src = l.rstrip("\n").split("\t")
        if n != cur:
            if cur is not None: yield cur, names, who
            cur, names, who = n, set(), {}
        if src == "NAME": names.add(k)
        else: who.setdefault(k, (t, src, int(sq)))
    if cur is not None: yield cur, names, who
with open("name_clash.jsonl", "w") as nc, open(f"{TMP}/collisions.partial.jsonl", "w") as co, open(f"{TMP}/owned.tsv", "w") as ow:
    for n, names, who in groups():
        if not who: continue
        own_name = {k for k in who if k in names}
        c["drop_own_name"] += len(own_name)
        who = {k: v for k, v in who.items() if k not in own_name}
        if not who: continue
        others = names - set(who)
        if others:
            for k, (t, src, _) in who.items():
                nc.write(json.dumps({"kid": k, "synonym": t, "source": src, "is_name_of": sorted(others)}, ensure_ascii=False) + "\n"); c["drop_other_name"] += 1
            continue
        if len(who) > 1:
            co.write(json.dumps({"norm": n, "claims": [{"kid": k, "text": t, "source": src} for k, (t, src, _) in who.items()]}, ensure_ascii=False) + "\n")
            c["collision_groups"] += 1; c["collision_claims"] += len(who); continue
        (k, (t, src, sq)), = who.items(); ow.write(f"{k}\t{sq}\t{t}\t{src}\n"); c["owned_" + src] += 1
guard()
need = set()                                            # names + works only for the collision keywords
for l in open(f"{TMP}/collisions.partial.jsonl"): need.update(x["kid"] for x in json.loads(l)["claims"])
info = {}
for l in open("vocab10.jsonl"):
    r = json.loads(l)
    if r["kid"] in need: info[r["kid"]] = (r["name"], r["n"])
with open("collisions.jsonl", "w") as co:
    for l in open(f"{TMP}/collisions.partial.jsonl"):
        g = json.loads(l)
        for x in g["claims"]: x["name"], x["works"] = info[x["kid"]]
        co.write(json.dumps(g, ensure_ascii=False) + "\n")
sort(f"{TMP}/owned.tsv", f"{TMP}/owned.sorted.tsv", ["-k1,1", "-k2,2n"])    # back to each keyword's original synonym order
n_kw = 0
with open("syn_owned.jsonl", "w") as f:
    cur, syns = None, []
    for l in open(f"{TMP}/owned.sorted.tsv"):
        k, _, t, src = l.rstrip("\n").split("\t")
        if k != cur:
            if cur: f.write(json.dumps({"kid": cur, "synonyms": syns}, ensure_ascii=False) + "\n"); n_kw += 1
            cur, syns = k, []
        syns.append({"text": t, "source": src})
    if cur: f.write(json.dumps({"kid": cur, "synonyms": syns}, ensure_ascii=False) + "\n"); n_kw += 1
c["keywords_with_llm_output"] = n_seen; c["keywords_with_any_owned_synonym"] = n_kw
for k, v in sorted(c.items()): print(f"{k}: {v:,}")
print(f"peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9:.2f} GB")
