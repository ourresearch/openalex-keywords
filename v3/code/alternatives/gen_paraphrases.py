"""Paraphrase-query eval for keyword search, the model-written half (239 queries; the other 50 were written by hand). Sample shipped
keywords (head: n >= 20K; torso: 1K-20K; seed 1464), ask Opus 5.5 for the query a researcher might type when looking for that concept
WITHOUT reusing its wording (synonym, lay term, spelling variant, abbreviation or its expansion, reordered phrase). Gold = the source
keyword. Input vocab.jsonl {kid, name, n} (the shipped vocabulary with works counts). Output eval_gen.json; eval_all.json = the 50
hand-written queries + these 239.  uv run --with anthropic python gen_paraphrases.py"""
import json, random, os, concurrent.futures as cf, anthropic
random.seed(1464)
V = [json.loads(l) for l in open("vocab.jsonl")]
head = [r for r in V if r["n"] >= 20000]; torso = [r for r in V if 1000 <= r["n"] < 20000]
sample = [dict(r, stratum="head") for r in random.sample(head, 120)] + [dict(r, stratum="torso") for r in random.sample(torso, 120)]
client = anthropic.Anthropic(api_key=os.environ.get("CLAUDE_API_KEY") or os.environ.get("ANTHROPIC_API_KEY"))
PROMPT = """A researcher is searching a list of scholarly keywords (research concepts) for this one:

  "{name}"

Write the search query they might plausibly type if they think of the concept in DIFFERENT words: a synonym, a near-synonym at the same specificity, a lay term, a British/American spelling variant, an abbreviation or its expansion, or a reworded phrase. Rules: keep the same concept and the same specificity (not broader, not narrower); do not reuse any content word of the keyword unless no natural alternative exists (then change the other words); 1 to 6 words; lowercase unless it is an acronym or proper noun.
If no natural different wording exists (e.g. a proper name or a dataset), answer SKIP.
Answer with the query only, no quotes."""
def one(r):
    m = client.messages.create(model="claude-opus-5-5", max_tokens=3000, messages=[{"role": "user", "content": PROMPT.format(name=r["name"])}])
    q = next(b.text for b in m.content if b.type == "text").strip().strip('"')
    return {"query": q, "gold": [r["kid"]], "src": "gen_" + r["stratum"], "keyword": r["name"], "n": r["n"]}
with cf.ThreadPoolExecutor(16) as ex:
    out = [o for o in ex.map(one, sample) if o["query"] and o["query"].upper() != "SKIP"]
json.dump(out, open("eval_gen.json", "w"), indent=0)
print(len(out), "queries;", sum(o["src"] == "gen_head" for o in out), "head")
