"""Fill titles and abstracts back into the released outside-judge sample (the repository ships ids and keyword lists only).
Reads data/eval/outside_judge/sample_a.jsonl, fetches each work from the OpenAlex API, writes sample_a_text.jsonl beside it
(abstract cut at 2,500 characters, as judged). Abstracts can change after 2026-09-30, so a few records may differ from what
the judge saw, and 4 of the 420 works no longer resolve (checked 2026-10-03). The output is git-ignored.
    OPENALEX_API_KEY=... python3 fetch_text.py"""
import json, os, time, urllib.parse, urllib.request

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "eval", "outside_judge")
KEY = os.environ.get("OPENALEX_API_KEY", "")


def get(ids):
    q = urllib.parse.urlencode({"filter": "openalex:" + "|".join(ids), "per-page": 100, "include_xpac": "true", "select": "id,display_name,abstract_inverted_index"})
    req = urllib.request.Request(f"https://api.openalex.org/works?{q}", headers={"Authorization": f"Bearer {KEY}"} if KEY else {})
    for i in range(5):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.load(r)["results"]
        except Exception as e:
            err = e
            time.sleep(2 + 3 * i)
    raise err


def abstract(inv):
    if not inv:
        return None
    return " ".join(w for _, w in sorted((i, w) for w, ii in inv.items() for i in ii))[:2500]


rows = [json.loads(l) for l in open(os.path.join(DATA, "sample_a.jsonl"))]
got = {}
for i in range(0, len(rows), 50):
    for w in get([r["work_id"] for r in rows[i:i + 50]]):
        got[w["id"].rsplit("/", 1)[-1]] = w
with open(os.path.join(DATA, "sample_a_text.jsonl"), "w") as f:
    for r in rows:
        w = got.get(r["work_id"], {})
        f.write(json.dumps({**r, "title": w.get("display_name"), "abstract": abstract(w.get("abstract_inverted_index"))}, ensure_ascii=False) + "\n")
print(f"{len(got)} of {len(rows)} works found")
