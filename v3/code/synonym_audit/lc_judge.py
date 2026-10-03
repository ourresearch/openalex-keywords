"""Work-level: is this work about long COVID (the illness)? 40 works per source string merged into keywords/long-covid. Opus 5.5 medium.
Input lc_samples.jsonl (from lc_pull.py; carries work text, not released). -> lc_judged.jsonl, released without titles as
data/synonyms/audit_v31/long_covid_judged.jsonl. Prompt: prompts/19_long_covid_check.md.
    CLAUDE_API_KEY=... uv run --with anthropic python lc_judge.py"""
import json, os, threading
from concurrent.futures import ThreadPoolExecutor
import anthropic
client = anthropic.Anthropic(api_key=os.environ['CLAUDE_API_KEY'], max_retries=6)
SYSTEM = ("You judge whether a scholarly work is about a given subject, for a precision audit of a subject index. 'yes' if the subject is a main topic, "
          "'partly' if it is a clear secondary topic or substantial component, 'no' if absent, mentioned in passing, or a different sense of the words. "
          "Judge only from the record; if there is no abstract, judge from title and venue.")
SCHEMA = {"type": "object", "properties": {"verdict": {"type": "string", "enum": ["yes", "partly", "no"]}, "reason": {"type": "string"}}, "required": ["verdict", "reason"], "additionalProperties": False}
subj = "long COVID (post-COVID-19 condition: symptoms persisting after SARS-CoV-2 infection)"
rows = [json.loads(l) for l in open('lc_samples.jsonl')]
out = open('lc_judged.jsonl', 'w'); lock = threading.Lock()
def one(r):
    rec = f"Title: {r['title']}\nVenue: {r['venue']}\nType: {r['type']}; year: {r['year']}\nAbstract: {r['abstract'] or '(none available)'}"
    try:
        m = client.messages.create(model='claude-opus-5-5', max_tokens=8000, system=SYSTEM, messages=[{'role': 'user', 'content': f'Subject: "{subj}"\n\nWork:\n{rec}\n\nIs this work substantially about the subject?'}],
                                   extra_body={'output_config': {'effort': 'medium', 'format': {'type': 'json_schema', 'schema': SCHEMA}}})
        t = "".join(b.text for b in m.content if b.type == 'text'); v = json.loads(t) if t else {'verdict': None, 'reason': m.stop_reason}
    except Exception as e: v = {'verdict': None, 'reason': repr(e)[:200]}
    with lock: out.write(json.dumps({**{k: r[k] for k in ('src', 'work_id', 'title')}, 'no_abs': not r['abstract'], **v}, ensure_ascii=False) + '\n'); out.flush()
with ThreadPoolExecutor(12) as ex: list(ex.map(one, rows))
