"""#1476 rung-3 check: a second judge from another family (GPT-6 Astra via OpenRouter, reasoning medium) with a different prompt (each
sense described in a sentence, venue included, returns the sense label text) on 500 random works from data/gold_works.jsonl (hash order).
-> data/astra_labels.jsonl (labels made 2026-09-30 came from api.openai.com directly; same model)
uv run --with httpx python astra_senses.py [--n 500] [--out data/astra_labels.jsonl]"""
import argparse, asyncio, hashlib, json, os, httpx
ap = argparse.ArgumentParser(); ap.add_argument('--n', type=int, default=500); ap.add_argument('--out', default='data/astra_labels.jsonl'); a = ap.parse_args()
J = {r['kid'].rsplit('/', 1)[-1]: r for r in map(json.loads, open('data/homonym_judged.jsonl'))}
W = sorted(map(json.loads, open('data/gold_works.jsonl')), key=lambda r: hashlib.md5((r['kid'] + str(r['work_id'])).encode()).hexdigest())[:a.n]
key = os.environ['OPENROUTER_API_KEY']
async def one(cl, sem, r, out):
    h = r['kid'].rsplit('/', 1)[-1]; S = sorted(J[h]['senses'], key=lambda s: -s['share'])
    labels = [s['label'] for s in S] + ['none']
    desc = "\n".join(f"- \"{s['label']}\": the sense used in works from {', '.join(s['fields'][:4]) or 'various fields'}" for s in S)
    text = (f"The keyword \"{h.replace('-', ' ')}\" has these distinct meanings in scholarly literature:\n{desc}\n- \"none\": the work uses none of these "
            f"meanings or is not about the keyword.\n\nRead the work and say which meaning it uses. Base it on what the work is about, not on its field.\n\n"
            f"Title: {r['title']}\nVenue: {r.get('venue') or 'unknown'}\nAbstract: {r.get('abstract') or '(none)'}")
    schema = {"type": "object", "properties": {"meaning": {"type": "string", "enum": labels}}, "required": ["meaning"], "additionalProperties": False}
    body = {"model": "openai/gpt-6-astra", "messages": [{"role": "user", "content": text}], "reasoning": {"effort": "medium"}, "max_tokens": 16000,
            "response_format": {"type": "json_schema", "json_schema": {"name": "sense", "schema": schema, "strict": True}}}
    async with sem:
        for att in range(4):
            try:
                resp = await cl.post("https://openrouter.ai/api/v1/chat/completions", json=body, headers={"Authorization": f"Bearer {key}"}, timeout=600)
                d = resp.json()
                t = d['choices'][0]['message'].get('content')
                m = json.loads(t)['meaning'] if t else None
                out.write(json.dumps({'kid': r['kid'], 'work_id': r['work_id'], 'choice': labels.index(m) if m in labels else None, 'model': d.get('model')}) + '\n'); out.flush()
                return
            except Exception: await asyncio.sleep(5 * (att + 1))
async def main():
    out = open(a.out, 'w'); sem = asyncio.Semaphore(16)
    async with httpx.AsyncClient() as cl: await asyncio.gather(*(one(cl, sem, r, out) for r in W))
asyncio.run(main()); print('done')
