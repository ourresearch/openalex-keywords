"""OpenRouter chat-completions call with a strict JSON schema -> (parsed json or None, cost $, status, served model).
Used by the bake-off scripts (syn_gen.py, syn_judge.py). Needs OPENROUTER_API_KEY. No max_tokens is sent (provider default)."""
import os, json, httpx, asyncio
_client = None
def _get_client():
    # one shared client: a fresh AsyncClient per call holds its own TLS context and pool (≈ MBs each at high concurrency)
    global _client
    if _client is None:
        _client = httpx.AsyncClient(timeout=300, limits=httpx.Limits(max_connections=200, max_keepalive_connections=200))
    return _client
async def or_json(model, system, user, schema, reasoning='low', timeout=600):
    body = {'model': model, 'messages': [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}],
            'response_format': {'type': 'json_schema', 'json_schema': {'name': 'out', 'strict': True, 'schema': schema}}, 'reasoning': {'effort': reasoning}, 'usage': {'include': True}}
    c = _get_client()
    for t in range(6):
        try:
            r = await c.post('https://openrouter.ai/api/v1/chat/completions', headers={'Authorization': 'Bearer ' + os.environ['OPENROUTER_API_KEY']}, json=body, timeout=timeout)
        except httpx.TransportError:
            await asyncio.sleep(2 ** t); continue
        if r.status_code in (429, 500, 502, 503, 504): await asyncio.sleep(2 ** t); continue
        d = r.json()
        if 'choices' not in d: return None, 0, f"err {r.status_code} {str(d)[:200]}", model
        txt = d['choices'][0]['message'].get('content') or ''
        try: return json.loads(txt), d.get('usage', {}).get('cost', 0) or 0, 'ok', d.get('model')
        except Exception: return None, d.get('usage', {}).get('cost', 0) or 0, 'bad_json', d.get('model')
    return None, 0, 'retries', model
