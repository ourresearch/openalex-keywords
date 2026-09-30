"""Synonym review tool for Jason (#1322, 2026-09-29): serves items.json and a one-keystroke review page; every decision is appended to decisions.jsonl
(last decision per item wins). Only paths under /<TOKEN>/ respond, so the public Funnel URL is unguessable. python3 server.py"""
import json, os, time, threading
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
HERE = os.path.dirname(os.path.abspath(__file__))
TOKEN = open(f'{HERE}/token.txt').read().strip()
LOCK = threading.Lock()
DEC = f'{HERE}/decisions.jsonl'
class H(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype='application/json'):
        b = body.encode() if isinstance(body, str) else body
        self.send_response(code); self.send_header('Content-Type', ctype); self.send_header('Cache-Control', 'no-store'); self.send_header('Content-Length', str(len(b))); self.end_headers(); self.wfile.write(b)
    def log_message(self, *a): pass
    def do_GET(self):
        p = self.path.split('?')[0]
        if p in (f'/{TOKEN}', f'/{TOKEN}/'): return self._send(200, open(f'{HERE}/index.html', 'rb').read(), 'text/html; charset=utf-8')
        if p == f'/{TOKEN}/items': return self._send(200, open(f'{HERE}/items.json', 'rb').read())
        if p == f'/{TOKEN}/decisions':
            d = {}
            if os.path.exists(DEC):
                for l in open(DEC):
                    r = json.loads(l); d[str(r['i'])] = r
            return self._send(200, json.dumps(d))
        self._send(404, '{}')
    def do_POST(self):
        if self.path != f'/{TOKEN}/decide': return self._send(404, '{}')
        r = json.loads(self.rfile.read(int(self.headers.get('Content-Length', 0))) or b'{}')
        if not isinstance(r.get('i'), int) or r.get('d') not in ('A', 'B', 'X', 'S', 'U'): return self._send(400, '{"ok":false}')
        r['t'] = time.strftime('%Y-%m-%dT%H:%M:%S')
        with LOCK, open(DEC, 'a') as f: f.write(json.dumps(r) + '\n')
        self._send(200, '{"ok":true}')
ThreadingHTTPServer(('127.0.0.1', 8765), H).serve_forever()
