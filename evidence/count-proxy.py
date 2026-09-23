import json, sys, threading
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.request import urlopen, Request
from urllib.error import HTTPError

UP = sys.argv[1]
pairs, fails, lock = Counter(), Counter(), threading.Lock()

class H(BaseHTTPRequestHandler):
    def _pass(self, body):
        try:
            model = json.loads(body).get("model", "(none)") if body else "(empty)"
        except Exception:
            model = "(unparsed)"
        with lock:
            pairs[(self.command, self.path, model)] += 1
        req = Request(UP + self.path, data=body if body else None,
                      method=self.command,
                      headers={"Content-Type": "application/json"})
        try:
            with urlopen(req) as r:
                data, code = r.read(), r.status
        except HTTPError as e:
            data, code = e.read(), e.code
            with lock:
                fails[(self.command, self.path, code)] += 1
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)
    def do_POST(self):
        self._pass(self.rfile.read(int(self.headers.get("Content-Length", 0))))
    def do_GET(self):
        self._pass(b"")
    def log_message(self, *a): pass

srv = ThreadingHTTPServer(("127.0.0.1", int(sys.argv[2])), H)
try:
    srv.serve_forever()
except KeyboardInterrupt:
    pass
finally:
    print(json.dumps({
        "requests": [{"method": m, "path": p, "model": md, "n": n} for (m, p, md), n in pairs.items()],
        "failures": [{"method": m, "path": p, "status": s, "n": n} for (m, p, s), n in fails.items()],
    }, indent=1), flush=True)
