"""Local server that tells the browser never to cache, so an edit is always what you see.
   python3 tools/serve.py [port]     (default 8791)
(The plain `python3 -m http.server` lets Chrome keep old copies of some modules, and a page made
of old and new files breaks in confusing ways.)"""
import http.server, pathlib, sys, functools
ROOT = pathlib.Path(__file__).resolve().parent.parent
class H(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Cache-Control', 'no-store, max-age=0')
        super().end_headers()
    def log_message(self, *a): pass
port = int(sys.argv[1]) if len(sys.argv) > 1 else 8791
http.server.ThreadingHTTPServer(('127.0.0.1', port), functools.partial(H, directory=str(ROOT))).serve_forever()
