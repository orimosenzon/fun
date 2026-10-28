"""טעינה של סרטונים "אמיתיים": ארוכים, 4K, HEVC. מדפיס את יומן האבחון של האפליקציה.

הרצה:  python3 test_load_real.py קובץ [קובץ ...]
עובר אם הסרטון נטען, התמונות הממוזערות נבנו, ואין קפיאה ארוכה או שגיאה ביומן.
"""
import functools, http.server, re, sys, threading, time
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
h = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(ROOT))
h.log_message = lambda *a: None
srv = http.server.ThreadingHTTPServer(("127.0.0.1", 8773), h)
threading.Thread(target=srv.serve_forever, daemon=True).start()

bad = []
with sync_playwright() as p:
    b = p.chromium.launch(channel="chrome", headless=False, args=["--autoplay-policy=no-user-gesture-required"])
    for f in sys.argv[1:]:
        pg = b.new_page(viewport={"width": 1500, "height": 900})
        pg.goto("http://127.0.0.1:8773/index.html")
        pg.wait_for_function("document.body.classList.contains('ready')")
        t0 = time.time()
        pg.set_input_files("#fileInput", f)
        try:
            pg.wait_for_function("window.__clipper.dur > 0", timeout=60000, polling=500)
            pg.wait_for_function("/thumbs done|thumbs: video did not load/.test(C.log.text())", timeout=180000, polling=1000)
            pg.wait_for_timeout(3000)
        except Exception as e:
            print("  ✗ timeout:", str(e).splitlines()[0])
        log = pg.evaluate("C.log.text()")
        print(f"\n===== {Path(f).name} ({time.time() - t0:.0f}s) =====")
        print("\n".join(l for l in log.splitlines()[1:] if " start {" not in l))
        if re.search(r" ERROR |freeze \{\"ms\":[3-9]\d{3}|freeze \{\"ms\":\d{5}", log) or "thumbs done" not in log:
            bad.append(Path(f).name)
        pg.close()
    b.close()
srv.shutdown()
print("\nנכשל: " + ", ".join(bad) if bad else "\nהכול עבר")
sys.exit(1 if bad else 0)
