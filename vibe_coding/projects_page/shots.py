#!/usr/bin/env python3
"""Screenshot every project in the "פרויקטים live" section.

Opens each app in headless Chromium, the way a visitor would, and saves a
640×400 WebP to projects_page/shots/<anchor>.webp. build.py shows the image if
it exists and falls back to the project's icon if not.

    python3 projects_page/shots.py              # only projects with no shot yet
    python3 projects_page/shots.py vedit 2048   # redo these
    python3 projects_page/shots.py --all        # redo everything

Local pages are served over http (some apps fetch their data files, which
file:// blocks). Needs playwright and Pillow.
"""
import argparse
import functools
import http.server
import io
import json
import threading

from PIL import Image
from playwright.sync_api import sync_playwright

import build

VIEWPORT = {"width": 1280, "height": 800}
OUT_SIZE = (640, 400)

# Per-project tweaks, keyed by anchor, applied in this order: "files" loads a
# file into an <input type=file>, "fill" types into a field, "click" presses
# buttons, "keys" plays keystrokes, then "wait" ms for canvases and WebGL to
# settle. The point is a shot of the app doing its thing, not its start screen.
ARROWS = ["ArrowUp", "ArrowLeft", "ArrowDown", "ArrowRight"]
TWEAKS = {
    "2048": {"keys": ARROWS * 25, "wait": 1200},
    "letras": {"fill": [("#searchInput", "Despacito")], "keys": ["Enter"], "wait": 6000},
    "fable": {"keys": ["Enter"], "wait": 4000},
    "opus5": {"click": ["#startBtn"], "wait": 4000},
    "dereh_kitzur": {"click": ["#welcome-sheet .sheet-x"], "wait": 4000},
    "animate": {"files": [("#loadMovieInput", "animate/a1.clp")], "wait": 1500},
    "manhattan_project": {"click": ["text=Edge Ring"], "wait": 2500},
}


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


def serve(root):
    handler = functools.partial(QuietHandler, directory=str(root))
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


def capture(page, url, tweak):
    page.goto(url, wait_until="load", timeout=45000)
    try:
        page.wait_for_load_state("networkidle", timeout=8000)
    except Exception:
        pass  # pages that poll or stream never go idle; the fixed wait covers them
    for sel, path in tweak.get("files", []):
        page.set_input_files(sel, str(build.ROOT / path))
        page.wait_for_timeout(800)
    for sel, text in tweak.get("fill", []):
        page.fill(sel, text)
    for sel in tweak.get("click", []):
        page.click(sel, timeout=5000)
        page.wait_for_timeout(400)
    for key in tweak.get("keys", []):
        page.keyboard.press(key)
        page.wait_for_timeout(150)
    page.wait_for_timeout(tweak.get("wait", 2500))
    png = page.screenshot()
    img = Image.open(io.BytesIO(png)).convert("RGB").resize(OUT_SIZE, Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, "WEBP", quality=80, method=6)
    return buf.getvalue()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("only", nargs="*", help="anchors to (re)capture")
    ap.add_argument("--all", action="store_true", help="recapture every project")
    args = ap.parse_args()

    data = json.loads(build.DATA.read_text(encoding="utf-8"))
    build.assign_anchors(data)
    todo = []
    for _, p, link in build.try_projects(data):
        a = p["_anchor"]
        out = build.SHOTS / f"{a}.webp"
        if args.only and a not in args.only:
            continue
        if not (args.all or args.only) and out.exists():
            continue
        todo.append((a, link["href"], out))

    if not todo:
        print("כל צילומי המסך קיימים. --all כדי לצלם הכל מחדש.")
        return 0

    build.SHOTS.mkdir(exist_ok=True)
    httpd = serve(build.ROOT)
    base = f"http://127.0.0.1:{httpd.server_address[1]}/"
    failed = []
    with sync_playwright() as pw:
        browser = pw.chromium.launch(args=["--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
        for a, href, out in todo:
            url = href if href.startswith("http") else base + href
            page = browser.new_page(viewport=VIEWPORT, locale="he-IL")
            try:
                out.write_bytes(capture(page, url, TWEAKS.get(a, {})))
                print(f"  ✓ {a}  ({out.stat().st_size // 1024}KB)")
            except Exception as e:
                failed.append(a)
                print(f"  ✗ {a}: {str(e).splitlines()[0]}")
            page.close()
        browser.close()
    httpd.shutdown()
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
