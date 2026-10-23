#!/usr/bin/env python3
"""Screenshot every project in the "פרויקטים live" section.

Opens each app in headless Chromium, the way a visitor would, and saves two
WebPs from one 1440×900 capture: a 640×400 thumbnail for the gallery at
projects_page/shots/<anchor>.webp, and the full size for the stage at the top
of the page at projects_page/shots/big/<anchor>.webp. build.py shows the images
if they exist and falls back to the project's icon if not.

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

GPU_ARGS = ["--use-angle=vulkan", "--enable-features=Vulkan", "--ignore-gpu-blocklist", "--enable-gpu"]
VIEWPORT = {"width": 1440, "height": 900}
OUT_SIZE = (640, 400)

# Per-project tweaks, keyed by anchor, applied in this order: "settle" ms for a
# slow start screen, "files" loads a file into an <input type=file>, "fill"
# types into a field, "click" presses buttons (a missing one is skipped),
# "keys" plays keystrokes, then "wait" ms for canvases and WebGL to settle.
# "gpu" runs the page in Chrome on the real graphics card, for scenes too heavy
# for the software renderer. "frame" scrolls an element into view and crops the
# shot to it (padded out to 16:10), shot at "dpr" device pixels so the crop
# stays sharp. "hash" is added to the URL, to open the app on a given state.
# The point is a shot of the app doing its thing, not its start screen.
ARROWS = ["ArrowUp", "ArrowLeft", "ArrowDown", "ArrowRight"]
TWEAKS = {
    "2048": {"keys": ARROWS * 25, "wait": 1200},
    "letras": {"fill": [("#searchInput", "Despacito")], "keys": ["Enter"], "wait": 6000},
    "fable": {"keys": ["Enter"], "wait": 4000},
    "opus5": {"click": ["#startBtn"], "wait": 4000},
    "dereh_kitzur": {"click": ["#welcome-sheet .sheet-x", "text=אחר כך"], "wait": 4000},
    "animate": {"files": [("#loadMovieInput", "animate/a1.clp")], "wait": 1500},
    "manhattan_project": {"click": ["text=Edge Ring"], "wait": 2500},
    "orbit": {"wait": 15000},
    "jetbike_game": {"gpu": True, "settle": 12000, "keys": ["Enter"], "wait": 8000},
    "optimus": {"click": ["#go"], "wait": 6000},
    "roadster": {"click": ["#go"], "wait": 5000},
    "lagrange": {"frame": "#lab-mtn", "dpr": 2, "wait": 3000},
    "hanadiv": {"hash": "#d=30&t=11:30", "wait": 9000},
    "phonebot": {"click": ["[data-mode=map]"], "frame": ".simgrid", "wait": 12000},
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
    page.wait_for_timeout(tweak.get("settle", 0))
    for sel, path in tweak.get("files", []):
        page.set_input_files(sel, str(build.ROOT / path))
        page.wait_for_timeout(800)
    for sel, text in tweak.get("fill", []):
        page.fill(sel, text)
    for sel in tweak.get("click", []):
        try:
            page.click(sel, timeout=5000)
        except Exception:
            print(f"    (no {sel} to click, shooting as is)")
        page.wait_for_timeout(400)
    for key in tweak.get("keys", []):
        page.keyboard.press(key)
        page.wait_for_timeout(150)
    if "frame" in tweak:
        page.locator(tweak["frame"]).evaluate("e => e.scrollIntoView({block: 'center', behavior: 'instant'})")
    page.wait_for_timeout(tweak.get("wait", 2500))
    clip = frame_clip(page.locator(tweak["frame"]).bounding_box()) if "frame" in tweak else None
    big = Image.open(io.BytesIO(page.screenshot(clip=clip, timeout=90000))).convert("RGB")
    big = big.resize((VIEWPORT["width"], VIEWPORT["height"]), Image.LANCZOS)
    return webp(big.resize(OUT_SIZE, Image.LANCZOS), 80), webp(big, 74)


def frame_clip(box, pad=16):
    """The element's box plus a margin, widened or heightened to the viewport's 16:10."""
    w, h = box["width"] + 2 * pad, box["height"] + 2 * pad
    ratio = VIEWPORT["width"] / VIEWPORT["height"]
    w, h = max(w, h * ratio), max(h, w / ratio)
    cx, cy = box["x"] + box["width"] / 2, box["y"] + box["height"] / 2
    return {"x": cx - w / 2, "y": cy - h / 2, "width": w, "height": h}


def webp(img, quality):
    buf = io.BytesIO()
    img.save(buf, "WEBP", quality=quality, method=6)
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
        if not (args.all or args.only) and out.exists() and (build.SHOTS_BIG / out.name).exists():
            continue
        todo.append((a, link["href"], out))

    if not todo:
        print("כל צילומי המסך קיימים. --all כדי לצלם הכל מחדש.")
        return 0

    build.SHOTS_BIG.mkdir(parents=True, exist_ok=True)
    httpd = serve(build.ROOT)
    base = f"http://127.0.0.1:{httpd.server_address[1]}/"
    failed = []
    with sync_playwright() as pw:
        browser = pw.chromium.launch(args=["--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
        gpu = None
        for a, href, out in todo:
            tweak = TWEAKS.get(a, {})
            url = (href if href.startswith("http") else base + href) + tweak.get("hash", "")
            opts = {"viewport": VIEWPORT, "locale": "he-IL", "device_scale_factor": tweak.get("dpr", 1)}
            if tweak.get("gpu"):
                gpu = gpu or pw.chromium.launch(channel="chrome", args=GPU_ARGS)
                page = gpu.new_page(**opts)
            else:
                page = browser.new_page(**opts)
            try:
                small, big = capture(page, url, tweak)
                out.write_bytes(small)
                (build.SHOTS_BIG / out.name).write_bytes(big)
                print(f"  ✓ {a}  ({len(small) // 1024}KB, big {len(big) // 1024}KB)")
            except Exception as e:
                failed.append(a)
                print(f"  ✗ {a}: {str(e).splitlines()[0]}")
            page.close()
        browser.close()
        if gpu:
            gpu.close()
    httpd.shutdown()
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
