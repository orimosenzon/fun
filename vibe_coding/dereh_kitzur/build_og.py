#!/usr/bin/env python3
"""Build the link-preview card that WhatsApp, Telegram and Facebook show.

Why a generated JPEG and not one of the trail photos: every image the app
carries is webp, and webp is exactly what the WhatsApp preview crawler will not
render. It fetches the file, fails to decode it, and falls back to showing the
bare domain - which looks identical to having no og:image at all, so the bug is
invisible unless you know to look for it.

The card is rendered in Chromium rather than drawn with Pillow because the text
is Hebrew: the browser already has the bidi algorithm and a real font stack, and
Pillow has neither.

    python3 build_og.py            # rebuild web/og.jpg from live trail data
    python3 build_og.py --photo X  # use web/img/X.webp as the background
"""

import argparse
import base64
import io
import json
import pathlib
import subprocess
import sys
import urllib.request

from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parent
WEB = ROOT / "web"
OUT = WEB / "og.jpg"

# Jacaranda in bloom over a path that leads somewhere, evening light, a dog and
# no faces. Chosen over the greener tunnel shots because the purple reads at
# thumbnail size, and a preview card is first seen at about 300px wide.
PHOTO = "72b4d97bc41f5e"

TRAILS = "https://raw.githubusercontent.com/orimosenzon/derech-kitzur-data/main/data/trails.json"

# WhatsApp is the tightest consumer: it wants a landscape image it can decode
# quickly, and drops the large preview for files that are too heavy.
SIZE = (1200, 630)
MAX_BYTES = 300 * 1024


def stats():
    """Segment count, straight from the data repo, so the card cannot go stale."""
    try:
        with urllib.request.urlopen(TRAILS, timeout=20) as r:
            data = json.load(r)
        return len(data.get("segments", []))
    except Exception as e:                                  # offline rebuild
        print(f"  ! could not read live trails ({e}), leaving the count out")
        return 0


def card_html(photo, segments):
    # A data URI and not a file:// path: the page is built with set_content, so
    # its origin is about:blank and Chromium refuses to fetch a local file from
    # there. It fails silently - a broken-image glyph in the corner and an
    # otherwise perfectly rendered card.
    raw = (WEB / "img" / f"{photo}.webp").read_bytes()
    src = "data:image/webp;base64," + base64.b64encode(raw).decode()
    count = f"{segments} קיצורי דרך שתושבים מיפו ברגל" if segments else \
            "קיצורי דרך שתושבים מיפו ברגל"
    return f"""<!doctype html>
<html lang="he" dir="rtl"><head><meta charset="utf-8">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Heebo:wght@400;500;800&display=swap" rel="stylesheet">
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ width: {SIZE[0]}px; height: {SIZE[1]}px; overflow: hidden;
         font-family: Heebo, 'Noto Sans Hebrew', 'DejaVu Sans', sans-serif;
         background: #0a1a0e; }}
  .card {{ position: relative; width: 100%; height: 100%; }}
  /* object-position keeps the blossoms and the path; a plain centre crop cuts
     the top of the tree and leaves a band of empty dirt at the bottom. */
  img {{ width: 100%; height: 100%; object-fit: cover; object-position: 50% 42%; }}
  .scrim {{ position: absolute; inset: 0;
            background: linear-gradient(to top,
              rgba(5,18,9,.94) 0%, rgba(5,18,9,.86) 22%,
              rgba(5,18,9,.45) 46%, rgba(5,18,9,.05) 70%, transparent 100%); }}
  .badge {{ position: absolute; top: 44px; right: 56px;
            padding: 10px 22px; border-radius: 999px;
            background: rgba(5,18,9,.62); backdrop-filter: blur(6px);
            border: 1px solid rgba(255,255,255,.22);
            color: #fff; font-size: 26px; font-weight: 500; letter-spacing: .4px; }}
  .text {{ position: absolute; right: 56px; bottom: 52px; left: 56px; }}
  h1 {{ color: #fff; font-size: 96px; font-weight: 800; line-height: 1;
        letter-spacing: -1px; text-shadow: 0 3px 24px rgba(0,0,0,.55); }}
  .rule {{ width: 108px; height: 7px; border-radius: 4px; margin: 22px 0 20px;
           background: #5cc46a; }}
  p {{ color: rgba(255,255,255,.94); font-size: 36px; font-weight: 400;
       text-shadow: 0 2px 14px rgba(0,0,0,.6); }}
</style></head><body>
  <div class="card">
    <img src="{src}" alt="">
    <div class="scrim"></div>
    <div class="badge">פרדס חנה-כרכור</div>
    <div class="text">
      <h1>דרך קיצור</h1>
      <div class="rule"></div>
      <p>{count}</p>
    </div>
  </div>
</body></html>"""


def render(html):
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        b = p.chromium.launch()
        page = b.new_page(viewport={"width": SIZE[0], "height": SIZE[1]},
                          device_scale_factor=2)
        page.set_content(html, wait_until="networkidle")
        # Webfonts land after networkidle often enough to be worth waiting on:
        # a card that falls back to DejaVu is not obviously wrong, just ugly.
        try:
            page.wait_for_function("document.fonts.ready.then(() => true)", timeout=8000)
        except Exception:
            print("  ! webfont wait timed out, rendering with what loaded")
        shot = page.screenshot(type="png")
        b.close()
    return shot


def save(png):
    """Down to 1200x630 and under the size WhatsApp will still show large."""
    im = Image.open(io.BytesIO(png)).convert("RGB").resize(SIZE, Image.LANCZOS)
    for q in (88, 84, 80, 74, 68):
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=q, optimize=True, progressive=False)
        if buf.tell() <= MAX_BYTES:
            OUT.write_bytes(buf.getvalue())
            return q, buf.tell()
    OUT.write_bytes(buf.getvalue())
    return q, buf.tell()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--photo", default=PHOTO, help="basename in web/img, without .webp")
    a = ap.parse_args()

    if not (WEB / "img" / f"{a.photo}.webp").exists():
        sys.exit(f"no such photo: web/img/{a.photo}.webp")

    n = stats()
    print(f"rendering og card from {a.photo}.webp ({n or '?'} segments)")
    q, size = save(render(card_html(a.photo, n)))
    print(f"  wrote {OUT.relative_to(ROOT)}  {SIZE[0]}x{SIZE[1]}  q{q}  {size/1024:.0f}KB")


if __name__ == "__main__":
    main()
