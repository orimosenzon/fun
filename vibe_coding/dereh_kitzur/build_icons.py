#!/usr/bin/env python3
"""Build the app icons for the home screen: web/icon-512.png, icon-192.png
and apple-touch-icon.png (180).

One design, three sizes: the app's green, and a shortcut drawn as the glowing
gold line the map draws it in. Full-bleed and with the line inside the middle
80 %, so the same file serves as a plain icon and as a maskable one - Android
cuts its own circle or squircle out of it, iOS rounds the corners itself.

Rendered in Chromium like build_og.py, because the glow is an SVG filter and
the browser already draws it exactly as the map does.

    python3 build_icons.py
"""

import io
import pathlib

from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parent
WEB = ROOT / "web"
SIZES = {"icon-512.png": 512, "icon-192.png": 192, "apple-touch-icon.png": 180}

HTML = """<!doctype html>
<html><head><meta charset="utf-8"><style>
  * { margin: 0; padding: 0; }
  body { width: 512px; height: 512px; overflow: hidden; }
  svg { display: block; }
</style></head><body>
<svg width="512" height="512" viewBox="0 0 512 512" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="#2e7d32"/>
      <stop offset="1" stop-color="#0f3d14"/>
    </linearGradient>
    <filter id="glow" x="-40%" y="-40%" width="180%" height="180%">
      <feGaussianBlur stdDeviation="16"/>
    </filter>
    <filter id="soft" x="-40%" y="-40%" width="180%" height="180%">
      <feGaussianBlur stdDeviation="5"/>
    </filter>
  </defs>
  <rect width="512" height="512" fill="url(#bg)"/>
  <!-- the ground: a hint of a block and a street, so the line is a shortcut through something -->
  <g stroke="rgba(255,255,255,.13)" stroke-width="14" fill="none" stroke-linecap="round">
    <path d="M60,150 H452"/>
    <path d="M60,362 H452"/>
  </g>
  <!-- the shortcut, the way the map draws it: a wide gold halo, the line, a pale filament -->
  <g fill="none" stroke-linecap="round" stroke-linejoin="round">
    <path d="M118,388 C170,388 176,300 232,268 S330,232 356,190 S392,132 394,124" stroke="#f2a900" stroke-width="58" opacity=".55" filter="url(#glow)"/>
    <path d="M118,388 C170,388 176,300 232,268 S330,232 356,190 S392,132 394,124" stroke="#ffc233" stroke-width="30" opacity=".9" filter="url(#soft)"/>
    <path d="M118,388 C170,388 176,300 232,268 S330,232 356,190 S392,132 394,124" stroke="#ffe08a" stroke-width="18"/>
    <path d="M118,388 C170,388 176,300 232,268 S330,232 356,190 S392,132 394,124" stroke="#fffaea" stroke-width="6" opacity=".9"/>
  </g>
  <!-- where it starts and where it gets you -->
  <circle cx="118" cy="388" r="22" fill="#fff8dc" stroke="#f2a900" stroke-width="8"/>
  <circle cx="394" cy="124" r="22" fill="#fff8dc" stroke="#f2a900" stroke-width="8"/>
</svg>
</body></html>"""


def render():
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        b = p.chromium.launch()
        page = b.new_page(viewport={"width": 512, "height": 512}, device_scale_factor=2)
        page.set_content(HTML, wait_until="load")
        shot = page.screenshot(type="png", omit_background=False)
        b.close()
    return shot


def main():
    im = Image.open(io.BytesIO(render())).convert("RGB")
    for name, px in SIZES.items():
        out = im.resize((px, px), Image.LANCZOS)
        out.save(WEB / name, "PNG", optimize=True)
        print(f"  wrote web/{name}  {px}x{px}  {(WEB / name).stat().st_size / 1024:.0f}KB")


if __name__ == "__main__":
    main()
