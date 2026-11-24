#!/usr/bin/env python3
"""צילום דף מהוויקי החי, במחשב ובטלפון.
  python3 tpl_gfx/shoot.py <כותרת הדף> <תיקיית פלט> [desk|phone]
"""
import sys, os, urllib.parse
from playwright.sync_api import sync_playwright

title, out = sys.argv[1], sys.argv[2]
which = sys.argv[3:] or ["desk", "phone"]
VPS = {"desk": {"width": 1280, "height": 900}, "phone": {"width": 390, "height": 800}}
os.makedirs(out, exist_ok=True)
with sync_playwright() as p:
    for w in which:
        br = p.chromium.launch(args=["--no-zygote", "--single-process", "--disable-gpu"], timeout=90000)
        pg = br.new_page(viewport=VPS[w], device_scale_factor=1)
        pg.goto("https://pardespedia.info/index.php?title=" + urllib.parse.quote(title.replace(" ", "_")), wait_until="networkidle", timeout=90000)
        sw = pg.evaluate("document.documentElement.scrollWidth")
        print(w, "scrollWidth", sw, "viewport", VPS[w]["width"], "OVERFLOW" if sw > VPS[w]["width"] + 1 else "ok")
        fn = os.path.join(out, f"{w}.png")
        pg.locator(".mw-parser-output").first.screenshot(path=fn)
        print(fn)
        br.close()
