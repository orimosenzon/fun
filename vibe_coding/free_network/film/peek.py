"""Stills from the film at chosen seconds, to look at while working on it.
  python3 film/peek.py 3 12.5 40      writes film/out/peek_<seconds>.jpg
(a static server on the project root is needed: python3 -m http.server 8812)"""
import sys
from pathlib import Path
from playwright.sync_api import sync_playwright

OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)
with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 1920, "height": 1080})
    page.on("console", lambda m: print("console:", m.text) if m.type in ("error", "warning") else None)
    page.on("pageerror", lambda e: print("error:", e))
    page.goto("http://127.0.0.1:8812/film/index.html?still")
    page.wait_for_function("window.film", timeout=60000)
    print("duration", page.evaluate("film.duration"), page.evaluate("JSON.stringify(film.sound)"))
    for arg in sys.argv[1:]:
        page.evaluate("t => film.seek(t)", float(arg))
        page.wait_for_timeout(120)
        page.screenshot(path=OUT / f"peek_{arg}.jpg", type="jpeg", quality=80)
    browser.close()
