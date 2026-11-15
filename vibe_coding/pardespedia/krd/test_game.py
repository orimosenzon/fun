"""Plays the "למי מגיע הקרדיט?" game on the live page with the engine injected."""
import os, sys
from playwright.sync_api import sync_playwright
S = sys.argv[1] if len(sys.argv) > 1 else "."
ONLY = sys.argv[2] if len(sys.argv) > 2 else None
URL = "https://pardespedia.info/wiki/גן_יד_לבנים"
HERE = os.path.dirname(os.path.abspath(__file__))
errs = []
with sync_playwright() as p:
    br = p.chromium.launch(args=['--no-zygote', '--single-process', '--disable-gpu', '--js-flags=--max-old-space-size=256'], timeout=90000)
    for name, vp in (("desk", {'width': 1200, 'height': 900}), ("phone", {'width': 390, 'height': 844})):
        if ONLY and name != ONLY: continue
        pg = br.new_page(viewport=vp); pg.on('pageerror', lambda e: errs.append(str(e)))
        pg.goto(URL, wait_until='networkidle', timeout=60000); pg.wait_for_timeout(600)
        if name == "desk":
            print("fallback: questions", pg.locator('.krd-q').count(), "events", pg.locator('.krd-ev').count(),
                  "periods", pg.locator('.krd-period').count(),
                  "explain hidden", not pg.locator('.krd-explain-body').first.is_visible())
            pg.locator('.krd-quiz').screenshot(path=f"{S}/fb.png")
        if pg.locator('.krd-js').count() == 0:
            pg.add_script_tag(path=os.path.join(HERE, "krd_game.js")); pg.wait_for_timeout(800)
        print(name, "static hidden:", not pg.locator('.krd-questions').is_visible())
        pg.locator('.krd-quiz').screenshot(path=f"{S}/{name}_intro.png")
        pg.click('text=מתחילים'); pg.wait_for_timeout(300)
        wrong = {1, 4, 7}
        nq = pg.locator('.krd-q').count()
        for i in range(nq):
            corr = pg.evaluate(f"document.querySelectorAll('.krd-q')[{i}].getAttribute('data-correct').trim()")
            keys = [b.get_attribute('data-key') for b in pg.locator('.krd-choice').all()]
            pick = corr if i not in wrong else [k for k in keys if k != corr][0]
            if i == 3: pg.locator('.krd-quiz').screenshot(path=f"{S}/{name}_q_before.png")
            pg.locator(f'.krd-choice[data-key="{pick}"]').click(); pg.wait_for_timeout(250)
            if i == 4: pg.locator('.krd-quiz').screenshot(path=f"{S}/{name}_q_after.png")
            pg.locator('button.krd-btn').last.click(); pg.wait_for_timeout(900)
            if i == 0:
                top = pg.evaluate("document.querySelector('.krd-quiz').getBoundingClientRect().top")
                print(name, "box top after next:", round(top))
        print(name, "score:", pg.locator('.krd-quiz span[style*="260%"]').inner_text())
        print(name, "timeline:", pg.locator('.krd-timeline').is_visible(), "era:", pg.locator('.krd-era').count(),
              "story:", pg.locator('.krd-story').is_visible())
        pg.locator('.krd-quiz').screenshot(path=f"{S}/{name}_results.png")
        pg.locator('.krd-era').screenshot(path=f"{S}/{name}_era.png")
    br.close()
print("errors:", errs)
