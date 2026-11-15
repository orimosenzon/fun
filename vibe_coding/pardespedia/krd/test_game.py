"""Plays the "למי מגיע הקרדיט?" game on the live page.

Usage: python3 krd/test_game.py <screenshot dir> [desk|phone] [local]
If the engine is not yet in MediaWiki:Common.js it is injected from krd_game.js.
"local" tests krd_game.js even when an older engine is already live: the widget
is replaced by a fresh copy from the parse API and the local engine is injected.
Chrome is launched with --single-process (the machine is often short on memory),
so each run opens one page only: run desk and phone separately.
"""
import os, sys
from playwright.sync_api import sync_playwright

S = sys.argv[1] if len(sys.argv) > 1 else "."
ONLY = sys.argv[2] if len(sys.argv) > 2 else "desk"
LOCAL = "local" in sys.argv[3:]
PAGE = os.environ.get("KRD_PAGE", "גן יד לבנים")  # KRD_PAGE=<title> tests another article
URL = "https://pardespedia.info/wiki/" + PAGE.replace(" ", "_")
HERE = os.path.dirname(os.path.abspath(__file__))
VP = {"desk": {'width': 1200, 'height': 900}, "phone": {'width': 390, 'height': 844}}[ONLY]
WRONG = {1, 4, 7}
errs, fails = [], []


def check(cond, label):
    print(("OK   " if cond else "FAIL ") + label)
    if not cond:
        fails.append(label)


with sync_playwright() as p:
    br = p.chromium.launch(args=['--no-zygote', '--single-process', '--disable-gpu'], timeout=90000)
    pg = br.new_page(viewport=VP)
    pg.on('pageerror', lambda e: errs.append(str(e)))
    pg.goto(URL, wait_until='networkidle', timeout=60000)
    pg.wait_for_timeout(700)
    pg.locator('.krd-quiz').screenshot(path=f"{S}/{ONLY}_nojs.png")
    print("no-engine widget height:", pg.locator('.krd-quiz').bounding_box()['height'])
    injected = pg.locator('.krd-js').count() == 0
    if LOCAL and not injected:
        pg.evaluate("""async () => {
            const r = await fetch('/api.php?action=parse&page=' + encodeURIComponent(document.title.split(' – ')[0]) + '&prop=text&format=json&formatversion=2');
            const doc = new DOMParser().parseFromString((await r.json()).parse.text, 'text/html');
            document.querySelector('.krd-quiz').replaceWith(document.importNode(doc.querySelector('.krd-quiz'), true));
        }""")
        injected = True
    if injected:
        pg.add_script_tag(path=os.path.join(HERE, "krd_game.js"))
        pg.wait_for_timeout(800)
    print("engine:", "injected" if injected else "from Common.js")

    h = pg.locator('.krd-quiz').bounding_box()['height']
    check(h < 130, f"invite widget is small ({h:.0f}px)")
    check(not pg.locator('.krd-questions').is_visible(), "questions hidden before play")
    pg.locator('.krd-quiz').screenshot(path=f"{S}/{ONLY}_invite.png")

    pg.locator('.krd-head button').click()
    pg.wait_for_timeout(300)
    nq = pg.locator('.krd-q').count()
    for i in range(nq):
        corr = pg.evaluate(f"document.querySelectorAll('.krd-q')[{i}].getAttribute('data-correct').trim()")
        keys = [b.get_attribute('data-key') for b in pg.locator('.krd-choice').all()]
        pick = corr if i not in WRONG else [k for k in keys if k != corr][0]
        if i == 3:
            pg.locator('.krd-quiz').screenshot(path=f"{S}/{ONLY}_question.png")
        print('q', i, 'pick', pick, 'shown:', pg.locator('.krd-quiz').inner_text()[:60].replace('\n', ' | '), flush=True)
        pg.locator(f'.krd-choice[data-key="{pick}"]').click(timeout=8000)
        pg.wait_for_timeout(700)
        if i == 1:
            # nothing may reveal right/wrong during the quiz
            check(pg.locator('.krd-quiz').inner_text().count('נכון') == 0, "no right/wrong shown mid-quiz")
            check('שאלה 3 מתוך' in pg.locator('.krd-quiz').inner_text(), "auto-advanced to question 3")
            pg.locator('button.krd-link', has_text='הקודמת').click(); pg.wait_for_timeout(300)
            check(pg.locator('.krd-choice.krd-picked').count() == 1, "back: previous answer still marked")
            pg.locator('button.krd-link', has_text='הבאה').click(); pg.wait_for_timeout(300)
        if i == 2:
            top = pg.evaluate("document.querySelector('.krd-quiz').getBoundingClientRect().top")
            check(top > 40, f"box not under the sticky bar (top={top:.0f})")

    score = pg.locator('.krd-quiz span[style*="250%"]').inner_text()
    check(score.replace(' ', '') == f"{nq - len(WRONG)}/{nq}", f"score {score}")
    check(pg.locator('.krd-era').count() == 1, "graphic timeline drawn")
    check(pg.locator('.krd-story').is_visible(), "story shown")
    check(pg.get_by_text('התשובה הנכונה:').count() == len(WRONG), "correct answers listed for the wrong ones")
    if pg.locator('.krd-slice').count():
        n = pg.locator('.krd-slice').count()
        check(pg.locator('.krd-pie').is_visible(), "credit pie shown")
        check(pg.locator('.krd-pie-seg').count() == n, f"pie has {n} slices")
        check(pg.locator('.krd-pie-row').count() == n, "pie legend row per slice")
        check(not pg.locator('.krd-pie-slices').is_visible(), "fallback bars hidden")
        box = pg.locator('.krd-pie').bounding_box()
        check(box['width'] <= VP['width'], "pie fits the viewport")
        pg.locator('.krd-pie').screenshot(path=f"{S}/{ONLY}_pie.png")
    pg.locator('.krd-quiz').screenshot(path=f"{S}/{ONLY}_results.png")
    pg.locator('.krd-era').screenshot(path=f"{S}/{ONLY}_era.png")

    pg.locator('button.krd-btn', has_text='סגירה').click(); pg.wait_for_timeout(400)
    h2 = pg.locator('.krd-quiz').bounding_box()['height']
    check(h2 < 130, f"closed back to the small widget ({h2:.0f}px)")
    pg.locator('.krd-head button').click(); pg.wait_for_timeout(300)
    check('שאלה 1 מתוך' in pg.locator('.krd-quiz').inner_text(), "replay starts at question 1")
    br.close()

check(not errs, "no JS errors " + "; ".join(errs))
print("FAILED:" if fails else "ALL PASSED", ", ".join(fails))
