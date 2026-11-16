"""Drives admin.html and the player with stubbed Firebase to test (1) a quiz's own heading (text and an optional
image) in place of the usual Hebrew × English line, written in the admin or taken over from another quiz.
Letters are tested in test_letters_levels_focus.py.
Usage: python3 tools/test_heading_letters.py http://localhost:8766/app/   (a server rooted at the project folder)"""
import json
import sys
from playwright.sync_api import sync_playwright

import fake_firebase
from fake_firebase import stub, check, IMG

BASE = sys.argv[1]

with sync_playwright() as p:
    b = p.chromium.launch()
    ctx = b.new_context(viewport={"width": 1200, "height": 1000})
    shared = []

    # ---- heading ----
    admin = ctx.new_page()
    admin.add_init_script("window.__PATCH = { qh: { heading: { text: 'ציור × מילים\\nכשהתמונה מדברת', image: '%s' } } };" % IMG)
    stub(admin)
    admin.on("dialog", lambda d: shared.append(d.default_value))  # no clipboard in headless: links come in a prompt
    admin.goto(BASE + "admin.html#quiz/qm")
    admin.wait_for_selector("#quiz-list")
    check("עברית" in admin.inner_text("#heading-preview"), "admin: no heading yet, the preview shows the usual line")
    check(admin.is_hidden("#heading-reset"), "admin: nothing to reset yet")
    admin.fill("#heading-text", "שורה ראשית\nוהסבר מתחתיה")
    check(admin.inner_text("#heading-preview .langs") == "שורה ראשית", "admin: the preview follows the typing, first line bold")
    admin.wait_for_timeout(900)
    h = admin.evaluate("window.__store.quizzes.qm.heading")
    check(h == {"text": "שורה ראשית\nוהסבר מתחתיה", "image": ""}, f"admin: the heading is saved after a pause: {h}")
    opts = admin.locator("#heading-from option").all_text_contents()
    check(any("ציור × מילים" in o and "קשה" in o for o in opts), f"admin: another quiz's heading is offered with its quiz: {opts}")
    # Taking over the other quiz's heading, image and all.
    val = admin.eval_on_selector("#heading-from", "s => [...s.options].find(o => o.text.includes('ציור')).value")
    admin.select_option("#heading-from", val)
    admin.wait_for_timeout(200)
    h = admin.evaluate("window.__store.quizzes.qm.heading")
    check(h["text"] == "ציור × מילים\nכשהתמונה מדברת" and h["image"] == IMG, f"admin: the heading taken over, image too: {h}")
    check(admin.input_value("#heading-text") == "ציור × מילים\nכשהתמונה מדברת", "admin: the text box shows it")
    check(admin.get_attribute("#heading-preview img", "src") == IMG, "admin: the preview shows the image")
    # A new quiz: write a heading or take one from the existing quizzes.
    admin.goto(BASE + "admin.html")
    admin.wait_for_selector(".new-quiz")
    admin.click(".new-quiz")
    admin.wait_for_selector("#quiz-view:visible")
    opts = admin.locator("#heading-from option").all_text_contents()
    check(len(opts) >= 2 and not admin.is_disabled("#heading-from"), f"admin: a new quiz can take an existing heading: {opts}")

    # The player sees the quiz's heading instead of the usual line; a quiz without one keeps the usual line.
    page = ctx.new_page()
    page.add_init_script("window.__PATCH = { qm: { heading: %s } };" % json.dumps(h))
    stub(page)
    page.goto(BASE + "index.html?q=qm")
    page.wait_for_selector("#start-btn:visible")
    check(page.is_hidden("#motif") and page.is_visible("#quiz-heading"), "player: the quiz's heading replaces the usual line")
    check(page.inner_text("#quiz-heading .langs") == "ציור × מילים" and page.inner_text("#quiz-heading .rest") == "כשהתמונה מדברת", "player: headline and explanation")
    check(page.get_attribute("#quiz-heading img", "src") == IMG, "player: the heading image")
    check("עברית ואנגלית" not in page.inner_text("#intro"), "player: the intro doesn't repeat the usual Hebrew/English line")
    page.click("#start-btn")
    check(page.is_visible("#quiz-heading"), "player: the heading stays while playing")
    page.goto(BASE + "index.html?q=qe")
    page.wait_for_selector("#start-btn:visible")
    check(page.is_visible("#motif") and page.is_hidden("#quiz-heading"), "player: no heading of its own, the usual line")
    page.goto(BASE + "index.html")
    page.wait_for_selector(".gallery-card")
    check(page.is_visible("#motif"), "player: the start screen keeps the usual line")

    b.close()

print("ALL PASS" if fake_firebase.ok else "SOME FAILED")
sys.exit(0 if fake_firebase.ok else 1)
