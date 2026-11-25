"""Drives admin.html and the player with stubbed Firebase to test saving a riddle before it has an answer:
the form saves it (text or picture answer type), the admin marks it "בלי תשובה עדיין", and players skip it
until it gets an answer.
Usage: python3 tools/test_no_answer.py http://localhost:8767/vibe_coding/כרמית/app/ [screenshot-dir]"""
import sys
from pathlib import Path
from playwright.sync_api import sync_playwright

import fake_firebase
from fake_firebase import stub, check

BASE = sys.argv[1]
SHOTS = sys.argv[2] if len(sys.argv) > 2 else None
PICTURE = str(Path(__file__).resolve().parent.parent / "app" / "favicon.png")

with sync_playwright() as p:
    b = p.chromium.launch()
    ctx = b.new_context(viewport={"width": 1200, "height": 1000})
    a = ctx.new_page()
    stub(a)
    store = lambda expr: a.evaluate(f"(() => {{ const s = window.__store; return {expr}; }})()")
    a.goto(BASE + "admin.html#pool")
    a.wait_for_selector("#tab-pool:visible .pool-tile")
    before = store("Object.keys(s.riddles).length")

    # ---- a text riddle with no answer is saved ----
    a.click("#pool-new")
    a.wait_for_selector("#riddle-dialog[open]")
    a.set_input_files("#image-file", PICTURE)
    a.wait_for_selector("#image-preview:not(.hidden)")
    a.fill("#question", "TEXT-DRAFT")
    a.click("#save-btn")
    a.wait_for_selector("#riddle-dialog", state="hidden")
    saved = store("Object.values(s.riddles).find((r) => r.question === 'TEXT-DRAFT')")
    check(saved is not None and saved["answers"] == [], "text: saved without an answer")
    check("לא יראו" in a.inner_text("#toast"), f"text: the toast says players won't see it: {a.inner_text('#toast')}")

    # ---- a picture-answer riddle with no answer picture is saved ----
    a.click("#pool-new")
    a.wait_for_selector("#riddle-dialog[open]")
    a.set_input_files("#image-file", PICTURE)
    a.wait_for_selector("#image-preview:not(.hidden)")
    a.fill("#question", "IMAGE-DRAFT")
    a.check('input[name="answer-type"][value="image"]')
    a.click("#save-btn")
    a.wait_for_selector("#riddle-dialog", state="hidden")
    saved = store("Object.values(s.riddles).find((r) => r.question === 'IMAGE-DRAFT')")
    check(saved is not None and saved["answerType"] == "image" and saved["answerImage"] == "", "image: saved without an answer picture")
    check(store("Object.keys(s.riddles).length") == before + 2, "both new riddles are in the pool")

    # ---- the admin marks them ----
    marks = a.locator("#pool-grid .tag.no-answer")
    check(marks.count() == 2 and marks.first.inner_text() == "בלי תשובה עדיין", "pool: both marked 'בלי תשובה עדיין'")
    if SHOTS:
        a.screenshot(path=f"{SHOTS}/no-answer-pool.png")

    # ---- adding the answer later clears the mark ----
    a.locator("#pool-grid .pool-tile", has_text="בלי תשובה").first.get_by_role("button", name="עריכה").click()
    a.wait_for_selector("#riddle-dialog[open]")
    if a.is_checked('input[name="answer-type"][value="text"]'):
        a.fill("#answers-list .answer-text", "LATER")
    else:
        a.set_input_files("#answer-file", PICTURE)
        a.wait_for_selector("#answer-preview:not(.hidden)")
    a.click("#save-btn")
    a.wait_for_selector("#riddle-dialog", state="hidden")
    check(a.locator("#pool-grid .tag.no-answer").count() == 1, "pool: the riddle that got an answer lost the mark")

    # ---- the player skips a riddle without an answer ----
    pl = ctx.new_page()
    stub(pl)
    pl.add_init_script("window.__RIDDLES = { r2: { answers: [] } };")
    pl.goto(BASE + "index.html?q=qe")
    pl.wait_for_selector("#start-btn:visible")
    pl.click("#start-btn")
    pl.wait_for_function("document.getElementById('progress-text').textContent.trim() !== ''")
    check(pl.inner_text("#progress-text").replace(" ", "") == "01/01", f"player: only the answered riddle: {pl.inner_text('#progress-text')}")
    b.close()

sys.exit(0 if fake_firebase.ok else 1)
