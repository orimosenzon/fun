"""Drives admin.html with stubbed Firebase to test making a riddle image from a sentence: typing in
"ליצור שקופית ממשפט" draws a slide into the riddle picture, the font size changes it, the riddle saves with
its sentence (`slide`), editing reopens the sentence, and choosing a real picture instead drops `slide`.
Usage: python3 tools/test_text_slide.py http://localhost:8767/vibe_coding/כרמית/app/ [screenshot-dir]"""
import sys
from pathlib import Path
from playwright.sync_api import sync_playwright

import fake_firebase
from fake_firebase import stub, check

BASE = sys.argv[1]
SHOTS = sys.argv[2] if len(sys.argv) > 2 else None
PICTURE = str(Path(__file__).resolve().parent.parent / "app" / "favicon.png")
SENTENCE = "איזה מספר מתאים לך לי לעולם"

with sync_playwright() as p:
    b = p.chromium.launch()
    ctx = b.new_context(viewport={"width": 1200, "height": 1000})
    a = ctx.new_page()
    stub(a)
    store = lambda expr: a.evaluate(f"(() => {{ const s = window.__store; return {expr}; }})()")
    src = lambda: a.get_attribute("#image-preview", "src") or ""
    a.goto(BASE + "admin.html#pool")
    a.wait_for_selector("#tab-pool:visible .pool-tile")

    # ---- typing a sentence draws the slide ----
    a.click("#pool-new")
    a.wait_for_selector("#riddle-dialog[open]")
    check(not a.is_visible("#slide-text"), "the slide maker starts closed")
    a.click("#slide-maker summary")
    a.fill("#slide-text", SENTENCE)
    a.wait_for_selector("#image-preview:not(.hidden)")
    first = src()
    check(first.startswith("data:image/jpeg"), "the sentence became the riddle picture")
    size = a.evaluate("(() => { const i = document.getElementById('image-preview'); return [i.naturalWidth, i.naturalHeight]; })()")
    check(size == [1280, 720], f"the slide is 1280×720: {size}")
    if SHOTS:
        a.locator("#riddle-dialog").screenshot(path=f"{SHOTS}/slide-form.png")

    # ---- the font size changes it ----
    a.eval_on_selector("#slide-size", "(el) => { el.value = 140; el.dispatchEvent(new Event('input')); }")
    a.wait_for_function("(old) => document.getElementById('image-preview').src !== old", arg=first)
    check(a.inner_text("#slide-size-out") == "140", "the size shows next to the slider")

    # ---- saved with its sentence ----
    a.fill("#answers-list .answer-text", "7")
    a.click("#save-btn")
    a.wait_for_selector("#riddle-dialog", state="hidden")
    saved = store(f"Object.entries(s.riddles).find(([, r]) => r.slide?.text === '{SENTENCE}')")
    check(saved is not None and saved[1]["slide"]["size"] == 140 and saved[1]["image"].startswith("data:image/jpeg"),
          "saved with the picture and the sentence")
    if SHOTS:
        Path(f"{SHOTS}/slide.jpg").write_bytes(__import__("base64").b64decode(saved[1]["image"].split(",", 1)[1]))

    # ---- editing reopens the sentence; a real picture instead drops it ----
    # The pool lists newest first, so the first tile is the slide riddle.
    a.locator("#pool-grid .pool-tile").first.get_by_role("button", name="עריכה").click()
    a.wait_for_selector("#riddle-dialog[open]")
    check(a.is_visible("#slide-text") and a.input_value("#slide-text") == SENTENCE, "editing reopens the sentence")
    check(a.input_value("#slide-size") == "140", "and its size")
    a.set_input_files("#image-file", PICTURE)
    a.wait_for_function("(old) => document.getElementById('image-preview').src !== old", arg=saved[1]["image"])
    a.click("#save-btn")
    a.wait_for_selector("#riddle-dialog", state="hidden")
    check(store(f"s.riddles['{saved[0]}'].slide") is None, "a chosen picture replaces the slide")

    # ---- a new riddle starts with the slide maker empty and closed ----
    a.click("#pool-new")
    a.wait_for_selector("#riddle-dialog[open]")
    check(not a.is_visible("#slide-text") and a.input_value("#slide-text") == "", "a new riddle starts without a sentence")
    b.close()

sys.exit(0 if fake_firebase.ok else 1)
