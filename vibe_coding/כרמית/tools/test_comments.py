"""Drives the player and admin.html with stubbed Firebase to test the free-text feedback beside the difficulty
question: one quiet line until pressed, a text box once pressed, sent with the button or on moving on, with or
without a difficulty level; nothing is saved from Ori's and Carmit's own devices; the statistics tab lists what
was written and can delete it.
Usage: python3 tools/test_comments.py http://localhost:8766/app/ [screenshot-dir]"""
import sys
from playwright.sync_api import sync_playwright

import fake_firebase
from fake_firebase import stub, check

BASE = sys.argv[1]
SHOTS = sys.argv[2] if len(sys.argv) > 2 else None

with sync_playwright() as p:
    b = p.chromium.launch()

    # ---- a real player ----
    ctx = b.new_context(viewport={"width": 420, "height": 900})
    page = ctx.new_page()
    stub(page)
    page.add_init_script("window.__NO_USER = true")
    comments = lambda: page.evaluate("Object.values(window.__store.comments)")
    page.goto(BASE + "index.html?q=qe")
    page.click("#start-btn")
    check(not page.is_visible("#rate-more"), "player: nothing about feedback before answering")
    page.fill("#answer-input", "ans1")
    page.click("#submit-btn")
    check(page.is_visible("#rate-more") and not page.is_visible("#rate-text"), "player: after answering, one line and no text box")
    size = page.evaluate("parseFloat(getComputedStyle(document.querySelector('#rate-more')).fontSize)")
    check(size <= 14, f"player: the line is small ({size}px)")
    if SHOTS:
        page.screenshot(path=f"{SHOTS}/comment-closed.png")
    page.click("#rate-more")
    check(page.is_visible("#rate-text") and page.evaluate("document.activeElement.id") == "rate-text", "player: pressing it opens the text box, ready to type")
    # Choosing a level while writing keeps the box and the words.
    page.fill("#rate-text", "חידה מקסימה")
    page.click(".rate-btn[data-rate=hard]")
    check(page.is_visible("#rate-text") and page.input_value("#rate-text") == "חידה מקסימה", "player: choosing a level keeps the text being written")
    if SHOTS:
        page.screenshot(path=f"{SHOTS}/comment-open.png")
    check(comments() == [], "player: nothing is sent while writing")
    page.click("#rate-send")
    cs = comments()
    check(len(cs) == 1 and cs[0]["text"] == "חידה מקסימה" and cs[0]["level"] == "hard" and cs[0]["riddle"] == "r1" and cs[0]["quiz"] == "qe"
          and cs[0]["correct"] is True and cs[0]["hint"] is False, f"player: sent with the level and how it went: {cs}")
    check(not page.is_visible("#rate-text") and "תודה" in page.inner_text("#rate-more"), "player: the box closes with a thank you")
    if SHOTS:
        page.screenshot(path=f"{SHOTS}/comment-sent.png")
    # A changed level follows the words already sent; editing overwrites.
    page.click(".rate-btn[data-rate=easy]")
    check(comments()[0]["level"] == "easy", "player: a changed level reaches the comment too")
    page.click("#rate-more")
    check(page.input_value("#rate-text") == "חידה מקסימה", "player: reopening shows what was sent")
    page.fill("#rate-text", "חידה מקסימה, תודה")
    page.click("#rate-send")
    cs = comments()
    check(len(cs) == 1 and cs[0]["text"] == "חידה מקסימה, תודה", "player: an edit overwrites, not a second comment")
    # Cancel drops what was not sent.
    page.click("#rate-more")
    page.fill("#rate-text", "משהו אחר")
    page.click("#rate-more")
    check(comments()[0]["text"] == "חידה מקסימה, תודה" and not page.is_visible("#rate-text"), "player: cancel drops the change")
    # Second riddle: words without a level, sent by moving on.
    page.click("#next-btn")
    check(not page.is_visible("#rate-more"), "player: hidden again on the next riddle")
    page.fill("#answer-input", "wrong")
    page.click("#submit-btn")
    check(page.is_visible("#rate-more") and "להוסיף" in page.inner_text("#rate-more") and not page.is_visible("#rate-text"),
          "player: the next riddle starts with a closed, empty line")
    page.click("#rate-more")
    page.fill("#rate-text", "  לא הבנתי את הרמז  ")
    page.click("#prev-btn")
    cs = sorted(comments(), key=lambda c: c["riddle"])
    check(len(cs) == 2 and cs[1]["text"] == "לא הבנתי את הרמז" and cs[1]["level"] == "" and cs[1]["correct"] is False and cs[1]["riddle"] == "r2",
          f"player: moving on sends what was written, with no level: {cs[-1]}")
    check("תודה" in page.inner_text("#rate-more") and page.get_attribute(".rate-btn[data-rate=easy]", "aria-checked") == "true",
          "player: back on the first riddle, its level and its thank you")
    check(len(page.evaluate("Object.values(window.__store.ratings)")) == 1, "player: the ratings are as before, one per rated riddle")
    ctx.close()

    # ---- the difficulty question switched off: no feedback line either ----
    ctx = b.new_context(viewport={"width": 420, "height": 900})
    page = ctx.new_page()
    stub(page)
    page.add_init_script("window.__NO_USER = true; window.__PATCH = { qe: { askRating: false } };")
    page.goto(BASE + "index.html?q=qe")
    page.click("#start-btn")
    page.fill("#answer-input", "ans1")
    page.click("#submit-btn")
    page.wait_for_selector("#next-btn:visible")
    check(not page.is_visible("#rate-more"), "switched off: no feedback line")
    ctx.close()

    # ---- Ori or Carmit: nothing is saved ----
    ctx = b.new_context(viewport={"width": 420, "height": 900})
    page = ctx.new_page()
    stub(page)
    page.goto(BASE + "index.html?p=qe")
    page.click("#start-btn")
    page.fill("#answer-input", "ans1")
    page.click("#submit-btn")
    page.wait_for_selector("#rate-note:visible")
    page.click("#rate-more")
    page.fill("#rate-text", "בדיקה")
    page.click("#rate-send")
    check(page.evaluate("Object.keys(window.__store.comments).length") == 0, "own device: the words are not saved")
    ctx.close()

    # ---- the statistics tab ----
    ctx = b.new_context(viewport={"width": 1200, "height": 1000})
    a = ctx.new_page()
    stub(a)
    a.add_init_script("""window.__COMMENTS = {
      k1: { quiz: 'qe', riddle: 'r1', text: 'ישן יותר', level: 'hard', correct: true, hint: true, created: 1700000000000 },
      k2: { quiz: 'qe', riddle: 'r2', text: 'חדש יותר\\nבשתי שורות <b>x</b>', level: '', correct: false, hint: false, created: 1700000100000 },
      k3: { quiz: 'gone', riddle: 'gone', text: 'על חידה שנמחקה', level: 'easy', correct: true, hint: false, created: 1600000000000 } };""")
    a.goto(BASE + "admin.html#stats")
    a.wait_for_selector("#comments-body .comment")
    texts = a.locator("#comments-body .comment-text").all_inner_texts()
    check(texts == ["חדש יותר\nבשתי שורות <b>x</b>", "ישן יותר", "על חידה שנמחקה"], f"stats: everything written, newest first, as plain text: {texts}")
    check(a.inner_text("#comments-count") == "3", "stats: the count")
    first = a.locator("#comments-body .comment").nth(0).inner_text()
    second = a.locator("#comments-body .comment").nth(1).inner_text()
    check("ANS2" in first and "קל" in first and "לא פתר" in first and "לא סימן רמת קושי" in first, f"stats: riddle, quiz and how it went: {first!r}")
    check("ANS1" in second and "נעזר ברמז" in second and 'סימן "קשה"' in second, f"stats: hint and level: {second!r}")
    check("חידה שנמחקה" in a.locator("#comments-body .comment").nth(2).inner_text(), "stats: a comment on a deleted riddle still shows")
    if SHOTS:
        a.locator("#comments-body").scroll_into_view_if_needed()
        a.screenshot(path=f"{SHOTS}/comment-stats.png", full_page=True)
    a.locator("#comments-body .comment").nth(1).get_by_role("button", name="מחיקה").click()   # the confirm is accepted by the stub
    a.wait_for_timeout(150)
    check(a.evaluate("Object.keys(window.__store.comments).sort().join()") == "k2,k3" and a.locator("#comments-body .comment").count() == 2,
          "stats: a comment can be deleted")
    ctx.close()
    b.close()

sys.exit(0 if fake_firebase.ok else 1)
