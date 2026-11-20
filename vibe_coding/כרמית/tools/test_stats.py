"""Drives the player and admin.html with stubbed Firebase to test (1) the "how hard was this riddle" buttons: a
rating is saved once per riddle per visit, a changed mind overwrites it, and it survives going back; (2) visits:
one document per browser per day per page, with started/finished; (3) Ori and Carmit's own browsers: no ratings
saved, visits marked own; (4) the statistics tab: real visitors apart from own ones, and the ratings per quiz.
Usage: python3 tools/test_stats.py http://localhost:8766/app/ [screenshot-dir]"""
import json
import sys
from datetime import date, timedelta
from playwright.sync_api import sync_playwright

import fake_firebase
from fake_firebase import stub, check

BASE = sys.argv[1]
SHOTS = sys.argv[2] if len(sys.argv) > 2 else None

with sync_playwright() as p:
    b = p.chromium.launch()

    # ---- a real player: signed out, a browser that never saw the admin ----
    ctx = b.new_context(viewport={"width": 420, "height": 900})
    page = ctx.new_page()
    stub(page)
    page.add_init_script("window.__NO_USER = true")
    page.goto(BASE + "index.html?q=qe")
    page.wait_for_selector("#start-btn:visible")
    page.wait_for_function("Object.keys(window.__store.visits).length === 1")
    v = page.evaluate("Object.values(window.__store.visits)[0]")
    check(v["quiz"] == "qe" and v["solo"] is True and v["own"] is False and "started" not in v, f"player: the visit is recorded: {v}")
    page.click("#start-btn")
    check(not page.is_visible("#rate"), "player: no rating buttons before answering")
    page.fill("#answer-input", "ans1")
    page.click("#submit-btn")
    check(page.is_visible("#rate"), "player: rating buttons after answering")
    check(not page.is_visible("#rate-note"), "player: no admin note for a real player")
    if SHOTS:
        page.screenshot(path=f"{SHOTS}/rate-phone.png", full_page=True)
    page.click(".rate-btn[data-rate=hard]")
    page.click(".rate-btn[data-rate=medium]")
    rs = page.evaluate("Object.values(window.__store.ratings)")
    check(len(rs) == 1 and rs[0]["level"] == "medium" and rs[0]["correct"] is True and rs[0]["quiz"] == "qe" and rs[0]["riddle"] == "r1",
          f"player: one rating, the last choice: {rs}")
    page.click("#next-btn")
    check(not page.is_visible("#rate"), "player: buttons hidden on the next riddle")
    page.click("#prev-btn")
    check(page.get_attribute(".rate-btn[data-rate=medium]", "aria-checked") == "true", "player: going back shows the choice")
    page.click("#next-btn")
    page.click("#skip-btn")
    page.wait_for_selector("#report-screen:visible")
    vs = page.evaluate("Object.values(window.__store.visits)")
    check(len(vs) == 1 and vs[0].get("started") is True and vs[0].get("finished") is True, f"player: started and finished on the same visit document: {vs}")
    check(len(page.evaluate("Object.values(window.__store.ratings)")) == 1, "player: a skipped riddle has no rating")
    ctx.close()

    # ---- Ori or Carmit: signed in ----
    ctx = b.new_context(viewport={"width": 420, "height": 900})
    page = ctx.new_page()
    stub(page)
    page.goto(BASE + "index.html?p=qe")
    page.wait_for_selector("#start-btn:visible")
    page.wait_for_function("Object.keys(window.__store.visits).length === 1")
    check(page.evaluate("Object.values(window.__store.visits)[0].own") is True, "admin playing: the visit is marked own")
    page.click("#start-btn")
    page.fill("#answer-input", "wrong")
    page.click("#submit-btn")
    check(page.is_visible("#rate-note"), "admin playing: told the rating is not counted")
    page.click(".rate-btn[data-rate=easy]")
    check(page.evaluate("Object.keys(window.__store.ratings).length") == 0, "admin playing: no rating saved")
    ctx.close()

    # ---- a browser that once signed in to the admin stays "own" after signing out ----
    ctx = b.new_context(viewport={"width": 1200, "height": 1000})
    admin = ctx.new_page()
    stub(admin)
    day = lambda n: (date.today() - timedelta(days=n)).isoformat()
    visits = {}
    def visit(visitor, n, quiz, **kw):
        visits[f"{visitor}-{n}-{quiz}"] = {"visitor": visitor, "day": day(n), "quiz": quiz, "solo": False, "own": False, **kw}
    visit("a", 0, ""); visit("a", 0, "qe", started=True, finished=True); visit("a", 3, "qe", started=True)
    visit("b", 0, "qe", solo=True, started=True); visit("c", 2, ""); visit("c", 2, "qm", started=True, finished=True)
    visit("d", 20, "qe"); visit("old", 40, "qe")
    # Carmit: one day before her browser was marked, one after. Both must count as hers.
    visit("carmit", 5, "qe"); visit("carmit", 1, "qe", own=True)
    ratings = {}
    for i, (quiz, riddle, level, correct) in enumerate(
            [("qe", "r1", "easy", True)] * 5 + [("qe", "r1", "medium", True)] * 2 + [("qe", "r2", "hard", False)] * 4
            + [("qe", "r2", "medium", True)] + [("qm", "r3", "hard", False)] * 2 + [("gone", "r1", "easy", True)]):
        ratings[f"x{i}"] = {"quiz": quiz, "riddle": riddle, "level": level, "correct": correct, "hint": False}
    admin.add_init_script(f"window.__VISITS = {json.dumps(visits)}; window.__RATINGS = {json.dumps(ratings)};")
    admin.goto(BASE + "admin.html#stats")
    admin.wait_for_selector(".stat-quiz")
    tiles = admin.locator("#visits-body .stat-tile b").all_inner_texts()
    check(tiles == ["5", "3", "2", "6"], f"stats: real visitors total / week / today / visit-days, without Carmit: {tiles}")
    own = admin.inner_text("#visits-body")
    check("2 ביקורים מ-דפדפן אחד" in own, "stats: Carmit's visits apart, including the day before her browser was marked")
    row = admin.locator(".visits-table tr", has_text="קל").locator("td.num").all_inner_texts()
    check(row == ["4", "2", "1", "1"], f"stats: quiz row opened / started / finished / from a sent link: {row}")
    check(admin.locator(".day-col").count() == 30, "stats: thirty days in the chart")
    check(admin.inner_text("#stats-count") == "14", "stats: ratings counted, without the deleted quiz")
    card = admin.locator(".stat-quiz", has_text="קל").first
    text = card.inner_text()
    check("12 דירוגים" in text and "קלה 5 · בינונית 3 · קשה 4" in text, "stats: the quiz's ratings")
    check("רמה בינונית" in text and "שונה ממה שהוגדר" in text, "stats: players' verdict differs from the set level")
    check("מעט מדי" in admin.locator(".stat-quiz", has_text="בינוני").last.inner_text(), "stats: two ratings are too few for a verdict")
    card.locator("summary").click()
    r2 = card.locator(".stat-riddle", has_text="ANS2").inner_text()
    check("5 דירוגים" in r2 and "20% פתרו" in r2, f"stats: per riddle: {r2!r}")
    if SHOTS:
        admin.screenshot(path=f"{SHOTS}/stats-desk.png", full_page=True)
        admin.set_viewport_size({"width": 400, "height": 900})
        admin.screenshot(path=f"{SHOTS}/stats-phone.png", full_page=True)
    player = ctx.new_page()
    stub(player)
    player.add_init_script("window.__NO_USER = true")
    player.goto(BASE + "index.html")
    player.wait_for_function("Object.keys(window.__store.visits).length === 1")
    mine = player.evaluate("Object.values(window.__store.visits)")
    check(len(mine) == 1 and mine[0]["own"] is True and mine[0]["quiz"] == "", f"own browser, signed out: still marked own: {mine}")
    b.close()

sys.exit(0 if fake_firebase.ok else 1)
