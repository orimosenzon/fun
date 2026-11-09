"""Drives the player (index.html) and admin.html with stubbed Firebase to test the path: the quizzes Carmit
chose for players, in her order. Only the next quiz is open, finishing it opens the one after, progress lasts
for the visit only, and the admin's "what the player sees" tab sets the path.
Usage: python3 tools/test_path.py http://localhost:8766/app/   (a server rooted at the project folder)"""
import sys
from playwright.sync_api import sync_playwright

import fake_firebase
from fake_firebase import stub, check

BASE = sys.argv[1]


def play_through(page, answer):
    page.click("#start-btn")
    while True:
        page.fill("#answer-input", answer)
        page.click("#submit-btn")
        page.wait_for_selector("#next-btn:visible")
        page.click("#next-btn")
        if page.is_visible("#report-screen"):
            break


def titles(page):
    return [t.strip() for t in page.locator(".gallery-card b").all_text_contents()]


with sync_playwright() as p:
    b = p.chromium.launch()

    # No path set yet: every quiz shows, easy first, so the game never goes empty.
    ctx = b.new_context(viewport={"width": 420, "height": 900})
    page = ctx.new_page()
    stub(page)
    page.goto(BASE + "index.html")
    page.wait_for_selector(".gallery-card")
    t = titles(page)
    check(len(t) == 4 and t[-2:] == ["בינוני", "קשה"], f"without a path: all quizzes, easy first: {t}")
    check(page.locator("#start-screen h2:text-is('חידונים')").count() == 0, "no 'quizzes' heading")
    y = lambda sel: page.locator(sel).first.bounding_box()["y"]
    check(y("h1") < y(".motif") < y(".lede-top") < y("#gallery") < y(".about"),
          "Carmit's order: title, Hebrew x English, the lede, the quizzes, then the about")
    check(page.evaluate("() => document.documentElement.scrollWidth <= innerWidth"), "no sideways scroll at phone width")
    ctx.close()

    # A path of three, in Carmit's order: hard, easy, medium. The old quiz is off the path.
    ctx = b.new_context(viewport={"width": 420, "height": 900})
    ctx.add_init_script("window.__PATH = ['qh', 'qe', 'qm'];")
    page = ctx.new_page()
    stub(page)
    page.goto(BASE + "index.html")
    page.wait_for_selector(".gallery-card")
    check(titles(page) == ["קשה", "קל", "בינוני"], f"only the path, in its order: {titles(page)}")
    check(page.locator("a.gallery-card").count() == 1, "only one quiz is open")
    check(page.locator("a.gallery-card.next b").text_content() == "קשה", "the open one is the first on the path")
    check(page.locator(".gallery-card.locked").count() == 2, "the rest are locked")

    page.click("a.gallery-card.next")
    page.wait_for_url("**?p=qh")
    page.wait_for_selector("#start-btn:visible")
    check("קשה" in page.text_content("#quiz-level"), "the quiz page shows its level")
    play_through(page, "wrong")  # no threshold: finishing is enough
    page.wait_for_selector("#next-level-btn:visible")
    check("קל" in page.text_content("#next-level-btn"), f"the report leads to the next quiz: {page.text_content('#next-level-btn')!r}")
    page.click("#next-level-btn")
    page.wait_for_url("**?p=qe")
    page.wait_for_selector("#start-btn:visible")
    check(page.text_content("#quiz-name") == "קל", "the next-quiz button opens it")

    page.goto(BASE + "index.html")
    page.wait_for_selector(".gallery-card")
    check(page.locator(".gallery-card.done").count() == 1, "the finished quiz shows as done")
    check(page.locator("a.gallery-card.next b").text_content() == "קל", "the second quiz is now the open one")

    # A new visit starts again from the first quiz.
    page2 = ctx.new_page()
    stub(page2)
    page2.goto(BASE + "index.html")
    page2.wait_for_selector(".gallery-card")
    check(page2.locator("a.gallery-card.next b").text_content() == "קשה", "a new visit starts again from the first")

    # Admin: the path tab shows the path and what's off it, and editing it saves positions.
    admin = ctx.new_page()
    stub(admin)
    admin.goto(BASE + "admin.html#path")
    admin.wait_for_selector("#path-list .path-item")
    on = [t.strip() for t in admin.locator("#path-list .path-item .meta b").all_text_contents()]
    off = [t.strip() for t in admin.locator("#path-out .path-item .meta b").all_text_contents()]
    check(on == ["קשה", "קל", "בינוני"] and off == ["ישן"], f"admin: path and the rest: {on} / {off}")
    admin.click("#path-out .path-item")
    admin.wait_for_timeout(200)
    check(admin.evaluate("window.__store.quizzes.qold.path") == 3, "admin: clicking adds to the end of the path")
    admin.click("#path-list .path-item >> nth=0 >> .icon-btn")
    admin.wait_for_timeout(200)
    pos = admin.evaluate("Object.fromEntries(Object.entries(window.__store.quizzes).map(([k, q]) => [k, q.path]))")
    check(pos == {"qold": 2, "qe": 0, "qm": 1, "qh": None}, f"admin: removing renumbers the rest: {pos}")
    admin.drag_and_drop("#path-list .path-item >> nth=2", "#path-list .path-item >> nth=0", target_position={"x": 20, "y": 5})
    admin.wait_for_timeout(200)
    pos = admin.evaluate("Object.fromEntries(Object.entries(window.__store.quizzes).map(([k, q]) => [k, q.path]))")
    check(pos == {"qold": 0, "qe": 1, "qm": 2, "qh": None}, f"admin: dragging reorders: {pos}")

    # Admin: the level picker shows the quiz's level and saves a change.
    admin.goto(BASE + "admin.html#quiz/qold")
    admin.wait_for_selector("#quiz-list")
    check(admin.is_checked('input[name="quiz-level"][value="easy"]'), "admin: an old quiz shows as easy")
    admin.click('label.seg:has(input[name="quiz-level"][value="hard"])')
    admin.wait_for_timeout(200)
    check(admin.evaluate("window.__store.quizzes.qold.level") == "hard", "admin: choosing hard saves level=hard")
    b.close()

print("ALL PASS" if fake_firebase.ok else "SOME FAILED")
sys.exit(0 if fake_firebase.ok else 1)
