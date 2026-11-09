"""Drives the player with stubbed Firebase to test (1) the way into the admin: one clear button on the start
screen only, and for Ori or Carmit signed in, a way to the admin from every screen; (2) a quiz sent on its own
(?q=<id>): the player sees only that quiz, never the start screen or the other quizzes.
Usage: python3 tools/test_entry_solo.py http://localhost:8766/app/   (a server rooted at the project folder)"""
import json
import sys
from playwright.sync_api import sync_playwright

import fake_firebase
from fake_firebase import stub, check

BASE = sys.argv[1]


def finish(page):
    page.click("#start-btn")
    while not page.is_visible("#report-screen"):
        page.click("#skip-btn")


with sync_playwright() as p:
    b = p.chromium.launch()

    # Signed out.
    ctx = b.new_context(viewport={"width": 420, "height": 900})
    ctx.add_init_script("window.__NO_USER = true; window.__PATH = ['qe', 'qm'];")
    page = ctx.new_page()
    stub(page)
    page.goto(BASE + "index.html")
    page.wait_for_selector(".gallery-card")
    page.wait_for_timeout(200)
    check(page.is_visible("#admin-btn") and "כניסה לניהול" in page.text_content("#admin-btn"), "start screen: one clear 'enter the admin' button")
    check(page.locator("a[href='admin.html']").count() == 1, "start screen: that is the only link to the admin")
    check(page.locator("footer").count() == 0, "no footer link anymore")
    page.click("a.gallery-card.next")
    page.wait_for_url("**?p=qe")
    page.wait_for_selector("#start-btn:visible")
    page.wait_for_timeout(200)
    check(page.is_hidden("#admin-btn"), "signed out: no admin button inside a quiz")
    check(page.is_visible("#home-btn"), "on the path: the way back to all quizzes")
    finish(page)
    check(page.is_hidden("#admin-btn"), "signed out: no admin button on the report")

    # A quiz sent on its own.
    page.goto(BASE + "index.html?q=qm")
    page.wait_for_selector("#start-btn:visible")
    page.wait_for_timeout(200)
    check(page.text_content("#quiz-name") == "בינוני", "solo: the quiz opens")
    check(page.is_hidden("#home-btn") and page.is_hidden("#admin-btn"), "solo: no way back to all quizzes, no admin button")
    check(page.is_hidden(".about") and page.is_hidden(".lede-top") and page.is_hidden("#gallery"), "solo: no start-screen content")
    check(page.get_attribute("#logo-link", "href") is None, "solo: the logo leads nowhere")
    finish(page)
    page.wait_for_timeout(300)
    check(page.is_hidden("#report-home-btn") and page.is_hidden("#next-level-btn"), "solo: the report leads nowhere else")
    check("סיימתם את כל" not in page.text_content("#score-sub"), "solo: no word about other quizzes")
    links = page.evaluate("[...document.querySelectorAll('a[href]')].filter(a => a.offsetParent).map(a => a.getAttribute('href'))")
    check(links == [], f"solo: no visible links at all: {links}")
    ctx.close()

    # Signed in as Ori: the way to the admin on every screen, the solo quiz too.
    ctx = b.new_context(viewport={"width": 420, "height": 900})
    ctx.add_init_script("window.__PATH = ['qe', 'qm'];")
    page = ctx.new_page()
    stub(page)
    for url in ["index.html", "index.html?p=qe", "index.html?q=qm"]:
        page.goto(BASE + url)
        page.wait_for_selector("#admin-btn:visible")
        check("לממשק הניהול" in page.text_content("#admin-btn"), f"signed in: the way to the admin on {url}")
    finish(page)
    check(page.is_visible("#admin-btn"), "signed in: and on the report")

    # A note sent along: typed in the admin, kept in the quiz under a short id, shown at the top of the quiz.
    admin = ctx.new_page()
    stub(admin)
    shared = []
    admin.on("dialog", lambda d: shared.append(d.default_value))  # no clipboard in headless: the link comes in a prompt
    admin.goto(BASE + "admin.html#quiz/qm")
    admin.wait_for_selector("#quiz-list")
    admin.fill("#share-note", "היי דנה,\nחידון קטן בשבילך")
    admin.click("#share-btn")
    admin.wait_for_timeout(300)
    notes = admin.evaluate("window.__store.quizzes.qm.notes")
    check(isinstance(notes, dict) and list(notes.values()) == ["היי דנה,\nחידון קטן בשבילך"], f"admin: the note is kept in the quiz: {notes}")
    nid = list(notes)[0]
    admin.click("#share-btn")
    admin.wait_for_timeout(300)
    check(len(admin.evaluate("window.__store.quizzes.qm.notes")) == 1, "admin: the same note again reuses its id")
    url = shared[-1] if shared else admin.evaluate("navigator.clipboard.readText()")
    check(url.endswith(f"index.html?q=qm&n={nid}"), f"admin: the link carries the note id: {url}")
    # Each fake page has its own store: hand the player what the admin saved.
    page = ctx.new_page()
    page.add_init_script(f"window.__PATCH = {{ qm: {{ notes: {json.dumps(notes)} }} }};")
    stub(page)
    page.goto(BASE + f"index.html?q=qm&n={nid}")
    page.wait_for_selector("#quiz-note:visible")
    check(page.inner_text("#quiz-note") == "היי דנה,\nחידון קטן בשבילך", "player: the note shows, line breaks kept")
    check(page.locator("#quiz-note").bounding_box()["y"] < page.locator("#quiz-name").bounding_box()["y"], "player: the note is at the top of the quiz")
    page.goto(BASE + "index.html?q=qm")
    page.wait_for_selector("#start-btn:visible")
    check(page.is_hidden("#quiz-note"), "player: no note without one in the link")
    b.close()

print("ALL PASS" if fake_firebase.ok else "SOME FAILED")
sys.exit(0 if fake_firebase.ok else 1)
