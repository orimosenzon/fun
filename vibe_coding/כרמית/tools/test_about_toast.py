"""Two requests from Carmit (25/11/2026): (1) a quiz can show "behind the riddles" in the link sent on its own
(quiz.showAbout, a switch in the quiz screen, off by default); (2) the notes the admin shows after a click stay long
enough to read, stay while the pointer is on them, and close on a click.
Usage: python3 tools/test_about_toast.py http://localhost:8767/vibe_coding/כרמית/app/ [shots-dir]"""
import sys
from playwright.sync_api import sync_playwright

import fake_firebase
from fake_firebase import stub, check

BASE = sys.argv[1]
SHOTS = sys.argv[2] if len(sys.argv) > 2 else None

with sync_playwright() as p:
    b = p.chromium.launch()

    # Admin: the switch saves, and starts off.
    page = b.new_page(viewport={"width": 1100, "height": 900})
    stub(page)
    page.goto(BASE + "admin.html#quiz/qe")
    page.wait_for_selector("#quiz-view:visible")
    check(page.is_checked("input[name='quiz-about'][value='no']"), "the switch starts off")
    page.click("input[name='quiz-about'][value='yes'] + span")
    page.wait_for_timeout(200)
    check(page.evaluate("window.__store.quizzes.qe.showAbout") is True, "switching it on saves showAbout")
    if SHOTS:
        page.locator("input[name='quiz-about']").first.scroll_into_view_if_needed()
        page.screenshot(path=f"{SHOTS}/about_switch.png")

    # The note after locking the quiz: still there after 3 seconds (it used to go after 2.2).
    page.click("#quiz-lock")
    page.wait_for_selector("#toast.show")
    page.wait_for_timeout(3000)
    check(page.is_visible("#toast.show"), "a note is still showing after 3 seconds")
    page.hover("#toast")
    page.wait_for_timeout(9000)
    check(page.locator("#toast.show").count() == 1, "it stays while the pointer is on it")
    page.click("#toast")
    page.wait_for_timeout(100)
    check(page.locator("#toast.show").count() == 0, "a click closes it")
    page.close()

    # Player: the link sent on its own shows "behind the riddles" only when switched on.
    for on in (False, True):
        ctx = b.new_context(viewport={"width": 420, "height": 900})
        ctx.add_init_script("window.__NO_USER = true;" + ("window.__PATCH = { qe: { showAbout: true } };" if on else ""))
        page = ctx.new_page()
        stub(page)
        page.goto(BASE + "index.html?q=qe")
        page.wait_for_selector("#start-btn:visible")
        page.wait_for_timeout(200)
        check(page.is_visible(".about-sketch") == on, f"solo link, switch {'on' if on else 'off'}: behind the riddles {'shown' if on else 'hidden'}")
        if on and SHOTS:
            page.locator(".about").scroll_into_view_if_needed()
            page.screenshot(path=f"{SHOTS}/about_solo.png")
        ctx.close()
    b.close()

sys.exit(0 if fake_firebase.ok else 1)
