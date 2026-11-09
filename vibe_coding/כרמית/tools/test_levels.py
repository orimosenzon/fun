"""Drives the player (index.html) and admin.html with stubbed Firebase to test difficulty levels:
the gallery groups quizzes by level, medium and hard stay locked until the level below is finished,
progress lasts only for the visit, and the admin can set a quiz's level.
Usage: python3 tools/test_levels.py http://localhost:8766/app/   (a server rooted at the project folder)"""
import sys
from playwright.sync_api import sync_playwright

BASE = sys.argv[1]
from fake_firebase import stub, check
import fake_firebase


def play_through(page, answer):
    page.click("#start-btn")
    while True:
        page.fill("#answer-input", answer)
        page.click("#submit-btn")
        page.wait_for_selector("#next-btn:visible")
        page.click("#next-btn")
        if page.is_visible("#report-screen"):
            break


with sync_playwright() as p:
    b = p.chromium.launch()
    ctx = b.new_context(viewport={"width": 420, "height": 900})
    page = ctx.new_page()
    stub(page)

    page.goto(BASE + "index.html")
    page.wait_for_selector(".level")
    heads = page.locator(".level-head").all_text_contents()
    check(len(heads) == 3 and "קלה" in heads[0] and "בינונית" in heads[1] and "קשה" in heads[2], f"three levels in order: {heads}")
    check(page.locator("#level-easy .gallery-card").count() == 2, "a quiz without a level counts as easy")
    check(page.locator("#level-medium.locked").count() == 1 and page.locator("#level-hard.locked").count() == 1, "medium and hard start locked")
    check(page.locator("#level-medium a.gallery-card").count() == 0, "a locked quiz is not a link")
    about_top = page.locator(".about").bounding_box()["y"] < page.locator("#gallery").bounding_box()["y"]
    check(about_top, "the about sits above the gallery")
    check(page.is_visible(".motif"), "the Hebrew x English line shows in the header")
    check(page.locator("body").evaluate("() => document.documentElement.scrollWidth <= innerWidth"), "no sideways scroll at phone width")

    # Finish the easy quiz, even with wrong answers: no threshold.
    page.click("#level-easy a.gallery-card >> nth=0")
    page.wait_for_url("**?q=*")
    page.wait_for_selector("#start-btn:visible")
    check("קלה" in page.text_content("#quiz-level"), f"the quiz page shows its level: {page.text_content('#quiz-level')!r} {page.url}")
    check(page.is_visible(".motif"), "the Hebrew x English line shows on the quiz page")
    play_through(page, "wrong")
    check(page.is_hidden("text=סיבוב נוסף"), "no 'another round' button")
    page.wait_for_selector("#next-level-btn:visible")
    check("בינונית" in page.text_content("#next-level-btn"), "the report invites to the medium level")
    page.click("#next-level-btn")  # one medium quiz: straight into it
    page.wait_for_url("**?q=qm")
    page.wait_for_selector("#start-btn:visible")
    check("בינוני" == page.text_content("#quiz-name"), "the next-level button opens the medium quiz")
    play_through(page, "ANS3")
    page.wait_for_selector("#next-level-btn:visible")
    page.click("#report-home-btn")
    page.wait_for_selector(".level")
    check(page.locator(".level.locked").count() == 0, "after easy and medium, everything is open")

    # A new visit (new tab, fresh sessionStorage) starts again from easy.
    page2 = ctx.new_page()
    stub(page2)
    page2.goto(BASE + "index.html")
    page2.wait_for_selector(".level")
    check(page2.locator("#level-medium.locked").count() == 1, "a new visit starts again from easy")

    # Admin: the level picker shows the quiz's level and saves a change.
    admin = ctx.new_page()
    stub(admin)
    admin.goto(BASE + "admin.html#quiz/qold")
    admin.wait_for_selector("#quiz-list")
    check(admin.is_checked('input[name="quiz-level"][value="easy"]'), "admin: an old quiz shows as easy")
    admin.click('label.seg:has(input[name="quiz-level"][value="hard"])')
    admin.wait_for_timeout(200)
    check(admin.evaluate("window.__store.quizzes.qold.level") == "hard", "admin: choosing hard saves level=hard")
    b.close()

print("ALL PASS" if fake_firebase.ok else "SOME FAILED")
sys.exit(0 if fake_firebase.ok else 1)
