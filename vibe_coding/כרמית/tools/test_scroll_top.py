"""Every screen opens at its top: the next riddle and the results, even when the player scrolled down to press
'next', and the page after a reload. Usage: python3 tools/test_scroll_top.py http://localhost:8766/app/"""
import sys
from playwright.sync_api import sync_playwright

import fake_firebase
from fake_firebase import stub, check

BASE = sys.argv[1]
y = lambda page: page.evaluate("scrollY")
bottom = lambda page: page.evaluate("scrollTo(0, document.body.scrollHeight)")

with sync_playwright() as p:
    b = p.chromium.launch()
    page = b.new_page(viewport={"width": 360, "height": 420})  # small, so every screen scrolls
    stub(page)
    page.goto(BASE + "index.html?q=qe")  # two riddles: ANS1, ANS2
    page.wait_for_selector("#start-btn:visible")
    page.click("#start-btn")

    page.fill("#answer-input", "ANS1")
    page.click("#submit-btn")
    bottom(page)
    check(y(page) > 0, f"the riddle screen scrolls: {y(page)}")
    page.click("#next-btn")
    page.wait_for_timeout(100)
    check(y(page) == 0, f"the next riddle opens at the top: {y(page)}")

    page.fill("#answer-input", "ANS2")
    page.click("#submit-btn")
    bottom(page)
    page.click("#next-btn")
    page.wait_for_selector("#report-screen:visible")
    page.wait_for_timeout(100)
    check(y(page) == 0, f"the results open at the top: {y(page)}")

    bottom(page)
    page.reload()
    page.wait_for_selector("#start-btn:visible")
    page.wait_for_timeout(300)
    check(y(page) == 0, f"a reloaded page opens at the top: {y(page)}")
    b.close()

print("ALL PASS" if fake_firebase.ok else "SOME FAILED")
