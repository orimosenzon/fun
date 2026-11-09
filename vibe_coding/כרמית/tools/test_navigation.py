"""Drives one quiz with stubbed Firebase to test moving between riddles: skip, back to the previous riddle,
an answered riddle shown as it was (no second try), a skipped one answerable on return, and the report.
Usage: python3 tools/test_navigation.py http://localhost:8766/app/   (a server rooted at the project folder)"""
import sys
from playwright.sync_api import sync_playwright

import fake_firebase
from fake_firebase import stub, check

BASE = sys.argv[1]

with sync_playwright() as p:
    b = p.chromium.launch()
    page = b.new_page(viewport={"width": 420, "height": 900})
    stub(page)
    page.goto(BASE + "index.html?q=qe")  # two riddles: ANS1, ANS2
    page.wait_for_selector("#start-btn:visible")
    page.click("#start-btn")

    check(page.is_hidden("#prev-btn") or page.is_disabled("#prev-btn"), "no 'previous' on the first riddle")
    check(page.is_visible("#skip-btn"), "'skip' shows on an open riddle")
    page.click("#skip-btn")
    check(page.text_content("#progress-text").startswith("02"), "skip moves to riddle 2")
    check("לתוצאות" in page.text_content("#skip-btn"), "on the last riddle, skip leads to the results")

    page.click("#prev-btn")
    check(page.text_content("#progress-text").startswith("01"), "back returns to riddle 1")
    check(page.is_enabled("#answer-input") and page.input_value("#answer-input") == "", "a skipped riddle can be answered on return")
    page.fill("#answer-input", "ANS1")
    page.click("#submit-btn")
    check("correct" in page.get_attribute("#feedback", "class"), "answering the skipped riddle works")
    check(page.is_hidden("#skip-btn"), "no skip once answered")
    page.click("#next-btn")

    page.fill("#answer-input", "nope")
    page.click("#submit-btn")
    page.click("#prev-btn")
    check(page.input_value("#answer-input") == "ANS1" and page.is_disabled("#answer-input"), "an answered riddle comes back as it was, locked")
    check(page.is_visible("#feedback") and page.is_visible("#next-btn"), "its feedback and 'next' show again")
    page.click("#next-btn")
    check(page.input_value("#answer-input") == "nope" and page.is_disabled("#answer-input"), "riddle 2 also comes back as it was")
    page.click("#next-btn")

    page.wait_for_selector("#report-screen:visible")
    check(page.text_content("#score-headline").endswith("1/2"), f"score counts each riddle once: {page.text_content('#score-headline')!r}")
    check(page.locator("#report-table tbody tr").count() == 2, "two rows in the report")

    # Skipping to the end: a skipped riddle shows as skipped in the report.
    page.goto(BASE + "index.html?q=qe")
    page.wait_for_selector("#start-btn:visible")
    page.click("#start-btn")
    page.click("#skip-btn")
    page.click("#skip-btn")
    page.wait_for_selector("#report-screen:visible")
    rows = page.locator("#report-table tbody tr").all_inner_texts()
    check(len(rows) == 2 and all("דילגתם" in r for r in rows), "skipped riddles show as skipped in the report")
    check(page.text_content("#score-headline").endswith("0/2"), "skipping everything scores 0/2")
    b.close()

print("ALL PASS" if fake_firebase.ok else "SOME FAILED")
sys.exit(0 if fake_firebase.ok else 1)
