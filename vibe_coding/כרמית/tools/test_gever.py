"""The "גבר גבר" tab: writing a request to Claude, a screenshot with it, Claude's answer showing under the request,
the count of new answers on the tab, continuing a request, and withdrawing one Claude has not seen yet.
Usage: python3 tools/test_gever.py http://localhost:8767/vibe_coding/כרמית/app/ [screenshot.png]"""
import base64
import sys
import tempfile
from playwright.sync_api import sync_playwright

import fake_firebase
from fake_firebase import stub, check

BASE = sys.argv[1]
SHOT = sys.argv[2] if len(sys.argv) > 2 else None

GEVER = {
    "g1": {"by": "carmitconsari@gmail.com", "text": "הכפתור של הרמז קטן מדי בטלפון", "image": "", "followUp": "",
           "status": "done", "created": 1000, "reply": "הגדלתי את כפתור הרמז בטלפון.", "replied": 2000},
    "g2": {"by": "carmitconsari@gmail.com", "text": "אפשר צבע אחר לכותרת?", "image": "", "followUp": "",
           "status": "question", "created": 3000, "reply": "איזה צבע? ורוד או תכלת?", "replied": 4000},
}

with sync_playwright() as p:
    b = p.chromium.launch()
    page = b.new_page(viewport={"width": 1100, "height": 900})
    page.add_init_script(f"window.__GEVER = {__import__('json').dumps(GEVER)};")
    stub(page)
    page.goto(BASE + "admin.html")
    page.wait_for_selector("#home-view:visible")
    check(page.text_content("#gever-count").strip() == "2 חדש", f"the tab counts answers not seen yet: {page.text_content('#gever-count')!r}")

    page.click(".tab[data-tab=gever]")
    page.wait_for_selector("#tab-gever:visible")
    check(page.text_content("#gever-count").strip() == "", "opening the tab clears the count")
    items = page.locator(".gever-item")
    check(items.count() == 2, "both requests show")
    check("איזה צבע" in items.nth(0).inner_text(), "the newest thread first, with Claude's answer")
    check("לקלוד יש שאלה" in items.nth(0).inner_text() and "לענות לקלוד" in items.nth(0).inner_text(), "a question asks for an answer")
    check("✓ טופלה" in items.nth(1).inner_text() and "הגדלתי" in items.nth(1).inner_text(), "a done request shows its answer")

    # Empty: an error, nothing sent.
    page.click("#gever-send")
    check(page.is_visible("#gever-error"), "an empty request is not sent")

    # A new request with a screenshot.
    png = tempfile.NamedTemporaryFile(suffix=".png", delete=False)  # a real 1x1 PNG: the canvas needs a decodable image
    png.write(base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="))
    png.close()
    page.set_input_files("#gever-file", png.name)
    page.wait_for_selector("#gever-img-box:visible")
    page.fill("#gever-text", "שהתוצאות יראו גם כמה זמן לקח")
    page.click("#gever-send")
    page.wait_for_function("Object.values(window.__store.gever).length === 3")
    new = page.evaluate("Object.values(window.__store.gever).find((g) => g.text.startsWith('שהתוצאות'))")
    check(new["status"] == "new" and new["by"] == "orimosenzon@gmail.com" and new["followUp"] == "", f"saved as a new request: {new}")
    check(new["image"].startswith("data:image/jpeg"), "with the screenshot")
    check(sorted(new) == ["by", "created", "followUp", "image", "status", "text"], f"only the fields the rules allow: {sorted(new)}")
    check(page.input_value("#gever-text") == "" and page.is_hidden("#gever-img-box"), "the form is cleared")
    first = page.locator(".gever-item").nth(0).inner_text()
    check("שהתוצאות" in first and "ממתינה לקלוד" in first, "the new request shows first, waiting")
    if SHOT:
        page.screenshot(path=SHOT, full_page=True)

    # Answering Claude's question: the answer is a follow-up, shown under the question.
    page.locator(".gever-item", has_text="איזה צבע").get_by_text("לענות לקלוד").click()
    check(page.is_visible("#gever-follow") and "אפשר צבע" in page.text_content("#gever-follow"), "the form says what it continues")
    page.fill("#gever-text", "ורוד")
    page.keyboard.press("Control+Enter")
    page.wait_for_function("Object.values(window.__store.gever).length === 4")
    follow = page.evaluate("Object.values(window.__store.gever).find((g) => g.text === 'ורוד')")
    check(follow["followUp"] == "g2", "the answer points to the question")
    order = page.locator(".gever-item").all_inner_texts()
    check("אפשר צבע" in order[0] and "ורוד" in order[1], "the thread with the answer is now first, the answer under the question")
    check("followup" in page.locator(".gever-item").nth(1).get_attribute("class"), "the answer is indented")

    # Withdrawing: only while new.
    check(page.locator(".gever-item", has_text="הגדלתי").get_by_text("ביטול הבקשה").count() == 0, "a handled request can't be withdrawn")
    page.locator(".gever-item", has_text="שהתוצאות").get_by_text("ביטול הבקשה").click()
    page.wait_for_function("Object.values(window.__store.gever).length === 3")
    check(page.locator(".gever-item", has_text="שהתוצאות").count() == 0, "a withdrawn request is gone")

    page.goto(BASE + "admin.html")
    page.wait_for_selector("#tab-quizzes:visible")
    page.goto(BASE + "admin.html#gever")  # a fresh page: the fake store starts again from the two above
    page.reload()
    page.wait_for_selector("#tab-gever:visible")
    check(page.locator(".gever-item").count() == 2, f"#gever opens the tab: {page.locator('.gever-item').count()}")
    b.close()

print("ALL PASS" if fake_firebase.ok else "SOME FAILED")
