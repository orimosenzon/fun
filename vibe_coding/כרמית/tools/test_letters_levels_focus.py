"""Drives admin.html and the player with stubbed Firebase to test (1) letters: written on their own, marked inside
a quiz as the letter before it (רישא) or after it (סיפא), and sent by the quiz's one share link; (2) a quiz without a level, for
quizzes that mix difficulties; (3) on a phone the answer box is not focused on a new riddle, so the keyboard
doesn't cover the picture; with a real keyboard it still is.
Usage: python3 tools/test_letters_levels_focus.py http://localhost:8766/app/   (a server rooted at the project folder)"""
import json
import sys
from playwright.sync_api import sync_playwright

import fake_firebase
from fake_firebase import stub, check

BASE = sys.argv[1]


def finish_quiz(page):
    page.click("#start-btn")
    while not page.is_visible("#report-screen"):
        page.click("#skip-btn")


with sync_playwright() as p:
    b = p.chromium.launch()
    ctx = b.new_context(viewport={"width": 1200, "height": 1000})
    copied = []
    shorts = []

    # ---- letters: written on their own, marked before/after inside a quiz, one share link ----
    admin = ctx.new_page()
    stub(admin)
    # Catch what the share button copies.
    admin.add_init_script("""
      navigator.clipboard.writeText = async (t) => { window.__copied = t; };
      navigator.clipboard.write = async (items) => { window.__copied = await (await items[0].getType('text/plain')).text(); };
    """)
    admin.goto(BASE + "admin.html#letters")
    admin.wait_for_selector("#tab-letters:visible")

    def write_letter(name, text, link=""):
        admin.click("#letter-new-home" if admin.is_visible("#letter-new-home") else "#letter-new")
        admin.fill("#letter-name", name)
        admin.fill("#letter-text", text)
        admin.fill("#letter-link-url", link)
        admin.click("#letter-save")
        admin.wait_for_timeout(300)

    def share():
        """Copies the share link: a short link; appends where its code leads, as the old long form, to `copied`."""
        admin.evaluate("window.__copied = null")
        admin.click("#share-btn")
        admin.wait_for_function("window.__copied")
        short = admin.evaluate("window.__copied")
        check("/carmit/?" in short, f"admin: a short link: {short}")
        code = short.split("?")[1]
        link = admin.evaluate(f"window.__store.links['{code}']")
        url = BASE + f"index.html?q={link['q']}" + (f"&b={link['b']}" if link["b"] else "") + (f"&a={link['a']}" if link["a"] else "")
        copied.append(url)
        shorts.append(short)

    admin.click("#letter-new-home")
    admin.click("#letter-save")
    check(admin.is_visible("#letter-error"), "admin: an empty letter is refused")
    admin.click("#letter-cancel")
    write_letter("פתיחה", "שלום מלפני", "example.com/w")
    write_letter("סגירה", "להתראות מאחרי")
    write_letter("עוד אחד", "סתם מכתב")
    letters = admin.evaluate("window.__store.letters")
    by = {l["name"]: lid for lid, l in letters.items()}
    check(len(by) == 3 and all("quizId" not in l for l in letters.values()), f"admin: three letters, none tied to a quiz: {list(by)}")
    check(admin.locator("#all-letters li").count() == 3, "admin: the letters tab lists them")
    check(admin.locator("#all-letters button", has_text="העתקת").count() == 0, "admin: no links in the letters tab")

    admin.goto(BASE + "admin.html#quiz/qe")
    admin.wait_for_selector("#quiz-view:visible")
    check(admin.locator("#letters-list li").count() == 3, "admin: the quiz lists every letter")
    check("בלי מכתבים" in admin.inner_text("#share-with"), "admin: nothing marked, the link goes without letters")
    share()
    check(copied[-1].endswith("index.html?q=qe"), f"admin: plain link: {copied[-1]}")

    def mark(name, role):
        admin.locator(f'#letters-list li[data-id="{by[name]}"] label.seg:has(input[value="{role}"])').click()
        admin.wait_for_timeout(200)

    mark("פתיחה", "opening")
    mark("סגירה", "closing")
    q = admin.evaluate("window.__store.quizzes.qe")
    check(q.get("opening") == by["פתיחה"] and q.get("closing") == by["סגירה"], f"admin: the choice is saved in the quiz: {q}")
    check("פתיחה" in admin.inner_text("#share-with") and "סגירה" in admin.inner_text("#share-with"), "admin: the share line names both")
    # Marking another letter "before" takes the place.
    mark("עוד אחד", "opening")
    q = admin.evaluate("window.__store.quizzes.qe")
    check(q["opening"] == by["עוד אחד"] and admin.locator("#letters-list li.picked").count() == 2, "admin: one letter per place")
    mark("פתיחה", "opening")
    share()
    both = copied[-1]
    check(f"b={by['פתיחה']}" in both and f"a={by['סגירה']}" in both, f"admin: the one link carries both letters: {both}")
    check(admin.get_attribute("#play-link", "href").endswith(both.split("/app/")[1]), "admin: 'play it' opens the same quiz and letters")
    n_links = len(admin.evaluate("window.__store.links"))
    share()
    check(shorts[-1] == shorts[-2] and len(admin.evaluate("window.__store.links")) == n_links,
          "admin: the same choice gets the same short link again")
    mark("סגירה", "none")
    share()
    before_only = copied[-1]
    check("a=" not in before_only and "b=" in before_only, f"admin: before only: {before_only}")
    mark("פתיחה", "none")
    mark("סגירה", "closing")
    share()
    after_only = copied[-1]
    check("b=" not in after_only and "a=" in after_only, f"admin: after only: {after_only}")

    letters = admin.evaluate("window.__store.letters")
    page = ctx.new_page()
    page.add_init_script(f"window.__NO_USER = true; window.__LETTERS = {json.dumps(letters)};")
    stub(page)

    def play(url):
        page.goto(url)
        page.wait_for_selector("#start-btn:visible")

    play(both)
    check(page.inner_text("#letter-text") == "שלום מלפני" and page.is_visible("#letter-cta-top"), "player: the letter before, with its button")
    check(page.is_hidden("#letter-after"), "player: the letter after waits for the end")
    check(page.is_hidden("#gallery") and page.is_hidden("#home-btn"), "player: only this quiz, nothing else")
    finish_quiz(page)
    check(page.inner_text("#letter-after-text") == "להתראות מאחרי", "player: the letter after, with the results")
    check(page.is_visible("#letter-cta") and page.get_attribute("#letter-cta", "href") == "https://example.com/w",
          "player: the button of the letter before comes back at the end")
    page.screenshot(path="/tmp/claude-1000/letter_after.png", full_page=True)

    play(after_only)
    check(page.is_hidden("#letter"), "player: after only, nothing above the quiz")
    finish_quiz(page)
    check(page.is_visible("#letter-after") and page.is_hidden("#letter-end"), "player: the letter after, no button")

    # The short page forwards to the quiz with its letters (Firestore's REST answer faked).
    links = admin.evaluate("window.__store.links")
    both_code = [c for c, l in links.items() if l["b"] and l["a"]][0]
    page.route("**/firestore.googleapis.com/**", lambda r: r.fulfill(json={"fields": {k: {"stringValue": v} for k, v in links[r.request.url.split("/links/")[1].split("?")[0]].items() if k in "qba"}}))
    page.goto(shorts[0].split("?")[0] + f"?{both_code}")
    page.wait_for_selector("#start-btn:visible")
    check(page.inner_text("#letter-text") == "שלום מלפני", f"short page: forwarded to the quiz with its letter: {page.url}")
    meta = page.request.get(shorts[0].split("?")[0]).text()
    check('og:title" content="חידות כרמית כונסרי"' in meta and "logo.jpg" in meta, "short page: preview title and logo")

    play(before_only)
    finish_quiz(page)
    check(page.is_hidden("#letter-after") and page.is_visible("#letter-cta"), "player: before only, its button at the end")

    play(BASE + "index.html?q=qe")
    check(page.is_hidden("#letter"), "player: a plain link, no letter")
    finish_quiz(page)
    check(page.is_hidden("#letter-after") and page.is_hidden("#letter-end"), "player: a plain link, nothing after")

    play(BASE + f"index.html?q=qe&m={by['פתיחה']}")
    check(page.inner_text("#letter-text") == "שלום מלפני", "player: an old m= link still shows its letter before")
    play(BASE + "index.html?q=qe&b=nosuch&a=nosuch")
    check(page.is_hidden("#letter"), "player: deleted letters just leave the quiz")

    # Deleting from inside the quiz, and from the edit dialog.
    admin.goto(BASE + "admin.html#quiz/qe")
    admin.wait_for_selector("#letters-list li")
    admin.locator(f'#letters-list li[data-id="{by["עוד אחד"]}"] button.del').click()
    admin.wait_for_timeout(300)
    check(by["עוד אחד"] not in admin.evaluate("window.__store.letters") and admin.locator("#letters-list li").count() == 2,
          "admin: a letter deleted from inside the quiz")
    admin.locator(f'#letters-list li[data-id="{by["פתיחה"]}"] button', has_text="עריכה").click()
    check(admin.is_visible("#letter-delete"), "admin: the edit dialog offers deleting")
    admin.click("#letter-delete")
    admin.wait_for_timeout(300)
    check(by["פתיחה"] not in admin.evaluate("window.__store.letters") and not admin.is_visible("#letter-dialog"),
          "admin: deleted from the dialog, which closes")
    admin.click("#letter-new")
    check(admin.is_hidden("#letter-delete"), "admin: a new letter has nothing to delete")
    admin.click("#letter-cancel")

    # Deleting a letter clears it from the quiz that had it marked.
    admin.goto(BASE + "admin.html#letters")
    admin.wait_for_selector("#tab-letters:visible")
    check("מסומן ב: קל (אחרי)" in admin.inner_text("#all-letters"), "admin: the letters tab says where a letter is marked")
    admin.locator(f'#all-letters li[data-id="{by["סגירה"]}"] button.del').click()
    admin.wait_for_timeout(300)
    check(by["סגירה"] not in admin.evaluate("window.__store.letters") and admin.evaluate("window.__store.quizzes.qe.closing") == "",
          "admin: deleted, and the quiz no longer points to it")

    # ---- no level ----
    admin.goto(BASE + "admin.html#quiz/qm")
    admin.wait_for_selector("#quiz-view:visible")
    admin.click('label.seg:has(input[value="none"])')
    admin.wait_for_timeout(500)
    check(admin.evaluate("window.__store.quizzes.qm.level") == "none", "admin: 'no level' saved")
    admin.goto(BASE + "admin.html#quiz/qm")
    admin.wait_for_selector("#quiz-view:visible")
    check(admin.is_checked('input[name="quiz-level"][value="none"]'), "admin: 'no level' shown after reload")
    # A fresh page loses the fake store, so it starts from what was saved.
    lists = ctx.new_page()
    lists.add_init_script("window.__PATCH = { qm: { level: 'none' } };")
    stub(lists)
    lists.goto(BASE + "admin.html")
    lists.wait_for_selector("#quiz-list .q-item, .quiz-card, .q-card")
    lists.wait_for_timeout(300)
    body = lists.inner_text("body")
    check("בינוני" in body and "רמה בינונית" not in body, "admin: no level label on that quiz in the lists")

    page = ctx.new_page()
    page.add_init_script(f"window.__NO_USER = true; window.__PATCH = {{ qm: {{ level: 'none' }} }};")
    stub(page)
    page.goto(BASE + "index.html?q=qm")
    page.wait_for_selector("#start-btn:visible")
    check(page.is_hidden("#quiz-level"), "player: no level badge")
    page.goto(BASE + "index.html")
    page.wait_for_selector(".gallery-card")
    cards = page.locator(".gallery-card").all_inner_texts()
    mixed = [c for c in cards if "בינוני" in c]
    check(mixed and "רמה" not in mixed[0], f"player: the gallery card has no level: {mixed}")
    check("בינוני" in cards[-1], f"player: without a set path, the no-level quiz comes last: {cards}")

    # ---- focus: desktop vs phone ----
    page.goto(BASE + "index.html?q=qe")
    page.wait_for_selector("#start-btn:visible")
    page.click("#start-btn")
    check(page.evaluate("document.activeElement.id") == "answer-input", "desktop: the answer box is focused")
    ctx.close()

    phone = b.new_context(viewport={"width": 390, "height": 844}, is_mobile=True, has_touch=True)
    pg = phone.new_page()
    pg.add_init_script("window.__NO_USER = true;")
    stub(pg)
    pg.goto(BASE + "index.html?q=qe")
    pg.wait_for_selector("#start-btn:visible")
    pg.tap("#start-btn")
    check(pg.evaluate("document.activeElement.id") != "answer-input", "phone: first riddle, the box is not focused")
    pg.tap("#answer-input")
    check(pg.evaluate("document.activeElement.id") == "answer-input", "phone: tapping the box focuses it")
    pg.fill("#answer-input", "משהו")
    pg.press("#answer-input", "Enter")
    check(pg.is_visible("#feedback") and pg.is_visible("#next-btn"), "phone: Enter answers and stays to show the result")
    pg.tap("#next-btn")
    check(pg.evaluate("document.activeElement.id") != "answer-input", "phone: next riddle, the box is not focused")
    # A phone that claims hover and a fine pointer (Samsung, newer Android): the tap itself decides.
    odd = b.new_context(viewport={"width": 412, "height": 900}, has_touch=True)
    pg = odd.new_page()
    # Answer every media query about pointers the way such a phone does: hover, fine pointer.
    pg.add_init_script("""window.__NO_USER = true;
      const real = window.matchMedia.bind(window);
      window.matchMedia = (q) => /hover|pointer/.test(q)
        ? { matches: /hover: hover|pointer: fine/.test(q), media: q, addEventListener() {}, removeEventListener() {} } : real(q);""")
    stub(pg)
    pg.goto(BASE + "index.html?q=qe")
    pg.wait_for_selector("#start-btn:visible")
    check(pg.evaluate("matchMedia('(hover: hover)').matches"), "odd phone: the browser claims hover")
    pg.tap("#start-btn")
    check(pg.evaluate("document.activeElement.id") != "answer-input", "odd phone: started with a finger, the box is not focused")
    pg.tap("#skip-btn")
    check(pg.evaluate("document.activeElement.id") != "answer-input", "odd phone: next riddle, still not focused")
    pg.click("#prev-btn")
    check(pg.evaluate("document.activeElement.id") == "answer-input", "odd phone: with a mouse click, focused again")
    b.close()

print("ALL PASS" if fake_firebase.ok else "SOME FAILED")
sys.exit(0 if fake_firebase.ok else 1)
