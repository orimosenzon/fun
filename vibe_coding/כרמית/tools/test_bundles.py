"""Drives admin.html and the player with stubbed Firebase to test bundles (מקבצים): collections of quizzes with a
link of their own. Admin: create, rename, add / remove / order quizzes, serial or not, the link to send, delete,
and a deleted quiz leaving its bundles. Player: a serial bundle opens one quiz after the other with the rest
dimmed and locked, a bundle that is not serial has all open, and "home" from a quiz leads back to the bundle.
Usage: python3 tools/test_bundles.py http://localhost:8767/vibe_coding/כרמית/app/ [screenshot-dir]
(with a server rooted at the fun repo the short page fun/carmit/ is tested too)"""
import sys
from playwright.sync_api import sync_playwright

import fake_firebase
from fake_firebase import stub, check

BASE = sys.argv[1]
SHOTS = sys.argv[2] if len(sys.argv) > 2 else None
BUNDLES = """window.__BUNDLES = {
  b1: { title: 'מקבץ סדרתי', quizIds: ['qe', 'qm', 'gone', 'qh'], serial: true, created: 2 },
  b2: { title: 'מקבץ פתוח', quizIds: ['qh', 'qe'], serial: false, created: 1 } };"""

with sync_playwright() as p:
    b = p.chromium.launch()

    # ================= admin =================
    ctx = b.new_context(viewport={"width": 1200, "height": 1000})
    a = ctx.new_page()
    stub(a)
    store = lambda expr: a.evaluate(f"(() => {{ const s = window.__store; return {expr}; }})()")
    a.goto(BASE + "admin.html#bundles")
    a.wait_for_selector("#tab-bundles:visible #bundle-new")
    check(a.locator("#bundle-grid .bundle-card").count() == 1, "tab: no bundles yet, only the 'new' card")

    # ---- create ----
    a.click("#bundle-new")
    a.wait_for_function("location.hash.startsWith('#bundle/new')")
    bid = a.evaluate("location.hash.slice(8)")
    a.wait_for_selector("#bundle-view:visible")
    made = store(f"s.bundles['{bid}']")
    check(made["title"] == "מקבץ חדש" and made["quizIds"] == [] and made["serial"] is False, f"create: an empty bundle, not serial: {made}")
    check(a.is_checked("input[name=bundle-serial][value=no]") and a.locator("#bundle-out .q-item").count() == 4, "create: every quiz is offered")

    # ---- rename ----
    a.fill("#bundle-title", "חידוני החורף")
    a.wait_for_function(f"window.__store.bundles['{bid}'].title === 'חידוני החורף'")
    check(True, "rename: saved by itself")

    # ---- add, remove, order ----
    row = lambda box, name: a.locator(f"{box} .q-item").filter(has=a.locator("b", has_text=name)).first
    for name in ["קל", "בינוני", "קשה"]:
        row("#bundle-out", name).locator(".add").click()
        a.wait_for_timeout(80)
    ids = lambda: store(f"s.bundles['{bid}'].quizIds")
    check(ids() == ["qe", "qm", "qh"], f"add: in the order they were added: {ids()}")
    check(a.locator("#bundle-in .q-item").count() == 3 and a.locator("#bundle-out .q-item").count() == 1 and a.inner_text("#bundle-in-count") == "3",
          "add: the lists follow")
    row("#bundle-in", "בינוני").locator(".icon-btn").click()
    a.wait_for_timeout(80)
    check(ids() == ["qe", "qh"] and store("'qm' in s.quizzes"), "remove: out of the bundle, the quiz itself stays")
    row("#bundle-out", "בינוני").locator(".add").click()
    a.wait_for_timeout(80)
    check(ids() == ["qe", "qh", "qm"], f"add again: to the end: {ids()}")
    # Drag the last one to the top.
    a.evaluate("""() => {
      const list = document.querySelector('#bundle-in');
      const first = list.querySelector('.q-item').getBoundingClientRect();
      const dt = new DataTransfer();
      dt.setData('text/plain', 'quiz:qm');
      list.dispatchEvent(new DragEvent('drop', { dataTransfer: dt, clientY: first.top + 2, bubbles: true, cancelable: true }));
    }""")
    a.wait_for_timeout(80)
    check(ids() == ["qm", "qe", "qh"], f"order: a dropped quiz lands where it was dropped: {ids()}")

    # ---- serial ----
    a.click("label:has(input[name=bundle-serial][value=yes])")
    a.wait_for_timeout(80)
    check(store(f"s.bundles['{bid}'].serial") is True and "נעולים" in a.inner_text("#bundle-serial-help"), "serial: saved, and explained")
    if SHOTS:
        a.screenshot(path=f"{SHOTS}/bundle-edit.png", full_page=True)

    # ---- the link ----
    check(a.get_attribute("#bundle-play", "href").endswith(f"index.html?s={bid}"), "link: 'see as a player' opens the bundle")
    a.click("#bundle-share")
    a.wait_for_function("Object.keys(window.__store.links).length === 1")
    a.wait_for_timeout(300)
    a.click("#bundle-share")
    a.wait_for_timeout(300)
    links = store("Object.values(s.links)")
    check(len(links) == 1 and links[0]["s"] == bid and links[0]["q"] == "" and links[0]["b"] == "" and links[0]["a"] == "",
          f"link: one short code for the bundle, however often it is copied: {links}")
    code = store("Object.keys(s.links)[0]")

    # ---- the tab ----
    a.goto(BASE + "admin.html#bundles")   # a fresh load: the fake store starts over, so seed what was made
    ctx.close()
    ctx = b.new_context(viewport={"width": 1200, "height": 1000})
    a = ctx.new_page()
    stub(a)
    a.add_init_script(BUNDLES)
    a.goto(BASE + "admin.html#bundles")
    a.wait_for_selector("#bundle-grid a.bundle-card")
    cards = a.locator("#bundle-grid a.bundle-card").all_inner_texts()
    check(len(cards) == 2 and "מקבץ סדרתי" in cards[0] and "3 חידונים" in cards[0] and "סדרתי" in cards[0]
          and "2 חידונים" in cards[1] and "כולם פתוחים" in cards[1], f"tab: each bundle with its size and kind: {cards}")
    check(a.inner_text("#bundle-count") == "2", "tab: the count")
    if SHOTS:
        a.screenshot(path=f"{SHOTS}/bundle-tab.png")

    # ---- a deleted quiz leaves its bundles ----
    a.goto(BASE + "admin.html#quiz/qh")
    a.wait_for_function("document.querySelector('#quiz-title').value === 'קשה'")
    a.click("#quiz-delete")   # the confirm is accepted by the stub
    a.wait_for_timeout(200)
    check(store("s.bundles.b1.quizIds") == ["qe", "qm", "gone"] and store("s.bundles.b2.quizIds") == ["qe"], "deleted quiz: out of every bundle")

    # ---- delete a bundle ----
    a.goto(BASE + "admin.html#bundle/b2")
    a.wait_for_selector("#bundle-view:visible")
    a.click("#bundle-delete")   # the confirm is accepted by the stub
    a.wait_for_function("location.hash === '#bundles'")
    a.wait_for_selector("#tab-bundles:visible")
    check(store("Object.keys(s.bundles).join()") == "b1" and store("'qe' in s.quizzes") and a.locator("#bundle-grid a.bundle-card").count() == 1,
          "delete: the bundle is gone, its quizzes are not")
    # The other tabs still work as before.
    a.goto(BASE + "admin.html#path")
    a.wait_for_selector("#tab-path:visible")
    a.locator("#path-out .q-item").first.locator(".add").click()
    a.wait_for_timeout(100)
    check(a.locator("#path-list .q-item").count() == 1, "the path tab still adds a quiz (it shares the rows with bundles)")
    ctx.close()

    # ---- phone ----
    ctx = b.new_context(viewport={"width": 390, "height": 800})
    a = ctx.new_page()
    stub(a)
    a.add_init_script(BUNDLES)
    a.goto(BASE + "admin.html#bundle/b1")
    a.wait_for_selector("#bundle-view:visible")
    check(a.evaluate("document.documentElement.scrollWidth <= innerWidth"), "phone: the bundle screen is no wider than the screen")
    a.goto(BASE + "admin.html#bundles")
    a.wait_for_selector("#bundle-grid a.bundle-card")
    check(a.evaluate("document.documentElement.scrollWidth <= innerWidth"), "phone: the bundles tab is no wider than the screen")
    ctx.close()

    # ================= player =================
    ctx = b.new_context(viewport={"width": 420, "height": 900})
    page = ctx.new_page()
    stub(page)
    page.add_init_script("window.__NO_USER = true;" + BUNDLES)
    cards = lambda: page.evaluate("[...document.querySelectorAll('.gallery-card')].map((c) => [c.querySelector('b').textContent, c.className.replace('gallery-card ', ''), c.tagName, c.getAttribute('href')])")

    # ---- a serial bundle: only the first is open, the rest dimmed and locked ----
    page.goto(BASE + "index.html?s=b1")
    page.wait_for_selector(".gallery-card")
    c = cards()
    check(c == [["קל", "next", "A", "?p=qe&s=b1"], ["בינוני", "locked", "DIV", None], ["קשה", "locked", "DIV", None]],
          f"serial: the first is the only link, a deleted quiz is skipped: {c}")
    check(page.inner_text("#bundle-name") == "מקבץ סדרתי" and "הראשון" in page.inner_text("#intro"), "serial: the bundle's name and where to start")
    dim = page.evaluate("[...document.querySelectorAll('.gallery-card.locked')].map((c) => +getComputedStyle(c).opacity)")
    check(all(o < 0.6 for o in dim) and page.evaluate("+getComputedStyle(document.querySelector('.gallery-card.next')).opacity") == 1,
          f"serial: locked quizzes are dimmed, the open one is not: {dim}")
    check(not page.is_visible("#admin-btn"), "bundle: no sign-in button for whoever got the link")
    page.wait_for_function("Object.keys(window.__store.visits).length === 1")
    check(page.evaluate("Object.values(window.__store.visits)[0].quiz") == "s:b1", "bundle: its screen is counted as a visit of its own")
    if SHOTS:
        page.screenshot(path=f"{SHOTS}/bundle-serial.png")

    # ---- play the first quiz: everything leads back to the bundle, and on to the next ----
    page.click(".gallery-card.next")
    page.wait_for_selector("#start-btn:visible")
    check(page.get_attribute("#home-btn", "href") == "?s=b1" and page.get_attribute("#logo-link", "href") == "?s=b1", "quiz: home is the bundle")
    page.click("#start-btn")
    for ans in ["ans1", "ans2"]:
        page.fill("#answer-input", ans)
        page.click("#submit-btn")
        page.click("#next-btn")
    page.wait_for_selector("#report-screen:visible")
    page.wait_for_selector("#next-level-btn:visible")
    check("בינוני" in page.inner_text("#next-level-btn") and page.get_attribute("#next-level-btn", "href") == "?p=qm&s=b1"
          and page.get_attribute("#report-home-btn", "href") == "?s=b1", "results: on to the next quiz of the bundle, or back to the bundle")
    page.click("#report-home-btn")
    page.wait_for_selector(".gallery-card")
    c = [x[:2] for x in cards()]
    check(c == [["קל", "done"], ["בינוני", "next"], ["קשה", "locked"]], f"serial: finishing one opens the next: {c}")
    if SHOTS:
        page.screenshot(path=f"{SHOTS}/bundle-serial-2.png")

    # ---- a bundle that is not serial: all open, also the one already finished ----
    page.goto(BASE + "index.html?s=b2")
    page.wait_for_selector(".gallery-card")
    c = cards()
    check(c == [["קשה", "open", "A", "?p=qh&s=b2"], ["קל", "done open", "A", "?p=qe&s=b2"]], f"not serial: every quiz is a link, in the bundle's order: {c}")
    check(page.locator(".gallery-card.locked").count() == 0 and "בוחרים" in page.inner_text("#intro"), "not serial: nothing locked")
    if SHOTS:
        page.screenshot(path=f"{SHOTS}/bundle-open.png")

    # ---- a bundle that is gone ----
    page.goto(BASE + "index.html?s=nope")
    page.wait_for_function("document.querySelector('#intro').textContent.includes('לא נמצא')")
    check(page.locator(".gallery-card").count() == 0, "a deleted bundle: says so, shows nothing")

    # ---- the main start screen is as it was ----
    page.goto(BASE + "index.html")
    page.wait_for_selector(".gallery-card")
    check(not page.is_visible("#bundle-name") and page.get_attribute(".gallery-card.next", "href").startswith("?p=")
          and "&s=" not in page.get_attribute(".gallery-card.next", "href"), "main screen: the path, with no bundle in its links")

    # ---- the short page forwards a bundle's code to the bundle ----
    if "/vibe_coding/" in BASE:
        short = BASE.split("/vibe_coding/")[0] + "/carmit/?" + code
        page.route("**/firestore.googleapis.com/**", lambda r: r.fulfill(json={"fields": {"q": {"stringValue": ""}, "b": {"stringValue": ""},
                                                                                         "a": {"stringValue": ""}, "s": {"stringValue": "b1"}}}))
        page.goto(short)
        page.wait_for_url("**/index.html?s=b1")
        page.wait_for_selector(".gallery-card")
        check(page.inner_text("#bundle-name") == "מקבץ סדרתי", "short link: forwards to the bundle")
    else:
        print("SKIP short link (needs a server rooted at the fun repo)")
    ctx.close()
    b.close()

sys.exit(0 if fake_firebase.ok else 1)
