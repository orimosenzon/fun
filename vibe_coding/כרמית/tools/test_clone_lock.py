"""Drives admin.html and the player with stubbed Firebase to test (1) cloning a riddle (in the pool and inside a
quiz) and cloning a whole quiz; (2) a locked quiz: nothing in it can change, its riddles can't be edited or deleted
from anywhere, and editing one from another quiz swaps in an editable copy there; (3) the per-quiz switch for
the "how hard was it" question.
Usage: python3 tools/test_clone_lock.py http://localhost:8766/app/ [screenshot-dir]"""
import sys
from playwright.sync_api import sync_playwright

import fake_firebase
from fake_firebase import stub, check

BASE = sys.argv[1]
SHOTS = sys.argv[2] if len(sys.argv) > 2 else None

with sync_playwright() as p:
    b = p.chromium.launch()
    ctx = b.new_context(viewport={"width": 1200, "height": 1000})
    a = ctx.new_page()
    stub(a)
    store = lambda expr: a.evaluate(f"(() => {{ const s = window.__store; return {expr}; }})()")
    a.goto(BASE + "admin.html#pool")
    a.wait_for_selector("#tab-pool:visible .pool-grid > *")

    # ---- clone a riddle in the pool ----
    a.locator("#pool-grid > *", has_text="ANS3").get_by_role("button", name="שיבוט").click()
    a.wait_for_selector("#riddle-dialog[open]")
    check(a.inner_text("#form-title") == "עריכת העותק", "pool: the clone opens for editing")
    check(store("Object.keys(s.riddles).length") == 5, "pool: a fifth riddle exists")
    a.fill("#answers-list .answer-text", "COPY3")
    a.click("#save-btn")
    a.wait_for_timeout(200)
    ans = store("Object.values(s.riddles).map((r) => r.answers[0]).sort()")
    check(ans == ["ANS1", "ANS2", "ANS3", "ANS4", "COPY3"], f"pool: the copy changed, the original did not: {ans}")
    check(store("s.quizzes.qm.riddleIds") == ["r3"], "pool: the copy joined no quiz")

    # ---- clone a riddle inside a quiz: right after the original ----
    a.goto(BASE + "admin.html#quiz/qe")
    a.wait_for_selector("#quiz-list .q-item")
    a.locator("#quiz-list .q-item", has_text="ANS1").get_by_role("button", name="שיבוט").click()
    a.wait_for_selector("#riddle-dialog[open]")
    a.click("#cancel-btn")
    ids = store("s.quizzes.qe.riddleIds")
    check(len(ids) == 3 and ids[0] == "r1" and ids[2] == "r2", f"quiz: the copy sits right after the original: {ids}")

    # ---- the ask-about-difficulty switch ----
    check(a.is_checked("input[name=quiz-ask][value=yes]"), "ask: on by default")
    a.click("label:has(input[name=quiz-ask][value=no])")
    a.wait_for_timeout(100)
    check(store("s.quizzes.qe.askRating") is False, "ask: switched off and saved")

    # ---- clone the quiz ----
    a.click("#quiz-clone")
    a.wait_for_function("location.hash.startsWith('#quiz/new')")
    cid = a.evaluate("location.hash.slice(6)")
    c = store(f"s.quizzes['{cid}']")
    check(c["title"] == "קל (עותק)" and c["riddleIds"] == ids and c["level"] == "easy" and c["askRating"] is False and "locked" not in c,
          f"clone quiz: same riddles and settings, not locked: {c['title']}")
    check(a.input_value("#quiz-title") == "קל (עותק)", "clone quiz: the copy is open")

    # ---- lock the original ----
    a.goto(BASE + "admin.html#quiz/qe")
    a.wait_for_selector("#quiz-list .q-item")
    a.click("#quiz-lock")
    a.wait_for_selector("#lock-banner:visible")
    check(store("s.quizzes.qe.locked") is True, "lock: saved")
    check(a.is_disabled("#quiz-title") and a.is_disabled("#quiz-delete") and a.is_disabled("#quiz-new-riddle")
          and a.is_disabled("input[name=quiz-level][value=hard]") and a.is_disabled("#heading-text"), "lock: the controls are off")
    check(not a.is_visible(".pane-pool") and a.locator("#quiz-list .icon-btn:visible").count() == 0
          and a.locator("#quiz-list .q-item").first.get_by_role("button", name="עריכה").count() == 0,
          "lock: no adding, removing or editing riddles")
    check(a.is_enabled("#share-btn") and a.is_enabled("#quiz-clone") and a.is_enabled("input[name=quiz-ask][value=yes]"),
          "lock: sending, cloning and the difficulty question stay open")
    before = store("JSON.stringify(s.quizzes.qe)")
    if SHOTS:
        a.screenshot(path=f"{SHOTS}/locked.png", full_page=True)

    # ---- its riddles are locked everywhere ----
    a.goto(BASE + "admin.html#pool")
    a.wait_for_selector("#tab-pool:visible .pool-grid > *")
    check(a.evaluate("document.querySelector('#quiz-grid').textContent").count("🔒") == 1, "grid: the locked quiz carries a lock")
    tile = a.locator("#pool-grid > *", has_text="ANS2")
    tile.get_by_role("button", name="מחיקה").click()   # the alert is accepted by the stub
    check(store("'r2' in s.riddles"), "locked riddle: can't be deleted from the pool")
    # Editing it from the clone (not locked, shares the riddle): an editable copy takes its place there.
    a.goto(BASE + f"admin.html#quiz/{cid}")
    a.wait_for_selector("#quiz-list .q-item")
    a.locator("#quiz-list .q-item", has_text="ANS2").get_by_role("button", name="עריכה").click()
    a.wait_for_selector("#riddle-dialog[open]")
    a.fill("#answers-list .answer-text", "ANS2-NEW")
    a.click("#save-btn")
    a.wait_for_timeout(200)
    check(store("s.riddles.r2.answers[0]") == "ANS2", "locked riddle: the original is untouched")
    cids = store(f"s.quizzes['{cid}'].riddleIds")
    check("r2" not in cids and len(cids) == 3 and store(f"s.riddles['{cids[2]}'].answers[0]") == "ANS2-NEW",
          f"locked riddle: in the other quiz the edited copy replaced it: {cids}")
    check(store("JSON.stringify(s.quizzes.qe)") == before, "lock: the locked quiz is exactly as it was")
    # A locked quiz is not offered as a target for copying riddles.
    a.click("#pick-all")
    opts = a.locator("#pick-target option").all_inner_texts()
    check("קל" not in opts and "בינוני" in opts, f"lock: not a target for copy/move: {opts}")

    # ---- unlock ----
    a.goto(BASE + "admin.html#quiz/qe")
    a.wait_for_selector("#lock-banner:visible")
    a.click("#quiz-lock")
    a.wait_for_selector("#lock-banner", state="hidden")
    check(store("s.quizzes.qe.locked") is False and a.is_enabled("#quiz-title") and a.is_visible(".pane-pool"), "unlock: editable again")
    ctx.close()

    # ---- the player: no difficulty question when switched off ----
    for patch, want in [("{ qe: { askRating: false } }", False), ("{}", True)]:
        ctx = b.new_context(viewport={"width": 420, "height": 900})
        page = ctx.new_page()
        stub(page)
        page.add_init_script(f"window.__NO_USER = true; window.__PATCH = {patch};")
        page.goto(BASE + "index.html?q=qe")
        page.click("#start-btn")
        page.fill("#answer-input", "ans1")
        page.click("#submit-btn")
        page.wait_for_selector("#next-btn:visible")
        check(page.is_visible("#rate") == want, f"player: difficulty question shown = {want}")
        ctx.close()
    b.close()

sys.exit(0 if fake_firebase.ok else 1)
