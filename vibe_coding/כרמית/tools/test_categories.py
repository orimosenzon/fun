"""Drives admin.html and the player with stubbed Firebase to test (1) categories of riddles: the filter above the
pool, changing a riddle's category from its tile and from the riddle form, adding, renaming and deleting categories
(a deleted category's riddles go back to "כללי"), and the category filter when adding riddles to a quiz;
(2) the "this picture was made by AI" mark: set in the riddle form, shown to the player as a small line.
Usage: python3 tools/test_categories.py http://localhost:8766/app/ [screenshot-dir]"""
import sys
from pathlib import Path
from playwright.sync_api import sync_playwright

import fake_firebase
from fake_firebase import stub, check

BASE = sys.argv[1]
SHOTS = sys.argv[2] if len(sys.argv) > 2 else None
PICTURE = str(Path(__file__).resolve().parent.parent / "app" / "favicon.png")
INIT = """
  window.__CATEGORIES = { c1: { name: 'יש לי שיר', created: 1 }, c2: { name: 'מי? אני', created: 2 } };
  window.__RIDDLES = { r1: { category: 'c1', aiImage: true }, r3: { category: 'gone' } };
"""

with sync_playwright() as p:
    b = p.chromium.launch()
    ctx = b.new_context(viewport={"width": 1200, "height": 1000})
    a = ctx.new_page()
    stub(a)
    a.add_init_script(INIT)
    store = lambda expr: a.evaluate(f"(() => {{ const s = window.__store; return {expr}; }})()")
    chips = lambda: [t.replace("\n", " ") for t in a.locator("#cat-chips .cat-chip").all_inner_texts()]
    tile = lambda ans: a.locator("#pool-grid .pool-tile", has_text=ans)
    a.goto(BASE + "admin.html#pool")
    a.wait_for_selector("#tab-pool:visible .pool-tile")

    # ---- the bar: every category with its count; a riddle whose category is gone counts as "כללי" ----
    check(chips() == ["הכול 4", "יש לי שיר 1", "מי? אני 0", "כללי 3"], f"bar: categories and counts: {chips()}")
    check(tile("ANS1").locator(".cat-select").input_value() == "c1" and tile("ANS3").locator(".cat-select").input_value() == "",
          "tile: shows the riddle's category, and כללי for a category that no longer exists")
    check(tile("ANS1").locator(".ai-mark").count() == 1 and tile("ANS2").locator(".ai-mark").count() == 0, "tile: the AI mark only where set")

    # ---- change a category from the tile ----
    tile("ANS2").locator(".cat-select").select_option("c2")
    a.wait_for_timeout(150)
    check(store("s.riddles.r2.category") == "c2", "tile: the new category is saved")
    check(chips() == ["הכול 4", "יש לי שיר 1", "מי? אני 1", "כללי 2"], f"bar: counts follow: {chips()}")

    # ---- filter ----
    a.locator("#cat-chips .cat-chip", has_text="מי? אני").click()
    check(a.locator("#pool-grid .pool-tile").count() == 1 and tile("ANS2").count() == 1, "filter: only the riddles of the category")
    a.locator("#cat-chips .cat-chip", has_text="כללי").click()
    check(a.locator("#pool-grid .pool-tile").count() == 2, "filter: כללי shows the riddles without a category")
    a.locator("#cat-chips .cat-chip", has_text="הכול").click()
    check(a.locator("#pool-grid .pool-tile").count() == 4, "filter: all")
    if SHOTS:
        a.screenshot(path=f"{SHOTS}/cat-pool.png", full_page=True)

    # ---- the categories window: add ----
    a.click("#cat-manage")
    a.wait_for_selector("#cat-dialog[open]")
    check(a.locator("#cat-list .cat-row").count() == 3 and a.locator("#cat-list .cat-row.general .link-btn").count() == 0,
          "window: both categories and כללי, which has no delete")
    a.fill("#cat-name", "מחכים לאות")
    a.click("#cat-add")
    a.wait_for_timeout(150)
    names = store("Object.values(s.categories).map((c) => c.name)")
    check("מחכים לאות" in names and len(names) == 3, f"add: saved: {names}")
    check(a.locator("#cat-list .cat-rename").evaluate_all("(xs) => xs.map((x) => x.value)") == ["יש לי שיר", "מי? אני", "מחכים לאות"],
          "add: the new category is last in the list")
    for bad in ["מחכים  לאות", "כללי", "   "]:
        a.fill("#cat-name", bad)
        a.click("#cat-add")
        a.wait_for_timeout(100)
        check(a.is_visible("#cat-error") and store("Object.keys(s.categories).length") == 3, f"add: '{bad}' is refused")
    if SHOTS:
        a.screenshot(path=f"{SHOTS}/cat-window.png")

    # ---- rename ----
    row = a.locator("#cat-list .cat-row").nth(1)
    row.locator(".cat-rename").fill("מי אני?")
    row.locator(".cat-rename").press("Enter")
    a.wait_for_timeout(150)
    check(store("s.categories.c2.name") == "מי אני?", "rename: saved")
    row = a.locator("#cat-list .cat-row").nth(1)
    row.locator(".cat-rename").fill("יש לי שיר")
    row.locator(".cat-rename").press("Enter")
    a.wait_for_timeout(150)
    check(store("s.categories.c2.name") == "מי אני?" and a.locator("#cat-list .cat-rename").nth(1).input_value() == "מי אני?",
          "rename: a name that is taken is refused and the old one comes back")

    # ---- delete: the riddles go back to כללי, none is deleted ----
    a.locator("#cat-list .cat-row").nth(0).get_by_role("button", name="מחיקה").click()   # the confirm is accepted by the stub
    a.wait_for_timeout(200)
    check(store("!('c1' in s.categories) && s.riddles.r1.category === '' && Object.keys(s.riddles).length === 4"),
          "delete: the category is gone, its riddle is in כללי, no riddle was deleted")
    a.click("#cat-done")
    check(chips() == ["הכול 4", "מי אני? 1", "מחכים לאות 0", "כללי 3"], f"bar: after rename and delete: {chips()}")

    # ---- a new riddle starts in the category the pool is showing; a new category can be made from the form ----
    a.locator("#cat-chips .cat-chip", has_text="מי אני?").click()
    a.click("#pool-new")
    a.wait_for_selector("#riddle-dialog[open]")
    check(a.input_value("#riddle-category") == "c2", "form: a new riddle starts in the filtered category")
    a.select_option("#riddle-category", "+new")
    a.wait_for_selector("#cat-dialog[open]")
    a.fill("#cat-name", "שפת הקודש")
    a.click("#cat-add")
    a.wait_for_selector("#cat-dialog", state="hidden")
    new_id = store("Object.keys(s.categories).find((id) => s.categories[id].name === 'שפת הקודש')")
    check(a.is_visible("#riddle-dialog") and a.input_value("#riddle-category") == new_id, "form: the category made on the way is chosen")
    a.set_input_files("#image-file", PICTURE)
    a.wait_for_selector("#image-preview:not(.hidden)")
    a.fill("#answers-list .answer-text", "NEWONE")
    a.check("#ai-image")
    if SHOTS:
        a.screenshot(path=f"{SHOTS}/cat-form.png")
    a.click("#save-btn")
    a.wait_for_selector("#riddle-dialog", state="hidden")
    saved = store("Object.values(s.riddles).find((r) => r.answers[0] === 'NEWONE')")
    check(saved["category"] == new_id and saved["aiImage"] is True, "form: the category and the AI mark are saved with the riddle")

    # ---- editing: the form shows what was saved, and un-marking saves too ----
    a.locator("#cat-chips .cat-chip", has_text="הכול").click()
    tile("NEWONE").get_by_role("button", name="עריכה").click()
    a.wait_for_selector("#riddle-dialog[open]")
    check(a.input_value("#riddle-category") == new_id and a.is_checked("#ai-image"), "edit: the form opens with the category and the mark")
    a.uncheck("#ai-image")
    a.select_option("#riddle-category", "")
    a.click("#save-btn")
    a.wait_for_selector("#riddle-dialog", state="hidden")
    saved = store("Object.values(s.riddles).find((r) => r.answers[0] === 'NEWONE')")
    check(saved["category"] == "" and saved["aiImage"] is False, "edit: back to כללי and without the mark")
    tile("ANS4").get_by_role("button", name="עריכה").click()
    a.wait_for_selector("#riddle-dialog[open]")
    check(a.input_value("#riddle-category") == "" and not a.is_checked("#ai-image"), "edit: an old riddle opens as כללי, not marked")
    a.click("#cancel-btn")

    # ---- a new category straight from a tile ----
    tile("ANS4").locator(".cat-select").select_option("+new")
    a.wait_for_selector("#cat-dialog[open]")
    a.fill("#cat-name", "חדשה")
    a.click("#cat-add")
    a.wait_for_selector("#cat-dialog", state="hidden")
    a.wait_for_timeout(150)
    fresh = store("Object.keys(s.categories).find((id) => s.categories[id].name === 'חדשה')")
    check(fresh and store("s.riddles.r4.category") == fresh, "tile: a category made from the tile is given to that riddle")

    # ---- in a quiz: the category shows on the riddle, and filters what can be added ----
    a.goto(BASE + "admin.html#quiz/qm")
    a.wait_for_function("document.querySelector('#quiz-title').value === 'בינוני'")
    opts = a.locator("#side-source option").all_inner_texts()
    check("קטגוריה: מי אני?" in opts and "קטגוריה: כללי" in opts, f"quiz: categories are offered as a filter: {opts}")
    a.select_option("#side-source", "cat:c2")
    check(a.locator("#side-pool .side-item").count() == 1 and "ANS2" in a.inner_text("#side-pool"), "quiz: the filter leaves the category's riddles")
    a.click("#side-pool .side-item")
    a.wait_for_timeout(150)
    check(store("s.quizzes.qm.riddleIds") == ["r3", "r2"], "quiz: added from the category")
    check(a.locator("#quiz-list .q-item", has_text="ANS2").locator(".cat-tag").inner_text() == "מי אני?"
          and a.locator("#quiz-list .q-item", has_text="ANS3").locator(".cat-tag").count() == 0, "quiz: the category shows on its riddle only")
    ctx.close()

    # ---- a locked quiz: the category of its riddle can still change (the player never sees it) ----
    ctx = b.new_context(viewport={"width": 1200, "height": 1000})
    a = ctx.new_page()
    stub(a)
    a.add_init_script(INIT + "window.__PATCH = { qe: { locked: true } };")
    a.goto(BASE + "admin.html#pool")
    a.wait_for_selector("#tab-pool:visible .pool-tile")
    tile("ANS2").locator(".cat-select").select_option("c2")
    a.wait_for_timeout(150)
    check(store("s.riddles.r2.category") == "c2", "locked quiz: the category of its riddle can change")
    ctx.close()

    # ---- phone: the bar and the window fit ----
    ctx = b.new_context(viewport={"width": 390, "height": 800})
    a = ctx.new_page()
    stub(a)
    a.add_init_script(INIT)
    a.goto(BASE + "admin.html#pool")
    a.wait_for_selector("#tab-pool:visible .pool-tile")
    check(a.evaluate("document.documentElement.scrollWidth <= innerWidth"), "phone: the pool is no wider than the screen")
    a.click("#cat-manage")
    a.wait_for_selector("#cat-dialog[open]")
    check(a.evaluate("(() => { const d = document.querySelector('#cat-dialog'); return d.scrollWidth <= d.clientWidth + 1; })()"), "phone: the window fits")
    if SHOTS:
        a.screenshot(path=f"{SHOTS}/cat-phone.png")
    ctx.close()

    # ---- the player: a small line under a picture made by AI, and only there ----
    ctx = b.new_context(viewport={"width": 420, "height": 900})
    page = ctx.new_page()
    stub(page)
    page.add_init_script("window.__NO_USER = true;" + INIT)
    page.goto(BASE + "index.html?q=qe")
    page.click("#start-btn")
    page.wait_for_selector("#riddle-image:visible")
    check(page.is_visible("#ai-note") and "בינה מלאכותית" in page.inner_text("#ai-note"), "player: the AI line shows under a marked picture")
    size = page.evaluate("parseFloat(getComputedStyle(document.querySelector('#ai-note')).fontSize)")
    check(size <= 12, f"player: the line is small ({size}px)")
    if SHOTS:
        page.screenshot(path=f"{SHOTS}/ai-player.png")
    page.fill("#answer-input", "ans1")
    page.click("#submit-btn")
    page.click("#next-btn")
    page.wait_for_function("document.querySelector('#progress-text').textContent.startsWith('02')")
    check(not page.is_visible("#ai-note"), "player: no line on a riddle that is not marked")
    ctx.close()
    b.close()

sys.exit(0 if fake_firebase.ok else 1)
