"""Drives admin.html with stubbed Firebase (in-memory Firestore, signed-in Ori) to test copy/move between quizzes."""
import sys
from playwright.sync_api import sync_playwright

BASE = sys.argv[1]  # e.g. http://localhost:8765/app/admin.html
IMG = "data:image/gif;base64,R0lGODlhAQABAAAAACw="

APP = "export const initializeApp = () => ({});"
FS = """
const store = window.__store = { riddles: {}, quizzes: {} };
let n = 0;
const img = '%s';
for (let i = 1; i <= 6; i++) store.riddles['r' + i] = { image: img, answerType: 'text', answers: ['ANS' + i], primary: 0, question: 'Q' + i, created: i };
store.quizzes.qa = { title: 'אלף', cover: '', riddleIds: ['r1', 'r2', 'r3', 'r4'], created: 2, shuffle: location.hash === '#shuf' };
store.quizzes.qb = { title: 'בית', cover: '', riddleIds: ['r5', 'r2'], created: 1 };
window.__writes = [];
export const getFirestore = () => ({});
export const collection = (db, name) => ({ name });
export const doc = (db, name, id) => ({ name, id });
export const query = (c) => c;
export const orderBy = () => null;
export const serverTimestamp = () => Date.now();
export const arrayRemove = (v) => ({ __remove: v });
const snapDoc = (name, id) => ({ id, exists: () => id in store[name], data: () => structuredClone(store[name][id]) });
export const getDocs = async (c) => ({ docs: Object.keys(store[c.name]).sort((a, b) => store[c.name][b].created - store[c.name][a].created).map((id) => snapDoc(c.name, id)) });
export const getDoc = async (d) => snapDoc(d.name, d.id);
export const addDoc = async (c, data) => { const id = 'new' + (++n); store[c.name][id] = structuredClone(data); window.__writes.push(['add', c.name, id]); return { id }; };
export const updateDoc = async (d, patch) => { Object.assign(store[d.name][d.id], structuredClone(patch)); window.__writes.push(['update', d.name, d.id]); };
export const deleteDoc = async (d) => { delete store[d.name][d.id]; };
""" % IMG
AUTH = """
export const getAuth = () => ({ currentUser: { email: 'orimosenzon@gmail.com' } });
export const onAuthStateChanged = (auth, cb) => setTimeout(() => cb(auth.currentUser), 0);
export const signInWithPopup = async () => {};
export const signOut = async () => {};
export class GoogleAuthProvider {}
"""

ok = True
def check(cond, msg):
    global ok
    print(("PASS " if cond else "FAIL ") + msg)
    ok &= bool(cond)

with sync_playwright() as p:
    b = p.chromium.launch()
    page = b.new_page(viewport={"width": 1300, "height": 900})
    page.on("pageerror", lambda e: check(False, f"page error: {e}"))
    page.on("dialog", lambda d: d.accept("מפוצל") if d.type == "prompt" else d.accept())
    for name, body in [("firebase-app.js", APP), ("firebase-firestore.js", FS), ("firebase-auth.js", AUTH)]:
        page.route(f"**/{name}", lambda r, _req=None, body=body: r.fulfill(body=body, content_type="text/javascript"))
    page.goto(BASE + "#quiz/qa")
    page.wait_for_selector("#quiz-list .q-item")
    ids = lambda q: page.evaluate(f"window.__store.quizzes['{q}'].riddleIds")

    check(page.is_hidden("#pick-actions"), "actions hidden with nothing checked")
    boxes = page.locator("#quiz-list .pick")
    boxes.nth(0).click()
    boxes.nth(2).click(modifiers=["Shift"])
    check(page.text_content("#pick-count") == "נבחרו 3", "shift-click picks a range of 3")
    check(page.locator("#quiz-list .q-item.picked").count() == 3, "3 rows highlighted")
    opts = page.eval_on_selector_all("#pick-target option", "os => os.map(o => o.textContent)")
    check("אלף" not in opts and "בית" in opts, f"target list excludes current quiz: {opts}")

    # copy r1,r2,r3 to בית (r2 already there)
    page.select_option("#pick-target", "qb")
    page.click("#pick-copy")
    page.wait_for_timeout(200)
    check(ids("qb") == ["r5", "r2", "r1", "r3"], f"copy appends, skips dup: {ids('qb')}")
    check(ids("qa") == ["r1", "r2", "r3", "r4"], "copy leaves source intact")
    check("(אחת מהן כבר הייתה שם)" in page.text_content("#toast"), f"toast: {page.text_content('#toast')}")
    check(page.is_hidden("#pick-actions"), "selection cleared after copy")

    # move r4 to a new quiz
    boxes.nth(3).click()
    page.select_option("#pick-target", "new")
    page.click("#pick-move")
    page.wait_for_timeout(200)
    new = page.evaluate("Object.entries(window.__store.quizzes).find(([k]) => k.startsWith('new'))")
    check(new and new[1]["title"] == "מפוצל" and new[1]["riddleIds"] == ["r4"], f"new quiz created: {new}")
    check(ids("qa") == ["r1", "r2", "r3"], f"move removes from source: {ids('qa')}")
    check(page.locator("#quiz-list .q-item").count() == 3, "list re-rendered")
    check(page.input_value("#pick-target") == new[0], "new quiz becomes the selected target")

    # select all
    page.click("#pick-all")
    check(page.text_content("#pick-count") == "נבחרו 3", "select all")
    page.click("#pick-clear")
    check(page.text_content("#pick-count") == "בחירת הכל", "clear")

    # pull: filter side pool by quiz בית
    page.select_option("#side-source", "qb")
    shown = page.eval_on_selector_all("#side-pool .side-item .ans", "es => es.map(e => e.textContent)")
    check(shown == ["ANS5"], f"side pool from בית shows only what's missing: {shown}")
    page.select_option("#side-source", "")
    page.fill("#side-search", "ans")
    check(page.is_visible("#side-add-all"), f"add-all visible: {page.text_content('#side-add-all')}")
    page.click("#side-add-all")
    page.wait_for_timeout(200)
    check(sorted(ids("qa")) == ["r1", "r2", "r3", "r4", "r5", "r6"], f"add all: {ids('qa')}")

    # order: fixed / shuffle
    check(page.is_checked('input[name="quiz-order"][value="fixed"]'), "order defaults to fixed")
    page.click('.seg:has(input[value="shuffle"])')
    page.wait_for_timeout(200)
    check(page.evaluate("window.__store.quizzes.qa.shuffle") is True, "shuffle saved")
    check("אקראי" in page.text_content("#order-help"), "help text explains random order")

    page.fill("#side-search", "")
    page.locator("#quiz-list .pick").nth(1).click()
    page.screenshot(path=sys.argv[2])

    # the game: fixed order always starts with Q1, shuffled order varies between starts
    game = BASE.replace("admin.html", "index.html?q=qa")
    def first_riddle(hash_):
        page.goto("about:blank")
        page.goto(game + hash_)
        page.click("#start-btn")
        return page.text_content("#riddle-question")
    fixed = {first_riddle("") for _ in range(4)}
    check(fixed == {"Q1"}, f"fixed order starts with Q1: {fixed}")
    shuf = {first_riddle("#shuf") for _ in range(10)}
    check(len(shuf) > 1, f"shuffled order varies: {sorted(shuf)}")
    b.close()

print("ALL OK" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
