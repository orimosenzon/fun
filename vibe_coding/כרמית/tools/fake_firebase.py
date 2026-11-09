"""Fake Firebase for the Playwright tests: an in-memory Firestore with four quizzes (an old one without a level,
then easy, medium and hard) and Ori signed in. stub(page) routes the Firebase SDK imports to it."""

IMG = "data:image/gif;base64,R0lGODlhAQABAAAAACw="

APP = "export const initializeApp = () => ({});"
FS = """
const img = '%s';
const store = window.__store = { riddles: {}, quizzes: {} };
for (let i = 1; i <= 4; i++) store.riddles['r' + i] = { image: img, answerType: 'text', answers: ['ANS' + i], primary: 0, created: i };
store.quizzes.qold = { title: 'ישן', cover: '', riddleIds: ['r1'], created: 4 };
store.quizzes.qe = { title: 'קל', cover: '', riddleIds: ['r1', 'r2'], level: 'easy', created: 3 };
store.quizzes.qm = { title: 'בינוני', cover: '', riddleIds: ['r3'], level: 'medium', created: 2 };
store.quizzes.qh = { title: 'קשה', cover: '', riddleIds: ['r4'], level: 'hard', created: 1 };
// A test sets window.__PATH (an init script) to put quizzes on the player's path, in that order.
(window.__PATH || []).forEach((id, i) => { store.quizzes[id].path = i; });
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
export const addDoc = async (c, data) => { const id = 'new' + Date.now(); store[c.name][id] = structuredClone(data); return { id }; };
export const updateDoc = async (d, patch) => { Object.assign(store[d.name][d.id], structuredClone(patch)); sessionStorage.setItem('__last', JSON.stringify(patch)); };
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


def stub(page):
    page.on("pageerror", lambda e: check(False, f"page error: {e}"))
    page.on("dialog", lambda d: d.accept())
    for name, body in [("firebase-app.js", APP), ("firebase-firestore.js", FS), ("firebase-auth.js", AUTH)]:
        page.route(f"**/{name}", lambda r, _req=None, body=body: r.fulfill(body=body, content_type="text/javascript"))


