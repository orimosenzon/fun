// Firebase for the riddles: the game reads riddles/*, the admin page (Ori and Carmit) writes them.
// The web config is public by design; the security is in ../firebase/firestore.rules.
const SDK = 'https://www.gstatic.com/firebasejs/10.14.1/';
const CONFIG = {
  apiKey: 'AIzaSyBH4Mu3MdsovurJBUiAFmCdnvp6VQW0nW0',
  authDomain: 'carmit-riddles.firebaseapp.com',
  projectId: 'carmit-riddles',
  appId: '1:1027859208328:web:d1562fcdd3405725b4e73d',
};
export const ADMINS = ['orimosenzon@gmail.com', 'carmitconsari@gmail.com'];

const [appMod, fs] = await Promise.all([import(`${SDK}firebase-app.js`), import(`${SDK}firebase-firestore.js`)]);
export const app = appMod.initializeApp(CONFIG);
export const db = fs.getFirestore(app);
export { fs };

export async function loadRiddles() {
  const snap = await fs.getDocs(fs.query(fs.collection(db, 'riddles'), fs.orderBy('created', 'desc')));
  return snap.docs.map((d) => ({ id: d.id, ...d.data() }));
}

// A quiz (חידון) is { title, cover (data URL or ''), riddleIds (ordered) }; the riddles stay in the pool.
export async function loadQuizzes() {
  const snap = await fs.getDocs(fs.query(fs.collection(db, 'quizzes'), fs.orderBy('created', 'desc')));
  return snap.docs.map((d) => ({ id: d.id, ...d.data() }));
}

export async function loadQuiz(id) {
  const snap = await fs.getDoc(fs.doc(db, 'quizzes', id));
  if (!snap.exists()) return null;
  const quiz = { id: snap.id, ...snap.data() };
  const docs = await Promise.all(quiz.riddleIds.map((rid) => fs.getDoc(fs.doc(db, 'riddles', rid))));
  // A riddle deleted from the pool simply drops out of the quiz.
  quiz.riddles = docs.filter((d) => d.exists()).map((d) => ({ id: d.id, ...d.data() }));
  return quiz;
}

// A bundle (מקבץ) is a titled, ordered collection of quizzes with a link of its own: { title, quizIds, serial }.
// serial: only the first quiz is open, and finishing one opens the next (like the path on the start screen);
// otherwise all of them are open and the player chooses.
export async function loadBundles() {
  const snap = await fs.getDocs(fs.query(fs.collection(db, 'bundles'), fs.orderBy('created', 'desc')));
  return snap.docs.map((d) => ({ id: d.id, ...d.data() }));
}

// For the player: the bundle with its quizzes in order. A quiz deleted since, or one with no riddles, drops out.
export async function loadBundle(id) {
  const snap = await fs.getDoc(fs.doc(db, 'bundles', id));
  if (!snap.exists()) return null;
  const bundle = { id: snap.id, ...snap.data() };
  const docs = await Promise.all((bundle.quizIds || []).map((qid) => fs.getDoc(fs.doc(db, 'quizzes', qid))));
  bundle.quizzes = docs.filter((d) => d.exists()).map((d) => ({ id: d.id, ...d.data() })).filter((q) => q.riddleIds.length);
  return bundle;
}

// A letter: { name (for the admin only), text, image, linkText, linkUrl }, not tied to any quiz. A sent link names
// the letter before the quiz (b=<id>) and the one after it (a=<id>); old links used m=<id> for the one before.
export async function loadLetter(id) {
  const snap = await fs.getDoc(fs.doc(db, 'letters', id));
  return snap.exists() ? { id: snap.id, ...snap.data() } : null;
}

export async function loadLetters() {
  const snap = await fs.getDocs(fs.query(fs.collection(db, 'letters'), fs.orderBy('created', 'desc')));
  return snap.docs.map((d) => ({ id: d.id, ...d.data() }));
}

// A category: { name }. Every riddle belongs to one (riddle.category = the category's id); a riddle without one,
// or whose category was deleted since, is in the built-in "כללי". In the order Carmit made them.
export const GENERAL = 'כללי';
export async function loadCategories() {
  const snap = await fs.getDocs(fs.collection(db, 'categories'));
  const at = (c) => (c && c.toMillis ? c.toMillis() : +c || 0);
  return snap.docs.map((d) => ({ id: d.id, ...d.data() })).sort((a, b) => at(a.created) - at(b.created));
}

// Only web, mail, phone and WhatsApp links; a bare address gets https://.
export function safeUrl(u) {
  const s = (u || '').trim();
  if (!s) return '';
  const full = /^[a-z][a-z0-9+.-]*:/i.test(s) ? s : `https://${s}`;
  return /^(https?:|mailto:|tel:)/i.test(full) ? full : '';
}

// A quiz's heading: { text, image }, shown under the title instead of the usual Hebrew × English line.
// The first line of the text is the headline, the rest is the explanation under it.
export const hasHeading = (h) => !!(h && ((h.text || '').trim() || h.image));

export function fillHeading(box, h) {
  box.innerHTML = '';
  if (h.image) {
    const img = document.createElement('img');
    img.src = h.image;
    img.alt = '';
    box.appendChild(img);
  }
  const [first, ...rest] = (h.text || '').trim().split('\n');
  if (first) {
    const b = document.createElement('span');
    b.className = 'langs';
    b.textContent = first;
    box.appendChild(b);
  }
  if (rest.join('\n').trim()) {
    const p = document.createElement('span');
    p.className = 'rest';
    p.textContent = rest.join('\n').trim();
    box.appendChild(p);
  }
}

// Difficulty levels, in the order the player climbs them. A quiz saved before levels existed counts as easy.
export const LEVELS = [
  { id: 'easy', name: 'קלה' },
  { id: 'medium', name: 'בינונית' },
  { id: 'hard', name: 'קשה' },
];
// 'none': a quiz that mixes riddles of different difficulty, so it carries no level label.
export const NO_LEVEL = 'none';
export const levelOf = (q) => (q.level === NO_LEVEL || LEVELS.some((L) => L.id === q.level) ? q.level : 'easy');
// "רמה קלה" etc., or '' for a quiz without a level.
export const levelLabel = (q) => {
  const L = LEVELS.find((x) => x.id === levelOf(q));
  return L ? `רמה ${L.name}` : '';
};

// After answering a riddle the player can say how hard it was for them (the same three words as the quiz levels).
// One document per riddle per visit, so changing one's mind overwrites instead of counting twice.
// Anyone can write a rating; only Ori and Carmit can read them (the statistics tab in the admin).
export function saveRating(id, { quiz, riddle, level, correct, hint }) {
  return fs.setDoc(fs.doc(db, 'ratings', id), { quiz, riddle, level, correct, hint, created: fs.serverTimestamp() });
}

// A few words the player chose to write about a riddle, beside the three buttons (which may be left unpressed:
// level is then ''). One document per riddle per visit, as with the ratings. Read only by Ori and Carmit.
export function saveComment(id, { quiz, riddle, text, level, correct, hint }) {
  return fs.setDoc(fs.doc(db, 'comments', id), { quiz, riddle, text, level, correct, hint, created: fs.serverTimestamp() });
}

export async function loadComments() {
  const snap = await fs.getDocs(fs.collection(db, 'comments'));
  return snap.docs.map((d) => ({ id: d.id, ...d.data() }));
}

export async function loadRatings() {
  const snap = await fs.getDocs(fs.collection(db, 'ratings'));
  return snap.docs.map((d) => ({ id: d.id, ...d.data() }));
}

// Visits, for the statistics tab. A browser gets a random id of its own (nothing about the person), and each
// day it opens the start screen or a quiz leaves one document: { visitor, day, quiz ('' = the start screen), solo,
// own, started, finished }. "own" marks Ori and Carmit themselves: any browser that ever signed in to the admin.
const OWN_KEY = 'carmit-own-device';
const VISITOR_KEY = 'carmit-visitor';
export function markOwnDevice() {
  try { localStorage.setItem(OWN_KEY, '1'); } catch { /* private mode */ }
}
export function isOwnDevice() {
  try { return localStorage.getItem(OWN_KEY) === '1'; } catch { return false; }
}
export const randomId = (bytes = 12) =>
  Array.from(crypto.getRandomValues(new Uint8Array(bytes)), (b) => b.toString(36).padStart(2, '0')).join('');
function visitorId() {
  try {
    let id = localStorage.getItem(VISITOR_KEY);
    if (!id) localStorage.setItem(VISITOR_KEY, id = randomId());
    return id;
  } catch { return randomId(); }
}
export const dayKey = (d = new Date()) =>
  `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`;

// flags: { started: true } when the player presses start, { finished: true } at the results. Merged into the
// day's document, so coming back the same day never erases that they finished.
export function saveVisit({ quiz, solo, own }, flags = {}) {
  const visitor = visitorId();
  const day = dayKey();
  return fs.setDoc(fs.doc(db, 'visits', `${visitor}-${day}-${quiz || 'home'}`),
    { visitor, day, quiz: quiz || '', solo: !!solo, own: !!own, ...flags, created: fs.serverTimestamp() }, { merge: true });
}

export async function loadVisits() {
  const snap = await fs.getDocs(fs.collection(db, 'visits'));
  return snap.docs.map((d) => ({ id: d.id, ...d.data() }));
}

// The path: the quizzes players see, in Carmit's order (quiz.path is the position; no path = not shown).
// Until any quiz has a position, every quiz with riddles is on it, easy first, so the game never goes empty.
export function playerPath(quizzes) {
  const playable = quizzes.filter((q) => q.riddleIds.length);
  const placed = playable.filter((q) => Number.isInteger(q.path)).sort((a, b) => a.path - b.path);
  if (placed.length) return placed;
  // A quiz without a level goes after the levelled ones.
  const rank = (q) => { const i = LEVELS.findIndex((L) => L.id === levelOf(q)); return i < 0 ? LEVELS.length : i; };
  return playable.map((q, i) => [q, i]).sort(([a, i], [b, j]) => rank(a) - rank(b) || i - j).map(([q]) => q);
}

// Calls back with true when Ori or Carmit is signed in (the sign-in is shared with admin.html).
// Loaded after the game itself, so a player never waits for it.
export async function watchAdmin(cb) {
  const au = await import(`${SDK}firebase-auth.js`);
  au.onAuthStateChanged(au.getAuth(app), (user) => cb(!!user && ADMINS.includes((user.email || '').toLowerCase())));
}

// Shown wherever a quiz has no cover of its own.
export const DEFAULT_COVER = 'logo.jpg';

export function primaryAnswer(r) {
  return r.answerType === 'text' ? (r.answers[r.primary] ?? r.answers[0] ?? '') : '';
}

// Lenient comparison: case, spaces, punctuation and Hebrew final letters don't matter.
const FINALS = { 'ך': 'כ', 'ם': 'מ', 'ן': 'נ', 'ף': 'פ', 'ץ': 'צ' };
export function normalize(s) {
  return (s || '').toLowerCase().normalize('NFKD')
    .replace(/[֑-ׇ]/g, '')
    .replace(/[ךםןףץ]/g, (c) => FINALS[c])
    .replace(/[^\p{L}\p{N}]/gu, '');
}

export function isCorrect(userAnswer, riddle) {
  const u = normalize(userAnswer);
  return !!u && riddle.answers.some((a) => normalize(a) === u);
}

// "גבר גבר": what Carmit (or Ori) asks Claude to change in the app. Claude checks once an hour (tools/gever_cron.sh),
// makes the change, and answers on the same document. A request:
// { by (email), text, image (data URL or ''), followUp (id of an earlier request it continues, or ''), created,
//   status: 'new' | 'working' | 'done' | 'question', reply, replied }. Only the first six are written here; the
// status after 'new', the reply and its time are written by Claude's script (as the project owner, past the rules).
export async function loadGever() {
  const snap = await fs.getDocs(fs.collection(db, 'gever'));
  return snap.docs.map((d) => ({ id: d.id, ...d.data() }));
}

export function sendGever({ by, text, image = '', followUp = '' }) {
  return fs.addDoc(fs.collection(db, 'gever'), { by, text, image, followUp, status: 'new', created: fs.serverTimestamp() });
}
