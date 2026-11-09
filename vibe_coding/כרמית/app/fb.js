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

// A marketing letter: { quizId, name (for the admin only), text, image, linkText, linkUrl }. Its link opens that
// one quiz on its own with the letter above it; the link button shows again after the quiz.
export async function loadLetter(id) {
  const snap = await fs.getDoc(fs.doc(db, 'letters', id));
  return snap.exists() ? { id: snap.id, ...snap.data() } : null;
}

export async function loadLetters() {
  const snap = await fs.getDocs(fs.query(fs.collection(db, 'letters'), fs.orderBy('created', 'desc')));
  return snap.docs.map((d) => ({ id: d.id, ...d.data() }));
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
export const levelOf = (q) => (LEVELS.some((L) => L.id === q.level) ? q.level : 'easy');

// The path: the quizzes players see, in Carmit's order (quiz.path is the position; no path = not shown).
// Until any quiz has a position, every quiz with riddles is on it, easy first, so the game never goes empty.
export function playerPath(quizzes) {
  const playable = quizzes.filter((q) => q.riddleIds.length);
  const placed = playable.filter((q) => Number.isInteger(q.path)).sort((a, b) => a.path - b.path);
  if (placed.length) return placed;
  const rank = (q) => LEVELS.findIndex((L) => L.id === levelOf(q));
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
