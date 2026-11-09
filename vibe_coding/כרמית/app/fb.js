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

// Difficulty levels, in the order the player climbs them. A quiz saved before levels existed counts as easy.
export const LEVELS = [
  { id: 'easy', name: 'קלה' },
  { id: 'medium', name: 'בינונית' },
  { id: 'hard', name: 'קשה' },
];
export const levelOf = (q) => (LEVELS.some((L) => L.id === q.level) ? q.level : 'easy');

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
