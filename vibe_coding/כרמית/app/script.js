// The game plays one quiz (חידון), index.html?q=<id>, in the order Carmit set, or shuffled if the quiz says so.
// Without ?q= the start screen lists all the quizzes to choose from, by difficulty level.
// Every visit starts at the easy level; finishing a quiz opens the next level, for this visit only
// (sessionStorage, so a new visit starts again from easy, as Carmit wants).
import { loadQuiz, loadQuizzes, primaryAnswer, isCorrect, DEFAULT_COVER, LEVELS, levelOf } from './fb.js';

const $ = (id) => document.getElementById(id);
const startScreen = $('start-screen');
const quizScreen = $('quiz-screen');
const reportScreen = $('report-screen');

const intro = $('intro');
const startBtn = $('start-btn');
const progressText = $('progress-text');
const progressFill = $('progress-fill');
const riddleImage = $('riddle-image');
const riddleQuestion = $('riddle-question');
const hintBox = $('hint-box');
const hintBtn = $('hint-btn');
const answerInput = $('answer-input');
const submitBtn = $('submit-btn');
const answerImage = $('answer-image');
const selfCheck = $('self-check');
const feedback = $('feedback');
const nextBtn = $('next-btn');
const prevBtn = $('prev-btn');
const skipBtn = $('skip-btn');

const scoreHeadline = $('score-headline');
const scoreSub = $('score-sub');
const reportBody = document.querySelector('#report-table tbody');

let RIDDLES = [];
let LEVEL = 'easy';      // the level of the quiz being played
let SHUFFLE = false;     // the quiz asks for a fresh random order on every start
let order = [];
let current = 0;
let results = [];        // results[i] is what happened at position i: answered or skipped
let hintUsed = false;
let answered = false;

const PRAISES = ['פיצחתם.', 'בדיוק.', 'חד. נכון.', 'אלגנטי.', 'נכון מאוד.'];
const CONSOLATIONS = ['לא הפעם. התשובה:', 'לא בדיוק. התשובה:', 'כמעט. התשובה:'];

// ---------- levels ----------

const DONE_KEY = 'carmit-levels-done';
function doneLevels() {
  try { return new Set(JSON.parse(sessionStorage.getItem(DONE_KEY)) || []); } catch { return new Set(); }
}
function markDone(level) {
  const done = doneLevels();
  done.add(level);
  try { sessionStorage.setItem(DONE_KEY, JSON.stringify([...done])); } catch { /* private mode: the level opens for this page only */ }
}
// A level is open once every lower level that has quizzes was finished. A level with no quizzes never blocks.
function openLevels(quizzes) {
  const done = doneLevels();
  const open = new Set();
  for (const L of LEVELS) {
    open.add(L.id);
    const present = quizzes.some((q) => levelOf(q) === L.id);
    if (present && !done.has(L.id)) break;
  }
  return open;
}

function pick(arr) {
  return arr[Math.floor(Math.random() * arr.length)];
}

const riddle = () => RIDDLES[order[current]];
const isImageAnswer = (r) => r.answerType === 'image';

function startQuiz() {
  order = RIDDLES.map((_, i) => i);
  if (SHUFFLE) {
    for (let i = order.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [order[i], order[j]] = [order[j], order[i]];
    }
  }
  current = 0;
  results = [];
  startScreen.classList.add('hidden');
  reportScreen.classList.add('hidden');
  quizScreen.classList.remove('hidden');
  showRiddle();
}

function showRiddle() {
  const r = riddle();
  hintUsed = false;
  answered = false;
  const pad = (n) => String(n).padStart(2, '0');
  progressText.textContent = `${pad(current + 1)} / ${pad(order.length)}`;
  progressFill.style.width = `${(current / order.length) * 100}%`;
  riddleImage.src = r.image;
  riddleQuestion.textContent = r.question || '';
  riddleQuestion.classList.toggle('hidden', !r.question);
  hintBox.textContent = '';
  hintBox.classList.add('hidden');
  answerInput.value = '';
  answerInput.disabled = false;
  // An image answer can't be typed: the player thinks, reveals it, and says whether they got it.
  answerInput.classList.toggle('hidden', isImageAnswer(r));
  submitBtn.innerHTML = isImageAnswer(r) ? 'הצגת התשובה' : 'בדיקה <kbd>↵</kbd>';
  answerImage.classList.add('hidden');
  answerImage.removeAttribute('src');
  selfCheck.classList.add('hidden');
  feedback.textContent = '';
  feedback.className = 'feedback hidden';
  nextBtn.classList.add('hidden');
  submitBtn.disabled = false;
  hintBtn.disabled = !r.hint;
  hintBtn.textContent = r.hint ? 'רמז' : 'אין רמז';
  prevBtn.disabled = current === 0;
  skipBtn.classList.remove('hidden');
  skipBtn.textContent = current + 1 < order.length ? 'דלג ←' : 'דלג לתוצאות ←';
  // Back on a riddle already answered: show it as it was. The answer was already seen, so no second try.
  const done = results[current];
  if (done && !done.skipped) {
    hintUsed = done.hintUsed;
    if (hintUsed) useHint();
    answered = true;
    if (isImageAnswer(r)) {
      answerImage.src = r.answerImage;
      answerImage.classList.remove('hidden');
    } else {
      answerInput.value = done.userAnswer;
    }
    showDone(done);
    return;
  }
  if (!isImageAnswer(r)) answerInput.focus();
}

function useHint() {
  const r = riddle();
  if (!r.hint) return;
  hintUsed = true;
  hintBox.textContent = `רמז: ${r.hint}`;
  hintBox.classList.remove('hidden');
}

function submitAnswer() {
  if (answered) return;
  const r = riddle();
  if (isImageAnswer(r)) {
    answered = true;
    submitBtn.disabled = true;
    answerImage.src = r.answerImage;
    answerImage.classList.remove('hidden');
    selfCheck.classList.remove('hidden');
    // The answer is in view now: the player grades themselves before moving on.
    prevBtn.disabled = true;
    skipBtn.classList.add('hidden');
    return;
  }
  answered = true;
  const userAnswer = answerInput.value;
  finish(isCorrect(userAnswer, r), userAnswer.trim() || '(לא ניתנה תשובה)');
}

function selfGrade(correct) {
  selfCheck.classList.add('hidden');
  finish(correct, correct ? 'צדקתי' : 'לא צדקתי');
}

function finish(correct, userAnswer) {
  results[current] = { riddle: riddle(), userAnswer, hintUsed, correct };
  showDone(results[current]);
  nextBtn.focus();
}

function showDone({ correct }) {
  const r = riddle();
  feedback.classList.remove('hidden');
  if (correct) {
    feedback.textContent = pick(PRAISES);
    feedback.className = 'feedback correct';
  } else {
    feedback.textContent = isImageAnswer(r) ? 'לא הפעם. הלאה.' : `${pick(CONSOLATIONS)} `;
    if (!isImageAnswer(r)) {
      const ans = document.createElement('span');
      ans.className = 'ans';
      ans.textContent = primaryAnswer(r);
      feedback.appendChild(ans);
    }
    feedback.className = 'feedback wrong';
  }

  answerInput.disabled = true;
  submitBtn.disabled = true;
  nextBtn.classList.remove('hidden');
  nextBtn.textContent = current + 1 < order.length ? 'הבאה ←' : 'לתוצאות ←';
  progressFill.style.width = `${((current + 1) / order.length) * 100}%`;
  prevBtn.disabled = current === 0;
  skipBtn.classList.add('hidden');
}

function skipRiddle() {
  results[current] = { riddle: riddle(), userAnswer: '(דילגתם)', hintUsed, correct: false, skipped: true };
  nextRiddle();
}

function prevRiddle() {
  if (current === 0) return;
  current--;
  showRiddle();
}

function nextRiddle() {
  current++;
  if (current >= order.length) {
    showReport();
  } else {
    showRiddle();
  }
}

function cell(child) {
  const td = document.createElement('td');
  if (typeof child === 'string') td.textContent = child;
  else td.appendChild(child);
  return td;
}

function thumb(src) {
  const img = document.createElement('img');
  img.src = src;
  img.className = 'thumb';
  img.alt = '';
  return img;
}

function showReport() {
  quizScreen.classList.add('hidden');
  reportScreen.classList.remove('hidden');
  results = results.filter(Boolean);
  const score = results.filter((r) => r.correct).length;
  scoreHeadline.innerHTML = `הציון שלכם<span class="score"><b>${score}</b>/${results.length}</span>`;
  scoreSub.textContent =
    score === results.length
      ? 'ניקוד מושלם. אין מה להוסיף.'
      : score >= results.length * 0.7
      ? 'תוצאה מרשימה.'
      : 'יש עוד מה לפצח.';
  markDone(LEVEL);
  offerNextLevel();
  reportBody.innerHTML = '';
  results.forEach((res, i) => {
    const r = res.riddle;
    const tr = document.createElement('tr');
    tr.className = res.correct ? 'row-correct' : 'row-wrong';
    tr.append(
      cell(String(i + 1)),
      cell(thumb(r.image)),
      cell(res.userAnswer),
      cell(isImageAnswer(r) ? thumb(r.answerImage) : primaryAnswer(r)),
      cell(res.hintUsed ? 'כן' : 'לא'),
      cell(res.skipped ? '–' : res.correct ? '✓' : '✗'),
    );
    reportBody.appendChild(tr);
  });
}

startBtn.addEventListener('click', startQuiz);
hintBtn.addEventListener('click', useHint);
submitBtn.addEventListener('click', submitAnswer);
$('self-yes').addEventListener('click', () => selfGrade(true));
$('self-no').addEventListener('click', () => selfGrade(false));
nextBtn.addEventListener('click', nextRiddle);
skipBtn.addEventListener('click', skipRiddle);
prevBtn.addEventListener('click', prevRiddle);
$('home-btn').addEventListener('click', (e) => {
  const midQuiz = !quizScreen.classList.contains('hidden') && (current > 0 || answered);
  if (midQuiz && !confirm('לצאת מהחידון? ההתקדמות לא תישמר.')) e.preventDefault();
});
answerInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter') {
    if (!answered) submitAnswer();
    else nextRiddle();
  }
});

// After a quiz: if the next level has quizzes, invite the player there.
async function offerNextLevel() {
  const i = LEVELS.findIndex((L) => L.id === LEVEL);
  const next = LEVELS[i + 1];
  if (!next) return;
  try {
    const quizzes = (await loadQuizzes()).filter((q) => q.riddleIds.length && levelOf(q) === next.id);
    if (!quizzes.length) return;
    const btn = $('next-level-btn');
    btn.textContent = `ממשיכים לרמה ${next.name} ←`;
    btn.href = quizzes.length === 1 ? `?q=${quizzes[0].id}` : `index.html#level-${next.id}`;
    btn.classList.remove('hidden');
    $('report-home-btn').className = 'secondary-btn link-as-btn';
  } catch (e) {
    console.error(e);
  }
}

function showGallery(quizzes) {
  const playable = quizzes.filter((q) => q.riddleIds.length);
  intro.textContent = playable.length ? 'מתחילים מהרמה הקלה. כל חידון שמסיימים פותח את הרמה הבאה.' : 'עוד אין חידונים. בקרוב.';
  const gallery = $('gallery');
  gallery.classList.remove('hidden');
  const open = openLevels(playable);
  LEVELS.forEach((L, i) => {
    const qs = playable.filter((q) => levelOf(q) === L.id);
    if (!qs.length) return;
    const locked = !open.has(L.id);
    const section = document.createElement('section');
    section.className = `level level-${L.id}${locked ? ' locked' : ''}`;
    section.id = `level-${L.id}`;
    const head = document.createElement('h3');
    head.className = 'level-head';
    head.innerHTML = `<span class="idx">0${i + 1}</span> רמה ${L.name}${locked ? ' <span class="lock">🔒</span>' : ''}`;
    section.appendChild(head);
    if (locked) {
      const prev = LEVELS.slice(0, i).reverse().find((P) => playable.some((q) => levelOf(q) === P.id));
      const note = document.createElement('p');
      note.className = 'level-note';
      note.textContent = `נפתחת אחרי שמסיימים חידון ברמה ${prev.name}.`;
      section.appendChild(note);
    }
    const grid = document.createElement('div');
    grid.className = 'gallery-grid';
    for (const q of qs) grid.appendChild(galleryCard(q, locked));
    section.appendChild(grid);
    gallery.appendChild(section);
  });
  if (location.hash) document.querySelector(location.hash)?.scrollIntoView({ behavior: 'smooth' });
}

function galleryCard(q, locked) {
  const a = document.createElement(locked ? 'div' : 'a');
  a.className = `gallery-card${locked ? ' locked' : ''}`;
  if (!locked) a.href = `?q=${q.id}`;
  const img = document.createElement('img');
  img.src = q.cover || DEFAULT_COVER;
  img.alt = '';
  if (!q.cover) img.className = 'default';
  const title = document.createElement('b');
  title.textContent = q.title;
  const n = document.createElement('span');
  n.textContent = `${q.riddleIds.length} חידות`;
  a.append(img, title, n);
  return a;
}

try {
  const quizId = new URLSearchParams(location.search).get('q');
  if (!quizId) {
    showGallery(await loadQuizzes());
  } else {
    $('home-btn').classList.remove('hidden');
    const quiz = await loadQuiz(quizId);
    if (!quiz) {
      intro.innerHTML = 'החידון הזה לא נמצא. אולי הוא נמחק.<br><a href="index.html">לכל החידונים</a>';
    } else {
      RIDDLES = quiz.riddles;
      LEVEL = levelOf(quiz);
      // On a quiz page the cover and the start button come first; the about moves down.
      document.querySelector('.about-sketch').prepend(document.querySelector('.about'));
      SHUFFLE = !!quiz.shuffle;
      document.title = `${quiz.title} · חידות לכרמית`;
      $('quiz-intro').classList.remove('hidden');
      // Without a cover of its own the header logo already stands in for it.
      $('quiz-cover').src = quiz.cover;
      $('quiz-cover').classList.toggle('hidden', !quiz.cover);
      $('quiz-name').textContent = quiz.title;
      $('quiz-level').textContent = `רמה ${LEVELS.find((L) => L.id === LEVEL).name}`;
      $('quiz-level').className = `level-badge level-${LEVEL}`;
      intro.innerHTML = `<span class="num">${RIDDLES.length}</span> חידות של משחקי מילים, עברית ואנגלית.<br>מוכנים?`;
      startBtn.classList.remove('hidden');
      startBtn.disabled = RIDDLES.length === 0;
    }
  }
} catch (e) {
  console.error(e);
  intro.textContent = 'לא הצלחנו לטעון את החידות. בדקו את החיבור לאינטרנט ונסו לרענן את הדף.';
}
