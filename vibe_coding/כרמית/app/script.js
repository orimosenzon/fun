// The game plays one quiz (חידון), index.html?q=<id>, in the order Carmit set, or shuffled if the quiz says so.
// Without ?q= the start screen shows the path: the quizzes Carmit chose for players, in her order.
// The player has no choice: only the next quiz on the path is open, and finishing it opens the one after.
// Progress lasts for this visit only (sessionStorage), so a new visit starts again from the first quiz.
// Two ways into a quiz: ?p=<id> is a step on the path; ?q=<id> is a quiz sent to someone on its own (solo):
// they see only that quiz, never the start screen or the other quizzes.
// A marketing letter (?m=<id>) shows above a solo quiz, and its link button again after it.
import { loadQuiz, loadQuizzes, loadLetter, primaryAnswer, isCorrect, DEFAULT_COVER, LEVELS, levelOf, playerPath, watchAdmin,
  hasHeading, fillHeading, safeUrl } from './fb.js';

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
let QUIZ_ID = null;      // the quiz being played
let SHUFFLE = false;     // the quiz asks for a fresh random order on every start
let order = [];
let current = 0;
let results = [];        // results[i] is what happened at position i: answered or skipped
let hintUsed = false;
let answered = false;

const PRAISES = ['פיצחתם.', 'בדיוק.', 'חד. נכון.', 'אלגנטי.', 'נכון מאוד.'];
const CONSOLATIONS = ['לא הפעם. התשובה:', 'לא בדיוק. התשובה:', 'כמעט. התשובה:'];

// ---------- the path ----------

const DONE_KEY = 'carmit-quizzes-done';
function doneQuizzes() {
  try { return new Set(JSON.parse(sessionStorage.getItem(DONE_KEY)) || []); } catch { return new Set(); }
}
function markDone(id) {
  const done = doneQuizzes();
  done.add(id);
  try { sessionStorage.setItem(DONE_KEY, JSON.stringify([...done])); } catch { /* private mode: progress lasts for this page only */ }
}
// The next quiz on the path is the first one not finished in this visit.
const nextOnPath = (path) => path.find((q) => !doneQuizzes().has(q.id));

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
  if (!SOLO) {
    markDone(QUIZ_ID);
    offerNextQuiz();
  }
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

// After a quiz on the path: straight on to the next one.
async function offerNextQuiz() {
  try {
    const path = playerPath(await loadQuizzes());
    if (!path.some((q) => q.id === QUIZ_ID)) return; // a quiz sent by link, outside the path
    const next = nextOnPath(path);
    if (!next) {
      scoreSub.textContent += ' סיימתם את כל החידונים.';
      return;
    }
    const btn = $('next-level-btn');
    btn.textContent = `לחידון הבא: ${next.title} ←`;
    btn.href = `?p=${next.id}`;
    btn.classList.remove('hidden');
    $('report-home-btn').className = 'secondary-btn link-as-btn';
  } catch (e) {
    console.error(e);
  }
}

function showGallery(quizzes) {
  const path = playerPath(quizzes);
  const done = doneQuizzes();
  const next = nextOnPath(path);
  intro.textContent = !path.length ? 'עוד אין חידונים. בקרוב.'
    : next === path[0] ? 'מתחילים מהרמה הקלה'
    : next ? 'ממשיכים מהחידון הבא' : 'סיימתם את כל החידונים. כל הכבוד!';
  const gallery = $('gallery');
  gallery.classList.remove('hidden');
  const grid = document.createElement('div');
  grid.className = 'gallery-grid';
  path.forEach((q, i) => grid.appendChild(galleryCard(q, i, done.has(q.id) ? 'done' : q === next ? 'next' : 'locked')));
  gallery.appendChild(grid);
}

// Only the next quiz is a link; finished ones show a check, later ones a lock.
function galleryCard(q, i, state) {
  const a = document.createElement(state === 'next' ? 'a' : 'div');
  a.className = `gallery-card ${state}`;
  if (state === 'next') a.href = `?p=${q.id}`;
  const img = document.createElement('img');
  img.src = q.cover || DEFAULT_COVER;
  img.alt = '';
  if (!q.cover) img.className = 'default';
  const step = document.createElement('i');
  step.className = 'step';
  step.textContent = state === 'done' ? '✓' : state === 'locked' ? '🔒' : String(i + 1).padStart(2, '0');
  const title = document.createElement('b');
  title.textContent = q.title;
  const n = document.createElement('span');
  const level = LEVELS.find((L) => L.id === levelOf(q)).name;
  n.textContent = `${q.riddleIds.length} חידות · רמה ${level}`;
  a.append(img, step, title, n);
  return a;
}

function showLetter(letter) {
  $('letter').classList.remove('hidden');
  $('letter-text').textContent = letter.text || '';
  $('letter-text').classList.toggle('hidden', !letter.text);
  if (letter.image) {
    $('letter-img').src = letter.image;
    $('letter-img').classList.remove('hidden');
  }
  const url = safeUrl(letter.linkUrl);
  if (!url) return;
  const label = letter.linkText || 'לפרטים נוספים';
  for (const a of [$('letter-cta-top'), $('letter-cta')]) {
    a.href = url;
    a.textContent = `${label} ↗`;
  }
  $('letter-cta-top').classList.remove('hidden');
  $('letter-end').classList.remove('hidden');
}

const params = new URLSearchParams(location.search);
const SOLO = !params.has('p') && params.has('q');
if (SOLO) {
  document.body.classList.add('solo');
  $('logo-link').removeAttribute('href');
}

// The way into the admin: one clear button on the start screen; once Ori or Carmit is signed in, on every screen.
const ON_START = !params.has('p') && !params.has('q');
if (ON_START) $('admin-btn').classList.remove('hidden');
watchAdmin((isAdmin) => {
  $('admin-btn').textContent = isAdmin ? '⚙ לממשק הניהול' : '🔑 כניסה לניהול';
  $('admin-btn').classList.toggle('hidden', !isAdmin && !ON_START);
}).catch((e) => console.error(e));

try {
  const quizId = params.get('p') || params.get('q');
  if (!quizId) {
    showGallery(await loadQuizzes());
  } else {
    if (!SOLO) $('home-btn').classList.remove('hidden');
    const [quiz, letter] = await Promise.all([
      loadQuiz(quizId),
      SOLO && params.get('m') ? loadLetter(params.get('m')).catch(() => null) : null,
    ]);
    if (!quiz) {
      intro.innerHTML = SOLO ? 'החידון הזה לא נמצא. אולי הוא נמחק.'
        : 'החידון הזה לא נמצא. אולי הוא נמחק.<br><a href="index.html">לכל החידונים</a>';
    } else {
      RIDDLES = quiz.riddles;
      QUIZ_ID = quiz.id;
      SHUFFLE = !!quiz.shuffle;
      document.title = `${quiz.title} · חידות לכרמית`;
      $('quiz-intro').classList.remove('hidden');
      // Without a cover of its own the header logo already stands in for it.
      $('quiz-cover').src = quiz.cover;
      $('quiz-cover').classList.toggle('hidden', !quiz.cover);
      $('quiz-name').textContent = quiz.title;
      // A note sent along with the link (?n=<id>), shown at the top of the quiz.
      const note = (quiz.notes || {})[params.get('n')];
      if (note) {
        $('quiz-note').textContent = note;
        $('quiz-note').classList.remove('hidden');
      }
      if (letter) showLetter(letter);
      $('quiz-level').textContent = `רמה ${LEVELS.find((L) => L.id === levelOf(quiz)).name}`;
      $('quiz-level').className = `level-badge level-${levelOf(quiz)}`;
      // The quiz's own heading, set in the admin, replaces the usual Hebrew × English line.
      const custom = hasHeading(quiz.heading);
      if (custom) {
        fillHeading($('quiz-heading'), quiz.heading);
        $('quiz-heading').classList.remove('hidden');
        $('motif').classList.add('hidden');
      }
      intro.innerHTML = custom ? `<span class="num">${RIDDLES.length}</span> חידות.<br>מוכנים?`
        : `<span class="num">${RIDDLES.length}</span> חידות של משחקי מילים, עברית ואנגלית.<br>מוכנים?`;
      startBtn.classList.remove('hidden');
      startBtn.disabled = RIDDLES.length === 0;
    }
  }
} catch (e) {
  console.error(e);
  intro.textContent = 'לא הצלחנו לטעון את החידות. בדקו את החיבור לאינטרנט ונסו לרענן את הדף.';
}
