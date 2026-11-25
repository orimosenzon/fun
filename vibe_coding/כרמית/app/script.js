// The game plays one quiz (חידון), index.html?q=<id>, in the order Carmit set, or shuffled if the quiz says so.
// Without ?q= the start screen shows the path: the quizzes Carmit chose for players, in her order.
// The player has no choice: only the next quiz on the path is open, and finishing it opens the one after.
// Progress lasts for this visit only (sessionStorage), so a new visit starts again from the first quiz.
// Two ways into a quiz: ?p=<id> is a step on the path; ?q=<id> is a quiz sent to someone on its own (solo):
// they see only that quiz, never the start screen or the other quizzes.
// A bundle (מקבץ) is a collection of quizzes Carmit sends as one link, ?s=<bundle>: its own start screen with its
// quizzes, opening one after the other if the bundle is serial, or all open if not. A quiz played from it is
// ?p=<quiz>&s=<bundle>, and everything that would lead to the start screen leads back to the bundle.
// A solo quiz comes with the letters marked in it right now: one before it (quiz.opening) and one after it
// (quiz.closing). Links used to name the letters themselves (&b=, &a=, and &m= before that); those are read only
// for a quiz from before letters were marked in quizzes, so that what Carmit sees in the admin is what is shown.
import { loadQuiz, loadQuizzes, loadBundle, loadLetter, primaryAnswer, isCorrect, DEFAULT_COVER, levelOf, levelLabel, playerPath, watchAdmin,
  hasHeading, fillHeading, safeUrl, saveRating, saveComment, saveVisit, isOwnDevice, randomId } from './fb.js?v=202611251653';

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
let ASK_RATING = true;   // after each riddle, ask how hard it was (Carmit can switch it off per quiz)
let order = [];
let current = 0;
let results = [];        // results[i] is what happened at position i: answered or skipped
let hintUsed = false;
let answered = false;

// Was the last tap a finger? Then focusing the answer box would pop up the keyboard. Judged by the tap itself:
// some Android phones (Samsung, newer Android) claim hover and a fine pointer, so a media query alone says "computer".
let lastPointer = matchMedia('(any-pointer: coarse)').matches ? 'touch' : '';
addEventListener('pointerdown', (e) => { lastPointer = e.pointerType; }, { capture: true, passive: true });
const fingerFirst = () => lastPointer === 'touch' || lastPointer === 'pen';

// How hard the riddle was for the player: asked once the answer is known, never required.
const rateBox = $('rate');
const rateBtns = [...document.querySelectorAll('.rate-btn')];
// One id per visit: with the riddle's id it names the rating document, so a changed mind overwrites.
const VISIT = randomId();
// Ori or Carmit themselves: signed in to the admin now, or a browser that ever was. Their ratings stay out of
// the statistics, and their visits are counted apart from the real players'.
let OWN = isOwnDevice();
// Visits are written once we know who is playing (the sign-in check comes after the game has loaded).
let whoKnown = false;
let visitFlags = {};
let visitBase = null;    // { quiz, solo } once the page knows what it shows
function noteVisit(flags = {}) {
  Object.assign(visitFlags, flags);
  if (!whoKnown || !visitBase) return;
  saveVisit({ ...visitBase, own: OWN }, visitFlags).catch((e) => console.error(e));
}

function showRating(level) {
  rateBox.classList.toggle('hidden', !ASK_RATING);
  showComment(false);
  rateBtns.forEach((b) => {
    const on = b.dataset.rate === level;
    b.classList.toggle('on', on);
    b.setAttribute('aria-checked', String(on));
  });
}

function rate(level) {
  const res = results[current];
  if (!res || res.skipped) return;
  res.rating = level;
  const writing = !rateTextBox.classList.contains('hidden');
  const draft = rateText.value;
  showRating(level);
  // Choosing a level while writing leaves the text box as it is.
  if (writing) { showComment(true); rateText.value = draft; }
  if (OWN) return;
  if (res.comment) sendComment(res);   // words already sent carry the level chosen with them
  // The game never waits for this, and a rating that didn't get through is not the player's problem.
  saveRating(`${VISIT}-${res.riddle.id}`, { quiz: QUIZ_ID, riddle: res.riddle.id, level, correct: res.correct, hint: res.hintUsed })
    .catch((e) => console.error(e));
}
rateBtns.forEach((b) => b.addEventListener('click', () => rate(b.dataset.rate)));

// Beside the three buttons the player can also write a few words about the riddle. Until asked for, it is one
// quiet line; pressing it opens a text box. What was written is sent with "שליחה", or on moving to another riddle.
const rateMore = $('rate-more');
const rateTextBox = $('rate-text-box');
const rateText = $('rate-text');

function showComment(open) {
  const sent = results[current]?.comment || '';
  rateTextBox.classList.toggle('hidden', !open);
  rateMore.setAttribute('aria-expanded', String(open));
  rateMore.classList.toggle('sent', !!sent && !open);
  rateMore.textContent = open ? '✕ ביטול' : sent ? '✓ תודה, קיבלנו. לחיצה לעריכה' : '✎ רוצים להוסיף כמה מילים?';
  rateText.value = sent;
}

function sendComment(res) {
  saveComment(`${VISIT}-${res.riddle.id}`, { quiz: QUIZ_ID, riddle: res.riddle.id, text: res.comment, level: res.rating || '',
    correct: res.correct, hint: res.hintUsed }).catch((e) => console.error(e));
}

// Keeps what is in the text box, if anything new. Emptying the box does not take back words already sent.
function keepComment() {
  const res = results[current];
  const text = rateText.value.trim();
  if (!res || res.skipped || rateTextBox.classList.contains('hidden') || !text || text === res.comment) return;
  res.comment = text;
  if (!OWN) sendComment(res);
}

rateMore.addEventListener('click', () => {
  const open = rateTextBox.classList.contains('hidden');
  showComment(open);   // closing it this way drops what was not sent
  if (open) rateText.focus();
});
$('rate-send').addEventListener('click', () => {
  keepComment();
  showComment(false);
});

$('copyright-year').textContent = new Date().getFullYear();

const PRAISES =['פיצחתם.', 'בדיוק.', 'חד. נכון.', 'אלגנטי.', 'נכון מאוד.'];
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

// Every screen opens at its top: the browser keeps the old scroll position when the page swaps
// screens in place (results, the next riddle) and when a page is reloaded or reopened.
if ('scrollRestoration' in history) history.scrollRestoration = 'manual';
const toTop = () => window.scrollTo({ top: 0, left: 0, behavior: 'instant' });

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
  noteVisit({ started: true });
  startScreen.classList.add('hidden');
  reportScreen.classList.add('hidden');
  quizScreen.classList.remove('hidden');
  showRiddle();
}

function showRiddle() {
  toTop();
  const r = riddle();
  hintUsed = false;
  answered = false;
  const pad = (n) => String(n).padStart(2, '0');
  progressText.textContent = `${pad(current + 1)} / ${pad(order.length)}`;
  progressFill.style.width = `${(current / order.length) * 100}%`;
  riddleImage.src = r.image;
  $('ai-note').classList.toggle('hidden', !r.aiImage);   // a small line under a picture made by AI
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
  rateBox.classList.add('hidden');
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
  // On a touch screen focusing the box would pop up the keyboard and hide most of the picture,
  // so there the player taps the box when ready to type. With a real keyboard, type straight away.
  if (!isImageAnswer(r) && !fingerFirst()) answerInput.focus({ preventScroll: true });   // the picture stays in view
  else if (document.activeElement === answerInput) answerInput.blur();
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

function showDone({ correct, rating }) {
  const r = riddle();
  showRating(rating);
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
  keepComment();
  current--;
  showRiddle();
}

function nextRiddle() {
  keepComment();
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
  toTop();
  noteVisit({ finished: true });
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
    // Answering moves the focus to "next"; without this the same Enter press would click it too,
    // and the player would never see whether they were right.
    e.preventDefault();
    if (!answered) submitAnswer();
    else nextRiddle();
  }
});

// After a quiz on the path, or in a bundle: straight on to the next one not finished yet.
async function offerNextQuiz() {
  try {
    const path = BUNDLE ? (await loadBundle(BUNDLE))?.quizzes || [] : playerPath(await loadQuizzes());
    if (!path.some((q) => q.id === QUIZ_ID)) return; // a quiz sent by link, outside the path
    const next = nextOnPath(path);
    if (!next) {
      scoreSub.textContent += ' סיימתם את כל החידונים.';
      return;
    }
    const btn = $('next-level-btn');
    btn.textContent = `לחידון הבא: ${next.title} ←`;
    btn.href = quizHref(next);
    btn.classList.remove('hidden');
    $('report-home-btn').className = 'secondary-btn link-as-btn';
  } catch (e) {
    console.error(e);
  }
}

const quizHref = (q) => `?p=${q.id}${BUNDLE ? `&s=${BUNDLE}` : ''}`;

// The quizzes of the path, or of a bundle. Serial: only the next one is open. Not serial (a bundle can be):
// all are open, also the finished ones.
function showGallery(path, serial = true, first = 'מתחילים מהרמה הקלה') {
  const done = doneQuizzes();
  const next = nextOnPath(path);
  intro.textContent = !path.length ? 'עוד אין חידונים. בקרוב.'
    : !next ? 'סיימתם את כל החידונים. כל הכבוד!'
    : !serial ? 'בוחרים חידון ומתחילים'
    : next === path[0] ? first : 'ממשיכים מהחידון הבא';
  const gallery = $('gallery');
  gallery.classList.remove('hidden');
  const grid = document.createElement('div');
  grid.className = 'gallery-grid';
  path.forEach((q, i) => {
    const state = serial ? (done.has(q.id) ? 'done' : q === next ? 'next' : 'locked') : done.has(q.id) ? 'done open' : 'open';
    grid.appendChild(galleryCard(q, i, state));
  });
  gallery.appendChild(grid);
}

// On a serial list only the next quiz is a link: finished ones show a check, later ones a lock, dimmed.
function galleryCard(q, i, state) {
  const link = state !== 'done' && state !== 'locked';
  const a = document.createElement(link ? 'a' : 'div');
  a.className = `gallery-card ${state}`;
  if (link) a.href = quizHref(q);
  const img = document.createElement('img');
  img.src = q.cover || DEFAULT_COVER;
  img.alt = '';
  if (!q.cover) img.className = 'default';
  const step = document.createElement('i');
  step.className = 'step';
  step.textContent = state.startsWith('done') ? '✓' : state === 'locked' ? '🔒' : String(i + 1).padStart(2, '0');
  const title = document.createElement('b');
  title.textContent = q.title;
  const n = document.createElement('span');
  n.textContent = [`${q.riddleIds.length} חידות`, levelLabel(q)].filter(Boolean).join(' · ');
  a.append(img, step, title, n);
  return a;
}

function fillLetter(box, img, p, letter) {
  box.classList.remove('hidden');
  p.textContent = letter.text || '';
  p.classList.toggle('hidden', !letter.text);
  if (letter.image) {
    img.src = letter.image;
    img.classList.remove('hidden');
  }
}

function fillButton(a, letter) {
  a.href = safeUrl(letter.linkUrl);
  a.textContent = `${letter.linkText || 'לפרטים נוספים'} ↗`;
}

// The letter before the quiz (רישא) sits above it with its button; the one after (סיפא) is on the results
// screen with its own button. With no button after, the button of the letter before comes back at the end.
function showLetters(before, after) {
  if (before) {
    fillLetter($('letter'), $('letter-img'), $('letter-text'), before);
    if (safeUrl(before.linkUrl)) {
      fillButton($('letter-cta-top'), before);
      $('letter-cta-top').classList.remove('hidden');
    }
  }
  if (after) fillLetter($('letter-after'), $('letter-after-img'), $('letter-after-text'), after);
  const end = [after, before].find((l) => l && safeUrl(l.linkUrl));
  if (end) {
    fillButton($('letter-cta'), end);
    $('letter-end').classList.remove('hidden');
  }
}

const params = new URLSearchParams(location.search);
const SOLO = !params.has('p') && params.has('q');
// The bundle this page belongs to: its own screen, or a quiz played from it. (An id is letters and digits only.)
const BUNDLE = (params.get('s') || '').replace(/[^\w-]/g, '') || null;
if (SOLO) {
  document.body.classList.add('solo');
  $('logo-link').removeAttribute('href');
}
if (BUNDLE && !SOLO) {
  // Whoever got a bundle stays in it: the ways "home" lead back to the bundle, not to the main start screen.
  for (const id of ['logo-link', 'home-btn', 'report-home-btn']) $(id).href = `?s=${BUNDLE}`;
  $('logo-link').title = 'לכל החידונים במקבץ';
}

// The way into the admin: one clear button on the start screen; once Ori or Carmit is signed in, on every screen.
const ON_START = !params.has('p') && !params.has('q') && !BUNDLE;
if (ON_START) $('admin-btn').classList.remove('hidden');
watchAdmin((isAdmin) => {
  OWN = OWN || isAdmin;
  $('rate-note').classList.toggle('hidden', !OWN);
  whoKnown = true;
  noteVisit();
  $('admin-btn').textContent = isAdmin ? '⚙ לממשק הניהול' : '🔑 כניסה לניהול';
  $('admin-btn').classList.toggle('hidden', !isAdmin && !ON_START);
}).catch((e) => {
  console.error(e);
  whoKnown = true;
  noteVisit();
});
$('rate-note').classList.toggle('hidden', !OWN);

try {
  const quizId = params.get('p') || params.get('q');
  if (!quizId && BUNDLE) {
    const bundle = await loadBundle(BUNDLE);
    if (!bundle) {
      intro.textContent = 'המקבץ הזה לא נמצא. אולי הוא נמחק.';
    } else {
      document.title = `${bundle.title} · חידות לכרמית`;
      $('bundle-name').textContent = bundle.title;
      $('bundle-name').classList.remove('hidden');
      showGallery(bundle.quizzes, !!bundle.serial, 'מתחילים מהחידון הראשון');
      // Counted apart from the main start screen ('') and from any quiz: the statistics tab shows it as the bundle.
      visitBase = { quiz: `s:${bundle.id}`, solo: false };
      noteVisit();
    }
  } else if (!quizId) {
    showGallery(playerPath(await loadQuizzes()));
    visitBase = { quiz: '', solo: false };
    noteVisit();
  } else {
    if (!SOLO) $('home-btn').classList.remove('hidden');
    // A letter that was deleted since the link was sent just isn't there.
    const letter = (id) => (SOLO && id ? loadLetter(id).catch(() => null) : null);
    const quiz = await loadQuiz(quizId);
    const marked = quiz && (quiz.opening !== undefined || quiz.closing !== undefined);
    const [before, after] = await Promise.all(marked ? [letter(quiz.opening), letter(quiz.closing)]
      : [letter(params.get('b') || params.get('m')), letter(params.get('a'))]);
    if (!quiz) {
      intro.innerHTML = SOLO ? 'החידון הזה לא נמצא. אולי הוא נמחק.'
        : `החידון הזה לא נמצא. אולי הוא נמחק.<br><a href="${BUNDLE ? `?s=${BUNDLE}` : 'index.html'}">לכל החידונים</a>`;
    } else {
      RIDDLES = quiz.riddles;
      QUIZ_ID = quiz.id;
      visitBase = { quiz: quiz.id, solo: SOLO };
      noteVisit();
      SHUFFLE = !!quiz.shuffle;
      ASK_RATING = quiz.askRating !== false;
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
      showLetters(before, after);
      $('quiz-level').textContent = levelLabel(quiz);
      $('quiz-level').className = `level-badge level-${levelOf(quiz)}`;
      $('quiz-level').classList.toggle('hidden', !levelLabel(quiz));
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
