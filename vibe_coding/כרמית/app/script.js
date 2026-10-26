import { loadRiddles, primaryAnswer, isCorrect } from './fb.js';

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

const scoreHeadline = $('score-headline');
const scoreSub = $('score-sub');
const reportBody = document.querySelector('#report-table tbody');
const restartBtn = $('restart-btn');

let RIDDLES = [];
let order = [];
let current = 0;
let results = [];
let hintUsed = false;
let answered = false;

const PRAISES = ['פיצחתם.', 'בדיוק.', 'חד. נכון.', 'אלגנטי.', 'נכון מאוד.'];
const CONSOLATIONS = ['לא הפעם. התשובה:', 'לא בדיוק. התשובה:', 'כמעט. התשובה:'];

function shuffle(arr) {
  const a = arr.slice();
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

function pick(arr) {
  return arr[Math.floor(Math.random() * arr.length)];
}

const riddle = () => RIDDLES[order[current]];
const isImageAnswer = (r) => r.answerType === 'image';

function startQuiz() {
  order = shuffle(RIDDLES.map((_, i) => i));
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
  const r = riddle();
  results.push({ riddle: r, userAnswer, hintUsed, correct });

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
  nextBtn.focus();
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
  const score = results.filter((r) => r.correct).length;
  scoreHeadline.innerHTML = `הציון שלכם<span class="score"><b>${score}</b>/${results.length}</span>`;
  scoreSub.textContent =
    score === results.length
      ? 'ניקוד מושלם. אין מה להוסיף.'
      : score >= results.length * 0.7
      ? 'תוצאה מרשימה.'
      : 'יש עוד מה לפצח. סיבוב נוסף?';
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
      cell(res.correct ? '✓' : '✗'),
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
restartBtn.addEventListener('click', startQuiz);
answerInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter') {
    if (!answered) submitAnswer();
    else nextRiddle();
  }
});

try {
  RIDDLES = await loadRiddles();
  intro.innerHTML = `<span class="num">${RIDDLES.length}</span> חידות של משחקי מילים, עברית ואנגלית.<br>מוכנים?`;
  startBtn.disabled = RIDDLES.length === 0;
} catch (e) {
  console.error(e);
  intro.textContent = 'לא הצלחנו לטעון את החידות. בדקו את החיבור לאינטרנט ונסו לרענן את הדף.';
}
