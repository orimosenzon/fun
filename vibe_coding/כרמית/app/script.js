(function () {
  const startScreen = document.getElementById('start-screen');
  const quizScreen = document.getElementById('quiz-screen');
  const reportScreen = document.getElementById('report-screen');

  const startBtn = document.getElementById('start-btn');
  const progressText = document.getElementById('progress-text');
  const riddleImage = document.getElementById('riddle-image');
  const riddleQuestion = document.getElementById('riddle-question');
  const hintBox = document.getElementById('hint-box');
  const hintBtn = document.getElementById('hint-btn');
  const answerInput = document.getElementById('answer-input');
  const submitBtn = document.getElementById('submit-btn');
  const feedback = document.getElementById('feedback');
  const nextBtn = document.getElementById('next-btn');

  const scoreHeadline = document.getElementById('score-headline');
  const scoreSub = document.getElementById('score-sub');
  const reportBody = document.querySelector('#report-table tbody');
  const restartBtn = document.getElementById('restart-btn');

  let order = [];
  let current = 0;
  let results = [];
  let hintUsed = false;
  let answered = false;

  const PRAISES = ['כל הכבוד! 🎉', 'מעולה! 💪', 'וואו, צדקתם! ⭐', 'איזה כישרון! 🏆', 'נכון מאוד! 👏'];
  const CONSOLATIONS = ['לא נורא, ננסה את הבאה! התשובה הייתה:', 'הפעם לא. התשובה הנכונה היא:', 'קרוב, אבל לא. התשובה:'];

  function shuffle(arr) {
    const a = arr.slice();
    for (let i = a.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [a[i], a[j]] = [a[j], a[i]];
    }
    return a;
  }

  function normalize(s) {
    return (s || '').trim().toLowerCase().replace(/\s+/g, ' ');
  }

  function isCorrect(userAnswer, correctAnswer) {
    const u = normalize(userAnswer);
    if (!u) return false;
    const c = normalize(correctAnswer);
    if (u === c) return true;
    return c.split(' ').includes(u);
  }

  function pick(arr) {
    return arr[Math.floor(Math.random() * arr.length)];
  }

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
    const riddle = RIDDLES[order[current]];
    hintUsed = false;
    answered = false;
    progressText.textContent = `חידה ${current + 1} מתוך ${order.length}`;
    riddleImage.src = riddle.image;
    riddleQuestion.textContent = riddle.question;
    hintBox.textContent = '';
    hintBox.classList.add('hidden');
    answerInput.value = '';
    answerInput.disabled = false;
    feedback.classList.add('hidden');
    feedback.textContent = '';
    feedback.className = 'feedback hidden';
    nextBtn.classList.add('hidden');
    submitBtn.disabled = false;
    hintBtn.disabled = !riddle.hint;
    hintBtn.textContent = riddle.hint ? '💡 רמז' : '💡 אין רמז לחידה זו';
    answerInput.focus();
  }

  function useHint() {
    const riddle = RIDDLES[order[current]];
    if (!riddle.hint) return;
    hintUsed = true;
    hintBox.textContent = `רמז: ${riddle.hint}`;
    hintBox.classList.remove('hidden');
  }

  function submitAnswer() {
    if (answered) return;
    answered = true;
    const riddle = RIDDLES[order[current]];
    const userAnswer = answerInput.value;
    const correct = isCorrect(userAnswer, riddle.answer);

    results.push({
      question: riddle.question,
      userAnswer: userAnswer.trim() || '(לא ניתנה תשובה)',
      correctAnswer: riddle.answer,
      hintUsed,
      correct,
    });

    feedback.classList.remove('hidden');
    if (correct) {
      feedback.textContent = pick(PRAISES);
      feedback.className = 'feedback correct';
    } else {
      feedback.textContent = `${pick(CONSOLATIONS)} ${riddle.answer}`;
      feedback.className = 'feedback wrong';
    }

    answerInput.disabled = true;
    submitBtn.disabled = true;
    nextBtn.classList.remove('hidden');
    nextBtn.textContent = current + 1 < order.length ? 'לחידה הבאה ←' : 'לדוח הסופי ←';
  }

  function nextRiddle() {
    current++;
    if (current >= order.length) {
      showReport();
    } else {
      showRiddle();
    }
  }

  function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
  }

  function showReport() {
    quizScreen.classList.add('hidden');
    reportScreen.classList.remove('hidden');
    const score = results.filter((r) => r.correct).length;
    scoreHeadline.textContent = `הציון הסופי: ${score} מתוך ${results.length}`;
    scoreSub.textContent =
      score === results.length
        ? 'ניקוד מושלם! 🏆'
        : score >= results.length * 0.7
        ? 'כל הכבוד, תוצאה מצוינת! ⭐'
        : 'יפה מאוד, אפשר תמיד לשחק שוב! 💪';
    reportBody.innerHTML = '';
    results.forEach((r, i) => {
      const tr = document.createElement('tr');
      tr.className = r.correct ? 'row-correct' : 'row-wrong';
      tr.innerHTML = `
        <td>${i + 1}</td>
        <td>${escapeHtml(r.question)}</td>
        <td>${escapeHtml(r.userAnswer)}</td>
        <td>${escapeHtml(r.correctAnswer)}</td>
        <td>${r.hintUsed ? 'כן' : 'לא'}</td>
        <td>${r.correct ? '✔️' : '❌'}</td>
      `;
      reportBody.appendChild(tr);
    });
  }

  startBtn.addEventListener('click', startQuiz);
  hintBtn.addEventListener('click', useHint);
  submitBtn.addEventListener('click', submitAnswer);
  nextBtn.addEventListener('click', nextRiddle);
  restartBtn.addEventListener('click', startQuiz);
  answerInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
      if (!answered) submitAnswer();
      else nextRiddle();
    }
  });
})();
