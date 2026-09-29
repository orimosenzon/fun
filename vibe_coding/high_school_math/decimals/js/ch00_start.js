'use strict';
/* פתיחה: ברוך הבא, מקרא הצבעים, חמש שאלות "לפני" (חוזרים אליהן בפרק 11) ומפת הדרך */

// השאלות האלה נשמרות, ובפרק האתגר עידו רואה מה ענה בהתחלה ומה הוא עונה עכשיו
const PRETEST = [
  { id: 'cmp', q: `מה גדול יותר: ${M('0.8')} או ${M('0.75')}?`, type: 'choice', options: ['0.8', '0.75', 'הם שווים'], answer: '0.8',
    why: `${M('0.8')} הוא 8 עשיריות, כלומר 80 מאיות. ${M('0.75')} הוא רק 75 מאיות. בשקלים: 80 אגורות מול 75 אגורות. (פרק 5)` },
  { id: 'count', q: `סופרים בקפיצות של עשירית: ${M('0.7, 0.8, 0.9')}, ... מה המספר הבא?`, type: 'choice', options: ['0.10', '1', '0.91'], answer: '1',
    why: `עשר עשיריות הן שלם אחד, אז אחרי ${M('0.9')} בא ${M('1')} (אפשר לכתוב גם ${M('1.0')}). ${M('0.10')} הוא בכלל עשר מאיות, וזה שווה ל-${M('0.1')}. (פרק 1)` },
  { id: 'add', q: `כמה זה ${M('2.9 + 0.2')}?`, type: 'input', answer: N.of(31, 1),
    why: `9 עשיריות ועוד 2 עשיריות הן 11 עשיריות, כלומר שלם אחד ועוד עשירית. ${M('2.9 + 0.2 = 3.1')}. (פרק 6)` },
  { id: 'money', q: `כמה אגורות יש ב-${M('1.5')} שקלים?`, type: 'input', answer: N.of(150), unit: 'אגורות',
    why: `${M('1.5')} זה שקל וחצי. חצי שקל הוא 50 אגורות, אז ${M('1.5')} שקלים הם 150 אגורות. (פרק 2)` },
  { id: 'between', q: `כתוב מספר כלשהו שנמצא בין ${M('0.4')} ל-${M('0.5')}.`, type: 'input', answer: null,
    ok: v => v !== null && v > N.of(4, 1) && v < N.of(5, 1),
    why: `יש אינסוף כאלה! למשל ${M('0.45')}, ${M('0.41')}, ${M('0.499')}. בין כל שני מספרים יש עוד מספרים, רק צריך "לעשות זום". (פרק 4)` },
];
function pretestCorrect(t, a) {
  if (a === undefined || a === null) return null;
  if (t.type === 'choice') return a === t.answer;
  const v = N.parse(a);
  return t.ok ? t.ok(v) : v === t.answer;
}

App.add({
  id: 'ch0', short: 'פתיחה', icon: '👋',
  render(el) {
    el.innerHTML = `
      <h1>היי עידו! 👋</h1>
      <p class="lead">זה המקום שלך ללמוד שברים עשרוניים. מההתחלה, בקצב שלך, ובעיקר בידיים.</p>
      <p>שבר עשרוני הוא מספר עם נקודה, כמו ${M('0.5')} או ${M('3.75')}. הם בכל מקום: מחירים בסופר, ציונים, מרחקים, משקל, זמנים בריצה. הרבה אנשים למדו אותם בעל פה ולא באמת הבינו. כאן נבנה אותם מאפס, ונבין למה כל כלל עובד. ככה לא צריך לזכור כמעט כלום.</p>

      <div class="card">
        <h4>איך זה עובד</h4>
        <ul>
          <li>יש 11 פרקים, וכל פרק מחולק ל<b>תחנות</b> קצרות. עוברים תחנה אחרי תחנה עם הכפתור "לתחנה הבאה".</li>
          <li>כמעט בכל תחנה יש משהו לעשות: לצבוע, לגרור, לבנות, לשחק. <b>קודם תנסה, ורק אחר כך תקרא</b> את ההסבר.</li>
          <li>טעויות זה בסדר גמור. כל טעות מראה לך בדיוק מה עוד לא ברור, ואז מסבירים לך.</li>
          <li>ההתקדמות נשמרת. אפשר לסגור ולחזור בדיוק לאותו מקום.</li>
        </ul>
      </div>

      <div class="card">
        <h4>🎨 הצבעים של המספרים</h4>
        <p>בכל האפליקציה, לכל מקום במספר יש צבע קבוע. תסתכל על הצבע, והוא יגיד לך מה הספרה שווה:</p>
        <div class="ro-big bignum">${digitsHTML('32.451')}</div>
        <div class="legend">
          <span><b class="t-p1">3</b> עשרות</span><span><b class="t-p0">2</b> שלמים</span><span><b>.</b> הנקודה העשרונית</span>
          <span><b class="t-pm1">4</b> עשיריות</span><span><b class="t-pm2">5</b> מאיות</span><span><b class="t-pm3">1</b> אלפיות</span>
        </div>
        <p class="muted small">אל תדאג אם המילים האלה עוד לא אומרות לך הרבה. בשביל זה אנחנו כאן.</p>
      </div>

      <div class="card" id="pre-card"></div>

      <h3>מפת הדרך</h3>
      <div class="journey" id="journey"></div>
      <button data-go="ch1" class="btn primary big">בוא נתחיל ←</button>`;
    App.renderJourney(el.querySelector('#journey'));
    this.drawPre(el.querySelector('#pre-card'));
  },

  drawPre(box) {
    const pre = Store.get('pre', null);
    if (pre && pre.done) {
      box.innerHTML = `<h4>✅ ענית על שאלות הפתיחה</h4><p class="muted">נחזור אליהן בפרק האתגר בסוף, ונראה מה השתנה.</p>`;
      return;
    }
    const ans = (pre && pre.ans) || {};
    box.innerHTML = `<h4>🤔 חמש שאלות לפני שמתחילים</h4>
      <p>תענה מה שנראה לך. <b>אל תדאג אם אתה לא בטוח</b>, ואל תחפש בגוגל. זה לא מבחן. לא אגיד לך עכשיו אם צדקת. בסוף הדרך נחזור לשאלות האלה ונראה מה השתנה.</p>
      <div class="pre-list">${PRETEST.map((t, k) => `<div class="pre-q" data-k="${k}">
        <div class="qline">${k + 1}. ${t.q}</div>
        ${t.type === 'choice'
          ? `<div class="opts">${t.options.map(o => `<button class="opt ${ans[t.id] === o ? 'on' : ''}" data-v="${esc(o)}">${/\d/.test(o) ? M(o) : o}</button>`).join('')}</div>`
          : `<div class="inrow"><input class="num-in" dir="ltr" inputmode="decimal" value="${esc(ans[t.id] || '')}">${t.unit ? `<span class="unit">${t.unit}</span>` : ''}</div>`}
      </div>`).join('')}</div>
      <div class="controls"><button class="btn primary" id="pre-save">שמור את התשובות</button><button class="btn" id="pre-skip">דלג</button><span class="muted small" id="pre-msg"></span></div>`;
    const save = () => Store.set('pre', { ans, done: false });
    box.querySelectorAll('.pre-q').forEach(qd => {
      const t = PRETEST[+qd.dataset.k];
      qd.querySelectorAll('.opt').forEach(b => b.onclick = () => {
        qd.querySelectorAll('.opt').forEach(x => x.classList.remove('on')); b.classList.add('on'); ans[t.id] = b.dataset.v; save();
      });
      const inp = qd.querySelector('input'); if (inp) inp.oninput = () => { ans[t.id] = inp.value.trim(); save(); };
    });
    box.querySelector('#pre-save').onclick = () => {
      const missing = PRETEST.filter(t => !ans[t.id]).length;
      if (missing) { box.querySelector('#pre-msg').textContent = `חסרות עוד ${missing} תשובות. אם אין לך מושג, אפשר לנחש.`; return; }
      Store.set('pre', { ans, done: true, when: Date.now() }); this.drawPre(box);
    };
    box.querySelector('#pre-skip').onclick = () => { Store.set('pre', { ans, done: true, skipped: true }); this.drawPre(box); };
  },
});
