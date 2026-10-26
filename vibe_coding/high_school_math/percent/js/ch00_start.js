'use strict';
/* פתיחה: ברוך הבא, ארבע חידות "לפני" (חוזרים אליהן באתגר הגמר) ומפת הדרך */

// התשובות נשמרות, ובאתגר הגמר עומר רואה מה ענה בהתחלה ומה הוא עונה עכשיו
const PRETEST = [
  { id: 'updown', q: `מחיר של אופניים עלה ב-${M('50%')}, ואחרי חודש ירד ב-${M('50%')}. עכשיו המחיר...`, type: 'choice',
    options: ['גבוה מהמחיר בהתחלה', 'נמוך מהמחיר בהתחלה', 'בדיוק כמו בהתחלה'], answer: 'נמוך מהמחיר בהתחלה',
    why: `העלייה היא ${M('× 1.5')} והירידה היא ${M('× 0.5')}. ביחד ${M('× 0.75')}, כלומר ירידה של ${M('25%')}. החצי שירד הוא חצי ממחיר <b>גדול יותר</b>. (פרק 5)` },
  { id: 'swap', q: `כמה זה ${M('8%')} מ-${M('50')}?`, type: 'input', answer: 4,
    why: `${M('8%')} מ-50 שווה ל-${M('50%')} מ-8, וחצי מ-8 זה 4. אפשר להחליף בין המספרים תמיד. (פרק 2)` },
  { id: 'back', q: `אחרי הנחה של ${M('20%')} חולצה עולה ${ILS(80)}. כמה עלתה לפני ההנחה?`, type: 'input', answer: 100, unit: '₪',
    why: `80 הם ${M('80%')} מהמחיר המקורי, אז ${M('1%')} הוא 1, והמחיר המקורי ${ILS(100)}. לא 96! ה-${M('20%')} מחושבים מהמחיר המקורי, לא מ-80. (פרק 4)` },
  { id: 'melon', q: `אבטיח שוקל ${M('10')} ק״ג, ו-${M('99%')} ממנו מים. הוא נשאר בשמש, חלק מהמים התאדו, ועכשיו רק ${M('98%')} ממנו מים. כמה הוא שוקל עכשיו?`, type: 'choice',
    options: ['9.9 ק״ג', '9.8 ק״ג', '9 ק״ג', '5 ק״ג'], answer: '5 ק״ג',
    why: `החלק היבש (${M('1%')}, כלומר ${M('0.1')} ק״ג) לא מתאדה. עכשיו הוא ${M('2%')} מהמשקל, אז המשקל כולו ${M('0.1 × 50 = 5')} ק״ג. מטורף אבל נכון. (פרק 8)` },
];
function pretestCorrect(t, a) {
  if (a === undefined || a === null || a === '') return null;
  if (t.type === 'choice') return a === t.answer;
  return close(num(a), t.answer);
}

App.add({
  id: 'ch0', short: 'פתיחה', icon: '👋',
  render(el) {
    el.innerHTML = `
      <h1>היי עומר! 👋</h1>
      <p class="lead">אחוזים נראים כמו נושא קטן: "כמה זה ${M('20%')} מ-150?" וזהו. אבל מתחת לזה מסתתרים כמה מהרעיונות הכי חזקים, ולפעמים הכי מטעים, במתמטיקה של החיים.</p>
      <p>בנקים, מבצעים בסופר, סקרים, כותרות בעיתון, סטטיסטיקה של שחקני כדורסל, ריבית על חיסכון, וירוסים שמתפשטים. בכולם יש אחוזים, ובכולם קל מאוד לטעות. אנשים מבוגרים, עיתונאים ואפילו רופאים נופלים במלכודות שתכיר כאן.</p>

      <div class="card">
        <h4>איך זה עובד</h4>
        <ul>
          <li>יש 10 פרקים, וכל פרק מחולק ל<b>תחנות</b> קצרות. בכל תחנה יש משהו לעשות: לצבוע, לגרור, להזיז מחוונים, לפתור.</li>
          <li><b>קודם תנסה, ורק אחר כך תקרא</b> את ההסבר. הרבה פעמים תגלה את הכלל לבד.</li>
          <li>נסתכל על אחוזים מהרבה כיוונים: ריבוע של מאה, פס כפול, מכפיל, גרף, עוגה. מי שרואה רעיון מכמה כיוונים באמת מבין אותו.</li>
          <li>בפרק 9 אתה הגיבור. עשר בעיות שקורות לעומר אחד שאתה אולי מכיר 😉</li>
          <li>ההתקדמות נשמרת. אפשר לסגור ולחזור בדיוק לאותו מקום.</li>
        </ul>
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
      box.innerHTML = `<h4>✅ ענית על חידות הפתיחה</h4><p class="muted">נחזור אליהן באתגר הגמר, ונראה אם משהו השתנה.</p>`;
      return;
    }
    const ans = (pre && pre.ans) || {};
    box.innerHTML = `<h4>🤔 ארבע חידות לפני שמתחילים</h4>
      <p>תענה מה שנראה לך, בלי מחשבון ובלי גוגל. זה לא מבחן, ולא אגיד לך עכשיו אם צדקת. חלק מהחידות האלה מפילות כמעט את כולם. בסוף הדרך נחזור אליהן.</p>
      <div class="pre-list">${PRETEST.map((t, k) => `<div class="pre-q" data-k="${k}">
        <div class="qline">${k + 1}. ${t.q}</div>
        ${t.type === 'choice'
          ? `<div class="opts">${t.options.map(o => `<button class="opt ${ans[t.id] === o ? 'on' : ''}" data-v="${esc(o)}">${o}</button>`).join('')}</div>`
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
      if (missing) { box.querySelector('#pre-msg').textContent = `חסרות עוד ${missing} תשובות. אם אין לך מושג, תנחש.`; return; }
      Store.set('pre', { ans, done: true, when: Date.now() }); this.drawPre(box);
    };
    box.querySelector('#pre-skip').onclick = () => { Store.set('pre', { ans, done: true, skipped: true }); this.drawPre(box); };
  },
});
