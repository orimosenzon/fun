'use strict';
/* פרק 7: ריבית דריבית. החיסכון של עומר (פשוטה מול דריבית), חוק ה-72, ו-1% כל יום */

App.add({
  id: 'ch7', short: 'ריבית דריבית', icon: '🌱', title: 'ריבית דריבית: אחוזים שמתרבים',
  desc: 'מכפיל שחוזר על עצמו, וחוק ה-72',
  intro: `<p>מה קורה כשמפעילים את אותו מכפיל שוב ושוב? אלברט איינשטיין כנראה אף פעם לא אמר ש"ריבית דריבית היא הכוח החזק ביקום", אבל הציטוט הזה מסתובב כבר עשרות שנים כי משהו בו מרגיש נכון.</p>`,
  steps: [
    {
      t: 'החיסכון של עומר',
      r(el, done) {
        el.innerHTML = `<p>עומר קיבל ${ILS(1000)} מתנה לבר מצווה ושם אותם בחיסכון. יש שתי תוכניות:</p>
          <div class="vs"><div class="panel"><b class="t-simple">ריבית פשוטה</b>: כל שנה מקבלים ${M('r%')} <u>מה-1000 המקוריים</u>. אותו סכום כל שנה.</div>
          <div class="panel"><b class="t-comp">ריבית דריבית</b>: כל שנה מקבלים ${M('r%')} <u>ממה שיש עכשיו</u>, כולל הריבית של השנים הקודמות. כלומר כל שנה ${M('× (1 + r)')}.</div></div>
          <p class="do">שחק עם הריבית ועם מספר השנים.</p>
          <div class="controls"><label class="sl wide"><span>ריבית</span><input type="range" min="0.5" max="15" step="0.5" value="5" class="rt"><output></output></label>
            <label class="sl wide"><span>שנים</span><input type="range" min="1" max="50" step="1" value="10" class="yr"><output></output></label></div>
          <div class="play"><svg class="board cc" viewBox="0 0 520 300"></svg><div class="panel co"></div></div>
          <div class="mh"></div>
          <div class="after" hidden><div class="aha"><b>פשוטה גדלה בקו ישר. דריבית גדלה כמו כדור שלג.</b> בהתחלה הן כמעט זהות, ואחר כך הדריבית בורחת.
          <br>הנוסחה: אחרי ${M('n')} שנים יש ${M('1000 × (1 + r)ⁿ')}. זו <b>חזקה</b>, ולכן קוראים לזה גדילה <b>מעריכית</b> (אקספוננציאלית).
          <br>זה עובד גם נגדך: חוב בכרטיס אשראי עם ריבית גבוהה גדל בדיוק ככה.</div></div>`;
        const rt = el.querySelector('.rt'), yr = el.querySelector('.yr'), svg = el.querySelector('.cc');
        let mis;
        function draw() {
          const r = +rt.value / 100, n = +yr.value;
          rt.nextElementSibling.textContent = fmt(r * 100, 1) + '%'; yr.nextElementSibling.textContent = n;
          const simple = [], comp = [];
          for (let t = 0; t <= n; t++) { simple.push([t, 1000 * (1 + r * t)]); comp.push([t, 1000 * (1 + r) ** t]); }
          const fs = 1000 * (1 + r * n), fc = 1000 * (1 + r) ** n;
          lineChart(svg, [{ pts: simple, cls: 'c3', dots: n <= 20 }, { pts: comp, cls: 'c1', dots: n <= 20 }], { xMax: n, yMax: Math.max(fc, fs) * 1.08, xStep: n <= 10 ? 1 : n <= 25 ? 5 : 10, xLbl: 'שנים', hlines: [{ y: 2000, t: 'כפול' }] });
          el.querySelector('.co').innerHTML = `<p class="t-simple">פשוטה: <b>${ILS(fs, 0)}</b></p><p class="t-comp">דריבית: <b>${ILS(fc, 0)}</b></p><p class="muted small">ההפרש: ${ILS(fc - fs, 0)}</p>`;
          mis && mis.check({ r: Math.round(r * 1000) / 10, n, fc });
        }
        mis = Missions(el.querySelector('.mh'), [
          { t: `ב-${M('6%')}: אחרי כמה שנים לכל הפחות הדריבית מכפילה את הכסף?`, ok: s => s.r === 6 && s.n === 12 },
          { t: `מצא ריבית שמכפילה את הכסף תוך 10 שנים (בערך)`, ok: s => s.n === 10 && s.fc > 1950 && s.fc < 2100 },
          { t: `50 שנה ב-${M('10%')}. כמה יש בפשוטה, וכמה בדריבית?`, ok: s => s.r === 10 && s.n === 50 },
        ], () => { el.querySelector('.after').hidden = false; done(); });
        rt.oninput = yr.oninput = draw; draw();
      },
    },
    {
      t: 'חוק ה-72',
      r(el, done) {
        el.innerHTML = `<p>שמת לב? ב-${M('6%')} הכסף הוכפל אחרי 12 שנים, וב-${M('7.2%')} אחרי 10. ${M('6 × 12 = 72')}, ${M('7.2 × 10 = 72')}.</p>
          <div class="def"><b>חוק ה-72:</b> בריבית של ${M('r%')} לשנה, הכסף מוכפל אחרי <b>בערך ${M('72 ÷ r')}</b> שנים.</div>
          <table class="clothes"><tr><th>ריבית</th><th>לפי החוק</th><th>בדיוק</th></tr>${[2, 4, 6, 8, 9, 12, 18].map(r => `<tr><td>${P(r)}</td><td>${fmt(72 / r, 1)}</td><td>${fmt(Math.log(2) / Math.log(1 + r / 100), 2)}</td></tr>`).join('')}</table>
          <p class="muted small">למה דווקא 72? החשבון המדויק נותן מספר קרוב ל-70, אבל ל-72 יש המון מחלקים (2, 3, 4, 6, 8, 9, 12), אז קל לחשב איתו בראש. את הסיבה העמוקה תפגוש כשתלמד לוגריתמים.</p>
          <p class="do">5 שאלות. תשובה מקורבת מספיקה.</p><div class="qh"></div>`;
        const qs = shuffle([
          () => { const r = pick([3, 4, 8, 9, 12]); return numQ(`בערך אחרי כמה שנים מוכפל חיסכון בריבית של ${P(r)} לשנה?`, 72 / r, { tol: .6, explain: `${M(`72 ÷ ${r} = ${fmt(72 / r, 1)}`)}. בדיוק: ${fmt(Math.log(2) / Math.log(1 + r / 100), 2)}.` }); },
          () => numQ(`באיזו ריבית שנתית (בערך) כסף מוכפל תוך 6 שנים?`, 12, { suffix: '%', unit: '%', tol: .6, explain: `${M('72 ÷ 6 = 12')}, כלומר ${M('12%')}.` }),
          () => numQ(`בריבית של ${M('8%')}, פי כמה (בערך) יגדל הכסף אחרי 27 שנים?`, 8, { tol: .5, hint: 'אחרי כמה שנים הוא מוכפל? וכמה הכפלות יש ב-27 שנים?', explain: `הכפלה כל ${M('72 ÷ 8 = 9')} שנים. ב-27 שנים יש 3 הכפלות: ${M('2 × 2 × 2 = 8')}. בדיוק: ${fmt(1.08 ** 27, 2)}.` }),
          () => numQ(`אוכלוסייה של עיר גדלה ב-${M('2%')} בשנה. בעוד כמה שנים בערך היא תוכפל?`, 36, { tol: 1.6, explain: `${M('72 ÷ 2 = 36')} שנים.` }),
          () => numQ(`האינפלציה ${M('4%')} לשנה. בעוד כמה שנים בערך המחירים יוכפלו?`, 18, { tol: 1, explain: `${M('72 ÷ 4 = 18')}. כלומר מה שעולה היום 50 יעלה 100 בעוד 18 שנה.` }),
          () => numQ(`בריבית של ${M('9%')}, ${ILS(1000)} יהפכו בערך לכמה אחרי 24 שנים?`, 8000, { tol: 900, explain: `הכפלה כל ${M('72 ÷ 9 = 8')} שנים, אז 3 הכפלות: ${M('1000 × 8 = 8000')}. בדיוק: ${ILS(1000 * 1.09 ** 24, 0)}.` }),
        ]);
        Quiz(el.querySelector('.qh'), { rounds: 5, gen: i => qs[i](), onDone: () => done() });
      },
    },
    {
      t: 'אחוז אחד כל יום',
      r(el, done) {
        el.innerHTML = `<p>שני חברים מתחילים ללמוד שחמט באותה רמה. אחד משתפר ב-${M('1%')} כל יום. השני "רק" נחלש ב-${M('1%')} כל יום. כמה זה משנה?</p>
          <div class="controls"><label class="sl wide"><span>יום</span><input type="range" min="0" max="365" step="1" value="30" class="dd"><output></output></label></div>
          <div class="play"><svg class="board dc" viewBox="0 0 520 300"></svg><div class="panel do2"></div></div>
          <div class="mh"></div>
          <div class="after" hidden><div class="aha"><b>${M('1.01³⁶⁵ ≈ 37.8')} ו-${M('0.99³⁶⁵ ≈ 0.026')}.</b> הבדל קטן בכל יום, ופער של יותר מפי 1000 בסוף שנה. כשמכפילים שוב ושוב, הבדלים קטנים במכפיל הופכים לענקיים בתוצאה.
          <br>(זה מודל, לא חוק טבע. אף אחד לא משתפר ב-${M('1%')} כל יום לנצח. אבל הרעיון של הצטברות אמיתי לגמרי.)</div></div>`;
        const dd = el.querySelector('.dd'), svg = el.querySelector('.dc');
        let mis;
        function draw() {
          const d = +dd.value; dd.nextElementSibling.textContent = d;
          const up = [], dn = []; for (let t = 0; t <= Math.max(d, 1); t++) { up.push([t, 1.01 ** t]); dn.push([t, .99 ** t]); }
          lineChart(svg, [{ pts: up, cls: 'c1' }, { pts: dn, cls: 'c2' }], { xMax: Math.max(d, 10), yMax: Math.max(1.01 ** d * 1.1, 2), xLbl: 'ימים', hlines: [{ y: 1, t: 'נקודת ההתחלה' }] });
          el.querySelector('.do2').innerHTML = `<p class="t-comp">משתפר: <b>${M('× ' + fmt(1.01 ** d, 2))}</b></p><p class="t-down">נחלש: <b>${M('× ' + fmt(.99 ** d, 3))}</b></p><p class="muted small">היחס ביניהם: פי ${fmt((1.01 / .99) ** d, 1)}</p>`;
          mis && mis.check(d);
        }
        mis = Missions(el.querySelector('.mh'), [
          { t: 'באיזה יום המשתפר כבר טוב פי 2 מההתחלה?', ok: d => d === 70 },
          { t: 'באיזה יום הוא טוב פי 10?', ok: d => d === 232 },
          { t: 'סוף שנה: יום 365', ok: d => d === 365 },
        ], () => { el.querySelector('.after').hidden = false; done(); });
        dd.oninput = draw; draw();
      },
    },
  ],
});
