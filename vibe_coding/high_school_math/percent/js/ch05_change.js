'use strict';
/* פרק 5: שינוי באחוזים כמכפיל. מכונת המכפיל, עלייה ואז ירידה, הנחה על הנחה, חזרה אחורה, וכמה צריך כדי לחזור */

// פסים אופקיים להשוואה: [{ v, t, cls }], max לסקאלה
function hbars(svg, rows, max) {
  S.clear(svg);
  const W = 640, L = 120, R = W - 70, h = 30, gap = 14;
  rows.forEach((r, k) => {
    const y = 8 + k * (h + gap), w = (R - L) * r.v / max;
    S.text(svg, L - 10, y + h / 2 + 6, r.t, { class: 'hb-t', 'text-anchor': 'end' });
    S.el('rect', { x: L, y, width: Math.max(w, 1), height: h, rx: 6, class: 'hb ' + (r.cls || '') }, svg);
    S.text(svg, L + w + 8, y + h / 2 + 6, fmt(r.v), { class: 'hb-v' });
  });
  if (rows.ref) S.el('line', { x1: L + (R - L) * rows.ref / max, x2: L + (R - L) * rows.ref / max, y1: 2, y2: rows.length * (h + gap), class: 'hb-ref' }, svg);
}
const pctTxt = c => (c > 0 ? '+' : c < 0 ? '−' : '') + fmt(Math.abs(c), 2) + '%';

App.add({
  id: 'ch5', short: 'המכפיל', icon: '📈', title: 'שינוי באחוזים: המכפיל',
  desc: 'למה +50% ואז −50% לא מחזיר להתחלה',
  intro: `<p>זה הפרק הכי חשוב. רעיון אחד פשוט: <b>שינוי באחוזים הוא כפל</b>. ברגע שרואים את זה, כל המלכודות של אחוזים מתפרקות.</p>`,
  steps: [
    {
      t: 'מכונת המכפיל',
      r(el, done) {
        el.innerHTML = `<p>משחק מחשב עולה ${ILS(80)}. מה קורה למחיר כשהוא משתנה באחוזים?</p>
          <p class="do">הזז את המחוון וצפה במכפיל.</p>
          <div class="controls"><label class="sl wide"><span>שינוי</span><input type="range" min="-90" max="200" step="5" value="0" class="ch"><output></output></label></div>
          <div class="machine2"><span class="mbox">${ILS(80)}</span><span class="marrow">×<b class="mul"></b></span><span class="mbox out"></span></div>
          <svg class="board hb2" viewBox="0 0 640 90"></svg>
          <div class="mh"></div>
          <div class="after" hidden><div class="aha"><b>עלייה של ${M('p%')} = כפל ב-${M('(1 + p/100)')}. ירידה של ${M('p%')} = כפל ב-${M('(1 − p/100)')}.</b>
          <br>${M('+20%')} זה ${M('× 1.2')}. ${M('−15%')} זה ${M('× 0.85')}. ${M('+100%')} זה ${M('× 2')}. ו-${M('−100%')}? ${M('× 0')}, הכול נעלם. לכן אי אפשר לרדת ביותר מ-${M('100%')}, אבל אפשר לעלות כמה שרוצים.
          <br>למה זה טוב? כי במקום שני צעדים ("חשב ${M('15%')}, ואז הורד") עושים צעד אחד: ${M('80 × 0.85 = 68')}.</div></div>`;
        const ch = el.querySelector('.ch'), svg = el.querySelector('.hb2');
        let mis;
        function draw() {
          const c = +ch.value, m = 1 + c / 100;
          ch.nextElementSibling.textContent = pctTxt(c);
          el.querySelector('.mul').textContent = fmt(m);
          el.querySelector('.out').innerHTML = ILS(80 * m);
          const rows = [{ v: 80, t: 'לפני', cls: 'a' }, { v: 80 * m, t: 'אחרי', cls: c >= 0 ? 'up' : 'down' }]; rows.ref = 80;
          hbars(svg, rows, 240);
          mis && mis.check({ c, m });
        }
        mis = Missions(el.querySelector('.mh'), [
          { t: `איזה שינוי נותן ${M('× 0.85')}?`, ok: s => s.c === -15 },
          { t: 'איזה שינוי מכפיל את המחיר?', ok: s => s.c === 100 },
          { t: 'איזה שינוי חוצה את המחיר?', ok: s => s.c === -50 },
          { t: `איזה שינוי נותן ${M('× 3')}?`, ok: s => s.c === 200 },
          { t: `מחיר חדש ${ILS(100)}: כמה אחוז עלייה?`, ok: s => s.c === 25 },
        ], () => { el.querySelector('.after').hidden = false; done(); });
        ch.oninput = draw; draw();
      },
    },
    {
      t: 'עלייה ואז ירידה',
      r(el, done) {
        el.innerHTML = `<p>מניה של חברת רובוטים שווה ${ILS(100)}. ביום שני עולה ב-${M('a%')}, ביום שלישי יורדת באותו ${M('a%')}. חוזרת ל-100?</p>
          <p class="do">שחק עם ${M('a')} ועם הסדר.</p>
          <div class="controls"><label class="sl wide"><span>${M('a')}</span><input type="range" min="0" max="100" step="5" value="10" class="aa"><output></output></label>
            <button class="btn sm ord">סדר: קודם עלייה</button></div>
          <div class="play"><div><svg class="board hb3" viewBox="0 0 640 132"></svg><div class="panel ud"></div></div>
            <div><svg class="board udc" viewBox="0 0 360 230"></svg><div class="muted small" style="text-align:center">המחיר הסופי כפונקציה של ${M('a')}</div></div></div>
          <div class="mh"></div>
          <div class="after" hidden><div class="aha"><b>${M('(1 + a)(1 − a) = 1 − a²')}</b>. זה תמיד פחות מ-1! העלייה והירידה הן אותו <u>אחוז</u>, אבל לא אותה <u>כמות</u>: הירידה מחושבת מסכום גדול יותר.
          <br>והסדר? לא משנה. ${M('1.5 × 0.5 = 0.5 × 1.5')}. כפל הוא חילופי.
          <br>זו גם הסיבה שהגרף הוא פרבולה הפוכה: ${M('y = 100(1 − a²)')}, כש-${M('a')} כשבר עשרוני.</div></div>`;
        const aa = el.querySelector('.aa'), svg = el.querySelector('.hb3'), cs = el.querySelector('.udc');
        let upFirst = true, mis, flipped = false;
        el.querySelector('.ord').onclick = e => { upFirst = !upFirst; flipped = true; e.target.textContent = upFirst ? 'סדר: קודם עלייה' : 'סדר: קודם ירידה'; draw(); };
        function draw() {
          const a = +aa.value / 100;
          aa.nextElementSibling.textContent = fmt(a * 100) + '%';
          const m1 = upFirst ? 1 + a : 1 - a, m2 = upFirst ? 1 - a : 1 + a, v1 = 100 * m1, v2 = v1 * m2;
          const rows = [{ v: 100, t: 'ראשון', cls: 'a' }, { v: v1, t: 'שני', cls: upFirst ? 'up' : 'down' }, { v: v2, t: 'שלישי', cls: 'fin' }]; rows.ref = 100;
          hbars(svg, rows, 200);
          el.querySelector('.ud').innerHTML = `${M(`100 × ${fmt(m1)} × ${fmt(m2)} = ${fmt(v2)}`)} · ${v2 < 100 - 1e-9 ? `הפסד של <b>${P(100 - v2)}</b>` : 'חזרה בדיוק ל-100'}`;
          const pts = []; for (let x = 0; x <= 100; x += 2) pts.push([x, 100 * (1 - (x / 100) ** 2)]);
          lineChart(cs, [{ pts, cls: 'c1' }], { xMax: 100, yMax: 100, xStep: 20, xLbl: 'a %', marks: [{ x: a * 100, y: v2, t: fmt(v2) }] });
          mis && mis.check({ a: Math.round(a * 100), v2, upFirst, flipped });
        }
        mis = Missions(el.querySelector('.mh'), [
          { t: `${M('a = 50%')}: לאן מגיעים?`, ok: s => s.a === 50 },
          { t: `מצא ${M('a')} שבו מפסידים בדיוק ${M('4%')}`, ok: s => s.a === 20 },
          { t: 'הפוך את הסדר: האם זה משנה משהו?', ok: s => s.flipped && !s.upFirst },
          { t: `מצא ${M('a')} שבו נשאר רק ${M('19%')}`, ok: s => s.a === 90 },
        ], () => { el.querySelector('.after').hidden = false; done(); });
        aa.oninput = draw; draw();
      },
    },
    {
      t: 'הנחה על הנחה',
      r(el, done) {
        el.innerHTML = `<p>בחנות כתוב: "${M('20%')} הנחה, ועוד ${M('10%')} הנחה בקופה". זה ${M('30%')}?</p>
          <div class="panel">${M('× 0.8 × 0.9 = × 0.72')} ← כלומר הנחה של <b>${M('28%')}</b>, לא ${M('30%')}. ההנחה השנייה היא ${M('10%')} מהמחיר <u>שכבר ירד</u>.</div>
          <p class="do">6 שאלות. חשוב עם מכפילים.</p><div class="qh"></div>`;
        const pool = shuffle([[-20, -10], [-50, -50], [10, 10], [-10, 20], [-25, -20], [50, -20], [20, 20], [-30, -30], [100, -50], [-40, 50], [10, -10], [25, -20]]);
        Quiz(el.querySelector('.qh'), {
          rounds: 6,
          gen(i) {
            const [a, b] = pool[i], m = (1 + a / 100) * (1 + b / 100), tot = Math.round((m - 1) * 10000) / 100;
            const word = c => c > 0 ? `עלייה של ${M(c + '%')}` : `ירידה של ${M(-c + '%')}`;
            return numQ(`${word(a)}, ואחריה ${word(b)}. מה השינוי הכולל באחוזים? (ירידה: כתוב מספר שלילי)`, tot, {
              suffix: '%', unit: '%', hint: `המכפילים: ${M(`× ${fmt(1 + a / 100)}`)} ו-${M(`× ${fmt(1 + b / 100)}`)}. כפול אותם.`,
              explain: `${M(`${fmt(1 + a / 100)} × ${fmt(1 + b / 100)} = ${fmt(m, 4)}`)}, כלומר ${tot === 0 ? 'אין שינוי בכלל!' : tot > 0 ? `עלייה של ${P(tot)}` : `ירידה של ${P(-tot)}`}${a + b !== tot ? ` (ולא ${M(pctTxt(a + b))}).` : '. במקרה הזה דווקא יצא כמו סכום.'}`,
            });
          },
          onDone: () => done(),
        });
      },
    },
    {
      t: 'חזרה אחורה',
      r(el, done) {
        el.innerHTML = `<p>מכפילים עובדים גם הפוך. אם אחרי עלייה של ${M('25%')} המחיר ${ILS(150)}, אז ${M('מקורי × 1.25 = 150')}, ולכן ${M('מקורי = 150 ÷ 1.25 = 120')}.</p>
          <div class="tip">כפל ← חילוק. <b>לא</b> מורידים ${M('25%')} מ-150 (זה נותן 112.5, טעות קלסית).</div>
          <p class="do">6 שאלות.</p><div class="qh"></div>`;
        const pool = shuffle([[25, 150, 'מחיר הכרטיס להופעה עלה ב-25% ועכשיו הוא'], [-20, 64, 'אחרי ירידה של 20% במשקל, הכלב של השכנים שוקל ק״ג'], [10, 77, 'אחרי תוספת של 10%, המשכורת של המדריך בחוג היא'], [-40, 90, 'במבצע של 40% הנחה, הקורקינט עולה'], [50, 33, 'הצמח של עומר גדל ב-50% וגובהו עכשיו ס״מ'], [-15, 170, 'אחרי 15% הנחה, האוזניות עולות'], [8, 54, 'אחרי התייקרות של 8%, הארוחה עולה'], [-75, 30, 'אחרי ירידה של 75%, נשארו במחסן ארגזים']]);
        Quiz(el.querySelector('.qh'), {
          rounds: 6,
          gen(i) {
            const [c, v, t] = pool[i], m = 1 + c / 100, o = v / m;
            const txt = t.replace(/(\d+)%/, (_, d) => M(d + '%'));
            const unit = /ק״ג|ס״מ|ארגזים/.exec(t);
            return numQ(`${unit ? txt.replace(unit[0], `${M(String(v))} ${unit[0]}`) : `${txt} ${ILS(v)}`}. כמה היה לפני?`, o, {
              hint: `מה המכפיל? ${M(`? × ${fmt(m)} = ${v}`)}`,
              explain: `${M(`${v} ÷ ${fmt(m)} = ${fmt(o)}`)}. בדיקה: ${M(`${fmt(o)} × ${fmt(m)} = ${v}`)} ✓`,
            });
          },
          onDone: () => done(),
        });
      },
    },
    {
      t: 'כמה צריך כדי לחזור?',
      r(el, done) {
        el.innerHTML = `<p>קבוצת כדורגל איבדה ${M('x%')} מהאוהדים. בכמה אחוזים היא צריכה לגדול עכשיו כדי לחזור לגודל המקורי?</p>
          <p class="do">הזז את ${M('x')}.</p>
          <div class="controls"><label class="sl wide"><span>ירידה ${M('x')}</span><input type="range" min="0" max="95" step="5" value="10" class="xx"><output></output></label></div>
          <div class="play"><div class="panel back"></div><svg class="board bc" viewBox="0 0 360 230"></svg></div>
          <div class="mh"></div>
          <div class="after" hidden><div class="aha"><b>אחרי ירידה של ${M('x')} נשאר ${M('1 − x')}, וכדי לחזור צריך לכפול ב-${M(F(1, '1 − x'))}.</b>
          <br>ירידה של ${M('50%')} דורשת עלייה של ${M('100%')}. ירידה של ${M('90%')} דורשת ${M('900%')}! לכן משקיעים אומרים: הכי חשוב זה לא להפסיד הרבה. ירידות קשות "לתקן" הרבה יותר ממה שנראה.
          <br>הגרף ממריא לאינסוף ליד ${M('x = 100%')}: מאפס אי אפשר לחזור בשום אחוז.</div></div>`;
        const xx = el.querySelector('.xx'), cs = el.querySelector('.bc');
        let mis;
        function draw() {
          const x = +xx.value / 100, need = 100 * x / (1 - x);
          xx.nextElementSibling.textContent = fmt(x * 100) + '%';
          el.querySelector('.back').innerHTML = `<p>${M(`× ${fmt(1 - x)}`)} ואז ${M(`× ${fmt(1 / (1 - x), 3)}`)} = ${M('× 1')}</p><p style="font-size:1.4rem">צריך לעלות ב-<b>${P(need, 1)}</b></p>`;
          const pts = []; for (let t = 0; t <= 92; t += 1) pts.push([t, 100 * t / (100 - t)]);
          lineChart(cs, [{ pts, cls: 'c2' }, { pts: [[0, 0], [92, 92]], cls: 'c3' }], { xMax: 95, yMax: 1000, xStep: 20, xLbl: 'x %', marks: [{ x: x * 100, y: Math.min(need, 1000), t: fmt(need, 0) + '%' }], hlines: [] });
          mis && mis.check(Math.round(x * 100));
        }
        mis = Missions(el.querySelector('.mh'), [
          { t: `ירידה של ${M('20%')}: כמה צריך לעלות?`, ok: x => x === 20 },
          { t: `מצא את הירידה שדורשת עלייה של ${M('100%')}`, ok: x => x === 50 },
          { t: `מצא את הירידה שדורשת עלייה של ${M('300%')}`, ok: x => x === 75 },
          { t: `ומה עם ${M('90%')}?`, ok: x => x === 90 },
        ], () => { el.querySelector('.after').hidden = false; done(); });
        xx.oninput = draw; draw();
      },
    },
  ],
});
