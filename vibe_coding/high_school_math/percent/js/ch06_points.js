'use strict';
/* פרק 6: אחוזים מול נקודות אחוז, ויחסי מול מוחלט (כותרות מפחידות עם מערך של 1000 נקודות) */

App.add({
  id: 'ch6', short: 'נקודות אחוז', icon: '⚖️', title: 'אחוזים, נקודות אחוז, וכותרות מפחידות',
  desc: 'מ-4% ל-5% זה עלייה של 1% או של 25%?',
  intro: `<p>כשהדבר שמשתנה הוא <b>בעצמו אחוז</b> (ריבית, אחוז הצבעה, סיכוי), יש שתי דרכים לתאר את השינוי. שתיהן נכונות, ושתיהן משמשות להטעות.</p>`,
  steps: [
    {
      t: 'נקודת אחוז',
      r(el, done) {
        el.innerHTML = `<p>הריבית בבנק עלתה מ-${M('4%')} ל-${M('5%')}. העיתון אחד כותב "עלתה ב-${M('1%')}", ועיתון שני כותב "עלתה ב-${M('25%')}". מי צודק?</p>
          <p class="do">הזז את שני המחוונים.</p>
          <div class="controls"><label class="sl wide"><span>לפני</span><input type="range" min="0.5" max="60" step="0.5" value="4" class="pa"><output></output></label>
            <label class="sl wide"><span>אחרי</span><input type="range" min="0.5" max="60" step="0.5" value="5" class="pb"><output></output></label></div>
          <div class="vs"><div class="panel big2"><span class="muted small">שינוי ב<b>נקודות אחוז</b> (חיסור)</span><div class="pp"></div></div>
            <div class="panel big2"><span class="muted small">שינוי ב<b>אחוזים</b> (יחסי)</span><div class="pr"></div></div></div>
          <div class="mh"></div>
          <div class="after" hidden><div class="aha"><b>נקודת אחוז</b> היא הפרש פשוט בין שני אחוזים: ${M('5% − 4% = 1')} נקודת אחוז.
          <br><b>שינוי באחוזים</b> הוא כמה השתנה ביחס להתחלה: ${M('1 ÷ 4 = 25%')}.
          <br>שניהם נכונים. הבעיה מתחילה כשמישהו אומר "${M('1%')}" כשהוא מתכוון לנקודת אחוז, או בוחר את המספר שנשמע דרמטי יותר.</div></div>`;
        const pa = el.querySelector('.pa'), pb = el.querySelector('.pb');
        let mis;
        function draw() {
          const a = +pa.value, b = +pb.value, pts = b - a, rel = 100 * (b - a) / a;
          pa.nextElementSibling.textContent = fmt(a, 1) + '%'; pb.nextElementSibling.textContent = fmt(b, 1) + '%';
          el.querySelector('.pp').innerHTML = `${M((pts >= 0 ? '+' : '−') + fmt(Math.abs(pts), 1))} נקודות`;
          el.querySelector('.pr').innerHTML = M(pctTxt(rel));
          mis && mis.check({ a, b, pts, rel });
        }
        mis = Missions(el.querySelector('.mh'), [
          { t: `נקודת אחוז אחת שהיא עלייה של ${M('50%')}`, ok: s => s.pts === 1 && Math.abs(s.rel - 50) < 1e-9 },
          { t: `עלייה של ${M('100%')} שהיא רק חצי נקודת אחוז`, ok: s => s.pts === .5 && Math.abs(s.rel - 100) < 1e-9 },
          { t: `עלייה של 5 נקודות שהיא רק ${M('10%')}`, ok: s => s.pts === 5 && Math.abs(s.rel - 10) < 1e-9 },
          { t: `ירידה של ${M('50%')} שהיא ירידה של 20 נקודות`, ok: s => s.pts === -20 && Math.abs(s.rel + 50) < 1e-9 },
        ], () => { el.querySelector('.after').hidden = false; done(); });
        pa.oninput = pb.oninput = draw; draw();
      },
    },
    {
      t: 'תרגול: נקודות או אחוזים',
      r(el, done) {
        el.innerHTML = `<p class="do">6 שאלות.</p><div class="qh"></div>`;
        const items = shuffle([
          ['מפלגה עלתה בסקרים מ-8% ל-10%.', 8, 10], ['אחוז ההצבעה ירד מ-70% ל-63%.', 70, 63], ['הסיכוי לגשם עלה מ-20% ל-30%.', 20, 30],
          ['הריבית ירדה מ-6% ל-4.5%.', 6, 4.5], ['אחוז הזריקות של עומר מהעונשין עלה מ-60% ל-75%.', 60, 75], ['סוללת הטלפון ירדה מ-80% ל-20%.', 80, 20],
          ['ציון המבחן עלה מ-50 ל-60 (מתוך 100).', 50, 60], ['אחוז האבטלה עלה מ-3% ל-4%.', 3, 4],
        ]);
        Quiz(el.querySelector('.qh'), {
          rounds: 6,
          gen(i) {
            const [t, a, b] = items[i], txt = t.replace(/(\d+(\.\d+)?)%/g, (_, d) => M(d + '%')), pts = b - a, rel = 100 * (b - a) / a;
            if (i % 2 === 0) return numQ(`${txt} בכמה <b>נקודות אחוז</b> השתנה? (ירידה: מספר שלילי)`, pts, { hint: 'נקודות אחוז זה חיסור פשוט.', explain: `${M(`${fmt(b)} − ${fmt(a)} = ${fmt(pts)}`)} נקודות. (ביחס להתחלה זה ${M(pctTxt(rel))}.)` });
            return numQ(`${txt} בכמה <b>אחוזים</b> השתנה, ביחס להתחלה? (ירידה: מספר שלילי)`, rel, { suffix: '%', unit: '%', tol: 0.051, hint: `השינוי חלקי ההתחלה: ${M(`(${fmt(b)} − ${fmt(a)}) ÷ ${fmt(a)}`)}`, explain: `${M(`${fmt(pts)} ÷ ${fmt(a)} = ${fmt(rel / 100, 4)}`)}, כלומר ${M(pctTxt(rel))}. (בנקודות אחוז זה ${M(fmt(pts))}.)` });
          },
          onDone: () => done(),
        });
      },
    },
    {
      t: 'כותרות מפחידות',
      r(el, done) {
        el.innerHTML = `<p>כותרת: <b>"מחקר: מי שאוכל נקניקיות כל יום, הסיכון שלו לחלות ב-X עולה ב-${M('100%')}!"</b> מפחיד. אבל ${M('100%')} ממה?</p>
          <p class="do">כל נקודה היא אדם אחד מתוך 1000. הזז את המחוונים.</p>
          <div class="controls"><label class="sl wide"><span>חולים בלי נקניקיות (מתוך 1000)</span><input type="range" min="1" max="100" step="1" value="2" class="rb"><output></output></label>
            <label class="sl wide"><span>עלייה יחסית</span><input type="range" min="0" max="300" step="25" value="100" class="rr"><output></output></label></div>
          <div class="vs"><div><b>בלי</b><svg class="board ia1" viewBox="0 0 400 250"></svg></div><div><b>עם</b><svg class="board ia2" viewBox="0 0 400 250"></svg></div></div>
          <div class="panel heads"></div><div class="mh"></div>
          <div class="after" hidden><div class="aha"><b>סיכון יחסי</b> ("עולה ב-${M('100%')}") אומר פי כמה. <b>סיכון מוחלט</b> ("עוד 2 מתוך 1000") אומר כמה אנשים באמת. כשהסיכון הבסיסי קטן, גם הכפלה שלו קטנה.
          <br>כותרות אוהבות סיכון יחסי, כי הוא נשמע דרמטי. השאלה שתמיד צריך לשאול: <b>כמה זה מתוך כמה, לפני ואחרי?</b></div></div>`;
        const rb = el.querySelector('.rb'), rr = el.querySelector('.rr');
        function arr(svg, n) {
          S.clear(svg);
          for (let k = 0; k < 1000; k++) { const c = k % 40, r = Math.floor(k / 40); S.el('circle', { cx: 5 + c * 9.8, cy: 5 + r * 9.8, r: 3.6, class: k < n ? 'ia-on' : 'ia-off' }, svg); }
        }
        let mis;
        function draw() {
          const b = +rb.value, rel = +rr.value, a = Math.round(b * (1 + rel / 100));
          rb.nextElementSibling.textContent = b; rr.nextElementSibling.textContent = '+' + rel + '%';
          arr(el.querySelector('.ia1'), b); arr(el.querySelector('.ia2'), Math.min(a, 1000));
          el.querySelector('.heads').innerHTML = `<p>📰 כותרת דרמטית: <b>"הסיכון עולה ב-${M(rel + '%')}!"</b></p>
            <p>🧮 כותרת כנה: <b>"מ-${b} ל-${a} מתוך 1000"</b>, כלומר עוד ${a - b} אנשים מתוך 1000 (${M(fmt((a - b) / 10, 1))} נקודות אחוז).</p>`;
          mis && mis.check({ b, rel, d: a - b });
        }
        mis = Missions(el.querySelector('.mh'), [
          { t: `"עלייה של ${M('100%')}" שהיא רק עוד אדם אחד מתוך 1000`, ok: s => s.rel === 100 && s.d === 1 },
          { t: `"עלייה של ${M('300%')}" שהיא עוד 3 אנשים`, ok: s => s.rel === 300 && s.d === 3 },
          { t: `עלייה של "רק" ${M('25%')} שהיא עוד 20 אנשים`, ok: s => s.rel === 25 && s.d === 20 },
        ], () => { el.querySelector('.after').hidden = false; done(); });
        rb.oninput = rr.oninput = draw; draw();
      },
    },
  ],
});
