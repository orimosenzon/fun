'use strict';
/* פרק 3: כמה אחוז? חלק מתוך שלם. איזה מבחן הכי טוב, שבר לאחוז, וסקר בכיתה עם עוגה */

App.add({
  id: 'ch3', short: 'כמה אחוז?', icon: '📊', title: 'כמה אחוז זה?',
  desc: 'חלק מתוך שלם, השוואות, וסקר עם עוגה',
  intro: `<p>עד עכשיו ידענו את האחוז וחיפשנו כמות. עכשיו הפוך: יודעים את החלק ואת השלם, ושואלים <b>כמה אחוז</b>. ככה משווים דברים שאי אפשר להשוות ישר.</p>`,
  steps: [
    {
      t: 'איזה מבחן הכי טוב?',
      r(el, done) {
        const exams = [['📐 מתמטיקה', 17, 20], ['🧲 פיזיקה', 21, 25], ['🇬🇧 אנגלית', 44, 50], ['🏛️ היסטוריה', 26, 30]];
        el.innerHTML = `<p>עומר קיבל ארבעה מבחנים, וכל אחד עם מספר נקודות אחר. באיזה הוא הצליח הכי טוב?</p>
          <p class="do">לחץ על המבחנים לפי הסדר: מהטוב ביותר לפחות טוב. בלי מחשבון!</p>
          <div class="exams"></div><div class="ef"></div>
          <div class="after" hidden><div class="aha"><b>אחוז הופך כל שבר למכנה 100</b>, ואז אפשר להשוות. ${F(17, 20)} = ${F(85, 100)} (כפול 5), ${F(21, 25)} = ${F(84, 100)} (כפול 4), ${F(44, 50)} = ${F(88, 100)} (כפול 2). ההיסטוריה קשה יותר: ${M('26 ÷ 30 = 0.8666...')}, כלומר ${M('86.7%')}.
          <br>כלל: <b>${M('חלק ÷ שלם × 100')}</b>.</div></div>`;
        const order = exams.map((e, k) => k).sort((a, b) => exams[b][1] / exams[b][2] - exams[a][1] / exams[a][2]);
        let got = [], wrongs = 0;
        function draw() {
          el.querySelector('.exams').innerHTML = exams.map(([n, a, b], k) => {
            const pos = got.indexOf(k), p = 100 * a / b;
            return `<button class="exam ${pos >= 0 ? 'placed' : ''}" data-k="${k}" ${pos >= 0 ? 'disabled' : ''}>
              ${pos >= 0 ? `<span class="rank">${pos + 1}</span>` : ''}<b>${n}</b><span class="score">${M(`${a} / ${b}`)}</span>
              ${pos >= 0 ? `<span class="ebar"><i style="width:${p}%"></i></span><span>${P(p, 1)}</span>` : ''}</button>`;
          }).join('');
          el.querySelectorAll('.exam').forEach(b => b.onclick = () => {
            const k = +b.dataset.k;
            if (k === order[got.length]) { got.push(k); el.querySelector('.ef').innerHTML = ''; draw(); if (got.length === exams.length) { el.querySelector('.after').hidden = false; done(); } }
            else { wrongs++; b.classList.add('shake'); setTimeout(() => b.classList.remove('shake'), 500); el.querySelector('.ef').innerHTML = `<div class="trap">עוד לא. ${wrongs > 1 ? `רמז: נסה להפוך כל שבר לשבר עם מכנה 100. למשל ${F(17, 20)} = ${F('?', 100)}.` : 'תחשוב שוב.'}</div>`; }
          });
        }
        draw();
      },
    },
    {
      t: 'משבר לאחוז',
      r(el, done) {
        el.innerHTML = `<div class="vs"><div class="panel"><b>דרך 1: מכנה 100</b><br>${F(7, 25)} = ${F(28, 100)} = ${M('28%')}<div class="muted small">עובד כשהמכנה מתחלק ב-100 או מחלק אותו</div></div>
          <div class="panel"><b>דרך 2: חילוק</b><br>${M('7 ÷ 25 = 0.28')} ← ${M('28%')}<div class="muted small">עובד תמיד</div></div>
          <div class="panel"><b>דרך 3: צמצום ואז מכנה 100</b><br>${F(18, 24)} = ${F(3, 4)} = ${M('75%')}<div class="muted small">לפעמים שבר מכוער מתגלה כיפה</div></div></div>
          <p class="do">8 שאלות. אם התשובה לא שלמה, עגל לעשירית.</p><div class="qh"></div>`;
        const items = [[7, 25], [9, 20], [18, 24], [3, 8], [12, 40], [27, 30], [45, 60], [13, 50], [7, 8], [33, 300], [6, 15], [1, 40], [56, 70], [2, 3], [11, 12], [150, 120]];
        const pool = shuffle(items);
        Quiz(el.querySelector('.qh'), {
          rounds: 8,
          gen(i) {
            const [a, b] = pool[i], p = 100 * a / b, nice = Math.abs(p * 10 - Math.round(p * 10)) < 1e-9;
            const ctx = pick([`כמה אחוז זה ${M(`${a}`)} מתוך ${M(String(b))}?`, `עומר פתר ${a} מתוך ${b} חידות. כמה אחוז מהחידות פתר?`, `בבחינה היו ${b} שאלות ועומר ענה נכון על ${a}. מה הציון באחוזים?`]);
            return numQ(ctx, p, { suffix: '%', unit: '%', tol: nice ? undefined : 0.051, hint: `${M(`${a} ÷ ${b}`)} ואז כפול 100. או: לצמצם, ואז מכנה 100.`, explain: `${F(a, b)} = ${M(fmt(a / b, 4))} = ${P(p, 1)}${nice ? '' : ' (בערך)'}${a > b ? '. יותר מ-100%, כי החלק גדול מהשלם.' : ''}` });
          },
          onDone: () => done(),
        });
      },
    },
    {
      t: 'סקר בכיתה',
      r(el, done) {
        const cats = [['⚽ כדורגל', '#2563eb'], ['♟️ שחמט', '#ea580c'], ['🎮 גיימינג', '#7c3aed'], ['📚 קריאה', '#16a34a']];
        const cnt = [10, 3, 12, 7];
        el.innerHTML = `<p>עומר עשה סקר בכיתה: "מה התחביב הכי אהוב עליך?" כל ילד בחר תחביב אחד.</p>
          <p class="do">שנה את מספר הילדים בכל תחביב, ותראה מה קורה לעוגה ולאחוזים.</p>
          <div class="play"><div class="survey"></div><div><svg class="board pie" viewBox="0 0 220 220" style="max-width:260px;margin:auto"></svg><div class="tot" style="text-align:center"></div></div></div>
          <div class="mh"></div>
          <div class="after" hidden><div class="aha"><b>אחוזים מספרים על החלוקה, לא על הכמות.</b> כשמוסיפים ילדים לגיימינג, האחוז של <u>כל</u> השאר יורד, למרות שאף אחד מהם לא עזב. וסכום כל האחוזים תמיד ${M('100%')}, כי כל ילד בדיוק בפרוסה אחת.
          <br>עוגה היא בעצם ריבוע המאה, רק מגולגל לעיגול.</div></div>`;
        let mis;
        const svg = el.querySelector('.pie');
        function draw() {
          const tot = cnt.reduce((a, b) => a + b, 0);
          el.querySelector('.survey').innerHTML = cats.map(([n, c], k) => `<div class="srow"><span class="sw" style="background:${c}"></span><span class="sn">${n}</span>
            <button class="btn sm round2" data-k="${k}" data-d="-1">−</button><b class="sc">${cnt[k]}</b><button class="btn sm round2" data-k="${k}" data-d="1">+</button>
            <span class="sp2">${tot ? P(100 * cnt[k] / tot, 1) : ''}</span></div>`).join('');
          el.querySelectorAll('.round2').forEach(b => b.onclick = () => { const k = +b.dataset.k; cnt[k] = clamp(cnt[k] + +b.dataset.d, 0, 40); draw(); });
          el.querySelector('.tot').innerHTML = `סך הכול: <b>${tot}</b> ילדים`;
          pie(svg, cats.map(([, c], k) => ({ v: cnt[k], col: c })));
          const pc = k => tot ? 100 * cnt[k] / tot : 0;
          mis && mis.check({ tot, pc, cnt });
        }
        mis = Missions(el.querySelector('.mh'), [
          // בדיקות בשלמים, בלי שגיאות עיגול
          { t: `גרום לשחמט להיות בדיוק ${M('12.5%')}`, ok: s => s.tot && s.cnt[1] * 8 === s.tot },
          { t: `גרום לגיימינג להיות בדיוק חצי מהכיתה`, ok: s => s.tot && s.cnt[2] * 2 === s.tot },
          { t: `גרום לכל ארבעת התחביבים להיות שווים. כמה אחוז כל אחד?`, ok: s => s.tot && s.cnt.every(c => c === s.cnt[0]) },
          { t: `כדורגל ${M('40%')}, שחמט ${M('10%')}, והשאר כרצונך`, ok: s => s.tot && s.cnt[0] * 5 === s.tot * 2 && s.cnt[1] * 10 === s.tot },
        ], () => { el.querySelector('.after').hidden = false; done(); });
        draw();
      },
    },
  ],
});
