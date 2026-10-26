'use strict';
/* פרק 1: מה זה אחוז. ריבוע מאה לצביעה, ארבעה בגדים לאותו מספר, "אחוז ממה?" עם שתי פיצות, ויותר מ-100% */

function gcd(a, b) { a = Math.abs(a); b = Math.abs(b); return b ? gcd(b, a % b) : a; }
// k/100 מצומצם, למשל 25 ← 1/4
function reduced(k, d = 100) { const g = gcd(k, d) || 1; return [k / g, d / g]; }

App.add({
  id: 'ch1', short: 'מה זה אחוז', icon: '💯', title: 'מה זה בכלל אחוז?',
  desc: 'מאה ריבועים, ארבעה בגדים, ואחוז ממה',
  intro: `<p>המילה percent באה מלטינית: <b>per centum</b>, "לכל מאה". הסימן % התפתח, לפי ההסבר המקובל, מקיצור של המילים האיטלקיות per cento, שהלך והתקצר עד שנשארו שני עיגולים וקו.</p>
    <p>אז ${M('37%')} פירושו פשוט: <b>37 מכל 100</b>.</p>`,
  steps: [
    {
      t: 'מאה ריבועים',
      r(el, done) {
        el.innerHTML = `<p class="do">צבע ריבועים בלחיצה או בגרירה. לחיצה על ריבוע צבוע מוחקת אותו.</p>
          <div class="play"><div class="gh"></div><div><div class="panel ro"></div><div class="mh"></div></div></div>
          <div class="after" hidden><div class="aha"><b>אחוז הוא שבר שהמכנה שלו תמיד 100.</b> בגלל זה קל להשוות אחוזים: כולם "מדברים באותה שפה". ${M('25%')} הוא ${F(25, 100)}, וזה בדיוק ${F(1, 4)}. אותו מספר, רק בבגד אחר.
          <br>ושים לב: <b>לא משנה איזה ריבועים צבעת</b>, רק כמה. רבע יכול להיות פינה, פס, או ריבועים מפוזרים.</div></div>`;
        const ro = el.querySelector('.ro');
        let mis;
        const grid = Grid100(el.querySelector('.gh'), {
          onChange(n, set) {
            const [a, b] = reduced(n);
            ro.innerHTML = `<div class="bignum" style="text-align:center">${M(n + '%')}</div>
              <div style="text-align:center">${n} מתוך 100 · ${M(fmt(n / 100))} · ${n && n < 100 ? F(a, b) : M(String(n / 100))}</div>`;
            if (mis) mis.check({ n, set });
          },
        });
        mis = Missions(el.querySelector('.mh'), [
          { t: `צבע ${M('37%')}`, ok: s => s.n === 37 },
          { t: `עכשיו רק ${M('5%')}`, ok: s => s.n === 5 },
          { t: `צבע רבע מהריבוע. כמה אחוז זה?`, ok: s => s.n === 25 },
          { t: `צבע ${M('20%')} כך שבכל שורה יהיו בדיוק 2 ריבועים צבועים`, ok: s => s.n === 20 && [...Array(10).keys()].every(r => [...s.set].filter(k => Math.floor(k / 10) === r).length === 2) },
          { t: `צבע ${M('100%')}. מה זה אומר?`, ok: s => s.n === 100, after: () => note(`${M('100%')} זה <b>הכול</b>. השלם כולו.`) },
        ], () => { el.querySelector('.after').hidden = false; done(); });
        function note(h) { const x = document.createElement('div'); x.className = 'tip flash'; x.innerHTML = h; el.querySelector('.mh').appendChild(x); }
        grid.set([]);
      },
    },
    {
      t: 'ארבעה בגדים',
      r(el, done) {
        el.innerHTML = `<p>כל מספר אפשר לכתוב בארבע דרכים: <b>אחוז</b>, <b>שבר עשרוני</b>, <b>שבר רגיל</b>, ו<b>"x מכל y"</b>.</p>
          <table class="clothes"><tr><th>אחוז</th><th>עשרוני</th><th>שבר</th><th>במילים</th></tr>
            <tr><td>${M('50%')}</td><td>${M('0.5')}</td><td>${F(1, 2)}</td><td>אחד מכל שניים</td></tr>
            <tr><td>${M('20%')}</td><td>${M('0.2')}</td><td>${F(1, 5)}</td><td>אחד מכל חמישה</td></tr>
            <tr><td>${M('12.5%')}</td><td>${M('0.125')}</td><td>${F(1, 8)}</td><td>אחד מכל שמונה</td></tr>
            <tr><td>${M('150%')}</td><td>${M('1.5')}</td><td>${F(3, 2)}</td><td>שלושה על כל שניים</td></tr></table>
          <div class="tip">אחוז ← עשרוני: מחלקים ב-100, כלומר מזיזים את הנקודה שני מקומות שמאלה. ${M('7% = 0.07')}, ${M('0.5% = 0.005')}.</div>
          <p class="do">עכשיו אתה. 8 שאלות.</p><div class="qh"></div>`;
        const items = [
          [1, 4, 25], [3, 4, 75], [1, 5, 20], [2, 5, 40], [1, 10, 10], [1, 8, 12.5], [3, 8, 37.5], [1, 20, 5], [1, 25, 4], [1, 50, 2], [3, 2, 150], [1, 1, 100], [1, 3, 33.33], [1, 200, 0.5],
        ];
        Quiz(el.querySelector('.qh'), {
          rounds: 8,
          gen(i) {
            const [a, b, p] = pick(items);
            const fr = b === 1 ? M(String(a)) : F(a, b);
            const kind = i % 4;
            if (kind === 0) return numQ(`כתוב את ${fr} באחוזים.`, p, { suffix: '%', unit: '%', tol: p === 33.33 ? 0.34 : undefined, hint: `אחוז = כמה מתוך 100. נסה להפוך את המכנה ל-100, או לחלק ${M(`${a} ÷ ${b}`)} ולכפול ב-100.`, explain: `${fr} = ${M(fmt(a / b, 4))} = ${P(p)}${p === 33.33 ? ' (בעצם 33.333... לנצח)' : ''}` });
            if (kind === 1) return numQ(`כתוב את ${P(p)} כשבר עשרוני.`, a / b, { tol: p === 33.33 ? 0.004 : undefined, hint: `מחלקים ב-100: הנקודה זזה שני מקומות שמאלה.`, explain: `${P(p)} = ${M(fmt(p, 2) + ' ÷ 100')} = ${M(fmt(a / b, 4))}` });
            if (kind === 2) {
              const wrong = shuffle(items.filter(x => x[2] !== p)).slice(0, 3);
              return { q: `איזה שבר שווה ל-${P(p)}?`, type: 'choice', options: shuffle([[a, b], ...wrong.map(w => [w[0], w[1]])]).map(([x, y]) => ({ h: y === 1 ? M(String(x)) : F(x, y), v: x / y })), answer: a / b, hint: 'כמה זה מתוך 100?', explain: `${P(p)} = ${fr}` };
            }
            const d = pick([20, 40, 200, 50, 25]);
            const k = Math.round(d * p / 100 * 100) / 100;
            if (!Number.isInteger(k)) return numQ(`כמה אחוז זה ${M(`${a}`)} מכל ${M(String(b))}?`, p, { suffix: '%', unit: '%', tol: p === 33.33 ? 0.34 : undefined, explain: `${fr} = ${P(p)}` });
            return numQ(`בקופסה ${d} כדורים, ${P(p)} מהם אדומים. כמה כדורים אדומים?`, k, { hint: `${P(p)} זה ${fr}. כמה זה ${fr} מתוך ${d}?`, explain: `${fr} מ-${d} = ${M(String(k))}` });
          },
          onDone: () => done(),
        });
      },
    },
    {
      t: 'אחוז ממה?',
      r(el, done) {
        el.innerHTML = `<p>עומר הזמין פיצה <b>קטנה</b> (קוטר 20 ס״מ), והחבר שלו נועם הזמין פיצה <b>ענקית</b> (קוטר 40 ס״מ). עומר אכל ${M('50%')} מהפיצה שלו, ונועם רק ${M('25%')} מהשלו. מי אכל יותר?</p>
          <p class="do">הזז את המחוונים ותבדוק.</p>
          <div class="pizzas"><div class="pz"><svg viewBox="0 0 220 220" class="board pza"></svg><label class="sl"><span>עומר</span><input type="range" min="0" max="100" step="5" value="50" class="sa"><output></output></label></div>
            <div class="pz"><svg viewBox="0 0 220 220" class="board pzb"></svg><label class="sl"><span>נועם</span><input type="range" min="0" max="100" step="5" value="25" class="sb"><output></output></label></div></div>
          <div class="panel pro"></div><div class="mh"></div>
          <div class="after" hidden><div class="aha"><b>אחוז לבד לא אומר כמה.</b> הוא תמיד "אחוז <u>ממשהו</u>". כשהשלם שונה, אותו אחוז שווה כמות אחרת. הפיצה של נועם גדולה פי 4 בשטח (קוטר כפול, שטח פי ${M('2² = 4')}), אז ${M('25%')} ממנה שווים ל-${M('100%')} מהפיצה של עומר.
          <br>בכל פעם שמישהו אומר לך אחוז, תשאל: <b>ממה?</b></div></div>`;
        const A = el.querySelector('.pza'), B = el.querySelector('.pzb'), sa = el.querySelector('.sa'), sb = el.querySelector('.sb');
        const area = d => Math.PI * (d / 2) ** 2;
        function drawPizza(svg, r, p) {
          S.clear(svg);
          S.el('circle', { cx: 110, cy: 110, r, class: 'pz-crust' }, svg);
          S.el('circle', { cx: 110, cy: 110, r: r - 5, class: 'pz-cheese' }, svg);
          if (p > 0) {
            const a0 = -Math.PI / 2, a1 = a0 + 2 * Math.PI * p / 100, rr = r + 1;
            if (p >= 100) S.el('circle', { cx: 110, cy: 110, r: rr, class: 'pz-eaten' }, svg);
            else S.el('path', { d: `M110,110 L${110 + rr * Math.cos(a0)},${110 + rr * Math.sin(a0)} A${rr},${rr} 0 ${p > 50 ? 1 : 0} 1 ${110 + rr * Math.cos(a1)},${110 + rr * Math.sin(a1)} Z`, class: 'pz-eaten' }, svg);
          }
        }
        let mis;
        function draw() {
          const pa = +sa.value, pb = +sb.value;
          sa.nextElementSibling.textContent = pa + '%'; sb.nextElementSibling.textContent = pb + '%';
          drawPizza(A, 50, pa); drawPizza(B, 100, pb);
          const ea = area(20) * pa / 100, eb = area(40) * pb / 100;
          el.querySelector('.pro').innerHTML = `עומר אכל <b>${fmt(ea, 0)}</b> סמ״ר · נועם אכל <b>${fmt(eb, 0)}</b> סמ״ר
            <div class="muted small">(השטח הלבן הוא מה שנאכל. ${ea > eb + .01 ? 'עומר אכל יותר' : eb > ea + .01 ? 'נועם אכל יותר' : 'אכלו בדיוק אותו דבר!'})</div>`;
          mis && mis.check({ pa, pb, eq: pa > 0 && Math.abs(ea - eb) < .01 });
        }
        mis = Missions(el.querySelector('.mh'), [
          { t: 'עומר אוכל את כל הפיצה שלו. כמה צריך נועם לאכול כדי לאכול אותו דבר?', ok: s => s.pa === 100 && s.eq },
          { t: 'מצא עוד זוג שבו הם אוכלים בדיוק אותה כמות', ok: s => s.eq && s.pa !== 100 },
        ], () => { el.querySelector('.after').hidden = false; done(); });
        sa.oninput = sb.oninput = draw; draw();
      },
    },
    {
      t: 'יותר מ-100%?',
      r(el, done) {
        el.innerHTML = `<p>אם ${M('100%')} זה הכול, איך אפשר יותר? אפשר, כשמשווים לשלם <b>אחר</b>. בשנה שעברה היו בחוג הרובוטיקה 40 ילדים. השנה יש 60. מספר הילדים השנה הוא ${M('150%')} מבשנה שעברה.</p>
          <p class="do">גרור את הידית על הפס. השלם (${M('100%')}) הוא 40 הילדים של שנה שעברה.</p>
          <div class="bh"></div><div class="mh"></div>
          <div class="after" hidden><div class="aha"><b>מתחת לפס של האחוזים יש פס של כמויות</b>, והם זזים יחד. ${M('100%')} הוא נקודת הייחוס, וממנה אפשר ללכת ימינה כמה שרוצים.
          <br>אבל זהירות: "${M('150%')} מבשנה שעברה" זה לא "עלייה של ${M('150%')}". העלייה היא ${M('50%')}. נחזור לזה בפרק 5.</div>
          <div class="tip">ומה עם שחקן שאומר "נתתי ${M('110%')} מעצמי"? פיזיקלית אי אפשר. זה רק אומר שהוא נתן יותר ממה שחשב שהמקסימום שלו.</div></div>`;
        let mis;
        PctBar(el.querySelector('.bh'), { whole: 40, maxPct: 300, step: 5, unit: 'ילדים', pct: 100, onChange: (p, v) => mis && mis.check({ p, v }) });
        mis = Missions(el.querySelector('.mh'), [
          { t: 'הגע ל-60 ילדים. כמה אחוז זה?', ok: s => s.v === 60 },
          { t: 'כמה אחוז יהיה אם מספר הילדים יוכפל?', ok: s => s.p === 200 },
          { t: 'עכשיו 110 ילדים', ok: s => s.v === 110 },
          { t: `ועכשיו ${M('5%')}: כמה ילדים זה?`, ok: s => s.p === 5 },
        ], () => { el.querySelector('.after').hidden = false; done(); });
      },
    },
  ],
});
