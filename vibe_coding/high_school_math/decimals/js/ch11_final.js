'use strict';
/* פרק 11: אתגר הגמר. שאלות מכל הפרקים, חזרה לחמש שאלות הפתיחה ("אז ועכשיו"), ותעודה. */

// מחולל שאלה אחת לכל נושא
const FINAL_GENS = {
  tenths() { const t = ri(11, 29); return { q: 'כמה שוקולד יש כאן?', type: 'input', answer: N.of(t, 1), fig: f => f.appendChild(barsSVG(t, { s: 80 })), hint: 'טבלאות שלמות לפני הנקודה, חתיכות אחריה.', explain: M(tenthStr(t)), topic: 'עשיריות' }; },
  hund() { const n = ri(1, 9); return { q: `כתוב כמספר עשרוני: <b>${HEB.count(n, 'מאית', 'מאיות')}</b>`, type: 'input', answer: N.of(n, 2), hint: 'מאיות הן הספרה השנייה אחרי הנקודה.', explain: `${M('0.0' + n)}: ה-0 שומר את מקום העשיריות.`, topic: 'מאיות' }; },
  place() {
    const s = `${ri(1, 9)}.${ri(1, 9)}${ri(1, 9)}${ri(1, 9)}`, j = ri(1, 3), d = s[1 + j];
    return choiceQ(`כמה שווה הספרה המודגשת: ${M(s.slice(0, 1 + j) + '<u>' + d + '</u>' + s.slice(2 + j))}?`, N.str(N.of(+d, j)), [d, N.str(N.of(+d, j === 1 ? 2 : 1)), N.str(N.of(+d, j === 3 ? 2 : 3))], { fmt: M, hint: 'באיזה מקום אחרי הנקודה היא?', explain: `היא במקום ה${['', 'עשיריות', 'מאיות', 'אלפיות'][j]}: ${M(N.str(N.of(+d, j)))}.`, topic: 'בית הספרות' });
  },
  compare() {
    let [a, b] = trapPair(pick(['long', 'short', 'zeroMid', 'nines', 'wzero', 'thou'])); if (Math.random() < .5) [a, b] = [b, a];
    const right = N.parse(a) > N.parse(b) ? a : b;
    return { q: 'מי גדול יותר?', type: 'choice', options: [{ h: digitsHTML(a), v: a }, { h: digitsHTML(b), v: b }], answer: right, hint: 'השלם אפסים והשווה משמאל.', explain: compareWhy(a, b), topic: 'השוואה' };
  },
  between() {
    const a = ri(11, 98), va = N.of(a, 2), vb = N.of(a + 1, 2);
    return { q: `כתוב מספר שנמצא בין ${M(N.str(va))} ל-${M(N.str(vb))}.`, type: 'input', show: N.str(va + N.of(5, 3)), accept: (s, v) => v !== null && v > va && v < vb, hint: 'הוסף ספרה בסוף: אלפיות.', explain: `למשל ${M(N.str(va + N.of(5, 3)))}.`, topic: 'ציר המספרים' };
  },
  add() {
    const a = N.of(ri(11, 99), 1), b = N.of(ri(11, 99), 2);
    return { q: `${M(`${N.str(a)} + ${N.str(b)} = ?`)}`, type: 'input', answer: a + b, hint: 'יישר נקודות והשלם 0.', explain: `${M(`${N.str(a, 2)} + ${N.str(b)} = ${N.str(a + b)}`)}`, topic: 'חיבור' };
  },
  sub() {
    const a = N.of(ri(3, 9)), b = N.of(ri(11, 299), 2);
    return { q: `${M(`${N.str(a)} − ${N.str(b)} = ?`)}`, type: 'input', answer: a - b, hint: `${M(N.str(a) + ' = ' + N.str(a, 2))}, ואז לפרוט.`, explain: `${M(`${N.str(a, 2)} − ${N.str(b, 2)} = ${N.str(a - b)}`)}`, topic: 'חיסור' };
  },
  shift() {
    const n = pick([`${ri(1, 9)}.${ri(1, 9)}${ri(1, 9)}`, `0.${ri(1, 9)}${ri(1, 9)}`]), k = ri(1, 3), f = 10 ** k, up = Math.random() < .6;
    const v = N.parse(n), r = up ? v * f : Math.round(v / f);
    if (!up && N.places(v) + k > 4) return FINAL_GENS.shift();
    return { q: `${M(`${n} ${up ? '×' : '÷'} ${f} = ?`)}`, type: 'input', answer: r, hint: `הספרות זזות ${k} מקומות ${up ? 'שמאלה' : 'ימינה'}.`, explain: M(`${n} ${up ? '×' : '÷'} ${f} = ${N.str(r)}`), topic: 'כפול 10' };
  },
  mul() {
    const a = N.of(ri(2, 9), 1), b = pick([N.of(ri(2, 9), 1), N.of(ri(2, 9))]);
    return { q: `${M(`${N.str(a)} × ${N.str(b)} = ?`)}`, type: 'input', answer: N.mul(a, b), hint: 'כפול בלי נקודות, ואז ספור ספרות אחרי הנקודה.', explain: M(`${N.str(a)} × ${N.str(b)} = ${N.str(N.mul(a, b))}`), topic: 'כפל' };
  },
  div() {
    const d = pick(['0.5', '0.2', '0.4', '0.25']), q = ri(2, 9), a = N.parse(d) * q;
    return { q: `${M(`${N.str(a)} ÷ ${d} = ?`)}`, type: 'input', answer: N.of(q), hint: 'כפול את שניהם ב-10 או ב-100 עד שהמחלק שלם.', explain: M(`${N.str(a)} ÷ ${d} = ${q}`), topic: 'חילוק' };
  },
  frac() {
    const [n, d] = pick([[1, 4], [3, 4], [2, 5], [1, 8], [3, 20], [7, 25]]);
    return { q: `כתוב כשבר עשרוני: ${F(n, d)}`, type: 'input', answer: fracDec(n, d), hint: 'הפוך את המכנה ל-10, 100 או 1000.', explain: `${M(`${F(n, d)} = ${M(N.str(fracDec(n, d)))}`)}`, topic: 'שברים' };
  },
  pct() { const p = pick([30, 45, 8, 75]); return { q: `${M(p + '%')} כשבר עשרוני:`, type: 'input', answer: N.of(p, 2), hint: 'אחוז = מאית.', explain: `${M(`${M(p + '%')} = ${M(N.str(N.of(p, 2)))}`)}`, topic: 'אחוזים' }; },
};

let certEl = null;
function drawCert() {
  const doneCh = App.chapters.slice(1, 11).filter(c => App.isDone(c.id)).length, best = Store.get('best', 0);
  const d = new Date(), date = `${d.getDate()}.${d.getMonth() + 1}.${d.getFullYear()}`;
  certEl.innerHTML = `
    <div class="cert">
      <div style="font-size:3rem">🏅</div>
      <h2 style="justify-content:center;margin:4px 0">תעודת מומחה לשברים עשרוניים</h2>
      <p style="font-size:1.2rem">מוענקת ל<b>עידו</b></p>
      <p>על מסע מלא בעולם השברים העשרוניים: עשיריות, מאיות ואלפיות, ציר המספרים, השוואה, ארבע פעולות החשבון, שברים ואחוזים.</p>
      <p class="muted">פרקים שהושלמו: ${doneCh} מתוך 10 · שיא באתגר הגמר: ${best} מתוך 15 · ${date}</p>
    </div>
    <div class="controls noprint" style="justify-content:center"><button class="btn pr">🖨️ להדפיס</button><button class="btn primary" data-go="ch12">לדף הסיכום ←</button></div>`;
  certEl.querySelector('.pr').onclick = () => { document.body.classList.add('print-cert'); window.print(); setTimeout(() => document.body.classList.remove('print-cert'), 500); };
}

App.add({
  id: 'ch11', short: 'אתגר הגמר', icon: '🏆', title: 'אתגר הגמר',
  desc: 'שאלות מכל הפרקים, ומה השתנה מאז ההתחלה',
  intro: `<p>הגעת לסוף הדרך! עכשיו נראה שהכל מתחבר.</p>`,
  steps: [
    {
      t: 'אתגר: 15 שאלות',
      r(el, done) {
        el.innerHTML = `<p>שאלה אחת לפחות מכל נושא, בסדר אקראי. אם משהו לא יושב, זה סימן טוב לחזור לפרק שלו (הנושא כתוב ליד כל שאלה).</p><div class="qhost"></div>`;
        const keys = Object.keys(FINAL_GENS), order = shuffle(keys.concat(shuffle(['compare', 'add', 'mul', 'shift']).slice(0, 15 - keys.length)));
        Quiz(el.querySelector('.qhost'), {
          rounds: 15,
          onDone(score) { const b = Store.get('best', 0); if (score > b) Store.set('best', score); done(); },
          gen(i) { const q = FINAL_GENS[order[i]](); q.q = `<span class="tag">${q.topic}</span> ` + q.q; return q; },
        });
      },
    },
    {
      t: 'אז ועכשיו',
      r(el, done) {
        const pre = Store.get('pre', {}) || {}, then = pre.ans || {};
        el.innerHTML = `<p>בהתחלה לא אמרתי לך אם צדקת. עכשיו הגיע הזמן. אלה חמש השאלות מההתחלה: תענה עליהן שוב, ואז נגלה מה נכון ונשווה למה שענית אז.</p>
          <div class="pre2">${PRETEST.map((t, k) => `<div class="card pq" data-k="${k}"><div class="qline">${k + 1}. ${t.q}</div>
            ${t.type === 'choice' ? `<div class="opts">${t.options.map(o => `<button class="opt" data-v="${esc(o)}">${/\d/.test(o) ? M(o) : o}</button>`).join('')}</div>`
              : `<div class="inrow"><input class="num-in" dir="ltr" inputmode="decimal">${t.unit ? `<span class="unit">${t.unit}</span>` : ''}</div>`}
            <div class="pr-res"></div></div>`).join('')}</div>
          <div class="controls"><button class="btn primary big reveal">גלה את התשובות</button></div><div class="sumry"></div>`;
        const now = {};
        el.querySelectorAll('.pq').forEach(qd => {
          const t = PRETEST[+qd.dataset.k];
          qd.querySelectorAll('.opt').forEach(b => b.onclick = () => { qd.querySelectorAll('.opt').forEach(x => x.classList.remove('on')); b.classList.add('on'); now[t.id] = b.dataset.v; });
          const inp = qd.querySelector('input'); if (inp) inp.oninput = () => { now[t.id] = inp.value.trim(); };
        });
        el.querySelector('.reveal').onclick = () => {
          if (PRETEST.some(t => !now[t.id])) { el.querySelector('.sumry').innerHTML = '<p class="muted">ענה קודם על כל חמש השאלות.</p>'; return; }
          let before = 0, after = 0;
          el.querySelectorAll('.pq').forEach(qd => {
            const t = PRETEST[+qd.dataset.k], a0 = then[t.id], ok0 = pretestCorrect(t, a0), ok1 = pretestCorrect(t, now[t.id]);
            if (ok0) before++; if (ok1) after++;
            qd.querySelectorAll('.opt, input').forEach(x => x.disabled = true);
            const show = a => /\d/.test(a) ? M(a) : a;
            qd.querySelector('.pr-res').innerHTML = `<div class="vs" style="margin-top:8px">
              <div class="panel"><div class="muted small">בהתחלה ענית</div>${a0 ? show(a0) + (ok0 ? ' ✅' : ' ❌') : '<span class="muted">(לא ענית)</span>'}</div>
              <div class="panel"><div class="muted small">עכשיו ענית</div>${show(now[t.id])} ${ok1 ? '✅' : '❌'}</div></div>
              <div class="${ok1 ? 'aha' : 'trap'}" style="margin-top:6px">${t.why}</div>`;
          });
          el.querySelector('.reveal').disabled = true;
          const any = Object.keys(then).length > 0;
          el.querySelector('.sumry').innerHTML = `<div class="card big-idea" style="text-align:center"><h4>${any ? `בהתחלה: ${before} מתוך 5 · עכשיו: ${after} מתוך 5` : `עכשיו: ${after} מתוך 5`}</h4>
            <p>${after === 5 ? (any && before < 5 ? 'תראה כמה התקדמת! 🎉' : 'מושלם! 🎉') : 'על מה שעוד לא יושב, שווה לחזור לפרק שכתוב בהסבר.'}</p></div>`;
          if (after >= 4) confetti();
          Store.set('post', { ans: now, before, after, when: Date.now() });
          done();
        };
      },
    },
    {
      t: 'תעודה', auto: true,
      r(el) { certEl = el; drawCert(); },
      onShow() { if (certEl) drawCert(); },
    },
  ],
});
