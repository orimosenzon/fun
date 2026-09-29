'use strict';
/* פרק 6: חיבור וחיסור. יישור לפי הנקודה, מחשבון עמודות עם העברה ופריטה, ומכולת. */

/* מחשבון עמודות: עידו ממלא ספרה אחרי ספרה, מימין לשמאל.
   בחיסור, כשחסר, הוא לוחץ "לפרוט" ורואה את הספרה משמאל יורדת ב-1 */
function ColCalc(el, a, b, op, onDone) {
  const va = N.parse(a), vb = N.parse(b), res = op === '+' ? va + vb : va - vb;
  const fl = s => (s.split('.')[1] || '').length, il = s => s.split('.')[0].length;
  const Fd = Math.max(fl(a), fl(b)), rs = N.str(res, Fd), RI = il(rs);
  const I = Math.max(il(a), il(b), RI);
  const places = []; for (let p = I - 1; p >= 0; p--) places.push(p); for (let p = -1; p >= -Fd; p--) places.push(p);
  const dg = (s, p) => { const [i, f = ''] = s.split('.'); if (p >= 0) { const k = i.length - 1 - p; return k >= 0 ? +i[k] : null; } const c = f[-p - 1]; return c === undefined ? null : +c; };
  const top = {}, bot = {}, cur = {}, want = {};
  places.forEach(p => { top[p] = dg(a, p); bot[p] = dg(b, p); cur[p] = top[p] ?? 0; want[p] = p < RI ? dg(rs, p) : null; });
  const carry = {};
  const cell = (p, row) => `<div class="c" data-row="${row}" data-p="${p}"></div>`;
  const rowHTML = (row, sign = '') => `<div class="row ${row === 'carry' ? 'carry' : ''}"><div class="c op">${sign}</div>${places.map(p => (p === -1 ? `<div class="c d">${row === 'a' || row === 'b' || row === 'r' ? '.' : ''}</div>` : '') + cell(p, row)).join('')}</div>`;
  el.innerHTML = `<div class="cc-wrap" style="text-align:center"><div class="colcalc">${rowHTML('carry')}${rowHTML('a')}${rowHTML('b', op === '+' ? '+' : '−')}<div class="line"></div>${rowHTML('r')}</div></div>
    <div class="cc-msg tip">מתחילים מהעמודה הימנית ביותר, ה${PLACE_NAME[-Fd] || 'שלמים'}.</div><div class="controls cc-ctl"></div>`;
  const C = (row, p) => el.querySelector(`.c[data-row="${row}"][data-p="${p}"]`);
  const msg = h => { const m = el.querySelector('.cc-msg'); m.innerHTML = h; m.classList.remove('flash'); void m.offsetWidth; m.classList.add('flash'); };
  function paintNums() {
    for (const p of places) {
      const ca = C('a', p), cb = C('b', p);
      if (cur[p] !== (top[p] ?? 0)) ca.innerHTML = `<span class="strike">${top[p] ?? 0}</span><span class="newd">${cur[p]}</span>`;
      else ca.innerHTML = top[p] === null ? (p < 0 ? '<span class="ghost">0</span>' : '') : `<span class="dg ${PLACE_CLS[p]}">${top[p]}</span>`;
      cb.innerHTML = bot[p] === null ? (p < 0 ? '<span class="ghost">0</span>' : '') : `<span class="dg ${PLACE_CLS[p]}">${bot[p]}</span>`;
      C('carry', p).textContent = carry[p] ? carry[p] : '';
    }
  }
  // שורת התוצאה: שדה קלט לכל עמודה שיש בה ספרה בתוצאה
  const order = places.slice().reverse().filter(p => want[p] !== null);
  order.forEach(p => { C('r', p).innerHTML = `<input maxlength="1" inputmode="numeric" disabled data-p="${p}" data-w="${want[p]}">`; });
  let k = 0, wrong = 0;
  const bottom = p => bot[p] ?? 0;
  function focusCol() {
    places.forEach(p => ['a', 'b', 'r'].forEach(r => C(r, p).classList.remove('curcol')));
    if (k >= order.length) return;
    const p = order[k];
    ['a', 'b', 'r'].forEach(r => C(r, p).classList.add('curcol'));
    const inp = C('r', p).querySelector('input');
    const ctl = el.querySelector('.cc-ctl'); ctl.innerHTML = '';
    if (op === '-' && cur[p] < bottom(p)) {
      inp.disabled = true;
      msg(`בעמודת ה<b>${PLACE_NAME[p]}</b> למעלה יש ${cur[p]}, וצריך להוריד ${bottom(p)}. <b>אין מספיק!</b> פורטים אחד מהעמודה שמשמאל ל-10 קטנים.`);
      ctl.innerHTML = `<button class="btn primary brk">🔨 לפרוט מהעמודה משמאל</button>`;
      ctl.querySelector('.brk').onclick = async () => { ctl.innerHTML = ''; await borrow(p); focusCol(); };
      return;
    }
    inp.disabled = false; setTimeout(() => inp.focus({ preventScroll: true }), 20);
    const ci = carry[p] || 0;
    msg(op === '+' ? `עמודת ה<b>${PLACE_NAME[p]}</b>: ${M(`${cur[p]} + ${bottom(p)}${ci ? ' + ' + ci : ''} = ?`)} ${ci ? '(ה-1 הקטן הוא מה שהעברנו)' : ''}`
                   : `עמודת ה<b>${PLACE_NAME[p]}</b>: ${M(`${cur[p]} − ${bottom(p)} = ?`)}`);
    inp.oninput = () => {
      const v = inp.value.replace(/\D/g, ''); inp.value = v; if (!v) return;
      const exp = want[p], s = op === '+' ? cur[p] + bottom(p) + ci : cur[p] - bottom(p);
      if (+v === exp) {
        inp.classList.remove('wrong'); inp.classList.add('right'); inp.disabled = true; wrong = 0;
        if (op === '+' && s >= 10 && places.includes(p + 1)) {
          carry[p + 1] = 1; paintNums();
          const up = { 0: 'עשרת אחת', '-1': 'שלם אחד', '-2': 'עשירית אחת', '-3': 'מאית אחת', 1: 'מאה אחת' }[p];
          k++;
          msg(`${M(String(s))} ${PLACE_NAME[p]}! כותבים ${exp}, ואת ה-10 הנוספים מעבירים: 10 ${PLACE_NAME[p]} = ${up}. ה-1 הקטן עולה לעמודה משמאל.`);
          setTimeout(focusCol, 1600);
        } else { k++; if (k >= order.length) finish(); else focusCol(); }
      } else {
        wrong++; inp.classList.add('wrong');
        let h = op === '+' ? `${M(`${cur[p]} + ${bottom(p)}${ci ? ' + ' + ci : ''} = ${s}`)}.` : `${M(`${cur[p]} − ${bottom(p)} = ${s}`)}.`;
        if (op === '+' && s >= 10) h += ` זה יותר מ-9, אז כותבים רק את ${M(String(s % 10))} ומעבירים 1 שמאלה.`;
        msg(wrong >= 2 ? `<b>כמעט.</b> ${h}` : `<b>לא בדיוק.</b> ${op === '+' ? 'חבר את כל מה שיש בעמודה הזאת.' : 'חסר את הספרה התחתונה מהעליונה.'} נסה שוב.`);
        setTimeout(() => { inp.value = ''; }, 500);
      }
    };
  }
  async function borrow(p) {
    const q = p + 1;
    if (!places.includes(q)) return;
    if (cur[q] === 0) await borrow(q);
    cur[q] -= 1; cur[p] += 10; paintNums();
    const from = { 0: 'שלם', '-1': 'עשירית', '-2': 'מאית', 1: 'עשרת' }[q], to = PLACE_NAME[p];
    msg(`פרטנו ${from} אחת ל-10 ${to}. עכשיו למעלה יש ${cur[p]} ${to}.`);
    await sleep(1300);
  }
  function finish() {
    focusCol();
    msg(`<b>✓ ${M(`${a} ${op === '+' ? '+' : '−'} ${b} = ${N.str(res)}`)}</b>${N.str(res) !== rs ? ` (אפשר לכתוב גם ${M(rs)})` : ''}`);
    onDone && onDone();
  }
  paintNums(); focusCol();
}

// סדרת תרגילים במחשבון העמודות
function CalcSeries(el, list, op, onAll) {
  let i = 0;
  el.innerHTML = `<div class="muted ser-p"></div><div class="ser-host"></div><div class="controls ser-n"></div>`;
  function go() {
    el.querySelector('.ser-p').textContent = `תרגיל ${i + 1} מתוך ${list.length}`;
    el.querySelector('.ser-n').innerHTML = '';
    ColCalc(el.querySelector('.ser-host'), list[i][0], list[i][1], op, () => {
      const last = i === list.length - 1;
      el.querySelector('.ser-n').innerHTML = `<button class="btn primary">${last ? 'סיימתי ✓' : 'לתרגיל הבא ←'}</button>`;
      el.querySelector('.ser-n button').onclick = () => { if (last) { confetti(); el.querySelector('.ser-n').innerHTML = '<div class="aha">🏆 כל התרגילים נפתרו!</div>'; onAll(); } else { i++; go(); } };
    });
  }
  go();
}

App.add({
  id: 'ch6', short: 'חיבור וחיסור', icon: '➕', title: 'חיבור וחיסור',
  desc: 'למה מיישרים נקודות, מעבירים ופורטים',
  intro: `<p>הכלל היחיד שצריך: <b>מחברים רק דברים מאותו סוג</b>. שלמים עם שלמים, עשיריות עם עשיריות, מאיות עם מאיות. בדיוק כמו שלא מחברים שקלים עם אגורות.</p>`,
  steps: [
    {
      t: 'למה מיישרים נקודות?',
      r(el, done) {
        el.innerHTML = `
          <p>כשמחברים מספרים שלמים, מיישרים אותם לימין. אבל בשברים עשרוניים זה מלכודת! תסתכל על הצבעים: בכל עמודה צריך להיות צבע אחד בלבד.</p>
          <p class="do">הזז את המספר התחתון עם החצים, עד שכל עמודה תהיה בצבע אחד.</p>
          <div class="al-host"></div><div class="al-fb"></div>`;
        const ex = [['2.5', '1.25'], ['12', '0.4'], ['3.07', '15.6']];
        let e = 0;
        function show() {
          const [a, b] = ex[e], ghostDot = s => s.includes('.') ? s : s + '.';
          const A = ghostDot(a), B = ghostDot(b), COLS = 9;
          const cls = (s, j) => { const d = s.indexOf('.'); if (j === d) return 'dot'; const p = j < d ? d - 1 - j : -(j - d); return PLACE_CLS[p]; };
          const offA = COLS - 2 - A.length; // המספר העליון מיושר לימין, כמו שיודעים ממספרים שלמים
          let offB = COLS - 2 - B.length;
          const host = el.querySelector('.al-host');
          host.innerHTML = `<div style="text-align:center"><div class="colcalc al"></div></div>
            <div class="controls" style="justify-content:center;direction:ltr"><button class="btn round lft">◀</button><button class="btn round rgt">▶</button></div>
            ${!a.includes('.') ? `<p class="tip">ל-${M(a)} אין נקודה? יש לו, היא פשוט נסתרת בסוף: ${M(a + ' = ' + a + '.0')}. סימנו אותה בחיוור.</p>` : ''}`;
          const draw = () => {
            const row = (s, off) => Array.from({ length: COLS }, (_, j) => { const k = j - off; if (k < 0 || k >= s.length) return '<div class="c"></div>';
              const c = cls(s, k), ch = s[k]; return `<div class="c ${c === 'dot' ? 'd' : ''}">${c === 'dot' ? `<span class="${s === B && !b.includes('.') || s === A && !a.includes('.') ? 'ghost' : ''}">.</span>` : `<span class="dg ${c}">${ch}</span>`}</div>`; }).join('');
            host.querySelector('.al').innerHTML = `<div class="row">${row(A, offA)}</div><div class="row">${row(B, offB)}</div>`;
            const aligned = offA + A.indexOf('.') === offB + B.indexOf('.');
            host.querySelector('.lft').disabled = offB <= 0 || aligned && e >= ex.length; host.querySelector('.rgt').disabled = offB + B.length >= COLS;
            const fbx = el.querySelector('.al-fb');
            if (aligned) {
              const sum = N.str(N.parse(a) + N.parse(b));
              fbx.innerHTML = `<div class="aha"><b>עכשיו כל עמודה היא סוג אחד!</b> הנקודות אחת מתחת לשנייה, והעמודות: שלמים מתחת לשלמים, עשיריות מתחת לעשיריות.
                ${e === 0 ? `<br>בשקלים: ${M('2.50')} ₪ + ${M('1.25')} ₪ = 3 שקלים ו-75 אגורות = ${M('3.75')} ₪.` : ''}
                <br>${M(`${a} + ${b} = ${sum}`)}</div>
                ${e < ex.length - 1 ? '<button class="btn primary nx">לדוגמה הבאה ←</button>' : ''}`;
              host.querySelector('.lft').disabled = true; host.querySelector('.rgt').disabled = true;
              if (e < ex.length - 1) fbx.querySelector('.nx').onclick = () => { e++; show(); }; else done();
            } else {
              fbx.innerHTML = offB === COLS - 2 - B.length ? `<div class="trap">ככה מיישרים מספרים שלמים, לימין. אבל תראה: ${e === 0 ? 'ה-5 (עשיריות, כחול) יושב מעל 5 של מאיות (כתום)' : 'הצבעים בעמודות מעורבבים'}. זה כמו לחבר שקלים עם אגורות.</div>` : '';
            }
          };
          host.querySelector('.lft').onclick = () => { offB--; draw(); };
          host.querySelector('.rgt').onclick = () => { offB++; draw(); };
          draw();
        }
        show();
      },
    },
    {
      t: 'חיבור בעמודות',
      r(el, done) {
        el.innerHTML = `
          <p>עכשיו אתה מחבר. <b>מיישרים את הנקודות</b>, משלימים אפסים (החיוורים), ומחברים עמודה-עמודה <b>מימין לשמאל</b>. כשעמודה מגיעה ל-10 או יותר, זה בדיוק כמו בקוביות מפרק 3: 10 מאותו סוג מתאחדים לאחד מהסוג שמשמאל.</p>
          <p class="do">כתוב בכל פעם את הספרה של העמודה הצהובה.</p>
          <div class="ser"></div>`;
        CalcSeries(el.querySelector('.ser'), [['2.9', '0.2'], ['3.45', '1.7'], ['0.08', '0.5'], ['4', '0.35'], ['6.75', '2.48']], '+', done);
      },
    },
    {
      t: 'חיסור בעמודות',
      r(el, done) {
        el.innerHTML = `
          <p>בחיסור עושים אותו דבר: מיישרים נקודות ומחסרים מימין לשמאל. ומה אם למעלה יש פחות ממה שצריך להוריד? <b>פורטים</b>: לוקחים אחד מהעמודה שמשמאל והופכים אותו ל-10 קטנים. בדיוק כמו לפרוט שקל לעשרה מטבעות של 10 אגורות.</p>
          <p class="do">כתוב את הספרה בעמודה הצהובה. כשצריך, לחץ על "לפרוט".</p>
          <div class="ser"></div>`;
        CalcSeries(el.querySelector('.ser'), [['5.8', '2.3'], ['1', '0.35'], ['5.2', '1.75'], ['3.04', '1.6'], ['10', '2.5']], '-', done);
      },
    },
    {
      t: 'משחק: המכולת',
      r(el, done) {
        el.innerHTML = `<p>אתה בקופה של המכולת. חשב כמה עולה הסל, וכמה עודף להחזיר. מותר להשתמש בדף ועט.</p><p class="tip">טיפ: לפני שמחשבים, <b>מעריכים</b>. במבה ב-${M('4.90')} זה כמעט 5 שקלים. ככה תדע אם התשובה הגיונית.</p><div class="qhost"></div>`;
        const ITEMS = [['🥛', 'חלב', '6.90'], ['🍞', 'לחם', '8.5'], ['🥜', 'במבה', '4.90'], ['🧃', 'שוקו', '6.50'], ['🍫', 'שוקולד', '7.25'], ['🥖', 'לחמנייה', '1.20'], ['🍎', 'תפוח', '2.35'], ['🍦', 'ארטיק', '9.80'], ['🧀', 'גבינה', '5.45'], ['🍬', 'מסטיק', '0.95']];
        const figOf = items => `<div class="shop">${items.map(([e, n, p]) => `<div class="item in"><div class="ie">${e}</div><div>${n}</div><div class="ip">${M(p)} ₪</div></div>`).join('')}</div>`;
        let basket = null;
        Quiz(el.querySelector('.qhost'), {
          rounds: 6, onDone: done,
          gen(i) {
            if (i % 2 === 0) {
              basket = shuffle(ITEMS).slice(0, i === 4 ? 3 : 2);
              const tot = basket.reduce((s, x) => s + N.parse(x[2]), 0);
              return { q: 'כמה עולה כל הסל?', type: 'input', unit: '₪', answer: tot, fig: figOf(basket),
                hint: 'יישר נקודות! אם לאחד המחירים יש רק ספרה אחת אחרי הנקודה, השלם 0.',
                explain: `${M(basket.map(x => x[2]).join(' + ') + ' = ' + N.str(tot, 2))} ₪` };
            }
            const tot = basket.reduce((s, x) => s + N.parse(x[2]), 0), paid = tot <= N.of(20) ? N.of(20) : N.of(50);
            return { q: `הלקוח משלם בשטר של ${N.str(paid)} ₪. כמה עודף מחזירים?`, type: 'input', unit: '₪', answer: paid - tot,
              fig: figOf(basket) + `<p>הסל עולה ${M(N.str(tot, 2))} ₪.</p>`,
              hint: `${M(N.str(paid) + ' = ' + N.str(paid, 2))}. תחסר בעמודות, ואל תשכח לפרוט.`,
              explain: `${M(`${N.str(paid, 2)} − ${N.str(tot, 2)} = ${N.str(paid - tot, 2)}`)} ₪. בדיקה: ${M(`${N.str(tot, 2)} + ${N.str(paid - tot, 2)} = ${N.str(paid, 2)}`)} ✓` };
          },
        });
      },
    },
    {
      t: 'מה לקחת מהפרק', auto: true,
      r(el) {
        el.innerHTML = `<div class="learned"><ul>
          <li><b>מיישרים את הנקודות</b>, לא את הקצוות. ככה כל עמודה היא סוג אחד.</li>
          <li>למספר שלם יש נקודה נסתרת בסוף: ${M('4 = 4.0 = 4.00')}.</li>
          <li>משלימים אפסים כדי שיהיה קל: ${M('0.5 + 0.08')} → ${M('0.50 + 0.08 = 0.58')}.</li>
          <li>בחיבור: עמודה שמגיעה ל-10 או יותר מעבירה 1 שמאלה. ${M('2.9 + 0.2 = 3.1')}, לא ${M('2.11')}.</li>
          <li>בחיסור: כשלמעלה חסר, פורטים אחד מהעמודה משמאל ל-10. ${M('1 − 0.35 = 1.00 − 0.35 = 0.65')}.</li>
          <li>בודקים חיסור עם חיבור: ${M('0.65 + 0.35 = 1')} ✓</li>
        </ul></div>`;
      },
    },
  ],
});
