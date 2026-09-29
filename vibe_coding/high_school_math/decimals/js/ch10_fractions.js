'use strict';
/* פרק 10: שבר רגיל ושבר עשרוני. מכונת שברים על ריבוע המאה, משחק זוגות, ואחוזים. */

// שבר n/d כמספר עשרוני, אם הוא נגמר תוך 3 ספרות. אחרת null
function fracDec(n, d) { const v = n * U / d; return Number.isInteger(v) && N.places(v) <= 3 ? v : null; }
function gcd(a, b) { return b ? gcd(b, a % b) : a; }

App.add({
  id: 'ch10', short: 'שבר ועשרוני', icon: '🔄', title: 'שבר רגיל ושבר עשרוני',
  desc: 'חצי, רבע, שמינית ושליש, ומה זה בכלל אחוז',
  intro: `<p>${M('0.5')} ו-${F(1, 2)} הם אותו מספר בשני בגדים שונים. בפרק הזה תלמד להחליף בגדים: מכל שבר רגיל לשבר עשרוני, ובחזרה.</p>`,
  steps: [
    {
      t: 'מכונת השברים',
      r(el, done) {
        el.innerHTML = `
          <p>הסוד: <b>מנסים להפוך את המכנה (המספר למטה) ל-10, 100 או 1000</b>. כי את ${F(25, 100)} כבר יודעים לכתוב: ${M('0.25')}.</p>
          <p class="do">בחר מכנה ומונה, וצפה בריבוע ובחישוב.</p>
          <div class="controls"><span class="muted small">מכנה:</span>${[2, 4, 5, 10, 20, 25, 50, 8, 3].map(d => `<button class="btn sm dn" data-d="${d}">${d}</button>`).join('')}</div>
          <div class="controls"><label class="sl"><span>מונה</span><input type="range" class="nm" min="1" max="1" value="1"><output></output></label></div>
          <div class="play">
            <div class="fg"></div>
            <div><div class="panel fro"></div><div class="mhost"></div></div>
          </div>
          <div class="after" hidden><div class="aha"><b>שבר הוא בעצם חילוק.</b> ${M(`${F(1, 4)} = ${M('1 ÷ 4')}`)}. כשאפשר להגיע למכנה 10, 100 או 1000, השבר העשרוני נגמר. כשאי אפשר (כמו ב-${F(1, 3)}), החילוק ממשיך לנצח, ומקבלים <b>שבר עשרוני אינסופי</b>: ${M('0.333...')}.
            <br><br><b>ואחוז?</b> אחוז הוא פשוט מאית! המילה "אחוז" מגיעה מ"אחד מתוך מאה". ${M('25%')} = 25 מאיות = ${M(`${M('0.25')} = ${F(1, 4)}`)}.</div></div>`;
        let d = 4, n = 1;
        const nm = el.querySelector('.nm');
        const svg = S.svg(300, 300, 'board'); svg.style.maxWidth = '320px'; el.querySelector('.fg').appendChild(svg);
        const mis = Missions(el.querySelector('.mhost'), [
          { t: `מצא כמה זה ${F(1, 4)}`, ok: s => s === '1/4' },
          { t: `מצא כמה זה ${F(3, 5)}`, ok: s => s === '3/5' },
          { t: `מצא כמה זה ${F(1, 8)}`, ok: s => s === '1/8', after: () => note(`את 8 אי אפשר להפוך ל-100, אבל אפשר ל-1000: ${M('8 × 125 = 1000')}. אז ${M(`${F(1, 8)} = ${F(125, 1000)} = ${M('0.125')}`)}.`) },
          { t: `ומה קורה עם ${F(1, 3)}?`, ok: s => s === '1/3' },
        ], () => { el.querySelector('.after').hidden = false; done(); });
        function note(h) { const x = document.createElement('div'); x.className = 'tip flash'; x.innerHTML = h; el.querySelector('.mhost').appendChild(x); }
        function draw() {
          el.querySelectorAll('.dn').forEach(b => b.classList.toggle('on', +b.dataset.d === d));
          nm.max = d - 1; if (n > d - 1) n = d - 1; nm.value = n; nm.nextElementSibling.textContent = n;
          // ציור: n/d מתוך 100 תאים, עם תא חלקי אם צריך
          S.clear(svg);
          const X = 10, s = 280, w = s / 10, cellsF = n * 100 / d;
          const full = Math.floor(cellsF + 1e-9), part = cellsF - full;
          drawGrid(svg, X, X, s, cellsOf(full), { byRole: true });
          if (part > 1e-6) { const { c, r } = gridCell(full); S.el('rect', { x: X + c * w + .6, y: X + r * w + .6 + (w - 1.2) * (1 - part), width: w - 1.2, height: (w - 1.2) * part, fill: 'var(--pm3)' }, svg); }
          const v = fracDec(n, d), ro = el.querySelector('.fro');
          if (v !== null) {
            const p = Math.max(1, N.places(v)), T = 10 ** (d === 8 ? 3 : d <= 10 && 10 % d === 0 ? 1 : 2), k = T / d;
            ro.innerHTML = `<div style="font-size:1.25rem;text-align:center" dir="ltr">${F(n, d)} ${k !== 1 ? `<span class="muted small">(× ${k} למעלה ולמטה)</span>` : ''} = ${F(n * k, T)} = <span class="bignum" style="font-size:2rem">${digitsHTML(N.str(v))}</span></div>
              <div class="muted small" style="margin-top:6px">${d === 10 || d === 100 ? 'המכנה כבר 10, קל!' : `למה ${k}? כי ${M(`${d} × ${k} = ${T}`)}, ואת ${F('?', T)} כבר יודעים לכתוב.`}</div>
              <div class="small">צבוע: ${N.str(N.of(Math.round(cellsF * 10), 1))} ריבועים מתוך 100${Number.isInteger(cellsF) ? '' : ' (הוורוד הוא חלק מריבוע)'}.</div>`;
          } else {
            const reps = N.str(N.of(Math.floor(n * 1e6 / d), 6)).slice(0, 8);
            ro.innerHTML = `<div style="font-size:1.25rem;text-align:center" dir="ltr">${F(n, d)} = ${M(`${n} ÷ ${d}`)} = <span class="bignum" style="font-size:2rem">${M(reps + '...')}</span></div>
              <div class="small" style="margin-top:6px">את 3 אי אפשר להכפיל ולקבל 10, 100 או 1000. בחילוק ארוך תמיד נשארת שארית, והספרה ${M(String(Math.floor(n * 10 / d)))} חוזרת לנצח. <b>זה שבר עשרוני אינסופי.</b> כותבים ${M(reps.slice(0, 5) + '...')} ובדרך כלל מעגלים ל-${M(reps.slice(0, 4))}.</div>
              <div class="small">צבוע: ${N.str(N.of(Math.round(cellsF * 100), 2))} ריבועים... ואף פעם לא בדיוק.</div>`;
          }
          mis.check(`${n}/${d}`);
        }
        el.querySelectorAll('.dn').forEach(b => b.onclick = () => { d = +b.dataset.d; n = Math.min(n, d - 1); draw(); });
        nm.oninput = () => { n = +nm.value; draw(); };
        draw();
      },
    },
    {
      t: 'משחק זוגות',
      r(el, done) {
        el.innerHTML = `<p class="do">הפוך שני קלפים. אם הם שווים (שבר רגיל ושבר עשרוני של אותו מספר), הם נשארים פתוחים. מצא את כל 8 הזוגות.</p>
          <div class="ro-row"><span class="muted mv"></span><button class="btn sm again">ערבב מחדש 🔁</button></div><div class="memory"></div><div class="mfb"></div>`;
        const PAIRS = [[1, 2], [1, 4], [3, 4], [1, 5], [1, 10], [1, 100], [2, 5], [1, 8]];
        let cards, open, moves, gone, lock;
        function deal() {
          cards = shuffle(PAIRS.flatMap(([n, d], k) => [{ k, h: F(n, d) }, { k, h: M(N.str(fracDec(n, d))) }]));
          open = []; moves = 0; gone = 0; lock = false; el.querySelector('.mfb').innerHTML = ''; draw();
        }
        function draw() {
          el.querySelector('.mv').textContent = `ניסיונות: ${moves} · זוגות: ${gone} מתוך 8`;
          const M_ = el.querySelector('.memory');
          M_.innerHTML = cards.map((c, j) => `<button class="mcard ${c.gone ? 'gone' : open.includes(j) ? 'open' : ''}" data-j="${j}">${c.gone || open.includes(j) ? c.h : '?'}</button>`).join('');
          M_.querySelectorAll('.mcard').forEach(b => b.onclick = () => flip(+b.dataset.j));
        }
        function flip(j) {
          if (lock || cards[j].gone || open.includes(j)) return;
          open.push(j); draw();
          if (open.length < 2) return;
          moves++;
          const [x, y] = open;
          if (cards[x].k === cards[y].k) {
            cards[x].gone = cards[y].gone = true; gone++; open = []; draw();
            if (gone === 8) { confetti(); el.querySelector('.mfb').innerHTML = `<div class="aha">🏆 מצאת את כל הזוגות ב-${moves} ניסיונות! ${moves <= 14 ? 'זיכרון מעולה.' : ''} כדאי לזכור את הזוגות האלה בעל פה: הם מופיעים כל הזמן.</div>`; done(); }
          } else { lock = true; setTimeout(() => { open = []; lock = false; draw(); }, 1100); draw(); }
        }
        el.querySelector('.again').onclick = deal;
        deal();
      },
    },
    {
      t: 'משחק: החלפת בגדים',
      r(el, done) {
        el.innerHTML = `<div class="qhost"></div>`;
        const kinds = shuffle(['f2d', 'f2d', 'f2d', 'd2f', 'd2f', 'pct', 'mixed', 'cmp']);
        const nice = [[1, 2], [1, 4], [3, 4], [1, 5], [2, 5], [3, 5], [4, 5], [1, 20], [3, 20], [7, 25], [1, 50], [9, 50], [3, 10], [7, 100], [3, 8], [1, 8]];
        Quiz(el.querySelector('.qhost'), {
          rounds: 8, onDone: done,
          gen(i) {
            const k = kinds[i];
            if (k === 'f2d') {
              const [n, d] = pick(nice), v = fracDec(n, d), T = d === 8 ? 1000 : 10 % d === 0 ? 10 : 100;
              return { q: `כתוב כשבר עשרוני: ${F(n, d)}`, type: 'input', answer: v, hint: `בכמה צריך לכפול את ${d} כדי לקבל ${T}? כפול גם את המונה באותו מספר.`,
                explain: `${M(`${F(n, d)} = ${F(n * T / d, T)} = ${M(N.str(v))}`)}` };
            }
            if (k === 'd2f') {
              const [n, d] = pick(nice.filter(([a, b]) => b <= 25 && ![10].includes(b))), v = N.str(fracDec(n, d));
              const wrong = [[n, d * 10], [d, n], [n + 1, d], [Math.round(N.num(fracDec(n, d)) * 10), 100]].filter(([a, b]) => a > 0 && a / b !== n / d).slice(0, 3);
              return { q: `איזה שבר שווה ל-${M(v)}?`, type: 'choice', options: shuffle([[n, d], ...wrong]).map(([a, b]) => ({ h: F(a, b), v: a + '/' + b })), answer: n + '/' + d,
                hint: `קודם כתוב את ${M(v)} כשבר עם מכנה 10, 100 או 1000, ואז צמצם.`,
                explain: (() => { const T = 10 ** N.places(fracDec(n, d)), top = fracDec(n, d) * T / U; return `${M(`${M(v)} = ${F(top, T)} = ${F(n, d)}`)} (מצמצמים: מחלקים למעלה ולמטה ב-${top / n})`; })() };
            }
            if (k === 'pct') {
              const p = pick([25, 50, 10, 75, 5, 40, 12, 60]);
              return Math.random() < .5
                ? { q: `${M(p + '%')} כשבר עשרוני זה...`, type: 'input', answer: N.of(p, 2), hint: 'אחוז = מאית.', explain: `${M(p + '%')} = ${p} מאיות = ${M(N.str(N.of(p, 2)))}` }
                : { q: `${M(N.str(N.of(p, 2)))} זה כמה אחוזים?`, type: 'input', answer: N.of(p), unit: '%', hint: 'כמה מאיות יש במספר הזה?', explain: `${M(N.str(N.of(p, 2)))} = ${p} מאיות = ${M(p + '%')}` };
            }
            if (k === 'mixed') {
              const w = ri(1, 5), [n, d] = pick([[1, 2], [1, 4], [3, 4], [1, 5], [2, 5]]), v = N.of(w) + fracDec(n, d);
              return { q: `כתוב כשבר עשרוני: ${M(String(w))} ${F(n, d)} &nbsp;(${HEB.count(w, 'שלם', 'שלמים', false)} ועוד ${F(n, d)})`, type: 'input', answer: v,
                hint: `השלמים הולכים לפני הנקודה. כמה זה ${F(n, d)}?`, explain: `${w} + ${M(`${M(N.str(fracDec(n, d)))} = ${M(N.str(v))}`)}` };
            }
            const [n, d] = pick([[3, 4], [1, 4], [2, 5], [1, 8], [3, 5]]), fv = fracDec(n, d);
            let x; do { x = fv + pick([-1, 1]) * N.of(ri(1, 9), 2); } while (x <= 0 || x >= U);
            const right = fv > x ? 'f' : 'x';
            return { q: 'מי גדול יותר?', type: 'choice', options: [{ h: F(n, d), v: 'f' }, { h: M(N.str(x)), v: 'x' }], answer: right,
              hint: 'הפוך את השבר לשבר עשרוני, ואז השווה כמו בפרק 5.', explain: `${M(`${F(n, d)} = ${M(N.str(fv))}`)}, ולכן ${M(right === 'f' ? `${N.str(fv)} > ${N.str(x)}` : `${N.str(x)} > ${N.str(fv)}`)}.` };
          },
        });
      },
    },
    {
      t: 'מה לקחת מהפרק', auto: true,
      r(el) {
        el.innerHTML = `<div class="learned"><ul>
          <li>שבר → עשרוני: הופכים את המכנה ל-10, 100 או 1000. ${M(`${F(3, 5)} = ${F(6, 10)} = ${M('0.6')}`)}.</li>
          <li>עשרוני → שבר: כותבים עם מכנה 10, 100 או 1000 ומצמצמים. ${M(`${M('0.25')} = ${F(25, 100)} = ${F(1, 4)}`)}.</li>
          <li>שבר הוא חילוק: ${M(`${F(1, 3)} = ${M('1 ÷ 3 = 0.333...')}`)}. יש שברים שהם שבר עשרוני אינסופי.</li>
          <li>אחוז = מאית: ${M('40% = 0.4')}, ${M('0.07 = 7%')}.</li>
          <li>ששת הזוגות שכדאי לזכור: ${M(`${F(1, 2)} = ${M('0.5')}`)}, ${M(`${F(1, 4)} = ${M('0.25')}`)}, ${M(`${F(3, 4)} = ${M('0.75')}`)}, ${M(`${F(1, 5)} = ${M('0.2')}`)}, ${M(`${F(1, 10)} = ${M('0.1')}`)}, ${M(`${F(1, 8)} = ${M('0.125')}`)}.</li>
        </ul></div>`;
      },
    },
  ],
});
