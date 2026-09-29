'use strict';
/* פרק 9: חילוק. מחלקים כסף בין חברים (ופורטים כשצריך), כמה פעמים נכנס, והטריק של כפל שני המספרים ב-10. */

const COIN_OF = { 100: ['shekel', '1<br>₪'], 10: ['ten', '10<br>אג׳'], 1: ['one', '1<br>אג׳'] };
const coinHTML = (u, sm) => `<span class="coin ${COIN_OF[u][0]}" style="${sm ? 'transform:scale(.72);margin:-7px' : ''};cursor:default">${COIN_OF[u][1]}</span>`;

App.add({
  id: 'ch9', short: 'חילוק', icon: '➗', title: 'חילוק שברים עשרוניים',
  desc: 'מחלקים כסף, פורטים, ולמה חילוק יכול להגדיל',
  intro: `<p>יש שתי שאלות שחילוק עונה עליהן: <b>כמה כל אחד מקבל?</b> (מחלקים ${M('7.5')} ₪ בין 3 חברים) ו<b>כמה פעמים זה נכנס?</b> (כמה חצאים יש ב-3). נבין את שתיהן.</p>`,
  steps: [
    {
      t: 'מחלקים כסף בין חברים',
      r(el, done) {
        el.innerHTML = `
          <p class="do">לחץ "הצעד הבא" ותראה איך מחלקים כסף בהוגנות. כשמטבע לא מתחלק, <b>פורטים</b> אותו למטבעות קטנים.</p>
          <div class="controls pr-pick"></div>
          <div class="panel"><div class="muted small">הקופה באמצע:</div><div class="pile coins" style="min-height:60px"></div></div>
          <div class="friends" style="display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin:10px 0"></div>
          <div class="dmsg tip"></div>
          <div class="controls"><button class="btn primary nx">הצעד הבא ←</button></div>
          <div class="after" hidden><div class="aha"><b>זה בדיוק חילוק ארוך!</b> מחלקים קודם את הגדולים (שלמים). מה שנשאר פורטים ל-10 עשיריות ומחלקים אותן. מה שנשאר מזה פורטים למאיות, וכן הלאה. הנקודה בתשובה נמצאת בדיוק איפה שעוברים משקלים לאגורות.</div></div>`;
        const probs = [['7.5', 3], ['9.6', 4], ['5', 4]];
        const finished = new Set();
        let pi = 0, pile, fr, steps, si, n;
        el.querySelector('.pr-pick').innerHTML = probs.map(([a, k], j) => `<button class="btn sm pp" data-j="${j}">${M(`${a} ÷ ${k}`)}</button>`).join('');
        el.querySelectorAll('.pp').forEach(b => b.onclick = () => load(+b.dataset.j));
        function load(j) {
          pi = j; const [a, k] = probs[j]; n = k;
          el.querySelectorAll('.pp').forEach(b => b.classList.toggle('on', +b.dataset.j === j));
          const ag = Math.round(N.parse(a) / 1e4);
          pile = { 100: Math.floor(ag / 100), 10: Math.floor(ag % 100 / 10), 1: ag % 10 };
          fr = Array.from({ length: n }, () => ({ 100: 0, 10: 0, 1: 0 }));
          // בונים מראש את רשימת הצעדים
          steps = []; const P = { ...pile };
          for (const u of [100, 10, 1]) {
            const q = Math.floor(P[u] / n);
            if (P[u] > 0) steps.push({ kind: 'share', u, q, rem: P[u] - q * n, had: P[u] });
            P[u] -= q * n;
            if (P[u] > 0 && u > 1) { steps.push({ kind: 'break', u, c: P[u] }); P[u / 10] += 10 * P[u]; P[u] = 0; }
          }
          steps.push({ kind: 'end' });
          si = 0; draw();
          say(`מחלקים ${M(a)} ₪ בין ${n} חברים. בקופה: ${pile[100]} שקלים${pile[10] ? ` ו-${pile[10]} מטבעות של 10 אג׳` : ''}.`);
          el.querySelector('.nx').disabled = false;
        }
        const name = { 100: 'שקלים', 10: 'מטבעות של 10 אג׳', 1: 'אגורות' };
        function say(h) { const m = el.querySelector('.dmsg'); m.innerHTML = h; m.classList.remove('flash'); void m.offsetWidth; m.classList.add('flash'); }
        function draw() {
          const P = el.querySelector('.pile'); P.innerHTML = [100, 10, 1].map(u => Array(pile[u]).fill(coinHTML(u, true)).join('')).join('') || '<span class="muted">ריקה</span>';
          el.querySelector('.friends').innerHTML = fr.map((f, k) => {
            const tot = f[100] * 100 + f[10] * 10 + f[1];
            return `<div class="panel" style="text-align:center"><div>🧒 חבר ${k + 1}</div><div class="coins" style="justify-content:center;min-height:44px">${[100, 10, 1].map(u => Array(f[u]).fill(coinHTML(u, true)).join('')).join('')}</div><b>${M(N.str(N.of(tot, 2), tot % 100 ? 2 : 0))} ₪</b></div>`;
          }).join('');
        }
        el.querySelector('.nx').onclick = () => {
          const st = steps[si++];
          if (st.kind === 'share') {
            pile[st.u] -= st.q * n; fr.forEach(f => f[st.u] += st.q); draw();
            say(st.q ? `מחלקים את ה${name[st.u]}: ${st.had} ÷ ${n}, כל אחד מקבל <b>${st.q}</b>${st.rem ? `, ונשארים ${st.rem}` : ', ולא נשאר כלום'}.`
                     : `יש רק ${st.had} ${name[st.u]}, פחות מ-${n}. אי אפשר לתת לכל אחד אפילו אחד.`);
          } else if (st.kind === 'break') {
            pile[st.u / 10] += 10 * st.c; pile[st.u] = 0; draw();
            say(`נשארו ${st.c} ${name[st.u]} שאי אפשר לחלק. <b>פורטים!</b> ${st.u === 100 ? 'כל שקל הופך ל-10 מטבעות של 10 אגורות' : 'כל מטבע של 10 אגורות הופך ל-10 אגורות בודדות'}. עכשיו יש ${pile[st.u / 10]} ${name[st.u / 10]}.`);
          } else {
            const f = fr[0], tot = f[100] * 100 + f[10] * 10 + f[1];
            say(`<b>סיימנו!</b> כל חבר קיבל ${f[100] === 1 ? 'שקל אחד' : f[100] + ' שקלים'} ו-${f[10] * 10 + f[1]} אגורות. ${M(`${probs[pi][0]} ÷ ${n} = ${N.str(N.of(tot, 2))}`)}`);
            el.querySelector('.nx').disabled = true; finished.add(pi);
            if (finished.size >= 2) { el.querySelector('.after').hidden = false; done(); }
            else say(el.querySelector('.dmsg').innerHTML + `<br>עכשיו נסה תרגיל אחר מהכפתורים למעלה.`);
          }
        };
        load(0);
      },
    },
    {
      t: 'כמה פעמים זה נכנס?',
      r(el, done) {
        el.innerHTML = `
          <p>${M('3 ÷ 0.5')} שואל: <b>כמה חצאים יש ב-3?</b> יש לך 3 טבלאות שוקולד, ואתה חותך אותן לחתיכות של חצי טבלה. כמה חתיכות יצאו?</p>
          <p class="do">בחר כמה טבלאות וגודל חתיכה, וספור כמה חתיכות יוצאות.</p>
          <div class="controls">
            <span class="muted small">טבלאות:</span>${[1, 2, 3, 4].map(t => `<button class="btn sm tb" data-t="${t}">${t}</button>`).join('')}
            <span style="width:10px"></span><span class="muted small">גודל חתיכה:</span>${['0.5', '0.25', '0.2', '0.1'].map(d => `<button class="btn sm pc" data-d="${d}">${M(d)}</button>`).join('')}
          </div>
          <div class="cut-h"></div>
          <div class="panel ro"></div>
          <div class="mhost"></div>
          <div class="after" hidden><div class="aha"><b>הפתעה נוספת:</b> כשמחלקים במספר <b>קטן מ-1</b>, התוצאה <b>גדולה</b> מהמספר שהתחלנו איתו! ${M('3 ÷ 0.5 = 6')}. כי חתיכות קטנות נכנסות הרבה פעמים. בדיוק ההפך מכפל: כפל ב-${M('0.5')} מקטין לחצי, וחילוק ב-${M('0.5')} מכפיל פי 2.</div></div>`;
        let T = 3, D = '0.5';
        const mis = Missions(el.querySelector('.mhost'), [
          { t: `גלה כמה זה ${M('3 ÷ 0.5')}`, ok: s => s === '3|0.5' },
          { t: `גלה כמה זה ${M('2 ÷ 0.25')}`, ok: s => s === '2|0.25', after: () => note(`בכל טבלה יש 4 רבעים, אז בשתי טבלאות 8. ${M('2 ÷ 0.25 = 8')}`) },
          { t: `גלה כמה זה ${M('1 ÷ 0.1')}`, ok: s => s === '1|0.1', after: () => note(`בשלם יש 10 עשיריות: ${M('1 ÷ 0.1 = 10')}. חילוק ב-${M('0.1')} זה כמו כפל ב-10!`) },
          { t: `גלה כמה זה ${M('4 ÷ 0.2')}`, ok: s => s === '4|0.2' },
        ], () => { el.querySelector('.after').hidden = false; done(); });
        function note(h) { const d = document.createElement('div'); d.className = 'tip flash'; d.innerHTML = h; el.querySelector('.mhost').appendChild(d); }
        function draw() {
          el.querySelectorAll('.tb').forEach(b => b.classList.toggle('on', +b.dataset.t === T));
          el.querySelectorAll('.pc').forEach(b => b.classList.toggle('on', b.dataset.d === D));
          const s = 120, gap = 18, d = N.num(N.parse(D)), W = T * s + (T - 1) * gap + 12;
          const svg = S.svg(W, s + 40, 'board'); svg.style.maxWidth = W + 'px';
          let cnt = 0;
          for (let b = 0; b < T; b++) {
            const x = 6 + b * (s + gap);
            drawBar(svg, x, 6, s, 10);
            const per = Math.round(1 / d);
            for (let k = 0; k < per; k++) {
              cnt++;
              S.el('rect', { x: x + k * d * s + 1, y: 6, width: d * s - 2, height: s, fill: cnt % 2 ? 'rgba(250,204,21,.38)' : 'rgba(59,130,246,.3)', stroke: '#fff', 'stroke-width': 2, rx: 3 }, svg);
              S.text(svg, x + (k + .5) * d * s, s + 30, String(cnt), { 'text-anchor': 'middle', 'font-size': per > 5 ? 11 : 15, 'font-weight': 700, fill: '#374151' });
            }
          }
          const H = el.querySelector('.cut-h'); H.innerHTML = ''; H.appendChild(svg);
          el.querySelector('.ro').innerHTML = `${T} טבלאות, חתוכות לחתיכות של ${M(D)}: יוצאות <b>${cnt} חתיכות</b>. <span class="bignum" style="font-size:1.6rem">${M(`${T} ÷ ${D} = ${cnt}`)}</span>`;
          mis.check(T + '|' + D);
        }
        el.querySelectorAll('.tb').forEach(b => b.onclick = () => { T = +b.dataset.t; draw(); });
        el.querySelectorAll('.pc').forEach(b => b.onclick = () => { D = b.dataset.d; draw(); });
        draw();
      },
    },
    {
      t: 'הטריק: מזיזים את שניהם',
      r(el, done) {
        el.innerHTML = `
          <p>איך מחשבים ${M('4.8 ÷ 0.6')}? נחשוב בכסף: ${M('4.8')} ₪ זה 48 מטבעות של 10 אגורות. ${M('0.6')} ₪ זה 6 מטבעות כאלה. כמה ערימות של 6 מטבעות אפשר לעשות מ-48 מטבעות? ${M('48 ÷ 6 = 8')}.</p>
          <div class="def"><b>הטריק:</b> אם כופלים את <b>שני</b> המספרים ב-10, התשובה של החילוק לא משתנה. (השאלה "כמה פעמים 6 נכנס ב-48" היא אותה שאלה כמו "כמה פעמים ${M('0.6')} נכנס ב-${M('4.8')}".)<br>
          אז כופלים את שניהם ב-10 שוב ושוב, <b>עד שהמחלק (המספר שמחלקים בו) נהיה שלם</b>. ואז מחלקים כרגיל.</div>
          <p class="do">לחץ "× 10 לשניהם" עד שהמספר השני יהיה שלם, ואז כתוב את התשובה.</p>
          <div class="panel" style="text-align:center"><div class="chain bignum" style="font-size:1.7rem;line-height:1.7"></div>
            <div class="controls" style="justify-content:center"><button class="btn primary x10">× 10 לשניהם</button></div>
            <div class="inrow" style="justify-content:center"><input class="num-in ans" dir="ltr" inputmode="decimal" placeholder="התשובה"><button class="btn chk">בדיקה</button></div>
            <div class="tfb"></div></div>
          <div class="muted rnd"></div>`;
        const probs = [['4.8', '0.6'], ['1.2', '0.04'], ['7', '0.5'], ['0.36', '1.2']];
        let r = 0, a, b, chain;
        function load() {
          [a, b] = probs[r]; chain = [[a, b]];
          el.querySelector('.rnd').textContent = `תרגיל ${r + 1} מתוך ${probs.length}`;
          el.querySelector('.ans').value = ''; el.querySelector('.ans').disabled = false; el.querySelector('.chk').disabled = false;
          el.querySelector('.tfb').innerHTML = ''; draw();
        }
        function draw() {
          el.querySelector('.chain').innerHTML = M(chain.map(([x, y]) => `${x} ÷ ${y}`).join(' = ') + ' = ?');
          const last = chain[chain.length - 1];
          el.querySelector('.x10').disabled = Number.isInteger(N.num(N.parse(last[1])));
        }
        el.querySelector('.x10').onclick = () => {
          const [x, y] = chain[chain.length - 1];
          chain.push([N.str(N.parse(x) * 10), N.str(N.parse(y) * 10)]); draw();
          if (Number.isInteger(N.num(N.parse(chain[chain.length - 1][1])))) el.querySelector('.tfb').innerHTML = `<p class="muted">עכשיו המספר השני שלם. אפשר לחלק כרגיל.</p>`;
        };
        el.querySelector('.chk').onclick = () => {
          const v = N.parse(el.querySelector('.ans').value), right = N.div(N.parse(a), N.parse(b));
          const f = el.querySelector('.tfb');
          if (v === right) {
            const [x, y] = chain[chain.length - 1];
            f.innerHTML = `<div class="qz-fb ok"><b>נכון! ✓</b> ${M(`${a} ÷ ${b} = ${x} ÷ ${y} = ${N.str(right)}`)}${r === 3 ? `. שים לב: כאן חילקנו במספר גדול מ-1, ולכן התוצאה קטנה.` : ''}</div>
              <button class="btn primary nxt" style="margin-top:8px">${r === probs.length - 1 ? 'סיימתי ✓' : 'לתרגיל הבא ←'}</button>`;
            el.querySelector('.ans').disabled = true; el.querySelector('.chk').disabled = true;
            f.querySelector('.nxt').onclick = () => { if (r === probs.length - 1) { f.innerHTML = '<div class="aha">🏆 כל הכבוד!</div>'; confetti(); done(); } else { r++; load(); } };
          } else {
            const [, y] = chain[chain.length - 1];
            f.innerHTML = `<div class="qz-fb bad">${Number.isInteger(N.num(N.parse(y))) ? `לא בדיוק. חשב את ${M(chain[chain.length - 1].join(' ÷ '))}. ${r === 3 ? 'התוצאה יכולה להיות קטנה מ-1.' : ''}` : 'קודם תלחץ "× 10 לשניהם" עד שהמספר השני יהיה שלם. ככה הרבה יותר קל.'}</div>`;
          }
        };
        load();
      },
    },
    {
      t: 'משחק: חילוק',
      r(el, done) {
        el.innerHTML = `<div class="qhost"></div>`;
        const kinds = shuffle(['byWhole', 'byWhole', 'byWhole', 'byDec', 'byDec', 'byDec', 'size', 'size']);
        Quiz(el.querySelector('.qhost'), {
          rounds: 8, onDone: done,
          gen(i) {
            if (kinds[i] === 'byWhole') {
              const n = ri(2, 5), q = pick([N.of(ri(11, 49), 1), N.of(ri(101, 399), 2)]), a = q * n;
              if (N.places(a) > 2) return this.gen(i);
              return { q: `${M(`${N.str(a)} ÷ ${n} = ?`)}`, type: 'input', answer: q, hint: `תחשוב על ${M(N.str(a))} שקלים שמתחלקים בין ${n} חברים. קודם השקלים, ואז לפרוט את מה שנשאר.`,
                explain: `${M(`${N.str(a)} ÷ ${n} = ${N.str(q)}`)}. בדיקה: ${M(`${N.str(q)} × ${n} = ${N.str(a)}`)} ✓` };
            }
            if (kinds[i] === 'byDec') {
              const d = pick(['0.2', '0.5', '0.4', '0.3', '0.25', '0.05', '1.5']), q = ri(2, 12), a = N.parse(d) * q;
              const k = N.places(N.parse(d)), f = 10 ** k;
              return { q: `${M(`${N.str(a)} ÷ ${d} = ?`)}`, type: 'input', answer: N.of(q), hint: `כפול את שני המספרים ב-${f}, כדי ש-${M(d)} יהיה שלם.`,
                explain: `${M(`${N.str(a)} ÷ ${d} = ${N.str(a * f)} ÷ ${N.str(N.parse(d) * f)} = ${q}`)}` };
            }
            const n = ri(4, 30), d = pick(['0.5', '0.1', '2', '0.25', '4']), small = N.parse(d) < U;
            return choiceQ(`בלי לחשב: ${M(`${n} ÷ ${d}`)} גדול מ-${M(String(n))} או קטן ממנו?`, small ? `גדול מ-${n}` : `קטן מ-${n}`, [small ? `קטן מ-${n}` : `גדול מ-${n}`, `שווה ל-${n}`],
              { hint: `כמה פעמים ${M(d)} נכנס ב-${n}? הרבה או מעט?`, explain: `${M(d)} ${small ? 'קטן מ-1, אז הוא נכנס הרבה פעמים' : 'גדול מ-1, אז הוא נכנס פחות פעמים'}: ${M(`${n} ÷ ${d} = ${N.str(N.div(N.of(n), N.parse(d)))}`)}` });
          },
        });
      },
    },
    {
      t: 'מה לקחת מהפרק', auto: true,
      r(el) {
        el.innerHTML = `<div class="learned"><ul>
          <li>חילוק במספר שלם = חלוקה בין חברים. מחלקים קודם את השלמים, ומה שנשאר פורטים לעשיריות, ואז למאיות. ${M('7.5 ÷ 3 = 2.5')}.</li>
          <li>חילוק במספר עשרוני = "כמה פעמים זה נכנס?". ${M('3 ÷ 0.5 = 6')}, כי בכל שלם יש 2 חצאים.</li>
          <li><b>חילוק במספר קטן מ-1 מגדיל.</b> חילוק במספר גדול מ-1 מקטין.</li>
          <li>הטריק: כופלים את שני המספרים ב-10 (או 100) עד שהמחלק שלם. ${M('4.8 ÷ 0.6 = 48 ÷ 6 = 8')}.</li>
          <li>בודקים חילוק עם כפל: ${M('8 × 0.6 = 4.8')} ✓</li>
        </ul></div>`;
      },
    },
  ],
});
