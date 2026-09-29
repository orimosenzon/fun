'use strict';
/* פרק 2: מאיות. חותכים כל עשירית לעשרה, מקבלים ריבוע של 100. ואחר כך כסף: שקל = 100 אגורות. */

// ריבוע מאה קטן ומסודר עם n מאיות צבועות
function gridSVG(n, s = 150, opt = {}) {
  const svg = S.svg(s + 8, s + 8, 'board'); svg.style.maxWidth = (opt.maxW || s + 8) + 'px';
  drawGrid(svg, 4, 4, s, cellsOf(n), { byRole: true });
  return svg;
}
const hundStr = n => N.str(N.of(n, 2), 2);

// ארנק: לחיצה על מטבע מוסיפה, לחיצה על מטבע במגש מורידה
const COINS = [{ k: 'shekel', a: 100, h: '1<br>₪' }, { k: 'ten', a: 10, h: '10<br>אג׳' }, { k: 'one', a: 1, h: '1<br>אג׳' }];
function Wallet(el, onChange) {
  let tray = [];
  el.innerHTML = `<div class="tray"></div><div class="coins" style="margin-top:10px">${COINS.map(c => `<button class="coin ${c.k}" data-a="${c.a}" title="הוסף">${c.h}</button>`).join('')}
    <button class="btn sm clr">לרוקן את המגש</button></div>`;
  const T = el.querySelector('.tray');
  const draw = () => {
    T.innerHTML = tray.slice().sort((a, b) => b - a).map(a => { const c = COINS.find(x => x.a === a); return `<button class="coin ${c.k}" data-rm="${a}" title="הורד">${c.h}</button>`; }).join('');
    T.querySelectorAll('[data-rm]').forEach(b => b.onclick = () => { tray.splice(tray.indexOf(+b.dataset.rm), 1); draw(); onChange(sum()); });
  };
  const sum = () => tray.reduce((s, a) => s + a, 0);
  el.querySelectorAll('.coins .coin').forEach(b => b.onclick = () => { if (tray.length < 40) { tray.push(+b.dataset.a); draw(); onChange(sum()); } });
  el.querySelector('.clr').onclick = () => { tray = []; draw(); onChange(0); };
  draw();
  return { reset() { tray = []; draw(); onChange(0); }, get sum() { return sum(); } };
}

App.add({
  id: 'ch2', short: 'מאיות', icon: '🟧', title: 'מאיות: חותכים עוד פעם',
  desc: 'ריבוע של 100, ולמה 0.5 גדול פי 10 מ-0.05',
  intro: `<p>עשירית היא חתיכה די גדולה. מה אם רוצים משהו קטן יותר? עושים שוב אותו טריק: חותכים כל עשירית ל-10 חלקים שווים.</p>`,
  steps: [
    {
      t: 'חותכים כל חתיכה לעשר',
      r(el, done) {
        el.innerHTML = `
          <div class="play">
            <div class="cut-host"></div>
            <div>
              <p>הנה שוב טבלת השוקולד, עם 10 הפסים שלה. כל פס הוא עשירית.</p>
              <p class="do">לחץ על הכפתור, ונחתוך כל פס ל-10 ריבועים קטנים.</p>
              <button class="btn primary cut">✂️ לחתוך</button>
              <div class="qhost"></div>
            </div>
          </div>`;
        const host = el.querySelector('.cut-host');
        const svg = S.svg(300, 300, 'board'); svg.style.maxWidth = '330px'; host.appendChild(svg);
        const X = 10, s = 280;
        drawBar(svg, X, X, s, 10);
        el.querySelector('.cut').onclick = async e => {
          e.target.disabled = true;
          const lines = S.el('g', {}, svg);
          for (let r = 1; r < 10; r++) {
            const ln = S.el('line', { x1: X, y1: X + r * s / 10, x2: X, y2: X + r * s / 10, stroke: '#fff', 'stroke-width': 3 }, lines);
            const t0 = performance.now();
            await new Promise(res => { const f = now => { const k = clamp((now - t0) / 160, 0, 1); ln.setAttribute('x2', X + s * k); k < 1 ? requestAnimationFrame(f) : res(); }; requestAnimationFrame(f); });
          }
          await sleep(250);
          S.clear(svg); drawGrid(svg, X, X, s, cellsOf(100), { color: 'choc' });
          const q = el.querySelector('.qhost');
          q.innerHTML = `<div class="modal-q">כמה ריבועים קטנים יש עכשיו בכל הטבלה?
            <div class="opts" style="margin-top:8px">${[10, 20, 100, 1000].map(o => `<button class="opt" data-v="${o}">${o}</button>`).join('')}</div><div class="fbq"></div></div>`;
          q.querySelectorAll('.opt').forEach(b => b.onclick = () => {
            const ok = +b.dataset.v === 100;
            b.classList.add(ok ? 'right' : 'wrong');
            if (!ok) { q.querySelector('.fbq').innerHTML = `<p class="muted">לא בדיוק. כמה ריבועים יש בפס אחד? וכמה פסים יש?</p>`; return; }
            q.querySelectorAll('.opt').forEach(x => x.disabled = true);
            q.querySelector('.fbq').innerHTML = `<p><b>100!</b> 10 פסים, ובכל פס 10 ריבועים: ${M('10 × 10 = 100')}.</p>
              <p class="def">כל ריבוע קטן הוא <b>מאית</b> מהטבלה, כי יש 100 כאלה בשלם.<br>כותבים אותה כך: ${M(`${M('0.01')} = ${F(1, 100)}`)}</p>
              <p>ובתוך כל פס (עשירית) יש 10 מאיות. זה אותו דפוס כמו קודם: <b>כל חלק מתחלק ל-10 חלקים קטנים יותר.</b></p>`;
            done();
          });
        };
      },
    },
    {
      t: 'צובעים מאיות',
      r(el, done) {
        el.innerHTML = `
          <p>עכשיו אתה הצייר. כל ריבוע קטן = מאית. שים לב לצבעים: פס <b class="t-pm1">מלא</b> נצבע בכחול, כי הוא עשירית שלמה. ריבועים <b class="t-pm2">בודדים</b> נשארים כתומים, כי הם מאיות.</p>
          <p class="do">גרור על הריבוע כדי לצבוע (או למחוק). אפשר גם לבחור מכחול של פס שלם.</p>
          <div class="play">
            <div>
              <div class="grid-host"></div>
              <div class="controls">
                <span class="muted small">מכחול:</span>
                <button class="btn sm on tool" data-t="cell">▪ ריבוע אחד</button>
                <button class="btn sm tool" data-t="col">▮ פס שלם</button>
                <button class="btn sm arrange">🧹 סדר לי</button>
                <button class="btn sm clr">נקה</button>
              </div>
            </div>
            <div><div class="panel readout"></div><div class="mhost"></div></div>
          </div>
          <div class="after" hidden><div class="aha">
            <b>מה גילית:</b> הספרה הראשונה אחרי הנקודה סופרת <b class="t-pm1">פסים (עשיריות)</b>, והשנייה סופרת <b class="t-pm2">ריבועים (מאיות)</b>.
            ${M('0.23')} = 2 עשיריות ו-3 מאיות = 23 מאיות. שתי הדרכים נכונות, וזה אותו מספר בדיוק.</div></div>`;
        const cells = Array(100).fill(false);
        let tool = 'cell';
        const host = el.querySelector('.grid-host'), ro = el.querySelector('.readout');
        const svg = S.svg(300, 300, 'board cells-paint'); svg.style.maxWidth = '380px'; host.appendChild(svg);
        const X = 10, s = 280, w = s / 10;
        const count = () => cells.filter(Boolean).length;
        const mis = Missions(el.querySelector('.mhost'), [
          { t: `צבע בדיוק ${M('0.23')}`, ok: n => n === 23, after: () => note(`2 פסים שלמים ועוד 3 ריבועים. או בקיצור: 23 מאיות.`) },
          { t: `צבע בדיוק ${M('0.4')}`, ok: n => n === 40, after: () => note(`שים לב: ${M('0.4')} זה 4 פסים, וזה בדיוק 40 ריבועים. לכן ${M('0.4 = 0.40')}. אפס בסוף המספר לא משנה כלום.`) },
          { t: `צבע בדיוק ${M('0.07')}`, ok: n => n === 7, after: () => note(`${M('0.07')} זה רק 7 ריבועים קטנים. ה-0 שאחרי הנקודה אומר: אין אף פס שלם.`) },
          { t: `צבע בדיוק ${M('0.7')}`, ok: n => n === 70, after: () => note(`${M('0.7')} = 70 ריבועים, פי 10 יותר מ-${M('0.07')}! אפס אחד במקום הלא נכון משנה הכל.`) },
          { t: 'צבע בדיוק רבע מהטבלה', ok: n => n === 25, after: () => note(`רבע מ-100 זה 25, אז רבע = ${M('0.25')}. כמו 25 אגורות, שהן רבע שקל.`) },
        ], () => { el.querySelector('.after').hidden = false; done(); });
        function note(h) { const d = document.createElement('div'); d.className = 'tip flash'; d.innerHTML = h; el.querySelector('.mhost').appendChild(d); }
        function draw() {
          S.clear(svg); drawGrid(svg, X, X, s, cells, { byRole: true });
          const n = count(), t = Math.floor(n / 10), h = n % 10;
          ro.innerHTML = `
            <div class="ro-row"><span class="ro-lbl">צבעת</span><b>${n} ריבועים = ${n} מאיות</b></div>
            <div class="ro-row"><span class="ro-lbl">כשבר</span>${F(n, 100)}</div>
            <div class="ro-big"><div class="bignum">${digitsHTML(hundStr(n))}</div>
              <div class="place-tags"><span class="t-p0">שלמים</span><span>.</span><span class="t-pm1">עשיריות</span><span class="t-pm2">מאיות</span></div></div>
            <div class="ro-row small"><span class="ro-lbl">אם מסדרים</span><span><b class="t-pm1">${t} פסים</b> ועוד <b class="t-pm2">${h} ריבועים</b></span></div>`;
        }
        let painting = null;
        const cellAt = e => {
          const p = S.pt(svg, e), c = Math.floor((p.x - X) / w), rv = Math.floor((p.y - X) / w);
          if (c < 0 || c > 9 || rv < 0 || rv > 9) return -1;
          return c * 10 + (9 - rv);
        };
        const paint = i => {
          if (i < 0) return;
          if (tool === 'col') { const c = Math.floor(i / 10); for (let j = 0; j < 10; j++) cells[c * 10 + j] = painting; }
          else cells[i] = painting;
          draw();
        };
        svg.addEventListener('pointerdown', e => {
          const i = cellAt(e); if (i < 0) return; e.preventDefault();
          svg.setPointerCapture(e.pointerId);
          painting = tool === 'col' ? !cells.slice(Math.floor(i / 10) * 10, Math.floor(i / 10) * 10 + 10).every(Boolean) : !cells[i];
          paint(i);
        });
        svg.addEventListener('pointermove', e => { if (painting !== null) paint(cellAt(e)); });
        const end = () => { if (painting !== null) { painting = null; mis.check(count()); } };
        svg.addEventListener('pointerup', end); svg.addEventListener('pointercancel', end);
        el.querySelectorAll('.tool').forEach(b => b.onclick = () => { tool = b.dataset.t; el.querySelectorAll('.tool').forEach(x => x.classList.toggle('on', x === b)); });
        el.querySelector('.arrange').onclick = () => { const n = count(); cellsOf(n).forEach((v, i) => cells[i] = v); draw(); };
        el.querySelector('.clr').onclick = () => { cells.fill(false); draw(); };
        draw();
      },
    },
    {
      t: 'האפס הערמומי',
      r(el, done) {
        el.innerHTML = `
          <p>שלושה מספרים שנראים כמעט אותו דבר:</p>
          <div class="duel"><span class="dcard" style="cursor:default">${digitsHTML('0.5')}</span><span class="dcard" style="cursor:default">${digitsHTML('0.05')}</span><span class="dcard" style="cursor:default">${digitsHTML('0.50')}</span></div>
          <div class="modal-q">מה נכון?
            <div class="opts" style="margin-top:8px">
              <button class="opt" data-v="a">${M('0.05')} הכי גדול, כי יש בו הכי הרבה ספרות</button>
              <button class="opt" data-v="b">${M('0.50')} הכי גדול, כי 50 גדול מ-5</button>
              <button class="opt" data-v="c">${M('0.5')} ו-${M('0.50')} שווים, ו-${M('0.05')} הכי קטן</button>
              <button class="opt" data-v="d">שלושתם שווים</button>
            </div><div class="fbq"></div></div>
          <div class="reveal" hidden>
            <div class="vs" style="text-align:center">
              <div><div class="g1"></div><b>${M('0.5')}</b><div class="muted small">5 פסים = 50 ריבועים</div></div>
              <div><div class="g2"></div><b>${M('0.05')}</b><div class="muted small">רק 5 ריבועים</div></div>
              <div><div class="g3"></div><b>${M('0.50')}</b><div class="muted small">5 פסים ו-0 ריבועים</div></div>
            </div>
            <div class="aha"><b>שני סוגים של אפסים:</b>
              <ul>
                <li><b>אפס בסוף</b>, אחרי הספרה האחרונה, לא משנה את הערך. ${M('0.5 = 0.50 = 0.500')}. זה כמו לומר "5 פסים ו-0 ריבועים נוספים".</li>
                <li><b>אפס באמצע</b>, מיד אחרי הנקודה, <b>שומר מקום</b>. ב-${M('0.05')} הוא אומר "אין עשיריות בכלל", ודוחף את ה-5 למקום של המאיות. ככה ה-5 שווה פי 10 פחות.</li>
              </ul></div>
            <p>ועוד משהו חשוב: <b>לא סופרים ספרות</b> כדי לדעת מי גדול. ${M('0.05')} ארוך יותר מ-${M('0.5')}, אבל קטן ממנו פי 10. נשחק בזה הרבה בפרק 5.</p>
          </div>`;
        el.querySelector('.g1').appendChild(gridSVG(50, 130)); el.querySelector('.g2').appendChild(gridSVG(5, 130)); el.querySelector('.g3').appendChild(gridSVG(50, 130));
        el.querySelectorAll('.opt').forEach(b => b.onclick = () => {
          const ok = b.dataset.v === 'c';
          el.querySelectorAll('.opt').forEach(x => x.disabled = true);
          b.classList.add(ok ? 'right' : 'wrong'); el.querySelector('[data-v="c"]').classList.add('right');
          el.querySelector('.fbq').innerHTML = ok ? '<p><b>בדיוק!</b> בוא נראה את זה בריבועים:</p>' : '<p><b>מלכודת קלאסית, ואתה לא לבד.</b> תסתכל על הריבועים:</p>';
          el.querySelector('.reveal').hidden = false; done();
        });
      },
    },
    {
      t: 'כסף: שקלים ואגורות',
      r(el, done) {
        el.innerHTML = `
          <p>יש מקום שבו אתה כבר משתמש במאיות כל יום, אפילו בלי לשים לב: <b>כסף</b>. בשקל אחד יש 100 אגורות. אז אגורה היא מאית שקל!</p>
          <table class="pv" style="direction:rtl"><tr><th class="p0">שלם</th><th class="pm1">עשירית</th><th class="pm2">מאית</th></tr>
            <tr><td class="p0" style="font-size:1rem">1 ₪</td><td class="pm1" style="font-size:1rem">10 אג׳ = ${M('0.1')} ₪</td><td class="pm2" style="font-size:1rem">1 אג׳ = ${M('0.01')} ₪</td></tr></table>
          <p class="muted small">(מטבע של אגורה אחת כבר לא קיים בישראל, אבל פה נעמיד פנים שכן.)</p>
          <p class="do">שלם בדיוק את המחיר שעל התווית. לחץ על מטבעות כדי לשים אותם במגש.</p>
          <div class="play">
            <div><div class="ro-row"><span class="ro-lbl">המחיר:</span><span class="price"></span><span class="muted rnd"></span></div><div class="wallet" style="margin-top:10px"></div></div>
            <div class="panel"><div class="ro-row"><span class="ro-lbl">במגש יש:</span><span class="bignum tot" style="font-size:2rem"></span></div><div class="coin-say muted"></div>
              <div class="controls"><button class="btn primary pay">💳 לשלם</button></div><div class="wfb"></div></div>
          </div>
          <div class="qhost"></div>`;
        const prices = ['2.30', '0.75', '1.05', '4.5', '0.08', '3.6'];
        let r = 0, cur = 0;
        const W = Wallet(el.querySelector('.wallet'), a => { cur = a; show(); });
        function show() {
          el.querySelector('.tot').innerHTML = digitsHTML(hundStr(cur)) + ' ₪';
          el.querySelector('.coin-say').innerHTML = `${Math.floor(cur / 100)} שקלים ו-${cur % 100} אגורות`;
        }
        function round() {
          if (r >= prices.length) return final();
          el.querySelector('.price').innerHTML = M(prices[r]) + ' ₪';
          el.querySelector('.rnd').textContent = `קנייה ${r + 1} מתוך ${prices.length}`;
          el.querySelector('.wfb').innerHTML = ''; W.reset();
        }
        el.querySelector('.pay').onclick = () => {
          const need = Math.round(N.parse(prices[r]) / 1e4), f = el.querySelector('.wfb');
          if (cur === need) {
            f.innerHTML = `<div class="qz-fb ok"><b>שולם! ✓</b> ${M(prices[r])} ₪ = ${Math.floor(need / 100)} שקלים ו-${need % 100} אגורות.${prices[r] === '4.5' ? ` שים לב: ה-5 הוא במקום של העשיריות, אז הוא 5 מטבעות של 10 אגורות, כלומר 50 אגורות.` : ''}</div>`;
            el.querySelector('.pay').disabled = true;
            setTimeout(() => { el.querySelector('.pay').disabled = false; r++; round(); }, prices[r] === '4.5' ? 3800 : 1900);
          } else {
            const hint = cur > need ? 'שמת יותר מדי.' : 'חסר עוד כסף.';
            f.innerHTML = `<div class="qz-fb bad"><b>${hint}</b> הספרה משמאל לנקודה אומרת כמה שקלים. הראשונה מימין לנקודה אומרת כמה מטבעות של 10 אגורות, והשנייה כמה אגורות בודדות.</div>`;
          }
        };
        function final() {
          el.querySelector('.play').hidden = true; el.querySelector('.do').hidden = true;
          el.querySelector('.qhost').innerHTML = `<div class="aha">🏆 שילמת את כל הקניות!</div>
            <div class="modal-q">ועכשיו שאלה: סבא אומר "${M('1.5')} שקלים זה שקל ו-5 אגורות". הוא צודק?
              <div class="opts" style="margin-top:8px"><button class="opt" data-v="a">כן</button><button class="opt" data-v="b">לא, זה שקל ו-50 אגורות</button><button class="opt" data-v="c">לא, זה 15 אגורות</button></div><div class="fbq"></div></div>`;
          el.querySelectorAll('.qhost .opt').forEach(b => b.onclick = () => {
            const ok = b.dataset.v === 'b';
            b.classList.add(ok ? 'right' : 'wrong');
            if (!ok) return;
            el.querySelectorAll('.qhost .opt').forEach(x => x.disabled = true);
            el.querySelector('.fbq').innerHTML = `<p>נכון. ה-5 יושב במקום של ה<b class="t-pm1">עשיריות</b>, אז הוא 5 עשיריות שקל = 50 אגורות. שקל ו-5 אגורות נכתב ${M('1.05')}.</p>`;
            done();
          });
        }
        show(); round();
      },
    },
    {
      t: 'משחק: מאיות',
      r(el, done) {
        el.innerHTML = `<div class="qhost"></div>`;
        const kinds = shuffle(['pic', 'toHund', 'fromWords', 'money', 'toAg', 'pic', 'bigHund', 'fromHund']);
        Quiz(el.querySelector('.qhost'), {
          rounds: 8, onDone: done,
          gen(i) {
            const k = kinds[i];
            if (k === 'pic') {
              const n = pick([ri(11, 99), ri(1, 9), ri(1, 9) * 10]);
              return { q: 'איזה מספר עשרוני צבוע כאן?', type: 'input', answer: N.of(n, 2), fig: f => f.appendChild(gridSVG(n, 170)),
                hint: 'ספור פסים כחולים (עשיריות) וריבועים כתומים (מאיות).', explain: `${Math.floor(n / 10)} פסים ו-${n % 10} ריבועים = ${M(N.str(N.of(n, 2)))}` };
            }
            if (k === 'toHund') {
              const t = ri(1, 9);
              return { q: `כמה מאיות יש ב-${M('0.' + t)}?`, type: 'input', answer: N.of(t * 10), unit: 'מאיות',
                hint: 'בכל עשירית (פס) יש 10 מאיות (ריבועים).', explain: `${t} פסים × 10 ריבועים = ${t * 10} מאיות. ${M(`0.${t} = 0.${t}0`)}` };
            }
            if (k === 'fromWords') {
              const n = pick([ri(1, 9), ri(11, 99)]);
              return { q: `כתוב כמספר עשרוני: <b>${HEB.count(n, 'מאית', 'מאיות')}</b>`, type: 'input', answer: N.of(n, 2),
                hint: n < 10 ? 'אין כאן אף עשירית שלמה. מה צריך לשים במקום של העשיריות?' : 'המאית היא הספרה השנייה אחרי הנקודה.',
                explain: `${M(hundStr(n))}${n < 10 ? `: ה-0 שומר את המקום של העשיריות.` : ''}` };
            }
            if (k === 'money') {
              const sh = ri(1, 9), ag = pick([ri(1, 9), ri(11, 99)]);
              return { q: `כתוב בשקלים, כמספר עשרוני: <b>${sh} שקלים ו-${ag} אגורות</b>`, type: 'input', answer: N.of(sh * 100 + ag, 2), unit: '₪',
                hint: 'אגורה היא מאית שקל. כמה ספרות צריך אחרי הנקודה?', explain: `${M(N.str(N.of(sh * 100 + ag, 2), 2))} ₪` };
            }
            if (k === 'toAg') {
              const v = pick([ri(11, 99), ri(1, 9) * 10, ri(101, 399)]);
              const s = N.str(N.of(v, 2));
              return { q: `כמה אגורות הן ${M(s)} שקלים?`, type: 'input', answer: N.of(v), unit: 'אגורות',
                hint: 'בכל שקל 100 אגורות.', explain: `${M(s)} ₪ = ${v} אגורות.` };
            }
            if (k === 'bigHund') {
              const n = ri(101, 299);
              return { q: `כמה מאיות יש בסך הכל ב-${M(hundStr(n))}?`, type: 'input', answer: N.of(n), unit: 'מאיות',
                hint: 'בכל שלם יש 100 מאיות.', explain: `${Math.floor(n / 100)} שלמים הם ${Math.floor(n / 100) * 100} מאיות, ועוד ${n % 100}: בסך הכל ${n}.` };
            }
            const t = ri(1, 9);
            return choiceQ(`מה שווה ל-${M('0.' + t)}?`, `0.${t}0`, [`0.0${t}`, `0.${t}${t}`, `${t}0`], { fmt: M,
              hint: 'תחשוב על הריבוע: כמה פסים? וכמה ריבועים בודדים?', explain: `${M(`0.${t} = 0.${t}0`)}: ${t} פסים ו-0 ריבועים נוספים.` });
          },
        });
      },
    },
    {
      t: 'מה לקחת מהפרק', auto: true,
      r(el) {
        el.innerHTML = `<div class="learned"><ul>
          <li><b>מאית</b> היא חלק אחד מתוך 100 של שלם. ${M(`${M('0.01')} = ${F(1, 100)}`)}. יש 10 מאיות בכל עשירית.</li>
          <li>הספרה השנייה אחרי הנקודה סופרת מאיות: ${M('0.37')} = 3 עשיריות ו-7 מאיות = 37 מאיות.</li>
          <li>אפס בסוף לא משנה: ${M('0.4 = 0.40')}. אפס שומר מקום כן משנה: ${M('0.05')} קטן פי 10 מ-${M('0.5')}.</li>
          <li>כסף: שקל = 100 אגורות, אז ${M('3.07')} ₪ = 3 שקלים ו-7 אגורות, ו-${M('1.5')} ₪ = שקל ו-50 אגורות.</li>
          <li>רבע = ${M('0.25')}, חצי = ${M('0.5')}, שלושה רבעים = ${M('0.75')}. בדיוק כמו במטבעות.</li>
        </ul></div>`;
      },
    },
  ],
});
