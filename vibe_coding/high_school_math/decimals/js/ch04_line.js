'use strict';
/* פרק 4: ציר המספרים. זכוכית מגדלת שנכנסת לתוך קטעים, משחק "פגע במטרה", ומספרים שבין לבין. */

// דיוק לתצוגה של ערך על ציר: לפי רוחב הטווח הנוכחי
const lineRound = (v, R) => { const p = Math.max(1, Math.round(-Math.log10(R)) + 2); return N.str(N.of(Math.round(v * 10 ** p), p)); };

App.add({
  id: 'ch4', short: 'ציר המספרים', icon: '🔍', title: 'ציר המספרים: זכוכית מגדלת',
  desc: 'איפה גרים המספרים, ומה מוצאים כשעושים זום',
  intro: `<p>כל מספר עשרוני גר במקום מסוים על ציר המספרים. בין 0 ל-1 יש הרבה יותר מקום ממה שנראה. בוא נעשה זום.</p>`,
  steps: [
    {
      t: 'זום לתוך הציר',
      r(el, done) {
        el.innerHTML = `
          <p class="do">לחץ על קטע בציר כדי להגדיל אותו. גרור את הסיכה הוורודה כדי להזיז אותה.</p>
          <div class="crumbs"></div>
          <div class="nl-host"></div>
          <div class="panel ro-row" style="margin-top:6px"><span class="ro-lbl">הסיכה על</span><span class="bignum pinval" style="font-size:2rem"></span><button class="btn sm out" style="margin-inline-start:auto">🔍− חזרה אחורה</button></div>
          <div class="mhost"></div>
          <div class="after" hidden><div class="aha"><b>מה ראית:</b> בין כל שתי שנתות יש עוד 10 שנתות קטנות. בין ${M('0.3')} ל-${M('0.4')} גרים ${M('0.31, 0.32, ..., 0.39')}. ובין ${M('0.37')} ל-${M('0.38')} גרים ${M('0.371, 0.372, ..., 0.379')}. אפשר להמשיך לעשות זום בלי סוף, ותמיד יהיו עוד מספרים באמצע.</div></div>`;
        const svg = S.svg(LW(), 170, 'board framed'); el.querySelector('.nl-host').appendChild(svg);
        const stack = [[0, 1]];
        const pin = { v: 0.2, drag: true, label: '', snap: 0.01 };
        let busy = false;
        const nl = new NumberLine(svg, {
          W: LW(), y: 105, a: 0, b: 1,
          onRender(L) {
            S.clear(L.gOver);
            if (busy || L.b - L.a < 0.0015) return;
            const t = (L.b - L.a) / 10;
            for (let j = 0; j < 10; j++) {
              const a = L.a + j * t, r = S.el('rect', { x: L.X(a), y: L.o.y - 22, width: L.X(a + t) - L.X(a), height: 44, class: 'zoom-seg' }, L.gOver);
              r.addEventListener('click', () => zoomIn(a, a + t));
            }
          },
        });
        const R = () => stack[stack.length - 1][1] - stack[stack.length - 1][0];
        function showPin() {
          pin.snap = R() / 100;
          el.querySelector('.pinval').innerHTML = digitsHTML(lineRound(pin.v, R()));
          nl.setPins([pin]);
        }
        pin.onMove = () => { el.querySelector('.pinval').innerHTML = digitsHTML(lineRound(pin.v, R())); };
        pin.onDrop = () => check();
        function crumbs() {
          el.querySelector('.crumbs').innerHTML = stack.map(([a, b], k) => `<button class="${k === stack.length - 1 ? 'cur' : ''}" data-k="${k}">${M(N.str(N.of(Math.round(a * 1e4), 4)) + ' – ' + N.str(N.of(Math.round(b * 1e4), 4)))}</button>`).join('<span>←</span>');
          el.querySelectorAll('.crumbs button').forEach(b => b.onclick = () => zoomBack(+b.dataset.k));
          el.querySelector('.out').disabled = stack.length === 1;
        }
        async function zoomIn(a, b) {
          if (busy) return; busy = true; nl.render();
          a = Math.round(a * 1e5) / 1e5; b = Math.round(b * 1e5) / 1e5;
          stack.push([a, b]);
          if (pin.v < a || pin.v > b) pin.v = (a + b) / 2;
          await nl.zoomTo(a, b); busy = false; nl.render(); crumbs(); showPin(); check();
        }
        async function zoomBack(k) {
          if (busy || k >= stack.length - 1) return; busy = true;
          stack.length = k + 1; const [a, b] = stack[k];
          await nl.zoomTo(a, b); busy = false; nl.render(); crumbs(); showPin(); check();
        }
        el.querySelector('.out').onclick = () => zoomBack(stack.length - 2);
        const near = (x, y) => Math.abs(x - y) < R() / 150;
        const inRange = (a, b) => Math.abs(nl.a - a) < 1e-6 && Math.abs(nl.b - b) < 1e-6;
        const mis = Missions(el.querySelector('.mhost'), [
          { t: `גרור את הסיכה ל-${M('0.7')}`, ok: () => near(pin.v, 0.7) },
          { t: `עשה זום לקטע שבין ${M('0.3')} ל-${M('0.4')}`, ok: () => inRange(0.3, 0.4) },
          { t: `שים את הסיכה על ${M('0.37')}`, ok: () => inRange(0.3, 0.4) && near(pin.v, 0.37) },
          { t: `עשה עוד זום, לקטע שבין ${M('0.37')} ל-${M('0.38')}`, ok: () => inRange(0.37, 0.38) },
          { t: `שים את הסיכה על ${M('0.375')}. זה באמצע, בין ${M('0.37')} ל-${M('0.38')}`, ok: () => inRange(0.37, 0.38) && near(pin.v, 0.375) },
          { t: 'חזור לציר המלא, ותראה איפה נמצאת עכשיו הסיכה', ok: () => stack.length === 1 },
        ], () => { el.querySelector('.after').hidden = false; done(); });
        function check() { mis.check(); }
        crumbs(); showPin();
      },
    },
    {
      t: 'משחק: פגע במטרה',
      r(el, done) {
        el.innerHTML = `
          <p>אני אומר מספר, ואתה שם את הסיכה בדיוק במקום שלו. אין זום הפעם, רק עין טובה. אחרי כל ניסיון נעשה זום ונראה כמה קרוב היית.</p>
          <div class="panel ro-row"><span class="ro-lbl">המטרה:</span><span class="bignum tgt" style="font-size:2.2rem"></span><span class="muted rnd"></span><span class="score" style="margin-inline-start:auto;font-weight:700"></span></div>
          <div class="nl-host" style="margin-top:8px"></div>
          <div class="controls"><button class="btn primary here">📍 כאן!</button><span class="res"></span></div>`;
        const targets = shuffle(['0.8', '0.25', '0.63', '0.09', '0.9', '0.47']).concat(['1.35']);
        let r = 0, pts = 0, nl, pin;
        const host = el.querySelector('.nl-host');
        function round() {
          const tg = targets[r], big = N.parse(tg) > U;
          host.innerHTML = ''; const svg = S.svg(LW(), 170, 'board framed'); host.appendChild(svg);
          nl = new NumberLine(svg, { W: LW(), y: 105, a: 0, b: big ? 2 : 1, dense: big });
          pin = { v: big ? 1 : 0.5, drag: true, snap: 0.001 }; nl.setPins([pin]);
          el.querySelector('.tgt').innerHTML = digitsHTML(tg);
          el.querySelector('.rnd').textContent = `סיבוב ${r + 1} מתוך ${targets.length}`;
          el.querySelector('.score').textContent = `ניקוד: ${pts}`;
          el.querySelector('.res').innerHTML = ''; el.querySelector('.here').disabled = false; el.querySelector('.here').textContent = '📍 כאן!';
          el.querySelector('.here').onclick = shoot;
        }
        async function shoot() {
          const btn = el.querySelector('.here'); btn.disabled = true;
          const tg = N.num(N.parse(targets[r])), e = Math.abs(pin.v - tg);
          const [p, word] = e <= 0.012 ? [3, '🎯 בול!'] : e <= 0.03 ? [2, 'קרוב מאוד!'] : e <= 0.08 ? [1, 'קרוב'] : [0, 'רחוק'];
          pts += p; pin.drag = false; pin.label = 'אתה';
          const tpin = { v: tg, cls: 'target', label: targets[r], up: false };
          nl.setPins([pin, tpin]);
          const lo = Math.floor(tg * 10 + 1e-9) / 10;
          const tip = e > 0.08 && Math.abs(tg - 0.09) < 1e-9 ? ` ${M('0.09')} זה פחות מעשירית אחת! הוא ממש קרוב ל-0.` : e > 0.08 && Math.abs(tg - 0.9) < 1e-9 ? ` ${M('0.9')} זה 9 עשיריות, כמעט 1.` : '';
          el.querySelector('.res').innerHTML = `<b>${word}</b> +${p} · ${M(targets[r])} גר בין ${M(N.str(N.of(Math.round(lo * 10), 1)))} ל-${M(N.str(N.of(Math.round(lo * 10) + 1, 1)))}.${tip}`;
          el.querySelector('.score').textContent = `ניקוד: ${pts}`;
          await sleep(700);
          const a = Math.min(lo, pin.v) - 0.02, b = Math.max(lo + 0.1, pin.v) + 0.02;
          await nl.zoomTo(Math.max(0, a), b, 900);
          const last = r === targets.length - 1;
          btn.disabled = false; btn.textContent = last ? 'לסיכום ←' : 'לסיבוב הבא ←';
          btn.onclick = () => { if (last) finish(); else { r++; round(); } };
        }
        function finish() {
          const max = targets.length * 3;
          if (pts >= max * .7) confetti();
          el.querySelector('.panel').hidden = true; host.innerHTML = '';
          el.querySelector('.controls').innerHTML = `<div class="qz-end" style="width:100%"><div class="stars">${pts >= max * .75 ? '⭐⭐⭐' : pts >= max * .45 ? '⭐⭐' : '⭐'}</div>
            <p><b>${pts} נקודות</b> מתוך ${max}.</p><p class="muted">הטריק: קודם מוצאים באיזו עשירית המספר גר (הספרה הראשונה אחרי הנקודה), ורק אז מדייקים בתוכה.</p>
            <button class="btn again">עוד משחק 🔁</button></div>`;
          el.querySelector('.again').onclick = () => { r = 0; pts = 0; el.querySelector('.panel').hidden = false; el.querySelector('.controls').innerHTML = `<button class="btn primary here">📍 כאן!</button><span class="res"></span>`; shuffleT(); round(); };
          done();
        }
        function shuffleT() { const last = targets.pop(); const s = shuffle(targets); targets.length = 0; targets.push(...s, last); }
        round();
      },
    },
    {
      t: 'מה יש בין לבין?',
      r(el, done) {
        el.innerHTML = `<p>בין 3 ל-4 אין אף מספר שלם. אבל מספרים עשרוניים? תמיד אפשר למצוא עוד אחד באמצע. תנסה.</p><div class="qhost"></div>`;
        const pairs = [['0.4', '0.5'], ['0.7', '0.8'], ['2.3', '2.4'], ['0.45', '0.46'], ['0.9', '1'], ['0.1', '0.11']];
        const mid = { '0.4': '0.45', '0.7': '0.75', '2.3': '2.35', '0.45': '0.455', '0.9': '0.95', '0.1': '0.105' };
        Quiz(el.querySelector('.qhost'), {
          rounds: pairs.length, onDone: done,
          gen(i) {
            const [a, b] = pairs[i], va = N.parse(a), vb = N.parse(b);
            let L;
            const lineFig = f => {
              const svg = S.svg(LW(), 150, 'board framed'); f.appendChild(svg);
              const w = vb - va, span = w * 10 >= U ? 1 : w * 10 / U;
              const lo = Math.floor(N.num(va) / span) * span;
              L = new NumberLine(svg, { W: LW(), y: 80, a: lo, b: lo + span, pad: 40 });
              L.setPins([{ v: N.num(va), cls: 'b', label: a, up: false }, { v: N.num(vb), cls: 'b', label: b, up: false }]);
            };
            return {
              q: `כתוב מספר שנמצא בין ${M(a)} ל-${M(b)}.`, type: 'input', fig: lineFig, show: mid[a],
              accept: (s, v) => v !== null && v > va && v < vb,
              onTry: async (v, ok) => {
                if (v === null || !L) return;
                L.setPins([{ v: N.num(va), cls: 'b', label: a, up: false }, { v: N.num(vb), cls: 'b', label: b, up: false }, { v: N.num(v), cls: ok ? 'target' : '', label: N.str(v) }]);
                if (ok) { const w = N.num(vb - va); await L.zoomTo(N.num(va) - w * .15, N.num(vb) + w * .15, 900); }
              },
              hint: `עשה זום בראש: אם תחלק את הקטע בין ${M(a)} ל-${M(b)} ל-10 חלקים, מה יהיו השנתות? ${i === 3 || i === 5 ? 'אפשר להוסיף ספרה נוספת בסוף, כמו אלפיות.' : ''}`,
              explain: i === 0 ? 'ראית? בזום הקטע נפתח, ויש בו עוד הרבה מספרים.' : i === 3 ? `${M('0.45 = 0.450')} ו-${M('0.46 = 0.460')}, ובין 450 ל-460 אלפיות יש מקום ל-${M('0.451, 0.452, ...')}` : i === 4 ? `${M('1 = 1.0')}. בין ${M('0.9')} ל-${M('1.0')} יש ${M('0.91, 0.92, ..., 0.99')}.` : `תמיד יש עוד מספר באמצע. תמיד.`,
            };
          },
        });
      },
    },
    {
      t: 'מה לקחת מהפרק', auto: true,
      r(el) {
        el.innerHTML = `<div class="learned"><ul>
          <li>על הציר, בין כל שני שלמים יש 10 עשיריות. בין כל שתי עשיריות יש 10 מאיות. בין כל שתי מאיות יש 10 אלפיות.</li>
          <li>כדי למצוא מספר על הציר: קודם השלמים, אחר כך באיזו עשירית, ורק אז מדייקים. ${M('0.63')} נמצא בין ${M('0.6')} ל-${M('0.7')}, קצת לפני האמצע.</li>
          <li>${M('0.09')} קרוב מאוד ל-0, ו-${M('0.9')} קרוב מאוד ל-1. הם רחוקים זה מזה!</li>
          <li>בין כל שני מספרים עשרוניים יש עוד מספרים. טריק: להוסיף 0 בסוף ואז לבחור באמצע. ${M('0.45 = 0.450')}, ${M('0.46 = 0.460')}, אז ${M('0.455')} באמצע.</li>
        </ul></div>`;
      },
    },
  ],
});
