'use strict';
/* פרק 7: כפל וחילוק ב-10, 100, 1000. הנקודה עומדת, הספרות קופצות. ומדידות: מטר, סנטימטר, גרם. */

// לוח קפיצות: ספרות על לוח מקומות, זזות בהנפשה
function ShiftBoard(el, start, onChange) {
  const PL = [3, 2, 1, 0, -1, -2, -3];
  const CW = Math.min(58, Math.floor(((el.clientWidth || 600) - 24) / PL.length)), FS = Math.round(CW * .66);
  el.innerHTML = `<div class="sh-wrap" style="overflow-x:auto"><div class="sh" style="position:relative;direction:ltr;margin:0 auto;width:${PL.length * CW + 20}px;height:120px">
    ${PL.map((p, j) => `<div style="position:absolute;top:0;left:${j * CW + (p < 0 ? 20 : 0)}px;width:${CW - 4}px;text-align:center;font-size:${CW < 50 ? .6 : .72}rem;color:#fff;overflow:hidden;white-space:nowrap;border-radius:6px;background:var(--${PLACE_CLS[p]})">${PLACE_NAME[p]}</div>
      <div style="position:absolute;top:26px;left:${j * CW + (p < 0 ? 20 : 0)}px;width:${CW - 4}px;height:64px;border:2px solid var(--line);border-radius:10px;background:#fff"></div>`).join('')}
    <div style="position:absolute;top:34px;left:${4 * CW - 2}px;font-size:${FS + 4}px;font-weight:800">.</div>
    <div class="sh-digits"></div></div></div>`;
  const box = el.querySelector('.sh-digits');
  const X = p => { const j = PL.indexOf(p); return j * CW + (p < 0 ? 20 : 0); };
  let digs = [];
  function load(str) {
    box.innerHTML = ''; digs = [];
    const [i, f = ''] = str.split('.');
    [...i].forEach((d, k) => digs.push({ d, p: i.length - 1 - k }));
    [...f].forEach((d, k) => digs.push({ d, p: -(k + 1) }));
    // בלי אפסים מובילים/סוגרים: הם יתווספו לפי הצורך
    while (digs.length > 1 && digs[0].d === '0' && digs[0].p > 0) digs.shift();
    digs.forEach(g => { g.el = document.createElement('div'); g.el.className = 'sh-d'; g.el.textContent = g.d; box.appendChild(g.el); });
    place(false); zeros();
  }
  function place(anim) {
    digs.forEach(g => {
      g.el.style.cssText = `position:absolute;top:30px;width:${CW - 4}px;text-align:center;font-size:${FS}px;font-weight:800;transition:${anim ? 'left .8s cubic-bezier(.5,0,.2,1), color .8s' : 'none'};left:${X(g.p)}px;color:var(--${PLACE_CLS[g.p]})`;
    });
  }
  function zeros() {
    box.querySelectorAll('.sh-z').forEach(z => z.remove());
    const ps = digs.map(g => g.p), hi = Math.max(0, ...ps), lo = Math.min(0, ...ps);
    for (let p = hi; p >= lo; p--) {
      if (ps.includes(p)) continue;
      if (p < Math.min(...ps) && p < 0) continue; // אפסים בסוף אחרי הנקודה לא צריך
      const z = document.createElement('div'); z.className = 'sh-z'; z.textContent = '0';
      z.style.cssText = `position:absolute;top:30px;left:${X(p)}px;width:${CW - 4}px;text-align:center;font-size:${FS}px;font-weight:800;color:var(--${PLACE_CLS[p]});opacity:.55;animation:pop .45s`;
      box.appendChild(z);
    }
  }
  const value = () => digs.reduce((s, g) => s + N.of(+g.d, -g.p), 0);
  const str = () => N.str(value());
  let busy = false;
  async function shift(s) {
    if (busy) return false;
    const ps = digs.map(g => g.p);
    if (Math.max(...ps) + s > 3 || Math.min(...ps) + s < -3) return false;
    busy = true;
    box.querySelectorAll('.sh-z').forEach(z => z.remove());
    digs.forEach(g => g.p += s); place(true);
    await sleep(850); zeros(); busy = false;
    onChange && onChange(str());
    return true;
  }
  const canShift = s => { const ps = digs.map(g => g.p); return Math.max(...ps) + s <= 3 && Math.min(...ps) + s >= -3; };
  load(start);
  return { shift, load, str, canShift, get busy() { return busy; } };
}

App.add({
  id: 'ch7', short: 'כפול 10', icon: '🚀', title: 'כפל וחילוק ב-10, 100, 1000',
  desc: 'הנקודה עומדת במקום, והספרות קופצות',
  intro: `<p>זוכר שכל מקום בבית הספרות שווה פי 10 מהמקום שמימינו? זה אומר שכפל ב-10 הוא הכי פשוט שיש: כל ספרה פשוט עוברת לגור מקום אחד שמאלה.</p>`,
  steps: [
    {
      t: 'מכונת הקפיצות',
      r(el, done) {
        el.innerHTML = `
          <p class="do">לחץ על הכפתורים ותסתכל טוב: מה זז? הספרות, או הנקודה?</p>
          <div class="sb"></div>
          <div class="controls" style="justify-content:center;direction:ltr;margin-bottom:0">${[1, 2, 3].map(k => `<button class="btn op" data-s="${k}">× ${10 ** k}</button>`).join('')}</div>
          <div class="controls" style="justify-content:center;direction:ltr;margin-top:6px">${[1, 2, 3].map(k => `<button class="btn op" data-s="${-k}">÷ ${10 ** k}</button>`).join('')}</div>
          <div class="controls" style="justify-content:center"><button class="btn sm rst">↺ חזרה ל-${M('3.47')}</button></div>
          <div class="panel hist" style="text-align:center;min-height:52px"></div>
          <div class="qx"></div>
          <div class="mhost"></div>
          <div class="after" hidden><div class="aha"><b>הנקודה לא זזה אף פעם.</b> הספרות זזות:
            <ul><li><b>כפל ב-10</b>: כל ספרה קופצת מקום אחד <b>שמאלה</b>, ונהיית שווה פי 10. ב-100: שני מקומות. ב-1000: שלושה.</li>
            <li><b>חילוק ב-10</b>: כל ספרה קופצת מקום אחד <b>ימינה</b>.</li>
            <li>מספר האפסים ב-10, 100, 1000 אומר כמה מקומות לקפוץ.</li>
            <li>כשנוצר מקום ריק בין הספרות לנקודה, ממלאים אותו ב-0 (החיוורים).</li></ul>
            בכתיבה נוח לחשוב "הנקודה זזה ימינה" בכפל. זה נותן אותה תוצאה, אבל עכשיו אתה יודע מה קורה באמת.</div></div>`;
        let prev = '3.47', asked = false, blocked = false;
        const B = ShiftBoard(el.querySelector('.sb'), '3.47', s => {
          el.querySelector('.hist').innerHTML = `<span class="bignum" style="font-size:1.8rem">${M(prev + ' ' + lastOp + ' = ')}${digitsHTML(s)}</span>`;
          prev = s; buttons(); mis.check(s);
        });
        let lastOp = '';
        function buttons() { el.querySelectorAll('.op').forEach(b => b.disabled = blocked || !B.canShift(+b.dataset.s)); }
        el.querySelectorAll('.op').forEach(b => b.onclick = () => { const s = +b.dataset.s; lastOp = (s > 0 ? '×' : '÷') + ' ' + 10 ** Math.abs(s); B.shift(s); el.querySelectorAll('.op').forEach(x => x.disabled = true); });
        el.querySelector('.rst').onclick = () => { B.load('3.47'); prev = '3.47'; el.querySelector('.hist').innerHTML = ''; buttons(); };
        const mis = Missions(el.querySelector('.mhost'), [
          { t: `לחץ על ${M('× 10')}. לאן זזו הספרות?`, ok: s => s === '34.7', after: trap },
          { t: `הגע ל-${M('3470')}`, ok: s => s === '3470' },
          { t: `חזור ל-${M('3.47')} (בלי הכפתור ↺)`, ok: s => s === '3.47' },
          { t: `הגע ל-${M('0.347')}`, ok: s => s === '0.347', after: () => note(`ב-${M('÷ 10')} כל ספרה עברה מקום אחד ימינה. ה-3 היה שלמים, ועכשיו הוא עשיריות.`) },
          { t: `ומ-${M('0.347')} הגע ל-${M('34.7')} בלחיצה אחת`, ok: s => s === '34.7' },
        ], () => { el.querySelector('.after').hidden = false; done(); });
        function note(h) { const d = document.createElement('div'); d.className = 'tip flash'; d.innerHTML = h; el.querySelector('.mhost').appendChild(d); }
        function trap() {
          if (asked) return; asked = true; blocked = true; buttons();
          const q = el.querySelector('.qx');
          q.innerHTML = `<div class="modal-q">רגע! חבר שלך אומר: "כשכופלים ב-10 פשוט מוסיפים 0 בסוף. אז ${M('3.47 × 10 = 3.470')}". מה דעתך?
            <div class="opts" style="margin-top:8px"><button class="opt" data-v="y">הוא צודק</button><button class="opt" data-v="n">הוא טועה</button></div><div class="fbq"></div></div>`;
          q.querySelectorAll('.opt').forEach(b => b.onclick = () => {
            q.querySelectorAll('.opt').forEach(x => x.disabled = true);
            const ok = b.dataset.v === 'n'; b.classList.add(ok ? 'right' : 'wrong'); if (!ok) q.querySelector('[data-v="n"]').classList.add('right');
            q.querySelector('.fbq').innerHTML = `<p>${ok ? '<b>נכון, הוא טועה.</b>' : '<b>זו טעות נפוצה מאוד.</b>'} ${M('3.470')} זה בדיוק ${M('3.47')}, כי אפס בסוף אחרי הנקודה לא משנה כלום (זוכר מפרק 2?). "להוסיף 0" עובד רק במספרים שלמים, כי שם ה-0 דוחף את כל הספרות שמאלה. הכלל האמיתי: <b>הספרות זזות מקום שמאלה</b>.</p>`;
            blocked = false; buttons();
          });
        }
        buttons();
      },
    },
    {
      t: 'משחק: קופצים',
      r(el, done) {
        el.innerHTML = `<p>אפשר לדמיין את לוח הקפיצות בראש, או לצייר על דף.</p><div class="qhost"></div>`;
        const kinds = shuffle(['mul', 'mul', 'div', 'div', 'which', 'which', 'mul', 'div']);
        const rnum = () => pick([`${ri(1, 9)}.${ri(1, 9)}${ri(1, 9)}`, `0.${ri(1, 9)}${ri(1, 9)}`, `${ri(11, 99)}.${ri(1, 9)}`, `0.0${ri(1, 9)}`, `${ri(2, 9)}`, `${ri(1, 9)}.${ri(1, 9)}`]);
        const ok = (v) => { const s = N.str(v); const [i, f = ''] = s.split('.'); return v > 0 && i.length <= 4 && f.length <= 4; };
        Quiz(el.querySelector('.qhost'), {
          rounds: 8, onDone: done,
          gen(i) {
            for (;;) {
              const n = rnum(), v = N.parse(n), k = ri(1, 3), f = 10 ** k;
              if (kinds[i] === 'mul' && ok(v * f)) return { q: `${M(`${n} × ${f} = ?`)}`, type: 'input', answer: v * f,
                hint: `כל ספרה קופצת ${k === 1 ? 'מקום אחד' : k + ' מקומות'} שמאלה.`, explain: `${M(`${n} × ${f} = ${N.str(v * f)}`)}: ${k === 1 ? 'מקום אחד' : k + ' מקומות'} שמאלה.` };
              if (kinds[i] === 'div' && ok(v / f) && Number.isInteger(v / f)) return { q: `${M(`${n} ÷ ${f} = ?`)}`, type: 'input', answer: v / f,
                hint: `כל ספרה קופצת ${k === 1 ? 'מקום אחד' : k + ' מקומות'} ימינה. אם צריך, ממלאים באפסים.`, explain: `${M(`${n} ÷ ${f} = ${N.str(v / f)}`)}: ${k === 1 ? 'מקום אחד' : k + ' מקומות'} ימינה.` };
              if (kinds[i] === 'which') {
                const up = Math.random() < .5, w = up ? v * f : v / f;
                if (!ok(w) || !Number.isInteger(w)) continue;
                const right = (up ? '× ' : '÷ ') + f;
                return choiceQ(`המכונה קיבלה ${digitsHTML(n)} והוציאה ${digitsHTML(N.str(w))}. מה היא עשתה?`, right, ['× 10', '× 100', '× 1000', '÷ 10', '÷ 100', '÷ 1000'].filter(x => x !== right).sort(() => Math.random() - .5).slice(0, 3),
                  { fmt: M, hint: 'המספר גדל או קטן? ובכמה מקומות זזו הספרות?', explain: `הספרות זזו ${k === 1 ? 'מקום אחד' : k + ' מקומות'} ${up ? 'שמאלה (המספר גדל)' : 'ימינה (המספר קטן)'}: ${M(right)}.` });
              }
            }
          },
        });
      },
    },
    {
      t: 'מודדים: סנטימטרים ומילימטרים',
      r(el, done) {
        el.innerHTML = `
          <p>בסרגל, כל סנטימטר מחולק ל-10 מילימטרים. אז מילימטר הוא <b>עשירית</b> סנטימטר! ${M('7.3')} ס״מ = 7 סנטימטרים ו-3 מילימטרים = 73 מ״מ.</p>
          <p class="do">גרור את הקצה של העיפרון (העיגול הוורוד) כדי לשנות את האורך שלו.</p>
          <div class="ruler-host"></div>
          <div class="panel ro-row"><span class="ro-lbl">אורך העיפרון</span><b class="rl-cm" style="font-size:1.4rem"></b><span class="muted">=</span><b class="rl-mm" style="font-size:1.4rem"></b></div>
          <div class="mhost"></div>
          <div class="after" hidden><div class="aha">מסנטימטרים למילימטרים: <b>כופלים ב-10</b> (${M('7.3 × 10 = 73')}). ממילימטרים לסנטימטרים: <b>מחלקים ב-10</b> (${M('45 ÷ 10 = 4.5')}). ובמטר יש 100 סנטימטרים, אז ${M('0.1')} מטר = 10 ס״מ.</div></div>`;
        const svg = S.svg(820, 150, 'board framed'); el.querySelector('.ruler-host').appendChild(svg);
        const x0 = 30, PX = 50; // 50 פיקסלים לסנטימטר
        const g = S.el('g', {}, svg);
        S.el('rect', { x: x0 - 10, y: 78, width: 15 * PX + 20, height: 62, rx: 6, fill: '#fef9c3', stroke: '#d6c46a' }, g);
        for (let mm = 0; mm <= 150; mm++) {
          const x = x0 + mm * PX / 10, big = mm % 10 === 0, mid = mm % 5 === 0;
          S.el('line', { x1: x, y1: 78, x2: x, y2: 78 + (big ? 24 : mid ? 16 : 9), stroke: '#57534e', 'stroke-width': big ? 1.6 : 1 }, g);
          if (big) S.text(g, x, 122, String(mm / 10), { 'text-anchor': 'middle', 'font-size': 15, 'font-weight': 600, fill: '#44403c' });
        }
        S.text(g, x0 + 15 * PX + 2, 134, 'ס״מ', { 'text-anchor': 'end', 'font-size': 12, fill: '#78716c', direction: 'rtl' });
        const pencil = S.el('g', {}, svg);
        let mm = 52;
        const mis = Missions(el.querySelector('.mhost'), [
          { t: `עשה עיפרון באורך ${M('7.3')} ס״מ`, ok: v => v === 73 },
          { t: 'עשה עיפרון באורך 45 מילימטרים', ok: v => v === 45, after: () => note(`45 מ״מ = ${M('4.5')} ס״מ. 45 עשיריות סנטימטר!`) },
          { t: `עשה עיפרון באורך ${M('0.1')} מטר (רמז: כמה סנטימטרים במטר?)`, ok: v => v === 100, after: () => note(`במטר יש 100 ס״מ, אז ${M('0.1')} מטר = עשירית מ-100 ס״מ = 10 ס״מ.`) },
          { t: `עשה עיפרון באורך ${M('12.05')} ס״מ... רגע, אפשר בכלל?`, ok: v => v === 120 || v === 121, after: () => note(`אי אפשר בדיוק! הסרגל מראה רק עשיריות ס״מ (מילימטרים). ${M('12.05')} ס״מ הוא בדיוק באמצע בין 12 ס״מ ל-${M('12.1')} ס״מ. כדי למדוד מאיות סנטימטר צריך כלי מדויק יותר.`) },
        ], () => { el.querySelector('.after').hidden = false; done(); });
        function note(h) { const d = document.createElement('div'); d.className = 'tip flash'; d.innerHTML = h; el.querySelector('.mhost').appendChild(d); }
        function draw() {
          S.clear(pencil);
          const L = mm * PX / 10, y = 40, tipLen = Math.min(26, L * .35);
          S.el('rect', { x: x0, y: y - 12, width: Math.min(14, L * .3), height: 24, fill: '#f472b6', rx: 3 }, pencil);
          S.el('rect', { x: x0 + Math.min(14, L * .3), y: y - 12, width: Math.max(0, L - tipLen - Math.min(14, L * .3)), height: 24, fill: '#facc15', stroke: '#ca8a04' }, pencil);
          S.el('path', { d: `M${x0 + L - tipLen} ${y - 12} L${x0 + L} ${y} L${x0 + L - tipLen} ${y + 12} Z`, fill: '#fde68a', stroke: '#ca8a04' }, pencil);
          S.el('path', { d: `M${x0 + L - tipLen * .3} ${y - 3.6} L${x0 + L} ${y} L${x0 + L - tipLen * .3} ${y + 3.6} Z`, fill: '#374151' }, pencil);
          S.el('line', { x1: x0 + L, y1: y, x2: x0 + L, y2: 78, stroke: '#db2777', 'stroke-dasharray': '3 3' }, pencil);
          const h = S.el('circle', { cx: x0 + L, cy: y, r: 13, class: 'handle', fill: '#fff', stroke: '#db2777', 'stroke-width': 3, style: 'cursor:grab' }, pencil);
          h.addEventListener('pointerdown', startDrag);
          el.querySelector('.rl-cm').innerHTML = `${digitsHTML(N.str(N.of(mm, 1)))} ס״מ`;
          el.querySelector('.rl-mm').innerHTML = `${M(String(mm))} מ״מ`;
        }
        function startDrag(e) {
          e.preventDefault(); svg.setPointerCapture(e.pointerId);
          const mv = ev => { mm = clamp(Math.round((S.pt(svg, ev).x - x0) / PX * 10), 5, 150); draw(); };
          const up = () => { svg.removeEventListener('pointermove', mv); svg.removeEventListener('pointerup', up); mis.check(mm); };
          svg.addEventListener('pointermove', mv); svg.addEventListener('pointerup', up);
        }
        draw();
      },
    },
    {
      t: 'משחק: המרת יחידות',
      r(el, done) {
        el.innerHTML = `
          <table class="pv" style="direction:rtl"><tr><th class="p0">1 מטר</th><th class="p0">1 ס״מ</th><th class="p0">1 ק״ג</th><th class="p0">1 ₪</th><th class="p0">1 ליטר</th></tr>
          <tr><td style="font-size:.95rem">100 ס״מ</td><td style="font-size:.95rem">10 מ״מ</td><td style="font-size:.95rem">1000 גרם</td><td style="font-size:.95rem">100 אגורות</td><td style="font-size:.95rem">1000 מ״ל</td></tr></table>
          <p class="tip">השאלה שתמיד עוזרת: <b>היחידה החדשה קטנה יותר?</b> אז יהיו ממנה <b>יותר</b>, וכופלים. גדולה יותר? יהיו פחות, ומחלקים.</p>
          <div class="qhost"></div>`;
        const U2 = [
          ['מטר', 'ס״מ', 100], ['ס״מ', 'מ״מ', 10], ['ק״ג', 'גרם', 1000], ['₪', 'אגורות', 100], ['ליטר', 'מ״ל', 1000], ['מטר', 'מ״מ', 1000], ['ק״מ', 'מטר', 1000],
        ];
        const kinds = shuffle([0, 1, 2, 3, 4, 5, 6, 0]).map((u, k) => [u, k % 2 === 0]);
        Quiz(el.querySelector('.qhost'), {
          rounds: 8, onDone: done,
          gen(i) {
            const [[big, small, f], down] = [U2[kinds[i][0]], kinds[i][1]];
                        if (down) {
              const n = pick([`${ri(1, 9)}.${ri(1, 9)}`, `0.${ri(1, 9)}${ri(0, 9)}`, `${ri(1, 4)}.${ri(0, 9)}${ri(1, 9)}`, `0.${ri(1, 9)}`]);
              const v = N.parse(n) * f;
              return { q: `${M(n)} ${big} = כמה ${small}?`, type: 'input', answer: v, unit: small,
                hint: `ב-${big} אחד יש ${f} ${small}. ${small} יותר קטן, אז יהיו יותר ממנו.`, explain: `${M(`${n} × ${f} = ${N.str(v)}`)} ${small}` };
            }
            const n = String(pick([ri(1, 9), ri(11, 99), ri(101, 999)]));
            const v = N.parse(n) / f;
            return { q: `${M(n)} ${small} = כמה ${big}?`, type: 'input', answer: v, unit: big,
              hint: `${f} ${small} הם ${big} אחד. ${big} יותר גדול, אז יהיו פחות ממנו: מחלקים ב-${f}.`, explain: `${M(`${n} ÷ ${f} = ${N.str(v)}`)} ${big}` };
          },
        });
      },
    },
    {
      t: 'מה לקחת מהפרק', auto: true,
      r(el) {
        el.innerHTML = `<div class="learned"><ul>
          <li>${M('× 10')}: כל ספרה זזה מקום אחד שמאלה. ${M('× 100')}: שניים. ${M('× 1000')}: שלושה. ${M('3.47 × 100 = 347')}.</li>
          <li>${M('÷ 10')}: כל ספרה זזה מקום אחד ימינה. ${M('3.47 ÷ 10 = 0.347')}.</li>
          <li>הנקודה עומדת. כשצריך, ממלאים מקומות ריקים ב-0: ${M('0.5 × 1000 = 500')}, ${M('4 ÷ 100 = 0.04')}.</li>
          <li>"להוסיף 0 בסוף" עובד רק במספרים שלמים. ${M('3.47 × 10')} הוא ${M('34.7')} ולא ${M('3.470')}.</li>
          <li>המרת יחידות: ליחידה קטנה יותר כופלים, ליחידה גדולה יותר מחלקים. ${M('2.5')} ק״ג = 2500 גרם.</li>
        </ul></div>`;
      },
    },
  ],
});
