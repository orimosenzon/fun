'use strict';
/* פרק 3: בית הספרות. לוח מקומות עם קוביות שמתאחדות ומתפרטות, אלפיות, המראה סביב השלמים, וקריאת מספרים. */

// לוח מקומות עם קוביות: שלמים | עשיריות | מאיות
function BlockBoard(el, opt = {}) {
  const cnt = opt.start ? opt.start.slice() : [0, 0, 0];
  const names = ['שלמים', 'עשיריות', 'מאיות'], keys = ['p0', 'pm1', 'pm2'];
  const CW = 200, s = 96;
  el.innerHTML = `<div class="bb-svg"></div>
    <div class="bb-btns" style="display:grid;grid-template-columns:repeat(3,1fr);direction:ltr;text-align:center;gap:4px;max-width:620px;margin:0 auto">
      ${[0, 1, 2].map(c => `<div class="controls" style="justify-content:center;margin:4px 0"><button class="btn round" data-c="${c}" data-d="-1">−</button><button class="btn round" data-c="${c}" data-d="1">+</button></div>`).join('')}
    </div><div class="bb-msg tip" hidden></div>`;
  const svg = S.svg(600, 330, 'board framed'); svg.style.maxWidth = '620px'; svg.style.margin = '0 auto';
  el.querySelector('.bb-svg').appendChild(svg);
  const msg = el.querySelector('.bb-msg');
  let busy = false, fresh = -1;

  function pos(c, j) {
    const x0 = c * CW;
    if (c === 0) return [x0 + 40 + j * 7, 44 + j * 7];
    if (c === 1) return [x0 + 16 + j * 17, 60];
    return [x0 + 52 + (j % 5) * 20, 90 + Math.floor(j / 5) * 22];
  }
  function drawBlock(g, c, x, y, cls) { return [blockFlat, blockRod, blockCube][c](g, x, y, s, cls); }

  function draw() {
    S.clear(svg);
    for (let c = 0; c < 3; c++) {
      S.el('rect', { x: c * CW + 4, y: 4, width: CW - 8, height: 322, rx: 12, class: 'col-bg' }, svg);
      S.text(svg, c * CW + CW / 2, 30, names[c], { class: 'col-head', 'text-anchor': 'middle', fill: `var(--${keys[c]})` });
      const g = S.el('g', { 'data-c': c }, svg);
      for (let j = 0; j < cnt[c]; j++) { const [x, y] = pos(c, j); drawBlock(g, c, x, y, fresh === c && j === cnt[c] - 1 ? 'new' : ''); }
      S.text(svg, c * CW + CW / 2, 305, String(Math.min(cnt[c], 99)), { class: 'col-digit', 'text-anchor': 'middle', fill: cnt[c] >= 10 ? 'var(--red)' : `var(--${keys[c]})` });
    }
    S.text(svg, CW, 305, '.', { class: 'col-digit', 'text-anchor': 'middle' });
    el.querySelectorAll('.bb-btns button').forEach(b => {
      const c = +b.dataset.c, d = +b.dataset.d;
      b.disabled = busy || (d > 0 ? (c === 0 ? cnt[0] >= 9 : value() + [100, 10, 1][c] > 999) : (cnt[c] === 0 && !cnt.slice(0, c).some(x => x > 0)));
    });
    opt.onChange && opt.onChange(value(), cnt.slice(), busy);
  }
  const value = () => cnt[0] * 100 + cnt[1] * 10 + cnt[2];
  const say = h => { msg.hidden = false; msg.innerHTML = h; msg.classList.remove('flash'); void msg.offsetWidth; msg.classList.add('flash'); };

  async function exchangeUp(c) {
    // 10 חתיכות בעמודה c מתאחדות לחתיכה אחת בעמודה c-1
    say(`יש 10 ${names[c]}! 10 ${names[c]} הן בדיוק ${c === 1 ? 'שלם אחד' : 'עשירית אחת'}, אז הן מתאחדות ועוברות עמודה שמאלה.`);
    await sleep(900);
    const g = svg.querySelector(`g[data-c="${c}"]`);
    const [tx, ty] = pos(c - 1, cnt[c - 1]), [fx, fy] = pos(c, 0);
    g.classList.add('blk', 'moving'); g.style.transform = `translate(${tx - fx}px, ${ty - fy}px)`; g.style.opacity = '.3';
    await sleep(750);
    cnt[c] = 0; cnt[c - 1]++; fresh = c - 1; draw();
    await sleep(500); fresh = -1;
  }
  async function breakDown(c) {
    // חתיכה אחת מהעמודה c-1 מתפרטת ל-10 בעמודה c
    if (cnt[c - 1] === 0) await breakDown(c - 1);
    say(`אין ${names[c]} להוריד. אז <b>פורטים</b>: ${c === 1 ? 'שלם אחד מתפרק ל-10 עשיריות' : 'עשירית אחת מתפרקת ל-10 מאיות'}. הערך לא משתנה, רק הצורה.`);
    await sleep(900);
    cnt[c - 1]--; cnt[c] += 10; draw();
    await sleep(900);
  }
  async function change(c, d) {
    if (busy) return; busy = true; msg.hidden = true;
    if (d > 0) { cnt[c]++; fresh = c; draw(); fresh = -1; await sleep(250); if (cnt[c] >= 10 && c > 0) { await exchangeUp(c); if (c === 2 && cnt[1] >= 10) await exchangeUp(1); } }
    else { if (cnt[c] === 0) await breakDown(c); cnt[c]--; draw(); }
    busy = false; draw();
    opt.onSettle && opt.onSettle(value(), cnt.slice());
  }
  el.querySelectorAll('.bb-btns button').forEach(b => b.onclick = () => change(+b.dataset.c, +b.dataset.d));
  draw();
  return { value, set(a) { cnt[0] = a[0]; cnt[1] = a[1]; cnt[2] = a[2]; draw(); } };
}

App.add({
  id: 'ch3', short: 'בית הספרות', icon: '🏠', title: 'בית הספרות: לכל ספרה יש מקום',
  desc: 'לוח המקומות, קוביות שמתאחדות, אלפיות וקריאה',
  intro: `<p>כבר ראית את הדפוס: שלם מתחלק ל-10 עשיריות, עשירית מתחלקת ל-10 מאיות. עכשיו נסדר את כל זה ב"בית" אחד, שבו לכל ספרה יש חדר משלה.</p>`,
  steps: [
    {
      t: 'לוח המקומות עם קוביות',
      r(el, done) {
        el.innerHTML = `
          <p>יש לנו שלושה סוגי חתיכות, וכל אחת גדולה פי 10 מזו שמימינה:</p>
          <div class="legend"><span><b class="t-p0">■ ריבוע</b> = שלם</span><span><b class="t-pm1">▮ מוט</b> = עשירית (10 מוטות = ריבוע)</span><span><b class="t-pm2">▪ קובייה</b> = מאית (10 קוביות = מוט)</span></div>
          <p class="do">הוסף והורד חתיכות עם + ו־−. בצע את המשימות, ותסתכל טוב מה קורה כשיש 10 מאותו סוג.</p>
          <div class="bb"></div>
          <div class="mhost"></div>
          <div class="after" hidden><div class="aha"><b>זה כל הסוד של השברים העשרוניים:</b> כל מקום שווה פי 10 מהמקום שמימינו. ב<b>כל</b> עמודה יכולות להיות לכל היותר 9 חתיכות. ברגע שיש 10, הן מתאחדות לחתיכה אחת בעמודה משמאל. וכשחסר, פורטים חתיכה מהעמודה משמאל ל-10 קטנות.
            <br>אותו דבר בדיוק קורה במספרים רגילים (10 אחדות = עשרת). בפרק 6 זה יהיה "להעביר 1" בחיבור ו"לפרוט" בחיסור.</div></div>`;
        const mis = Missions(el.querySelector('.mhost'), [
          { t: `בנה את המספר ${M('1.23')}`, ok: v => v === 123 },
          { t: `הוסף מאיות, אחת אחרי השנייה, עד שתגיע ל-${M('1.30')}`, ok: v => v === 130 },
          { t: 'עכשיו הורד מאית אחת. אבל רגע, אין מאיות! מה יקרה?', ok: v => v === 129 },
          { t: `בנה ${M('2.05')}`, ok: v => v === 205, after: () => { const d = document.createElement('div'); d.className = 'tip'; d.innerHTML = `בעמודת העשיריות אין כלום, ולכן כותבים שם 0. בלי ה-0 היה יוצא ${M('2.5')}, וזה מספר אחר לגמרי.`; el.querySelector('.mhost').appendChild(d); } },
        ], () => { el.querySelector('.after').hidden = false; done(); });
        BlockBoard(el.querySelector('.bb'), { onSettle: v => mis.check(v) });
      },
    },
    {
      t: 'כמה שווה כל ספרה?',
      r(el, done) {
        el.innerHTML = `<p>אותה ספרה יכולה להיות שווה המון או מעט, לפי <b>המקום</b> שבו היא יושבת. ה-3 ב-${M('3')} שווה פי 100 מה-3 ב-${M('0.03')}.</p><div class="qhost"></div>`;
        const kinds = shuffle(['click', 'click', 'click', 'value', 'value', 'value', 'expand', 'expand']);
        const plName = { 1: 'העשרות', 0: 'השלמים', '-1': 'העשיריות', '-2': 'המאיות', '-3': 'האלפיות' };
        function randNum() {
          const ds = shuffle([1, 2, 3, 4, 5, 6, 7, 8, 9]);
          const ni = ri(1, 2), nf = ri(2, 3);
          return { ip: ds.slice(0, ni).join(''), fp: ds.slice(ni, ni + nf).join('') };
        }
        Quiz(el.querySelector('.qhost'), {
          rounds: 8, onDone: done,
          gen(i) {
            const k = kinds[i];
            if (k === 'click') {
              const { ip, fp } = randNum(), str = ip + '.' + fp;
              const places = [...ip].map((_, j) => ip.length - 1 - j).concat([...fp].map((_, j) => -(j + 1)));
              const tp = pick(places.filter(p => p !== 1 || ip.length > 1)), digit = tp >= 0 ? ip[ip.length - 1 - tp] : fp[-tp - 1];
              const byName = Math.random() < .5;
              let wrongs = 0;
              return {
                q: byName ? `לחץ על הספרה שנמצאת במקום של <b>${plName[tp]}</b>.` : `לחץ על הספרה ששווה <b>${HEB.count(+digit, PLACE_ONE[tp], PLACE_NAME[tp], tp !== 0)}</b>.`,
                type: 'custom', maxTries: 2,
                mount(ans, submit) {
                  ans.innerHTML = `<div class="bignum" dir="ltr" style="text-align:center">${[...str].map((ch, j) => ch === '.' ? '<span class="dot">.</span>' : `<button class="opt dgbtn" data-p="${places[j < ip.length ? j : j - 1]}" style="font-size:2.4rem;padding:2px 14px;margin:2px">${ch}</button>`).join('')}</div>`;
                  ans.querySelectorAll('.dgbtn').forEach(b => b.onclick = () => {
                    const ok = +b.dataset.p === tp;
                    b.classList.add(ok ? 'right' : 'wrong');
                    if (ok) ans.querySelectorAll('.dgbtn').forEach(x => x.disabled = true); else { b.disabled = true; wrongs++; }
                    submit(ok);
                  });
                  this.reveal = () => { ans.querySelectorAll('.dgbtn').forEach(x => { x.disabled = true; if (+x.dataset.p === tp) x.classList.add('right'); }); };
                },
                hint: 'המקום הראשון מימין לנקודה הוא עשיריות, השני מאיות, השלישי אלפיות. משמאל לנקודה: שלמים, ואז עשרות.',
                explain: `במספר ${digitsHTML(str)} הספרה ${M(digit)} נמצאת במקום של ${plName[tp]}.`,
              };
            }
            if (k === 'value') {
              const { ip, fp } = randNum(), str = ip + '.' + fp;
              const j = ri(0, fp.length - 1), d = fp[j], p = j + 1;
              const val = p => N.str(N.of(+d, p));
              return choiceQ(`כמה שווה הספרה ${M(d)} במספר ${digitsHTML(str)}?`, val(p), [val(0), val(p === 1 ? 2 : 1), val(p === 3 ? 2 : 3), String(+d * 10)], {
                fmt: M, hint: 'ספור באיזה מקום היא אחרי הנקודה.',
                explain: `היא ${p === 1 ? 'ראשונה' : p === 2 ? 'שנייה' : 'שלישית'} אחרי הנקודה, כלומר ${HEB.count(+d, PLACE_ONE[-p], PLACE_NAME[-p])}: ${M(val(p))}.` });
            }
            const w = ri(1, 9), a = ri(1, 9), b = ri(1, 9), c = pick([0, ri(1, 9)]);
            const parts = shuffle([String(w), '0.' + a, '0.0' + b].concat(c ? ['0.00' + c] : []));
            const ans = N.of(w * 1000 + a * 100 + b * 10 + c, 3);
            return { q: `כמה זה ${M(parts.join(' + '))}?`, type: 'input', answer: ans,
              hint: 'שים כל חלק במקום שלו: שלמים, עשיריות, מאיות, אלפיות. הסדר שבו הם כתובים לא משנה.',
              explain: `${M(parts.join(' + ') + ' = ' + N.str(ans))}. כל חלק נכנס לחדר שלו בבית.` };
          },
        });
      },
    },
    {
      t: 'אלפיות, והמראה',
      r(el, done) {
        el.innerHTML = `
          <div class="play">
            <div class="zoomfig"></div>
            <div>
              <p>ואפשר להמשיך לחתוך! כל מאית (קובייה קטנה) מתחלקת ל-10 פרוסות דקות. כל פרוסה היא <b class="t-pm3">אלפית</b>, כי יש 1000 כאלה בשלם.</p>
              <p class="def">${M(`${M('0.001')} = ${F(1, 1000)}`)} &nbsp; היא הספרה השלישית אחרי הנקודה.</p>
              <p>אלפיות מופיעות הרבה במדידות:</p>
              <ul>
                <li>בקילוגרם יש 1000 גרם. אז ${M('0.250')} ק״ג גבינה = 250 גרם.</li>
                <li>במטר יש 1000 מילימטרים. אז ${M('1.005')} מטר = מטר ו-5 מ״מ.</li>
                <li>בליטר יש 1000 מיליליטר. בקבוק של ${M('0.5')} ליטר = 500 מ״ל.</li>
              </ul>
            </div>
          </div>
          <h4>🪞 המראה של בית הספרות</h4>
          <p class="do">לחץ על שמות המקומות בטבלה. מה תמונת המראה של כל מקום?</p>
          <table class="pv mirror-t"><tr>
            <th class="p2 clickable" data-p="2">מאות</th><th class="p1 clickable" data-p="1">עשרות</th><th class="p0 clickable" data-p="0">שלמים</th><th class="dotc"></th>
            <th class="pm1 clickable" data-p="-1">עשיריות</th><th class="pm2 clickable" data-p="-2">מאיות</th><th class="pm3 clickable" data-p="-3">אלפיות</th></tr>
            <tr><td class="p2">100</td><td class="p1">10</td><td class="p0">1</td><td class="dotc">.</td><td class="pm1" style="font-size:1.1rem">${F(1, 10)}</td><td class="pm2" style="font-size:1.1rem">${F(1, 100)}</td><td class="pm3" style="font-size:1.1rem">${F(1, 1000)}</td></tr></table>
          <div class="mir-msg tip">לחץ על שם של מקום.</div>
          <div class="after" hidden><div class="aha"><b>המרכז של הבית הוא השלמים, לא הנקודה.</b> עשרות ועשיריות הן תמונת מראה (10 ו-${F(1, 10)}), מאות ומאיות הן תמונת מראה (100 ו-${F(1, 100)}). לכן אין "אחדיות": מימין לשלמים באות מיד העשיריות. הנקודה היא רק שלט שאומר "פה נגמרים השלמים".</div></div>`;
        // ציור זום: קובייה אחת מתוך ריבוע המאה הופכת ל-10 פרוסות
        const svg = S.svg(330, 230, 'board'); svg.style.maxWidth = '360px';
        const cells = cellsOf(0); cells[0] = true;
        drawGrid(svg, 10, 30, 170, cells, { byRole: true });
        const cw = 17;
        S.el('rect', { x: 10, y: 30 + 9 * cw, width: cw, height: cw, fill: 'none', stroke: '#db2777', 'stroke-width': 3 }, svg);
        S.el('line', { x1: 10 + cw, y1: 30 + 9 * cw, x2: 215, y2: 60, stroke: '#db2777', 'stroke-width': 1.5, 'stroke-dasharray': '4 3' }, svg);
        S.el('line', { x1: 10 + cw, y1: 30 + 10 * cw, x2: 215, y2: 200, stroke: '#db2777', 'stroke-width': 1.5, 'stroke-dasharray': '4 3' }, svg);
        S.el('rect', { x: 215, y: 60, width: 140, height: 140, fill: '#fdba74', stroke: '#ea580c', 'stroke-width': 2 }, svg);
        for (let k = 0; k < 10; k++) S.el('rect', { x: 215 + k * 14 + 1, y: 61, width: 12, height: 138, fill: k === 0 ? '#db2777' : 'rgba(255,255,255,.35)', stroke: 'none' }, svg);
        S.text(svg, 95, 20, 'שלם = 100 מאיות', { 'text-anchor': 'middle', class: 'lbl', 'font-size': 14, 'font-weight': 600 });
        S.text(svg, 285, 50, 'מאית אחת = 10 אלפיות', { 'text-anchor': 'middle', 'font-size': 14, 'font-weight': 600, fill: '#db2777', direction: 'rtl' });
        svg.setAttribute('viewBox', '0 0 370 230');
        el.querySelector('.zoomfig').appendChild(svg);
        const texts = {
          2: `<b>מאות</b> = 100 שלמים. תמונת המראה: <b>מאיות</b> = שלם חלקי 100.`,
          1: `<b>עשרות</b> = 10 שלמים. תמונת המראה: <b>עשיריות</b> = שלם חלקי 10.`,
          0: `<b>השלמים הם מרכז המראה.</b> הם עומדים לבד, ואין להם תמונת מראה. לכן אין מקום של "אחדיות".`,
          '-1': `<b>עשיריות</b> = שלם חלקי 10. תמונת המראה: <b>עשרות</b> = 10 שלמים.`,
          '-2': `<b>מאיות</b> = שלם חלקי 100. תמונת המראה: <b>מאות</b> = 100 שלמים.`,
          '-3': `<b>אלפיות</b> = שלם חלקי 1000. תמונת המראה: <b>אלפים</b> = 1000 שלמים (לא נכנסו לטבלה).`,
        };
        const seen = new Set();
        el.querySelectorAll('.mirror-t th[data-p]').forEach(th => th.onclick = () => {
          const p = +th.dataset.p;
          el.querySelectorAll('.mirror-t th').forEach(x => x.classList.toggle('mirror', x === th || +x.dataset.p === -p && x.dataset.p !== undefined));
          el.querySelector('.mir-msg').innerHTML = texts[p]; seen.add(p);
          if (seen.size >= 3) { el.querySelector('.after').hidden = false; done(); }
        });
      },
    },
    {
      t: 'איך קוראים את זה?',
      r(el, done) {
        el.innerHTML = `
          <p>ביום-יום אומרים "שתיים נקודה שלוש חמש", וזה בסדר. אבל יש דרך קריאה שאומרת <b>מה</b> המספר באמת:</p>
          <p class="def">${M('2.35')} = "שני שלמים ושלושים וחמש <b class="t-pm2">מאיות</b>"<br>
          <span class="small">הטריק: קוראים את מה שאחרי הנקודה כמספר רגיל, ונותנים לו את השם של <b>המקום האחרון</b>. ב-${M('2.35')} הספרה האחרונה (5) במקום המאיות, אז "35 מאיות".</span></p>
          <div class="qhost"></div>`;
        const pool = shuffle(['r', 'r', 'r', 'w', 'w', 'w', 'w', 'r']);
        Quiz(el.querySelector('.qhost'), {
          rounds: 8, onDone: done,
          gen(i) {
            const p = ri(1, 3), whole = pick([0, 0, ri(1, 5)]);
            let f = p === 1 ? ri(1, 9) : p === 2 ? pick([ri(1, 9), ri(11, 99)]) : pick([ri(1, 9), ri(11, 99), ri(101, 999)]);
            if (f % 10 === 0) f++;
            const v = N.of(whole * 10 ** p + f, p), str = N.str(v, p);
            if (pool[i] === 'w') {
              return { q: `כתוב במספרים: <b>${HEB.read(v)}</b>`, type: 'input', answer: v,
                hint: `כמה ספרות צריך אחרי הנקודה בשביל ${['', 'עשיריות', 'מאיות', 'אלפיות'][p]}? ${p > 1 ? 'אם חסרות ספרות, אפסים שומרים מקום.' : ''}`,
                explain: `${digitsHTML(str)}` };
            }
            const q2 = p === 3 ? 2 : p + 1;
            const wrong1 = HEB.read(N.of(whole * 10 ** q2 + f, q2));
            const wrong2 = whole ? HEB.read(N.of(f * 10 ** p + whole, p)) : HEB.read(N.of(f, Math.max(1, p - 1)));
            return choiceQ(`איך קוראים את ${digitsHTML(str)}?`, HEB.read(v), [wrong1, wrong2], {
              hint: 'מה המקום של הספרה האחרונה אחרי הנקודה?',
              explain: `הספרה האחרונה נמצאת במקום ה${['', 'עשיריות', 'מאיות', 'אלפיות'][p]}.` });
          },
        });
      },
    },
    {
      t: 'מה לקחת מהפרק', auto: true,
      r(el) {
        el.innerHTML = `<div class="learned"><ul>
          <li>כל מקום שווה <b>פי 10</b> מהמקום שמימינו: ${M('1 = 10 × 0.1')}, ${M('0.1 = 10 × 0.01')}, ${M('0.01 = 10 × 0.001')}.</li>
          <li>בכל מקום אפשר לשים רק ספרה אחת, 0 עד 9. 10 חתיכות מתאחדות לחתיכה אחת משמאל. חסר? פורטים חתיכה משמאל ל-10.</li>
          <li>שמות המקומות אחרי הנקודה: <b class="t-pm1">עשיריות</b>, <b class="t-pm2">מאיות</b>, <b class="t-pm3">אלפיות</b>.</li>
          <li>המרכז של הבית הוא השלמים. עשרות ↔ עשיריות, מאות ↔ מאיות.</li>
          <li>קוראים את מה שאחרי הנקודה עם השם של המקום האחרון: ${M('0.042')} = ארבעים ושתיים אלפיות.</li>
          <li>אפשר לפרק מספר לחלקים: ${M('3.47 = 3 + 0.4 + 0.07')}.</li>
        </ul></div>`;
      },
    },
  ],
});
