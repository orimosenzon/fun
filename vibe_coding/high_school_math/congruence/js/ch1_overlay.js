'use strict';
/* פרק 1: להניח את המשולש הכתום על המשולש המקווקו */

App.inits.ch1 = function () {
  const root = document.getElementById('overlay-game');
  root.innerHTML = `
    <div class="levels" id="ov-levels"></div>
    <div class="msg" id="ov-msg"></div>
    <svg class="board" id="ov-svg" viewBox="0 0 720 420"></svg>
    <div class="controls">
      <button class="btn" id="ov-flip">↔ הפוך (שיקוף)</button>
      <button class="btn ghost" id="ov-measure">📏 הצג מידות</button>
      <button class="btn warn" id="ov-not">✋ הם לא חופפים!</button>
      <button class="btn ghost" id="ov-reset">↺ מההתחלה</button>
    </div>
    <p class="hint-line">גרור את המשולש כדי להזיז אותו. גרור את העיגול הוורוד כדי לסובב.</p>
    <div id="ov-result"></div>`;
  const $ = s => root.querySelector(s);
  const svg = $('#ov-svg');
  const PX = 40; // פיקסלים ליחידת אורך
  const center0 = pts => { const c = centroid(pts); return pts.map(p => V.sub(p, c)); };
  const SHAPES = [
    center0([[-100, 55], [95, 40], [-15, -80]]),
    center0([[-100, 55], [95, 40], [-3, -92]]),   // כמעט אותו דבר... אבל לא
  ];
  const TARGET_C = [505, 205];
  const targetPts = SHAPES[0].map(p => V.add(TARGET_C, p));
  const LEVELS = [
    { name: 'הזזה', start: { c: [180, 255], phi: 0, s: 1 }, shape: 0, hint: 'שלב 1: גרור את המשולש הכתום ושים אותו בדיוק על המשולש המקווקו.' },
    { name: 'סיבוב', start: { c: [175, 245], phi: 130 * RAD, s: 1 }, shape: 0, hint: 'שלב 2: עכשיו צריך גם לסובב. גרור את העיגול הוורוד.' },
    { name: 'היפוך', start: { c: [185, 240], phi: -35 * RAD, s: -1 }, shape: 0, hint: 'שלב 3: נסה להניח אותו על המטרה. ואם זה לא מסתדר, אולי צריך משהו נוסף?' },
    { name: 'חשוד', start: { c: [180, 250], phi: 60 * RAD, s: 1 }, shape: 1, hint: 'שלב 4: עוד אחד. אבל תיזהר, משהו פה חשוד...' },
  ];
  const solved = Store.get('ch1solved', [false, false, false, false]);
  let lvl = solved.indexOf(false); if (lvl < 0) lvl = 0;
  let st, busy = false, measures = false, won = false;

  const gTarget = S('g', {}, svg);
  D.poly(gTarget, targetPts, 'target');
  const gMov = S('g', {}, svg);
  const movPoly = S('polygon', { class: 'tri-orange movable' }, gMov);
  const rotLine = S('line', { class: 'rot-line', 'pointer-events': 'none' }, gMov);
  const rotHandle = S('circle', { r: 13, class: 'handle' }, gMov);
  const gTop = S('g', { 'pointer-events': 'none' }, svg);

  const L = () => LEVELS[lvl];
  const shape = () => SHAPES[L().shape];
  const movPts = () => shape().map(p => V.add(st.c, V.rot([p[0] * st.s, p[1]], st.phi)));
  const msg = (html, kind) => { const m = $('#ov-msg'); m.innerHTML = html; m.className = 'msg ' + (kind || ''); };

  function drawMeasures(g, pts, cls) {
    const c = centroid(pts);
    for (let i = 0; i < 3; i++) D.sideLabel(g, pts[i], pts[(i + 1) % 3], c, fmt(V.dist(pts[i], pts[(i + 1) % 3]) / PX), 'val ' + cls, 18);
  }
  function render() {
    const pts = movPts();
    movPoly.setAttribute('points', ptsAttr(pts));
    const R = Math.max(...shape().map(V.len)) + 30;
    const h = V.add(st.c, V.rot([0, -R], st.phi));
    rotLine.setAttribute('x1', st.c[0]); rotLine.setAttribute('y1', st.c[1]);
    rotLine.setAttribute('x2', h[0]); rotLine.setAttribute('y2', h[1]);
    place(rotHandle, h);
    rotLine.style.display = rotHandle.style.display = won ? 'none' : '';
    clearEl(gTop);
    const tc = centroid(targetPts);
    ['D', 'E', 'F'].forEach((n, i) => D.label(gTop, targetPts[i], n, tc, 20, 'lbl c-gray'));
    const mc = centroid(pts);
    ['A', 'B', 'C'].forEach((n, i) => D.label(gTop, pts[i], n, mc, 20, 'lbl c-orange'));
    if (measures) { drawMeasures(gTop, targetPts, 'c-gray'); drawMeasures(gTop, pts, 'c-orange'); }
  }

  function renderLevels() {
    $('#ov-levels').innerHTML = LEVELS.map((l, i) =>
      `<button class="pill ${i === lvl ? 'active' : ''} ${solved[i] ? 'done' : ''}" data-l="${i}">${solved[i] ? '✓' : i + 1} ${l.name}</button>`).join('');
    root.querySelectorAll('[data-l]').forEach(b => b.onclick = () => setLevel(+b.dataset.l));
  }
  function setLevel(i) {
    lvl = i; won = false; busy = false;
    st = Object.assign({}, L().start);
    $('#ov-result').innerHTML = '';
    msg(L().hint);
    renderLevels(); render();
  }

  // גרירה: הזזה
  let off;
  onDrag(svg, movPoly, {
    start: p => { if (busy || won) return false; off = V.sub(st.c, p); },
    move: p => { st.c = [clamp(p[0] + off[0], 30, 690), clamp(p[1] + off[1], 30, 390)]; render(); },
    end: check,
  });
  // גרירה: סיבוב
  let last;
  onDrag(svg, rotHandle, {
    start: p => { if (busy || won) return false; last = V.ang(V.sub(p, st.c)); },
    move: p => { const a = V.ang(V.sub(p, st.c)); st.phi += normAng(a - last); last = a; render(); },
    end: check,
  });

  function check() {
    const dPos = V.dist(st.c, TARGET_C), dAng = Math.abs(normAng(st.phi)) * DEG;
    if (L().shape === 0) {
      if (st.s === 1 && dPos < 24 && dAng < 12) snapWin();
      else if (st.s === -1 && dPos < 45) msg('הוא כמעט במקום, אבל נראה כמו <b>תמונת מראה</b> של המטרה. שום סיבוב לא יעזור כאן. אולי צריך להפוך אותו?', 'info');
      else if (dPos < 30) msg('המיקום טוב! עכשיו סובב אותו עם העיגול הוורוד.', 'info');
    } else if (dPos < 40) {
      msg('משהו לא מסתדר, נכון? לא משנה איך מסובבים, הקצוות לא נופלים בדיוק אחד על השני. אולי כדאי למדוד? 📏', 'info');
    }
  }
  function snapWin() {
    busy = true;
    const c0 = st.c, p0 = normAng(st.phi);
    animate(280, t => { st.c = V.lerp(c0, TARGET_C, ease(t)); st.phi = p0 * (1 - ease(t)); render(); }, () => {
      st.c = TARGET_C.slice(); st.phi = 0; busy = false; won = true;
      render(); win();
    });
  }

  $('#ov-flip').onclick = () => {
    if (busy || won) return;
    busy = true;
    const s0 = st.s;
    animate(450, t => { st.s = s0 * (1 - 2 * ease(t)); render(); }, () => { st.s = -s0; busy = false; render(); check(); });
  };
  $('#ov-measure').onclick = () => { measures = !measures; $('#ov-measure').classList.toggle('on', measures); render(); };
  $('#ov-reset').onclick = () => setLevel(lvl);
  $('#ov-not').onclick = () => {
    if (won) return;
    if (L().shape === 0) { msg('דווקא <b>כן</b> חופפים! נסה להזיז, לסובב, ואולי גם להפוך.', 'bad'); return; }
    won = true; solved[lvl] = true; Store.set('ch1solved', solved); renderLevels();
    const d = (i, j, pts) => fmt(V.dist(pts[i], pts[j]) / PX);
    const mp = SHAPES[1];
    msg('');
    $('#ov-result').innerHTML = `
      <div class="card win"><h4>🕵️ עין חדה! הם באמת לא חופפים.</h4>
      <p>הם נראים דומים מאוד, אבל תמדוד: ${M('AB = ' + d(0, 1, mp) + ',  DE = ' + d(0, 1, targetPts))} ו-${M('BC = ' + d(1, 2, mp) + ',  EF = ' + d(1, 2, targetPts))}, אבל ${M('AC = ' + d(2, 0, mp))} ואילו ${M('DF = ' + d(2, 0, targetPts))}.</p>
      <p>הזזה, סיבוב והיפוך לא משנים אורך של אף צלע. אז אין שום דרך להניח את המשולש הזה על המטרה. חפיפה היא עניין <b>מדויק</b>: "כמעט" לא נחשב.</p></div>`;
    measures = true; $('#ov-measure').classList.add('on'); render();
    celebrate();
    finish();
  };

  function win() {
    solved[lvl] = true; Store.set('ch1solved', solved); renderLevels();
    celebrate();
    const extra = [
      'הזזת את המשולש בלי לשנות אותו, והוא כיסה את המטרה בדיוק.',
      'הזזה וסיבוב לא משנים את המשולש: לא את אורכי הצלעות ולא את הזוויות.',
      'גם <b>היפוך</b> מותר! תחשוב על משולש שגזרת מנייר: מותר להרים אותו ולהפוך אותו על הצד השני. משולש ותמונת המראה שלו חופפים.',
    ][lvl];
    const T = targetPts;
    const side = (i, j) => fmt(V.dist(T[i], T[j]) / PX);
    const ang = i => fmtDeg(angleAt(T[(i + 1) % 3], T[i], T[(i + 2) % 3]));
    msg('');
    $('#ov-result').innerHTML = `
      <div class="card win"><h4>🎉 מושלם! ${M('△ABC ≅ △DEF')}</h4>
      <p>${extra}</p>
      <p><b>A</b> נחת על <b>D</b>, <b>B</b> נחת על <b>E</b> ו-<b>C</b> נחת על <b>F</b>. לכן כותבים ${M('△ABC ≅ △DEF')} בסדר הזה: האות הראשונה מתאימה לראשונה, השנייה לשנייה והשלישית לשלישית. הסדר מספר מי מתאים למי.</p>
      <p>וכשמשולש אחד מכסה את השני בדיוק, <b>כל שש החתיכות</b> שוות בזוגות:</p>
      <div class="pairs">
        <span>${M('AB = DE')} <i>${side(0, 1)}</i></span><span>${M('BC = EF')} <i>${side(1, 2)}</i></span><span>${M('AC = DF')} <i>${side(0, 2)}</i></span>
        <span>${M('∠A = ∠D')} <i>${ang(0)}</i></span><span>${M('∠B = ∠E')} <i>${ang(1)}</i></span><span>${M('∠C = ∠F')} <i>${ang(2)}</i></span>
      </div>
      ${lvl < 3 ? `<button class="btn primary" id="ov-next">לשלב ${lvl + 2} ←</button>` : ''}</div>`;
    const nb = $('#ov-next'); if (nb) nb.onclick = () => setLevel(lvl + 1);
    finish();
  }

  function finish() {
    if (!solved.every(Boolean)) return;
    App.markDone('ch1');
    document.getElementById('ch1-summary').innerHTML = `
      <div class="card idea"><h4>📌 מה למדנו</h4>
      <p>שני משולשים <b>חופפים</b> אם אפשר להניח אחד על השני כך שיכסה אותו בדיוק (בעזרת הזזה, סיבוב והיפוך). ואז כל 6 החתיכות שלהם שוות בזוגות: 3 צלעות ו-3 זוויות.</p>
      <p>אבל רגע. בתרגיל אמיתי אי אפשר לגזור משולשים ולהניח אותם אחד על השני. וגם לבדוק 6 שוויונות זה המון עבודה. <b>האם אפשר להסתפק בפחות?</b></p>
      ${App.nextButton('ch1')}</div>`;
  }

  setLevel(lvl);
  finish();
};
