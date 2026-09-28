'use strict';
/* פרק 5: תרגול אינסופי. שני משולשים עם נתונים, ועומר מחליט לפי איזה משפט הם חופפים. */

App.inits.ch5 = function () {
  const root = document.getElementById('practice');
  root.innerHTML = `
    <div class="score" id="pr-score"></div>
    <svg class="board" viewBox="0 0 720 320"></svg>
    <div class="qline">לפי איזה משפט המשולשים חופפים?</div>
    <div class="opts" id="pr-opts"></div>
    <div id="pr-fb"></div>`;
  const $ = s => root.querySelector(s);
  const svg = root.querySelector('svg');
  const ANS = ['צ.צ.צ', 'צ.ז.צ', 'ז.צ.ז', 'צ.צ.ז', 'אין מספיק נתונים'];
  const COLORS = ['#db2777', '#0d9488', '#7c3aed'];
  const SIDE_V = { a: ['B', 'C'], b: ['A', 'C'], c: ['A', 'B'] };
  const OPP = { A: 'a', B: 'b', C: 'c' };
  const others = v => ['A', 'B', 'C'].filter(x => x !== v);
  const TYPES = ['SSS', 'SAS', 'ASA', 'AAS', 'SSA+', 'SSA-', 'AAA'];
  const KEY = { SSS: 0, SAS: 1, ASA: 2, AAS: 2, 'SSA+': 3, 'SSA-': 4, AAA: 4 };

  let stats = Store.get('practice', { right: 0, total: 0, streak: 0, best: 0 });
  let q = null, answered = false, lastType = null;

  function randTri() {
    for (;;) {
      const A = ri(35, 100), B = ri(30, 110), C = 180 - A - B;
      if (C < 30) continue;
      const c = rf(5, 7), sC = Math.sin(C * RAD);
      const a = c * Math.sin(A * RAD) / sC, b = c * Math.sin(B * RAD) / sC;
      if (Math.max(a, b) > 9.5 || Math.min(a, b) < 3) continue;
      return { P: { A: [0, 0], B: [c, 0], C: [b * Math.cos(A * RAD), b * Math.sin(A * RAD)] }, ang: { A, B, C }, len: { a, b, c } };
    }
  }
  const pick = arr => arr[Math.floor(Math.random() * arr.length)];

  function gen(type) {
    for (let k = 0; k < 800; k++) {
      const T = randTri(), L = T.len, r = x => Math.round(x * 10) / 10;
      let sel = null, info = {};
      if (type === 'SSS') sel = ['a', 'b', 'c'];
      else if (type === 'AAA') sel = ['A', 'B', 'C'];
      else if (type === 'SAS') { const v = pick('ABC'); sel = [...['a', 'b', 'c'].filter(s => SIDE_V[s].includes(v)), v]; info.v = v; }
      else if (type === 'ASA') { const s = pick('abc'); sel = [...SIDE_V[s], s]; }
      else if (type === 'AAS') { const [x, y] = shuffle(['A', 'B', 'C']); sel = [x, y, OPP[pick([x, y])]]; }
      else {
        const v = pick('ABC'), opp = OPP[v], adj = pick(['a', 'b', 'c'].filter(s => s !== opp));
        const big = r(L[opp]) >= r(L[adj]) + 0.6, small = r(L[opp]) <= r(L[adj]) - 0.6;
        if ((type === 'SSA+' && big) || (type === 'SSA-' && small)) { sel = [adj, opp, v]; info = { v, opp, adj }; }
      }
      if (sel) return { T, sel, type, info };
    }
  }

  const val = (T, id) => 'abc'.includes(id) ? fmt(T.len[id]) : T.ang[id] + '°';
  const nm = (id, map) => 'abc'.includes(id) ? SIDE_V[id].map(v => map[v]).join('') : '∠' + map[id];

  // מיקום משולש על המסך: סיבוב (והיפוך) ואז מרכוז באזור
  function placeTri(P, center, rot, flip, U) {
    const pts = ['A', 'B', 'C'].map(k => { let p = [P[k][0], -P[k][1]]; if (flip) p = [-p[0], p[1]]; return V.rot(p, rot); });
    const xs = pts.map(p => p[0]), ys = pts.map(p => p[1]);
    const cx = (Math.min(...xs) + Math.max(...xs)) / 2, cy = (Math.min(...ys) + Math.max(...ys)) / 2;
    const out = {};
    ['A', 'B', 'C'].forEach((k, i) => { out[k] = [center[0] + (pts[i][0] - cx) * U, center[1] + (pts[i][1] - cy) * U]; });
    return out;
  }
  function fitU(P, rot, flip) {
    const t = placeTri(P, [0, 0], rot, flip, 1);
    const xs = Object.values(t).map(p => p[0]), ys = Object.values(t).map(p => p[1]);
    return Math.min(270 / (Math.max(...xs) - Math.min(...xs)), 230 / (Math.max(...ys) - Math.min(...ys)));
  }

  function drawTri(g, S2, names, cls, sel, T) {
    const pts = [S2.A, S2.B, S2.C], ctr = centroid(pts);
    D.poly(g, pts, cls);
    sel.forEach((id, i) => {
      const col = COLORS[i];
      if ('abc'.includes(id)) {
        const [p, q2] = SIDE_V[id];
        D.seg(g, S2[p], S2[q2], { class: 'selside', style: `stroke:${col}` });
        D.sideLabel(g, S2[p], S2[q2], ctr, val(T, id), { class: 'val', style: `fill:${col}` }, 17);
      } else {
        const [p, q2] = others(id);
        D.arc(g, S2[id], S2[p], S2[q2], 1, 22, { class: 'selarc', style: `stroke:${col}` }, { style: `fill:${col};opacity:.18` });
        D.angLabel(g, S2[id], S2[p], S2[q2], 42, val(T, id), { class: 'val', style: `fill:${col}` });
      }
    });
    ['A', 'B', 'C'].forEach(k => D.label(g, S2[k], names[k], ctr, 18, 'lbl'));
  }

  function next() {
    let type; do { type = pick(TYPES); } while (type === lastType);
    lastType = type;
    q = gen(type); answered = false;
    clearEl(svg);
    const r1 = rf(-0.5, 0.5), r2 = rf(0, 2 * Math.PI), f2 = Math.random() < 0.5;
    const U = Math.min(fitU(q.T.P, r1, false), fitU(q.T.P, r2, f2));
    q.n1 = { A: 'A', B: 'B', C: 'C' }; q.n2 = { A: 'D', B: 'E', C: 'F' };
    drawTri(svg, placeTri(q.T.P, [540, 160], r1, false, U), q.n1, 'tri-blue', q.sel, q.T);
    drawTri(svg, placeTri(q.T.P, [180, 160], r2, f2, U), q.n2, 'tri-orange', q.sel, q.T);
    $('#pr-opts').innerHTML = ANS.map((a, i) => `<button class="opt" data-i="${i}">${a}</button>`).join('');
    root.querySelectorAll('#pr-opts .opt').forEach(b => b.onclick = () => answer(+b.dataset.i, b));
    $('#pr-fb').innerHTML = '';
    renderScore();
  }

  function explain() {
    const t = q.type, n1 = id => M(nm(id, q.n1));
    const list = q.sel.map(id => M(nm(id, q.n1) + ' = ' + nm(id, q.n2))).join(', ');
    const base = `ידוע: ${list}. `;
    switch (t) {
      case 'SSS': return base + 'שלוש צלעות שוות בזוגות: <b>צ.צ.צ</b>.';
      case 'SAS': return base + `שתי צלעות, והזווית ${n1(q.info.v)} נמצאת <b>ביניהן</b> (שתי הצלעות יוצאות ממנה): <b>צ.ז.צ</b>.`;
      case 'ASA': return base + 'שתי זוויות, והצלע שביניהן (היא מחברת את שני הקודקודים של הזוויות): <b>ז.צ.ז</b>.';
      case 'AAS': return base + 'שתי זוויות וצלע שלא ביניהן. מחשבים את הזווית השלישית (180° פחות השתיים), ואז יש שתי זוויות והצלע שביניהן: <b>ז.צ.ז</b>.';
      case 'SSA+': return base + `הזווית ${n1(q.info.v)} לא בין שתי הצלעות. היא מול ${n1(q.info.opp)} (${val(q.T, q.info.opp)}), וזו הצלע הגדולה מבין השתיים (${n1(q.info.adj)} = ${val(q.T, q.info.adj)}). לכן <b>צ.צ.ז</b>.`;
      case 'SSA-': return base + `הזווית ${n1(q.info.v)} מול ${n1(q.info.opp)} (${val(q.T, q.info.opp)}), שהיא הצלע ה<b>קטנה</b> (${n1(q.info.adj)} = ${val(q.T, q.info.adj)}). זה המקרה המסוכן מהמעבדה: יכולים להיות שני משולשים שונים. <b>אין מספיק נתונים</b>.`;
      case 'AAA': return base + 'רק זוויות. יכולים להיות משולשים באותה צורה ובגדלים שונים. <b>אין מספיק נתונים</b>.';
    }
  }

  function answer(i, btn) {
    if (answered) return;
    answered = true;
    const ok = i === KEY[q.type];
    stats.total++;
    if (ok) { stats.right++; stats.streak++; stats.best = Math.max(stats.best, stats.streak); } else stats.streak = 0;
    Store.set('practice', stats);
    root.querySelectorAll('#pr-opts .opt').forEach((b, k) => {
      b.disabled = true;
      if (k === KEY[q.type]) b.classList.add('right');
    });
    if (!ok) btn.classList.add('wrong');
    if (ok && stats.streak > 0 && stats.streak % 5 === 0) celebrate();
    $('#pr-fb').innerHTML = `<div class="card ${ok ? 'win' : 'lose'}"><h4>${ok ? '✔ נכון!' : '✗ לא הפעם'}</h4><p>${explain()}</p>
      <button class="btn primary" id="pr-next">שאלה הבאה ←</button></div>`;
    $('#pr-next').onclick = next;
    if (stats.right >= 10) App.markDone('ch5');
    renderScore();
  }

  function renderScore() {
    $('#pr-score').innerHTML = `<span>✔ ${stats.right} מתוך ${stats.total}</span><span>🔥 רצף: ${stats.streak}</span><span>🏅 שיא: ${stats.best}</span>
      <button class="btn ghost small" id="pr-reset">איפוס</button>`;
    $('#pr-reset').onclick = () => { stats = { right: 0, total: 0, streak: 0, best: 0 }; Store.set('practice', stats); renderScore(); };
  }

  next();
};
