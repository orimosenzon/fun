// Labs 7–9: one linear program, three algorithms.
// max c·x  s.t.  A x ≤ b, x ≥ 0  (a small carpentry shop: tables x1, cabinets x2)
import { Plot, token } from '../plot.js';
import { $, fmt, fmtt, put, range } from '../ui.js';

const A = [[1, 0], [0, 2], [3, 2]];
const RES = ['חיתוך', 'ליטוש', 'עבודה'];
const VARS = ['x₁', 'x₂', 's₁', 's₂', 's₃'];
const SUB = ['₁', '₂', '₃'];
const st = { b: [4, 12, 18], c: [3, 5], step: 0, sens: 2 };
const listeners = [];
const changed = () => listeners.forEach(f => f());

/* ---------- simplex (dense tableau, Dantzig's rule) ---------- */
export function simplex(Am, b, c) {
  const m = Am.length, n = c.length, W = n + m + 1;
  const T = Am.map((row, i) => [...row, ...Array.from({ length: m }, (_, k) => +(k === i)), b[i]]);
  T.push([...c.map(v => -v), ...Array(m).fill(0), 0]);
  const basis = Array.from({ length: m }, (_, i) => n + i);
  const snap = () => {
    const x = Array(n).fill(0);
    basis.forEach((j, i) => { if (j < n) x[j] = T[i][W - 1]; });
    return { T: T.map(r => r.slice()), basis: basis.slice(), x, z: T[m][W - 1], y: T[m].slice(n, n + m), pivot: null };
  };
  const states = [snap()];
  for (let it = 0; it < 50; it++) {
    let e = -1, best = -1e-9;
    for (let j = 0; j < n + m; j++) if (T[m][j] < best) { best = T[m][j]; e = j; }
    if (e < 0) break;
    let r = -1, br = Infinity;
    for (let i = 0; i < m; i++) if (T[i][e] > 1e-9) { const ratio = T[i][W - 1] / T[i][e]; if (ratio < br - 1e-12) { br = ratio; r = i; } }
    if (r < 0) break;
    states[states.length - 1].pivot = { r, e, ratio: br };
    const pv = T[r][e];
    for (let j = 0; j < W; j++) T[r][j] /= pv;
    for (let i = 0; i <= m; i++) {
      if (i === r) continue;
      const k = T[i][e];
      if (k) for (let j = 0; j < W; j++) T[i][j] -= k * T[r][j];
    }
    basis[r] = e;
    states.push(snap());
  }
  return states;
}

function clipPoly(poly, a, b) {
  const out = [];
  for (let i = 0; i < poly.length; i++) {
    const P = poly[i], Q = poly[(i + 1) % poly.length];
    const fp = a[0] * P[0] + a[1] * P[1] - b, fq = a[0] * Q[0] + a[1] * Q[1] - b;
    if (fp <= 0) out.push(P);
    if ((fp < 0 && fq > 0) || (fp > 0 && fq < 0)) { const t = fp / (fp - fq); out.push([P[0] + t * (Q[0] - P[0]), P[1] + t * (Q[1] - P[1])]); }
  }
  return out;
}
const feasiblePoly = () => A.reduce((poly, a, i) => clipPoly(poly, a, st.b[i]), [[0, 0], [100, 0], [100, 100], [0, 100]]);
const allCons = () => [...A.map((a, i) => ({ a, b: st.b[i] })), { a: [-1, 0], b: 0 }, { a: [0, -1], b: 0 }];

function viewFor(poly, pad = 0.12) {
  const xs = poly.map(p => p[0]), ys = poly.map(p => p[1]);
  const X = Math.max(...xs, 3), Y = Math.max(...ys, 3), s = Math.max(X, Y);
  return [-pad * s, X + 2.5 * pad * s, -pad * s, Y + 2.5 * pad * s];
}

function drawProblem(p, poly, { labels = true } = {}) {
  p.clear();
  p.grid({});
  p.path(poly, { fill: token('--c-feas'), stroke: token('--c-feas-line'), width: 1.5, close: true });
  A.forEach((a, i) => {
    const b = st.b[i], aa = a[0] * a[0] + a[1] * a[1];
    const x0 = [a[0] * b / aa, a[1] * b / aa];
    p.infLine(x0[0], x0[1], -a[1], a[0], { stroke: token('--c-g'), width: 1.5, alpha: .8 });
    if (!labels) return;
    const lx = a[1] === 0 ? b / a[0] : Math.min(p.x1 * 0.92, Math.max(p.x0 + 0.5, (b - a[1] * p.y1 * 0.9) / (a[0] || 1)));
    const ly = a[1] === 0 ? p.y1 * 0.93 : a[0] === 0 ? b / a[1] : (b - a[0] * lx) / a[1];
    p.text(RES[i], a[0] === 0 ? p.x1 : lx, ly, { align: a[0] === 0 ? 'right' : 'left', dx: a[0] === 0 ? -6 : 6, dy: -9, color: token('--c-g'), font: '600 12px system-ui' });
  });
}

/* ---------- Lab 7: simplex, tableau and shadow prices ---------- */
function initSimplex() {
  const plane = new Plot($('#lp-plane'), { view: [0, 10, 0, 10] });
  const sens = new Plot($('#lp-sens'), { view: [0, 1, 0, 1], equal: false });
  let states, poly;
  const SMAX = [10, 24, 36];

  function recompute() {
    states = simplex(A, st.b, st.c);
    st.step = Math.min(st.step, states.length - 1);
    poly = feasiblePoly();
    plane.setView(viewFor(poly));
    render();
  }

  function render() {
    const fin = states[states.length - 1], cur = states[st.step];
    put('lp-x', `(${fmtt(fin.x[0])}, ${fmtt(fin.x[1])})`);
    put('lp-z', fmtt(fin.z));
    put('lp-y', `(${fin.y.map(v => fmtt(v)).join(', ')})`);
    put('lp-dual', fmtt(fin.y.reduce((a, v, i) => a + v * st.b[i], 0)));
    renderTableau(cur, st.step === states.length - 1);
    $('#lp-dual-text').textContent =
      `min   ${st.b.map((b, i) => `${fmtt(b)}·y${SUB[i]}`).join(' + ')}\n` +
      [0, 1].map(j => `      ${A.map((a, i) => a[j] ? `${a[j] === 1 ? '' : a[j]}y${SUB[i]}` : null).filter(Boolean).join(' + ')} ≥ ${fmtt(st.c[j])}`).join('\n') +
      `\n      y₁, y₂, y₃ ≥ 0\n\ny* = (${fin.y.map(v => fmtt(v)).join(', ')})     b·y* = ${fmtt(fin.y.reduce((a, v, i) => a + v * st.b[i], 0))}`;
    plane.invalidate(); sens.invalidate();
  }

  function renderTableau(s, final) {
    const m = A.length, W = s.T[0].length, pv = s.pivot;
    let h = '<table class="tab"><thead><tr><th>בסיס</th>';
    VARS.forEach((v, j) => { h += `<th class="${pv && pv.e === j ? 'enter' : ''}">${v}</th>`; });
    h += '<th>ערך</th></tr></thead><tbody>';
    for (let i = 0; i <= m; i++) {
      const isZ = i === m, leave = pv && pv.r === i;
      h += `<tr class="${isZ ? 'zrow' : ''}"><td class="${leave ? 'leave' : ''}">${isZ ? 'z' : VARS[s.basis[i]]}</td>`;
      for (let j = 0; j < W; j++) {
        const cls = ['num'];
        if (pv && pv.e === j) cls.push('enter');
        if (leave) cls.push('leave');
        if (pv && pv.r === i && pv.e === j) cls.push('piv');
        if (isZ && final && j >= 2 && j < W - 1) cls.push('price');
        h += `<td class="${cls.join(' ')}">${fmtt(s.T[i][j])}</td>`;
      }
      h += '</tr>';
    }
    $('#lp-tableau').innerHTML = h + '</tbody></table>';
    let msg;
    if (pv) {
      const leaving = VARS[s.basis[pv.r]];
      msg = `צעד ${st.step}: בשורת z המקדם השלילי ביותר הוא ${fmtt(s.T[m][pv.e])}, בעמודה ${VARS[pv.e]}. ` +
        `במחירים הנוכחיים כל יחידה של ${VARS[pv.e]} מוסיפה ${fmtt(-s.T[m][pv.e])} לרווח, ולכן היא נכנסת לבסיס. ` +
        `מבחן היחס: מגדילים אותה עד ${fmtt(pv.ratio)}, ואז ${leaving} מגיע לאפס ויוצא.`;
    } else {
      msg = `צעד ${st.step}: אין יותר מספרים שליליים בשורת z, ולכן הקודקוד הזה אופטימלי (z = ${fmtt(s.z)}). ` +
        `שלושת המספרים בשורת z מתחת ל-s₁, s₂, s₃ הם מחירי הצל של המשאבים: y = (${s.y.map(v => fmtt(v)).join(', ')}).`;
    }
    $('#lp-msg').textContent = msg;
  }

  plane.draw = p => {
    drawProblem(p, poly);
    const fin = states[states.length - 1], z = fin.z, c = st.c;
    for (const k of [0.25, 0.5, 0.75, 1]) {
      const v = z * k, cc = c[0] * c[0] + c[1] * c[1];
      p.infLine(c[0] * v / cc, c[1] * v / cc, -c[1], c[0], { stroke: token('--c-f'), width: k === 1 ? 2 : 1, dash: k === 1 ? [] : [5, 5], alpha: k === 1 ? 1 : .6 });
    }
    const path = states.slice(0, st.step + 1).map(s => s.x);
    p.path(path, { stroke: token('--ink'), width: 3 });
    path.forEach(q => p.dot(q[0], q[1], { r: 4, fill: token('--ink') }));
    const cur = states[st.step].x;
    if (st.step === states.length - 1) {
      const sc = 0.14 * (p.x1 - p.x0) / Math.hypot(c[0], c[1]);
      fin.y.forEach((y, i) => {
        if (y > 1e-9) p.arrow(cur[0], cur[1], y * A[i][0] * sc, y * A[i][1] * sc, { color: token('--c-lam'), width: 2.5, label: `y${SUB[i]}a${SUB[i]}` });
      });
      p.arrow(cur[0], cur[1], c[0] * sc, c[1] * sc, { color: token('--c-f'), width: 3, label: 'c' });
    }
    p.dot(cur[0], cur[1], { r: 7.5, fill: token('--ink'), stroke: token('--plot-bg') });
    p.text('x₁ שולחנות', p.x1, 0, { align: 'right', base: 'bottom', dx: -6, dy: -5, color: token('--muted') });
    p.text('x₂ ארונות', 0, p.y1, { align: 'left', base: 'top', dx: 8, dy: 5, color: token('--muted') });
  };

  sens.draw = p => {
    const i = st.sens, bmax = SMAX[i];
    const zAt = v => { const b = st.b.slice(); b[i] = v; const s = simplex(A, b, st.c); return s[s.length - 1].z; };
    const pts = [];
    for (let k = 0; k <= 160; k++) { const v = bmax * k / 160; pts.push([v, zAt(v)]); }
    const zmax = Math.max(...pts.map(q => q[1]));
    p.view = [0, bmax, 0, zmax * 1.1 + 1]; p.layout();
    p.clear();
    p.grid({});
    p.path(pts, { stroke: token('--c-f'), width: 2.5 });
    const fin = states[states.length - 1], y = fin.y[i], b = st.b[i], d = bmax * 0.18;
    p.path([[b - d, fin.z - y * d], [b + d, fin.z + y * d]], { stroke: token('--c-lam'), width: 2.5, dash: [6, 4] });
    p.dot(b, fin.z, { r: 6.5, fill: token('--ink'), stroke: token('--plot-bg') });
    p.text(`שיפוע = y${SUB[i]} = ${fmtt(y)}`, b, fin.z, { dx: 10, dy: 16, color: token('--c-lam'), font: '600 12px system-ui' });
    p.text(`שעות ${RES[i]} (b${SUB[i]})`, p.x1, p.y0, { align: 'right', base: 'bottom', dx: -8, dy: -16, color: token('--muted') });
    p.text('רווח מרבי', p.x1, p.y1, { align: 'right', base: 'top', dx: -8, dy: 6, color: token('--muted') });
  };

  ['lp-b1', 'lp-b2', 'lp-b3'].forEach((id, i) => range(id, v => { st.b[i] = v; changed(); }, v => fmtt(v, 1)));
  ['lp-c1', 'lp-c2'].forEach((id, i) => range(id, v => { st.c[i] = v; changed(); }, v => fmtt(v, 1)));
  $('#lp-step').addEventListener('click', () => { if (st.step < states.length - 1) { st.step++; render(); } });
  $('#lp-reset').addEventListener('click', () => { st.step = 0; render(); });
  $('#lp-solve').addEventListener('click', () => { st.step = states.length - 1; render(); });
  document.querySelectorAll('#lp-sens-pick button').forEach(btn => btn.addEventListener('click', () => {
    st.sens = +btn.dataset.k;
    document.querySelectorAll('#lp-sens-pick button').forEach(o => o.classList.toggle('on', o === btn));
    sens.invalidate();
  }));
  listeners.push(recompute);
  recompute();
}

/* ---------- Lab 8: the ellipsoid method ---------- */
function initEllipsoid() {
  const plot = new Plot($('#ell-plane'), { view: [0, 10, 0, 10] });
  let E, timer = null, poly, zstar, xstar, ball;
  const RATIO = (2 / 3) * Math.sqrt(4 / 3);

  function reset() {
    stop();
    poly = feasiblePoly();
    const s = simplex(A, st.b, st.c), fin = s[s.length - 1];
    zstar = fin.z; xstar = fin.x;
    const [x0, x1, y0, y1] = viewFor(poly, 0);
    const c = [(x0 + x1) / 2 + 1, (y0 + y1) / 2 - 1];
    const R = 1.25 * Math.max(...poly.map(q => Math.hypot(q[0] - c[0], q[1] - c[1])));
    ball = { c, R };
    E = { c, P: [[R * R, 0], [0, R * R]], k: 0, hist: [], best: -Infinity, bestX: null, cut: null, det0: R ** 4 };
    plot.setView([c[0] - R * 1.05, c[0] + R * 1.05, c[1] - R * 1.05, c[1] + R * 1.05]);
    readouts();
  }

  function step() {
    const c = E.c;
    let a = null, worst = 1e-12, kind;
    for (const k of allCons()) {
      const v = (k.a[0] * c[0] + k.a[1] * c[1] - k.b) / Math.hypot(k.a[0], k.a[1]);
      if (v > worst) { worst = v; a = k.a; }
    }
    if (a) kind = 'feas';
    else {
      kind = 'obj';
      const val = st.c[0] * c[0] + st.c[1] * c[1];
      if (val > E.best) { E.best = val; E.bestX = c.slice(); }
      a = [-st.c[0], -st.c[1]];
    }
    const P = E.P, Pa = [P[0][0] * a[0] + P[0][1] * a[1], P[1][0] * a[0] + P[1][1] * a[1]];
    const s = Math.sqrt(a[0] * Pa[0] + a[1] * Pa[1]), bt = [Pa[0] / s, Pa[1] / s];
    E.hist.push({ c: c.slice(), P: P.map(r => r.slice()) });
    E.cut = { c: c.slice(), a, kind, P: P.map(r => r.slice()) };
    E.c = [c[0] - bt[0] / 3, c[1] - bt[1] / 3];
    E.P = [[4 / 3 * (P[0][0] - 2 / 3 * bt[0] * bt[0]), 4 / 3 * (P[0][1] - 2 / 3 * bt[0] * bt[1])],
      [4 / 3 * (P[1][0] - 2 / 3 * bt[1] * bt[0]), 4 / 3 * (P[1][1] - 2 / 3 * bt[1] * bt[1])]];
    E.k++;
    readouts();
  }

  function readouts() {
    const det = E.P[0][0] * E.P[1][1] - E.P[0][1] * E.P[1][0];
    put('ell-k', E.k);
    put('ell-area', fmt(Math.sqrt(det / E.det0), 5));
    put('ell-theory', fmt(RATIO ** E.k, 5));
    put('ell-best', E.bestX ? fmt(E.best, 3) : '?');
    put('ell-gap', E.bestX ? fmt(zstar - E.best, 4) : '?');
    const kind = E.cut ? (E.cut.kind === 'feas' ? 'המרכז מחוץ לתחום: חותכים לפי האילוץ המופר' : 'המרכז חוקי: חותכים לפי המטרה ושומרים רק את מה שטוב ממנו') : 'כדור התחלתי שמכיל את כל התחום';
    put('ell-kind', kind);
    plot.invalidate();
  }

  const ellPts = (c, P) => {
    const l11 = Math.sqrt(P[0][0]), l21 = P[1][0] / l11, l22 = Math.sqrt(Math.max(P[1][1] - l21 * l21, 0)), out = [];
    for (let i = 0; i <= 120; i++) {
      const t = i / 120 * 2 * Math.PI, u = Math.cos(t), v = Math.sin(t);
      out.push([c[0] + l11 * u, c[1] + l21 * u + l22 * v]);
    }
    return out;
  };

  plot.draw = p => {
    if (!poly) return;
    drawProblem(p, poly, { labels: false });
    p.dot(xstar[0], xstar[1], { r: 5, fill: token('--c-f') });
    const recent = E.hist.slice(-8);
    recent.forEach((h, i) => p.path(ellPts(h.c, h.P), { stroke: token('--muted'), width: 1, alpha: 0.15 + 0.5 * (i + 1) / recent.length }));
    if (E.cut) {
      const { c, a } = E.cut, ctx = p.ctx;
      ctx.save();
      ctx.beginPath();
      ellPts(c, E.cut.P).forEach(([x, y], i) => i ? ctx.lineTo(p.X(x), p.Y(y)) : ctx.moveTo(p.X(x), p.Y(y)));
      ctx.clip();
      const d = [-a[1], a[0]], L = 1e3, n = Math.hypot(a[0], a[1]), an = [a[0] / n * L, a[1] / n * L];
      p.path([[c[0] + d[0] * L, c[1] + d[1] * L], [c[0] - d[0] * L, c[1] - d[1] * L],
        [c[0] - d[0] * L + an[0], c[1] - d[1] * L + an[1]], [c[0] + d[0] * L + an[0], c[1] + d[1] * L + an[1]]],
        { fill: token('--c-bad'), alpha: 0.13, close: true });
      ctx.restore();
      p.infLine(c[0], c[1], -a[1], a[0], { stroke: E.cut.kind === 'feas' ? token('--c-g') : token('--c-f'), width: 1.5, dash: [5, 4] });
    }
    p.path(ellPts(E.c, E.P), { stroke: token('--c-lam'), width: 2.5 });
    const inside = allCons().every(k => k.a[0] * E.c[0] + k.a[1] * E.c[1] <= k.b + 1e-9);
    p.dot(E.c[0], E.c[1], { r: 4.5, fill: inside ? token('--c-tan') : token('--c-bad') });
    if (E.bestX) p.dot(E.bestX[0], E.bestX[1], { r: 6, fill: token('--plot-bg'), stroke: token('--c-tan'), width: 2.5 });
  };

  const stop = () => { if (timer) { clearInterval(timer); timer = null; $('#ell-play').textContent = 'הפעלה'; } };
  $('#ell-step').addEventListener('click', () => { stop(); step(); });
  $('#ell-reset').addEventListener('click', reset);
  $('#ell-play').addEventListener('click', () => {
    if (timer) return stop();
    $('#ell-play').textContent = 'עצירה';
    timer = setInterval(() => { if (E.k >= 60) stop(); else step(); }, 380);
  });
  $('#ell-zoom').addEventListener('click', () => {
    const P = E.P, rx = Math.sqrt(P[0][0]), ry = Math.sqrt(P[1][1]), r = Math.max(rx, ry, 0.02) * 1.6;
    plot.setView([E.c[0] - r, E.c[0] + r, E.c[1] - r, E.c[1] + r]);
  });
  $('#ell-full').addEventListener('click', () => {
    const { c, R } = ball;
    plot.setView([c[0] - R * 1.05, c[0] + R * 1.05, c[1] - R * 1.05, c[1] + R * 1.05]);
  });
  listeners.push(reset);
  reset();
}

/* ---------- Lab 9: interior point, the central path ---------- */
function initCentralPath() {
  const plot = new Plot($('#ipm-plane'), { view: [0, 10, 0, 10] });
  const N = 141;
  let path = [], poly, xstar, ystar, idx = 40;

  function newton(x, mu, cons) {
    const c = st.c;
    const slacks = z => cons.map(k => k.b - k.a[0] * z[0] - k.a[1] * z[1]);
    const phi = z => { const s = slacks(z); return s.some(v => v <= 0) ? Infinity : -(c[0] * z[0] + c[1] * z[1]) - mu * s.reduce((a, v) => a + Math.log(v), 0); };
    for (let it = 0; it < 80; it++) {
      const s = slacks(x);
      const g = [-c[0], -c[1]], H = [[0, 0], [0, 0]];
      cons.forEach((k, j) => {
        g[0] += mu * k.a[0] / s[j]; g[1] += mu * k.a[1] / s[j];
        const w = mu / (s[j] * s[j]);
        H[0][0] += w * k.a[0] * k.a[0]; H[0][1] += w * k.a[0] * k.a[1]; H[1][1] += w * k.a[1] * k.a[1];
      });
      H[1][0] = H[0][1];
      const det = H[0][0] * H[1][1] - H[0][1] * H[1][0];
      const dx = [-(H[1][1] * g[0] - H[0][1] * g[1]) / det, -(-H[1][0] * g[0] + H[0][0] * g[1]) / det];
      const dec = -(g[0] * dx[0] + g[1] * dx[1]);
      if (dec / 2 < 1e-13) break;
      let t = 1;
      const f0 = phi(x);
      while (phi([x[0] + t * dx[0], x[1] + t * dx[1]]) > f0 - 0.25 * t * dec && t > 1e-12) t *= 0.5;
      x = [x[0] + t * dx[0], x[1] + t * dx[1]];
    }
    return x;
  }

  function recompute() {
    poly = feasiblePoly();
    const s = simplex(A, st.b, st.c), fin = s[s.length - 1];
    xstar = fin.x; ystar = fin.y;
    const cons = allCons();
    let x = [poly.reduce((a, q) => a + q[0], 0) / poly.length, poly.reduce((a, q) => a + q[1], 0) / poly.length];
    path = [];
    for (let i = 0; i < N; i++) {
      const mu = 10 ** (2 - 5 * i / (N - 1));
      x = newton(x, mu, cons);
      const sl = cons.map(k => k.b - k.a[0] * x[0] - k.a[1] * x[1]);
      path.push({ mu, x, y: sl.slice(0, 3).map(v => mu / v) });
    }
    plot.setView(viewFor(poly));
    readouts();
  }

  function readouts() {
    const q = path[idx];
    put('ipm-mu', q.mu < 0.01 ? q.mu.toExponential(1) : fmt(q.mu, q.mu < 1 ? 3 : 1));
    put('ipm-x', `(${fmt(q.x[0], 3)}, ${fmt(q.x[1], 3)})`);
    put('ipm-y', `(${q.y.map(v => fmt(v, 3)).join(', ')})`);
    put('ipm-ystar', `(${ystar.map(v => fmtt(v)).join(', ')})`);
    put('ipm-gap', fmt(5 * q.mu, 4));
    plot.invalidate();
  }

  plot.draw = p => {
    if (!poly) return;
    drawProblem(p, poly);
    p.path(path.map(q => q.x), { stroke: token('--c-lam'), width: 2.5 });
    [0, 20, 40, 60, 80, 100, 120, 140].forEach(i => p.dot(path[i].x[0], path[i].x[1], { r: 2.5, fill: token('--c-lam') }));
    p.dot(xstar[0], xstar[1], { r: 6, fill: token('--plot-bg'), stroke: token('--c-f'), width: 2.5 });
    const q = path[idx];
    p.dot(q.x[0], q.x[1], { r: 7.5, fill: token('--ink'), stroke: token('--plot-bg') });
  };

  range('ipm-t', v => { idx = v; readouts(); }, v => `${v}/${N - 1}`);
  listeners.push(recompute);
  recompute();
}

export function initLP() {
  initSimplex();
  initEllipsoid();
  initCentralPath();
}
