// Lab 5: KKT. Closest point to a target p inside disk ∩ two half-planes, found by checking every active set.
import { Plot, token } from '../plot.js';
import { $, fmt, dot, norm } from '../ui.js';

const CONS = [
  { name: 'מעגל', tex: 'x² + y² ≤ 4', type: 'circle', r: 2,
    g: ([x, y]) => x * x + y * y - 4, grad: ([x, y]) => [2 * x, 2 * y] },
  { name: 'אלכסון', tex: 'x + y ≤ 2', type: 'line', a: [1, 1], b: 2,
    g: ([x, y]) => x + y - 2, grad: () => [1, 1] },
  { name: 'רצפה', tex: 'y ≥ −1', type: 'line', a: [0, -1], b: 1,
    g: ([x, y]) => -y - 1, grad: () => [0, -1] },
];

function project(c, p) {
  if (c.type === 'circle') {
    const n = norm(p);
    return n < 1e-9 ? null : [c.r * p[0] / n, c.r * p[1] / n];
  }
  const k = (dot(c.a, p) - c.b) / dot(c.a, c.a);
  return [p[0] - k * c.a[0], p[1] - k * c.a[1]];
}

function intersect(c1, c2) {
  if (c1.type === 'circle') [c1, c2] = [c2, c1];
  if (c2.type === 'line') {
    const det = c1.a[0] * c2.a[1] - c1.a[1] * c2.a[0];
    if (Math.abs(det) < 1e-12) return [];
    return [[(c1.b * c2.a[1] - c1.a[1] * c2.b) / det, (c1.a[0] * c2.b - c1.b * c2.a[0]) / det]];
  }
  const aa = dot(c1.a, c1.a), x0 = [c1.a[0] * c1.b / aa, c1.a[1] * c1.b / aa];
  const h2 = c2.r * c2.r - dot(x0, x0);
  if (h2 < 0) return [];
  const h = Math.sqrt(h2), d = [-c1.a[1] / Math.sqrt(aa), c1.a[0] / Math.sqrt(aa)];
  return [[x0[0] + h * d[0], x0[1] + h * d[1]], [x0[0] - h * d[0], x0[1] - h * d[1]]];
}

// minimize |x − p|² subject to the enabled constraints; returns x*, multipliers and the active set
export function solveKKT(p, on) {
  const idx = CONS.map((_, i) => i).filter(i => on[i]);
  const cands = [{ x: p, act: [] }];
  for (const i of idx) { const x = project(CONS[i], p); if (x) cands.push({ x, act: [i] }); }
  for (const i of idx) for (const j of idx) if (i < j) for (const x of intersect(CONS[i], CONS[j])) cands.push({ x, act: [i, j] });
  let best = null;
  for (const c of cands) {
    if (idx.some(i => CONS[i].g(c.x) > 1e-7)) continue;
    const r = [2 * (p[0] - c.x[0]), 2 * (p[1] - c.x[1])];
    const lam = CONS.map(() => 0);
    if (c.act.length === 1) {
      const gg = CONS[c.act[0]].grad(c.x);
      lam[c.act[0]] = dot(r, gg) / dot(gg, gg);
    } else if (c.act.length === 2) {
      const g1 = CONS[c.act[0]].grad(c.x), g2 = CONS[c.act[1]].grad(c.x);
      const det = g1[0] * g2[1] - g1[1] * g2[0];
      if (Math.abs(det) < 1e-12) continue;
      lam[c.act[0]] = (r[0] * g2[1] - r[1] * g2[0]) / det;
      lam[c.act[1]] = (g1[0] * r[1] - g1[1] * r[0]) / det;
    }
    if (lam.some(l => l < -1e-7)) continue;
    const f = (c.x[0] - p[0]) ** 2 + (c.x[1] - p[1]) ** 2;
    if (!best || f < best.f - 1e-12) best = { x: c.x, lam, act: c.act, f };
  }
  return best;
}

export function initKKT() {
  const plot = new Plot($('#kkt-plane'), { view: [-3.2, 3.6, -2.6, 3.4] });
  let p = [2.4, 1.9];
  const on = [true, true, true];

  function halfPath(ctx, P, c) {
    if (c.type === 'circle') {
      ctx.moveTo(P.X(c.r), P.Y(0));
      ctx.arc(P.X(0), P.Y(0), c.r * P.sx, 0, 2 * Math.PI);
      return;
    }
    const aa = Math.sqrt(dot(c.a, c.a)), n = [c.a[0] / aa, c.a[1] / aa], d = [-n[1], n[0]];
    const x0 = [n[0] * c.b / aa, n[1] * c.b / aa], L = 100;
    const pts = [[x0[0] + L * d[0], x0[1] + L * d[1]], [x0[0] + L * d[0] - L * n[0], x0[1] + L * d[1] - L * n[1]],
      [x0[0] - L * d[0] - L * n[0], x0[1] - L * d[1] - L * n[1]], [x0[0] - L * d[0], x0[1] - L * d[1]]];
    pts.forEach(([x, y], i) => i ? ctx.lineTo(P.X(x), P.Y(y)) : ctx.moveTo(P.X(x), P.Y(y)));
    ctx.closePath();
  }

  function boundary(P, c, style) {
    if (c.type === 'circle') {
      const pts = [];
      for (let i = 0; i <= 200; i++) pts.push([c.r * Math.cos(i * Math.PI / 100), c.r * Math.sin(i * Math.PI / 100)]);
      P.path(pts, style);
    } else {
      const aa = dot(c.a, c.a);
      P.infLine(c.a[0] * c.b / aa, c.a[1] * c.b / aa, -c.a[1], c.a[0], style);
    }
  }

  function render() {
    const sol = solveKKT(p, on);
    const tb = $('#kkt-table tbody');
    tb.innerHTML = '';
    CONS.forEach((c, i) => {
      if (!on[i]) return;
      const gv = c.g(sol.x), l = sol.lam[i], active = sol.act.includes(i);
      const tr = document.createElement('tr');
      tr.innerHTML = `<td>${c.name}</td><td class="num">${c.tex}</td><td class="num">${fmt(gv, 3)}</td>` +
        `<td class="num price">${fmt(l, 3)}</td><td class="num">${fmt(l * gv, 3)}</td>` +
        `<td>${active ? (l > 1e-6 ? 'הדוק, דוחף' : 'הדוק, לא דוחף') : 'רפוי'}</td>`;
      tb.appendChild(tr);
    });
    const st = $('#kkt-status');
    const names = sol.act.map(i => CONS[i].name);
    if (!names.length) {
      st.className = 'status';
      st.textContent = 'המטרה בתוך התחום: אף אילוץ לא פעיל, כל הכופלים אפס, והפתרון הוא פשוט ∇f = 0.';
    } else {
      st.className = 'status ok';
      const top = sol.act.reduce((a, b) => sol.lam[b] > sol.lam[a] ? b : a);
      st.textContent = `אילוצים פעילים: ${names.join(' ו')}. כיוון הירידה של f נמצא בתוך החרוט של הגרדיאנטים שלהם. ` +
        `כל יחידה של הרחבה באילוץ ה${CONS[top].name} מקטינה את f* בערך ב-${fmt(sol.lam[top], 2)}.`;
    }
    $('#kkt-f').textContent = fmt(sol.f, 3);
    plot.invalidate();
  }

  plot.draw = P => {
    P.clear();
    P.grid({});
    const ctx = P.ctx;
    ctx.save();
    CONS.forEach((c, i) => { if (on[i]) { ctx.beginPath(); halfPath(ctx, P, c); ctx.clip(); } });
    ctx.fillStyle = token('--c-feas');
    ctx.fillRect(0, 0, P.w, P.h);
    CONS.forEach((c, i) => { if (on[i]) boundary(P, c, { stroke: token('--c-g'), width: 5 }); });
    ctx.restore();
    CONS.forEach((c, i) => { if (on[i]) boundary(P, c, { stroke: token('--c-g'), width: 1, dash: [3, 5], alpha: .6 }); });

    const sol = solveKKT(p, on), x = sol.x, rf = Math.sqrt(sol.f);
    for (const k of [0.5, 1.5, 2.2]) {
      if (rf * k < 0.05) continue;
      const pts = [];
      for (let i = 0; i <= 120; i++) pts.push([p[0] + rf * k * Math.cos(i * Math.PI / 60), p[1] + rf * k * Math.sin(i * Math.PI / 60)]);
      P.path(pts, { stroke: token('--contour'), width: 1 });
    }
    if (rf > 1e-3) {
      const pts = [];
      for (let i = 0; i <= 160; i++) pts.push([p[0] + rf * Math.cos(i * Math.PI / 80), p[1] + rf * Math.sin(i * Math.PI / 80)]);
      P.path(pts, { stroke: token('--c-f'), width: 2 });
    }

    const sc = 0.32, grads = sol.act.map(i => CONS[i].grad(x));
    if (grads.length) {
      const u = grads.map(g => [g[0] / norm(g) * 1.6, g[1] / norm(g) * 1.6]);
      const wedge = u.length === 2 ? [x, [x[0] + u[0][0], x[1] + u[0][1]], [x[0] + u[1][0], x[1] + u[1][1]]] : null;
      if (wedge) P.path(wedge, { fill: token('--c-feas'), stroke: token('--c-lam'), width: 1, close: true, alpha: .9 });
      u.forEach(v => P.path([x, [x[0] + v[0], x[1] + v[1]]], { stroke: token('--c-lam'), width: 1, dash: [3, 3] }));
      sol.act.forEach((i, k) => {
        const g = grads[k], l = sol.lam[i];
        P.arrow(x[0], x[1], l * g[0] * sc, l * g[1] * sc, { color: token('--c-lam'), width: 2.5, label: `λ${i + 1}∇g${i + 1}` });
      });
    }
    P.arrow(x[0], x[1], 2 * (p[0] - x[0]) * sc, 2 * (p[1] - x[1]) * sc, { color: token('--c-f'), width: 3, label: sol.act.length === 2 ? '−∇f' : '' });
    P.dot(p[0], p[1], { r: 8, fill: token('--plot-bg'), stroke: token('--c-f'), width: 3 });
    P.text('p', p[0], p[1], { dx: 11, dy: -10, color: token('--c-f'), font: '600 14px system-ui' });
    P.dot(x[0], x[1], { r: 6.5, fill: token('--ink'), stroke: token('--plot-bg') });
  };

  plot.onDrag({ move: (x, y) => { p = [x, y]; render(); } });
  [0, 1, 2].forEach(i => {
    const cb = $(`#kkt-on${i}`);
    cb.addEventListener('change', () => { on[i] = cb.checked; render(); });
  });
  render();
}
