// 3D lab: a linear program in three variables. The polytope, the simplex walking its edges under
// different pivot rules, and the Klee–Minty cube on which Dantzig's rule visits every vertex.
import { View3D, THREE, V, tube, ball, arrow, patch, translucent, segments, axesLines } from './view3d.js';
import { simplex } from '../demos/lp.js';
import { $, fmt, fmtt, put, range, segmented } from '../ui.js';

const PRESETS = {
  km: { A: [[1, 0, 0], [4, 1, 0], [8, 4, 1]], b: [5, 25, 125], c: [4, 2, 1], names: ['x₁ ≤ 5', '4x₁ + x₂ ≤ 25', '8x₁ + 4x₂ + x₃ ≤ 125'] },
  shop: { A: [[1, 0, 0.5], [0, 2, 0], [3, 2, 1]], b: [4, 12, 18], c: [3, 5, 2], names: ['חיתוך', 'ליטוש', 'עבודה'] },
};
const SUB = ['₁', '₂', '₃'];

function solve3(M, r) {
  const d = (m) => m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
  const D = d(M);
  if (Math.abs(D) < 1e-12) return null;
  return [0, 1, 2].map(k => d(M.map((row, i) => row.map((v, j) => j === k ? r[i] : v))) / D);
}

function polytope(A, b) {
  const cons = [...A.map((a, i) => ({ a, b: b[i] })), { a: [-1, 0, 0], b: 0 }, { a: [0, -1, 0], b: 0 }, { a: [0, 0, -1], b: 0 }];
  const verts = [];
  for (let i = 0; i < cons.length; i++) for (let j = i + 1; j < cons.length; j++) for (let k = j + 1; k < cons.length; k++) {
    const x = solve3([cons[i].a, cons[j].a, cons[k].a], [cons[i].b, cons[j].b, cons[k].b]);
    if (!x || cons.some(c => c.a[0] * x[0] + c.a[1] * x[1] + c.a[2] * x[2] > c.b + 1e-7)) continue;
    if (verts.some(v => Math.hypot(v.x[0] - x[0], v.x[1] - x[1], v.x[2] - x[2]) < 1e-7)) continue;
    const act = cons.map((c, m) => Math.abs(c.a[0] * x[0] + c.a[1] * x[1] + c.a[2] * x[2] - c.b) < 1e-7 ? m : -1).filter(m => m >= 0);
    verts.push({ x, act });
  }
  return { cons, verts };
}

export function initPolytope() {
  const view = new View3D($('#poly-3d'), { cam: [11.5, -8, 7.5], target: [2, 2, 2.2] });
  const s = { key: 'km', rule: 'dantzig', step: 0, c: null };
  let P, poly, states, scale;
  const stat = view.layer(), dyn = view.layer();
  const mFace = translucent(view, '--c-f', 0.16), mEdge = view.mat('--c-feas-line', { kind: 'line' });
  const mInk = view.mat('--ink'), mF = view.mat('--c-f'), mLam = view.mat('--c-lam'), mPath = view.mat('--c-tan');
  const mVert = view.mat('--muted'), mLevel = translucent(view, '--c-f', 0.16);
  const D = x => V(x[0] * scale[0], x[1] * scale[1], x[2] * scale[2]);   // display map
  const Dn = a => V(a[0] / scale[0], a[1] / scale[1], a[2] / scale[2]);  // normals: inverse transpose

  function setPreset(k) {
    s.key = k; P = PRESETS[k]; s.c = P.c.slice();
    ['poly-c1', 'poly-c2', 'poly-c3'].forEach((id, i) => cSliders[i].set(s.c[i]));
    poly = polytope(P.A, P.b);
    scale = [0, 1, 2].map(i => 4 / Math.max(...poly.verts.map(v => v.x[i]), 1e-9));
    stat.reset();
    // faces: vertices on each constraint, sorted by angle around their centroid
    poly.cons.forEach((c, m) => {
      const vs = poly.verts.filter(v => v.act.includes(m)).map(v => D(v.x));
      if (vs.length < 3) return;
      const cen = vs.reduce((a, v) => a.add(v), V(0, 0, 0)).multiplyScalar(1 / vs.length);
      const n = Dn(c.a).normalize(), u = vs[0].clone().sub(cen).normalize(), w = n.clone().cross(u);
      vs.sort((p, q) => Math.atan2(p.clone().sub(cen).dot(w), p.clone().sub(cen).dot(u)) - Math.atan2(q.clone().sub(cen).dot(w), q.clone().sub(cen).dot(u)));
      const pos = [];
      for (let i = 1; i + 1 < vs.length; i++) pos.push(...vs[0].toArray(), ...vs[i].toArray(), ...vs[i + 1].toArray());
      const g = new THREE.BufferGeometry();
      g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
      stat.add(new THREE.Mesh(g, mFace));
    });
    const edges = [];
    for (let i = 0; i < poly.verts.length; i++) for (let j = i + 1; j < poly.verts.length; j++) {
      const shared = poly.verts[i].act.filter(m => poly.verts[j].act.includes(m));
      if (shared.length >= 2) edges.push([...D(poly.verts[i].x).toArray(), ...D(poly.verts[j].x).toArray()]);
    }
    stat.add(segments(edges, mEdge));
    poly.verts.forEach(v => stat.add(ball(D(v.x), 0.05, mVert)));
    stat.add(axesLines(view, [0, 4.8, 0, 4.8, 0, 4.8], ['x₁', 'x₂', 'x₃']));
    document.querySelectorAll('#lab-poly [data-preset]').forEach(el => { el.hidden = el.dataset.preset !== k; });
    put('poly-nv', poly.verts.length);
    solveAll();
  }

  function solveAll() {
    states = simplex(P.A, P.b, s.c, s.rule);
    s.step = 0;
    build();
  }

  function build() {
    dyn.reset();
    const pts = states.slice(0, s.step + 1).map(q => D(q.x));
    for (let i = 0; i + 1 < pts.length; i++) dyn.add(tube([pts[i], pts[i].clone().lerp(pts[i + 1], 0.5), pts[i + 1]], 0.045, mPath));
    pts.forEach(p => dyn.add(ball(p, 0.08, mPath)));
    const cur = states[s.step], p = D(cur.x), cd = Dn(s.c);
    dyn.add(ball(p, 0.13, mInk));
    if (cd.length() > 1e-9) dyn.add(patch(p, cd, 5.5, mLevel));
    const last = s.step === states.length - 1;
    if (last && cd.length() > 1e-9) {
      const k = 1.3 / cd.length();
      dyn.add(arrow(p, cd.clone().multiplyScalar(k), mF, 0.045));
      const lc = view.label('c', 'f'); lc.position.copy(p.clone().addScaledVector(cd, k * 1.12)); dyn.add(lc);
      cur.y.forEach((y, i) => {
        if (y < 1e-9) return;
        const v = Dn(P.A[i]).multiplyScalar(y * k);
        dyn.add(arrow(p, v, mLam, 0.035));
        const l = view.label(`y${SUB[i]}a${SUB[i]}`, 'lam'); l.position.copy(p.clone().addScaledVector(v, 1.12)); dyn.add(l);
      });
      // active x_j ≥ 0: their multipliers are the reduced costs left in the z row
      [0, 1, 2].forEach(j => {
        const mu = cur.T[P.A.length][j];
        if (mu < 1e-9) return;
        const e = [0, 0, 0]; e[j] = -1;
        const v = Dn(e).multiplyScalar(mu * k);
        dyn.add(arrow(p, v, mLam, 0.035));
        const l = view.label(`μ${SUB[j]}`, 'lam'); l.position.copy(p.clone().addScaledVector(v, 1.12)); dyn.add(l);
      });
    }
    view.request();
    put('poly-step', `${s.step} / ${states.length - 1}`);
    put('poly-x', `(${cur.x.map(v => fmtt(v, 2)).join(', ')})`);
    put('poly-z', fmtt(cur.z, 2));
    put('poly-y', last ? `(${cur.y.map(v => fmtt(v, 3)).join(', ')})` : '…');
    const pv = cur.pivot;
    put('poly-msg', pv
      ? `הצעד הבא: ${['x₁', 'x₂', 'x₃', 's₁', 's₂', 's₃'][pv.e]} נכנס לבסיס. המסלול ממשיך לאורך צלע לקודקוד שכן.`
      : `אופטימום אחרי ${states.length - 1} צעדים. כל הקודקודים "מתחת" למישור הכחול, ו-c הוא צירוף אי-שלילי של הנורמלים של הפאות הפעילות.`);
  }

  const cSliders = ['poly-c1', 'poly-c2', 'poly-c3'].map((id, i) => range(id, v => { s.c[i] = v; solveAll(); }, v => fmtt(v, 1)));
  segmented('poly-rule', r => { s.rule = r; if (P) solveAll(); });
  $('#poly-next').addEventListener('click', () => { if (s.step < states.length - 1) { s.step++; build(); } });
  $('#poly-reset').addEventListener('click', () => { s.step = 0; build(); });
  let timer = null;
  $('#poly-play').addEventListener('click', () => {
    if (timer) { clearInterval(timer); timer = null; return; }
    s.step = 0; build();
    timer = setInterval(() => { if (s.step >= states.length - 1) { clearInterval(timer); timer = null; } else { s.step++; build(); } }, 700);
  });
  segmented('poly-presets', setPreset)('km');
}
