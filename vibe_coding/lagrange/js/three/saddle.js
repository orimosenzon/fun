// 3D lab: the surface L(x, λ). Straight lines along λ (the regulator), parabolas along x (the planner),
// the dual curve q(λ) along the valley floor, and a saddle point exactly when there is no duality gap.
import { View3D, THREE, V, col, surface, tube, ball, segments, axesLines } from './view3d.js';
import { $, fmt, put, range, segmented } from '../ui.js';

const PRESETS = {
  convex: { f: x => x * x, g: x => 1 - x, X: [-1, 3], L: [0, 4], zs: 0.28, xs: 1, sol: { x: 1, lam: 2 } },
  gap: { f: x => 4 - x * x, g: x => x, X: [-1, 2], L: [0, 4], zs: 0.28, xs: 1.3, sol: null },
};

export function initSaddle() {
  const view = new View3D($('#sad-3d'), { cam: [6.8, -8.2, 3.6], target: [0, 0, 0.9] });
  let P, s = { x: 1.6, lam: 0.8 };
  const stat = view.layer(), dyn = view.layer();
  const mLam = view.mat('--c-lam'), mF = view.mat('--c-f'), mInk = view.mat('--ink');
  const L = (x, l) => P.f(x) + l * P.g(x);
  // scene coordinates: centered box, x and λ horizontal, L vertical
  const S = (x, l, z) => V((x - (P.X[0] + P.X[1]) / 2) * P.xs, l - 2, z * P.zs);
  const xs = n => Array.from({ length: n + 1 }, (_, i) => P.X[0] + (P.X[1] - P.X[0]) * i / n);
  const ls = n => Array.from({ length: n + 1 }, (_, i) => P.L[0] + (P.L[1] - P.L[0]) * i / n);
  const q = l => { let m = Infinity, at = 0; for (const x of xs(600)) { const v = L(x, l); if (v < m) { m = v; at = x; } } return { q: m, x: at }; };

  function setPreset(k) {
    P = PRESETS[k];
    stat.reset();
    const lo = col('--surf-lo'), hi = col('--surf-hi');
    let zmin = Infinity, zmax = -Infinity;
    for (const x of xs(40)) for (const l of ls(40)) { zmin = Math.min(zmin, L(x, l)); zmax = Math.max(zmax, L(x, l)); }
    const mat = new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.9, side: THREE.DoubleSide, transparent: true, opacity: 0.92 });
    const mesh = surface((u, v) => 0, [0, 1, 0, 1], 80, mat);
    const pos = mesh.geometry.getAttribute('position'), colors = [];
    for (let k2 = 0; k2 < pos.count; k2++) {
      const x = P.X[0] + (P.X[1] - P.X[0]) * pos.getX(k2), l = P.L[0] + (P.L[1] - P.L[0]) * pos.getY(k2), z = L(x, l), p = S(x, l, z);
      pos.setXYZ(k2, p.x, p.y, p.z);
      const c = lo.clone().lerp(hi, (z - zmin) / (zmax - zmin)); colors.push(c.r, c.g, c.b);
    }
    mesh.geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
    mesh.geometry.computeVertexNormals();
    stat.add(mesh);
    view.onTheme = () => setPreset(k);
    // ruling lines: straight along λ, parabolas along x
    const lines = [];
    for (const x of xs(16)) { const a = S(x, P.L[0], L(x, P.L[0])), b = S(x, P.L[1], L(x, P.L[1])); lines.push([a.x, a.y, a.z + 0.01, b.x, b.y, b.z + 0.01]); }
    for (const l of ls(8)) {
      const pts = xs(60).map(x => S(x, l, L(x, l)));
      for (let i = 0; i + 1 < pts.length; i++) lines.push([pts[i].x, pts[i].y, pts[i].z + 0.01, pts[i + 1].x, pts[i + 1].y, pts[i + 1].z + 0.01]);
    }
    stat.add(segments(lines, view.mat('--contour', { kind: 'line', transparent: true, opacity: 0.8 })));
    // the dual function along the valley floor
    stat.add(tube(ls(80).map(l => { const r = q(l); return S(r.x, l, r.q).add(V(0, 0, 0.02)); }), 0.04, mLam));
    const a0 = S(P.X[0], P.L[0], Math.min(0, zmin)), ax = new THREE.Group();
    ax.add(axesLines(view, [0, (P.X[1] - P.X[0]) * P.xs + 0.4, 0, 4.4, 0, (zmax - Math.min(0, zmin)) * P.zs + 0.5], ['x', 'λ', 'L']));
    ax.position.copy(a0);
    stat.add(ax);
    // best dual point; and the saddle, if there is one
    let bl = 0, bq = -Infinity;
    for (const l of ls(400)) { const r = q(l); if (r.q > bq) { bq = r.q; bl = l; } }
    const bx = q(bl).x;
    stat.add(ball(S(bx, bl, bq), 0.11, view.mat('--ink')));
    const ld = view.label(`max q = d* = ${fmt(bq, 2)}`, 'lam'); ld.position.copy(S(bx, bl, bq).add(V(0, 0, 0.45)));
    stat.add(ld);
    if (P.sol) {
      const sp = S(P.sol.x, P.sol.lam, L(P.sol.x, P.sol.lam));
      const ls2 = view.label('נקודת אוכף', 'ink'); ls2.position.copy(sp.clone().add(V(0.3, 0, 0.75)));
      stat.add(ls2);
    }
    document.querySelectorAll('#lab-saddle [data-preset]').forEach(el => { el.hidden = el.dataset.preset !== k; });
    xSlider.el.min = P.X[0]; xSlider.el.max = P.X[1];
    s.x = Math.min(P.X[1], Math.max(P.X[0], s.x)); xSlider.set(s.x);
    build();
  }

  function build() {
    dyn.reset();
    // slice at fixed λ: what the planner minimizes
    const sl = xs(80).map(x => S(x, s.lam, L(x, s.lam)).add(V(0, 0, 0.03)));
    dyn.add(tube(sl, 0.03, mLam));
    const r = q(s.lam);
    dyn.add(ball(S(r.x, s.lam, r.q), 0.075, mLam));
    // slice at fixed x: what the regulator maximizes (a straight line of slope g(x))
    const a = S(s.x, P.L[0], L(s.x, P.L[0])).add(V(0, 0, 0.03)), b = S(s.x, P.L[1], L(s.x, P.L[1])).add(V(0, 0, 0.03));
    dyn.add(tube([a, a.clone().lerp(b, 0.5), b], 0.03, mF));
    const gx = P.g(s.x), bestL = gx > 1e-9 ? P.L[1] : gx < -1e-9 ? P.L[0] : s.lam;
    dyn.add(ball(S(s.x, bestL, L(s.x, bestL)), 0.075, mF));
    dyn.add(ball(S(s.x, s.lam, L(s.x, s.lam)), 0.06, mInk));
    view.request();

    put('sad-L', fmt(L(s.x, s.lam), 3));
    put('sad-q', fmt(r.q, 3) + '  (x = ' + fmt(r.x, 2) + ')');
    put('sad-g', fmt(gx, 3));
    put('sad-reg', gx > 1e-9 ? 'מעלה את λ בלי גבול' : gx < -1e-9 ? 'λ = 0' : 'אדיש: כל λ טוב');
  }

  const xSlider = range('sad-x', v => { s.x = v; build(); }, v => fmt(v, 2));
  range('sad-lam', v => { s.lam = v; build(); }, v => fmt(v, 2));
  segmented('sad-presets', setPreset)('convex');
}
