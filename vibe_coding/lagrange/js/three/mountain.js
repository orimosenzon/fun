// 3D lab: the mountain f(x, y), the trail g = c standing on it as a curtain, and f*(c).
import { View3D, THREE, V, col, surface, tube, ball, arrow, patch, translucent, segments } from './view3d.js';
import { Plot, token, grad } from '../plot.js';
import { $, fmt, put, range, dot } from '../ui.js';

const f = (x, y) =>
  2.2 * Math.exp(-((x - 1) ** 2 + (y - 1.1) ** 2) / 0.9) +
  1.6 * Math.exp(-((x + 1.4) ** 2 + (y + 1.3) ** 2) / 0.7);
const trail = (t, c) => [t, c + 0.6 * Math.sin(1.4 * t)];
const G = (x, y) => y - 0.6 * Math.sin(1.4 * x);
const T0 = -3.2, T1 = 3.2, BOX = [-3.2, 3.2, -3.2, 3.2];

// the best point on the trail g = c, by dense sampling plus golden-section refinement
function best(c) {
  let bt = T0, bf = -Infinity;
  for (let i = 0; i <= 800; i++) {
    const t = T0 + (T1 - T0) * i / 800, v = f(...trail(t, c));
    if (v > bf) { bf = v; bt = t; }
  }
  let a = Math.max(T0, bt - 0.01), b = Math.min(T1, bt + 0.01);
  const gr = (Math.sqrt(5) - 1) / 2;
  for (let i = 0; i < 60; i++) {
    const p = b - gr * (b - a), q = a + gr * (b - a);
    if (f(...trail(p, c)) > f(...trail(q, c))) b = q; else a = p;
  }
  const t = (a + b) / 2, [x, y] = trail(t, c), gf = grad(f, x, y), gg = grad(G, x, y);
  return { t, x, y, f: f(x, y), lam: dot(gf, gg) / dot(gg, gg), gf, gg };
}

export function initMountain() {
  const view = new View3D($('#mtn-3d'), { cam: [5.2, -8.6, 3.9], target: [0, 0, 0.55] });
  const graph = new Plot($('#mtn-graph'), { view: [-1.6, 1.6, 0, 2.5], equal: false });
  const s = { c: 0, t: -2.2 };
  const cs = [], fs = [];
  for (let i = 0; i <= 160; i++) { const c = -1.6 + 3.2 * i / 160; cs.push(c); fs.push(best(c).f); }

  // static: the mountain and its contour lines
  const lo = col('--surf-lo'), hi = col('--surf-hi');
  const surfMat = new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.9, side: THREE.DoubleSide, transparent: true, opacity: 0.62, depthWrite: false });
  let mesh = surface(f, BOX, 110, surfMat, z => lo.clone().lerp(hi, Math.min(1, z / 2.2)));
  view.scene.add(mesh);
  view.onTheme = () => {
    const a = col('--surf-lo'), b = col('--surf-hi'), cattr = mesh.geometry.getAttribute('color'), pos = mesh.geometry.getAttribute('position');
    for (let k = 0; k < pos.count; k++) { const c = a.clone().lerp(b, Math.min(1, pos.getZ(k) / 2.2)); cattr.setXYZ(k, c.r, c.g, c.b); }
    cattr.needsUpdate = true;
  };
  const contour = [];
  const N = 140, h = 6.4 / N;
  for (let L = 0.2; L < 2.2; L += 0.2) {
    for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) {
      const x = -3.2 + i * h, y = -3.2 + j * h;
      const v = [f(x, y), f(x + h, y), f(x + h, y + h), f(x, y + h)].map(q => q - L);
      const P = [[x, y], [x + h, y], [x + h, y + h], [x, y + h]], pts = [];
      for (let k = 0; k < 4; k++) {
        const a = v[k], b = v[(k + 1) % 4];
        if ((a > 0) !== (b > 0)) { const r = a / (a - b), A = P[k], B = P[(k + 1) % 4]; pts.push([A[0] + r * (B[0] - A[0]), A[1] + r * (B[1] - A[1]), L + 0.012]); }
      }
      if (pts.length >= 2) contour.push([...pts[0], ...pts[1]]);
    }
  }
  const cl = segments(contour, view.mat('--contour', { kind: 'line', transparent: true, opacity: 0.7 }));
  view.scene.add(cl);
  $('#mtn-hide').addEventListener('change', e => { mesh.visible = cl.visible = !e.target.checked; view.request(); });
  const ground = patch([0, 0, -0.004], [0, 0, 1], 6.4, view.mat('--grid', { side: THREE.DoubleSide }));
  view.scene.add(ground);
  const peak = ball([1, 1.1, f(1, 1.1)], 0.07, view.mat('--muted'));
  const peakLbl = view.label('פסגה', 'muted'); peakLbl.position.set(1, 1.1, f(1, 1.1) + 0.3);
  view.scene.add(peak, peakLbl);

  const dyn = view.layer();
  const mOrange = view.mat('--c-g'), mBlue = view.mat('--c-f'), mInk = view.mat('--ink'), mLam = view.mat('--c-lam');
  const mCurtain = translucent(view, '--c-g', 0.42), mLevel = translucent(view, '--c-f', 0.1);

  function build() {
    dyn.reset();
    const pts = [], top = [], wall = [];
    for (let i = 0; i <= 200; i++) {
      const t = T0 + (T1 - T0) * i / 200, [x, y] = trail(t, s.c);
      if (Math.abs(y) > 3.2) continue;
      pts.push([x, y, 0.01]); top.push([x, y, f(x, y) + 0.015]);
    }
    for (let i = 0; i + 1 < pts.length; i++) {
      const a = pts[i], b = pts[i + 1], A = top[i], B = top[i + 1];
      wall.push(...a, ...b, ...B, ...a, ...B, ...A);
    }
    const wg = new THREE.BufferGeometry();
    wg.setAttribute('position', new THREE.Float32BufferAttribute(wall, 3));
    dyn.add(new THREE.Mesh(wg, mCurtain));
    dyn.add(tube(pts, 0.03, mOrange), tube(top, 0.04, mOrange));

    const o = best(s.c);
    dyn.add(ball([o.x, o.y, o.f], 0.1, mInk));
    dyn.add(patch([o.x, o.y, o.f], [0, 0, 1], 6.4, mLevel));
    dyn.add(segments([[o.x, o.y, 0, o.x, o.y, o.f]], view.mat('--ink', { kind: 'line' })));
    const sc = 0.9;
    dyn.add(arrow([o.x, o.y, 0.02], [o.gg[0] / Math.hypot(...o.gg) * sc * 0.8, o.gg[1] / Math.hypot(...o.gg) * sc * 0.8, 0], mOrange, 0.03));
    if (Math.hypot(...o.gf) > 1e-3) dyn.add(arrow([o.x, o.y, 0.02], [o.gf[0] / Math.hypot(...o.gf) * sc, o.gf[1] / Math.hypot(...o.gf) * sc, 0], mBlue, 0.022));
    const lbl = view.label(`f* = ${fmt(o.f, 3)}`, 'f'); lbl.position.set(o.x, o.y, o.f + 0.35);
    dyn.add(lbl);

    const [wx, wy] = trail(s.t, s.c);
    if (Math.abs(wy) <= 3.2) {
      dyn.add(ball([wx, wy, f(wx, wy) + 0.02], 0.075, mLam));
      dyn.add(segments([[wx, wy, 0, wx, wy, f(wx, wy)]], view.mat('--c-lam', { kind: 'line' })));
    }
    view.request();

    put('mtn-f', fmt(o.f, 3));
    put('mtn-lam', fmt(o.lam, 3));
    const d = 0.01, der = (best(s.c + d).f - best(s.c - d).f) / (2 * d);
    put('mtn-der', fmt(der, 3));
    put('mtn-walk', Math.abs(wy) <= 3.2 ? fmt(f(wx, wy), 3) : '?');
    graph.invalidate();
  }

  graph.draw = p => {
    p.clear();
    p.grid({});
    p.path(cs.map((c, i) => [c, fs[i]]), { stroke: token('--c-f'), width: 2.5 });
    const o = best(s.c), d = 0.45;
    p.path([[s.c - d, o.f - o.lam * d], [s.c + d, o.f + o.lam * d]], { stroke: token('--c-lam'), width: 2.5, dash: [6, 4] });
    p.dot(s.c, o.f, { r: 6.5, fill: token('--ink'), stroke: token('--plot-bg') });
    p.text('c', p.x1, p.y0, { align: 'right', base: 'bottom', dx: -8, dy: -16, color: token('--muted') });
    p.text('f*(c)', p.x1, p.y1, { align: 'right', base: 'top', dx: -8, dy: 6, color: token('--muted') });
  };

  range('mtn-c', v => { s.c = v; build(); }, v => fmt(v, 2));
  range('mtn-t', v => { s.t = v; build(); }, v => fmt(v, 2));
  build();
}
