// Lab 6: the geometry of duality. The set S = {(g(x), f(x))}, supporting lines of slope −λ, and q(λ).
import { Plot, token } from '../plot.js';
import { $, fmt, put, range, segmented } from '../ui.js';

const PRESETS = {
  convex: { f: x => x * x, g: x => 1 - x, dom: [-3, 3], viewS: [-2.6, 4.4, -1.5, 9.5], viewQ: [0, 4, -0.6, 1.5] },
  gap: { f: x => 4 - x * x, g: x => x, dom: [-1, 2], viewS: [-1.7, 2.7, -1.5, 5.2], viewQ: [0, 4, -1.4, 3.6] },
};
const LMAX = 4;

export function initDuality() {
  const sPlot = new Plot($('#dual-s'), { view: [0, 1, 0, 1], equal: false });
  const qPlot = new Plot($('#dual-qplot'), { view: [0, 1, 0, 1], equal: false });
  let P, pts, pstar, pstarU, valueFn, lam = 0.6, best = { lam: 0, q: 0 };

  const q = l => {
    let m = Infinity, at = null;
    for (const [u, t] of pts) { const v = t + l * u; if (v < m) { m = v; at = [u, t]; } }
    return { q: m, at };
  };

  function setPreset(k) {
    P = PRESETS[k];
    pts = [];
    for (let i = 0; i <= 1200; i++) { const x = P.dom[0] + (P.dom[1] - P.dom[0]) * i / 1200; pts.push([P.g(x), P.f(x)]); }
    pstar = Infinity;
    for (const [u, t] of pts) if (u <= 1e-9 && t < pstar) { pstar = t; pstarU = u; }
    const sorted = [...pts].sort((a, b) => a[0] - b[0]);
    valueFn = []; let run = Infinity;
    for (const [u, t] of sorted) { run = Math.min(run, t); valueFn.push([u, run]); }
    best = { lam: 0, q: -Infinity };
    for (let i = 0; i <= 4000; i++) { const l = LMAX * i / 4000, v = q(l).q; if (v > best.q) best = { lam: l, q: v }; }
    sPlot.setView(P.viewS); qPlot.setView(P.viewQ);
    document.querySelectorAll('#lab-dual [data-preset]').forEach(el => { el.hidden = el.dataset.preset !== k; });
    update();
  }

  function update() {
    const r = q(lam);
    put('dual-q', fmt(r.q, 3));
    put('dual-p', fmt(pstar, 3));
    put('dual-d', fmt(best.q, 3) + '  (λ* = ' + fmt(best.lam, 2) + ')');
    put('dual-gap', fmt(pstar - best.q, 3));
    sPlot.invalidate(); qPlot.invalidate();
  }

  sPlot.draw = p => {
    p.clear();
    p.path([[p.x0, p.y0], [0, p.y0], [0, p.y1], [p.x0, p.y1]], { fill: token('--c-feas'), close: true });
    p.grid({});
    p.path(valueFn.filter(([u]) => u <= p.x1), { stroke: token('--c-tan'), width: 2, dash: [5, 4] });
    p.path(pts, { stroke: token('--c-f'), width: 3.5 });
    const r = q(lam);
    p.path([[p.x0, r.q - lam * p.x0], [p.x1, r.q - lam * p.x1]], { stroke: token('--c-lam'), width: 2.5 });
    p.dot(r.at[0], r.at[1], { r: 5, fill: token('--c-lam') });
    p.path([[pstarU, pstar], [0, pstar]], { stroke: token('--c-f'), width: 1, dash: [3, 3] });
    p.dot(pstarU, pstar, { r: 5.5, fill: token('--c-f') });
    p.dot(0, pstar, { r: 4, fill: token('--plot-bg'), stroke: token('--c-f'), width: 2 });
    p.text('p*', 0, pstar, { dx: 8, dy: -10, color: token('--c-f'), font: '600 13px system-ui' });
    p.dot(0, r.q, { r: 6, fill: token('--plot-bg'), stroke: token('--c-lam'), width: 3 });
    p.text('q(λ)', 0, r.q, { dx: 9, dy: 12, color: token('--c-lam'), font: '600 13px system-ui' });
    p.text('u = g(x)', p.x1, p.y0, { align: 'right', base: 'bottom', dx: -8, dy: -16, color: token('--muted') });
    p.text('t = f(x)', p.x1, p.y1, { align: 'right', base: 'top', dx: -8, dy: 6, color: token('--muted') });
  };

  qPlot.draw = p => {
    p.clear();
    p.grid({});
    p.path([[p.x0, pstar], [p.x1, pstar]], { stroke: token('--c-f'), width: 1.5, dash: [6, 4] });
    p.text('p*', p.x1, pstar, { align: 'right', dx: -6, dy: -10, color: token('--c-f'), font: '600 13px system-ui' });
    p.fn(l => q(l).q, 0, LMAX, { stroke: token('--c-lam'), width: 2.5 }, 300);
    p.dot(best.lam, best.q, { r: 4, fill: token('--plot-bg'), stroke: token('--c-lam'), width: 2 });
    p.dot(lam, q(lam).q, { r: 6.5, fill: token('--ink'), stroke: token('--plot-bg') });
    p.text('λ', p.x1, p.y0, { align: 'right', base: 'bottom', dx: -8, dy: -16, color: token('--muted') });
    p.text('q(λ)', p.x1, p.y1, { align: 'right', base: 'top', dx: -8, dy: 24, color: token('--muted') });
  };

  const slider = range('dual-lam', v => { lam = v; update(); }, v => fmt(v, 2));
  qPlot.onDrag({ move: x => { lam = Math.max(0, Math.min(LMAX, x)); slider.set(lam); update(); } });
  $('#dual-best').addEventListener('click', () => { lam = best.lam; slider.set(lam); update(); });
  segmented('dual-presets', setPreset)('convex');
}
