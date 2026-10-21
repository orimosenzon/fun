// Lab 1: drag a point along the constraint g = c and watch ∇f against ∇g.
import { Plot, token, grad, levelsBetween } from '../plot.js';
import { $, fmt, put, segmented, dot, norm } from '../ui.js';

const TAU = 2 * Math.PI;
const hills = (x, y) =>
  2.2 * Math.exp(-((x - 1) ** 2 + (y - 1.1) ** 2) / 0.9) +
  1.6 * Math.exp(-((x + 1.4) ** 2 + (y + 1.3) ** 2) / 0.7);
const RA = [[1.5, 1], [1, -0.5]];

const PRESETS = {
  circle: {
    f: (x, y) => x + 2 * y, g: (x, y) => x * x + y * y,
    curve: t => [2 * Math.cos(t), 2 * Math.sin(t)], range: [0, TAU], closed: true,
    view: [-3, 3, -3, 3], t: 2.6,
  },
  ellipse: {
    f: (x, y) => x * y, g: (x, y) => x * x / 4 + y * y,
    curve: t => [2 * Math.cos(t), Math.sin(t)], range: [0, TAU], closed: true,
    view: [-3, 3, -2.4, 2.4], t: 0.2,
  },
  trail: {
    f: hills, g: (x, y) => y - 0.6 * Math.sin(1.4 * x),
    curve: t => [t, 0.6 * Math.sin(1.4 * t)], range: [-3.2, 3.2], closed: false,
    view: [-3.2, 3.2, -2.6, 2.6], t: -2.6,
  },
  rayleigh: {
    f: (x, y) => RA[0][0] * x * x + 2 * RA[0][1] * x * y + RA[1][1] * y * y, g: (x, y) => x * x + y * y,
    curve: t => [Math.cos(t), Math.sin(t)], range: [0, TAU], closed: true,
    view: [-1.9, 1.9, -1.9, 1.9], t: 2.2,
  },
};

export function initGeo() {
  const plane = new Plot($('#geo-plane'), { view: PRESETS.circle.view });
  const trace = new Plot($('#geo-trace'), { view: [0, 1, 0, 1], equal: false });
  let key, P, t, samples, crits, sf, sg;

  const phi = tt => { const [x, y] = P.curve(tt); return P.f(x, y); };
  const dphi = tt => (phi(tt + 1e-5) - phi(tt - 1e-5)) / 2e-5;

  function findCrits() {
    const out = [];
    for (let i = 0; i + 1 < samples.length; i++) {
      let a = samples[i].t, b = samples[i + 1].t, da = dphi(a), db = dphi(b);
      if (da === 0 || (da > 0) === (db > 0)) continue;
      const rising = da > 0;
      for (let k = 0; k < 50; k++) {
        const m = (a + b) / 2, dm = dphi(m);
        if ((dm > 0) === (da > 0)) { a = m; da = dm; } else b = m;
      }
      const tc = (a + b) / 2;
      if (out.some(c => Math.abs(c.t - tc) < 1e-4)) continue;
      const [x, y] = P.curve(tc);
      out.push({ t: tc, x, y, max: rising });
    }
    return out;
  }

  function setPreset(k) {
    key = k; P = PRESETS[k]; t = P.t;
    plane.setView(P.view);
    const [a, b] = P.range, N = 1500;
    samples = [];
    for (let i = 0; i <= N; i++) {
      const tt = a + (b - a) * i / N, [x, y] = P.curve(tt);
      samples.push({ t: tt, x, y, f: P.f(x, y) });
    }
    crits = findCrits();
    const fs = samples.map(s => s.f), lo = Math.min(...fs), hi = Math.max(...fs), pad = (hi - lo) * 0.14 || 1;
    trace.setView([a, b, lo - pad, hi + pad]);
    let mf = 0, mg = 0;
    for (const s of samples) {
      mf = Math.max(mf, norm(grad(P.f, s.x, s.y)));
      mg = Math.max(mg, norm(grad(P.g, s.x, s.y)));
    }
    const W = P.view[1] - P.view[0];
    sf = 0.24 * W / mf; sg = 0.17 * W / mg;
    document.querySelectorAll('#lab-geo [data-preset]').forEach(el => { el.hidden = el.dataset.preset !== k; });
    update();
  }

  function nearestT(x, y) {
    let best = samples[0], bd = Infinity;
    for (const s of samples) {
      const d = (s.x - x) ** 2 + (s.y - y) ** 2;
      if (d < bd) { bd = d; best = s; }
    }
    return best.t;
  }

  function state() {
    const [x, y] = P.curve(t), gf = grad(P.f, x, y), gg = grad(P.g, x, y);
    const lam = dot(gf, gg) / dot(gg, gg);
    const par = [lam * gg[0], lam * gg[1]], perp = [gf[0] - par[0], gf[1] - par[1]];
    return { x, y, gf, gg, lam, par, perp, rel: norm(perp) / (norm(gf) || 1) };
  }

  function update() {
    const s = state();
    put('geo-f', fmt(P.f(s.x, s.y), 3));
    put('geo-lam', fmt(s.lam, 3));
    put('geo-tan', fmt(norm(s.perp), 3));
    const ang = Math.acos(Math.min(1, Math.abs(dot(s.gf, s.gg)) / (norm(s.gf) * norm(s.gg) || 1))) * 180 / Math.PI;
    put('geo-angle', fmt(ang, 1) + '°');
    const st = $('#geo-status');
    if (s.rel < 0.015) {
      const c = crits.reduce((a, b) => Math.abs(b.t - t) < Math.abs(a.t - t) ? b : a, crits[0]);
      const kind = c && Math.abs(c.t - t) < 0.05 ? (c.max ? 'מקסימום מקומי' : 'מינימום מקומי') : 'נקודה קריטית';
      st.className = 'status ok';
      st.textContent = `החצים מקבילים: זו נקודה קריטית (${kind}), והכופל כאן הוא λ = ${fmt(s.lam, 3)}.`;
    } else {
      st.className = 'status';
      st.textContent = 'החץ הירוק עדיין לא התאפס: אפשר לזוז לאורך האילוץ בכיוון שלו ולהעלות את f. כאן עוד לא מקסימום.';
    }
    plane.invalidate(); trace.invalidate();
  }

  plane.draw = p => {
    p.clear();
    const F = p.field(key, P.f);
    p.heat(F);
    p.grid({ labels: true });
    p.contours(F, levelsBetween(F.min, F.max, 16), { color: token('--contour'), width: 1, alpha: .75 });
    const s = state();
    p.contours(F, [P.f(s.x, s.y)], { color: token('--c-f'), width: 2.2 });
    p.path(samples.map(q => [q.x, q.y]), { stroke: token('--c-g'), width: 4, close: P.closed });
    for (const c of crits) p.dot(c.x, c.y, { r: 4, fill: token('--plot-bg'), stroke: token('--ink'), width: 1.5 });
    const tl = [-s.gg[1], s.gg[0]];
    p.infLine(s.x, s.y, tl[0], tl[1], { stroke: token('--muted'), width: 1, dash: [4, 4] });
    p.arrow(s.x, s.y, s.gg[0] * sg, s.gg[1] * sg, { color: token('--c-g'), label: '∇g' });
    p.arrow(s.x, s.y, s.par[0] * sf, s.par[1] * sf, { color: token('--c-lam'), width: 2, dash: [5, 4], label: 'λ∇g' });
    p.arrow(s.x, s.y, s.perp[0] * sf, s.perp[1] * sf, { color: token('--c-tan'), width: 3 });
    p.arrow(s.x, s.y, s.gf[0] * sf, s.gf[1] * sf, { color: token('--c-f'), width: 3, label: '∇f' });
    p.dot(s.x, s.y, { r: 7, fill: token('--ink'), stroke: token('--plot-bg'), width: 2 });
  };

  trace.draw = p => {
    p.clear();
    p.grid({ axes: false, labels: true });
    for (const c of crits) {
      p.path([[c.t, p.y0], [c.t, p.y1]], { stroke: token('--line'), width: 1 });
      p.dot(c.t, phi(c.t), { r: 4, fill: token('--plot-bg'), stroke: token('--ink'), width: 1.5 });
    }
    p.path(samples.map(q => [q.t, q.f]), { stroke: token('--c-f'), width: 2.5 });
    const ft = phi(t), slope = dphi(t), d = (p.x1 - p.x0) * 0.08;
    p.path([[t - d, ft - slope * d], [t + d, ft + slope * d]], { stroke: token('--c-tan'), width: 2.5 });
    p.dot(t, ft, { r: 6, fill: token('--ink'), stroke: token('--plot-bg'), width: 2 });
    p.text('f לאורך האילוץ', p.x1, p.y1, { align: 'right', base: 'top', dx: -8, dy: 6, color: token('--muted') });
  };

  plane.onDrag({ move: (x, y) => { t = nearestT(x, y); update(); } });
  trace.onDrag({ move: x => { t = Math.min(P.range[1], Math.max(P.range[0], x)); update(); } });

  $('#geo-next').addEventListener('click', () => {
    if (!crits.length) return;
    const span = P.range[1] - P.range[0];
    const ahead = c => ((c.t - t - 1e-6) % span + span) % span;
    const next = P.closed ? crits.reduce((a, b) => ahead(b) < ahead(a) ? b : a)
      : (crits.find(c => c.t > t + 1e-6) || crits[0]);
    t = next.t; update();
  });

  segmented('geo-presets', setPreset)('circle');
}
