// Labs 2–4: what the multiplier means. Fence budget (price), bead on a wire (force), dice entropy (temperature).
import { Plot, token } from '../plot.js';
import { $, fmt, fmtt, put, range, norm } from '../ui.js';

/* ---------------- Lab 2: garden fence ---------------- */
export function initFence() {
  const plane = new Plot($('#fence-plane'), { view: [0, 1, 0, 1] });
  const value = new Plot($('#fence-value'), { view: [0, 220, 0, 1], equal: false });
  const s = { B: 120, px: 4, py: 2 };
  const DB = 20, BMAX = 220;
  const sol = (B = s.B) => {
    const x = B / (2 * s.px), y = B / (2 * s.py);
    return { x, y, f: x * y, lam: B / (2 * s.px * s.py) };
  };

  function update() {
    const o = sol(), o2 = sol(s.B + DB);
    const M = 1.12 * Math.max(s.B / s.px, s.B / s.py);
    plane.setView([-0.05 * M, M, -0.05 * M, M]);
    value.setView([0, BMAX, 0, sol(BMAX).f * 1.06]);
    put('fence-x', fmt(o.x, 2));
    put('fence-y', fmt(o.y, 2));
    put('fence-area', fmt(o.f, 1));
    put('fence-lam', fmt(o.lam, 3));
    put('fence-r1', fmt(o.y / s.px, 3));
    put('fence-r2', fmt(o.x / s.py, 3));
    put('fence-pred', '+' + fmt(o.lam * DB, 1));
    put('fence-act', '+' + fmt(o2.f - o.f, 1));
  }

  plane.draw = p => {
    p.clear();
    p.grid({});
    const o = sol(), M = p.x1;
    p.path([[0, 0], [o.x, 0], [o.x, o.y], [0, o.y]], { fill: token('--c-feas'), stroke: token('--c-feas-line'), width: 1.5, close: true });
    p.text(`שטח ${fmt(o.f, 0)}`, o.x / 2, o.y / 2, { align: 'center', color: token('--c-f'), font: '600 13px system-ui' });
    for (const k of [0.25, 0.5, 1.6, 2.6]) {
      const L = o.f * k;
      p.fn(x => L / x, L / p.y1, p.x1, { stroke: token('--contour'), width: 1, alpha: .8 });
    }
    p.fn(x => o.f / x, o.f / p.y1, p.x1, { stroke: token('--c-f'), width: 2.2 });
    p.path([[0, s.B / s.py], [s.B / s.px, 0]], { stroke: token('--c-g'), width: 3.5 });
    const L = 0.2 * M, gf = [o.y, o.x], gg = [s.px, s.py];
    p.arrow(o.x, o.y, gg[0] / norm(gg) * L * 0.75, gg[1] / norm(gg) * L * 0.75, { color: token('--c-g'), width: 6, head: 13 });
    p.arrow(o.x, o.y, gf[0] / norm(gf) * L, gf[1] / norm(gf) * L, { color: token('--c-f'), width: 2.5, label: '∇f ∥ ∇g' });
    p.dot(o.x, o.y, { r: 6.5, fill: token('--ink'), stroke: token('--plot-bg') });
    p.text('x', p.x1, 0, { align: 'right', base: 'bottom', dx: -6, dy: -4, color: token('--muted') });
    p.text('y', 0, p.y1, { align: 'left', base: 'top', dx: 6, dy: 4, color: token('--muted') });
  };

  value.draw = p => {
    p.clear();
    p.grid({});
    const o = sol(), o2 = sol(s.B + DB);
    p.fn(B => sol(B).f, 0, BMAX, { stroke: token('--c-f'), width: 2.5 });
    const d = 55;
    p.path([[s.B - d, o.f - o.lam * d], [s.B + d, o.f + o.lam * d]], { stroke: token('--c-lam'), width: 2, dash: [6, 4] });
    p.path([[s.B, o.f], [s.B + DB, o.f]], { stroke: token('--muted'), width: 1.5 });
    p.path([[s.B + DB, o.f], [s.B + DB, o.f + o.lam * DB]], { stroke: token('--c-lam'), width: 4 });
    p.dot(s.B + DB, o2.f, { r: 4.5, fill: token('--c-f') });
    p.dot(s.B, o.f, { r: 6.5, fill: token('--ink'), stroke: token('--plot-bg') });
    p.text('תקציב B (₪)', p.x1, p.y0, { align: 'right', base: 'bottom', dx: -8, dy: -16, color: token('--muted') });
    p.text('השטח המרבי', p.x1, p.y1, { align: 'right', base: 'top', dx: -8, dy: 6, color: token('--muted') });
  };

  range('fence-B', v => { s.B = v; update(); }, v => v + ' ₪');
  range('fence-px', v => { s.px = v; update(); }, v => v + ' ₪');
  range('fence-py', v => { s.py = v; update(); }, v => v + ' ₪');
  update();
}

/* ---------------- Lab 3: bead on a circular wire ---------------- */
export function initBead() {
  const plot = new Plot($('#bead-plane'), { view: [-3, 3, -3, 3] });
  const s = { R: 1.6, k: 1.5, mg: 1, anchor: [0.5, 0.9], th: -2.4, om: 0 };
  const U = (th, R = s.R) =>
    s.mg * R * Math.sin(th) + 0.5 * s.k * ((s.anchor[0] - R * Math.cos(th)) ** 2 + (s.anchor[1] - R * Math.sin(th)) ** 2);
  const dU = th => (U(th + 1e-5) - U(th - 1e-5)) / 2e-5;
  let running = false;

  function localMin(R, th0) {
    let a = th0 - 0.5, b = th0 + 0.5;
    const gr = (Math.sqrt(5) - 1) / 2;
    for (let i = 0; i < 80; i++) {
      const c = b - gr * (b - a), d = a + gr * (b - a);
      if (U(c, R) < U(d, R)) b = d; else a = c;
    }
    return U((a + b) / 2, R);
  }

  function forces() {
    const n = [Math.cos(s.th), Math.sin(s.th)], p = [s.R * n[0], s.R * n[1]];
    const Fg = [0, -s.mg], Fs = [s.k * (s.anchor[0] - p[0]), s.k * (s.anchor[1] - p[1])];
    const F = [Fg[0] + Fs[0], Fg[1] + Fs[1]];
    const lam = -(F[0] * n[0] + F[1] * n[1]);
    return { n, p, Fg, Fs, lam, N: [lam * n[0], lam * n[1]] };
  }

  function readouts() {
    const f = forces(), settled = Math.abs(s.om) < 2e-3 && Math.abs(dU(s.th)) < 2e-3;
    put('bead-lam', fmt(f.lam, 3));
    put('bead-dir', f.lam > 0.005 ? 'דוחף החוצה' : f.lam < -0.005 ? 'מושך פנימה' : 'כמעט לא עובד');
    put('bead-state', settled ? 'שיווי משקל' : 'בתנועה');
    const dR = 1e-3, der = (localMin(s.R + dR, s.th) - localMin(s.R - dR, s.th)) / (2 * dR);
    put('bead-der', settled ? fmt(der, 3) : '…');
    plot.invalidate();
    return settled;
  }

  function tick() {
    for (let i = 0; i < 40; i++) {
      s.om += (-dU(s.th) / (s.R * s.R) - 1.4 * s.om) * 0.004;
      s.th += s.om * 0.004;
    }
    running = !readouts();
    if (running) requestAnimationFrame(tick);
  }
  const kick = () => { if (!running) { running = true; requestAnimationFrame(tick); } };

  plot.draw = p => {
    p.clear();
    p.grid({ labels: false });
    const circ = [];
    for (let i = 0; i <= 200; i++) circ.push([s.R * Math.cos(i / 200 * 2 * Math.PI), s.R * Math.sin(i / 200 * 2 * Math.PI)]);
    p.path(circ, { stroke: token('--c-g'), width: 4 });
    p.dot(0, 0, { r: 3, fill: token('--muted') });
    const f = forces(), [bx, by] = f.p, [ax, ay] = s.anchor;
    const L = Math.hypot(ax - bx, ay - by), ux = (ax - bx) / L, uy = (ay - by) / L, zig = [];
    for (let i = 0; i <= 24; i++) {
      const t = i / 24, off = i === 0 || i === 24 ? 0 : (i % 2 ? 0.08 : -0.08);
      zig.push([bx + ux * L * t - uy * off, by + uy * L * t + ux * off]);
    }
    p.path(zig, { stroke: token('--muted'), width: 1.5 });
    p.dot(ax, ay, { r: 8, fill: token('--plot-bg'), stroke: token('--c-tan'), width: 3 });
    const sc = 0.6;
    p.arrow(bx, by, f.Fg[0] * sc, f.Fg[1] * sc, { color: token('--muted'), label: 'mg' });
    p.arrow(bx, by, f.Fs[0] * sc, f.Fs[1] * sc, { color: token('--c-tan'), label: 'קפיץ' });
    p.arrow(bx, by, f.N[0] * sc, f.N[1] * sc, { color: token('--c-lam'), width: 3.5, label: 'λ∇g' });
    p.dot(bx, by, { r: 8, fill: token('--ink'), stroke: token('--plot-bg') });
  };

  plot.onDrag({
    move: (x, y) => {
      s.anchor = [Math.max(-2.9, Math.min(2.9, x)), Math.max(-2.9, Math.min(2.9, y))];
      kick();
    },
  });
  range('bead-R', v => { s.R = v; kick(); }, v => fmtt(v, 2));
  range('bead-k', v => { s.k = v; kick(); }, v => fmtt(v, 2));
  range('bead-mg', v => { s.mg = v; kick(); }, v => fmtt(v, 2));
  kick();
}

/* ---------------- Lab 4: maximum entropy die ---------------- */
export function initEntropy() {
  const bars = new Plot($('#ent-bars'), { view: [0.4, 6.6, 0, 1], equal: false });
  const curve = new Plot($('#ent-curve'), { view: [1, 6, -0.08, 1.95], equal: false });
  const faces = [1, 2, 3, 4, 5, 6];
  let m = 4.5;

  const dist = beta => {
    const lo = Math.min(...faces.map(i => beta * i));
    const w = faces.map(i => Math.exp(-(beta * i - lo))), Z = w.reduce((a, b) => a + b);
    return w.map(v => v / Z);
  };
  const mean = p => p.reduce((a, v, i) => a + v * (i + 1), 0);
  const H = p => -p.reduce((a, v) => a + (v > 0 ? v * Math.log(v) : 0), 0);
  function solve(mm) {
    let a = -60, b = 60;
    for (let i = 0; i < 100; i++) { const c = (a + b) / 2; if (mean(dist(c)) > mm) a = c; else b = c; }
    const beta = (a + b) / 2, p = dist(beta);
    return { beta, p, H: H(p) };
  }
  const curvePts = [];
  for (let i = 0; i <= 200; i++) { const mm = 1.01 + 4.98 * i / 200; curvePts.push([mm, solve(mm).H]); }

  function update() {
    const o = solve(m);
    put('ent-beta', fmt(o.beta, 3));
    put('ent-H', fmt(o.H, 3));
    put('ent-T', Math.abs(o.beta) < 0.002 ? '∞' : fmt(1 / o.beta, 2));
    bars.setView([0.4, 6.6, 0, Math.max(0.42, Math.max(...o.p) * 1.15)]);
    curve.invalidate();
  }

  bars.draw = p => {
    p.clear();
    p.grid({ axes: false, stepX: 1 });
    const o = solve(m);
    p.path([[0.4, 1 / 6], [6.6, 1 / 6]], { stroke: token('--muted'), width: 1, dash: [4, 4] });
    o.p.forEach((v, i) => {
      const x = i + 1;
      p.path([[x - .32, 0], [x + .32, 0], [x + .32, v], [x - .32, v]], { fill: token('--c-f'), close: true, alpha: .85 });
      p.text(fmt(v, 3), x, v, { align: 'center', base: 'bottom', dy: -3, color: token('--ink'), font: '11px system-ui' });
    });
    p.text('הסתברות לכל פאה', p.x1, p.y1, { align: 'right', base: 'top', dx: -8, dy: 6, color: token('--muted') });
  };

  curve.draw = p => {
    p.clear();
    p.grid({ axes: false });
    p.path(curvePts, { stroke: token('--c-f'), width: 2.5 });
    const o = solve(m), d = 0.9;
    p.path([[m - d, o.H - o.beta * d], [m + d, o.H + o.beta * d]], { stroke: token('--c-lam'), width: 2.5, dash: [6, 4] });
    p.dot(m, o.H, { r: 6.5, fill: token('--ink'), stroke: token('--plot-bg') });
    p.text('ממוצע m', p.x1, p.y0, { align: 'right', base: 'bottom', dx: -8, dy: -16, color: token('--muted') });
    p.text('האנטרופיה המרבית', p.x1, p.y1, { align: 'right', base: 'top', dx: -8, dy: 6, color: token('--muted') });
  };

  range('ent-m', v => { m = v; update(); }, v => fmt(v, 2));
  update();
}
