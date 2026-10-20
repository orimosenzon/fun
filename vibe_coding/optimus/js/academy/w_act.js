// ווידג'טים של המפעילים: יחס תמסורת, בורג גלגלות פלנטרי, ומעטפת מומנט-מהירות עם נקודות מהסימולטור.
import { LinePlot, SERIES, setupCanvas } from './plot.js';
import { slider, readout, buttons, f, animate } from './ui.js';
import { JOINTS } from '../robot/spec.js';
import { torqueLimit } from '../ctrl/actuators.js';
import { OPPOINTS } from './oppoints.js';

// ================= יחס תמסורת =================
export function gear(root) {
  const cv = root.querySelector('canvas.plot'), ctrls = root.querySelector('.ctrls');
  const MOTOR = { T: 2.0, w: 6000 * 2 * Math.PI / 60, J: 1.2e-4, eta: 0.85 };
  const plot = new LinePlot(cv, { xLabel: 'יחס תמסורת N', yLabel: '% מהערך המרבי בגרף', xMin: 1, xMax: 160, yMin: 0, yMax: 100, xDec: 0, yDec: 0, unitY: '%' });
  let N = 50;
  const ro = readout(ctrls, [['T', 'מומנט ביציאה'], ['w', 'מהירות מרבית'], ['J', 'אינרציה מוחזרת'], ['F', 'כוח מכה בנחיתה']]);
  slider(ctrls, { label: 'יחס התמסורת', min: 1, max: 160, step: 1, value: N, fmt: (v) => `1:${v}`, onInput: (v) => { N = v; draw(); } });
  // מכה: רגל של 2 ק"ג על זרוע של 40 ס"מ פוגעת ברצפה ב-0.5 מ'/ש'. מגע קפיצי k=2·10⁵ נ'/מ'
  const impact = (n) => { const meff = 2 + MOTOR.J * n * n / (0.4 * 0.4); return 0.5 * Math.sqrt(2e5 * meff); };
  const tq = (n) => MOTOR.T * n * MOTOR.eta, sp = (n) => MOTOR.w / n, ji = (n) => MOTOR.J * n * n;
  const xs = Array.from({ length: 160 }, (_, i) => i + 1);
  const norm = (fn) => { const mx = fn(160) > fn(1) ? fn(160) : fn(1); return xs.map(x => [x, 100 * fn(x) / mx]); };
  function draw() {
    plot.vlines = [{ x: N, label: `1:${N}` }];
    plot.set([
      { name: 'מומנט', data: norm(tq) },
      { name: 'מהירות', data: norm(sp) },
      { name: 'אינרציה מוחזרת', data: norm(ji) },
      { name: 'כוח מכה', data: norm(impact), dash: [5, 4] },
    ]);
    ro.set({ T: `${f(tq(N), 0)} נ"מ`, w: `${f(sp(N), 1)} רד'/ש' (${f(sp(N) * 60 / 2 / Math.PI, 0)} סל"ד)`, J: `${f(ji(N), 3)} ק"ג·מ"ר`, F: `${f(impact(N), 0)} ניוטון` });
  }
  draw();
}

// ================= בורג גלגלות פלנטרי =================
export function screw(root) {
  const cv = root.querySelector('canvas.anim'), ctrls = root.querySelector('.ctrls');
  const st = { lead: 4, torque: 1.6, rpm: 5000, arm: 0.045 };
  slider(ctrls, { label: 'עלייה (lead)', min: 1, max: 10, step: 0.5, value: st.lead, fmt: (v) => `${v} מ"מ לסיבוב`, onInput: (v) => { st.lead = v; upd(); } });
  slider(ctrls, { label: 'מומנט מנוע', min: 0.4, max: 3, step: 0.1, value: st.torque, fmt: (v) => `${v.toFixed(1)} נ"מ`, onInput: (v) => { st.torque = v; upd(); } });
  slider(ctrls, { label: 'מהירות מנוע', min: 1000, max: 8000, step: 100, value: st.rpm, fmt: (v) => `${v} סל"ד`, onInput: (v) => { st.rpm = v; upd(); } });
  slider(ctrls, { label: 'זרוע מנוף במפרק', min: 0.02, max: 0.08, step: 0.002, value: st.arm, fmt: (v) => `${(v * 100).toFixed(1)} ס"מ`, onInput: (v) => { st.arm = v; upd(); } });
  const ro = readout(root.querySelector('.readout-wrap'), [['F', 'כוח במוט'], ['v', 'מהירות המוט'], ['tj', 'מומנט במפרק'], ['wj', 'מהירות במפרק'], ['P', 'הספק מכני']]);
  const eta = 0.85;
  const calc = () => {
    const F = 2 * Math.PI * eta * st.torque / (st.lead / 1000);
    const v = st.rpm / 60 * st.lead / 1000;
    return { F, v, tj: F * st.arm, wj: v / st.arm, P: F * v };
  };
  function upd() { const c = calc(); ro.set({ F: `${f(c.F, 0)} ניוטון (${f(c.F / 9.81, 0)} ק"ג)`, v: `${f(c.v * 1000, 0)} מ"מ/ש'`, tj: `${f(c.tj, 0)} נ"מ`, wj: `${f(c.wj, 1)} רד'/ש'`, P: `${f(c.P, 0)} ואט` }); }
  upd();
  let ang = 0, x = 0, dir = 1;
  animate(root, (dt) => {
    const { c, w, h } = setupCanvas(cv);
    c.clearRect(0, 0, w, h);
    const cy = h / 2, L = Math.min(w - 80, 560), x0 = (w - L) / 2;
    const vis = 0.25;   // תנועה מואטת לתצוגה
    ang += st.rpm / 60 * 2 * Math.PI * dt * vis * 0.05;
    x += dir * calc().v * dt * vis * 40;
    if (x > 0.25 * L) dir = -1; if (x < 0) dir = 1;
    // בורג מרכזי עם הברגה
    c.fillStyle = '#3a3d42'; c.fillRect(x0, cy - 10, L * 0.75, 20);
    c.strokeStyle = '#7a7f87'; c.lineWidth = 1.5;
    const pitch = st.lead * 3;
    for (let px = x0 - pitch + ((ang * pitch / (2 * Math.PI)) % pitch); px < x0 + L * 0.75; px += pitch) {
      if (px < x0) continue;
      c.beginPath(); c.moveTo(px, cy - 10); c.lineTo(px + pitch * 0.5, cy + 10); c.stroke();
    }
    // אום (הרוטור של המנוע) עם הגלגלות
    const nx = x0 + L * 0.3;
    c.fillStyle = 'rgba(214,185,128,.25)'; c.strokeStyle = '#d6b980'; c.lineWidth = 2;
    c.beginPath(); c.roundRect(nx, cy - 46, 110, 92, 10); c.fill(); c.stroke();
    for (let i = 0; i < 6; i++) {
      const a = ang * 0.45 + i * Math.PI / 3, ry = cy + Math.sin(a) * 24;
      const depth = Math.cos(a);
      c.globalAlpha = 0.45 + 0.55 * (depth + 1) / 2;
      c.fillStyle = '#c9ccd2'; c.beginPath(); c.roundRect(nx + 8, ry - 6, 94, 12, 6); c.fill();
      c.strokeStyle = '#8b8f96'; c.lineWidth = 1;
      for (let k = 0; k < 8; k++) { c.beginPath(); c.moveTo(nx + 14 + k * 11, ry - 6); c.lineTo(nx + 19 + k * 11, ry + 6); c.stroke(); }
    }
    c.globalAlpha = 1;
    // המוט שיוצא החוצה ודוחף את המפרק
    const rodEnd = x0 + L * 0.75 + x;
    c.fillStyle = '#9aa0a8'; c.fillRect(x0 + L * 0.55, cy - 7, rodEnd - (x0 + L * 0.55), 14);
    c.fillStyle = '#d6b980'; c.beginPath(); c.arc(rodEnd, cy, 11, 0, Math.PI * 2); c.fill();
    c.fillStyle = '#c3c2b7'; c.font = '13px Rubik'; c.textAlign = 'center';
    c.fillText('המנוע מסובב את האום', nx + 55, cy - 56);
    c.fillText('הגלגלות מתגלגלות בין האום לבורג', nx + 55, cy + 66);
    c.fillText('המוט נדחף', rodEnd, cy - 22);
  });
}

// ================= מעטפת מומנט-מהירות =================
export function torqueSpeed(root) {
  const cv = root.querySelector('canvas.plot');
  const J = Object.fromEntries(JOINTS.map(j => [j.name, j]));
  const choices = [['l_knee', 'ברך'], ['l_hip_pitch', 'ירך (כפיפה)'], ['l_ankle_pitch', 'קרסול'], ['l_hip_roll', 'ירך (צד)']];
  let sel = 'l_knee';
  buttons(root.querySelector('.btns'), choices, { selected: sel, onPick: (k) => { sel = k; draw(); } });
  const ACT = [['stand', 'עמידה', SERIES[2]], ['walk', 'הליכה 0.45 מ\'/ש\'', SERIES[0]], ['walkFast', 'הליכה מהירה', SERIES[1]], ['squat', 'סקוואט', SERIES[3]]];
  let hover = null;
  cv.addEventListener('pointermove', (e) => { const r = cv.getBoundingClientRect(); hover = [e.clientX - r.left, e.clientY - r.top]; draw(); });
  cv.addEventListener('pointerleave', () => { hover = null; draw(); });
  function draw() {
    const { c, w, h } = setupCanvas(cv);
    c.clearRect(0, 0, w, h);
    const j = J[sel], L = 52, R = 14, T = 30, B = 36;
    const xMax = j.speed * 1.05, yMax = j.torque * 1.1;
    const X = (v) => L + v / xMax * (w - L - R), Y = (t) => h - B - t / yMax * (h - T - B);
    c.font = '11px Rubik, sans-serif';
    // רשת
    c.strokeStyle = 'rgba(255,255,255,.07)'; c.fillStyle = '#8b8a83'; c.textAlign = 'right'; c.textBaseline = 'middle';
    const ys = yMax > 200 ? 50 : yMax > 100 ? 25 : 10;
    for (let t = 0; t <= yMax; t += ys) { c.beginPath(); c.moveTo(L, Y(t)); c.lineTo(w - R, Y(t)); c.stroke(); c.fillText(t, L - 6, Y(t)); }
    c.textAlign = 'center'; c.textBaseline = 'top';
    for (let v = 0; v <= xMax; v += 1) c.fillText(v, X(v), h - B + 5);
    c.fillStyle = '#c3c2b7'; c.fillText('מהירות המפרק (רד\'/ש\')', (L + w - R) / 2, h - 16);
    c.save(); c.translate(13, (T + h - B) / 2); c.rotate(-Math.PI / 2); c.textBaseline = 'middle'; c.fillText('מומנט (נ"מ)', 0, 0); c.restore();
    // מעטפת
    c.beginPath();
    for (let v = 0; v <= j.speed; v += j.speed / 100) { const t = torqueLimit(j, v, 1); v === 0 ? c.moveTo(X(v), Y(t)) : c.lineTo(X(v), Y(t)); }
    c.lineTo(X(j.speed), Y(0)); c.lineTo(X(0), Y(0)); c.closePath();
    c.fillStyle = 'rgba(214,185,128,.08)'; c.fill();
    c.strokeStyle = '#d6b980'; c.lineWidth = 2; c.stroke();
    // נקודות
    for (const [k, , col] of ACT) {
      const pts = OPPOINTS[k]?.pts[sel] || [];
      c.fillStyle = col; c.globalAlpha = 0.55;
      for (const [v, t] of pts) { c.beginPath(); c.arc(X(Math.min(v, xMax)), Y(Math.min(t, yMax)), 2.2, 0, Math.PI * 2); c.fill(); }
    }
    c.globalAlpha = 1;
    // מקרא
    let lx = w - R; c.textAlign = 'right'; c.textBaseline = 'middle';
    for (const [k, name, col] of [...ACT].reverse()) {
      const tw = c.measureText(name).width;
      c.fillStyle = '#c3c2b7'; c.fillText(name, lx, 12);
      c.fillStyle = col; c.beginPath(); c.arc(lx - tw - 8, 12, 4, 0, Math.PI * 2); c.fill();
      lx -= tw + 26;
    }
    c.fillStyle = '#d6b980'; c.textAlign = 'left'; c.fillText(`מעטפת המנוע: ${j.torque} נ"מ, ${j.speed} רד'/ש'`, L + 6, T + 8);
    if (hover) {
      const v = (hover[0] - L) / (w - L - R) * xMax, t = (h - B - hover[1]) / (h - T - B) * yMax;
      if (v >= 0 && t >= 0) { c.fillStyle = '#f0efec'; c.textAlign = 'left'; c.fillText(`${v.toFixed(1)} רד'/ש', ${t.toFixed(0)} נ"מ`, hover[0] + 10, hover[1] - 10); }
    }
  }
  draw();
  addEventListener('resize', draw);
}
