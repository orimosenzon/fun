// ווידג'טים של שיווי משקל והליכה: מטוטלת הפוכה על כף רגל, הליכה במודל LIPM ממבט על, ומיזוג חיישנים.
import { LinePlot, SERIES, setupCanvas } from './plot.js';
import { slider, checkbox, readout, buttons, f, animate } from './ui.js';

const GRAV = 9.81, Z = 0.94, W = Math.sqrt(GRAV / Z);
const clamp = (x, a, b) => Math.max(a, Math.min(b, x));

// ================= מטוטלת הפוכה על כף רגל (מבט מהצד) =================
export function pendulum(root) {
  const cv = root.querySelector('canvas.anim'), ctrls = root.querySelector('.ctrls');
  const FOOT = [-0.07, 0.175];   // עקב ובהונות ביחס לקרסול (מ')
  const st = { x: 0.03, v: 0, foot: 0, J: 0.25, auto: true, step: null, msg: '', trail: [] };
  slider(ctrls, { label: 'עוצמת הדחיפה (Δv)', min: 0.05, max: 0.9, step: 0.05, value: st.J, fmt: (v) => `${v.toFixed(2)} מ'/ש'`, onInput: (v) => { st.J = v; } });
  checkbox(ctrls, { label: 'צעד אוטומטי כשהקרסול לא מספיק', value: true, onInput: (v) => { st.auto = v; } });
  const row = document.createElement('div'); row.className = 'btns'; ctrls.appendChild(row);
  const mk = (t, fn) => { const b = document.createElement('button'); b.textContent = t; b.onclick = fn; row.appendChild(b); };
  mk('דחיפה קדימה ▶', () => { st.v += st.J; st.msg = ''; });
  mk('◀ דחיפה אחורה', () => { st.v -= st.J; st.msg = ''; });
  mk('איפוס', () => { Object.assign(st, { x: 0.03, v: 0, foot: 0, step: null, msg: '', trail: [] }); });
  const ro = readout(ctrls, [['xi', 'נקודת הלכידה ביחס לקרסול'], ['p', 'ZMP'], ['s', 'מצב']]);
  animate(root, (dt) => {
    dt = Math.min(dt, 0.03);
    const sub = 10;
    for (let i = 0; i < sub; i++) {
      const h = dt / sub;
      const xi = st.x + st.v / W;
      // ZMP: משוב על נקודת הלכידה, חתוך לסוליה
      let p = clamp(st.foot + 0.06 + 2.5 * (xi - st.foot - 0.06), st.foot + FOOT[0] * 0.9, st.foot + FOOT[1] * 0.9);
      if (st.step) {
        st.step.t += h;
        if (st.step.t >= 0.32) { st.foot = st.step.to; st.step = null; st.msg = 'צעד!'; }
      } else if (st.auto && (xi > st.foot + FOOT[1] * 0.95 || xi < st.foot + FOOT[0] * 0.95)) {
        // צעד: הרגל נוחתת קצת מעבר לאיפה ש-ξ תהיה בנחיתה
        const xiEnd = p + (xi - p) * Math.exp(W * 0.32);
        st.step = { t: 0, from: st.foot, to: xiEnd - 0.04 * Math.sign(xi - st.foot) };
      }
      const a = W * W * (st.x - p);
      st.v += a * h; st.x += st.v * h;
      st.p = p;
      if (Math.abs(st.x - st.foot) > 0.6) { st.msg = 'נפל'; st.v = 0; }
    }
    st.v *= 0.9995;
    const xi = st.x + st.v / W;
    st.trail.push(xi); if (st.trail.length > 120) st.trail.shift();
    const inside = xi > st.foot + FOOT[0] && xi < st.foot + FOOT[1];
    ro.set({ xi: `${f((xi - st.foot) * 100, 1)} ס"מ`, p: `${f((st.p - st.foot) * 100, 1)} ס"מ`, s: st.msg || (st.step ? 'צועד…' : inside ? 'הקרסול מחזיק' : 'מחוץ לסוליה!') });
    // ציור
    const { c, w, h } = setupCanvas(cv);
    c.clearRect(0, 0, w, h);
    const S = Math.min(w / 2.0, h / 1.25), gy = h - 30, cx = w / 2 - st.foot * S * 0.4;
    const X = (x) => cx + (x - st.foot * 0.6) * S, Yh = (z) => gy - z * S;
    c.strokeStyle = 'rgba(255,255,255,.2)'; c.beginPath(); c.moveTo(0, gy); c.lineTo(w, gy); c.stroke();
    // כף הרגל (ואם צועדים: הרגל השנייה בדרך)
    const drawFoot = (fx, col) => { c.fillStyle = col; c.fillRect(X(fx + FOOT[0]), gy - 6, (FOOT[1] - FOOT[0]) * S, 8); };
    drawFoot(st.foot, '#d6b980');
    if (st.step) { const k = st.step.t / 0.32; const fx = st.step.from + (st.step.to - st.step.from) * k; drawFoot(fx, 'rgba(214,185,128,.5)'); c.strokeStyle = 'rgba(214,185,128,.5)'; c.lineWidth = 3; c.beginPath(); c.moveTo(X(st.x), Yh(Z)); c.lineTo(X(fx), gy - 4 - Math.sin(k * Math.PI) * 30); c.stroke(); }
    // רגל ומרכז מסה
    c.strokeStyle = '#a8946d'; c.lineWidth = 6; c.beginPath(); c.moveTo(X(st.foot), gy - 4); c.lineTo(X(st.x), Yh(Z)); c.stroke();
    c.fillStyle = '#f0efec'; c.beginPath(); c.arc(X(st.x), Yh(Z), 14, 0, Math.PI * 2); c.fill();
    // ZMP ו-ξ
    c.fillStyle = '#ff4d3d'; c.beginPath(); c.arc(X(st.p), gy + 12, 6, 0, Math.PI * 2); c.fill();
    c.strokeStyle = 'rgba(47,140,255,.35)'; c.beginPath(); st.trail.forEach((t, i) => i ? c.lineTo(X(t), gy + 22) : c.moveTo(X(t), gy + 22)); c.stroke();
    c.fillStyle = inside ? '#2f8cff' : '#ff9b97'; c.beginPath(); c.arc(X(xi), gy + 22, 6, 0, Math.PI * 2); c.fill();
    // וקטור כוח הרצפה: מה-ZMP אל מרכז המסה
    c.strokeStyle = 'rgba(255,210,61,.8)'; c.lineWidth = 2; c.beginPath(); c.moveTo(X(st.p), gy); c.lineTo(X(st.p + (st.x - st.p) * 0.5), Yh(Z * 0.5)); c.stroke();
    c.fillStyle = '#c3c2b7'; c.font = '12px Rubik'; c.textAlign = 'center';
    c.fillText('ZMP', X(st.p), gy + 36 > h ? gy - 14 : gy + 34);
    c.fillText('מרכז מסה', X(st.x), Yh(Z) - 22);
  });
}

// ================= הליכה ב-LIPM, מבט על =================
export function lipm(root) {
  const cv = root.querySelector('canvas.anim'), ctrls = root.querySelector('.ctrls');
  const st = { v: 0.4, T: 0.45, pushJ: 0.35 };
  slider(ctrls, { label: 'מהירות', min: 0, max: 0.8, step: 0.05, value: st.v, fmt: (v) => `${v.toFixed(2)} מ'/ש'`, onInput: (v) => { st.v = v; } });
  slider(ctrls, { label: 'זמן צעד', min: 0.3, max: 0.8, step: 0.05, value: st.T, fmt: (v) => `${v.toFixed(2)} ש'`, onInput: (v) => { st.T = v; } });
  slider(ctrls, { label: 'עוצמת דחיפה', min: 0.1, max: 0.8, step: 0.05, value: st.pushJ, fmt: (v) => `${v.toFixed(2)} מ'/ש'`, onInput: (v) => { st.pushJ = v; } });
  const row = document.createElement('div'); row.className = 'btns'; ctrls.appendChild(row);
  const mk = (t, fn) => { const b = document.createElement('button'); b.textContent = t; b.onclick = fn; row.appendChild(b); };
  let S;
  const reset = () => { S = { c: [0, 0], cd: [0, 0], stance: { p: [0, -0.1], side: -1 }, t: 0, trail: [], feet: [{ p: [0, 0.1], side: 1 }, { p: [0, -0.1], side: -1 }], fell: false }; };
  reset();
  mk('דחיפה הצידה', () => { S.cd[1] += st.pushJ; });
  mk('דחיפה קדימה', () => { S.cd[0] += st.pushJ; });
  mk('איפוס', reset);
  cv.addEventListener('pointerdown', (e) => {
    const r = cv.getBoundingClientRect(), view = S.view; if (!view) return;
    const x = view.ox + (e.clientX - r.left - view.w / 2) / view.s, y = view.oy - (e.clientY - r.top - view.h / 2) / view.s;
    const d = [x - S.c[0], y - S.c[1]], L = Math.hypot(...d) || 1;
    S.cd[0] += d[0] / L * st.pushJ; S.cd[1] += d[1] / L * st.pushJ;
  });
  const HW = 0.1;
  animate(root, (dt) => {
    dt = Math.min(dt, 0.03);
    const sub = 12, h = dt / sub;
    for (let i = 0; i < sub; i++) {
      S.t += h;
      const p = S.stance.p;
      // ZMP בתוך כף הרגל, עם משוב קטן על ξ
      const xi = [S.c[0] + S.cd[0] / W, S.c[1] + S.cd[1] / W];
      // היעד: מחזוריות. בהליכה ב-v, ξ בסוף הצעד צריכה להיות v·T/(1−e^{−ωT}) לפני הרגל העומדת
      const a = [W * W * (S.c[0] - p[0]), W * W * (S.c[1] - p[1])];
      S.cd[0] += a[0] * h; S.cd[1] += a[1] * h; S.c[0] += S.cd[0] * h; S.c[1] += S.cd[1] * h;
      if (S.t >= st.T) {
        // מיקום הרגל הבאה: ξ הצפויה בנחיתה, פחות ההיסט הנומינלי של הליכה מחזורית
        const e = Math.exp(W * st.T);
        const side = -S.stance.side;
        const bx = st.v * st.T / (e - 1), by = -side * HW * 2 / (e + 1);
        let nx = xi[0] - bx, ny = xi[1] - by;
        // מגבלת אורך צעד
        const dx = clamp(nx - p[0], -0.4, 0.6), dy = side > 0 ? clamp(ny - p[1], 0.12, 0.6) : clamp(ny - p[1], -0.6, -0.12);
        S.stance = { p: [p[0] + dx, p[1] + dy], side };
        S.feet.push(S.stance); if (S.feet.length > 14) S.feet.shift();
        S.t = 0;
      }
      if (Math.hypot(S.c[0] - S.stance.p[0], S.c[1] - S.stance.p[1]) > 0.9) { S.fell = true; reset(); }
    }
    S.trail.push(S.c.slice()); if (S.trail.length > 400) S.trail.shift();
    const { c, w, h: H } = setupCanvas(cv);
    c.clearRect(0, 0, w, H);
    const s = Math.min(w / 4, H / 2.2), ox = S.c[0] + 0.6, oy = S.c[1] * 0.6;
    S.view = { ox, oy, s, w, h: H };
    const X = (x) => w / 2 + (x - ox) * s, Y = (y) => H / 2 - (y - oy) * s;
    // רשת
    c.strokeStyle = 'rgba(255,255,255,.05)';
    for (let gx = Math.floor(ox - 3); gx < ox + 3; gx += 0.5) { c.beginPath(); c.moveTo(X(gx), 0); c.lineTo(X(gx), H); c.stroke(); }
    // כפות רגליים
    S.feet.forEach((ft, i) => {
      const cur = ft === S.stance;
      c.fillStyle = cur ? 'rgba(214,185,128,.45)' : 'rgba(255,255,255,.08)'; c.strokeStyle = cur ? '#d6b980' : 'rgba(255,255,255,.25)';
      c.beginPath(); c.rect(X(ft.p[0] - 0.12), Y(ft.p[1] + 0.05), 0.245 * s, 0.1 * s); c.fill(); c.stroke();
    });
    // מסלול מרכז המסה
    c.strokeStyle = 'rgba(255,255,255,.6)'; c.lineWidth = 2; c.beginPath();
    S.trail.forEach((q, i) => i ? c.lineTo(X(q[0]), Y(q[1])) : c.moveTo(X(q[0]), Y(q[1]))); c.stroke();
    const xi = [S.c[0] + S.cd[0] / W, S.c[1] + S.cd[1] / W];
    c.strokeStyle = 'rgba(47,140,255,.6)'; c.beginPath(); c.moveTo(X(S.c[0]), Y(S.c[1])); c.lineTo(X(xi[0]), Y(xi[1])); c.stroke();
    c.fillStyle = '#fff'; c.beginPath(); c.arc(X(S.c[0]), Y(S.c[1]), 6, 0, Math.PI * 2); c.fill();
    c.fillStyle = '#2f8cff'; c.beginPath(); c.arc(X(xi[0]), Y(xi[1]), 5, 0, Math.PI * 2); c.fill();
    c.fillStyle = '#ff4d3d'; c.beginPath(); c.arc(X(S.stance.p[0]), Y(S.stance.p[1]), 4, 0, Math.PI * 2); c.fill();
    c.fillStyle = '#8b8a83'; c.font = '12px Rubik'; c.textAlign = 'right';
    c.fillText('קדימה ←', w - 10, 18);
  });
}

// ================= מיזוג חיישנים: ג'ירו + מד תאוצה =================
export function imu(root) {
  const cv = root.querySelector('canvas.plot'), ctrls = root.querySelector('.ctrls');
  let alpha = 0.98, seed = 1;
  const plot = new LinePlot(cv, { xLabel: 'זמן (ש\')', yLabel: 'הטיה (מעלות)', xMin: 0, xMax: 10, yMin: -12, yMax: 12, xDec: 1, yDec: 1, unitY: '°' });
  slider(ctrls, { label: 'α (אמון בג\'ירו)', min: 0.8, max: 0.999, step: 0.001, value: alpha, fmt: (v) => v.toFixed(3), onInput: (v) => { alpha = v; draw(); } });
  const ro = readout(ctrls, [['g', 'שגיאה: ג\'ירו בלבד'], ['a', 'שגיאה: מד תאוצה'], ['c', 'שגיאה: מסנן משלים']]);
  const b = document.createElement('div'); b.className = 'btns'; ctrls.appendChild(b);
  const bt = document.createElement('button'); bt.textContent = 'רעש חדש'; bt.onclick = () => { seed++; draw(); }; b.appendChild(bt);
  function rnd() { seed = (seed * 16807) % 2147483647; return seed / 2147483647 - 0.5; }
  function draw() {
    const s0 = seed;
    const dt = 0.01, N = 1000;
    const truth = [], gyro = [], accl = [], comp = [];
    let g = 0, cf = 0, bias = 0.6;   // סחיפה של 0.6°/ש'
    for (let i = 0; i < N; i++) {
      const t = i * dt;
      // נדנוד הליכה + הטיה קדימה איטית + "מעידה"
      const th = 2.5 * Math.sin(2 * Math.PI * 0.9 * t) + 4 * Math.sin(2 * Math.PI * 0.08 * t) + (t > 6 && t < 7 ? 5 * Math.sin(Math.PI * (t - 6)) : 0);
      const thd = i ? (th - truth[i - 1][1]) / dt : 0;
      const gm = thd + bias + rnd() * 3;
      g += gm * dt;
      // מד התאוצה רואה גם את תאוצת הצעדים (ארטיפקט בתדר הצעדים) ורעש
      const am = th + 3 * Math.sin(2 * Math.PI * 1.8 * t + 1) + rnd() * 6;
      cf = alpha * (cf + gm * dt) + (1 - alpha) * am;
      truth.push([t, th]); gyro.push([t, g]); accl.push([t, am]); comp.push([t, cf]);
    }
    seed = s0;
    const rms = (arr) => Math.sqrt(arr.reduce((s, q, i) => s + (q[1] - truth[i][1]) ** 2, 0) / arr.length);
    plot.set([
      { name: 'אמת', data: truth, color: '#f0efec' },
      { name: 'ג\'ירו', data: gyro, color: SERIES[1] },
      { name: 'מד תאוצה', data: accl, color: SERIES[0] },
      { name: 'מסנן משלים', data: comp, color: SERIES[2] },
    ]);
    ro.set({ g: `${f(rms(gyro), 2)}°`, a: `${f(rms(accl), 2)}°`, c: `${f(rms(comp), 2)}°` });
  }
  draw();
  addEventListener('resize', () => plot.draw());
}
