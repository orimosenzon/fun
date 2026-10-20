// ווידג'טים של היד: אצבע עם גידים וצימוד של שורש כף היד, ומחשבון אחיזה עם חרוט חיכוך.
import { setupCanvas } from './plot.js';
import { slider, checkbox, readout, buttons, f, animate } from './ui.js';

// ================= אצבע עם גידים =================
// שלושה גידים: גיד i מסתיים בפרק i ועובר מעל כל המפרקים שלפניו. המנוע שולט באורך (משיכה x_i).
// אילוץ האורך: x_i = r·(θ0 + … + θi) + d·φ, כאשר φ זווית שורש כף היד ו-d המרחק של הגיד מהציר שלו.
export function tendon(root) {
  const cv = root.querySelector('canvas.anim'), ctrls = root.querySelector('.ctrls');
  const st = { x: [0.25, 0.45, 0.65], phi: 0, routed: false };
  const r = 0.008;     // זרוע מומנט של הגיד בכל מפרק אצבע (מ')
  const labels = ['גיד הבסיס (MCP)', 'גיד האמצע (PIP)', 'גיד הקצה (DIP)'];
  st.x.forEach((v, i) => slider(ctrls, { label: labels[i], min: 0, max: 1, step: 0.01, value: v, fmt: (q) => `${Math.round(q * 100)}%`, onInput: (q) => { st.x[i] = q; } }));
  slider(ctrls, { label: 'כיפוף שורש כף היד', min: -0.9, max: 0.9, step: 0.01, value: 0, fmt: (q) => `${Math.round(q * 57.3)}°`, onInput: (q) => { st.phi = q; } });
  checkbox(ctrls, { label: 'ניתוב הגידים דרך ציר השורש', value: false, onInput: (v) => { st.routed = v; } });
  const ro = readout(ctrls, [['a', 'זוויות המפרקים'], ['c', 'צימוד מהשורש']]);
  const ang = [0, 0, 0];
  animate(root, (dt) => {
    // הגידים יודעים רק למשוך: אם הגיד רפוי המפרק לא זז
    const d = st.routed ? 0.0012 : 0.012;
    // משיכה של 100% = 12 מ"מ של גיד
    const X = st.x.map(v => v * 0.012);
    // אורך הגיד שנאכל ע"י השורש
    const w = d * st.phi;
    const tgt = [0, 0, 0];
    tgt[0] = Math.max(0, (X[0] - w) / r);
    tgt[1] = Math.max(0, (X[1] - X[0]) / r);
    tgt[2] = Math.max(0, (X[2] - X[1]) / r);
    for (let i = 0; i < 3; i++) { const lim = [1.6, 1.8, 1.4][i]; ang[i] += (Math.min(lim, tgt[i]) - ang[i]) * Math.min(1, dt * 10); }
    ro.set({ a: ang.map(a => Math.round(a * 57.3) + '°').join(' / '), c: `${f(-d * st.phi / r * 57.3, 0)}° בבסיס` });
    const { c, w: W, h: H } = setupCanvas(cv);
    c.clearRect(0, 0, W, H);
    const s = Math.min(W / 430, H / 250);
    // אמה ושורש
    const ox = (W - 400 * s) / 2, oy = H * 0.36;
    const forearm = [[ox, oy], [ox + 150 * s, oy]];
    c.lineCap = 'round';
    c.strokeStyle = '#a8946d'; c.lineWidth = 34 * s; c.beginPath(); c.moveTo(...forearm[0]); c.lineTo(...forearm[1]); c.stroke();
    // מנועים באמה
    for (let i = 0; i < 3; i++) { c.fillStyle = ['#3987e5', '#d95926', '#199e70'][i]; c.fillRect(ox + 12 * s + i * 40 * s, oy - 8 * s, 30 * s, 16 * s); }
    // כף היד מסתובבת סביב השורש
    const wr = forearm[1];
    let a = st.phi;
    const palmL = 90 * s, seg = [52 * s, 34 * s, 26 * s];
    const pts = [wr];
    let p = [wr[0] + Math.cos(a) * palmL, wr[1] + Math.sin(a) * palmL];
    pts.push(p);
    for (let i = 0; i < 3; i++) { a += ang[i]; p = [p[0] + Math.cos(a) * seg[i], p[1] + Math.sin(a) * seg[i]]; pts.push(p); }
    c.strokeStyle = '#2a2b2e'; c.lineWidth = 26 * s; c.beginPath(); c.moveTo(...pts[0]); c.lineTo(...pts[1]); c.stroke();
    for (let i = 1; i < 4; i++) { c.strokeStyle = '#1a1b1e'; c.lineWidth = (18 - i * 2) * s; c.beginPath(); c.moveTo(...pts[i]); c.lineTo(...pts[i + 1]); c.stroke(); }
    // מפרקים
    for (let i = 0; i < 5; i++) { c.fillStyle = i === 0 ? '#d6b980' : '#8b8f96'; c.beginPath(); c.arc(...pts[i], (i === 0 ? 7 : 5) * s, 0, Math.PI * 2); c.fill(); }
    // גידים: מהמנוע, לאורך האמה, דרך השורש (במרחק d), לאורך האצבע בצד הכף (למטה)
    const off = (q, base, dist) => [q[0] - Math.sin(base) * dist, q[1] + Math.cos(base) * dist];
    for (let k = 0; k < 3; k++) {
      c.strokeStyle = ['#3987e5', '#d95926', '#199e70'][k]; c.lineWidth = 2;
      c.beginPath();
      c.moveTo(ox + 27 * s + k * 40 * s, oy + 8 * s);
      const dWr = (st.routed ? 2 : 14 + k * 2) * s;
      c.lineTo(wr[0] - 6 * s, oy + dWr);
      let base = st.phi;
      const q1 = off(pts[1], base, (9 + k * 2) * s);
      c.lineTo(...off(wr, base, dWr)); c.lineTo(...q1);
      for (let i = 0; i <= k; i++) { base += ang[i]; c.lineTo(...off(pts[i + 2], base, 6 * s)); }
      c.stroke();
    }
    c.fillStyle = '#c3c2b7'; c.font = `${13 * s + 2}px Rubik`; c.textAlign = 'center';
    c.fillText('מנועים באמה', ox + 70 * s, oy - 28 * s);
    c.fillText(st.routed ? 'הגידים עוברים דרך הציר' : 'הגידים עוברים מתחת לציר', wr[0], oy + 70 * s);
  });
}

// ================= מחשבון אחיזה =================
const OBJECTS = { egg: ['ביצה', 0.06, 0.45, 'egg'], cell: ['תא 4680', 0.355, 0.8, 'cyl'], cup: ['כוס מים', 0.35, 0.6, 'cyl'], box: ['ארגז', 2.2, 0.9, 'box'], tool: ['מקדחה', 1.6, 0.7, 'box'] };
export function grip(root) {
  const cv = root.querySelector('canvas.anim'), ctrls = root.querySelector('.ctrls');
  const st = { obj: 'cell', mu: 0.8, acc: 2, sf: 1.5 };
  buttons(ctrls, Object.entries(OBJECTS).map(([k, v]) => [k, v[0]]), { selected: st.obj, onPick: (k) => { st.obj = k; mu.value = OBJECTS[k][2]; } });
  const mu = slider(ctrls, { label: 'מקדם חיכוך μ', min: 0.2, max: 1.5, step: 0.05, value: st.mu, fmt: (v) => v.toFixed(2), onInput: (v) => { st.mu = v; } });
  slider(ctrls, { label: 'תאוצה כלפי מעלה', min: 0, max: 10, step: 0.5, value: st.acc, fmt: (v) => `${v} מ'/ש²`, onInput: (v) => { st.acc = v; } });
  slider(ctrls, { label: 'מקדם ביטחון', min: 1, max: 3, step: 0.1, value: st.sf, fmt: (v) => `×${v.toFixed(1)}`, onInput: (v) => { st.sf = v; } });
  const ro = readout(ctrls, [['W', 'כוח לשאת'], ['N', 'לחיצה בכל צד'], ['v', 'הערה']]);
  animate(root, (dt) => {
    const [name, m, , shape] = OBJECTS[st.obj];
    const W = m * (9.81 + st.acc);
    const N = W * st.sf / (2 * st.mu);
    const breakN = 30;
    ro.set({ W: `${f(W, 1)} ניוטון`, N: `${f(N, 1)} ניוטון (${f(N / 9.81 * 1000, 0)} גרם-כוח)`, v: st.obj === 'egg' ? (N > breakN ? '⚠ מעל ~30 נ\': סכנת שבירה' : 'בטווח הבטוח לביצה') : N > 60 ? 'מעל מה שאצבע של אופטימוס נותנת' : 'בתוך היכולת של היד' });
    const { c, w, h } = setupCanvas(cv);
    c.clearRect(0, 0, w, h);
    const cx = w / 2, cy = h / 2 + 4;
    const ow = shape === 'box' ? 150 : shape === 'egg' ? 84 : 78, oh = shape === 'box' ? 136 : shape === 'egg' ? 110 : 146;
    // החפץ
    c.fillStyle = shape === 'box' ? '#c49a6c' : shape === 'egg' ? '#efe6d4' : '#9aa3ad';
    c.beginPath();
    if (shape === 'egg') c.ellipse(cx, cy, ow / 2, oh / 2, 0, 0, Math.PI * 2); else c.roundRect(cx - ow / 2, cy - oh / 2, ow, oh, shape === 'box' ? 4 : 14);
    c.fill();
    // אצבעות/כפות משני הצדדים
    c.fillStyle = '#1d1e21';
    c.beginPath(); c.roundRect(cx - ow / 2 - 58, cy - 32, 56, 64, 12); c.fill();
    c.beginPath(); c.roundRect(cx + ow / 2 + 2, cy - 32, 56, 64, 12); c.fill();
    // חרוטי חיכוך
    const coneH = 95, th = Math.atan(st.mu);
    for (const sg of [-1, 1]) {
      const px = cx + sg * ow / 2;
      c.fillStyle = 'rgba(35,196,131,.12)'; c.strokeStyle = 'rgba(35,196,131,.6)'; c.lineWidth = 1;
      c.beginPath(); c.moveTo(px, cy);
      c.lineTo(px - sg * coneH, cy - coneH * Math.tan(th)); c.lineTo(px - sg * coneH, cy + coneH * Math.tan(th)); c.closePath(); c.fill(); c.stroke();
      // כוח הלחיצה (נורמלי) וכוח החיכוך (מעלה)
      const sc = 95 / Math.max(N, W / 2, 1);
      const nLen = N * sc, fLen = W / 2 * sc;
      c.strokeStyle = '#23c483'; c.lineWidth = 3;
      c.beginPath(); c.moveTo(px + sg * 58, cy); c.lineTo(px + sg * (58 - nLen), cy); c.stroke();
      c.strokeStyle = '#3987e5'; c.beginPath(); c.moveTo(px, cy); c.lineTo(px, cy - fLen); c.stroke();
      // הכוח הכולל במגע: אסור לו לצאת מהחרוט
      const ok = W / 2 <= st.mu * N + 1e-9;
      c.strokeStyle = ok ? '#f0efec' : '#ff4d3d'; c.setLineDash([4, 3]);
      c.beginPath(); c.moveTo(px, cy); c.lineTo(px - sg * nLen, cy - fLen); c.stroke(); c.setLineDash([]);
    }
    // משקל
    c.strokeStyle = '#d95926'; c.lineWidth = 3;
    const wl = Math.min(80, W * 4 + 10);
    c.beginPath(); c.moveTo(cx, cy); c.lineTo(cx, cy + wl); c.stroke();
    c.fillStyle = '#c3c2b7'; c.font = '13px Rubik'; c.textAlign = 'center';
    c.fillText(`${name}: ${m >= 1 ? m + ' ק"ג' : Math.round(m * 1000) + ' גרם'}`, cx, 22);
    c.fillStyle = '#23c483'; c.fillText('לחיצה', cx - ow / 2 - 70, cy - 36);
    c.fillStyle = '#3987e5'; c.fillText('חיכוך', cx + ow / 2 + 40, cy - 60);
    c.fillStyle = '#d95926'; c.fillText('משקל + תאוצה', cx, cy + wl + 16);
  });
}
