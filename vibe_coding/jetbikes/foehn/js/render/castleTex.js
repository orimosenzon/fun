// Every texture of the fortress is painted here on a canvas: old rubble masonry, slate, half-timbering,
// planks, and one atlas with the woven and painted things (tapestries, carpets, banners, shields,
// stained glass).
import { canvasTex, rng } from './meshkit.js';

const rgb = (c, v = 1) => `rgb(${Math.round(c[0] * v)},${Math.round(c[1] * v)},${Math.round(c[2] * v)})`;
function rrect(g, x, y, w, h, r) {
  r = Math.min(r, w / 2, h / 2);
  g.beginPath(); g.moveTo(x + r, y); g.lineTo(x + w - r, y); g.quadraticCurveTo(x + w, y, x + w, y + r); g.lineTo(x + w, y + h - r); g.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
  g.lineTo(x + r, y + h); g.quadraticCurveTo(x, y + h, x, y + h - r); g.lineTo(x, y + r); g.quadraticCurveTo(x, y, x + r, y); g.closePath();
}
const poly = (g, pts) => { g.beginPath(); pts.forEach((p, i) => (i ? g.lineTo(p[0], p[1]) : g.moveTo(p[0], p[1]))); g.closePath(); };

// rubble masonry laid in rough courses: stones of many sizes and three kinds of rock, worn corners,
// deep joints; 6.4 m to the tile
export function stoneTex() {
  return canvasTex(512, 512, (g, w, h) => {
    const r = rng(77);
    g.fillStyle = '#3a352d'; g.fillRect(0, 0, w, h);
    const rows = [70, 52, 64, 78, 56, 68, 48, 76];
    let y = 0;
    for (const rh of rows) {
      const ws = []; let sum = 0; while (sum < w - 50) { const bw = 44 + r() * r() * 150; ws.push(bw); sum += bw; }
      let x = r() * 90;
      for (const b0 of ws) {
        const bw = b0 * w / sum, kind = r(), v = 0.72 + 0.26 * r(), ins = 2 + r() * 2.6, dy = r() * 5, rad = 3 + r() * 9;
        const c = kind < 0.09 ? [122, 119, 118] : kind < 0.2 ? [172, 152, 126] : kind < 0.36 ? [154, 152, 140] : [176, 166, 144];
        const spots = []; for (let i = 0; i < 5; i++) spots.push([r(), r(), 4 + r() * 16, r() < 0.5 ? 0 : 255, 0.04 + r() * 0.07]);
        for (const ox of [0, w]) {
          const X = x - ox + ins, Y = y + ins + dy * 0.4, W = bw - 2 * ins, H = rh - 2 * ins - dy * 0.4;
          g.fillStyle = rgb(c, v); rrect(g, X, Y, W, H, rad); g.fill();
          g.save(); g.clip();
          for (const s of spots) { g.fillStyle = `rgba(${s[3]},${s[3]},${s[3]},${s[4]})`; g.beginPath(); g.arc(X + s[0] * W, Y + s[1] * H, s[2], 0, 6.3); g.fill(); }
          g.fillStyle = 'rgba(255,250,235,0.11)'; g.fillRect(X, Y, W, 4);
          g.fillStyle = 'rgba(0,0,0,0.16)'; g.fillRect(X, Y + H - 6, W, 6);
          g.restore();
        }
        x += bw;
      }
      y += rh;
    }
    for (let i = 0; i < 14000; i++) { const a = r() * 0.12; g.fillStyle = r() < 0.55 ? `rgba(0,0,0,${a})` : `rgba(255,250,235,${a})`; g.fillRect(r() * w, r() * h, 1 + r() * 2.5, 1 + r() * 2.5); }
    // cracks
    g.strokeStyle = 'rgba(20,18,14,0.5)'; g.lineWidth = 1.2;
    for (let i = 0; i < 9; i++) { let x = r() * w, yy = r() * h; g.beginPath(); g.moveTo(x, yy); for (let k = 0; k < 7; k++) { x += (r() - 0.5) * 26; yy += 8 + r() * 16; g.lineTo(x, yy); } g.stroke(); }
    // old drips
    for (let i = 0; i < 20; i++) { g.fillStyle = `rgba(40,36,26,${0.03 + r() * 0.05})`; g.fillRect(r() * w, r() * h, 3 + r() * 8, 40 + r() * 140); }
  });
}
// slate: staggered rows, 3.2 m to the tile
export function slateTex() {
  return canvasTex(256, 256, (g, w, h) => {
    const r = rng(5);
    g.fillStyle = '#191d24'; g.fillRect(0, 0, w, h);
    for (let row = 0; row < 8; row++) for (let k = -1; k < 9; k++) {
      const v = 0.72 + 0.42 * r(), br = r() < 0.15;
      g.fillStyle = br ? rgb([92, 84, 78], v) : rgb([70, 82, 100], v);
      g.fillRect(k * 32 + (row % 2) * 16 + 1, row * 32 + 1, 30, 30);
      g.fillStyle = 'rgba(0,0,0,0.3)'; g.fillRect(k * 32 + (row % 2) * 16 + 1, row * 32 + 25, 30, 6);
    }
    for (let i = 0; i < 2500; i++) { g.fillStyle = `rgba(0,0,0,${r() * 0.15})`; g.fillRect(r() * w, r() * h, 2, 2); }
  });
}
// half-timbering: two storeys of 3.2 m to the tile, posts every 1.6 m, braces and small windows
export function fachTex() {
  return canvasTex(512, 512, (g, w, h) => {
    const r = rng(31), T = '#3f2318';
    g.fillStyle = '#d6c9aa'; g.fillRect(0, 0, w, h);
    for (let i = 0; i < 160; i++) { g.fillStyle = `rgba(${r() < 0.5 ? '70,55,30' : '255,250,235'},${0.03 + r() * 0.06})`; g.beginPath(); g.arc(r() * w, r() * h, 6 + r() * 30, 0, 6.3); g.fill(); }
    for (let i = 0; i < 40; i++) { g.fillStyle = `rgba(60,50,30,${0.04 + r() * 0.05})`; g.fillRect(r() * w, r() * h, 3 + r() * 6, 30 + r() * 90); }
    g.fillStyle = T; g.strokeStyle = T;
    for (let s = 0; s < 2; s++) {
      const y0 = s * 256;                            // canvas rows: the storey's top is y0, its sill y0 + 256
      g.fillRect(0, y0, w, 12); g.fillRect(0, y0 + 236, w, 20); g.fillRect(0, y0 + 122, w, 11);
      for (let b = 0; b < 4; b++) {
        const x0 = b * 128; g.fillRect(x0 - 8, y0, 16, 256); if (b === 0) g.fillRect(w - 8, y0, 16, 256);
        const kind = (b + s * 2) % 4;
        g.lineWidth = 11;
        if (kind === 0) { g.beginPath(); g.moveTo(x0 + 8, y0 + 236); g.lineTo(x0 + 120, y0 + 133); g.stroke(); }
        else if (kind === 2) { g.beginPath(); g.moveTo(x0 + 120, y0 + 236); g.lineTo(x0 + 8, y0 + 133); g.stroke(); }
        else if (kind === 3) { g.beginPath(); g.moveTo(x0 + 8, y0 + 236); g.lineTo(x0 + 120, y0 + 133); g.moveTo(x0 + 120, y0 + 236); g.lineTo(x0 + 8, y0 + 133); g.stroke(); }
        if (kind === 1 || kind === 3) {              // a window in the upper panel
          g.fillStyle = T; g.fillRect(x0 + 34, y0 + 24, 60, 90);
          g.fillStyle = '#14181d'; g.fillRect(x0 + 40, y0 + 30, 48, 78);
          g.fillStyle = '#6d6a5c'; g.fillRect(x0 + 62, y0 + 30, 3, 78); g.fillRect(x0 + 40, y0 + 66, 48, 3);
          g.fillStyle = T;
        } else { g.fillRect(x0 + 58, y0 + 12, 11, 110); }
      }
    }
    for (let i = 0; i < 5000; i++) { g.fillStyle = `rgba(0,0,0,${r() * 0.08})`; g.fillRect(r() * w, r() * h, 1 + r() * 2, 1 + r() * 3); }
  });
}
// weathered oak planks, 3.2 m to the tile
export function woodTex() {
  return canvasTex(256, 256, (g, w, h) => {
    const r = rng(9);
    for (let k = 0; k < 8; k++) {
      const v = 0.7 + 0.5 * r();
      g.fillStyle = rgb([96, 72, 48], v); g.fillRect(k * 32, 0, 32, h);
      g.fillStyle = 'rgba(0,0,0,0.45)'; g.fillRect(k * 32, 0, 2, h);
      for (let i = 0; i < 14; i++) { g.fillStyle = `rgba(${r() < 0.5 ? '0,0,0' : '200,170,120'},${0.05 + r() * 0.1})`; g.fillRect(k * 32 + 3 + r() * 26, 0, 1 + r() * 1.5, h); }
      for (let i = 0; i < 2; i++) { g.fillStyle = 'rgba(15,12,10,0.8)'; g.beginPath(); g.arc(k * 32 + 16, 40 + i * 128 + r() * 6, 2.4, 0, 6.3); g.fill(); }
    }
    g.fillStyle = 'rgba(18,16,15,0.75)'; g.fillRect(0, 30, w, 9); g.fillRect(0, 158, w, 9);
  });
}

// ---------------- heraldry ----------------
function star(g, cx, cy, R) { g.beginPath(); for (let i = 0; i < 10; i++) { const a = -Math.PI / 2 + i * Math.PI / 5, q = i % 2 ? R * 0.42 : R; g.lineTo(cx + Math.cos(a) * q, cy + Math.sin(a) * q); } g.closePath(); g.fill(); }
function eagle(g, x, y, w, h, col = '#0d0d0d') {
  const X = (u) => x + u * w, Y = (v) => y + v * h, s = Math.min(w, h);
  g.fillStyle = col; g.strokeStyle = col; g.lineCap = 'round';
  for (const sg of [-1, 1]) {
    for (let i = 0; i < 7; i++) {                                  // the feathers of one wing
      const a = (-78 + i * 21) * Math.PI / 180, L = 0.36 - 0.012 * i * i * 0.5, sx = 0.5 + sg * 0.07, sy = 0.40 + i * 0.012;
      g.lineWidth = s * 0.05; g.beginPath(); g.moveTo(X(sx), Y(sy)); g.lineTo(X(sx + sg * Math.cos(a) * L * 0.95), Y(sy + Math.sin(a) * L)); g.stroke();
    }
    g.lineWidth = s * 0.035; g.beginPath(); g.moveTo(X(0.5 + sg * 0.05), Y(0.66)); g.lineTo(X(0.5 + sg * 0.17), Y(0.78)); g.stroke();   // leg
    g.lineWidth = s * 0.02;
    for (const d of [-0.05, 0, 0.05]) { g.beginPath(); g.moveTo(X(0.5 + sg * 0.17), Y(0.78)); g.lineTo(X(0.5 + sg * (0.2 + d)), Y(0.85 - Math.abs(d) * 0.4)); g.stroke(); }
  }
  g.beginPath(); g.ellipse(X(0.5), Y(0.53), w * 0.085, h * 0.19, 0, 0, 6.3); g.fill();
  poly(g, [[X(0.5), Y(0.66)], [X(0.38), Y(0.9)], [X(0.46), Y(0.86)], [X(0.5), Y(0.93)], [X(0.54), Y(0.86)], [X(0.62), Y(0.9)]]); g.fill();   // tail
  poly(g, [[X(0.46), Y(0.4)], [X(0.47), Y(0.25)], [X(0.55), Y(0.25)], [X(0.54), Y(0.4)]]); g.fill();                                   // neck
  g.beginPath(); g.arc(X(0.5), Y(0.22), s * 0.062, 0, 6.3); g.fill();
  poly(g, [[X(0.46), Y(0.19)], [X(0.35), Y(0.235)], [X(0.46), Y(0.26)]]); g.fill();                                                 // beak
  g.fillStyle = '#c8281e'; poly(g, [[X(0.38), Y(0.245)], [X(0.31), Y(0.27)], [X(0.4), Y(0.262)]]); g.fill();                          // tongue
  g.fillStyle = '#f2e6b0'; g.beginPath(); g.arc(X(0.485), Y(0.215), s * 0.013, 0, 6.3); g.fill();
}
// eight coats of arms; each fills the rectangle
function arms(g, k, x, y, w, h) {
  g.save(); g.beginPath(); g.rect(x, y, w, h); g.clip();
  const F = (c) => { g.fillStyle = c; g.fillRect(x, y, w, h); };
  switch (k % 8) {
    case 0: F('#d8a31c'); eagle(g, x, y + h * 0.02, w, h * 0.96); break;                                          // the imperial eagle
    case 1: {                                                                                                      // blue and white lozenges
      F('#e9e4d4'); g.fillStyle = '#2b62ad'; const a = w / 4, b = a * 1.7;
      for (let j = -1; j * b < h + b; j++) for (let i = -1; i < 6; i++) { const cx = x + i * a, cy = y + j * b; poly(g, [[cx, cy - b / 2], [cx + a / 2, cy], [cx, cy + b / 2], [cx - a / 2, cy]]); g.fill(); }
      break;
    }
    case 2: F('#e6dfcc'); g.fillStyle = '#0e0e0e'; g.fillRect(x + w * 0.4, y, w * 0.2, h); g.fillRect(x, y + h * 0.5 - w * 0.1 - h * 0.08, w, w * 0.2); break;   // the black cross of the Order
    case 3: {                                                                                                      // a white tower on red
      F('#96181a'); g.fillStyle = '#e6dfcc'; g.fillRect(x + w * 0.3, y + h * 0.36, w * 0.4, h * 0.46);
      for (let i = 0; i < 3; i++) g.fillRect(x + w * (0.27 + i * 0.165), y + h * 0.26, w * 0.13, h * 0.11);
      g.fillRect(x + w * 0.24, y + h * 0.82, w * 0.52, h * 0.05);
      g.fillStyle = '#1a1210'; g.beginPath(); g.arc(x + w * 0.5, y + h * 0.7, w * 0.07, Math.PI, 0); g.rect(x + w * 0.43, y + h * 0.7, w * 0.14, h * 0.12); g.fill(); g.fillRect(x + w * 0.47, y + h * 0.45, w * 0.06, h * 0.1);
      break;
    }
    case 4: F('#e6dfcc'); g.fillStyle = '#96181a'; g.fillRect(x, y, w / 2, h / 2); g.fillRect(x + w / 2, y + h / 2, w / 2, h / 2); break;
    case 5: F('#d8a31c'); g.fillStyle = '#111'; for (let i = 0; i < 3; i++) g.fillRect(x + w * (i * 2 + 1) / 6, y, w / 6, h); break;
    case 6: F('#27552f'); g.fillStyle = '#d8a31c'; poly(g, [[x, y + h * 0.7], [x + w / 2, y + h * 0.34], [x + w, y + h * 0.7], [x + w, y + h * 0.86], [x + w / 2, y + h * 0.5], [x, y + h * 0.86]]); g.fill(); break;
    default: F('#1d3a72'); g.fillStyle = '#d8a31c'; star(g, x + w * 0.3, y + h * 0.3, w * 0.13); star(g, x + w * 0.7, y + h * 0.3, w * 0.13); star(g, x + w * 0.5, y + h * 0.66, w * 0.13);
  }
  g.restore();
}
function wear(g, x, y, w, h, r, folds) {
  g.save(); g.beginPath(); g.rect(x, y, w, h); g.clip();
  g.fillStyle = 'rgba(95,78,50,0.16)'; g.fillRect(x, y, w, h);
  for (let i = 0; i < w * h / 90; i++) { g.fillStyle = `rgba(${r() < 0.6 ? '0,0,0' : '255,245,220'},${r() * 0.09})`; g.fillRect(x + r() * w, y + r() * h, 1 + r() * 3, 1 + r() * 2); }
  for (let yy = y; yy < y + h; yy += 3) { g.fillStyle = 'rgba(0,0,0,0.05)'; g.fillRect(x, yy, w, 1); }
  if (folds) for (let i = 0; i < folds; i++) { const fx = x + (i + 0.5) * w / folds, gr = g.createLinearGradient(fx - w / folds / 2, 0, fx + w / folds / 2, 0); gr.addColorStop(0, 'rgba(0,0,0,0.20)'); gr.addColorStop(0.5, 'rgba(255,255,255,0.07)'); gr.addColorStop(1, 'rgba(0,0,0,0.20)'); g.fillStyle = gr; g.fillRect(fx - w / folds / 2, y, w / folds, h); }
  g.restore();
}
function border(g, x, y, w, h, b, c0, c1) {
  g.fillStyle = c0; g.fillRect(x, y, w, b); g.fillRect(x, y + h - b, w, b); g.fillRect(x, y, b, h); g.fillRect(x + w - b, y, b, h);
  g.fillStyle = c1;
  const d = b * 0.34, n = Math.round(w / (b * 1.1));
  for (let i = 0; i < n; i++) { const cx = x + (i + 0.5) * w / n; for (const cy of [y + b / 2, y + h - b / 2]) { poly(g, [[cx, cy - d], [cx + d, cy], [cx, cy + d], [cx - d, cy]]); g.fill(); } }
  const m = Math.round(h / (b * 1.1));
  for (let i = 0; i < m; i++) { const cy = y + (i + 0.5) * h / m; for (const cx of [x + b / 2, x + w - b / 2]) { poly(g, [[cx, cy - d], [cx + d, cy], [cx, cy + d], [cx - d, cy]]); g.fill(); } }
  g.strokeStyle = c1; g.lineWidth = 2; g.strokeRect(x + b + 2, y + b + 2, w - 2 * b - 4, h - 2 * b - 4);
}
function flowers(g, x, y, w, h, r, n) {
  for (let i = 0; i < n; i++) {
    const fx = x + r() * w, fy = y + r() * h, c = ['#d9c56a', '#c9524a', '#e4dcc6', '#7d9ac4'][Math.floor(r() * 4)];
    g.strokeStyle = '#4c7a45'; g.lineWidth = 1.5; g.beginPath(); g.moveTo(fx, fy); g.lineTo(fx + (r() - 0.5) * 6, fy + 9); g.stroke();
    g.fillStyle = c; for (let p = 0; p < 5; p++) { g.beginPath(); g.arc(fx + Math.cos(p * 1.257) * 2.6, fy + Math.sin(p * 1.257) * 2.6, 1.7, 0, 6.3); g.fill(); }
  }
}
function tree(g, cx, base, s, r, fruit) {
  g.fillStyle = '#5a4026'; poly(g, [[cx - s * 0.06, base], [cx - s * 0.03, base - s * 0.5], [cx + s * 0.03, base - s * 0.5], [cx + s * 0.06, base]]); g.fill();
  for (let i = 0; i < 9; i++) { const a = i * 0.7, q = i ? s * 0.2 : 0; g.fillStyle = i % 2 ? '#3f6b3a' : '#56824a'; g.beginPath(); g.arc(cx + Math.cos(a) * q, base - s * 0.66 + Math.sin(a) * q * 0.8, s * 0.17, 0, 6.3); g.fill(); }
  g.fillStyle = fruit; for (let i = 0; i < 12; i++) { g.beginPath(); g.arc(cx + (r() - 0.5) * s * 0.6, base - s * 0.66 + (r() - 0.5) * s * 0.5, s * 0.022, 0, 6.3); g.fill(); }
}
function tapestry(g, k, x, y, w, h, r) {
  g.save(); g.beginPath(); g.rect(x, y, w, h); g.clip();
  const b = 34, X = (u) => x + b + u * (w - 2 * b), Y = (v) => y + b + v * (h - 2 * b), s = w - 2 * b;
  if (k === 0) {                                                   // the white stag
    g.fillStyle = '#243d2b'; g.fillRect(x, y, w, h); flowers(g, x + b, y + b, w - 2 * b, h - 2 * b, r, 170);
    tree(g, X(0.14), Y(0.9), s * 0.62, r, '#d9c56a'); tree(g, X(0.88), Y(0.86), s * 0.5, r, '#c9524a');
    const c = '#e6dcc2'; g.fillStyle = c; g.strokeStyle = c; g.lineCap = 'round';
    g.beginPath(); g.ellipse(X(0.5), Y(0.57), s * 0.19, s * 0.085, 0, 0, 6.3); g.fill();
    poly(g, [[X(0.61), Y(0.52)], [X(0.71), Y(0.36)], [X(0.77), Y(0.39)], [X(0.69), Y(0.6)]]); g.fill();
    g.beginPath(); g.ellipse(X(0.765), Y(0.355), s * 0.06, s * 0.036, 0.35, 0, 6.3); g.fill();
    poly(g, [[X(0.33), Y(0.52)], [X(0.285), Y(0.47)], [X(0.31), Y(0.56)]]); g.fill();
    g.lineWidth = s * 0.026; for (const [a, bq] of [[[0.63, 0.6], [0.645, 0.84]], [[0.67, 0.6], [0.71, 0.82]], [[0.36, 0.6], [0.33, 0.84]], [[0.4, 0.61], [0.43, 0.83]]]) { g.beginPath(); g.moveTo(X(a[0]), Y(a[1])); g.lineTo(X(bq[0]), Y(bq[1])); g.stroke(); }
    g.lineWidth = s * 0.014; g.strokeStyle = '#d9c56a';
    for (const [x0, lean] of [[0.745, -0.05], [0.785, 0.06]]) {
      g.beginPath(); g.moveTo(X(x0), Y(0.32)); g.quadraticCurveTo(X(x0 + lean), Y(0.22), X(x0 + lean * 0.4), Y(0.12)); g.stroke();
      for (const t of [0.26, 0.2, 0.15]) { g.beginPath(); g.moveTo(X(x0 + lean * 0.7), Y(t)); g.lineTo(X(x0 + lean * 0.7 + (lean < 0 ? -0.05 : 0.05)), Y(t - 0.05)); g.stroke(); }
    }
    g.fillStyle = '#1a1a1a'; g.beginPath(); g.arc(X(0.775), Y(0.345), s * 0.008, 0, 6.3); g.fill();
    border(g, x, y, w, h, b, '#5e1c1c', '#d0a53a');
  } else if (k === 1) {                                            // the castle on its rock under sun and stars
    g.fillStyle = '#1c2b52'; g.fillRect(x, y, w, h);
    g.fillStyle = '#d9b84a'; for (let i = 0; i < 26; i++) star(g, X(r()), Y(r() * 0.5), s * (0.008 + r() * 0.012));
    g.beginPath(); g.arc(X(0.82), Y(0.18), s * 0.07, 0, 6.3); g.fill();
    g.strokeStyle = '#d9b84a'; g.lineWidth = 3; for (let i = 0; i < 12; i++) { const a = i * 0.5236; g.beginPath(); g.moveTo(X(0.82) + Math.cos(a) * s * 0.09, Y(0.18) + Math.sin(a) * s * 0.09); g.lineTo(X(0.82) + Math.cos(a) * s * 0.13, Y(0.18) + Math.sin(a) * s * 0.13); g.stroke(); }
    g.fillStyle = '#2f5a3a'; g.beginPath(); g.moveTo(X(0), Y(1)); g.quadraticCurveTo(X(0.5), Y(0.42), X(1), Y(1)); g.fill();
    g.fillStyle = '#3f7048'; g.beginPath(); g.moveTo(X(0), Y(1)); g.quadraticCurveTo(X(0.2), Y(0.7), X(0.5), Y(1)); g.fill();
    const tw = (u, v, ww, hh) => { g.fillStyle = '#ddd2b6'; g.fillRect(X(u), Y(v), s * ww, s * hh); g.fillStyle = '#9c2a22'; poly(g, [[X(u) - s * 0.012, Y(v)], [X(u) + s * ww / 2, Y(v) - s * ww * 1.3], [X(u) + s * ww + s * 0.012, Y(v)]]); g.fill(); g.fillStyle = '#1c1a22'; g.fillRect(X(u) + s * ww * 0.38, Y(v) + s * hh * 0.2, s * ww * 0.24, s * 0.04); };
    g.fillStyle = '#ddd2b6'; g.fillRect(X(0.3), Y(0.52), s * 0.4, s * 0.17);
    for (let i = 0; i < 8; i++) g.fillRect(X(0.3 + i * 0.052), Y(0.495), s * 0.03, s * 0.03);
    tw(0.26, 0.4, 0.09, 0.3); tw(0.65, 0.4, 0.09, 0.3); tw(0.44, 0.27, 0.12, 0.42);
    g.fillStyle = '#1c1a22'; g.beginPath(); g.arc(X(0.5), Y(0.64), s * 0.035, Math.PI, 0); g.rect(X(0.465), Y(0.64), s * 0.07, s * 0.05); g.fill();
    flowers(g, x + b, Y(0.78), w - 2 * b, s * 0.2, r, 50);
    border(g, x, y, w, h, b, '#6b4a1c', '#d9b84a');
  } else if (k === 2) {                                            // the tree of life
    g.fillStyle = '#571a22'; g.fillRect(x, y, w, h);
    g.strokeStyle = '#cfa33c'; g.lineCap = 'round'; g.lineWidth = s * 0.035; g.beginPath(); g.moveTo(X(0.5), Y(0.97)); g.lineTo(X(0.5), Y(0.2)); g.stroke();
    for (const sg of [-1, 1]) for (let i = 0; i < 4; i++) {
      const y0 = 0.82 - i * 0.18, len = 0.36 - i * 0.05;
      g.lineWidth = s * 0.018; g.beginPath(); g.moveTo(X(0.5), Y(y0)); g.bezierCurveTo(X(0.5 + sg * len * 0.6), Y(y0), X(0.5 + sg * len), Y(y0 - 0.06), X(0.5 + sg * len), Y(y0 - 0.16)); g.stroke();
      g.beginPath(); g.arc(X(0.5 + sg * (len - 0.035)), Y(y0 - 0.16), s * 0.035, 0, 5); g.stroke();
      for (let l = 0; l < 4; l++) { const u = 0.5 + sg * len * (0.25 + l * 0.2), v = y0 - 0.015 - l * 0.02; g.fillStyle = l % 2 ? '#5d8a4c' : '#86a85e'; g.beginPath(); g.ellipse(X(u), Y(v - 0.03), s * 0.03, s * 0.014, -sg * 0.9, 0, 6.3); g.fill(); }
      g.fillStyle = '#d9584a'; g.beginPath(); g.arc(X(0.5 + sg * len * 0.8), Y(y0 + 0.01), s * 0.02, 0, 6.3); g.fill();
    }
    for (const sg of [-1, 1]) {                                    // two birds
      g.fillStyle = '#dcd3b8'; g.beginPath(); g.ellipse(X(0.5 + sg * 0.2), Y(0.3), s * 0.05, s * 0.03, 0, 0, 6.3); g.fill(); g.beginPath(); g.arc(X(0.5 + sg * 0.155), Y(0.275), s * 0.02, 0, 6.3); g.fill();
      poly(g, [[X(0.5 + sg * 0.24), Y(0.3)], [X(0.5 + sg * 0.31), Y(0.27)], [X(0.5 + sg * 0.3), Y(0.33)]]); g.fill();
      g.fillStyle = '#cfa33c'; poly(g, [[X(0.5 + sg * 0.137), Y(0.27)], [X(0.5 + sg * 0.11), Y(0.278)], [X(0.5 + sg * 0.137), Y(0.286)]]); g.fill();
    }
    g.fillStyle = '#cfa33c'; star(g, X(0.5), Y(0.13), s * 0.06);
    border(g, x, y, w, h, b, '#1f3a2a', '#cfa33c');
  } else {                                                         // the arms of the house between two banners
    g.fillStyle = '#b08a3c'; g.fillRect(x, y, w, h);
    for (let i = 0; i < 9; i++) for (let j = 0; j < 9; j++) { g.fillStyle = 'rgba(90,50,20,0.25)'; const cx = X((i + 0.5) / 9), cy = Y((j + 0.5) / 9), d = s * 0.03; poly(g, [[cx, cy - d], [cx + d, cy], [cx, cy + d], [cx - d, cy]]); g.fill(); }
    const sw = s * 0.36, sh = s * 0.44, sx = X(0.5) - sw / 2, sy = Y(0.36);
    g.save(); g.beginPath(); g.moveTo(sx, sy); g.lineTo(sx + sw, sy); g.lineTo(sx + sw, sy + sh * 0.5); g.quadraticCurveTo(sx + sw, sy + sh * 0.85, sx + sw / 2, sy + sh); g.quadraticCurveTo(sx, sy + sh * 0.85, sx, sy + sh * 0.5); g.closePath(); g.clip(); arms(g, 0, sx, sy, sw, sh); g.restore();
    g.strokeStyle = '#2a1a10'; g.lineWidth = 4; g.beginPath(); g.moveTo(sx, sy); g.lineTo(sx + sw, sy); g.lineTo(sx + sw, sy + sh * 0.5); g.quadraticCurveTo(sx + sw, sy + sh * 0.85, sx + sw / 2, sy + sh); g.quadraticCurveTo(sx, sy + sh * 0.85, sx, sy + sh * 0.5); g.closePath(); g.stroke();
    g.fillStyle = '#8f949c'; g.beginPath(); g.arc(X(0.5), Y(0.27), s * 0.075, Math.PI, 0); g.rect(X(0.5) - s * 0.075, Y(0.27), s * 0.15, s * 0.07); g.fill();   // helm
    g.fillStyle = '#16161a'; g.fillRect(X(0.5) - s * 0.055, Y(0.275), s * 0.11, s * 0.014);
    g.fillStyle = '#96181a'; g.beginPath(); g.ellipse(X(0.5), Y(0.14), s * 0.035, s * 0.075, 0, 0, 6.3); g.fill();
    for (const [sg, ak] of [[-1, 2], [1, 1]]) {
      g.strokeStyle = '#4a321c'; g.lineWidth = 5; g.beginPath(); g.moveTo(X(0.5 + sg * 0.36), Y(0.92)); g.lineTo(X(0.5 + sg * 0.36), Y(0.1)); g.stroke();
      arms(g, ak, X(0.5 + sg * 0.36) - (sg < 0 ? 0 : s * 0.0) - (sg > 0 ? s * 0.16 : 0) + (sg < 0 ? 0 : 0), Y(0.12), s * 0.16, s * 0.24);
    }
    g.fillStyle = '#e6dcc2'; g.beginPath(); g.moveTo(X(0.2), Y(0.9)); g.quadraticCurveTo(X(0.5), Y(0.84), X(0.8), Y(0.9)); g.lineTo(X(0.8), Y(0.96)); g.quadraticCurveTo(X(0.5), Y(0.9), X(0.2), Y(0.96)); g.closePath(); g.fill();
    border(g, x, y, w, h, b, '#3a1c14', '#e0bf5a');
  }
  g.restore();
  wear(g, x, y, w, h, r, 0);
}
function carpet(g, x, y, w, h, field, r) {
  g.save(); g.beginPath(); g.rect(x, y, w, h); g.clip();
  g.fillStyle = field; g.fillRect(x, y, w, h);
  g.fillStyle = '#2a1410'; g.fillRect(x, y, 10, h); g.fillRect(x + w - 10, y, 10, h);
  g.fillStyle = '#caa040'; for (const u of [16, 62]) { g.fillRect(x + u, y, 5, h); g.fillRect(x + w - u - 5, y, 5, h); }
  for (let i = 0; i < 8; i++) for (const cx of [x + 42, x + w - 42]) { const cy = y + (i + 0.5) * h / 8, d = 14; g.fillStyle = i % 2 ? '#caa040' : '#e6dcc2'; poly(g, [[cx, cy - d * 1.4], [cx + d, cy], [cx, cy + d * 1.4], [cx - d, cy]]); g.fill(); }
  const med = (cy) => {
    const cx = x + w / 2;
    g.fillStyle = '#caa040'; poly(g, [[cx, cy - 150], [cx + 120, cy], [cx, cy + 150], [cx - 120, cy]]); g.fill();
    g.fillStyle = '#1c2748'; poly(g, [[cx, cy - 132], [cx + 104, cy], [cx, cy + 132], [cx - 104, cy]]); g.fill();
    g.fillStyle = field; poly(g, [[cx, cy - 84], [cx + 66, cy], [cx, cy + 84], [cx - 66, cy]]); g.fill();
    g.fillStyle = '#caa040'; star(g, cx, cy, 34); g.fillStyle = '#e6dcc2'; for (const [dx, dy] of [[0, -106], [0, 106], [-84, 0], [84, 0]]) { g.beginPath(); g.arc(cx + dx, cy + dy, 7, 0, 6.3); g.fill(); }
  };
  med(y + h / 2); med(y - 0); med(y + h);
  g.fillStyle = '#caa040'; for (const cy of [y + h * 0.25, y + h * 0.75]) for (const dx of [-110, 110]) star(g, x + w / 2 + dx, cy, 13);
  g.restore();
  wear(g, x, y, w, h, r, 0);
}
function lancet(g, x, y, w, h) { g.beginPath(); g.moveTo(x, y + h); g.lineTo(x, y + w * 0.78); g.quadraticCurveTo(x + w * 0.02, y + w * 0.3, x + w / 2, y); g.quadraticCurveTo(x + w * 0.98, y + w * 0.3, x + w, y + w * 0.78); g.lineTo(x + w, y + h); g.closePath(); }
function glass(g, k, x, y, w, h, r) {
  g.clearRect(x, y, w, h);
  g.save(); lancet(g, x + 6, y + 6, w - 12, h - 12); g.clip();
  g.fillStyle = '#0c0c0e'; g.fillRect(x, y, w, h);
  const pal = [['#2a55c8', '#3c7be0', '#1b3a9a', '#c8302a', '#e0b23a'], ['#c8302a', '#e04a3a', '#8f1d1d', '#2a55c8', '#e0b23a'], ['#2e9a5a', '#58c070', '#2a55c8', '#e0b23a', '#c8302a'], ['#7a3ac0', '#2a55c8', '#e0b23a', '#c8302a', '#3c7be0']][k % 4];
  const a = 30, b = 44;
  for (let j = -1; j * b / 2 < h + b; j++) for (let i = -1; i * a < w + a; i++) {
    const cx = x + i * a + (j & 1 ? a / 2 : 0), cy = y + j * b / 2, q = r();
    g.fillStyle = pal[q < 0.5 ? 0 : q < 0.72 ? 1 : q < 0.84 ? 2 : q < 0.93 ? 3 : 4]; g.globalAlpha = 0.75 + r() * 0.25;
    poly(g, [[cx, cy - b / 2 + 2], [cx + a / 2 - 2, cy], [cx, cy + b / 2 - 2], [cx - a / 2 + 2, cy]]); g.fill();
  }
  g.globalAlpha = 1;
  const cx = x + w / 2, cy = y + h * 0.56, R = w * 0.3;
  g.fillStyle = '#0c0c0e'; g.beginPath(); g.arc(cx, cy, R + 7, 0, 6.3); g.fill();
  g.fillStyle = k % 2 ? '#1b3a9a' : '#a8221e'; g.beginPath(); g.arc(cx, cy, R, 0, 6.3); g.fill();
  if (k % 4 === 0) { g.fillStyle = '#f0c648'; g.fillRect(cx - 9, cy - R * 0.75, 18, R * 1.5); g.fillRect(cx - R * 0.5, cy - R * 0.35, R, 18); }
  else if (k % 4 === 1) { g.fillStyle = '#f0c648'; g.beginPath(); g.arc(cx, cy - R * 0.3, R * 0.36, 0, 6.3); g.fill(); g.fillStyle = '#e8d2b0'; g.beginPath(); g.arc(cx, cy - R * 0.3, R * 0.22, 0, 6.3); g.fill(); g.fillStyle = '#e6e2d6'; poly(g, [[cx - R * 0.2, cy - R * 0.08], [cx + R * 0.2, cy - R * 0.08], [cx + R * 0.55, cy + R * 0.85], [cx - R * 0.55, cy + R * 0.85]]); g.fill(); }
  else if (k % 4 === 2) { for (let i = 0; i < 6; i++) { g.fillStyle = i % 2 ? '#f0c648' : '#e8e0d0'; g.beginPath(); g.ellipse(cx + Math.cos(i * 1.047) * R * 0.45, cy + Math.sin(i * 1.047) * R * 0.45, R * 0.3, R * 0.2, i * 1.047, 0, 6.3); g.fill(); } g.fillStyle = '#c8302a'; g.beginPath(); g.arc(cx, cy, R * 0.2, 0, 6.3); g.fill(); }
  else arms(g, 4, cx - R * 0.6, cy - R * 0.65, R * 1.2, R * 1.3);
  g.restore();
  g.strokeStyle = '#8d2a22'; g.lineWidth = 9; lancet(g, x + 11, y + 11, w - 22, h - 22); g.stroke();
  g.strokeStyle = '#0c0c0e'; g.lineWidth = 3; lancet(g, x + 17, y + 17, w - 34, h - 34); g.stroke(); lancet(g, x + 6, y + 6, w - 12, h - 12); g.stroke();
}

export const ATLAS = 2048;
const cell = (x, y, w, h, pad = 3) => [(x + pad) / ATLAS, 1 - (y + h - pad) / ATLAS, (x + w - pad) / ATLAS, 1 - (y + pad) / ATLAS];
export const CELL = {
  tapestry: (k) => cell((k % 4) * 512, 0, 512, 512),
  carpet: (k) => cell((k % 2) * 512, 512, 512, 512, 0.5),
  glass: (k) => cell(1024 + (k % 4) * 256, 512, 256, 512, 1),
  banner: (k) => cell((k % 8) * 256, 1024, 256, 512),
  shield: (k) => cell((k % 8) * 256, 1536, 256, 256),
  shutter: () => cell(0, 1792, 128, 256),
};
export function atlasTex() {
  return canvasTex(ATLAS, ATLAS, (g) => {
    const r = rng(404);
    g.fillStyle = '#777'; g.fillRect(0, 0, ATLAS, ATLAS);
    for (let k = 0; k < 4; k++) tapestry(g, k, k * 512, 0, 512, 512, r);
    carpet(g, 0, 512, 512, 512, '#741420', r); carpet(g, 512, 512, 512, 512, '#1f3a6e', r);
    for (let k = 0; k < 4; k++) glass(g, k, 1024 + k * 256, 512, 256, 512, r);
    for (let k = 0; k < 8; k++) {
      const x = k * 256, y = 1024;
      arms(g, k, x, y, 256, 512);
      g.strokeStyle = '#c9a23e'; g.lineWidth = 8; g.strokeRect(x + 8, y + 8, 240, 496);
      wear(g, x, y, 256, 512, r, 3);
      arms(g, k, x, 1536, 256, 256); wear(g, x, 1536, 256, 256, r, 0);
    }
    // shutters: white and red flames on a board
    g.save(); g.beginPath(); g.rect(0, 1792, 128, 256); g.clip();
    g.fillStyle = '#e6dfcc'; g.fillRect(0, 1792, 128, 256); g.fillStyle = '#9c1c1c';
    for (let i = -2; i < 8; i++) { poly(g, [[0, 1792 + i * 48], [64, 1792 + i * 48 + 40], [128, 1792 + i * 48], [128, 1792 + i * 48 + 24], [64, 1792 + i * 48 + 64], [0, 1792 + i * 48 + 24]]); g.fill(); }
    g.fillStyle = '#1c1512'; g.fillRect(0, 1792 + 40, 128, 7); g.fillRect(0, 1792 + 200, 128, 7);
    g.restore();
    wear(g, 0, 1792, 128, 256, r, 0);
  }, false);
}
export function haloTex() {
  return canvasTex(64, 64, (g) => { const gr = g.createRadialGradient(32, 32, 0, 32, 32, 32); gr.addColorStop(0, 'rgba(255,220,150,1)'); gr.addColorStop(0.25, 'rgba(255,150,50,0.45)'); gr.addColorStop(1, 'rgba(255,90,10,0)'); g.fillStyle = gr; g.fillRect(0, 0, 64, 64); }, false);
}
