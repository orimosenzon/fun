// האקדמיה: המעבדות נטענות רק כשמגיעים אליהן
import { plot, fitCanvas } from '../app/plot.js';
import * as PO from '../core/po.js';
import { sphereSigmaNorm } from '../core/mie.js';
import { layerGamma, salisburyGamma, ferriteMu, ETA0 } from '../core/ram.js';
import * as C from '../core/complex.js';
import * as IR from '../core/ir.js';
import { waveDragDq, searsHaackDq } from '../core/aero.js';

const $ = (s) => document.querySelector(s);
const c0 = 299792458;
const fmtF = (g) => g >= 1 ? g.toFixed(g < 10 ? 2 : 1) + ' GHz' : (g * 1000).toFixed(0) + ' MHz';
const lamStr = (g) => { const l = c0 / (g * 1e9); return l >= 1 ? l.toFixed(2) + ' מ\'' : (l * 100).toFixed(l < 0.1 ? 2 : 1) + ' ס"מ'; };
function kpis(el, rows) { el.innerHTML = rows.map(([t, v, c, cls]) => `<div class="kpi"><span>${t}</span><b class="${cls || ''}">${v}</b>${c ? `<i>${c}</i>` : ''}</div>`).join(''); }
function seg(el, items, cur, cb) {
  el.innerHTML = '';
  for (const [k, n] of items) {
    const b = document.createElement('button'); b.className = 'btn' + (k === cur ? ' on' : ''); b.textContent = n;
    b.onclick = () => { el.querySelectorAll('.btn').forEach((x) => x.classList.toggle('on', x === b)); cb(k); };
    el.appendChild(b);
  }
}
const bind = (id, f) => { const e = $(id); e.addEventListener('input', f); return e; };

// ===== 1: משוואת המכ"ם =====
function labRadarEq() {
  const draw = () => {
    const sdb = +$('#re-s').value, R1 = +$('#re-r').value, s = 10 ** (sdb / 10);
    $('#re-sv').textContent = `${sdb} dBsm = ${s >= 0.01 ? s.toPrecision(2) : s.toExponential(0)} מ"ר`; $('#re-rv').textContent = R1 + ' ק"מ';
    const R = R1 * s ** 0.25, Rc = R1 * 10 ** 0.25;
    const { ctx, W, H } = fitCanvas($('#re-c'));
    ctx.clearRect(0, 0, W, H);
    const sc = (H * 0.44) / Rc, cx = W / 2, cy = H * 0.52;
    // גבול
    ctx.strokeStyle = '#ffb547'; ctx.setLineDash([6, 4]); ctx.beginPath(); ctx.moveTo(10, cy); ctx.lineTo(W - 10, cy); ctx.stroke(); ctx.setLineDash([]);
    const border = 600, nC = Math.ceil(border / (2 * Rc)), nS = Math.ceil(border / (2 * R));
    const bx = (W - 20) / border;
    ctx.fillStyle = 'rgba(255,138,92,.6)';
    for (let i = 0; i < nC; i++) { const x = 10 + (i + 0.5) * border / nC * bx; ctx.beginPath(); ctx.arc(x, cy - 22, 3, 0, 7); ctx.fill(); }
    ctx.fillStyle = '#7cf0c5';
    for (let i = 0; i < Math.min(nS, 400); i++) { const x = 10 + (i + 0.5) * border / nS * bx; ctx.beginPath(); ctx.arc(x, cy + 22, 2.5, 0, 7); ctx.fill(); }
    ctx.strokeStyle = '#ff8a5c'; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(cx, cy, Rc * sc, 0, 7); ctx.stroke();
    ctx.fillStyle = 'rgba(124,240,197,.15)'; ctx.strokeStyle = '#7cf0c5'; ctx.beginPath(); ctx.arc(cx, cy, Math.max(1, R * sc), 0, 7); ctx.fill(); ctx.stroke();
    ctx.fillStyle = '#ffd34d'; ctx.beginPath(); ctx.arc(cx, cy, 5, 0, 7); ctx.fill();
    ctx.font = '12px Rubik'; ctx.direction = 'rtl'; ctx.textAlign = 'center'; ctx.fillStyle = '#ff8a5c';
    ctx.fillText(`מטוס קונבנציונלי: ${Rc.toFixed(0)} ק"מ`, cx, cy - Rc * sc - 8);
    ctx.fillStyle = '#7cf0c5'; ctx.fillText(`שלך: ${R.toFixed(0)} ק"מ`, cx, cy + Math.max(14, R * sc + 16));
    ctx.textAlign = 'right'; ctx.fillStyle = '#8a9bb9';
    ctx.fillText(`מכ"מים לגבול: ${nC}`, W - 12, cy - 34); ctx.fillText(`מכ"מים לגבול: ${nS}`, W - 12, cy + 42);
    ctx.direction = 'ltr';
    const v = 0.28; // ק"מ לשנייה
    kpis($('#re-k'), [
      ['טווח גילוי', R.toFixed(0) + ' ק"מ', `פי ${(Rc / R).toFixed(1)} קצר יותר`],
      ['זמן התרעה', (R / v / 60).toFixed(1) + ' דק\'', `מול ${(Rc / v / 60).toFixed(1)} דק' (מאך 0.9)`],
      ['שטח מכוסה', (100 * (R / Rc) ** 2).toFixed(1) + '%', 'מהשטח שמכ"ם אחד מכסה'],
    ]);
  };
  bind('#re-s', draw); bind('#re-r', draw); draw();
  return draw;
}

// ===== 2: גופים קנוניים =====
function labCanon() {
  const draw = () => {
    const g = 10 ** +$('#cn-f').value, a = +$('#cn-a').value, lam = c0 / (g * 1e9), k = 2 * Math.PI / lam;
    $('#cn-fv').textContent = `${fmtF(g)} (λ = ${lamStr(g)})`; $('#cn-av').textContent = a.toFixed(2) + ' מ\'';
    const th = [], y = [], px = [], py = [];
    const span = Math.min(30, Math.max(3, 400 * lam / a / 2));
    for (let i = 0; i <= 600; i++) {
      const t = -span + 2 * span * i / 600, r = t * Math.PI / 180, x = k * a * Math.sin(r);
      const sinc = Math.abs(x) < 1e-9 ? 1 : Math.sin(x) / x;
      th.push(t); y.push(PO.dB(4 * Math.PI * a ** 4 / lam ** 2 * Math.cos(r) ** 2 * sinc * sinc));
    }
    const h = a / 2, mesh = { pos: [h, h, 0, -h, h, 0, -h, -h, 0, h, -h, 0], tri: [0, 1, 2, 0, 2, 3] }, P = PO.prepare(mesh);
    for (let i = 0; i <= 40; i++) {
      const t = -span + 2 * span * i / 40, r = t * Math.PI / 180;
      // הלוח במישור xy, המכ"ם מסתובב סביב ציר y
      px.push(t); py.push(PO.dB(PO.sigma(P, [Math.sin(r), 0, Math.cos(r)], k)));
    }
    const top = Math.ceil(Math.max(...y) / 10) * 10;
    plot($('#cn-c'), { series: [{ x: th, y, color: '#4fd1ff', width: 2, label: 'הנוסחה' }], points: px.map((x, i) => ({ x, y: py[i], color: '#7cf0c5', r: 3.5 })), xMin: -span, xMax: span, yMin: top - 70, yMax: top, xLabel: 'זווית מהניצב (מעלות)', yLabel: 'σ (dBsm)' });
    // השוואת גופים בגודל 1 מ'
    const shapes = [
      ['לוח 1×1, בניצב', 4 * Math.PI / lam ** 2],
      ['פינה משולשת, צלע 1', 12 * Math.PI / lam ** 2],
      ['פינה כפולה 1×1', 8 * Math.PI / lam ** 2],
      ['גליל ⌀0.5, אורך 1', 2 * Math.PI * 0.25 / lam],
      ['כדור ⌀1', Math.PI * 0.25],
      ['קצה ישר באורך 1', 1 / Math.PI],
      ['חוד חרוט 15°', lam * lam / (16 * Math.PI) * Math.tan(15 * Math.PI / 180) ** 4],
    ];
    const { ctx, W, H } = fitCanvas($('#cn-bars'));
    ctx.clearRect(0, 0, W, H);
    const lo = -80, hi = 60, x0 = 150, xw = W - x0 - 60, rh = (H - 10) / shapes.length;
    ctx.font = '12px Rubik';
    shapes.forEach(([n, s], i) => {
      const db = PO.dB(s), yv = 6 + i * rh, w = Math.max(2, (Math.max(lo, db) - lo) / (hi - lo) * xw);
      ctx.fillStyle = i === 0 ? '#4fd1ff' : '#3a6a9a'; ctx.fillRect(W - x0 - w, yv + 3, w, rh - 7);
      ctx.fillStyle = '#dfe8f7'; ctx.direction = 'rtl'; ctx.textAlign = 'right'; ctx.fillText(n, W - 6, yv + rh / 2 + 4);
      ctx.direction = 'ltr'; ctx.textAlign = 'right'; ctx.font = '11px JetBrains Mono'; ctx.fillStyle = '#8a9bb9';
      ctx.fillText(db.toFixed(0) + ' dBsm', W - x0 - w - 4, yv + rh / 2 + 4); ctx.font = '12px Rubik';
    });
  };
  bind('#cn-f', draw); bind('#cn-a', draw); draw();
  return draw;
}

// ===== 5: עורך צורת כנף =====
const ED_PRESETS = {
  square: ['ריבוע', [[5, 0], [5, 5], [-5, 5], [-5, 0]]],
  diamond: ['יהלום', [[8, 0], [0, 6], [-8, 0]]],
  f117: ['בהשראת F-117', [[10, 0], [-1.5, 6.6], [-3.5, 6.0], [-6.8, 2.2], [-10, 1.5], [-10, 0]]],
  b2: ['בהשראת B-2', [[10.5, 0], [-6.5, 26.2], [-8.6, 25], [-5.3, 19.8], [-9.7, 13], [-6.4, 8.2], [-10.5, 0]]],
  conv: ['כנף ישרה וגוף', [[9, 0], [6, 1.2], [1, 1.2], [0.4, 7], [-1.6, 7], [-2.4, 1.2], [-9, 1.2], [-9, 0]]],
};
function labEdges() {
  let pts = ED_PRESETS.b2[1].map((p) => [...p]);
  seg($('#ed-preset'), Object.entries(ED_PRESETS).map(([k, v]) => [k, v[0]]), 'b2', (k) => { pts = ED_PRESETS[k][1].map((p) => [...p]); draw(); });
  const full = () => { const r = pts.slice(); for (let i = pts.length - 2; i >= 1; i--) r.push([pts[i][0], -pts[i][1]]); return r; };
  const cv = $('#ed-c'); let view = null, drag = -1;
  const draw = () => {
    const g = 10 ** +$('#ed-f').value, lam = c0 / (g * 1e9), k = 2 * Math.PI / lam;
    $('#ed-fv').textContent = `${fmtF(g)} (λ = ${lamStr(g)})`;
    const F = full();
    const { ctx, W, H } = fitCanvas(cv);
    ctx.clearRect(0, 0, W, H);
    let ext = 1; for (const [x, y] of F) ext = Math.max(ext, Math.abs(x), Math.abs(y));
    const sc = Math.min(W, H) * 0.45 / ext, cx = W / 2, cy = H / 2;
    view = { sc, cx, cy };
    const X = (y) => cx - y * sc, Y = (x) => cy - x * sc;
    ctx.fillStyle = '#1a2846'; ctx.beginPath(); F.forEach(([x, y], i) => (i ? ctx.lineTo(X(y), Y(x)) : ctx.moveTo(X(y), Y(x)))); ctx.closePath(); ctx.fill();
    // קצוות לפי כיוון הניצב
    const edges = [];
    let area2 = 0; for (let i = 0; i < F.length; i++) { const a = F[i], b = F[(i + 1) % F.length]; area2 += a[0] * b[1] - b[0] * a[1]; }
    const orient = Math.sign(area2) || 1;
    for (let i = 0; i < F.length; i++) {
      const a = F[i], b = F[(i + 1) % F.length], L = Math.hypot(b[0] - a[0], b[1] - a[1]); if (L < 1e-6) continue;
      const t = [(b[0] - a[0]) / L, (b[1] - a[1]) / L], n = [orient * t[1], -orient * t[0]];
      const ang = ((Math.atan2(n[1], n[0]) * 180 / Math.PI) + 360) % 360;
      edges.push({ L, t, n, ang });
      ctx.strokeStyle = `hsl(${(ang % 180) * 2},85%,62%)`; ctx.lineWidth = 3; ctx.beginPath(); ctx.moveTo(X(a[1]), Y(a[0])); ctx.lineTo(X(b[1]), Y(b[0])); ctx.stroke();
    }
    ctx.fillStyle = '#ffffff';
    pts.forEach(([x, y]) => { ctx.beginPath(); ctx.arc(X(y), Y(x), 5, 0, 7); ctx.fill(); });
    ctx.fillStyle = '#8a9bb9'; ctx.font = '12px Rubik'; ctx.direction = 'rtl'; ctx.textAlign = 'center'; ctx.fillText('▲ אף', cx, 14); ctx.direction = 'ltr';
    // תבנית קוטבית במישור
    const n = 1440, sig = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      const az = i * 2 * Math.PI / n, r = [Math.cos(az), Math.sin(az)];
      let s = 0;
      for (const e of edges) {
        if (e.n[0] * r[0] + e.n[1] * r[1] <= 0) continue;
        const ct = e.t[0] * r[0] + e.t[1] * r[1], x = k * e.L * ct, sinc = Math.abs(x) < 1e-9 ? 1 : Math.sin(x) / x;
        s += e.L * e.L / Math.PI * (1 - ct * ct) * sinc * sinc;
      }
      sig[i] = s;
    }
    const sm = PO.smooth(sig, 4);
    polarDraw($('#ed-p'), [{ arr: sig, color: '#4fd1ff', w: 0.7, a: 0.5 }, { arr: sm, color: '#7cf0c5', w: 2.2 }], -40, 30);
    const nose = PO.sectorMean(sig, 0, 30), side = PO.sectorMean(sig, 90, 30);
    const mx = Math.max(...sig); let dirs = 0; for (let i = 0; i < n; i += 4) { const v = sm[i]; if (v > mx * 0.02 && v >= sm[(i + n - 8) % n] && v >= sm[(i + 8) % n]) dirs++; }
    kpis($('#ed-k'), [
      ['גזרת האף ±30°', PO.dB(nose).toFixed(1) + ' dBsm', 'ממוצע', nose < 0.01 ? 'good' : nose > 1 ? 'bad' : 'warn'],
      ['גזרת הצד', PO.dB(side).toFixed(1) + ' dBsm', 'ממוצע'],
      ['הקוץ הגבוה', PO.dB(mx).toFixed(1) + ' dBsm', 'L²/π של הקצה הארוך'],
      ['מספר קצוות', String(edges.length), `ב-${new Set(edges.map((e) => Math.round((e.ang % 180) / 3) % 60)).size} כיוונים שונים`],
    ]);
  };
  const toW = (e) => { const r = cv.getBoundingClientRect(); return { x: (view.cy - (e.clientY - r.top)) / view.sc, y: (view.cx - (e.clientX - r.left)) / view.sc }; };
  const hit = (p) => pts.findIndex(([x, y]) => Math.hypot(x - p.x, y - p.y) * view.sc < 10);
  cv.addEventListener('pointerdown', (e) => { const i = hit(toW(e)); if (i >= 0) { drag = i; cv.setPointerCapture(e.pointerId); } });
  cv.addEventListener('pointermove', (e) => {
    if (drag < 0) return; const p = toW(e);
    const end = drag === 0 || drag === pts.length - 1;
    pts[drag] = [p.x, end ? 0 : Math.max(0, p.y)]; draw();
  });
  cv.addEventListener('pointerup', () => { drag = -1; });
  cv.addEventListener('dblclick', (e) => {
    const p = toW(e), i = hit(p);
    if (i > 0 && i < pts.length - 1 && pts.length > 3) { pts.splice(i, 1); draw(); return; }
    let best = -1, bd = 1e9;
    for (let j = 0; j + 1 < pts.length; j++) {
      const a = pts[j], b = pts[j + 1], dx = b[0] - a[0], dy = b[1] - a[1], u = Math.max(0, Math.min(1, ((p.x - a[0]) * dx + (p.y - a[1]) * dy) / (dx * dx + dy * dy || 1)));
      const d = Math.hypot(a[0] + u * dx - p.x, a[1] + u * dy - p.y); if (d < bd) { bd = d; best = j; }
    }
    if (best >= 0 && bd * view.sc < 14) { pts.splice(best + 1, 0, [p.x, Math.max(0, p.y)]); draw(); }
  });
  bind('#ed-f', draw); draw();
  return draw;
}

// תרשים קוטבי כללי (0 = אף למעלה, חיובי = שמאל)
export function polarDraw(cv, lines, lo, hi) {
  const { ctx, W, H } = fitCanvas(cv);
  ctx.clearRect(0, 0, W, H);
  const cx = W / 2, cy = H / 2, R = Math.min(W, H) / 2 - 16;
  const rOf = (db) => R * Math.max(0, (Math.min(hi, db) - lo) / (hi - lo));
  const pt = (a, r) => [cx - r * Math.sin(a), cy - r * Math.cos(a)];
  ctx.font = '10px JetBrains Mono'; ctx.direction = 'ltr';
  for (let db = lo + 10; db <= hi; db += 10) { ctx.strokeStyle = '#1c2a46'; ctx.beginPath(); ctx.arc(cx, cy, rOf(db), 0, 7); ctx.stroke(); ctx.fillStyle = '#5a6d92'; ctx.fillText(db + '', cx + 3, cy - rOf(db) + 10); }
  for (let a = 0; a < 360; a += 30) { const [x, y] = pt(a * Math.PI / 180, R); ctx.strokeStyle = '#1c2a46'; ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(x, y); ctx.stroke(); }
  ctx.fillStyle = '#8a9bb9'; ctx.font = '11px Rubik'; ctx.direction = 'rtl'; ctx.textAlign = 'center'; ctx.fillText('אף', cx, 10); ctx.direction = 'ltr';
  for (const l of lines) {
    const n = l.arr.length; ctx.strokeStyle = l.color; ctx.lineWidth = l.w || 2; ctx.globalAlpha = l.a || 1; ctx.beginPath();
    for (let i = 0; i <= n; i++) { const [x, y] = pt(i * 2 * Math.PI / n, rOf(PO.dB(l.arr[i % n]))); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); }
    ctx.stroke(); ctx.globalAlpha = 1;
  }
  return { cx, cy, R, rOf, pt };
}

// ===== 6: מי =====
function labMie() {
  const kas = [], ys = [];
  for (let i = 0; i <= 500; i++) { const ka = 0.03 * Math.pow(100 / 0.03, i / 500); kas.push(ka); ys.push(sphereSigmaNorm(ka)); }
  const ray = kas.map((x) => 9 * x ** 4);
  const parts = [
    ['בורג או חריץ (1 ס"מ)', 0.005, '#8a9bb9'], ['קצה כנף מעוגל (רדיוס 3 ס"מ)', 0.03, '#b59cff'], ['טיל חיצוני (קוטר 18 ס"מ)', 0.09, '#ffb547'],
    ['פתח אוויר (1 מ\')', 0.5, '#ff8a5c'], ['גוף המטוס (קוטר 2 מ\')', 1, '#4fd1ff'], ['זנב (2.5 מ\')', 1.25, '#7cf0c5'],
  ];
  const draw = () => {
    const g = 10 ** +$('#mie-f').value, lam = c0 / (g * 1e9), k = 2 * Math.PI / lam;
    $('#mie-fv').textContent = `${fmtF(g)} (λ = ${lamStr(g)})`;
    const points = parts.map(([, a, c]) => { const ka = Math.min(100, Math.max(0.03, k * a)); return { x: ka, y: Math.log10(sphereSigmaNorm(ka)), color: c, r: 6 }; });
    plot($('#mie-c'), {
      series: [{ x: kas, y: ys.map(Math.log10), color: '#dfe8f7', width: 2, label: 'כדור מוליך (מי)' }, { x: kas, y: ray.map(Math.log10), color: '#b59cff', dash: [4, 4], width: 1.3, label: 'ריילי: 9(ka)⁴' }],
      xLog: true, xMin: 0.03, xMax: 100, yMin: -5, yMax: 1, xLabel: 'ka = 2π·רדיוס/λ', yLabel: 'σ / πa²', fmtY: (v) => '10^' + v, points,
      marks: [{ x: 1, label: 'תהודה', color: '#ffb547' }, { x: 10, label: 'אופטי', color: '#5ef0a0' }],
    });
    $('#mie-list').innerHTML = parts.map(([n, a, c]) => {
      const ka = k * a, reg = ka < 0.5 ? ['ריילי: כמעט לא מפזר', 'good'] : ka < 10 ? ['תהודה: מצלצל', 'bad'] : ['אופטי: פאות עובדות', 'warn'];
      return `<div class="kv" style="display:flex;justify-content:space-between;gap:8px;font-size:13px;margin:3px 0"><span><span style="color:${c}">●</span> ${n}</span><span class="${reg[1]}">ka=${ka.toFixed(ka < 1 ? 2 : 1)} · ${reg[0]}</span></div>`;
    }).join('');
  };
  bind('#mie-f', draw); draw();
  return draw;
}

// ===== 7א: פינה כפולה =====
function cornerTrace(A, inc, nR = 60) {
  // הקודקוד בראשית, חוצה הזווית כלפי מעלה. קירות בזוויות ±A/2 מהאנך
  const h = A / 2 * Math.PI / 180, walls = [[Math.sin(h), Math.cos(h)], [-Math.sin(h), Math.cos(h)]].map(([x, y]) => ({ a: [0, 0], b: [10 * x, 10 * y] }));
  const d0 = [Math.sin(inc * Math.PI / 180), -Math.cos(inc * Math.PI / 180)];
  const perp = [-d0[1], d0[0]];
  const rays = []; let back = 0;
  for (let i = 0; i < nR; i++) {
    const off = -6 + 12 * (i + 0.5) / nR;
    let p = [-d0[0] * 14 + perp[0] * off, -d0[1] * 14 + perp[1] * off], d = [...d0];
    const path = [p.slice()]; let bounces = 0, last = -1;
    for (let b = 0; b < 8; b++) {
      let best = null;
      walls.forEach((w, wi) => {
        if (wi === last) return;
        const ex = w.b[0] - w.a[0], ey = w.b[1] - w.a[1], den = d[0] * ey - d[1] * ex; if (Math.abs(den) < 1e-12) return;
        const t = ((w.a[0] - p[0]) * ey - (w.a[1] - p[1]) * ex) / den, u = ((w.a[0] - p[0]) * d[1] - (w.a[1] - p[1]) * d[0]) / den;
        if (t > 1e-9 && u >= 0 && u <= 1 && (!best || t < best.t)) best = { t, wi, ex, ey };
      });
      if (!best) break;
      p = [p[0] + best.t * d[0], p[1] + best.t * d[1]]; path.push(p.slice());
      const L = Math.hypot(best.ex, best.ey), n = [-best.ey / L, best.ex / L], dn = d[0] * n[0] + d[1] * n[1];
      d = [d[0] - 2 * dn * n[0], d[1] - 2 * dn * n[1]]; bounces++; last = best.wi;
    }
    path.push([p[0] + d[0] * 16, p[1] + d[1] * 16]);
    const ret = bounces > 0 && (d[0] * -d0[0] + d[1] * -d0[1]) > Math.cos(2 * Math.PI / 180);
    if (ret) back++;
    if (bounces) rays.push({ path, ret, bounces });
  }
  return { walls, rays, back, frac: back / nR, d0 };
}
function labCorner() {
  const draw = () => {
    const A = +$('#co-a').value, inc = +$('#co-i').value;
    $('#co-av').textContent = A + '°'; $('#co-iv').textContent = inc + '°';
    const r = cornerTrace(A, inc);
    const { ctx, W, H } = fitCanvas($('#co-c'));
    ctx.clearRect(0, 0, W, H);
    const sc = Math.min(W, H) / 26, X = (x) => W / 2 + x * sc, Y = (y) => H - 20 - y * sc;
    for (const ray of r.rays) {
      ctx.strokeStyle = ray.ret ? 'rgba(255,93,115,.85)' : 'rgba(255,211,77,.35)'; ctx.lineWidth = ray.ret ? 1.6 : 1;
      ctx.beginPath(); ray.path.forEach(([x, y], i) => (i ? ctx.lineTo(X(x), Y(y)) : ctx.moveTo(X(x), Y(y)))); ctx.stroke();
    }
    ctx.strokeStyle = '#dfe8f7'; ctx.lineWidth = 4;
    for (const w of r.walls) { ctx.beginPath(); ctx.moveTo(X(w.a[0]), Y(w.a[1])); ctx.lineTo(X(w.b[0]), Y(w.b[1])); ctx.stroke(); }
    kpis($('#co-k'), [['חוזרות למכ"ם', (100 * r.frac).toFixed(0) + '%', 'מכל האלומה', r.frac > 0.5 ? 'bad' : r.frac > 0.05 ? 'warn' : 'good']]);
    const as = [], fs = [];
    for (let a = 40; a <= 150; a += 1) { as.push(a); fs.push(100 * cornerTrace(a, inc, 80).frac); }
    plot($('#co-p'), { series: [{ x: as, y: fs, color: '#ff5d73', width: 2 }], xMin: 40, xMax: 150, yMin: 0, yMax: 100, xLabel: 'זווית הפינה (מעלות)', yLabel: '% חוזר', marks: [{ x: A, color: '#ffd34d' }] });
  };
  bind('#co-a', draw); bind('#co-i', draw); draw();
  return draw;
}

// ===== 7ב: צינור S ורשת =====
function labDuct() {
  const draw = () => {
    const bend = +$('#du-b').value, gdb = +$('#du-g').value, g = 10 ** (gdb / 10);
    $('#du-bv').textContent = bend.toFixed(2) + ' מ\''; $('#du-gv').textContent = gdb + ' dB';
    const Lx = 6, w = 0.9, N = 80;
    const cl = (x) => bend * (3 * (x / Lx) ** 2 - 2 * (x / Lx) ** 3); // smoothstep
    const top = [], bot = [];
    for (let i = 0; i <= N; i++) { const x = Lx * i / N; top.push([x, cl(x) + w / 2]); bot.push([x, cl(x) - w / 2]); }
    const segs = [];
    for (let i = 0; i < N; i++) { segs.push([top[i], top[i + 1]]); segs.push([bot[i], bot[i + 1]]); }
    segs.push([[Lx, cl(Lx) - w / 2], [Lx, cl(Lx) + w / 2], 'face']);
    const rays = []; let eRet = 0, nRays = 0, bSum = 0, los = false;
    for (let k = 0; k < 41; k++) for (const ang of [-6, 0, 6]) {
      const y0 = -w / 2 + w * (k + 0.5) / 41; let p = [-0.8, y0 - Math.tan(ang * Math.PI / 180) * 0.8], d = [Math.cos(ang * Math.PI / 180), Math.sin(ang * Math.PI / 180)];
      const path = [p.slice()]; let e = 1, bounces = 0, hitFace = false, lastSeg = -1;
      for (let it = 0; it < 60; it++) {
        let best = null;
        segs.forEach((s, si) => {
          if (si === lastSeg) return;
          const a = s[0], b = s[1], ex = b[0] - a[0], ey = b[1] - a[1], den = d[0] * ey - d[1] * ex; if (Math.abs(den) < 1e-12) return;
          const t = ((a[0] - p[0]) * ey - (a[1] - p[1]) * ex) / den, u = ((a[0] - p[0]) * d[1] - (a[1] - p[1]) * d[0]) / den;
          if (t > 1e-9 && u >= 0 && u <= 1 && (!best || t < best.t)) best = { t, si, ex, ey, face: s[2] === 'face' };
        });
        if (!best) break;
        p = [p[0] + best.t * d[0], p[1] + best.t * d[1]]; path.push(p.slice());
        const L = Math.hypot(best.ex, best.ey), n = [-best.ey / L, best.ex / L], dn = d[0] * n[0] + d[1] * n[1];
        d = [d[0] - 2 * dn * n[0], d[1] - 2 * dn * n[1]]; lastSeg = best.si;
        if (best.face) { if (!hitFace && bounces === 0) los = true; hitFace = true; } else { e *= g; bounces++; }
      }
      path.push([p[0] + d[0] * 1.5, p[1] + d[1] * 1.5]);
      nRays++;
      const ret = hitFace && d[0] < -0.95;
      if (ret) { eRet += e; bSum += bounces; }
      rays.push({ path, ret, e });
    }
    const { ctx, W, H } = fitCanvas($('#du-c'));
    ctx.clearRect(0, 0, W, H);
    const sc = Math.min(W / 8.2, H / 3.6), X = (x) => W - 14 - (x + 1.2) * sc, Y = (y) => H / 2 + 0.5 * bend * sc - y * sc;
    for (const r of rays) {
      ctx.strokeStyle = r.ret ? `rgba(255,93,115,${0.15 + 0.85 * Math.sqrt(r.e)})` : 'rgba(255,211,77,.12)'; ctx.lineWidth = 1;
      ctx.beginPath(); r.path.forEach(([x, y], i) => (i ? ctx.lineTo(X(x), Y(y)) : ctx.moveTo(X(x), Y(y)))); ctx.stroke();
    }
    ctx.strokeStyle = '#7cf0c5'; ctx.lineWidth = 3;
    for (const line of [top, bot]) { ctx.beginPath(); line.forEach(([x, y], i) => (i ? ctx.lineTo(X(x), Y(y)) : ctx.moveTo(X(x), Y(y)))); ctx.stroke(); }
    ctx.strokeStyle = '#dfe8f7'; ctx.lineWidth = 6; ctx.beginPath(); ctx.moveTo(X(Lx), Y(cl(Lx) - w / 2)); ctx.lineTo(X(Lx), Y(cl(Lx) + w / 2)); ctx.stroke();
    ctx.fillStyle = '#8a9bb9'; ctx.font = '12px Rubik'; ctx.direction = 'rtl'; ctx.textAlign = 'center';
    ctx.fillText('פני המדחס', X(Lx), Y(cl(Lx) + w / 2) - 8); ctx.fillText('← מהמכ"ם', X(-0.6), Y(0) - w * sc); ctx.direction = 'ltr';
    const rel = eRet / nRays;
    kpis($('#du-k'), [
      ['קו ראייה למדחס', los ? 'כן' : 'לא', los ? 'המדחס נראה ישירות' : 'רק דרך החזרות', los ? 'bad' : 'good'],
      ['החזרות בממוצע', eRet > 0 ? (bSum / Math.max(1, rays.filter((r) => r.ret).length)).toFixed(1) : '–', 'הלוך וחזור'],
      ['החזרה יחסית', eRet > 0 ? (10 * Math.log10(rel)).toFixed(1) + ' dB' : '< -60 dB', 'מול צינור ישר ממתכת'],
    ]);
  };
  const drawGrille = () => {
    const a = +$('#gr-a').value / 100, d = +$('#gr-d').value / 100;
    $('#gr-av').textContent = (a * 100).toFixed(2) + ' ס"מ'; $('#gr-dv').textContent = (d * 100).toFixed(1) + ' ס"מ';
    const fc = c0 / (2 * a) / 1e9, fs = [], ys = [];
    for (let i = 0; i <= 300; i++) {
      const f = 0.3 * Math.pow(60 / 0.3, i / 300); fs.push(f);
      if (f >= fc) ys.push(0); else { const al = 2 * Math.PI * fc * 1e9 / c0 * Math.sqrt(1 - (f / fc) ** 2); ys.push(Math.max(-150, -17.37 * al * d)); }
    }
    plot($('#gr-c'), { series: [{ x: fs, y: ys, color: '#4fd1ff', width: 2.5, label: 'הנחתה הלוך-חזור' }], xLog: true, xMin: 0.3, xMax: 60, yMin: -150, yMax: 5, xLabel: 'תדר (גיגה-הרץ)', yLabel: 'dB', marks: [{ x: fc, label: 'f_c', color: '#ff5d73' }, { x: 3, label: 'S', color: '#5a6d92' }, { x: 10, label: 'X', color: '#5a6d92' }] });
  };
  bind('#du-b', draw); bind('#du-g', draw); bind('#gr-a', drawGrille); bind('#gr-d', drawGrille);
  draw(); drawGrille();
  return () => { draw(); drawGrille(); };
}

// ===== 8: חומרים בולעים =====
function labRam() {
  const TYPES = {
    sal: { n: 'מסך סליסברי', p: { Rs: [377, 100, 1000, 1, 'התנגדות היריעה', 'Ω'], d: [7.5, 1, 30, 0.1, 'מרווח', 'מ"מ'], e: [1, 1, 4, 0.05, 'ε של המרווח', ''] } },
    dal: { n: 'שכבת דלנבך', p: { er: [7, 1, 20, 0.1, "ε'", ''], ei: [2.5, 0, 10, 0.05, "ε''", ''], mr: [1, 1, 5, 0.05, "μ'", ''], mi: [0, 0, 5, 0.05, "μ''", ''], d: [3, 0.2, 20, 0.05, 'עובי', 'מ"מ'] } },
    fer: { n: 'צבע פריט', p: { mus: [40, 2, 80, 1, 'μ סטטי', ''], fr: [1.5, 0.1, 5, 0.05, 'תדר הרלקסציה', 'GHz'], er: [8, 2, 20, 0.5, "ε'", ''], d: [1, 0.2, 5, 0.05, 'עובי', 'מ"מ'] } },
  };
  let type = 'sal'; const vals = {};
  for (const [k, t] of Object.entries(TYPES)) { vals[k] = {}; for (const [pk, p] of Object.entries(t.p)) vals[k][pk] = p[0]; }
  const model = (f) => {
    const v = vals[type];
    if (type === 'sal') return { G: salisburyGamma(f, v.Rs, v.d / 1000, v.e), eps: [v.e, 0], mu: [1, 0], d: v.d / 1000, sheet: v.Rs };
    if (type === 'dal') { const eps = [v.er, -v.ei], mu = [v.mr, -v.mi]; return { G: layerGamma(f, eps, mu, v.d / 1000), eps, mu, d: v.d / 1000 }; }
    const eps = [v.er, -0.6], mu = ferriteMu(f / 1e9, v.mus, v.fr); return { G: layerGamma(f, eps, mu, v.d / 1000), eps, mu, d: v.d / 1000 };
  };
  const ctl = () => {
    const box = $('#ram-ctl'); box.innerHTML = '';
    for (const [pk, p] of Object.entries(TYPES[type].p)) {
      const d = document.createElement('div'); d.className = 'ctl';
      d.innerHTML = `<label>${p[4]} <b>${vals[type][pk]} ${p[5]}</b></label><input type="range" min="${p[1]}" max="${p[2]}" step="${p[3]}" value="${vals[type][pk]}">`;
      const inp = d.querySelector('input'); inp.oninput = () => { vals[type][pk] = +inp.value; d.querySelector('b').textContent = `${inp.value} ${p[5]}`; draw(); };
      box.appendChild(d);
    }
  };
  seg($('#ram-type'), Object.entries(TYPES).map(([k, t]) => [k, t.n]), type, (k) => { type = k; ctl(); draw(); });
  let t0 = performance.now(), raf = 0;
  const draw = () => {
    const fs = [], ys = [];
    for (let i = 0; i <= 300; i++) { const f = 0.1 * Math.pow(200, i / 300); fs.push(f); ys.push(20 * Math.log10(Math.max(1e-5, C.abs(model(f * 1e9).G)))); }
    const fa = 10 ** +$('#ram-f').value;
    $('#ram-fv').textContent = `${fmtF(fa)}: החזרה ${(20 * Math.log10(Math.max(1e-5, C.abs(model(fa * 1e9).G)))).toFixed(1)} dB`;
    plot($('#ram-c'), { series: [{ x: fs, y: ys, color: '#7cf0c5', width: 2.5 }], xLog: true, xMin: 0.1, xMax: 20, yMin: -40, yMax: 1, xLabel: 'תדר (גיגה-הרץ)', yLabel: '|Γ| (dB)', marks: [{ x: fa, color: '#ffd34d' }, { x: 3, label: 'S', color: '#5a6d92' }, { x: 10, label: 'X', color: '#5a6d92' }] });
  };
  const anim = () => {
    raf = requestAnimationFrame(anim);
    const cv = $('#ram-w'); if (!cv.offsetParent) return;
    const fa = 10 ** +$('#ram-f').value * 1e9, m = model(fa), k0 = 2 * Math.PI * fa / c0;
    const n = C.sqrt(C.mul(m.eps, m.mu)), k1 = C.scale(n, k0);
    const G = m.G, wt = (performance.now() - t0) / 1000 * 2 * Math.PI * 0.6;
    // השדה בשכבה: E = C·sin(k1(d - z)), C = (1+Γ)/sin(k1 d)
    const sin = (z) => { const a = C.exp(C.mul([0, 1], z)), b = C.exp(C.mul([0, -1], z)); return C.div(C.sub(a, b), [0, 2]); };
    const Cc = C.div(C.add([1, 0], G), sin(C.scale(k1, m.d)));
    const { ctx, W, H } = fitCanvas(cv);
    ctx.clearRect(0, 0, W, H);
    const xa = W * 0.62, xl = W * 0.9, cy = H / 2, amp = H * 0.2;
    ctx.fillStyle = 'rgba(181,156,255,.15)'; ctx.fillRect(xa, 10, xl - xa, H - 20);
    ctx.fillStyle = '#8a9bb9'; ctx.fillRect(xl, 10, 8, H - 20);
    ctx.strokeStyle = '#1c2a46'; ctx.beginPath(); ctx.moveTo(10, cy); ctx.lineTo(xl, cy); ctx.stroke();
    const rot = [Math.cos(wt), Math.sin(wt)];
    const field = (E) => C.mul(E, rot)[0];
    // אוויר: 2 אורכי גל
    const lam0 = c0 / fa, zair = (px) => -(xa - px) / (xa - 10) * 2 * lam0;
    const env = [], re = [];
    for (let px = 10; px <= xa; px += 2) {
      const z = zair(px), E = C.add(C.exp([0, -k0 * z]), C.mul(G, C.exp([0, k0 * z])));
      env.push([px, C.abs(E)]); re.push([px, field(E)]);
    }
    for (let px = xa; px <= xl; px += 1) {
      const z = (px - xa) / (xl - xa) * m.d, E = C.mul(Cc, sin(C.scale(k1, m.d - z)));
      env.push([px, C.abs(E)]); re.push([px, field(E)]);
    }
    ctx.strokeStyle = 'rgba(79,209,255,.35)'; ctx.setLineDash([4, 4]);
    for (const sg of [1, -1]) { ctx.beginPath(); env.forEach(([x, v], i) => (i ? ctx.lineTo(x, cy - sg * v * amp) : ctx.moveTo(x, cy - sg * v * amp))); ctx.stroke(); }
    ctx.setLineDash([]); ctx.strokeStyle = '#4fd1ff'; ctx.lineWidth = 2.2;
    ctx.beginPath(); re.forEach(([x, v], i) => (i ? ctx.lineTo(x, cy - v * amp) : ctx.moveTo(x, cy - v * amp))); ctx.stroke();
    ctx.fillStyle = '#8a9bb9'; ctx.font = '12px Rubik'; ctx.direction = 'rtl'; ctx.textAlign = 'center';
    ctx.fillText('אוויר (שני אורכי גל)', (10 + xa) / 2, 24); ctx.fillText('השכבה (מוגדלת)', (xa + xl) / 2, 24); ctx.fillText('מתכת', xl + 4, H - 2);
    ctx.fillText(`|Γ| = ${C.abs(G).toFixed(3)}`, (10 + xa) / 2, H - 14); ctx.direction = 'ltr';
  };
  bind('#ram-f', draw); ctl(); draw(); anim();
  return draw;
}

// ===== 9: פלאנק =====
function labPlanck() {
  const draw = () => {
    const T = +$('#pl-t').value, Mn = +$('#pl-m').value;
    $('#pl-tv').textContent = `${T} K (${(T - 273.15).toFixed(0)}°C)`; $('#pl-mv').textContent = Mn.toFixed(2);
    const lam = [], ys = [], ysk = [];
    const Ts = IR.skinT(Mn, 11);
    for (let i = 0; i <= 300; i++) { const l = 0.7 + i * (20 - 0.7) / 300; lam.push(l); ys.push(Math.log10(IR.planck(l * 1e-6, T) * 1e-6 + 1e-9)); ysk.push(Math.log10(IR.planck(l * 1e-6, Ts) * 1e-6 + 1e-9)); }
    const r = plot($('#pl-c'), { series: [{ x: lam, y: ys, color: '#ff8a5c', width: 2.5, label: T + 'K' }, { x: lam, y: ysk, color: '#7cf0c5', width: 2, label: 'עור ' + Ts.toFixed(0) + 'K' }], xMin: 0.7, xMax: 20, yMin: -2, yMax: 5.5, xLabel: 'אורך גל (מיקרון)', yLabel: 'קרינות (W/m²·sr·µm)', fmtY: (v) => '10^' + v, marks: [{ x: 2898 / T, label: 'וין', color: '#ffd34d' }] });
    const ctx = $('#pl-c').getContext('2d');
    ctx.fillStyle = 'rgba(79,209,255,.08)'; ctx.fillRect(r.X(3), r.pad.t, r.X(5) - r.X(3), r.ph); ctx.fillRect(r.X(8), r.pad.t, r.X(12) - r.X(8), r.ph);
    ctx.fillStyle = '#4fd1ff'; ctx.font = '11px Rubik'; ctx.textAlign = 'center'; ctx.fillText('3–5', r.X(4), r.pad.t + r.ph - 6); ctx.fillText('8–12', r.X(10), r.pad.t + r.ph - 6);
    const tot = IR.SIGMA_SB * T ** 4 / Math.PI;
    kpis($('#pl-k'), [
      ['שיא (וין)', (2898 / T).toFixed(2) + ' µm', ''],
      ['הספק כולל', (IR.SIGMA_SB * T ** 4 / 1000).toFixed(T > 600 ? 0 : 2) + ' kW/m²', 'σT⁴'],
      ['בחלון 3–5', (100 * IR.bandRadiance(T, 3e-6, 5e-6) / tot).toFixed(1) + '%', 'מהקרינה'],
      ['בחלון 8–12', (100 * IR.bandRadiance(T, 8e-6, 12e-6) / tot).toFixed(1) + '%', 'מהקרינה'],
    ]);
    const Ta = IR.isaT(11), cont = IR.bandRadiance(Ts, 8e-6, 12e-6) / IR.bandRadiance(Ta, 8e-6, 12e-6);
    kpis($('#pl-k2'), [
      ['טמפרטורת העור', Ts.toFixed(0) + 'K', `${(Ts - 273.15).toFixed(0)}°C, אוויר ${(Ta - 273.15).toFixed(0)}°C`],
      ['ניגוד ב-8–12', '×' + cont.toFixed(1), 'קרינת העור מול הרקע'],
    ]);
  };
  bind('#pl-t', draw); bind('#pl-m', draw); draw();
  return draw;
}

// ===== 10: כלל השטחים =====
function labArea() {
  const draw = () => {
    const s = +$('#ar-s').value, x0 = +$('#ar-x').value;
    $('#ar-sv').textContent = (s * 100).toFixed(0) + '%'; $('#ar-xv').textContent = (x0 * 100).toFixed(0) + '%';
    const L = 1, N = 200, xs = [], A = [], SH = [];
    let V0 = 0; const Smax = 0.01;
    for (let i = 0; i <= N; i++) {
      const x = i / N, xi = 2 * x - 1, base = Smax * Math.pow(Math.max(0, 1 - xi * xi), 1.5);
      const bump = s * 0.35 * Smax * Math.exp(-(((x - x0) / 0.07) ** 2)) * Math.pow(Math.max(0, 1 - xi * xi), 0.5);
      xs.push(x); A.push(base + bump);
    }
    let V = 0; for (let i = 0; i < N; i++) V += 0.5 * (A[i] + A[i + 1]) / N;
    const Sm = 16 * V / (3 * Math.PI * L);
    for (let i = 0; i <= N; i++) { const xi = 2 * xs[i] - 1; SH.push(Sm * Math.pow(Math.max(0, 1 - xi * xi), 1.5)); }
    plot($('#ar-c'), { series: [{ x: xs, y: A.map((v) => v * 1000), color: '#7cf0c5', width: 2.5, label: 'הגוף', fill: 'rgba(124,240,197,.08)' }, { x: xs, y: SH.map((v) => v * 1000), color: '#ffb547', dash: [5, 4], label: 'סירס-האק, אותו נפח' }], xMin: 0, xMax: 1, yMin: 0, xLabel: 'x / L', yLabel: 'S(x) ×10⁻³' });
    // מקדמי פורייה n·A_n²
    const M = 400, nMax = 14, coef = new Float64Array(nMax + 1);
    const Sx = (x) => { const t = Math.max(0, Math.min(N, x * N)), i = Math.min(N - 1, Math.floor(t)), f = t - i; return A[i] * (1 - f) + A[i + 1] * f; };
    for (let k = 0; k < M; k++) {
      const th = (k + 0.5) * Math.PI / M, x = 0.5 * (1 - Math.cos(th)), dx = 1e-3;
      const dS = (Sx(Math.min(1, x + dx)) - Sx(Math.max(0, x - dx))) / (Math.min(1, x + dx) - Math.max(0, x - dx));
      for (let n = 1; n <= nMax; n++) coef[n] += dS * Math.sin(n * th) * 2 / M;
    }
    const contrib = []; for (let n = 1; n <= nMax; n++) contrib.push(Math.PI / 4 * n * coef[n] ** 2);
    const { ctx, W, H } = fitCanvas($('#ar-b'));
    ctx.clearRect(0, 0, W, H);
    const mx = Math.max(...contrib), bw = (W - 30) / nMax;
    contrib.forEach((v, i) => {
      const h = (H - 34) * v / mx; ctx.fillStyle = i === 1 ? '#ffb547' : '#7cf0c5';
      ctx.fillRect(W - 15 - (i + 1) * bw + 3, H - 18 - h, bw - 6, h);
      ctx.fillStyle = '#8a9bb9'; ctx.font = '10px JetBrains Mono'; ctx.textAlign = 'center'; ctx.fillText(String(i + 1), W - 15 - (i + 0.5) * bw, H - 5);
    });
    ctx.fillStyle = '#8a9bb9'; ctx.font = '12px Rubik'; ctx.direction = 'rtl'; ctx.textAlign = 'right'; ctx.fillText('התרומה של כל איבר n·Aₙ² (n=2 כתום: סירס-האק)', W - 6, 14); ctx.direction = 'ltr';
    const dq = waveDragDq(xs.map((x) => 1 - x), A, L), dqSH = searsHaackDq(V, L);
    kpis($('#ar-k'), [['גרר גל מול סירס-האק', '×' + (dq / dqSH).toFixed(2), 'באותו נפח ואורך', dq / dqSH > 1.5 ? 'bad' : dq / dqSH > 1.15 ? 'warn' : 'good']]);
  };
  bind('#ar-s', draw); bind('#ar-x', draw); draw();
  return draw;
}

// ===== 11: היסטוריה =====
function timeline() {
  const T = [
    ['1865', 'משוואות מקסוול', 'ג\'יימס קלרק מקסוול מאחד חשמל, מגנטיות ואור. כל מה שבאקדמיה הזו הוא פתרון מקורב שלהן.'],
    ['1896', 'זומרפלד פותר את הטריז', 'ארנולד זומרפלד מוצא פתרון מדויק לפיזור מחצי-מישור מוליך. מכאן יבואו מקדמי העקיפה של הקצוות.'],
    ['1908', 'גוסטב מי', 'הפתרון המדויק לכדור, שמראה את שלושת האזורים: ריילי, תהודה ואופטי.'],
    ['1935–1945', 'המכ"ם במלחמה', 'רשת Chain Home הבריטית. כבר אז הגרמנים ניסו ציפוי בולע לשנורקלים של צוללות (Schornsteinfeger) ובנו את ה-Horten Ho 229, כנף מעופפת.'],
    ['1962', 'אופימצב וקלר', 'פיוטר אופימצב מפרסם במוסקבה את "שיטת גלי השוליים בתאוריה הפיזיקלית של העקיפה". באותה שנה מתפרסם המאמר המסכם של קלר על GTD.'],
    ['1971', 'התרגום', 'חיל האוויר האמריקאי מתרגם את הספר של אופימצב לאנגלית. בברית המועצות הוא לא נחשב סוד.'],
    ['1975', 'Echo 1 והיהלום חסר התקווה', 'דניס אוברהולסר מהסקאנק וורקס של לוקהיד מוצא את המאמר, ועם המתמטיקאי ביל שרדר כותב תוכנה שמחשבת RCS של גוף מפאות שטוחות. המחשבים של אז לא יכלו לחשב משטחים מעוקלים, ולכן הצורה הייתה מפאות. הדגם, "היהלום חסר התקווה", החזיר במדידות כמו כדור מתכת קטן מאוד.'],
    ['1977', 'Have Blue', 'מטוס ההדגמה טס לראשונה ב-1 בדצמבר 1977. הוכחה שהרעיון עובד גם באוויר, עם מחשב טיסה שמחזיק צורה לא יציבה.'],
    ['1981', 'F-117', 'טיסה ראשונה ב-18 ביוני 1981, וחשיפה לציבור רק בנובמבר 1988. ספנה של 67.5°, פאות, רשת על פתחי האוויר ונחירים שטוחים.'],
    ['1982', 'Tacit Blue', 'נורת\'רופ מטיסה את "הלוויתן": גוף מעוקל, שמוכיח שאפשר לתכנן חמקנות גם עם משטחים חלקים כשיש כוח חישוב.'],
    ['1989', 'B-2', 'טיסה ראשונה ב-17 ביולי 1989. כנף מעופפת חלקה במוטה 52.4 מ\', שפת נגרר בשיני מסור מקבילות לקצה המוביל, בלי זנב. נבנו 21.'],
    ['1997', 'F-22', 'הטיסה הראשונה של דגם הייצור. חמקנות ויכולת תמרון יחד: קצוות מיושרים, זנבות מוטים, צינורות S, טיסה על-קולית בלי מבער.'],
    ['1999', 'הפלת ה-F-117', 'ב-27 במרץ 1999 סוללת S-125 יוגוסלבית בפיקוד זולטן דאני הפילה F-117 מעל סרביה. על פי הדיווחים נעזרו במכ"ם VHF ישן ובתכנון מסלולים צפויים.'],
    ['2006–2011', 'F-35, Su-57, J-20', 'F-35 טס ב-2006, ה-T-50 הרוסי (Su-57) ב-2010 וה-J-20 הסיני ב-2011. ה-F-35I "אדיר" נכנס לשירות בחיל האוויר הישראלי ב-2016.'],
    ['2023', 'B-21', 'המפציץ החדש טס לראשונה ב-10 בנובמבר 2023, כנף מעופפת כמו ה-B-2, עם כל מה שנלמד בשלושים השנים שבינתיים.'],
  ];
  $('#timeline').innerHTML = T.map(([y, t, d]) => `<div class="tl"><b class="y">${y}</b><b>${t}</b><p>${d}</p></div>`).join('');
}

// ===== טעינה עצלה =====
const LABS = {
  'lab-radar-eq': labRadarEq, 'lab-canon': labCanon, 'lab-edges': labEdges, 'lab-mie': labMie, 'lab-corner': labCorner,
  'lab-duct': labDuct, 'lab-ram': labRam, 'lab-planck': labPlanck, 'lab-area': labArea,
  'lab-po3d': () => import('./three_labs.js').then((m) => m.labPO3D()),
  'lab-rays': () => import('./three_labs.js').then((m) => m.labRays()),
};
const redraw = new Map();
const io = new IntersectionObserver((ents) => {
  for (const e of ents) {
    if (!e.isIntersecting || redraw.has(e.target.id)) continue;
    const f = LABS[e.target.id]; if (!f) continue;
    try { redraw.set(e.target.id, f() || null); } catch (err) { console.error(e.target.id, err); }
  }
}, { rootMargin: '300px' });
document.querySelectorAll('.lab').forEach((l) => io.observe(l));
window.addEventListener('resize', () => { for (const f of redraw.values()) if (typeof f === 'function') f(); });
timeline();
// הדגשת הפרק בתוכן העניינים
const tocLinks = [...document.querySelectorAll('#toc a')];
const io2 = new IntersectionObserver((ents) => {
  for (const e of ents) if (e.isIntersecting) tocLinks.forEach((a) => a.classList.toggle('on', a.getAttribute('href') === '#' + e.target.id));
}, { rootMargin: '-40% 0px -55% 0px' });
document.querySelectorAll('.chapter').forEach((c) => io2.observe(c));
