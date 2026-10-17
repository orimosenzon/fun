// גרף קווי קטן על קנבס: ציר אחד, רשת עדינה, מקרא, וקו-כוונת עם חלונית ערכים במעבר עכבר.
// הצבעים לפי הפלטה הקטגוריאלית המאומתת (מצב כהה), תמיד באותו סדר.

export const SERIES = ['#3987e5', '#d95926', '#199e70', '#c98500', '#d55181', '#9085e9', '#e66767', '#008300'];
const INK = '#f0efec', INK2 = '#c3c2b7', MUTED = '#8b8a83', GRID = 'rgba(255,255,255,0.07)', AXIS = 'rgba(255,255,255,0.22)';

export function setupCanvas(cv) {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const r = cv.getBoundingClientRect();
  const w = Math.max(10, r.width), h = Math.max(10, r.height);
  if (cv.width !== Math.round(w * dpr) || cv.height !== Math.round(h * dpr)) { cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr); }
  const c = cv.getContext('2d');
  c.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { c, w, h };
}

function niceStep(range, n = 5) {
  const raw = range / n, p = Math.pow(10, Math.floor(Math.log10(raw))), m = raw / p;
  return (m < 1.5 ? 1 : m < 3 ? 2 : m < 7 ? 5 : 10) * p;
}
const fmt = (v, d) => (Math.abs(v) >= 1000 ? Math.round(v).toLocaleString('en-US') : v.toFixed(d));

export class LinePlot {
  // opts: {xLabel, yLabel, xMin, xMax, yMin, yMax, xDec, yDec, unitX, unitY}
  constructor(cv, opts = {}) {
    this.cv = cv; this.o = { xMin: 0, yMin: 0, xDec: 1, yDec: 1, unitX: '', unitY: '', ...opts };
    this.series = []; this.markers = []; this.vlines = []; this.hover = null;
    cv.addEventListener('pointermove', (e) => { const r = cv.getBoundingClientRect(); this.hover = { x: e.clientX - r.left, y: e.clientY - r.top }; this.draw(); });
    cv.addEventListener('pointerleave', () => { this.hover = null; this.draw(); });
  }
  set(series, extra = {}) { this.series = series; Object.assign(this.o, extra); this.draw(); }
  draw() {
    const { c, w, h } = setupCanvas(this.cv);
    const o = this.o, L = 46, R = 12, T = this.series.length > 1 ? 30 : 14, B = 34;
    c.clearRect(0, 0, w, h);
    let xMax = o.xMax, yMax = o.yMax;
    if (xMax == null || yMax == null) {
      let mx = -Infinity, my = -Infinity;
      for (const s of this.series) for (const [x, y] of s.data) { mx = Math.max(mx, x); my = Math.max(my, y); }
      xMax = xMax ?? mx; yMax = yMax ?? (my * 1.1 || 1);
    }
    const X = (x) => L + (x - o.xMin) / (xMax - o.xMin) * (w - L - R);
    const Y = (y) => h - B - (y - o.yMin) / (yMax - o.yMin) * (h - T - B);
    this.X = X; this.Y = Y;
    c.font = '11px Rubik, sans-serif'; c.lineWidth = 1;
    // רשת וצירים
    const ys = niceStep(yMax - o.yMin, 4), xs = niceStep(xMax - o.xMin, 6);
    c.textAlign = 'right'; c.textBaseline = 'middle';
    for (let y = Math.ceil(o.yMin / ys) * ys; y <= yMax + 1e-9; y += ys) {
      c.strokeStyle = GRID; c.beginPath(); c.moveTo(L, Y(y)); c.lineTo(w - R, Y(y)); c.stroke();
      c.fillStyle = MUTED; c.fillText(fmt(y, ys < 1 ? 1 : 0), L - 6, Y(y));
    }
    c.textAlign = 'center'; c.textBaseline = 'top';
    for (let x = Math.ceil(o.xMin / xs) * xs; x <= xMax + 1e-9; x += xs) c.fillText(fmt(x, xs < 1 ? 1 : 0), X(x), h - B + 5);
    c.strokeStyle = AXIS; c.beginPath(); c.moveTo(L, Y(o.yMin)); c.lineTo(w - R, Y(o.yMin)); c.stroke();
    c.fillStyle = INK2; c.textAlign = 'center'; c.fillText(o.xLabel || '', (L + w - R) / 2, h - 15);
    c.save(); c.translate(11, (T + h - B) / 2); c.rotate(-Math.PI / 2); c.textBaseline = 'middle'; c.fillText(o.yLabel || '', 0, 0); c.restore();
    // קווים אנכיים מסומנים
    for (const v of this.vlines) {
      c.strokeStyle = v.color || AXIS; c.setLineDash([4, 4]); c.beginPath(); c.moveTo(X(v.x), T); c.lineTo(X(v.x), h - B); c.stroke(); c.setLineDash([]);
      c.fillStyle = INK2; c.textAlign = 'center'; c.textBaseline = 'bottom'; c.fillText(v.label || '', X(v.x), T - 1);
    }
    // סדרות
    c.lineJoin = 'round'; c.lineCap = 'round';
    this.series.forEach((s, i) => {
      c.strokeStyle = s.color || SERIES[i]; c.lineWidth = 2; c.setLineDash(s.dash || []);
      c.beginPath(); let first = true;
      let prevOut = false;
      for (const [x, y] of s.data) {
        if (!isFinite(y)) { first = true; continue; }
        const out = y > yMax || y < o.yMin;
        const px = X(x), py = Y(Math.min(Math.max(y, o.yMin), yMax));
        // קו שיוצא מהתחום נחתך בקצה ולא נמרח לאורכו
        if (out && prevOut) { first = true; continue; }
        first ? c.moveTo(px, py) : c.lineTo(px, py); first = false; prevOut = out;
      }
      c.stroke(); c.setLineDash([]);
    });
    // סמנים
    for (const m of this.markers) {
      c.fillStyle = '#1a1a19'; c.beginPath(); c.arc(X(m.x), Y(m.y), 6, 0, Math.PI * 2); c.fill();
      c.fillStyle = m.color || INK; c.beginPath(); c.arc(X(m.x), Y(m.y), 4.5, 0, Math.PI * 2); c.fill();
      if (m.label) { c.fillStyle = INK; c.textAlign = m.align || 'left'; c.textBaseline = 'bottom'; c.fillText(m.label, X(m.x) + (m.align === 'right' ? -8 : 8), Y(m.y) - 4); }
    }
    // מקרא (כשיש יותר מסדרה אחת)
    if (this.series.length > 1) {
      let lx = w - R; c.textAlign = 'right'; c.textBaseline = 'middle';
      this.series.forEach((s, i) => {
        const tw = c.measureText(s.name).width;
        c.fillStyle = INK2; c.fillText(s.name, lx, 12);
        c.fillStyle = s.color || SERIES[i]; c.fillRect(lx - tw - 16, 11, 10, 3);
        lx -= tw + 30;
      });
    }
    // כוונת
    if (this.hover && this.hover.x > L && this.hover.x < w - R) {
      const xv = o.xMin + (this.hover.x - L) / (w - L - R) * (xMax - o.xMin);
      c.strokeStyle = 'rgba(255,255,255,0.35)'; c.beginPath(); c.moveTo(this.hover.x, T); c.lineTo(this.hover.x, h - B); c.stroke();
      const rows = [`${o.xLabel ? o.xLabel.split('(')[0].trim() : 'x'}: ${fmt(xv, o.xDec)}${o.unitX}`];
      this.series.forEach((s, i) => {
        let best = null, bd = Infinity;
        for (const p of s.data) { const d = Math.abs(p[0] - xv); if (d < bd) { bd = d; best = p; } }
        if (best && isFinite(best[1])) {
          rows.push([s.color || SERIES[i], `${s.name}: ${fmt(best[1], o.yDec)}${o.unitY}`]);
          c.fillStyle = '#1a1a19'; c.beginPath(); c.arc(X(best[0]), Y(best[1]), 5, 0, Math.PI * 2); c.fill();
          c.fillStyle = s.color || SERIES[i]; c.beginPath(); c.arc(X(best[0]), Y(best[1]), 3.5, 0, Math.PI * 2); c.fill();
        }
      });
      const bw = Math.max(...rows.map(r => c.measureText(typeof r === 'string' ? r : r[1]).width)) + 26, bh = rows.length * 16 + 8;
      let bx = this.hover.x + 12; if (bx + bw > w) bx = this.hover.x - bw - 12;
      const by = T + 4;
      c.fillStyle = 'rgba(20,20,19,0.94)'; c.strokeStyle = 'rgba(255,255,255,0.15)';
      c.beginPath(); c.roundRect(bx, by, bw, bh, 6); c.fill(); c.stroke();
      c.textAlign = 'left'; c.textBaseline = 'middle';
      rows.forEach((r, i) => {
        const yy = by + 12 + i * 16;
        if (typeof r === 'string') { c.fillStyle = INK2; c.fillText(r, bx + 8, yy); }
        else { c.fillStyle = r[0]; c.fillRect(bx + 8, yy - 2, 8, 4); c.fillStyle = INK; c.fillText(r[1], bx + 20, yy); }
      });
    }
  }
}
