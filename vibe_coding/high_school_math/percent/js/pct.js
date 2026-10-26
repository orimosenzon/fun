'use strict';
/* כלים של אחוזים: מספרים עם סבולת, שאלת קלט מספרית, ריבוע מאה לצביעה, פס אחוזים כפול, גרף קווים, עוגה */

// קורא מספר שהתלמיד כתב: מוריד %, ₪ ורווחים. "1,250" הוא אלף מאתיים חמישים, "2,5" הוא 2.5
function num(s) {
  s = String(s).trim().replace(/[\s₪%]/g, '').replace(/[−–]/g, '-');
  if (/^-?\d{1,3}(,\d{3})+(\.\d+)?$/.test(s)) s = s.replace(/,/g, '');
  s = s.replace(',', '.');
  if (!/^-?(\d+\.?\d*|\.\d+)$/.test(s)) return null;
  return Number(s);
}
// אותו מספר עד עיגול: יחסית לגודל, ותמיד עד 0.005
const close = (a, b, tol) => a !== null && Math.abs(a - b) <= (tol ?? Math.max(0.005, Math.abs(b) * 1e-4));
// מספר יפה להצגה: עד 2 ספרות אחרי הנקודה (או p), בלי אפסים מיותרים, פסיקים באלפים
function fmt(x, p = 2) {
  const r = Math.round(x * 10 ** p) / 10 ** p;
  const [i, f] = Math.abs(r).toFixed(p).split('.');
  const ii = i.replace(/\B(?=(\d{3})+(?!\d))/g, ',');
  const ff = (f || '').replace(/0+$/, '');
  return (r < 0 ? '−' : '') + ii + (ff ? '.' + ff : '');
}
const P = (x, p = 2) => M(fmt(x, p) + '%');
const ILS = (x, p = 2) => M(fmt(x, p) + ' ₪');

// שאלת קלט מספרית. ans הוא המספר הנכון; extra: unit, hint, explain, tol, suffix
function numQ(q, ans, extra = {}) {
  return {
    q, type: 'input', show: fmt(ans, extra.p ?? 3) + (extra.suffix || ''),
    accept: s => close(num(s), ans, extra.tol),
    ...extra,
  };
}

/* ---------- ריבוע מאה לצביעה ----------
   onChange(count, set) נקרא בכל שינוי. גרירה צובעת או מוחקת לפי התא הראשון. */
function Grid100(host, opt = {}) {
  const W = 300, X = 6, s = W - 2 * X, w = s / 10;
  const svg = S.svg(W, W, 'board'); svg.style.maxWidth = (opt.max || 330) + 'px';
  host.appendChild(svg);
  S.el('rect', { x: X - 2, y: X - 2, width: s + 4, height: s + 4, rx: 6, class: 'grid-frame' }, svg);
  const on = new Set(opt.start || []), cells = [];
  for (let k = 0; k < 100; k++) {
    const c = k % 10, r = Math.floor(k / 10);
    cells.push(S.el('rect', { x: X + c * w + 1, y: X + r * w + 1, width: w - 2, height: w - 2, rx: 3, class: 'cell', 'data-k': k }, svg));
  }
  let mode = null;
  const paint = k => { if (k == null) return; if (mode) on.add(k); else on.delete(k); draw(); };
  const at = e => { const p = S.pt(svg, e); const c = Math.floor((p.x - X) / w), r = Math.floor((p.y - X) / w); return c >= 0 && c < 10 && r >= 0 && r < 10 ? r * 10 + c : null; };
  if (!opt.readonly) {
    svg.classList.add('cells-paint');
    svg.addEventListener('pointerdown', e => { const k = at(e); if (k == null) return; mode = !on.has(k); svg.setPointerCapture(e.pointerId); paint(k); });
    svg.addEventListener('pointermove', e => { if (mode !== null) paint(at(e)); });
    const up = () => { mode = null; };
    svg.addEventListener('pointerup', up); svg.addEventListener('pointercancel', up);
  }
  function draw() {
    cells.forEach((c, k) => c.setAttribute('class', 'cell' + (on.has(k) ? ' pc' : '')));
    opt.onChange && opt.onChange(on.size, on);
  }
  draw();
  return {
    svg, get count() { return on.size; },
    set(ks) { on.clear(); ks.forEach(k => on.add(k)); draw(); },
    fill(n) { this.set(Array.from({ length: n }, (_, k) => k)); },
  };
}

/* ---------- פס אחוזים כפול ----------
   למעלה אחוזים, למטה כמויות. ידית גרירה. opt: whole, maxPct, step, unit, onChange(pct, val) */
function PctBar(host, opt) {
  const W = 760, H = 150, L = 30, R = W - 30, Y = 62, BH = 34;
  const svg = S.svg(W, H, 'board'); host.appendChild(svg);
  const st = { whole: opt.whole, maxPct: opt.maxPct || 100, pct: opt.pct ?? 0, step: opt.step || 1 };
  const xOf = p => L + (R - L) * p / st.maxPct;
  const g = S.el('g', {}, svg);
  function draw() {
    S.clear(g);
    S.el('rect', { x: L, y: Y, width: xOf(100) - L, height: BH, rx: 6, class: 'pb-whole' }, g);
    if (st.maxPct > 100) S.el('rect', { x: xOf(100), y: Y, width: R - xOf(100), height: BH, rx: 6, class: 'pb-extra' }, g);
    S.el('rect', { x: L, y: Y, width: Math.max(0, xOf(st.pct) - L), height: BH, rx: 6, class: 'pb-fill' }, g);
    const tick = opt.ticks || (st.maxPct > 100 ? 25 : 10);
    for (let p = 0; p <= st.maxPct + 1e-9; p += tick) {
      const x = xOf(p), big = p % 50 === 0;
      S.el('line', { x1: x, x2: x, y1: Y - 6, y2: Y + BH + 6, class: 'pb-tick' + (big ? ' big' : '') }, g);
      if (big || st.maxPct <= 100) {
        S.text(g, x, Y - 12, p + '%', { class: 'pb-lbl top' + (big ? ' big' : ''), 'text-anchor': 'middle' });
        S.text(g, x, Y + BH + 24, fmt(st.whole * p / 100), { class: 'pb-lbl' + (big ? ' big' : ''), 'text-anchor': 'middle' });
      }
    }
    const x = xOf(st.pct), val = st.whole * st.pct / 100;
    S.el('line', { x1: x, x2: x, y1: 18, y2: Y + BH + 32, class: 'pb-cursor' }, g);
    const hd = S.el('g', { class: 'pb-handle', transform: `translate(${x},${Y + BH / 2})` }, g);
    S.el('circle', { r: 15 }, hd); S.el('circle', { r: 26, class: 'pin-hit' }, hd);
    const lab = `${fmt(st.pct, 1)}% = ${fmt(val)}${opt.unit ? ' ' + opt.unit : ''}`;
    S.text(g, clamp(x, L + 70, R - 70), 14, lab, { class: 'pb-read', 'text-anchor': 'middle' });
    opt.onChange && opt.onChange(st.pct, val);
  }
  let drag = false;
  const move = e => { const p = S.pt(svg, e); st.pct = clamp(Math.round(((p.x - L) / (R - L) * st.maxPct) / st.step) * st.step, 0, st.maxPct); draw(); };
  svg.addEventListener('pointerdown', e => { drag = true; svg.setPointerCapture(e.pointerId); move(e); });
  svg.addEventListener('pointermove', e => { if (drag) move(e); });
  svg.addEventListener('pointerup', () => { drag = false; });
  svg.addEventListener('pointercancel', () => { drag = false; });
  draw();
  return { svg, st, draw, set(k, v) { st[k] = v; draw(); } };
}

/* ---------- גרף קווים פשוט ----------
   series: [{ pts:[[x,y]], cls, label }]. opt: xMax, yMax, xLbl, yLbl, marks:[{x,y,t}] */
function lineChart(svg, series, opt) {
  S.clear(svg);
  const vb = svg.viewBox.baseVal, W = vb.width, H = vb.height, L = 58, B = H - 34, T = 14, R = W - 16;
  const xOf = x => L + (R - L) * x / opt.xMax, yOf = y => B - (B - T) * clamp(y, 0, opt.yMax) / opt.yMax;
  const yStep = niceStep(opt.yMax / 5), xStep = opt.xStep || niceStep(opt.xMax / 8);
  for (let y = 0; y <= opt.yMax + 1e-9; y += yStep) {
    S.el('line', { x1: L, x2: R, y1: yOf(y), y2: yOf(y), class: 'ch-grid' }, svg);
    S.text(svg, L - 6, yOf(y) + 4, fmt(y, 1), { class: 'ch-lbl', 'text-anchor': 'end' });
  }
  for (let x = 0; x <= opt.xMax + 1e-9; x += xStep) S.text(svg, xOf(x), B + 18, fmt(x, 1), { class: 'ch-lbl', 'text-anchor': 'middle' });
  S.el('line', { x1: L, x2: R, y1: B, y2: B, class: 'ch-axis' }, svg);
  S.el('line', { x1: L, x2: L, y1: T, y2: B, class: 'ch-axis' }, svg);
  if (opt.xLbl) S.text(svg, R, B + 32, opt.xLbl, { class: 'ch-lbl', 'text-anchor': 'end' });
  (opt.hlines || []).forEach(h => {
    S.el('line', { x1: L, x2: R, y1: yOf(h.y), y2: yOf(h.y), class: 'ch-h' }, svg);
    if (h.t) S.text(svg, L + 6, yOf(h.y) - 5, h.t, { class: 'ch-lbl hl' });
  });
  series.forEach(s => {
    S.el('polyline', { points: s.pts.map(([x, y]) => `${xOf(x)},${yOf(y)}`).join(' '), class: 'ch-line ' + (s.cls || '') }, svg);
    if (s.dots) s.pts.forEach(([x, y]) => S.el('circle', { cx: xOf(x), cy: yOf(y), r: 3.5, class: 'ch-dot ' + (s.cls || '') }, svg));
  });
  (opt.marks || []).forEach(m => {
    S.el('circle', { cx: xOf(m.x), cy: yOf(m.y), r: 6, class: 'ch-mark ' + (m.cls || '') }, svg);
    if (m.t) S.text(svg, clamp(xOf(m.x), L + 40, R - 40), yOf(m.y) - 12 < T + 12 ? yOf(m.y) + 24 : yOf(m.y) - 12, m.t, { class: 'ch-lbl mk', 'text-anchor': 'middle' });
  });
  return { xOf, yOf };
}
function niceStep(raw) {
  const p = 10 ** Math.floor(Math.log10(raw)), f = raw / p;
  return (f <= 1 ? 1 : f <= 2 ? 2 : f <= 2.5 ? 2.5 : f <= 5 ? 5 : 10) * p;
}

/* ---------- עוגה ---------- */
function pie(svg, parts, opt = {}) {
  S.clear(svg);
  const cx = 110, cy = 110, r = 96, tot = parts.reduce((a, p) => a + p.v, 0);
  if (!tot) { S.el('circle', { cx, cy, r, class: 'pie-empty' }, svg); return; }
  let a = -Math.PI / 2;
  parts.forEach(p => {
    if (!p.v) return;
    const da = 2 * Math.PI * p.v / tot;
    if (p.v === tot) S.el('circle', { cx, cy, r, fill: p.col }, svg);
    else {
      const x1 = cx + r * Math.cos(a), y1 = cy + r * Math.sin(a), x2 = cx + r * Math.cos(a + da), y2 = cy + r * Math.sin(a + da);
      S.el('path', { d: `M${cx},${cy} L${x1},${y1} A${r},${r} 0 ${da > Math.PI ? 1 : 0} 1 ${x2},${y2} Z`, fill: p.col, stroke: '#fff', 'stroke-width': 2 }, svg);
    }
    if (p.v / tot >= 0.06) {
      const m = a + da / 2;
      S.text(svg, cx + r * .62 * Math.cos(m), cy + r * .62 * Math.sin(m) + 5, fmt(100 * p.v / tot, 1) + '%', { class: 'pie-lbl', 'text-anchor': 'middle' });
    }
    a += da;
  });
}

// עזר לשאלה עם סיפור: כרטיס עם אימוג'י גדול
const story = (emo, html) => `<div class="story"><span class="semo">${emo}</span><div>${html}</div></div>`;
