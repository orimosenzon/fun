// util.js — עיצוב מספרים, יצירת אלמנטים וגרפים קטנים
export const $ = (sel, root = document) => root.querySelector(sel);
export const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

export function fmt(x, digits = 0) {
  if (!isFinite(x)) return '—';
  return x.toLocaleString('en-US', { minimumFractionDigits: digits, maximumFractionDigits: digits });
}
export function fmtDur(sec) {
  if (!isFinite(sec)) return '—';
  const s = Math.abs(sec);
  if (s < 60) return `${fmt(s, 0)} שנ׳`;
  if (s < 3600) return `${Math.floor(s / 60)} דק׳ ${Math.floor(s % 60)} שנ׳`;
  if (s < 86400 * 2) return `${Math.floor(s / 3600)} שע׳ ${Math.round((s % 3600) / 60)} דק׳`;
  if (s < 86400 * 365 * 2) return `${fmt(s / 86400, 1)} ימים`;
  return `${fmt(s / (86400 * 365.25), 0)} שנים`;
}
export function fmtT(sec) { // T+hh:mm:ss
  const s = Math.max(0, Math.floor(sec));
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), ss = s % 60;
  return (h ? h + ':' + String(m).padStart(2, '0') : String(m).padStart(2, '0')) + ':' + String(ss).padStart(2, '0');
}
export function fmtMass(kg) {
  if (kg >= 1000) return `${fmt(kg / 1000, kg >= 100000 ? 0 : 1)} טון`;
  return `${fmt(kg)} ק"ג`;
}
const HE_MONTHS = ['בינואר', 'בפברואר', 'במרץ', 'באפריל', 'במאי', 'ביוני', 'ביולי', 'באוגוסט', 'בספטמבר', 'באוקטובר', 'בנובמבר', 'בדצמבר'];
export function fmtDateHe(d) { return `${d.getUTCDate()} ${HE_MONTHS[d.getUTCMonth()]} ${d.getUTCFullYear()}`; }

export function h(html) {
  const t = document.createElement('template');
  t.innerHTML = html.trim();
  return t.content.firstElementChild;
}

// גרף קווי קטן על קנבס: series = [{data:[[x,y]], color}], marker x
export function drawChart(canvas, series, opts = {}) {
  const dpr = Math.min(window.devicePixelRatio, 2);
  const W = canvas.clientWidth, H = canvas.clientHeight;
  if (canvas.width !== W * dpr) { canvas.width = W * dpr; canvas.height = H * dpr; }
  const g = canvas.getContext('2d');
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, W, H);
  const pad = { l: 40, r: 8, t: 16, b: 18 };
  let xMax = opts.xMax ?? 0, yMax = opts.yMax ?? 0;
  for (const s of series) for (const [x, y] of s.data) { if (opts.xMax == null) xMax = Math.max(xMax, x); if (opts.yMax == null) yMax = Math.max(yMax, y); }
  yMax = yMax || 1; xMax = xMax || 1;
  const X = x => pad.l + (W - pad.l - pad.r) * x / xMax;
  const Y = y => H - pad.b - (H - pad.t - pad.b) * y / yMax;
  g.strokeStyle = 'rgba(255,255,255,0.08)'; g.lineWidth = 1;
  g.font = '10px JetBrains Mono, monospace'; g.fillStyle = '#7e8aa3';
  for (let i = 0; i <= 2; i++) { const y = yMax * i / 2; g.beginPath(); g.moveTo(pad.l, Y(y)); g.lineTo(W - pad.r, Y(y)); g.stroke(); g.textAlign = 'right'; g.fillText(fmt(y / (opts.yDiv ?? 1), 0), pad.l - 4, Y(y) + 3); }
  g.textAlign = 'left'; g.fillText(opts.xLabel ?? '', pad.l, H - 4);
  g.textAlign = 'right'; g.fillText(opts.title ?? '', W - pad.r, 11);
  for (const s of series) {
    g.strokeStyle = s.color; g.lineWidth = 1.8; g.beginPath();
    s.data.forEach(([x, y], i) => i ? g.lineTo(X(x), Y(y)) : g.moveTo(X(x), Y(y)));
    g.stroke();
  }
  if (opts.marker != null) {
    g.strokeStyle = 'rgba(251,191,36,0.8)'; g.beginPath(); g.moveTo(X(opts.marker), pad.t); g.lineTo(X(opts.marker), H - pad.b); g.stroke();
  }
}
