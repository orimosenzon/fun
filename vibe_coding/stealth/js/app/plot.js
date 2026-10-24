// גרף קווי מינימלי על קנבס, משותף לשולחן התכנון ולאקדמיה
export function fitCanvas(cv) {
  const r = cv.getBoundingClientRect(), d = Math.min(2, window.devicePixelRatio || 1);
  const w = Math.max(10, Math.round(r.width * d)), h = Math.max(10, Math.round(r.height * d));
  if (cv.width !== w || cv.height !== h) { cv.width = w; cv.height = h; }
  const ctx = cv.getContext('2d'); ctx.setTransform(d, 0, 0, d, 0, 0);
  return { ctx, W: r.width, H: r.height };
}

const COLORS = { grid: '#1c2a46', axis: '#5a6d92', text: '#8a9bb9' };

function niceTicks(a, b, n = 5) {
  const span = b - a, step0 = span / n, mag = Math.pow(10, Math.floor(Math.log10(step0)));
  const st = [1, 2, 2.5, 5, 10].map((s) => s * mag).find((s) => span / s <= n) || mag * 10;
  const out = []; for (let v = Math.ceil(a / st) * st; v <= b + 1e-9; v += st) out.push(+v.toFixed(10));
  return out;
}

// o: { series:[{x:[],y:[],color,width,dash,label,fill}], xMin,xMax,yMin,yMax, xLog, xLabel, yLabel, marks:[{x,label,color}], hmarks, fmtX, fmtY }
export function plot(cv, o) {
  const { ctx, W, H } = fitCanvas(cv);
  ctx.clearRect(0, 0, W, H);
  const pad = { l: 52, r: 14, t: 14, b: 34 };
  const pw = W - pad.l - pad.r, ph = H - pad.t - pad.b;
  let xMin = o.xMin, xMax = o.xMax, yMin = o.yMin, yMax = o.yMax;
  if (xMin == null || xMax == null || yMin == null || yMax == null) {
    let a = Infinity, b = -Infinity, c = Infinity, d = -Infinity;
    for (const s of o.series) for (let i = 0; i < s.x.length; i++) {
      const y = s.y[i]; if (!isFinite(y)) continue;
      a = Math.min(a, s.x[i]); b = Math.max(b, s.x[i]); c = Math.min(c, y); d = Math.max(d, y);
    }
    if (xMin == null) xMin = a; if (xMax == null) xMax = b;
    if (yMin == null) yMin = c - 0.05 * (d - c || 1); if (yMax == null) yMax = d + 0.05 * (d - c || 1);
  }
  const lx = (x) => o.xLog ? Math.log10(x) : x;
  const X = (x) => pad.l + (lx(x) - lx(xMin)) / (lx(xMax) - lx(xMin)) * pw;
  const Y = (y) => pad.t + (1 - (y - yMin) / (yMax - yMin)) * ph;
  ctx.font = '11px JetBrains Mono, monospace'; ctx.direction = 'ltr';
  ctx.strokeStyle = COLORS.grid; ctx.lineWidth = 1; ctx.fillStyle = COLORS.text;
  // רשת
  const yt = niceTicks(yMin, yMax, Math.max(3, Math.round(ph / 45)));
  ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
  for (const v of yt) { ctx.beginPath(); ctx.moveTo(pad.l, Y(v)); ctx.lineTo(pad.l + pw, Y(v)); ctx.stroke(); ctx.fillText(o.fmtY ? o.fmtY(v) : String(v), pad.l - 6, Y(v)); }
  let xt;
  if (o.xLog) { xt = []; for (let e = Math.floor(Math.log10(xMin)); e <= Math.ceil(Math.log10(xMax)); e++) for (const m of [1, 2, 5]) { const v = m * 10 ** e; if (v >= xMin * 0.999 && v <= xMax * 1.001) xt.push(v); } }
  else xt = niceTicks(xMin, xMax, Math.max(3, Math.round(pw / 70)));
  ctx.textAlign = 'center'; ctx.textBaseline = 'top';
  for (const v of xt) { ctx.beginPath(); ctx.moveTo(X(v), pad.t); ctx.lineTo(X(v), pad.t + ph); ctx.stroke(); ctx.fillText(o.fmtX ? o.fmtX(v) : String(+v.toPrecision(4)), X(v), pad.t + ph + 5); }
  ctx.strokeStyle = COLORS.axis; ctx.strokeRect(pad.l, pad.t, pw, ph);
  ctx.save(); ctx.beginPath(); ctx.rect(pad.l, pad.t, pw, ph); ctx.clip();
  for (const m of o.hmarks || []) {
    ctx.strokeStyle = m.color || '#ffb547'; ctx.setLineDash([4, 4]); ctx.beginPath(); ctx.moveTo(pad.l, Y(m.y)); ctx.lineTo(pad.l + pw, Y(m.y)); ctx.stroke(); ctx.setLineDash([]);
    if (m.label) { ctx.fillStyle = m.color || '#ffb547'; ctx.textAlign = 'left'; ctx.textBaseline = 'bottom'; ctx.direction = 'rtl'; ctx.fillText(m.label, pad.l + pw - 4, Y(m.y) - 2); ctx.direction = 'ltr'; }
  }
  for (const s of o.series) {
    ctx.strokeStyle = s.color || '#4fd1ff'; ctx.lineWidth = s.width || 2; ctx.setLineDash(s.dash || []);
    ctx.beginPath(); let pen = false;
    for (let i = 0; i < s.x.length; i++) {
      const y = s.y[i]; if (!isFinite(y)) { pen = false; continue; }
      const px = X(s.x[i]), py = Y(y);
      if (!pen) { ctx.moveTo(px, py); pen = true; } else ctx.lineTo(px, py);
    }
    ctx.stroke(); ctx.setLineDash([]);
    if (s.fill) { ctx.lineTo(X(s.x[s.x.length - 1]), Y(yMin)); ctx.lineTo(X(s.x[0]), Y(yMin)); ctx.closePath(); ctx.fillStyle = s.fill; ctx.fill(); }
  }
  for (const m of o.marks || []) {
    ctx.strokeStyle = m.color || '#b59cff'; ctx.setLineDash([3, 3]); ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(X(m.x), pad.t); ctx.lineTo(X(m.x), pad.t + ph); ctx.stroke(); ctx.setLineDash([]);
    if (m.label) { ctx.fillStyle = m.color || '#b59cff'; ctx.textAlign = 'center'; ctx.textBaseline = 'top'; ctx.fillText(m.label, X(m.x), pad.t + 3); }
  }
  for (const pt of o.points || []) {
    ctx.fillStyle = pt.color || '#fff'; ctx.beginPath(); ctx.arc(X(pt.x), Y(pt.y), pt.r || 4, 0, 7); ctx.fill();
  }
  ctx.restore();
  // תוויות צירים וכותרות
  ctx.fillStyle = COLORS.text; ctx.font = '12px Rubik, sans-serif';
  if (o.xLabel) { ctx.direction = 'rtl'; ctx.textAlign = 'center'; ctx.textBaseline = 'bottom'; ctx.fillText(o.xLabel, pad.l + pw / 2, H - 1); }
  if (o.yLabel) { ctx.save(); ctx.translate(12, pad.t + ph / 2); ctx.rotate(-Math.PI / 2); ctx.direction = 'rtl'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillText(o.yLabel, 0, 0); ctx.restore(); }
  // מקרא
  const leg = o.series.filter((s) => s.label);
  if (leg.length) {
    ctx.direction = 'rtl'; ctx.textAlign = 'right'; ctx.textBaseline = 'middle'; ctx.font = '12px Rubik, sans-serif';
    let y = pad.t + 12;
    for (const s of leg) {
      const x = pad.l + pw - 8;
      ctx.fillStyle = 'rgba(7,11,20,.75)'; const tw = ctx.measureText(s.label).width; ctx.fillRect(x - tw - 30, y - 9, tw + 34, 18);
      ctx.strokeStyle = s.color; ctx.lineWidth = 3; ctx.setLineDash(s.dash || []); ctx.beginPath(); ctx.moveTo(x - 2, y); ctx.lineTo(x - 18, y); ctx.stroke(); ctx.setLineDash([]);
      ctx.fillStyle = '#dfe8f7'; ctx.fillText(s.label, x - 24, y); y += 19;
    }
  }
  ctx.direction = 'ltr';
  return { X, Y, pad, pw, ph };
}
