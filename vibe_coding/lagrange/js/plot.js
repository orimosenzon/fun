// Canvas helper: world coordinates, grid, scalar fields (heat map + contours), arrows, dragging.

export function token(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

export function rgb(color) {
  let h = color.replace('#', '');
  if (h.length === 3) h = [...h].map(c => c + c).join('');
  const n = parseInt(h.slice(0, 6), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

export function grad(fn, x, y, h = 1e-5) {
  return [(fn(x + h, y) - fn(x - h, y)) / (2 * h), (fn(x, y + h) - fn(x, y - h)) / (2 * h)];
}

export function niceStep(span, target = 8) {
  const raw = span / target, p = 10 ** Math.floor(Math.log10(raw)), m = raw / p;
  return (m < 1.5 ? 1 : m < 3.5 ? 2 : m < 7.5 ? 5 : 10) * p;
}

const HEB = /[\u0590-\u05FF]/;
const registry = new Set();
export function redrawAll() {
  for (const p of registry) { p.fieldCache = null; p.invalidate(); }
}

export class Plot {
  constructor(canvas, { view = [-1, 1, -1, 1], equal = true } = {}) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.view = view;
    this.equal = equal;
    this.draw = null;
    this.fieldCache = null;
    this.w = this.h = 0;
    registry.add(this);
    new ResizeObserver(() => this.resize()).observe(canvas);
  }

  setView(view) {
    this.view = view;
    this.fieldCache = null;
    this.layout();
    this.invalidate();
  }

  resize() {
    const r = this.canvas.getBoundingClientRect();
    if (!r.width || !r.height) return;
    const dpr = window.devicePixelRatio || 1;
    this.w = r.width; this.h = r.height;
    this.canvas.width = Math.round(r.width * dpr);
    this.canvas.height = Math.round(r.height * dpr);
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.fieldCache = null;
    this.layout();
    this.invalidate();
  }

  layout() {
    if (!this.w) return;
    let [x0, x1, y0, y1] = this.view;
    if (this.equal) {
      const s = Math.min(this.w / (x1 - x0), this.h / (y1 - y0));
      const cx = (x0 + x1) / 2, cy = (y0 + y1) / 2, hw = this.w / s / 2, hh = this.h / s / 2;
      x0 = cx - hw; x1 = cx + hw; y0 = cy - hh; y1 = cy + hh;
    }
    Object.assign(this, { x0, x1, y0, y1, sx: this.w / (x1 - x0), sy: this.h / (y1 - y0) });
  }

  X(x) { return (x - this.x0) * this.sx; }
  Y(y) { return this.h - (y - this.y0) * this.sy; }
  ix(px) { return this.x0 + px / this.sx; }
  iy(py) { return this.y0 + (this.h - py) / this.sy; }

  invalidate() {
    if (this.pending) return;
    this.pending = true;
    requestAnimationFrame(() => {
      this.pending = false;
      if (this.w && this.draw) {
        this.ctx.direction = 'ltr';
        this.draw(this);
      }
    });
  }

  clear() {
    this.ctx.fillStyle = token('--plot-bg');
    this.ctx.fillRect(0, 0, this.w, this.h);
  }

  grid({ axes = true, labels = true, stepX, stepY } = {}) {
    const c = this.ctx;
    const sx = stepX ?? niceStep(this.x1 - this.x0), sy = stepY ?? niceStep(this.y1 - this.y0);
    c.lineWidth = 1;
    c.strokeStyle = token('--grid');
    c.beginPath();
    for (let k = Math.ceil(this.x0 / sx); k * sx <= this.x1; k++) {
      const X = Math.round(this.X(k * sx)) + .5; c.moveTo(X, 0); c.lineTo(X, this.h);
    }
    for (let k = Math.ceil(this.y0 / sy); k * sy <= this.y1; k++) {
      const Y = Math.round(this.Y(k * sy)) + .5; c.moveTo(0, Y); c.lineTo(this.w, Y);
    }
    c.stroke();
    const zx = Math.min(Math.max(this.X(0), 0), this.w), zy = Math.min(Math.max(this.Y(0), 0), this.h);
    if (axes) {
      c.strokeStyle = token('--axis');
      c.beginPath();
      if (this.x0 <= 0 && this.x1 >= 0) { c.moveTo(Math.round(zx) + .5, 0); c.lineTo(Math.round(zx) + .5, this.h); }
      if (this.y0 <= 0 && this.y1 >= 0) { c.moveTo(0, Math.round(zy) + .5); c.lineTo(this.w, Math.round(zy) + .5); }
      c.stroke();
    }
    if (labels) {
      c.fillStyle = token('--muted');
      c.font = '11px system-ui, sans-serif';
      const lab = v => String(+v.toFixed(6)).replace('-', '−');
      const ly = Math.min(Math.max(zy + 13, 12), this.h - 4);
      c.textAlign = 'center'; c.textBaseline = 'alphabetic';
      for (let k = Math.ceil(this.x0 / sx); k * sx <= this.x1; k++) {
        if (k === 0) continue;
        const X = this.X(k * sx);
        if (X > 10 && X < this.w - 10) c.fillText(lab(k * sx), X, ly);
      }
      const lx = Math.min(Math.max(zx + 4, 4), this.w - 30);
      c.textAlign = 'left'; c.textBaseline = 'middle';
      for (let k = Math.ceil(this.y0 / sy); k * sy <= this.y1; k++) {
        if (k === 0) continue;
        const Y = this.Y(k * sy);
        if (Y > 8 && Y < this.h - 8) c.fillText(lab(k * sy), lx, Y);
      }
    }
  }

  path(pts, { stroke, width = 2, dash, fill, close = false, alpha = 1 } = {}) {
    if (pts.length < 2) return;
    const c = this.ctx;
    c.save();
    c.globalAlpha = alpha;
    c.beginPath();
    pts.forEach(([x, y], i) => i ? c.lineTo(this.X(x), this.Y(y)) : c.moveTo(this.X(x), this.Y(y)));
    if (close) c.closePath();
    if (fill) { c.fillStyle = fill; c.fill(); }
    if (stroke) {
      c.strokeStyle = stroke; c.lineWidth = width; c.lineJoin = 'round'; c.lineCap = 'round';
      c.setLineDash(dash || []);
      c.stroke();
    }
    c.restore();
  }

  // graph of y = f(x), broken where f is not finite
  fn(f, a, b, style, n = 400) {
    let seg = [];
    for (let i = 0; i <= n; i++) {
      const x = a + (b - a) * i / n, y = f(x);
      if (Number.isFinite(y)) seg.push([x, y]);
      else { this.path(seg, style); seg = []; }
    }
    this.path(seg, style);
  }

  // infinite line through (x, y) in direction (dx, dy)
  infLine(x, y, dx, dy, style) {
    const L = 4 * Math.hypot(this.x1 - this.x0, this.y1 - this.y0) / Math.hypot(dx, dy);
    this.path([[x - L * dx, y - L * dy], [x + L * dx, y + L * dy]], style);
  }

  arrow(x, y, vx, vy, { color, width = 2.5, head = 10, dash, label, labelColor } = {}) {
    const c = this.ctx;
    const X0 = this.X(x), Y0 = this.Y(y), X1 = this.X(x + vx), Y1 = this.Y(y + vy);
    const L = Math.hypot(X1 - X0, Y1 - Y0);
    if (L < 2) return;
    const ux = (X1 - X0) / L, uy = (Y1 - Y0) / L, hd = Math.min(head, L * 0.6);
    c.save();
    c.strokeStyle = c.fillStyle = color;
    c.lineWidth = width; c.lineCap = 'round';
    c.setLineDash(dash || []);
    c.beginPath(); c.moveTo(X0, Y0); c.lineTo(X1 - ux * hd * .8, Y1 - uy * hd * .8); c.stroke();
    c.setLineDash([]);
    c.beginPath();
    c.moveTo(X1, Y1);
    c.lineTo(X1 - ux * hd - uy * hd * .5, Y1 - uy * hd + ux * hd * .5);
    c.lineTo(X1 - ux * hd + uy * hd * .5, Y1 - uy * hd - ux * hd * .5);
    c.closePath(); c.fill();
    if (label) {
      c.font = '600 13px system-ui, sans-serif';
      c.textAlign = 'center'; c.textBaseline = 'middle';
      c.fillStyle = labelColor || color;
      c.direction = HEB.test(label) ? 'rtl' : 'ltr';
      c.fillText(label, X1 + ux * 14, Y1 + uy * 14);
    }
    c.restore();
  }

  dot(x, y, { r = 5, fill, stroke, width = 2 } = {}) {
    const c = this.ctx;
    c.beginPath(); c.arc(this.X(x), this.Y(y), r, 0, 2 * Math.PI);
    if (fill) { c.fillStyle = fill; c.fill(); }
    if (stroke) { c.strokeStyle = stroke; c.lineWidth = width; c.stroke(); }
  }

  text(s, x, y, { color, align = 'left', base = 'middle', font = '12px system-ui, sans-serif', dx = 0, dy = 0 } = {}) {
    const c = this.ctx;
    c.font = font; c.fillStyle = color || token('--ink');
    c.direction = HEB.test(s) ? 'rtl' : 'ltr';
    c.textAlign = align; c.textBaseline = base;
    c.fillText(s, this.X(x) + dx, this.Y(y) + dy);
  }

  // samples fn on a pixel grid; cached until the view, size or key changes
  field(key, fn, cell = 4) {
    if (this.fieldCache && this.fieldCache.key === key) return this.fieldCache;
    const nx = Math.ceil(this.w / cell), ny = Math.ceil(this.h / cell);
    const v = new Float64Array((nx + 1) * (ny + 1));
    let min = Infinity, max = -Infinity;
    for (let j = 0; j <= ny; j++) {
      const y = this.iy(j * cell);
      for (let i = 0; i <= nx; i++) {
        const z = fn(this.ix(i * cell), y);
        v[j * (nx + 1) + i] = z;
        if (z < min) min = z;
        if (z > max) max = z;
      }
    }
    return (this.fieldCache = { key, nx, ny, cell, v, min, max, img: null });
  }

  heat(F, { lo = token('--heat-lo'), hi = token('--heat-hi') } = {}) {
    if (!F.img) {
      const W = F.nx + 1, H = F.ny + 1;
      const off = document.createElement('canvas');
      off.width = W; off.height = H;
      const octx = off.getContext('2d'), im = octx.createImageData(W, H);
      const a = rgb(lo), b = rgb(hi), span = F.max - F.min || 1;
      for (let k = 0; k < W * H; k++) {
        const t = Math.min(1, Math.max(0, (F.v[k] - F.min) / span));
        im.data[4 * k] = a[0] + (b[0] - a[0]) * t;
        im.data[4 * k + 1] = a[1] + (b[1] - a[1]) * t;
        im.data[4 * k + 2] = a[2] + (b[2] - a[2]) * t;
        im.data[4 * k + 3] = 255;
      }
      octx.putImageData(im, 0, 0);
      F.img = off;
    }
    const c = this.ctx;
    c.imageSmoothingEnabled = true;
    c.drawImage(F.img, -F.cell / 2, -F.cell / 2, (F.nx + 1) * F.cell, (F.ny + 1) * F.cell);
  }

  // marching squares
  contours(F, levels, { color, width = 1, alpha = 1, dash } = {}) {
    const { nx, ny, cell, v } = F, W = nx + 1, c = this.ctx;
    c.save();
    c.globalAlpha = alpha; c.strokeStyle = color; c.lineWidth = width;
    c.setLineDash(dash || []);
    c.beginPath();
    const pts = [];
    for (const L of levels) {
      for (let j = 0; j < ny; j++) {
        for (let i = 0; i < nx; i++) {
          const a = v[j * W + i] - L, b = v[j * W + i + 1] - L;
          const d = v[(j + 1) * W + i] - L, e = v[(j + 1) * W + i + 1] - L;
          pts.length = 0;
          if ((a > 0) !== (b > 0)) pts.push(i + a / (a - b), j);
          if ((b > 0) !== (e > 0)) pts.push(i + 1, j + b / (b - e));
          if ((e > 0) !== (d > 0)) pts.push(i + 1 - e / (e - d), j + 1);
          if ((d > 0) !== (a > 0)) pts.push(i, j + 1 - d / (d - a));
          for (let k = 0; k + 3 < pts.length; k += 4) {
            c.moveTo(pts[k] * cell, pts[k + 1] * cell);
            c.lineTo(pts[k + 2] * cell, pts[k + 3] * cell);
          }
        }
      }
    }
    c.stroke();
    c.restore();
  }

  onDrag({ down, move, up, hover }) {
    const cv = this.canvas;
    let active = false;
    const pos = e => {
      const r = cv.getBoundingClientRect(), px = e.clientX - r.left, py = e.clientY - r.top;
      return [this.ix(px), this.iy(py), px, py];
    };
    cv.addEventListener('pointerdown', e => {
      const [x, y, px, py] = pos(e);
      if (down && down(x, y, px, py) === false) return;
      active = true;
      cv.setPointerCapture(e.pointerId);
      move && move(x, y);
      e.preventDefault();
    });
    cv.addEventListener('pointermove', e => {
      const [x, y, px, py] = pos(e);
      if (active) move && move(x, y);
      else if (hover) cv.style.cursor = hover(x, y, px, py) ? 'grab' : 'default';
    });
    const end = () => { if (active) { active = false; up && up(); } };
    cv.addEventListener('pointerup', end);
    cv.addEventListener('pointercancel', end);
  }
}

export function levelsBetween(min, max, n) {
  const out = [];
  for (let k = 1; k <= n; k++) out.push(min + (max - min) * k / (n + 1));
  return out;
}
