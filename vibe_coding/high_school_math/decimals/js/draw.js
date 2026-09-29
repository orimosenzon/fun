'use strict';
/* ציורים משותפים: טבלת שוקולד, ריבוע מאה, קוביות של לוח המקומות, ציר מספרים עם זום */

const COL = { p0: '#7c3aed', pm1: '#2563eb', pm2: '#ea580c', pm3: '#db2777', choc: '#6b3e26', chocD: '#4a2a18', chocL: '#8a5636' };

/* טבלת שוקולד: ריבוע עם 10 פסים עומדים. כל פס הוא עשירית.
   filled: כמה פסים יש. opt.onPick(i): לחיצה על פס i */
function drawBar(g, x, y, s, filled, opt = {}) {
  const w = s / 10, gap = Math.max(1.5, s / 90);
  const grp = S.el('g', { class: 'bar' }, g);
  S.el('rect', { x: x - 3, y: y - 3, width: s + 6, height: s + 6, rx: 8, class: 'bar-frame' }, grp);
  for (let i = 0; i < 10; i++) {
    const on = i < filled;
    const r = S.el('rect', { x: x + i * w + gap / 2, y, width: w - gap, height: s, rx: 3, class: on ? 'piece on' : 'piece off' }, grp);
    if (on) S.el('rect', { x: x + i * w + gap / 2 + 2, y: y + 3, width: Math.max(1, w - gap - 7), height: s * .08, rx: 2, class: 'piece-shine' }, grp);
    if (opt.onPick) {
      const hit = S.el('rect', { x: x + i * w, y, width: w, height: s, class: 'hitbox' }, grp);
      hit.addEventListener('pointerdown', e => { e.preventDefault(); opt.onPick(i); });
    }
  }
  return grp;
}

/* ריבוע מאה: 10 עמודות (עשיריות) על 10 שורות (מאיות).
   תא i הוא עמודה floor(i/10), שורה מלמטה i%10. cells: מערך בוליאני של 100 */
function gridCell(i) { return { c: Math.floor(i / 10), r: 9 - (i % 10) }; }
function drawGrid(g, x, y, s, cells, opt = {}) {
  const w = s / 10;
  const grp = S.el('g', { class: 'grid100' }, g);
  S.el('rect', { x: x - 2, y: y - 2, width: s + 4, height: s + 4, rx: 6, class: 'grid-frame' }, grp);
  const full = [];
  for (let c = 0; c < 10; c++) full[c] = cells.slice(c * 10, c * 10 + 10).every(Boolean);
  for (let i = 0; i < 100; i++) {
    const { c, r } = gridCell(i);
    let cls = 'cell';
    if (cells[i]) cls += opt.byRole ? (full[c] ? ' tenth' : ' hund') : (opt.color ? ' ' + opt.color : ' hund');
    S.el('rect', { x: x + c * w + .6, y: y + r * w + .6, width: w - 1.2, height: w - 1.2, rx: 1.5, class: cls, 'data-i': i }, grp);
  }
  // קווים עבים בין העמודות כדי שיהיה ברור שכל עמודה היא עשירית
  for (let c = 1; c < 10; c++) S.el('line', { x1: x + c * w, y1: y, x2: x + c * w, y2: y + s, class: 'grid-col' }, grp);
  return grp;
}
// מערך תאים מלא לפי מספר מאיות, מסודר: עמודות מלאות קודם
const cellsOf = n => Array.from({ length: 100 }, (_, i) => i < n);

/* קוביות: שלם = ריבוע גדול, עשירית = מוט, מאית = קובייה קטנה */
function blockFlat(g, x, y, s, cls = '') {
  const grp = S.el('g', { class: 'blk flat ' + cls }, g);
  S.el('rect', { x, y, width: s, height: s, rx: 2, class: 'blk-body' }, grp);
  for (let k = 1; k < 10; k++) {
    S.el('line', { x1: x + k * s / 10, y1: y, x2: x + k * s / 10, y2: y + s, class: 'blk-line' }, grp);
    S.el('line', { x1: x, y1: y + k * s / 10, x2: x + s, y2: y + k * s / 10, class: 'blk-line faint' }, grp);
  }
  return grp;
}
function blockRod(g, x, y, s, cls = '') {
  const grp = S.el('g', { class: 'blk rod ' + cls }, g);
  S.el('rect', { x, y, width: s / 10, height: s, rx: 1.5, class: 'blk-body' }, grp);
  for (let k = 1; k < 10; k++) S.el('line', { x1: x, y1: y + k * s / 10, x2: x + s / 10, y2: y + k * s / 10, class: 'blk-line faint' }, grp);
  return grp;
}
function blockCube(g, x, y, s, cls = '') {
  const grp = S.el('g', { class: 'blk cube ' + cls }, g);
  S.el('rect', { x, y, width: s / 10, height: s / 10, rx: 1, class: 'blk-body' }, grp);
  return grp;
}

// רוחב הציר: בטלפון ציר צר יותר, כדי שהמספרים והסיכות לא יהיו זעירים
const LW = () => (innerWidth < 640 ? 440 : 800);

/* ציר מספרים עם זכוכית מגדלת.
   opt: { W, H, y, a, b, onTap(v), pins: [] } */
class NumberLine {
  constructor(svg, opt) {
    this.svg = svg; this.o = Object.assign({ W: 800, y: 90, pad: 34 }, opt);
    this.a = opt.a ?? 0; this.b = opt.b ?? 1;
    this.gTicks = S.el('g', {}, svg);
    this.gOver = S.el('g', {}, svg);
    this.gPins = S.el('g', {}, svg);
    this.pins = [];
    this.render();
  }
  get x0() { return this.o.pad; }
  get x1() { return this.o.W - this.o.pad; }
  X(v) { return this.x0 + (v - this.a) / (this.b - this.a) * (this.x1 - this.x0); }
  V(x) { return this.a + (x - this.x0) / (this.x1 - this.x0) * (this.b - this.a); }

  render() {
    const g = this.gTicks; S.clear(g);
    const { y } = this.o, R = this.b - this.a, W = this.x1 - this.x0;
    S.el('line', { x1: this.x0 - 14, y1: y, x2: this.x1 + 14, y2: y, class: 'nl-axis' }, g);
    const top = Math.floor(Math.log10(R) + 1e-9);
    const seen = new Set();
    for (let k = top; k >= top - 3; k--) {
      const t = 10 ** k, px = t / R * W;
      if (px < 3.5) continue;
      const alpha = clamp((px - 3.5) / 8, 0, 1);
      const start = Math.ceil(this.a / t - 1e-9), end = Math.floor(this.b / t + 1e-9);
      if (end - start > 1200) continue;
      for (let j = start; j <= end; j++) {
        const v = j * t, key = Math.round(v * 1e6);
        if (seen.has(key)) continue; seen.add(key);
        const x = this.X(v);
        const hgt = clamp(px / 7, 4, 16);
        S.el('line', { x1: x, y1: y - hgt, x2: x, y2: y + hgt, class: 'nl-tick', opacity: alpha }, g);
        if (px >= 34 || (px >= 26 && this.o.dense)) {
          const big = k === top || key % 1e6 === 0;
          S.text(g, x, y + hgt + 20, N.str(key), { class: 'nl-lbl' + (big ? ' big' : ''), 'text-anchor': 'middle', opacity: clamp((px - 26) / 14, 0, 1) });
        }
      }
    }
    this.o.onRender && this.o.onRender(this);
    this.drawPins();
  }

  // מעבר חלק לטווח חדש
  zoomTo(a, b, ms = 750) {
    const a0 = this.a, b0 = this.b, t0 = performance.now();
    // אינטרפולציה לוגריתמית של הרוחב, כדי שהזום ירגיש אחיד
    const la = Math.log(b0 - a0), lb = Math.log(b - a);
    const c0 = (a0 + b0) / 2, c1 = (a + b) / 2;
    return new Promise(res => {
      const f = now => {
        let t = clamp((now - t0) / ms, 0, 1); const e = t < .5 ? 2 * t * t : 1 - (-2 * t + 2) ** 2 / 2;
        const w = Math.exp(la + (lb - la) * e), c = c0 + (c1 - c0) * e;
        this.a = c - w / 2; this.b = c + w / 2;
        if (t >= 1) { this.a = a; this.b = b; }
        this.render();
        if (t < 1) requestAnimationFrame(f); else res();
      };
      requestAnimationFrame(f);
    });
  }

  setPins(pins) { this.pins = pins; this.drawPins(); }
  drawPins() {
    const g = this.gPins; S.clear(g); const { y } = this.o;
    for (const p of this.pins) {
      if (p.v < this.a - (this.b - this.a) * .02 || p.v > this.b + (this.b - this.a) * .02) continue;
      const x = this.X(p.v), up = p.up ?? true;
      const pg = S.el('g', { class: 'pin ' + (p.cls || '') }, g);
      if (up) {
        S.el('line', { x1: x, y1: y, x2: x, y2: y - 40, class: 'pin-stem' }, pg);
        S.el('circle', { cx: x, cy: y - 48, r: 11, class: 'pin-head' }, pg);
        if (p.label) S.text(pg, x, y - 66, p.label, { class: 'pin-lbl', 'text-anchor': 'middle' });
      } else {
        S.el('line', { x1: x, y1: y, x2: x, y2: y + 40, class: 'pin-stem' }, pg);
        S.el('path', { d: `M${x} ${y + 34} l-9 16 h18 z`, class: 'pin-head' }, pg);
        if (p.label) S.text(pg, x, y + 68, p.label, { class: 'pin-lbl', 'text-anchor': 'middle' });
      }
      if (p.drag) {
        pg.classList.add('draggable');
        S.el('circle', { cx: x, cy: up ? y - 40 : y + 40, r: 30, class: 'pin-hit' }, pg);
        pg.addEventListener('pointerdown', e => this.startDrag(e, p));
      }
    }
  }
  startDrag(e, p) {
    e.preventDefault();
    const svg = this.svg; svg.setPointerCapture && svg.setPointerCapture(e.pointerId);
    const move = ev => {
      const x = clamp(S.pt(svg, ev).x, this.x0, this.x1);
      let v = this.V(x);
      if (p.snap) v = Math.round(v / p.snap) * p.snap;
      p.v = clamp(v, this.a, this.b); this.drawPins(); p.onMove && p.onMove(p.v);
    };
    const up = () => { svg.removeEventListener('pointermove', move); svg.removeEventListener('pointerup', up); svg.removeEventListener('pointercancel', up); p.onDrop && p.onDrop(p.v); };
    svg.addEventListener('pointermove', move); svg.addEventListener('pointerup', up); svg.addEventListener('pointercancel', up);
    move(e);
  }
}
