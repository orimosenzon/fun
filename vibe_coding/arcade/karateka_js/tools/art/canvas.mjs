// A canvas for drawing the game's pictures, in the terms of the screen they are made for.
//
// The screen is 280 pixels by 192 rows, one bit per pixel, stored seven pixels to a byte. A lit
// pixel with a lit neighbour is white; a lone lit pixel is blue in an even column and orange in an
// odd one; an unlit pixel between two lit ones takes their colour too. So colours here are painted
// as bit patterns: 'w' lit, 'k' unlit, 'b' lit in even columns, 'o' lit in odd columns.
// A cell that was never painted is "clear": unlit when the shape replaces what is on the screen,
// and transparent when it is laid over it.
export class Canvas {
  constructor(w = 280, h = 192, x0 = 0, y0 = 0) { this.w = w; this.h = h; this.x0 = x0; this.y0 = y0; this.px = new Int8Array(w * h).fill(-1); }
  // x, y are in the picture's own coordinates; x0/y0 say which of them is the canvas's corner (x0 fixes the colours' columns)
  put(x, y, c) {
    x = Math.round(x) - this.x0; y = Math.round(y) - this.y0;
    if (x < 0 || y < 0 || x >= this.w || y >= this.h) return;
    const X = x + this.x0;
    this.px[y * this.w + x] = c === 'w' ? 1 : c === 'k' ? 0 : c === 'b' ? (X & 1 ? 0 : 1) : c === 'o' ? (X & 1 ? 1 : 0) : c === '.' ? -1 : c;
  }
  get(x, y) { x -= this.x0; y -= this.y0; return x < 0 || y < 0 || x >= this.w || y >= this.h ? -1 : this.px[y * this.w + x]; }
  rect(x1, y1, x2, y2, c) { for (let y = y1; y < y2; y++) for (let x = x1; x < x2; x++) this.put(x, y, c); }
  // c may be a function (x, y) => colour, for patterns and shading
  paint(test, c, box = [this.x0, this.y0, this.x0 + this.w, this.y0 + this.h]) {
    for (let y = box[1]; y < box[3]; y++) for (let x = box[0]; x < box[2]; x++) if (test(x, y)) this.put(x, y, typeof c === 'function' ? c(x, y) : c);
  }
  poly(points, c) {
    const ys = points.map((p) => p[1]), xs = points.map((p) => p[0]);
    this.paint((x, y) => { let inside = false; for (let i = 0, j = points.length - 1; i < points.length; j = i++) { const [xi, yi] = points[i], [xj, yj] = points[j]; if ((yi > y + 0.5) !== (yj > y + 0.5) && x + 0.5 < (xj - xi) * (y + 0.5 - yi) / (yj - yi) + xi) inside = !inside; } return inside; },
      c, [Math.floor(Math.min(...xs)), Math.floor(Math.min(...ys)), Math.ceil(Math.max(...xs)) + 1, Math.ceil(Math.max(...ys)) + 1]);
  }
  line(x1, y1, x2, y2, c, thick = 1) {
    const n = Math.max(Math.abs(x2 - x1), Math.abs(y2 - y1), 1) * 2;
    for (let i = 0; i <= n; i++) { const x = x1 + (x2 - x1) * i / n, y = y1 + (y2 - y1) * i / n; for (let a = 0; a < thick; a++) for (let b = 0; b < thick; b++) this.put(x + a, y + b, c); }
  }
  // rows of text: '#' white, 'k' black, 'b' blue, 'o' orange, anything else left alone
  stamp(x, y, rows) { rows.forEach((r, j) => [...r].forEach((ch, i) => { const c = ch === '#' ? 'w' : 'kbo'.includes(ch) ? ch : null; if (c) this.put(x + i, y + j, c); })); }

  // the smallest box of whole bytes around what was painted: [first byte column, first row, bytes wide, rows]
  bounds() {
    let x1 = 1e9, x2 = -1, y1 = 1e9, y2 = -1;
    for (let y = 0; y < this.h; y++) for (let x = 0; x < this.w; x++) if (this.px[y * this.w + x] >= 0) { x1 = Math.min(x1, x); x2 = Math.max(x2, x); y1 = Math.min(y1, y); y2 = Math.max(y2, y); }
    if (x2 < 0) return null;
    const c1 = Math.floor((x1 + this.x0) / 7), c2 = Math.floor((x2 + this.x0) / 7);
    return [c1, y1 + this.y0, c2 - c1 + 1, y2 - y1 + 1];
  }
  // A shape in the game's format: rows, bytes per row, then the rows; bit 0 of a byte is its leftmost pixel.
  // col is the byte column of the shape's left edge. litOnly: clear cells count as unlit (for shapes laid over the screen);
  // mask: the shape for clearing (AND): every painted cell becomes 0, everything else 1.
  shape(col, row, wide, rows, { mask = false } = {}) {
    const out = [rows, wide];
    for (let y = row; y < row + rows; y++) for (let b = 0; b < wide; b++) {
      let v = 0;
      for (let i = 0; i < 7; i++) { const p = this.get((col + b) * 7 + i, y); if (mask ? p < 0 : p > 0) v |= 1 << i; }
      out.push(v);
    }
    return out;
  }
}

export const hex = (bytes) => bytes.map((v) => (v & 0xff).toString(16).padStart(2, '0')).join('');

// Writing a data file: [[address, hex], ...]
export function dataFile(note, entries) {
  return `// ${note}\n// Made by tools/art (node tools/art/build.mjs). Do not edit: change the drawing code.\nexport default [\n${entries.map(([at, bytes, what]) => `  [0x${at.toString(16).padStart(4, '0')}, '${hex(bytes)}'],${what ? '   // ' + what : ''}`).join('\n')}\n];\n`;
}
