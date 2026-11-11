// Graphics primitives of Karateka, ported from the original 6502 routines.
// The port keeps the original memory layout (`mem`, 64K) so that every routine can be
// checked byte-for-byte against the original running in the reference emulator
// (tools/difftest.mjs). Addresses in comments are the original entry points.
//
// Zero-page parameter block used by all drawing routines:
//   $03/$04 shape pointer      $05 x (byte column, signed)   $06 y (top row)
//   $07 page ($20 or $40)      $0F mode (0 = OR, <0 = AND, >0 = replace)
//   $10 bit shift (0..6)       $0D/$0E height / width of the last shape drawn
// Shape format: [height, widthBytes, height*widthBytes bytes of 7-bit pixels].

export const ZP = { PTR: 0x03, X: 0x05, Y: 0x06, PAGE: 0x07, Y2: 0x08, X2: 0x09, H: 0x0d, W: 0x0e, MODE: 0x0f, SHIFT: 0x10 };

const ROW_LO = 0x0800, ROW_HI = 0x08c0, FLIP = 0x0900, MASK_LO = 0x1a34, MASK_HI = 0x1a3b;

export class Gfx {
  constructor(mem) { this.mem = mem; this.carry = 0; }

  rowAddr(row, col) {
    const m = this.mem, y = row & 0xff;
    return (((m[ROW_HI + y] + m[ZP.PAGE]) & 0xff) << 8) | ((m[ROW_LO + y] + col) & 0xff);
  }

  // $1A84: shift a 7-bit pixel byte left by `s`; the bits pushed out go to this.carry.
  shift(a, s) {
    if (s === 0) { this.carry = 0; return a; }
    this.carry = (a & 0x7f) >> (7 - s);
    return (a << s) & 0x7f;
  }

  header() { // $1A61
    const m = this.mem, p = m[3] | (m[4] << 8);
    m[0x1b] = m[3]; m[0x1c] = m[4];
    m[ZP.H] = m[p]; m[ZP.W] = m[p + 1];
    return p + 2;
  }

  // $1900: clear the page to black (palette bit set)
  clear() { const base = this.mem[ZP.PAGE] << 8; this.mem.fill(0x80, base, base + 0x2000); }

  // $1903: draw shape with its left edge at column x
  blit() { const src = this.header(); this.blitBody(src); }

  // $1906: draw shape with its right edge at column x
  blitRight() {
    const m = this.mem, src = this.header(), x = m[ZP.X];
    m[ZP.X] = (x - m[ZP.W] + 1) & 0xff;
    this.blitBody(src);
    m[ZP.X] = x;
  }

  // $1909: draw shape mirrored, left edge at x
  blitMirror() { const src = this.header(); this.mirrorBody(src); }

  // $190C: draw shape mirrored, right edge at x
  blitMirrorRight() {
    const m = this.mem, src = this.header(), x = m[ZP.X];
    m[ZP.X] = (x - m[ZP.W] + 1) & 0xff;
    this.mirrorBody(src);
    m[ZP.X] = x;
  }

  op(v, scr) { const mode = this.mem[ZP.MODE]; return mode === 0 ? v | scr : mode & 0x80 ? v & scr : v; }

  blitBody(src) { // $1927
    const m = this.mem, s = m[ZP.SHIFT], mode = m[ZP.MODE];
    const fullW = m[ZP.W];
    let w = fullW, col = 0, skip = 0;
    const x = m[ZP.X];
    m[0x15] = fullW; m[0x14] = 0; m[0x16] = 0; m[0x1a] = 0;
    if (x & 0x80) {
      skip = (-x) & 0xff; m[0x16] = skip;
      const t = x + w;
      if (t < 0x100) return;
      w = t & 0xff; m[ZP.W] = w;
    } else {
      if (x >= 40) return;
      col = x; m[0x14] = x;
      if (x + w >= 40) { w = 40 - x; m[ZP.W] = w; }
    }
    const trailing = ((x + fullW) & 0xff) < 40;
    for (let row = 0; row < m[ZP.H]; row++) {
      const addr = this.rowAddr(m[ZP.Y] + row, col);
      src += skip;
      let carry, y = 0, doLoop = true;
      if (skip) {
        this.shift(m[src - 1], s); carry = this.carry;
        if (w === 0) doLoop = false;
      } else carry = m[addr] & m[MASK_LO + s];
      let skipTrail = false;
      if (doLoop) {
        do {
          const v = ((this.shift(m[src + y], s) ^ carry) | 0x80);
          carry = this.carry;
          m[addr + y] = this.op(v, m[addr + y]);
          y++;
        } while (y < w);
        if (!trailing) skipTrail = true;
      }
      if (!skipTrail) {
        const scr = m[addr + y];
        let v;
        if (mode & 0x80) v = (m[MASK_HI + s] | carry) & scr;
        else if (mode === 0) v = scr | carry | 0x80;
        else v = (scr & m[MASK_HI + s]) | carry | 0x80;
        m[addr + y] = v;
      }
      src += fullW - skip;
    }
  }

  mirrorBody(src) { // $1AF4
    const m = this.mem, mode = m[ZP.MODE];
    const savedX = m[ZP.X], savedShift = m[ZP.SHIFT];
    if (savedShift) m[ZP.SHIFT] = 7 - savedShift; else m[ZP.X] = (savedX - 1) & 0xff;
    const s = m[ZP.SHIFT], x = m[ZP.X];
    const fullW = m[ZP.W];
    let w = fullW, col = 0, skip = 0, clip = 0;
    m[0x15] = fullW; m[0x14] = 0; m[0x16] = 0; m[0x1a] = 0; m[0x1d] = 0;
    let visible = true;
    if (x & 0x80) {
      w = (w + 1) & 0xff; m[ZP.W] = w;
      const t = x + w;
      if (t < 0x100 || (t & 0xff) === 0) visible = false;
      else { w = t & 0xff; m[ZP.W] = w; clip = 1; }
    } else if (x >= 40) visible = false;
    else {
      col = x; m[0x14] = x;
      if (x + w >= 40) {
        clip = 0xff;
        w = 40 - x; m[ZP.W] = w;
        if (w === 0) visible = false; else { skip = fullW - w; m[0x16] = skip; }
      }
    }
    if (visible) {
      for (let row = 0; row < m[ZP.H]; row++) {
        let addr = this.rowAddr(m[ZP.Y] + row, col);
        if (clip === 1) addr = (addr - 1) & 0xffff;
        src += skip;
        let si = 0, y = w, carry, doLoop = true;
        if (clip === 0xff) {
          this.shift(m[src], s); carry = this.carry;
          si = 1; y = (y - 1) & 0xff;
          if (y === 0) doLoop = false;
        } else carry = m[FLIP + (m[addr + y] | 0x80)] & m[MASK_LO + s];
        if (doLoop) {
          do {
            const v = (this.shift(m[src + si], s) ^ carry) | 0x80;
            carry = this.carry;
            si++;
            m[addr + y] = this.op(m[FLIP + v], m[addr + y]);
            y = (y - 1) & 0xff;
          } while (y !== 0);
        }
        if (!(doLoop && clip === 1)) {
          const scr = m[addr + y];
          let v;
          if (mode & 0x80) v = m[FLIP + (m[MASK_HI + s] | carry | 0x80)] & scr;
          else if (mode === 0) v = m[FLIP + (carry | 0x80)] | scr;
          else v = m[FLIP + (carry | 0x80)] | (m[FLIP + (m[MASK_HI + s] | 0x80)] & scr);
          m[addr + y] = v;
        }
        src += fullW - skip;
      }
    }
    m[ZP.SHIFT] = savedShift; m[ZP.X] = savedX;
  }

  // $0A8F: clip the fill rectangle; returns false when nothing is visible
  fillSetup() {
    const m = this.mem;
    if (m[ZP.X] >= 40) { if (m[ZP.X2] >= 41) return false; m[ZP.X] = 0; }
    else if (m[ZP.X2] >= 41) m[ZP.X2] = 40;
    m[ZP.H] = (m[ZP.Y2] - m[ZP.Y] - 1) & 0xff;
    m[ZP.W] = (m[ZP.X2] - m[ZP.X] - 1) & 0xff;
    return true;
  }

  // $0A00 / $0A03: fill columns [x, x2) of rows [y, y2) with a two-byte pattern
  // ($02 for even columns, $11 for odd). fill2 uses $12/$13 on odd rows.
  fill(alt = false) {
    const m = this.mem;
    if (!this.fillSetup()) return;
    let r = m[ZP.H];
    do {
      const addr = this.rowAddr(r + m[ZP.Y], m[ZP.X]);
      const odd = alt && (r & 1);
      const even = odd ? m[0x12] : m[0x02], oddc = odd ? m[0x13] : m[0x11];
      for (let y = m[ZP.W]; y < 0x80; y = (y - 1) & 0xff) m[addr + y] = (y & 1) ? oddc : even;
      r = (r - 1) & 0xff;
    } while (r < 0xc0);
  }

  // $0A06: switch off the rightmost pixel (bit 6 of column 39) on every row of the page
  clearRightEdge() {
    const m = this.mem, base = m[ZP.PAGE] << 8;
    for (let row = 0; row < 192; row++) m[base + (m[ROW_HI + row] << 8) + m[ROW_LO + row] + 39] &= 0xbf;
  }

  // $0780: scroll up two rows: row r of the draw page ($07) = row r+2 of page $E4
  scrollUp() {
    const m = this.mem, dst = m[ZP.PAGE] << 8, src = m[0xe4] << 8;
    for (let row = 0; row < 0xbe; row++) {
      const d = dst + (m[ROW_HI + row] << 8) + m[ROW_LO + row], s = src + (m[ROW_HI + row + 2] << 8) + m[ROW_LO + row + 2];
      m.copyWithin(d, s, s + 40);
    }
  }
}
