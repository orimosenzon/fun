// VIC-II video chip, rendered one raster line at a time.
// All five graphics modes, 8 sprites with expansion, multicolour, priority and
// both collision registers, raster interrupt, borders, bad-line cycle stealing.
// A line is drawn whole, with the registers as they are near its start (no cycle-exact splits).

export const MODELS = {
  // lines per frame, CPU cycles per line, CPU clock, first visible line, visible rows, pixel aspect
  ntsc: { name: 'ntsc', lines: 263, cpl: 65, hz: 1022727, firstLine: 27, rows: 248, par: 0.75 },
  pal: { name: 'pal', lines: 312, cpl: 63, hz: 985248, firstLine: 15, rows: 272, par: 0.9365 },
};

export const WIDTH = 384;      // 32 px border + 320 + 32 px border
const LEFT = 32;               // x of the 40-column window inside a row
const BUF = 640;
const DRAW_AT = 14;           // CPU cycle of a line at which it is drawn

// Colodore palette (Pepto's successor), RGB.
export const PALETTE = [
  [0x00, 0x00, 0x00], [0xff, 0xff, 0xff], [0x81, 0x33, 0x38], [0x75, 0xce, 0xc8],
  [0x8e, 0x3c, 0x97], [0x56, 0xac, 0x4d], [0x2e, 0x2c, 0x9b], [0xed, 0xf1, 0x71],
  [0x8e, 0x50, 0x29], [0x55, 0x38, 0x00], [0xc4, 0x6c, 0x71], [0x4a, 0x4a, 0x4a],
  [0x7b, 0x7b, 0x7b], [0xa9, 0xff, 0x9f], [0x70, 0x6d, 0xeb], [0xb2, 0xb2, 0xb2],
];

export class VIC {
  // mem: { ram, chargen, colorRam }, bank(): 0, 0x4000, 0x8000 or 0xc000
  constructor(model, mem) {
    this.model = model;
    this.mem = mem;
    this.bank = 0;
    this.reg = new Uint8Array(64);
    this.fb = new Uint8Array(WIDTH * model.rows);
    this.lineCol = new Uint8Array(BUF);
    this.lineFg = new Uint8Array(BUF);
    this.lineSpr = new Uint8Array(BUF);
    this.sprDma = new Uint8Array(8);
    this.sprMc = new Uint8Array(8);
    this.sprHalf = new Uint8Array(8);
    this.reset();
  }

  reset() {
    this.reg.fill(0);
    this.raster = 0; this.lineCycle = 0;
    this.vcbase = 0; this.rc = 7; this.display = false;
    this.denLatch = false; this.vborder = true;
    this.irqStatus = 0; this.irqMask = 0;
    this.collSS = 0; this.collSB = 0;
    this.sprDma.fill(0); this.sprMc.fill(0); this.sprHalf.fill(0);
    this.frameReady = false; this.frames = 0; this.drawn = false;
  }

  get irq() { return (this.irqStatus & this.irqMask) !== 0; }
  get rasterCompare() { return this.reg[0x12] | ((this.reg[0x11] & 0x80) << 1); }

  read(r) {
    r &= 0x3f;
    switch (r) {
      case 0x11: return (this.reg[0x11] & 0x7f) | ((this.raster & 0x100) >> 1);
      case 0x12: return this.raster & 0xff;
      case 0x13: case 0x14: return 0;
      case 0x16: return this.reg[r] | 0xc0;
      case 0x18: return this.reg[r] | 0x01;
      case 0x19: return this.irqStatus | 0x70 | (this.irq ? 0x80 : 0);
      case 0x1a: return this.irqMask | 0xf0;
      case 0x1e: { const v = this.collSS; this.collSS = 0; return v; }
      case 0x1f: { const v = this.collSB; this.collSB = 0; return v; }
      default:
        if (r >= 0x2f) return 0xff;
        if (r >= 0x20) return this.reg[r] | 0xf0;
        return this.reg[r];
    }
  }

  write(r, v) {
    r &= 0x3f;
    if (r >= 0x2f) return;
    switch (r) {
      case 0x19: this.irqStatus &= ~(v & 0x0f); return;
      case 0x1a: this.irqMask = v & 0x0f; return;
      case 0x1e: case 0x1f: return;
      case 0x11: case 0x12: {
        const before = this.rasterCompare;
        this.reg[r] = v;
        if (this.rasterCompare !== before && this.rasterCompare === this.raster) this.irqStatus |= 1;
        if (this.raster === 0x30 && (this.reg[0x11] & 0x10)) this.denLatch = true;
        return;
      }
      default: this.reg[r] = v;
    }
  }

  // Advance by c CPU cycles. Returns the cycles the VIC took from the CPU.
  // A line is drawn when the beam reaches its left edge, with the registers as they are then:
  // a raster interrupt handler that writes later in the line changes the lines below it, as on
  // the real chip (a screen split set up on the last line of a text row takes effect on the next).
  clock(c) {
    this.lineCycle += c;
    let stolen = 0;
    for (;;) {
      if (!this.drawn && this.lineCycle >= DRAW_AT) { this.drawn = true; this.renderLine(this.raster); }
      if (this.lineCycle < this.model.cpl) break;
      this.lineCycle -= this.model.cpl;
      const s = this.nextLine();
      stolen += s; this.lineCycle += s;
      this.drawn = false;
    }
    return stolen;
  }

  nextLine() {
    const reg = this.reg;
    const line = this.raster;

    // Sprite row counters, then the Y compare for sprites starting on the next line.
    let active = 0;
    for (let s = 0; s < 8; s++) {
      if (this.sprDma[s]) {
        if ((reg[0x17] >> s) & 1 && !this.sprHalf[s]) this.sprHalf[s] = 1;
        else { this.sprHalf[s] = 0; this.sprMc[s] += 3; if (this.sprMc[s] >= 63) this.sprDma[s] = 0; }
      }
      if (!this.sprDma[s] && (reg[0x15] >> s) & 1 && reg[1 + s * 2] === (line & 0xff)) {
        this.sprDma[s] = 1; this.sprMc[s] = 0; this.sprHalf[s] = 0;
      }
      active += this.sprDma[s];
    }

    // Row counter at the end of the line.
    if (this.rc === 7) { if (this.display) this.vcbase += 40; this.display = false; }
    if (this.display) this.rc = (this.rc + 1) & 7;

    this.raster++;
    if (this.raster >= this.model.lines) { this.raster = 0; this.vcbase = 0; this.denLatch = false; }
    const r = this.raster;
    if (r === this.rasterCompare) this.irqStatus |= 1;
    const ctrl1 = reg[0x11];
    if (r === 0x30 && (ctrl1 & 0x10)) this.denLatch = true;
    const rsel = ctrl1 & 0x08;
    if (r === (rsel ? 251 : 247)) this.vborder = true;
    else if (r === (rsel ? 51 : 55) && (ctrl1 & 0x10)) this.vborder = false;

    let stolen = active ? active * 2 + 1 : 0;
    if (this.denLatch && r >= 0x30 && r <= 0xf7 && (r & 7) === (ctrl1 & 7)) {
      this.rc = 0; this.display = true;
      stolen += 40;
    }
    return stolen;
  }

  vicRead(a) {
    a = (a & 0x3fff) | this.bank;
    if ((a & 0x7000) === 0x1000) return this.mem.chargen[a & 0x0fff];
    return this.mem.ram[a];
  }

  renderLine(line) {
    const m = this.model, reg = this.reg;
    let row = line - m.firstLine;
    if (row < 0) row += m.lines;
    const visible = row < m.rows;
    let anySprite = 0;
    for (let s = 0; s < 8; s++) anySprite |= this.sprDma[s];
    if (!visible && !anySprite) return;

    const col = this.lineCol, fg = this.lineFg;
    const border = reg[0x20] & 15;
    const ctrl1 = reg[0x11], ctrl2 = reg[0x16];
    fg.fill(0, 0, WIDTH);

    if (this.vborder) {
      col.fill(border, 0, WIDTH);
    } else {
      const b0 = reg[0x21] & 15;
      col.fill(b0, LEFT, LEFT + 328);
      if (this.display) this.renderGraphics(LEFT + (ctrl2 & 7));
    }

    if (anySprite) this.renderSprites();

    if (!this.vborder) {
      if (ctrl2 & 0x08) { col.fill(border, 0, LEFT); col.fill(border, LEFT + 320, WIDTH); }
      else { col.fill(border, 0, LEFT + 7); col.fill(border, LEFT + 311, WIDTH); }
    } else if (anySprite) {
      col.fill(border, 0, WIDTH);
    }

    if (visible) {
      this.fb.set(col.subarray(0, WIDTH), row * WIDTH);
      if (row === m.rows - 1) { this.frameReady = true; this.frames++; if (this.onFrame) this.onFrame(); }
    }
  }

  renderGraphics(x0) {
    const reg = this.reg, col = this.lineCol, fg = this.lineFg;
    const ram = this.mem.ram, colorRam = this.mem.colorRam;
    const ctrl1 = reg[0x11], ctrl2 = reg[0x16];
    const mode = ((ctrl1 & 0x60) | (ctrl2 & 0x10)) >> 4;   // ECM BMM MCM
    const vm = (reg[0x18] & 0xf0) << 6;
    const cb = (reg[0x18] & 0x0e) << 10;
    const rc = this.rc, vc = this.vcbase;
    const b0 = reg[0x21] & 15, b1 = reg[0x22] & 15, b2 = reg[0x23] & 15;

    for (let c = 0; c < 40; c++) {
      const cell = (vc + c) & 0x3ff;
      const ch = this.vicRead(vm | cell);
      const cr = colorRam[cell] & 15;
      const x = x0 + c * 8;
      let data, multi = false, c0 = b0, c1 = 0, c2 = 0, c3 = 0;

      switch (mode) {
        case 0: data = this.vicRead(cb + ch * 8 + rc); c3 = cr; break;
        case 1:
          data = this.vicRead(cb + ch * 8 + rc);
          if (cr & 8) { multi = true; c1 = b1; c2 = b2; c3 = cr & 7; } else c3 = cr;
          break;
        case 2: data = this.vicRead((cb & 0x2000) + cell * 8 + rc); c0 = ch & 15; c3 = ch >> 4; break;
        case 3: data = this.vicRead((cb & 0x2000) + cell * 8 + rc); multi = true; c1 = ch >> 4; c2 = ch & 15; c3 = cr; break;
        case 4: data = this.vicRead(cb + (ch & 63) * 8 + rc); c0 = reg[0x21 + (ch >> 6)] & 15; c3 = cr; break;
        default: data = 0; c0 = 0; break;   // invalid modes show black
      }

      if (multi) {
        for (let p = 0; p < 4; p++) {
          const bits = (data >> (6 - p * 2)) & 3;
          const color = bits === 0 ? c0 : bits === 1 ? c1 : bits === 2 ? c2 : c3;
          const px = x + p * 2;
          col[px] = col[px + 1] = color;
          if (bits & 2) fg[px] = fg[px + 1] = 1;
        }
      } else {
        for (let p = 0; p < 8; p++) {
          if (data & (0x80 >> p)) { col[x + p] = c3; fg[x + p] = 1; } else col[x + p] = c0;
        }
      }
    }
  }

  renderSprites() {
    const reg = this.reg, col = this.lineCol, fg = this.lineFg, spr = this.lineSpr;
    spr.fill(0);
    const vm = (reg[0x18] & 0xf0) << 6;
    const mm0 = reg[0x25] & 15, mm1 = reg[0x26] & 15;
    const wrap = this.model.cpl * 8;
    let ss = 0, sb = 0;

    for (let s = 7; s >= 0; s--) {
      if (!this.sprDma[s]) continue;
      const ptr = this.vicRead(vm | 0x3f8 | s) << 6;
      const mc = this.sprMc[s];
      const data = (this.vicRead(ptr + mc) << 16) | (this.vicRead(ptr + mc + 1) << 8) | this.vicRead(ptr + mc + 2);
      if (!data) continue;
      const bit = 1 << s;
      let x = reg[s * 2] | ((reg[0x10] >> s) & 1 ? 256 : 0);
      if (x >= wrap - 24) x -= wrap;
      x += LEFT - 24;
      const wide = (reg[0x1d] >> s) & 1 ? 2 : 1;
      const behind = (reg[0x1b] >> s) & 1;
      const own = reg[0x27 + s] & 15;
      const multi = (reg[0x1c] >> s) & 1;
      const units = multi ? 12 : 24, unitW = (multi ? 2 : 1) * wide;

      for (let u = 0; u < units; u++) {
        let color;
        if (multi) {
          const bits = (data >> (22 - u * 2)) & 3;
          if (!bits) continue;
          color = bits === 1 ? mm0 : bits === 2 ? own : mm1;
        } else {
          if (!(data & (0x800000 >> u))) continue;
          color = own;
        }
        const start = x + u * unitW;
        for (let px = start; px < start + unitW; px++) {
          if (px < 0 || px >= BUF) continue;
          const prev = spr[px];
          if (prev) ss |= prev | bit;
          spr[px] = prev | bit;
          if (fg[px]) { sb |= bit; if (!behind) col[px] = color; } else col[px] = color;
        }
      }
    }

    // The interrupt fires only for the first collision since the register was read.
    if (ss) { if (!this.collSS) this.irqStatus |= 4; this.collSS |= ss; }
    if (sb) { if (!this.collSB) this.irqStatus |= 2; this.collSB |= sb; }
  }
}
