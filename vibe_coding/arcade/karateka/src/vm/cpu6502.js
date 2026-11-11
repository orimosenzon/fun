// NMOS 6502 core, documented opcodes, cycle counted.
// Used only as a measuring instrument for the original game.

const M = { IMP: 0, ACC: 1, IMM: 2, ZP: 3, ZPX: 4, ZPY: 5, ABS: 6, ABX: 7, ABY: 8, IZX: 9, IZY: 10, IND: 11, REL: 12 };

const OPS = new Array(256).fill(null);
function def(name, list) {
  for (const [op, mode, cyc, pen] of list) OPS[op] = { name, mode, cyc, pen: !!pen, op };
}
const grp = (base) => [
  [base + 0x09, M.IMM, 2], [base + 0x05, M.ZP, 3], [base + 0x15, M.ZPX, 4], [base + 0x0d, M.ABS, 4],
  [base + 0x1d, M.ABX, 4, 1], [base + 0x19, M.ABY, 4, 1], [base + 0x01, M.IZX, 6], [base + 0x11, M.IZY, 5, 1],
];
def('ORA', grp(0x00)); def('AND', grp(0x20)); def('EOR', grp(0x40)); def('ADC', grp(0x60));
def('LDA', grp(0xa0)); def('CMP', grp(0xc0)); def('SBC', grp(0xe0));
def('STA', [[0x85, M.ZP, 3], [0x95, M.ZPX, 4], [0x8d, M.ABS, 4], [0x9d, M.ABX, 5], [0x99, M.ABY, 5], [0x81, M.IZX, 6], [0x91, M.IZY, 6]]);
const shf = (base) => [[base + 0x0a, M.ACC, 2], [base + 0x06, M.ZP, 5], [base + 0x16, M.ZPX, 6], [base + 0x0e, M.ABS, 6], [base + 0x1e, M.ABX, 7]];
def('ASL', shf(0x00)); def('ROL', shf(0x20)); def('LSR', shf(0x40)); def('ROR', shf(0x60));
def('DEC', [[0xc6, M.ZP, 5], [0xd6, M.ZPX, 6], [0xce, M.ABS, 6], [0xde, M.ABX, 7]]);
def('INC', [[0xe6, M.ZP, 5], [0xf6, M.ZPX, 6], [0xee, M.ABS, 6], [0xfe, M.ABX, 7]]);
def('BIT', [[0x24, M.ZP, 3], [0x2c, M.ABS, 4]]);
def('CPX', [[0xe0, M.IMM, 2], [0xe4, M.ZP, 3], [0xec, M.ABS, 4]]);
def('CPY', [[0xc0, M.IMM, 2], [0xc4, M.ZP, 3], [0xcc, M.ABS, 4]]);
def('LDX', [[0xa2, M.IMM, 2], [0xa6, M.ZP, 3], [0xb6, M.ZPY, 4], [0xae, M.ABS, 4], [0xbe, M.ABY, 4, 1]]);
def('LDY', [[0xa0, M.IMM, 2], [0xa4, M.ZP, 3], [0xb4, M.ZPX, 4], [0xac, M.ABS, 4], [0xbc, M.ABX, 4, 1]]);
def('STX', [[0x86, M.ZP, 3], [0x96, M.ZPY, 4], [0x8e, M.ABS, 4]]);
def('STY', [[0x84, M.ZP, 3], [0x94, M.ZPX, 4], [0x8c, M.ABS, 4]]);
def('JMP', [[0x4c, M.ABS, 3], [0x6c, M.IND, 5]]);
def('JSR', [[0x20, M.ABS, 6]]);
for (const [n, op] of [['BPL', 0x10], ['BMI', 0x30], ['BVC', 0x50], ['BVS', 0x70], ['BCC', 0x90], ['BCS', 0xb0], ['BNE', 0xd0], ['BEQ', 0xf0]]) def(n, [[op, M.REL, 2]]);
for (const [n, op, c] of [['BRK', 0x00, 7], ['CLC', 0x18, 2], ['CLD', 0xd8, 2], ['CLI', 0x58, 2], ['CLV', 0xb8, 2], ['DEX', 0xca, 2], ['DEY', 0x88, 2],
  ['INX', 0xe8, 2], ['INY', 0xc8, 2], ['NOP', 0xea, 2], ['PHA', 0x48, 3], ['PHP', 0x08, 3], ['PLA', 0x68, 4], ['PLP', 0x28, 4], ['RTI', 0x40, 6],
  ['RTS', 0x60, 6], ['SEC', 0x38, 2], ['SED', 0xf8, 2], ['SEI', 0x78, 2], ['TAX', 0xaa, 2], ['TAY', 0xa8, 2], ['TSX', 0xba, 2], ['TXA', 0x8a, 2],
  ['TXS', 0x9a, 2], ['TYA', 0x98, 2]]) def(n, [[op, M.IMP, c]]);

export const OPTABLE = OPS;
export const MODES = M;

export class CPU6502 {
  constructor(read, write) {
    this.read = read; this.write = write;
    this.a = 0; this.x = 0; this.y = 0; this.sp = 0xff; this.pc = 0;
    this.n = 0; this.v = 0; this.d = 0; this.i = 1; this.z = 0; this.c = 0;
    this.cycles = 0;
    this.illegal = 0;
  }
  reset() { this.pc = this.read(0xfffc) | (this.read(0xfffd) << 8); this.sp = 0xff; this.i = 1; this.d = 0; }
  getP() { return (this.n << 7) | (this.v << 6) | 0x20 | (this.d << 3) | (this.i << 2) | (this.z << 1) | this.c; }
  setP(p) { this.n = (p >> 7) & 1; this.v = (p >> 6) & 1; this.d = (p >> 3) & 1; this.i = (p >> 2) & 1; this.z = (p >> 1) & 1; this.c = p & 1; }
  push(v) { this.write(0x100 | this.sp, v); this.sp = (this.sp - 1) & 0xff; }
  pop() { this.sp = (this.sp + 1) & 0xff; return this.read(0x100 | this.sp); }
  nz(v) { this.n = (v >> 7) & 1; this.z = v === 0 ? 1 : 0; return v; }
  adc(m) {
    if (this.d) {
      let lo = (this.a & 15) + (m & 15) + this.c;
      let hi = (this.a >> 4) + (m >> 4);
      const bin = (this.a + m + this.c) & 0xff;
      if (lo > 9) { lo += 6; hi++; }
      this.z = bin === 0 ? 1 : 0;
      this.n = (hi >> 3) & 1;
      this.v = (~(this.a ^ m) & (this.a ^ (hi << 4)) & 0x80) ? 1 : 0;
      if (hi > 9) hi += 6;
      this.c = hi > 15 ? 1 : 0;
      this.a = ((hi << 4) | (lo & 15)) & 0xff;
    } else {
      const r = this.a + m + this.c;
      this.v = (~(this.a ^ m) & (this.a ^ r) & 0x80) ? 1 : 0;
      this.c = r > 0xff ? 1 : 0;
      this.a = this.nz(r & 0xff);
    }
  }
  sbc(m) {
    if (this.d) {
      const r = this.a - m - (1 - this.c);
      let lo = (this.a & 15) - (m & 15) - (1 - this.c);
      let hi = (this.a >> 4) - (m >> 4);
      if (lo < 0) { lo -= 6; hi--; }
      if (hi < 0) hi -= 6;
      this.v = ((this.a ^ m) & (this.a ^ r) & 0x80) ? 1 : 0;
      this.c = r >= 0 ? 1 : 0;
      this.nz(r & 0xff);
      this.a = ((hi << 4) | (lo & 15)) & 0xff;
    } else {
      this.adc(m ^ 0xff);
    }
  }
  step() {
    const opc = this.read(this.pc);
    const o = OPS[opc];
    const pc0 = this.pc;
    this.pc = (this.pc + 1) & 0xffff;
    if (!o) { this.illegal++; this.lastIllegal = pc0; this.cycles += 2; return; }
    let addr = -1, cyc = o.cyc;
    const rd = this.read;
    switch (o.mode) {
      case M.IMM: addr = this.pc; this.pc = (this.pc + 1) & 0xffff; break;
      case M.ZP: addr = rd(this.pc); this.pc = (this.pc + 1) & 0xffff; break;
      case M.ZPX: addr = (rd(this.pc) + this.x) & 0xff; this.pc = (this.pc + 1) & 0xffff; break;
      case M.ZPY: addr = (rd(this.pc) + this.y) & 0xff; this.pc = (this.pc + 1) & 0xffff; break;
      case M.ABS: addr = rd(this.pc) | (rd((this.pc + 1) & 0xffff) << 8); this.pc = (this.pc + 2) & 0xffff; break;
      case M.ABX: { const b = rd(this.pc) | (rd((this.pc + 1) & 0xffff) << 8); addr = (b + this.x) & 0xffff; if (o.pen && ((b ^ addr) & 0xff00)) cyc++; this.pc = (this.pc + 2) & 0xffff; break; }
      case M.ABY: { const b = rd(this.pc) | (rd((this.pc + 1) & 0xffff) << 8); addr = (b + this.y) & 0xffff; if (o.pen && ((b ^ addr) & 0xff00)) cyc++; this.pc = (this.pc + 2) & 0xffff; break; }
      case M.IZX: { const zp = (rd(this.pc) + this.x) & 0xff; addr = rd(zp) | (rd((zp + 1) & 0xff) << 8); this.pc = (this.pc + 1) & 0xffff; break; }
      case M.IZY: { const zp = rd(this.pc); const b = rd(zp) | (rd((zp + 1) & 0xff) << 8); addr = (b + this.y) & 0xffff; if (o.pen && ((b ^ addr) & 0xff00)) cyc++; this.pc = (this.pc + 1) & 0xffff; break; }
      case M.IND: { const p = rd(this.pc) | (rd((this.pc + 1) & 0xffff) << 8); addr = rd(p) | (rd((p & 0xff00) | ((p + 1) & 0xff)) << 8); this.pc = (this.pc + 2) & 0xffff; break; }
      case M.REL: { let off = rd(this.pc); this.pc = (this.pc + 1) & 0xffff; if (off & 0x80) off -= 256; addr = (this.pc + off) & 0xffff; break; }
    }
    let t;
    switch (o.name) {
      case 'LDA': this.a = this.nz(rd(addr)); break;
      case 'LDX': this.x = this.nz(rd(addr)); break;
      case 'LDY': this.y = this.nz(rd(addr)); break;
      case 'STA': this.write(addr, this.a); break;
      case 'STX': this.write(addr, this.x); break;
      case 'STY': this.write(addr, this.y); break;
      case 'ORA': this.a = this.nz(this.a | rd(addr)); break;
      case 'AND': this.a = this.nz(this.a & rd(addr)); break;
      case 'EOR': this.a = this.nz(this.a ^ rd(addr)); break;
      case 'ADC': this.adc(rd(addr)); break;
      case 'SBC': this.sbc(rd(addr)); break;
      case 'CMP': t = this.a - rd(addr); this.c = t >= 0 ? 1 : 0; this.nz(t & 0xff); break;
      case 'CPX': t = this.x - rd(addr); this.c = t >= 0 ? 1 : 0; this.nz(t & 0xff); break;
      case 'CPY': t = this.y - rd(addr); this.c = t >= 0 ? 1 : 0; this.nz(t & 0xff); break;
      case 'BIT': t = rd(addr); this.z = (this.a & t) === 0 ? 1 : 0; this.n = (t >> 7) & 1; this.v = (t >> 6) & 1; break;
      case 'ASL': if (o.mode === M.ACC) { this.c = this.a >> 7; this.a = this.nz((this.a << 1) & 0xff); } else { t = rd(addr); this.c = t >> 7; this.write(addr, this.nz((t << 1) & 0xff)); } break;
      case 'LSR': if (o.mode === M.ACC) { this.c = this.a & 1; this.a = this.nz(this.a >> 1); } else { t = rd(addr); this.c = t & 1; this.write(addr, this.nz(t >> 1)); } break;
      case 'ROL': if (o.mode === M.ACC) { t = (this.a << 1) | this.c; this.c = this.a >> 7; this.a = this.nz(t & 0xff); } else { const m = rd(addr); t = (m << 1) | this.c; this.c = m >> 7; this.write(addr, this.nz(t & 0xff)); } break;
      case 'ROR': if (o.mode === M.ACC) { t = (this.a >> 1) | (this.c << 7); this.c = this.a & 1; this.a = this.nz(t); } else { const m = rd(addr); t = (m >> 1) | (this.c << 7); this.c = m & 1; this.write(addr, this.nz(t)); } break;
      case 'INC': this.write(addr, this.nz((rd(addr) + 1) & 0xff)); break;
      case 'DEC': this.write(addr, this.nz((rd(addr) - 1) & 0xff)); break;
      case 'INX': this.x = this.nz((this.x + 1) & 0xff); break;
      case 'INY': this.y = this.nz((this.y + 1) & 0xff); break;
      case 'DEX': this.x = this.nz((this.x - 1) & 0xff); break;
      case 'DEY': this.y = this.nz((this.y - 1) & 0xff); break;
      case 'TAX': this.x = this.nz(this.a); break;
      case 'TAY': this.y = this.nz(this.a); break;
      case 'TXA': this.a = this.nz(this.x); break;
      case 'TYA': this.a = this.nz(this.y); break;
      case 'TSX': this.x = this.nz(this.sp); break;
      case 'TXS': this.sp = this.x; break;
      case 'PHA': this.push(this.a); break;
      case 'PHP': this.push(this.getP() | 0x10); break;
      case 'PLA': this.a = this.nz(this.pop()); break;
      case 'PLP': this.setP(this.pop()); break;
      case 'CLC': this.c = 0; break; case 'SEC': this.c = 1; break;
      case 'CLD': this.d = 0; break; case 'SED': this.d = 1; break;
      case 'CLI': this.i = 0; break; case 'SEI': this.i = 1; break;
      case 'CLV': this.v = 0; break;
      case 'NOP': break;
      case 'JMP': this.pc = addr; break;
      case 'JSR': t = (this.pc - 1) & 0xffff; this.push(t >> 8); this.push(t & 0xff); this.pc = addr; break;
      case 'RTS': t = this.pop(); t |= this.pop() << 8; this.pc = (t + 1) & 0xffff; break;
      case 'RTI': this.setP(this.pop()); t = this.pop(); t |= this.pop() << 8; this.pc = t; break;
      case 'BRK': t = (this.pc + 1) & 0xffff; this.push(t >> 8); this.push(t & 0xff); this.push(this.getP() | 0x10); this.i = 1; this.pc = rd(0xfffe) | (rd(0xffff) << 8); break;
      case 'BPL': if (!this.n) { cyc += 1 + (((this.pc ^ addr) & 0xff00) ? 1 : 0); this.pc = addr; } break;
      case 'BMI': if (this.n) { cyc += 1 + (((this.pc ^ addr) & 0xff00) ? 1 : 0); this.pc = addr; } break;
      case 'BVC': if (!this.v) { cyc += 1 + (((this.pc ^ addr) & 0xff00) ? 1 : 0); this.pc = addr; } break;
      case 'BVS': if (this.v) { cyc += 1 + (((this.pc ^ addr) & 0xff00) ? 1 : 0); this.pc = addr; } break;
      case 'BCC': if (!this.c) { cyc += 1 + (((this.pc ^ addr) & 0xff00) ? 1 : 0); this.pc = addr; } break;
      case 'BCS': if (this.c) { cyc += 1 + (((this.pc ^ addr) & 0xff00) ? 1 : 0); this.pc = addr; } break;
      case 'BNE': if (!this.z) { cyc += 1 + (((this.pc ^ addr) & 0xff00) ? 1 : 0); this.pc = addr; } break;
      case 'BEQ': if (this.z) { cyc += 1 + (((this.pc ^ addr) & 0xff00) ? 1 : 0); this.pc = addr; } break;
    }
    this.cycles += cyc;
  }
}
