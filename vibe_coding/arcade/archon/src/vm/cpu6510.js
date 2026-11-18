// MOS 6510 core: all 256 opcodes (documented and undocumented), cycle counted.
// Instruction-stepped: step() runs one instruction and returns the cycles it took.

const IMP = 0, ACC = 1, IMM = 2, ZP = 3, ZPX = 4, ZPY = 5, ABS = 6, ABX = 7, ABY = 8, IZX = 9, IZY = 10, IND = 11, REL = 12;

const NAMES = `
BRK ORA KIL SLO NOP ORA ASL SLO PHP ORA ASL ANC NOP ORA ASL SLO
BPL ORA KIL SLO NOP ORA ASL SLO CLC ORA NOP SLO NOP ORA ASL SLO
JSR AND KIL RLA BIT AND ROL RLA PLP AND ROL ANC BIT AND ROL RLA
BMI AND KIL RLA NOP AND ROL RLA SEC AND NOP RLA NOP AND ROL RLA
RTI EOR KIL SRE NOP EOR LSR SRE PHA EOR LSR ALR JMP EOR LSR SRE
BVC EOR KIL SRE NOP EOR LSR SRE CLI EOR NOP SRE NOP EOR LSR SRE
RTS ADC KIL RRA NOP ADC ROR RRA PLA ADC ROR ARR JMP ADC ROR RRA
BVS ADC KIL RRA NOP ADC ROR RRA SEI ADC NOP RRA NOP ADC ROR RRA
NOP STA NOP SAX STY STA STX SAX DEY NOP TXA XAA STY STA STX SAX
BCC STA KIL AHX STY STA STX SAX TYA STA TXS TAS SHY STA SHX AHX
LDY LDA LDX LAX LDY LDA LDX LAX TAY LDA TAX LXA LDY LDA LDX LAX
BCS LDA KIL LAX LDY LDA LDX LAX CLV LDA TSX LAS LDY LDA LDX LAX
CPY CMP NOP DCP CPY CMP DEC DCP INY CMP DEX AXS CPY CMP DEC DCP
BNE CMP KIL DCP NOP CMP DEC DCP CLD CMP NOP DCP NOP CMP DEC DCP
CPX SBC NOP ISC CPX SBC INC ISC INX SBC NOP SBC CPX SBC INC ISC
BEQ SBC KIL ISC NOP SBC INC ISC SED SBC NOP ISC NOP SBC INC ISC`.trim().split(/\s+/);

const ROW_EVEN = [IMP, IZX, IMP, IZX, ZP, ZP, ZP, ZP, IMP, IMM, ACC, IMM, ABS, ABS, ABS, ABS];
const ROW_ODD = [REL, IZY, IMP, IZY, ZPX, ZPX, ZPX, ZPX, IMP, ABY, IMP, ABY, ABX, ABX, ABX, ABX];
const MODE = new Uint8Array(256);
for (let op = 0; op < 256; op++) MODE[op] = (op & 0x10 ? ROW_ODD : ROW_EVEN)[op & 15];
MODE[0x20] = ABS; MODE[0x6c] = IND;
for (const op of [0x80, 0xa0, 0xc0, 0xe0, 0x82, 0xa2, 0xc2, 0xe2]) MODE[op] = IMM;
for (const op of [0x8a, 0xaa, 0xca, 0xea]) MODE[op] = IMP;
for (const op of [0x96, 0x97, 0xb6, 0xb7]) MODE[op] = ZPY;
for (const op of [0x9e, 0x9f, 0xbe, 0xbf]) MODE[op] = ABY;

const CYC = Uint8Array.from(`
7 6 2 8 3 3 5 5 3 2 2 2 4 4 6 6
2 5 2 8 4 4 6 6 2 4 2 7 4 4 7 7
6 6 2 8 3 3 5 5 4 2 2 2 4 4 6 6
2 5 2 8 4 4 6 6 2 4 2 7 4 4 7 7
6 6 2 8 3 3 5 5 3 2 2 2 3 4 6 6
2 5 2 8 4 4 6 6 2 4 2 7 4 4 7 7
6 6 2 8 3 3 5 5 4 2 2 2 5 4 6 6
2 5 2 8 4 4 6 6 2 4 2 7 4 4 7 7
2 6 2 6 3 3 3 3 2 2 2 2 4 4 4 4
2 6 2 6 4 4 4 4 2 5 2 5 5 5 5 5
2 6 2 6 3 3 3 3 2 2 2 2 4 4 4 4
2 5 2 5 4 4 4 4 2 4 2 4 4 4 4 4
2 6 2 8 3 3 5 5 2 2 2 2 4 4 6 6
2 5 2 8 4 4 6 6 2 4 2 7 4 4 7 7
2 6 2 8 3 3 5 5 2 2 2 2 4 4 6 6
2 5 2 8 4 4 6 6 2 4 2 7 4 4 7 7`.trim().split(/\s+/).map(Number));

// One more cycle when an indexed read crosses a page.
const PEN = new Uint8Array(256);
for (let op = 0; op < 256; op++) {
  const m = MODE[op];
  if (((m === ABX || m === ABY) && CYC[op] === 4) || (m === IZY && CYC[op] === 5)) PEN[op] = 1;
}

const branch = (c, addr, taken) => {
  if (!taken) return;
  c.extra += ((c.pc ^ addr) & 0xff00) ? 2 : 1;
  c.pc = addr;
};
const cmp = (c, r, m) => { const d = r - m; c.c = d >= 0 ? 1 : 0; c.nz(d & 0xff); };
// Read-modify-write: the 6502 writes the old value back before the new one
// (INC $D019 acknowledges interrupts this way).
const rmw = (c, addr, fn) => { const old = c.read(addr); c.write(addr, old); const v = fn(c, old); c.write(addr, v); return v; };
const asl = (c, v) => { c.c = v >> 7; return c.nz((v << 1) & 0xff); };
const lsr = (c, v) => { c.c = v & 1; return c.nz(v >> 1); };
const rol = (c, v) => { const r = ((v << 1) | c.c) & 0xff; c.c = v >> 7; return c.nz(r); };
const ror = (c, v) => { const r = (v >> 1) | (c.c << 7); c.c = v & 1; return c.nz(r); };
const inc = (c, v) => c.nz((v + 1) & 0xff);
const dec = (c, v) => c.nz((v - 1) & 0xff);

const EXEC = {
  ORA(c, a) { c.a = c.nz(c.a | c.read(a)); },
  AND(c, a) { c.a = c.nz(c.a & c.read(a)); },
  EOR(c, a) { c.a = c.nz(c.a ^ c.read(a)); },
  ADC(c, a) { c.adc(c.read(a)); },
  SBC(c, a) { c.sbc(c.read(a)); },
  CMP(c, a) { cmp(c, c.a, c.read(a)); },
  CPX(c, a) { cmp(c, c.x, c.read(a)); },
  CPY(c, a) { cmp(c, c.y, c.read(a)); },
  LDA(c, a) { c.a = c.nz(c.read(a)); },
  LDX(c, a) { c.x = c.nz(c.read(a)); },
  LDY(c, a) { c.y = c.nz(c.read(a)); },
  STA(c, a) { c.write(a, c.a); },
  STX(c, a) { c.write(a, c.x); },
  STY(c, a) { c.write(a, c.y); },
  BIT(c, a) { const m = c.read(a); c.z = (c.a & m) ? 0 : 1; c.n = m >> 7; c.v = (m >> 6) & 1; },
  ASL(c, a) { rmw(c, a, asl); }, LSR(c, a) { rmw(c, a, lsr); }, ROL(c, a) { rmw(c, a, rol); }, ROR(c, a) { rmw(c, a, ror); },
  INC(c, a) { rmw(c, a, inc); }, DEC(c, a) { rmw(c, a, dec); },
  ASL_A(c) { c.a = asl(c, c.a); }, LSR_A(c) { c.a = lsr(c, c.a); }, ROL_A(c) { c.a = rol(c, c.a); }, ROR_A(c) { c.a = ror(c, c.a); },
  INX(c) { c.x = c.nz((c.x + 1) & 0xff); }, INY(c) { c.y = c.nz((c.y + 1) & 0xff); },
  DEX(c) { c.x = c.nz((c.x - 1) & 0xff); }, DEY(c) { c.y = c.nz((c.y - 1) & 0xff); },
  TAX(c) { c.x = c.nz(c.a); }, TAY(c) { c.y = c.nz(c.a); }, TXA(c) { c.a = c.nz(c.x); }, TYA(c) { c.a = c.nz(c.y); },
  TSX(c) { c.x = c.nz(c.sp); }, TXS(c) { c.sp = c.x; },
  CLC(c) { c.c = 0; }, SEC(c) { c.c = 1; }, CLI(c) { c.i = 0; }, SEI(c) { c.i = 1; }, CLD(c) { c.d = 0; }, SED(c) { c.d = 1; }, CLV(c) { c.v = 0; },
  PHA(c) { c.push(c.a); }, PLA(c) { c.a = c.nz(c.pop()); },
  PHP(c) { c.push(c.getP() | 0x10); }, PLP(c) { c.setP(c.pop()); },
  JMP(c, a) { c.pc = a; },
  JSR(c, a) { const r = (c.pc - 1) & 0xffff; c.push(r >> 8); c.push(r & 0xff); c.pc = a; },
  RTS(c) { const lo = c.pop(); c.pc = ((lo | (c.pop() << 8)) + 1) & 0xffff; },
  RTI(c) { c.setP(c.pop()); const lo = c.pop(); c.pc = lo | (c.pop() << 8); },
  BRK(c) { c.pc = (c.pc + 1) & 0xffff; c.interrupt(0xfffe, true); },
  BPL(c, a) { branch(c, a, !c.n); }, BMI(c, a) { branch(c, a, c.n); },
  BVC(c, a) { branch(c, a, !c.v); }, BVS(c, a) { branch(c, a, c.v); },
  BCC(c, a) { branch(c, a, !c.c); }, BCS(c, a) { branch(c, a, c.c); },
  BNE(c, a) { branch(c, a, !c.z); }, BEQ(c, a) { branch(c, a, c.z); },
  NOP() {},
  // Undocumented.
  KIL(c) { c.pc = (c.pc - 1) & 0xffff; c.jammed = true; },
  SLO(c, a) { c.a = c.nz(c.a | rmw(c, a, asl)); },
  RLA(c, a) { c.a = c.nz(c.a & rmw(c, a, rol)); },
  SRE(c, a) { c.a = c.nz(c.a ^ rmw(c, a, lsr)); },
  RRA(c, a) { c.adc(rmw(c, a, ror)); },
  SAX(c, a) { c.write(a, c.a & c.x); },
  LAX(c, a) { c.a = c.x = c.nz(c.read(a)); },
  DCP(c, a) { cmp(c, c.a, rmw(c, a, dec)); },
  ISC(c, a) { c.sbc(rmw(c, a, inc)); },
  ANC(c, a) { c.a = c.nz(c.a & c.read(a)); c.c = c.n; },
  ALR(c, a) { c.a = lsr(c, c.a & c.read(a)); },
  ARR(c, a) { c.a = c.nz(((c.a & c.read(a)) >> 1) | (c.c << 7)); c.c = (c.a >> 6) & 1; c.v = ((c.a >> 6) ^ (c.a >> 5)) & 1; },
  XAA(c, a) { c.a = c.nz((c.a | 0xee) & c.x & c.read(a)); },
  LXA(c, a) { c.a = c.x = c.nz((c.a | 0xee) & c.read(a)); },
  AXS(c, a) { const d = (c.a & c.x) - c.read(a); c.c = d >= 0 ? 1 : 0; c.x = c.nz(d & 0xff); },
  LAS(c, a) { c.a = c.x = c.sp = c.nz(c.read(a) & c.sp); },
  TAS(c, a) { c.sp = c.a & c.x; c.write(a, c.sp & (((a >> 8) + 1) & 0xff)); },
  SHY(c, a) { c.write(a, c.y & (((a >> 8) + 1) & 0xff)); },
  SHX(c, a) { c.write(a, c.x & (((a >> 8) + 1) & 0xff)); },
  AHX(c, a) { c.write(a, c.a & c.x & (((a >> 8) + 1) & 0xff)); },
};

const HANDLER = new Array(256);
for (let op = 0; op < 256; op++) HANDLER[op] = EXEC[MODE[op] === ACC ? NAMES[op] + '_A' : NAMES[op]];

export const OPNAMES = NAMES, OPMODES = MODE, OPCYCLES = CYC;
export const MODES = { IMP, ACC, IMM, ZP, ZPX, ZPY, ABS, ABX, ABY, IZX, IZY, IND, REL };

export class CPU6510 {
  constructor(read, write) {
    this.read = read; this.write = write;
    this.a = 0; this.x = 0; this.y = 0; this.sp = 0xfd; this.pc = 0;
    this.n = 0; this.v = 0; this.d = 0; this.i = 1; this.z = 0; this.c = 0;
    this.irqLine = false;      // level: taken whenever I is clear
    this.nmiPending = false;   // edge: taken once
    this.jammed = false;
    this.extra = 0;
  }
  reset() { this.pc = this.read(0xfffc) | (this.read(0xfffd) << 8); this.sp = 0xfd; this.i = 1; this.d = 0; this.jammed = false; this.nmiPending = false; }
  getP() { return (this.n << 7) | (this.v << 6) | 0x20 | (this.d << 3) | (this.i << 2) | (this.z << 1) | this.c; }
  setP(p) { this.n = (p >> 7) & 1; this.v = (p >> 6) & 1; this.d = (p >> 3) & 1; this.i = (p >> 2) & 1; this.z = (p >> 1) & 1; this.c = p & 1; }
  push(v) { this.write(0x100 | this.sp, v); this.sp = (this.sp - 1) & 0xff; }
  pop() { this.sp = (this.sp + 1) & 0xff; return this.read(0x100 | this.sp); }
  nz(v) { this.n = v >> 7; this.z = v === 0 ? 1 : 0; return v; }
  interrupt(vector, brk) {
    this.push(this.pc >> 8); this.push(this.pc & 0xff);
    this.push(this.getP() | (brk ? 0x10 : 0));
    this.i = 1;
    this.pc = this.read(vector) | (this.read(vector + 1) << 8);
  }
  adc(m) {
    const a = this.a;
    if (this.d) {
      let lo = (a & 15) + (m & 15) + this.c;
      let hi = (a >> 4) + (m >> 4);
      this.z = ((a + m + this.c) & 0xff) === 0 ? 1 : 0;
      if (lo > 9) { lo += 6; hi++; }
      this.n = (hi >> 3) & 1;
      this.v = (~(a ^ m) & (a ^ (hi << 4)) & 0x80) ? 1 : 0;
      if (hi > 9) hi += 6;
      this.c = hi > 15 ? 1 : 0;
      this.a = ((hi << 4) | (lo & 15)) & 0xff;
    } else {
      const r = a + m + this.c;
      this.v = (~(a ^ m) & (a ^ r) & 0x80) ? 1 : 0;
      this.c = r > 0xff ? 1 : 0;
      this.a = this.nz(r & 0xff);
    }
  }
  sbc(m) {
    const a = this.a, borrow = 1 - this.c;
    const r = a - m - borrow;
    this.v = ((a ^ m) & (a ^ r) & 0x80) ? 1 : 0;
    if (this.d) {
      let lo = (a & 15) - (m & 15) - borrow;
      let hi = (a >> 4) - (m >> 4);
      if (lo < 0) { lo -= 6; hi--; }
      if (hi < 0) hi -= 6;
      this.c = r >= 0 ? 1 : 0;
      this.nz(r & 0xff);
      this.a = ((hi << 4) | (lo & 15)) & 0xff;
    } else {
      this.c = r >= 0 ? 1 : 0;
      this.a = this.nz(r & 0xff);
    }
  }

  step() {
    if (this.nmiPending) { this.nmiPending = false; this.interrupt(0xfffa, false); return 7; }
    if (this.irqLine && !this.i) { this.interrupt(0xfffe, false); return 7; }
    const op = this.read(this.pc);
    let pc = (this.pc + 1) & 0xffff;
    let addr = 0, base;
    this.extra = 0;
    switch (MODE[op]) {
      case IMM: addr = pc; pc++; break;
      case ZP: addr = this.read(pc); pc++; break;
      case ZPX: addr = (this.read(pc) + this.x) & 0xff; pc++; break;
      case ZPY: addr = (this.read(pc) + this.y) & 0xff; pc++; break;
      case ABS: addr = this.read(pc) | (this.read((pc + 1) & 0xffff) << 8); pc += 2; break;
      case ABX:
        base = this.read(pc) | (this.read((pc + 1) & 0xffff) << 8); pc += 2;
        addr = (base + this.x) & 0xffff;
        if (PEN[op] && ((base ^ addr) & 0xff00)) this.extra = 1;
        break;
      case ABY:
        base = this.read(pc) | (this.read((pc + 1) & 0xffff) << 8); pc += 2;
        addr = (base + this.y) & 0xffff;
        if (PEN[op] && ((base ^ addr) & 0xff00)) this.extra = 1;
        break;
      case IZX: base = (this.read(pc) + this.x) & 0xff; pc++; addr = this.read(base) | (this.read((base + 1) & 0xff) << 8); break;
      case IZY:
        base = this.read(pc); pc++;
        base = this.read(base) | (this.read((base + 1) & 0xff) << 8);
        addr = (base + this.y) & 0xffff;
        if (PEN[op] && ((base ^ addr) & 0xff00)) this.extra = 1;
        break;
      case IND:
        base = this.read(pc) | (this.read((pc + 1) & 0xffff) << 8); pc += 2;
        addr = this.read(base) | (this.read((base & 0xff00) | ((base + 1) & 0xff)) << 8);
        break;
      case REL: base = this.read(pc); pc = (pc + 1) & 0xffff; addr = (pc + (base < 0x80 ? base : base - 256)) & 0xffff; break;
      default: break;
    }
    this.pc = pc & 0xffff;
    HANDLER[op](this, addr);
    return CYC[op] + this.extra;
  }
}
