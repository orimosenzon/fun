// A Commodore 64: 6510, 64K RAM with ROM banking, VIC-II, two CIAs, SID,
// keyboard matrix and two joysticks. Runs in the browser and in Node.
// No disk drive is emulated: KERNAL LOAD from device 8+ is served straight
// from a D64 image.

import { CPU6510 } from './cpu6510.js';
import { VIC, MODELS, WIDTH, PALETTE } from './vic.js';
import { CIA } from './cia.js';
import { SID } from './sid.js';

export { MODELS, WIDTH, PALETTE };

// Keyboard matrix position: column (CIA1 port A bit) * 8 + row (port B bit).
const KEY_ROWS = [
  ['DEL', 'RETURN', 'RIGHT', 'F7', 'F1', 'F3', 'F5', 'DOWN'],
  ['3', 'W', 'A', '4', 'Z', 'S', 'E', 'LSHIFT'],
  ['5', 'R', 'D', '6', 'C', 'F', 'T', 'X'],
  ['7', 'Y', 'G', '8', 'B', 'H', 'U', 'V'],
  ['9', 'I', 'J', '0', 'M', 'K', 'O', 'N'],
  ['+', 'P', 'L', '-', '.', ':', '@', ','],
  ['POUND', '*', ';', 'HOME', 'RSHIFT', '=', 'UP_ARROW', '/'],
  ['1', 'LEFT_ARROW', 'CTRL', '2', 'SPACE', 'CBM', 'Q', 'STOP'],
];
export const KEYS = {};
KEY_ROWS.forEach((row, c) => row.forEach((name, r) => { KEYS[name] = c * 8 + r; }));

export const JOY = { UP: 1, DOWN: 2, LEFT: 4, RIGHT: 8, FIRE: 16 };

export class C64 {
  constructor({ kernal, basic, chargen, model = 'ntsc', sampleRate = 44100 }) {
    this.model = MODELS[model];
    this.kernal = kernal; this.basic = basic; this.chargen = chargen;
    this.ram = new Uint8Array(65536);
    this.colorRam = new Uint8Array(1024);
    this.cpu = new CPU6510((a) => this.read(a), (a, v) => this.write(a, v));
    this.vic = new VIC(this.model, this);
    this.cia1 = new CIA(this.model.hz);
    this.cia2 = new CIA(this.model.hz);
    this.sid = new SID(this.model.hz, sampleRate);
    this.keyCols = new Uint8Array(8);   // per column: rows held down
    this.joy = [0, 0];                  // [port 1, port 2], JOY bits
    this.disk = null;                   // D64
    this.traps = new Map();
    this.trapAt = new Uint8Array(65536);

    this.cia1.readPinsA = () => {
      let v = 0xff & ~this.joy[1];
      const rowsLow = ~this.cia1.portB & 0xff;       // reverse scan: rows driven low
      if (rowsLow) for (let c = 0; c < 8; c++) if (this.keyCols[c] & rowsLow) v &= ~(1 << c);
      return v;
    };
    this.cia1.readPinsB = () => {
      let v = 0xff & ~this.joy[0];
      const colsLow = ~(this.cia1.portA & ~this.joy[1]) & 0xff;
      for (let c = 0; c < 8; c++) if ((colsLow >> c) & 1) v &= ~this.keyCols[c];
      return v;
    };
    // CIA2 port A: bits 0-1 pick the VIC bank; the serial bus has nobody on it.
    this.cia2.onPortA = () => { this.vic.bank = (~this.cia2.portA & 3) << 14; };
    this.cia2.readPinsA = () => { const out = this.cia2.portA; return 0x3f | ((out & 0x10) ? 0 : 0x40) | ((out & 0x20) ? 0 : 0x80); };

    // KERNAL LOAD, after the vector at $0330.
    this.trap(0xf4a5, () => this.kernalLoad());
    this.reset();
  }

  reset() {
    this.ram.fill(0);
    // Power-on pattern. Some decrunchers assume memory is not all zero.
    for (let a = 0; a < 65536; a++) this.ram[a] = (a & 0x40) ? 0xff : 0x00;
    this.colorRam.fill(0);
    this.ddr = 0x2f; this.port = 0x37;
    this.updateBanks();
    this.vic.reset(); this.cia1.reset(); this.cia2.reset(); this.sid.reset();
    this.cia2.onPortA();
    this.cycles = 0;
    this.nmiLine = false;
    this.cpu.reset();
  }

  updateBanks() {
    const p = (this.port | ~this.ddr) & 7;
    const any = (p & 3) !== 0;
    this.basicOn = (p & 3) === 3;
    this.kernalOn = (p & 2) !== 0;
    this.ioOn = any && (p & 4) !== 0;
    this.charOn = any && !(p & 4);
  }

  read(a) {
    if (a < 0xa000) {
      if (a < 2) return a === 0 ? this.ddr : ((this.port & this.ddr) | (~this.ddr & 0x17));
      return this.ram[a];
    }
    if (a < 0xc000) return this.basicOn ? this.basic[a - 0xa000] : this.ram[a];
    if (a < 0xd000) return this.ram[a];
    if (a < 0xe000) {
      if (this.ioOn) return this.readIO(a);
      return this.charOn ? this.chargen[a - 0xd000] : this.ram[a];
    }
    return this.kernalOn ? this.kernal[a - 0xe000] : this.ram[a];
  }

  write(a, v) {
    if (a >= 0xd000 && a < 0xe000 && this.ioOn) { this.writeIO(a, v); return; }
    if (a < 2) {
      if (a === 0) this.ddr = v; else this.port = v;
      this.updateBanks();
      return;
    }
    this.ram[a] = v;
  }

  readIO(a) {
    if (a < 0xd400) return this.vic.read(a);
    if (a < 0xd800) { this.sid.clockTo(this.cycles); return this.sid.read(a); }
    if (a < 0xdc00) return this.colorRam[a - 0xd800] | 0xf0;
    if (a < 0xdd00) return this.cia1.read(a);
    if (a < 0xde00) return this.cia2.read(a);
    return 0xff;
  }

  writeIO(a, v) {
    if (a < 0xd400) this.vic.write(a, v);
    else if (a < 0xd800) { this.sid.clockTo(this.cycles); this.sid.write(a, v); }
    else if (a < 0xdc00) this.colorRam[a - 0xd800] = v & 15;
    else if (a < 0xdd00) this.cia1.write(a, v);
    else if (a < 0xde00) this.cia2.write(a, v);
  }

  // Call fn(machine) whenever the CPU is about to execute the instruction at addr.
  trap(addr, fn) { this.traps.set(addr, fn); this.trapAt[addr] = 1; }
  untrap(addr) { this.traps.delete(addr); this.trapAt[addr] = 0; }

  step() {
    const cpu = this.cpu;
    if (this.trapAt[cpu.pc]) this.traps.get(cpu.pc)(this);
    let c = cpu.step();
    c += this.vic.clock(c);
    this.cia1.clock(c); this.cia2.clock(c);
    this.cycles += c;
    cpu.irqLine = this.vic.irq || this.cia1.irq;
    const nmi = this.cia2.irq || this.restoreKey;
    if (nmi && !this.nmiLine) cpu.nmiPending = true;
    this.nmiLine = !!nmi;
    return c;
  }

  // Run until the VIC finishes drawing a frame.
  runFrame() {
    this.vic.frameReady = false;
    let guard = 0;
    while (!this.vic.frameReady && guard++ < 100000 && !this.cpu.jammed) this.step();
    this.sid.clockTo(this.cycles);
  }

  // Run for about n CPU cycles. onFrame is called each time the VIC finishes a
  // frame (the frame buffer is complete at that moment). Returns the frame count.
  runCycles(n, onFrame) {
    const end = this.cycles + n;
    let frames = 0;
    this.vic.frameReady = false;
    while (this.cycles < end && !this.cpu.jammed) {
      this.step();
      if (this.vic.frameReady) { this.vic.frameReady = false; frames++; if (onFrame) onFrame(); }
    }
    this.sid.clockTo(this.cycles);
    return frames;
  }

  rts() { const lo = this.cpu.pop(); this.cpu.pc = ((lo | (this.cpu.pop() << 8)) + 1) & 0xffff; }

  kernalLoad() {
    if (!this.kernalOn || !this.disk || this.ram[0xba] < 8) return;
    const ram = this.ram, cpu = this.cpu;
    const nameAt = ram[0xbb] | (ram[0xbc] << 8);
    const data = this.disk.load(ram.slice(nameAt, nameAt + ram[0xb7]));
    if (!data) { ram[0x90] = 0x42; cpu.a = 4; cpu.c = 1; this.rts(); return; }
    if (!cpu.a) {
      const addr = ram[0xb9] ? data[0] | (data[1] << 8) : ram[0xc3] | (ram[0xc4] << 8);
      const end = this.inject(data.subarray(2), addr);
      ram[0xae] = end & 0xff; ram[0xaf] = end >> 8;
    }
    ram[0x90] = 0x40;
    cpu.x = ram[0xae]; cpu.y = ram[0xaf]; cpu.c = 0;
    this.rts();
  }

  inject(bytes, addr) {
    for (let i = 0; i < bytes.length; i++) this.ram[(addr + i) & 0xffff] = bytes[i];
    return (addr + bytes.length) & 0xffff;
  }

  // Run the power-on sequence until BASIC waits for a key.
  bootToReady(maxFrames = 400) {
    for (let f = 0; f < maxFrames; f++) {
      this.runFrame();
      if (this.ram[0xcc] === 0 && this.cpu.pc >= 0xe5cd && this.cpu.pc <= 0xe5d5 && this.kernalOn) return true;
    }
    return false;
  }

  // Put a PRG (load address + bytes) in memory as if LOADed, and type RUN.
  runProgram(prg) {
    const addr = prg[0] | (prg[1] << 8);
    const end = this.inject(prg.subarray(2), addr);
    for (const p of [0x2d, 0x2f, 0x31, 0xae]) { this.ram[p] = end & 0xff; this.ram[p + 1] = end >> 8; }
    this.type('RUN\r');
  }

  // Into the KERNAL keyboard buffer (10 characters at most).
  type(text) {
    for (let i = 0; i < text.length && i < 10; i++) this.ram[0x277 + i] = text.charCodeAt(i);
    this.ram[0xc6] = Math.min(text.length, 10);
  }

  setKey(key, down) {
    const c = key >> 3, bit = 1 << (key & 7);
    if (down) this.keyCols[c] |= bit; else this.keyCols[c] &= ~bit;
  }

  // The 40x25 text screen as lines of ASCII (for tests and debugging).
  screenText() {
    const base = ((this.vic.reg[0x18] & 0xf0) << 6) | this.vic.bank;
    const lines = [];
    for (let r = 0; r < 25; r++) {
      let s = '';
      for (let c = 0; c < 40; c++) {
        const ch = this.ram[base + r * 40 + c] & 0x7f;
        s += ch === 0 ? '@' : ch < 27 ? String.fromCharCode(64 + ch) : ch < 32 ? '[\\]^_'[ch - 27] : ch < 64 ? String.fromCharCode(ch) : '.';
      }
      lines.push(s.replace(/\s+$/, ''));
    }
    return lines;
  }

  // The current frame as RGBA bytes.
  frameRGBA(out) {
    const fb = this.vic.fb;
    out = out || new Uint8ClampedArray(fb.length * 4);
    for (let i = 0, o = 0; i < fb.length; i++, o += 4) {
      const p = PALETTE[fb[i]];
      out[o] = p[0]; out[o + 1] = p[1]; out[o + 2] = p[2]; out[o + 3] = 255;
    }
    return out;
  }
}
