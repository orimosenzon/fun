// Pure (browser + node) Apple II+ machine. See apple2.mjs for the node wrapper.
import { CPU6502 } from './cpu6502.js';

export const CPU_HZ = 1020484;

export const TRANS62 = [0x96, 0x97, 0x9a, 0x9b, 0x9d, 0x9e, 0x9f, 0xa6, 0xa7, 0xab, 0xac, 0xad, 0xae, 0xaf, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6, 0xb7, 0xb9, 0xba, 0xbb, 0xbc, 0xbd, 0xbe, 0xbf, 0xcb, 0xcd, 0xce, 0xcf, 0xd3, 0xd6, 0xd7, 0xd9, 0xda, 0xdb, 0xdc, 0xdd, 0xde, 0xdf, 0xe5, 0xe6, 0xe7, 0xe9, 0xea, 0xeb, 0xec, 0xed, 0xee, 0xef, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf9, 0xfa, 0xfb, 0xfc, 0xfd, 0xfe, 0xff];
export const DOS_SKEW = [0, 7, 14, 6, 13, 5, 12, 4, 11, 3, 10, 2, 9, 1, 8, 15];

function nibblizeTrack(dsk, track, volume = 254) {
  const out = [];
  const p44 = (v) => { out.push((v >> 1) | 0xaa, v | 0xaa); };
  for (let phys = 0; phys < 16; phys++) {
    for (let i = 0; i < (phys === 0 ? 64 : 20); i++) out.push(0xff);
    out.push(0xd5, 0xaa, 0x96); p44(volume); p44(track); p44(phys); p44(volume ^ track ^ phys); out.push(0xde, 0xaa, 0xeb);
    for (let i = 0; i < 6; i++) out.push(0xff);
    out.push(0xd5, 0xaa, 0xad);
    const off = (track * 16 + DOS_SKEW[phys]) * 256;
    const data = dsk.subarray(off, off + 256);
    const nibs = new Uint8Array(0x158);
    let idx2 = 0x55;
    for (let idx6 = 0x101; idx6 >= 0; idx6--) {
      let val6 = data[idx6 % 0x100];
      let val2 = nibs[idx2];
      val2 = (val2 << 1) | (val6 & 1); val6 >>= 1;
      val2 = (val2 << 1) | (val6 & 1); val6 >>= 1;
      nibs[0x56 + idx6] = val6;
      nibs[idx2] = val2 & 0x3f;
      if (--idx2 < 0) idx2 = 0x55;
    }
    let last = 0;
    for (let i = 0; i < 0x156; i++) { out.push(TRANS62[last ^ nibs[i]]); last = nibs[i]; }
    out.push(TRANS62[last]);
    out.push(0xde, 0xaa, 0xeb);
  }
  return Uint8Array.from(out);
}

export class Machine {
  constructor({ rom, diskRom, dsk }) {
    this.ram = new Uint8Array(0x10000);
    this.rom = rom; // 12K at $D000
    if (this.rom.length !== 0x3000) throw new Error('unexpected ROM size ' + this.rom.length);
    this.diskRom = diskRom;
    if (this.diskRom.length !== 256) throw new Error('unexpected disk ROM size ' + this.diskRom.length);
    this.lcRam = new Uint8Array(0x4000); // D000-FFFF bank 2 + E000..; bank1 kept separately
    this.lcBank1 = new Uint8Array(0x1000);
    this.lcRead = false; this.lcWrite = false; this.lcBank2 = true; this.lcPre = false;
    this.key = 0; this.keyQueue = [];
    this.video = { text: true, mixed: false, page2: false, hires: false };
    this.videoLog = null; // optional array of [cycle, switch]
    this.speaker = []; // cycle stamps of toggles
    this.buttons = [0, 0, 0];
    this.paddleTimeout = Infinity; // no joystick attached
    this.paddleStart = 0;
    this.tracks = [];
    if (dsk) this.loadDisk(dsk);
    this.drive = { halfTrack: 0, motor: false, q6: 0, q7: 0, lastPos: -1, offAt: -1e9 };
    this.cpu = new CPU6502((a) => this.read(a), (a, v) => this.write(a, v));
    this.onWrite = null;
    this.cpu.reset();
  }
  loadDisk(dsk) {
    this.dsk = dsk;
    this.tracks = [];
    for (let t = 0; t < 35; t++) this.tracks.push(nibblizeTrack(dsk, t));
  }
  io(addr, isWrite, val) {
    const a = addr & 0xff;
    if (a < 0x10) return isWrite ? 0 : this.key;
    if (a < 0x20) { this.key &= 0x7f; return 0; }
    if (a >= 0x30 && a < 0x40) { this.speaker.push(this.cpu.cycles); return 0; }
    if (a >= 0x50 && a < 0x58) {
      const v = this.video;
      switch (a) {
        case 0x50: v.text = false; break; case 0x51: v.text = true; break;
        case 0x52: v.mixed = false; break; case 0x53: v.mixed = true; break;
        case 0x54: v.page2 = false; break; case 0x55: v.page2 = true; break;
        case 0x56: v.hires = false; break; case 0x57: v.hires = true; break;
      }
      if (this.videoLog) this.videoLog.push([this.cpu.cycles, a]);
      return 0;
    }
    if (a >= 0x61 && a <= 0x63) return this.buttons[a - 0x61] ? 0x80 : 0;
    if (a >= 0x64 && a <= 0x67) return (this.cpu.cycles - this.paddleStart) < this.paddleTimeout ? 0x80 : 0;
    if (a === 0x70) { this.paddleStart = this.cpu.cycles; return 0; }
    if (a >= 0x80 && a < 0x90) {
      this.lcBank2 = !(a & 8);
      const m = a & 3;
      this.lcRead = m === 0 || m === 3;
      if (a & 1) { if (!isWrite) { if (this.lcPre) this.lcWrite = true; this.lcPre = true; } else this.lcPre = false; } else { this.lcWrite = false; this.lcPre = false; }
      return 0;
    }
    if (a >= 0xe0 && a < 0xf0) return this.diskIO(a & 0xf);
    return 0;
  }
  diskIO(s) {
    const d = this.drive;
    if (s < 8) {
      if (s & 1) {
        const ph = s >> 1, cur = d.halfTrack & 3;
        if (ph === ((cur + 1) & 3)) d.halfTrack = Math.min(69, d.halfTrack + 1);
        else if (ph === ((cur + 3) & 3)) d.halfTrack = Math.max(0, d.halfTrack - 1);
      }
      return 0;
    }
    switch (s) {
      case 8: if (d.motor) d.offAt = this.cpu.cycles; d.motor = false; break; case 9: d.motor = true; break;
      case 0xc: d.q6 = 0; break; case 0xd: d.q6 = 1; break;
      case 0xe: d.q7 = 0; break; case 0xf: d.q7 = 1; break;
    }
    if (s === 0xc && !d.q7 && (d.motor || this.cpu.cycles - d.offAt < CPU_HZ)) { // real drives keep spinning ~1s after motor-off
      const trk = this.tracks[d.halfTrack >> 1];
      const pos = Math.floor(this.cpu.cycles / 32) % trk.length;
      const v = trk[pos];
      if (pos === d.lastPos) return v & 0x7f;
      d.lastPos = pos;
      return v;
    }
    return 0;
  }
  read(addr) {
    if (addr < 0xc000) return this.ram[addr];
    if (addr < 0xc100) return this.io(addr, false, 0);
    if (addr >= 0xc600 && addr < 0xc700) return this.diskRom[addr - 0xc600];
    if (addr < 0xd000) return 0xff;
    if (this.lcRead) {
      if (addr < 0xe000 && !this.lcBank2) return this.lcBank1[addr - 0xd000];
      return this.lcRam[addr - 0xc000];
    }
    return this.rom[addr - 0xd000];
  }
  write(addr, v) {
    if (addr < 0xc000) { this.ram[addr] = v; if (this.onWrite) this.onWrite(addr, v); return; }
    if (addr < 0xc100) { this.io(addr, true, v); return; }
    if (addr >= 0xd000 && this.lcWrite) {
      if (addr < 0xe000 && !this.lcBank2) this.lcBank1[addr - 0xd000] = v; else this.lcRam[addr - 0xc000] = v;
    }
  }
  pressKey(ch) { this.key = (typeof ch === 'string' ? ch.charCodeAt(0) : ch) | 0x80; }
  // Run until the given absolute cycle count.
  runTo(cycle) { const c = this.cpu; while (c.cycles < cycle) c.step(); }
  runSeconds(s) { this.runTo(this.cpu.cycles + Math.round(s * CPU_HZ)); }
  seconds() { return this.cpu.cycles / CPU_HZ; }
  hgrPage() { const base = this.video.page2 ? 0x4000 : 0x2000; return this.ram.slice(base, base + 0x2000); }
}

// --- Hi-res rendering -------------------------------------------------
export function hgrRowBase(y) { return ((y & 7) << 10) | (((y >> 3) & 7) << 7) | ((y >> 6) * 40); }

// Returns a 280x192 array of colour indices: 0 black, 1 white, 2 violet, 3 green, 4 blue, 5 orange
export function hgrToIndexed(page) {
  const out = new Uint8Array(280 * 192);
  const bits = new Uint8Array(282), hi = new Uint8Array(282);
  for (let y = 0; y < 192; y++) {
    const base = hgrRowBase(y);
    bits.fill(0);
    for (let c = 0; c < 40; c++) {
      const b = page[base + c];
      for (let i = 0; i < 7; i++) { bits[1 + c * 7 + i] = (b >> i) & 1; hi[1 + c * 7 + i] = b >> 7; }
    }
    for (let x = 0; x < 280; x++) {
      const on = bits[x + 1], l = bits[x], r = bits[x + 2];
      let col = 0;
      if (on) col = (l || r) ? 1 : (hi[x + 1] ? ((x & 1) ? 5 : 4) : ((x & 1) ? 3 : 2));
      else if (l && r) col = hi[x + 1] ? ((x & 1) ? 4 : 5) : ((x & 1) ? 2 : 3); // fill between two same-colour dots
      out[y * 280 + x] = col;
    }
  }
  return out;
}

export const PALETTE = [[0, 0, 0], [255, 255, 255], [255, 68, 253], [20, 245, 60], [20, 207, 253], [255, 106, 60]];

