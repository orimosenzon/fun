// The machine the game runs on, without any Apple ROM. The original program needs very
// little from the computer's firmware, and each piece is replaced here by JavaScript:
//   - the Disk II boot PROM ($C600, and the copy the game makes of it at $4600)
//   - WAIT ($FCA8), the monitor's delay loop, cycle exact
//   - SETKBD / SETVID / INIT ($FE89, $FE93, $FB2F), text screen set-up at boot
//   - IORTS ($FF58), a bare RTS, and the machine ID byte at $FBB3
// Everything else (DOS, the game) is code that lives on the disk itself.
import { Machine, DOS_SKEW, TRANS62 } from './machine.js';

const MARK = 0x02; // an illegal opcode; marks the two entry points of the stand-in boot PROM

export class GameMachine extends Machine {
  constructor(dsk) {
    const rom = new Uint8Array(0x3000);
    rom[0xfbb3 - 0xd000] = 0xea; // "Apple II+"
    const prom = new Uint8Array(256); prom[0x00] = MARK; prom[0x5c] = MARK;
    super({ rom, diskRom: prom, dsk });
    this.cpu.sp = 0xff;
    this.inGame = false;   // true once the game proper (not the title sequence) has been loaded
    this.keyboardMode = true; // the original starts in joystick mode and expects the player to press K
    this.cpu.pc = 0xc600;
    this.hookAt = new Uint8Array(0x10000); this.hooks = new Map();
  }

  // Extensions to the game: call fn when the game proper reaches addr, provided the code
  // there is the code we expect. fn returns true when it has replaced the instruction.
  hook(addr, bytes, fn) { this.hookAt[addr] = 1; this.hooks.set(addr, { bytes, fn }); }
  runHook(pc) {
    const h = this.hooks.get(pc), m = this.ram;
    for (let i = 0; i < h.bytes.length; i++) if (m[pc + i] !== h.bytes[i]) return false;
    return h.fn();
  }

  rts() { const c = this.cpu; let t = c.pop(); t |= c.pop() << 8; c.pc = (t + 1) & 0xffff; c.cycles += 6; }

  // Returns true when the instruction at pc was handled here instead of by the CPU.
  firmware(pc) {
    const c = this.cpu, m = this.ram;
    if (pc < 0xc000) {
      if (m[pc] !== MARK || ((pc & 0xff) !== 0 && (pc & 0xff) !== 0x5c)) return false;
      this.bootProm(pc & 0xff00, (pc & 0xff) === 0); return true;
    }
    switch (pc) {
      case 0xc600: this.bootProm(0xc600, true); return true;
      case 0xc65c: this.bootProm(0xc600, false); return true;
      case 0xff58: this.rts(); return true;
      case 0xfe89: m[0x38] = 0x1b; m[0x39] = 0xfd; this.rts(); return true;
      case 0xfe93: m[0x36] = 0xf0; m[0x37] = 0xfd; this.rts(); return true;
      case 0xfb2f:
        m[0x48] = 0; m[0x22] = 0; m[0x20] = 0; m[0x21] = 0x28; m[0x23] = 0x18; m[0x25] = 0x17; m[0x28] = 0xd0; m[0x29] = 0x07;
        this.video.text = true; this.video.page2 = false; this.video.hires = false;
        this.rts(); return true;
      case 0xfca8: this.wait(); return true;
    }
    throw new Error('unsupported firmware call $' + pc.toString(16));
  }

  // Monitor WAIT: SEC / PHA / SBC #1 / BNE / PLA / SBC #1 / BNE / RTS, counted cycle by cycle.
  wait() {
    const c = this.cpu;
    let a = c.a, carry = 1, cyc = 2;
    do {
      cyc += 3;
      let t = a;
      do { t = t - 1 - (1 - carry); carry = t >= 0 ? 1 : 0; t &= 0xff; cyc += t ? 5 : 4; } while (t);
      cyc += 4;
      a = a - 1 - (1 - carry); carry = a >= 0 ? 1 : 0; a &= 0xff; cyc += a ? 5 : 4;
    } while (a);
    c.a = 0; c.c = carry; c.z = 1; c.n = 0;
    c.cycles += cyc;
    this.rts();
  }

  // Boot PROM: read sector $3D of track $41 into ($26), then what the PROM's last bytes do.
  bootProm(base, cold) {
    const c = this.cpu, m = this.ram;
    if (cold) {
      TRANS62.forEach((n, i) => { m[0x356 + (n & 0x7f)] = i; });
      m[0x2b] = 0x60; m[0x26] = 0; m[0x27] = 0x08; m[0x3d] = 0; m[0x41] = 0;
      this.drive.halfTrack = 0;
    }
    this.drive.motor = true;
    if (cold) this.inGame = base !== 0xc600; // the game proper is started through the copy of the PROM
    for (;;) {
      const off = (m[0x41] * 16 + DOS_SKEW[m[0x3d] & 15]) * 256, dst = m[0x26] | (m[0x27] << 8);
      for (let i = 0; i < 256; i++) this.write((dst + i) & 0xffff, this.dsk[off + i]);
      this.drive.halfTrack = m[0x41] * 2;
      m[0x27] = (m[0x27] + 1) & 0xff; m[0x3d] = (m[0x3d] + 1) & 0xff;
      c.a = m[0x3d]; c.y = 0; c.c = m[0x3d] >= m[0x0800] ? 1 : 0; c.z = m[0x3d] === m[0x0800] ? 1 : 0;
      c.cycles += 20000;
      if (base !== 0xc600) { c.pc = base + 0xf8; return; } // the game's copy has its own ending patched in
      c.x = m[0x2b];
      if (c.c) { c.pc = 0x0801; return; }
    }
  }

  // $C4 is the game's input device flag (1 = keyboard, 0 = joystick; the K and J keys set it).
  selectInput() { if (this.inGame && !this.drive.motor) this.ram[0xc4] = this.keyboardMode ? 1 : 0; }

  step() {
    const c = this.cpu, pc = c.pc;
    if ((pc >= 0xc000 || this.ram[pc] === MARK) && this.firmware(pc)) return;
    if (this.hookAt[pc] && this.inGame && this.runHook(pc)) return;
    c.step();
  }
  runTo(cycle) { const c = this.cpu; while (c.cycles < cycle) this.step(); }
}
