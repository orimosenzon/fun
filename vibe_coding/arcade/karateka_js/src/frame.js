// The end of a frame, and the keyboard.
// The game draws on the page that is not showing and then swaps the two.
import { extend, u8 } from './core.js';
import { EDGE, PASS } from './data/timing.js';

export class Restart extends Error { constructor(level) { super('restart'); this.level = level; } }
export class StartOver extends Error { constructor() { super('start over'); } }   // the original jumps back into its set-up ($B763)

extend({
  strobe() { this.key &= 0x7f; },
  // $7609: show the page just drawn, and draw the next frame on the other one
  r7609() {
    const m = this.m; this.g.clearRightEdge(); this.cycles += EDGE;
    m[7] = m[7] === 0x40 ? 0x20 : 0x40;
    const shown = (m[7] ^ 0x60) << 8;
    this.flips.push({ at: this.cycles, show: m[7] ^ 0x60, page: m.slice(shown, shown + 0x2000) });
  },
  // $7600: end of frame: swap the pages, then play whatever sounds the frame asked for
  r7600() {
    const m = this.m; this.r7609(); this.cycles += PASS;
    if (m[0x86] === 0) { m[0x42] = 0; m[0x43] = 0; m[0x45] = 0; m[0x88] = 0; m[0x4e] = 0; return; }
    if (m[0x4e] !== 0 && --m[0x4e] === 0) this.r0DC0();
    if (m[0x42] !== 0) { m[0x42]--; this.r0C40(); }
    if (m[0x43] !== 0) { m[0x43]--; if (m[0x41] === 1) this.r0C46(); else this.r0C43(); }
    if (m[0x88] !== 0) { m[0x88] = 0; this.r1006(); }
    if (m[0x45] !== 0) { m[0x45] = 0; this.r0C49(); }
    if (m[0x20] === 0x24 || m[0x25] === 0x4b || m[0x25] === 0x4a) { if (m[0x36] !== 0) this.r0C4F(); else this.r0C52(); }
  },
  // $7603: the keys that are not moves: Esc pauses, Ctrl-S sound, J/K joystick or keyboard,
  // Ctrl-R start again, and the joystick settings. Returns the key in A.
  r7603() {
    const m = this.m; let a = this.a = this.key;
    if (!(a & 0x80)) return;
    if (m[0xd0] === 0 || m[0x36] !== 0 || m[0xc4] === 0) this.strobe();
    const flipBit = (addr) => { m[addr] ^= 1; this.r0C4C(); };
    switch (a) {
      case 0x9b: this.strobe(); this.paused = true; return;                 // the page resumes on the next key, and swallows it
      case 0x93: { const y = m[0x86] ^ 1; m[0x86] = y ^ m[0x4f]; m[0x4f] = y; if (m[0xd0] !== 0) { m[0xd5] = m[0x86]; m[0xd6] = m[0x4f]; } return this.r0C4C(); }
      case 0x82: return flipBit(0xc3);
      case 0x85: return flipBit(0xc0);
      case 0x99: return flipBit(0xc2);
      case 0x98: return flipBit(0xc1);
      case 0xca: case 0xea: if (m[0xc4] === 0) { this.a = 0; return; } m[0xc4] = 0; return this.r0C4C();
      case 0xcb: case 0xeb: if (m[0xc4] !== 0) { this.a = m[0xc4]; return; } m[0xc4] = 1; return this.r0C4C();
      case 0x92: this.strobe(); if (m[0xd2] & 0x80) { this.a = m[0xd2]; return; } throw new Restart(m[0xd0]);
      case 0x8e: m[0xc0] = 0; m[0xc1] = 0; m[0xc2] = 0; m[0xc3] = 0; return this.r0C4C();
    }
    this.a = 0x91;
  },

  // $6E00: the player's wish for this frame, from the keyboard (or the joystick)
  r6E00() {
    const m = this.m, cmd = (a) => { this.a = a; this.r6540(); };
    if (m[0xc4] === 0) {                       // joystick
      this.r6D00();
      if (m[0x2f] !== 0) return cmd(!(m[0x2a] & 0x80) ? 0x9b : m[0x47] === 0xff ? 0xc2 : 0);
      if (m[0x46] === 0xff) return cmd(0xc5);
      if (m[0x47] === 0xff) return cmd(0xd7);
      if (m[0x46] !== 0 || m[0x47] !== 0 || m[0x4d] === 5) return cmd(0);
      if (m[0x98] !== 0 && m[0x91] < 0x27 && m[0x94] & 0x80) return cmd(0);
      if (m[0x5e] !== 0 && m[0x4d] < 2 && u8(m[0x72] - m[0x22]) < 0x0f) return cmd(0);
      return cmd(m[0x2a] & 0x80 ? 0x9b : 0);
    }
    this.r7603();
    let a = this.key;
    const go = (c) => { this.strobe(); cmd(c); };
    if (a & 0x80) {
      if (a >= 0xe0) a -= 0x20;
      if (a === 0xa0) { m[0x29] = 0; if (m[0x2f] !== 0) m[0x46] = 0xff; return go(0x9b); }     // space
      if (a === 0x88) { m[0x29] = 0xff; return go(0); }                                        // left
      if (a === 0x95) { m[0x29] = 1; return go(0); }                                           // right
    }
    if (m[0x2f] !== 0) {                       // standing upright
      if (m[0x47] === 0xff) return go(0xc2);
      if (m[0x46] === 0xff) return go(0x9b);
      if (a === 0xc2) { m[0x47] = 0xff; return go(0xc2); }                                     // B: bow
      return go(0);
    }
    m[0x29] = 0;
    if (m[0x47] === 0xff) return go(0xd7);
    const punch = { 0xd1: 0xff, 0xc1: 0, 0xda: 1 }[a], kick = { 0xd7: 0xff, 0xd3: 0, 0xd8: 1 }[a];   // Q A Z, W S X
    if (punch !== undefined) { m[0x47] = 0xff; m[0x2a] = punch; return go(0xd7); }
    if (m[0x46] === 0xff) return go(0xc5);
    if (kick !== undefined) { m[0x46] = 0xff; m[0x2a] = kick; return go(0xc5); }
    go(0);
  },
  r6D00() { this.m[0x29] = 0; this.m[0x2a] = 0; },   // the joystick's stick: none here
  // the tunes ($1000 jump table): filled in by music.js
});
