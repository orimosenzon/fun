// Akuma's hall. When the player leaves a scene the game cuts to the fortress: Akuma on his
// dais, an opponent walks in, bows, is sent out with a wave of the hand. Then back to the player.
import { extend, u8 } from './core.js';
import { waitCycles } from './sound.js';
import { StartOver } from './frame.js';

extend({
  // pieces of the hall: columns and floor, drawn on the left and mirrored on the right
  r78D8() { const m = this.m, x = this.x; m[3] = m[0x7815 + x]; m[4] = m[0x7820 + x]; m[6] = m[0x7836 + x]; },
  r78F9() { const m = this.m; m[5] = m[0x782b + this.x]; this.r78D8(); m[0x10] = 0; this.r1903(); },
  r780C() { const m = this.m; m[5] = u8(0x26 - m[0x782b + this.x]); this.r78D8(); m[0x10] = 6; this.r190C(); },
  r791A() { const m = this.m; m[2] = 0xd5; m[0x11] = 0xaa; m[0x12] = 0x80; m[0x13] = 0x80; },
  // $7800: the hall itself (in full the first time on each page)
  r7800() {
    const m = this.m, both = (x) => { this.x = x; this.r78F9(); this.r780C(); };
    m[0x0f] = 1;
    if (m[0x83] !== 0) { m[0x83]--; this.r1900(); m[5] = 4; m[9] = 0x24; m[6] = 0x99; m[8] = 0xb6; this.r791A(); this.r0A03(); both(0); both(3); }
    both(1); both(4);
  },
  // $7803: what stands in front: the doorway on the right, the lever, the side walls
  r7803() {
    const m = this.m;
    m[0x0f] = 1; this.x = 2; this.r780C(); this.r78F9();
    m[0x0f] = 0; this.x = m[0x84]; if (this.x !== 0) this.r78F9();
    if (m[0xd0] >= 3 && m[0x44] === 1) { this.x = m[0xed]; if (this.x !== 0) this.r780C(); }
    this.r78B5(); m[5] = 0x25; m[9] = 0x28; this.blackRows(0x5f, 0xad);
  },
  r78B5() { const m = this.m; m[5] = 0; m[9] = 3; this.blackRows(0x5f, 0xad); },
  blackRows(y, y2) { const m = this.m; m[6] = y; m[8] = y2; m[2] = 0x80; m[0x11] = 0x80; this.r0A00(); },
  r7AB1() { this.a = 0x19; this.r7AB3(); },
  r7AB3() { const m = this.m; m[5] = this.a; m[9] = 0x26; this.blackRows(0xa9, 0xac); },
  r7AC1() { const m = this.m; if (m[0x7c] & 0x80) this.r6786(); m[5] = u8(m[0x72] + 1); m[9] = u8(m[5] + 7); this.blackRows(0xa1, 0xa2); },
  r7ADC() { const m = this.m; m[5] = 4; m[9] = 0x14; m[6] = 0xa1; m[8] = 0xa2; m[2] = 0xd5; m[0x11] = 0xaa; this.r0A00(); },

  // Akuma: $8F his body, $8A his arm, $56 his head; all drawn mirrored at the right of the hall
  r7D6A() { const m = this.m, x = this.x; m[3] = m[0x7d0c + x]; m[4] = m[0x7d1d + x]; m[6] = m[0x7d3f + x]; m[5] = u8(0x28 - m[0x7d2e + x]); this.r190C(); },
  r7D03() {
    const m = this.m, piece = (x) => { this.x = x; this.r7D6A(); };
    m[0x10] = 0; m[0x0f] = 0xff; piece(0x0a); piece(0x0b);
    m[0x0f] = 1; piece(0x0c);
    m[0x0f] = 0; m[5] = 0x13; m[9] = 0x19; this.blackRows(0x7c, 0x8f);
    piece(0); piece(m[0x8f]); piece(m[0x8a]); piece(0x0d);
    m[0x0f]++; piece(m[0x56]); piece(0x0f);
  },
  r7CDC() { this.r7D03(); this.sceneFrame(); },
  // $7C43: end of a frame of the scene
  sceneFrame() { if (this.m[0xd0] === 0) throw new StartOver(); this.r7603(); this.r7600(); },

  // $7606: bring the new picture in column by column, left to right, over the one that is showing
  r7606() {
    const m = this.m; this.g.clearRightEdge();
    const from = m[7] << 8, to = (m[7] ^ 0x60) << 8; m[0x80] = m[7] ^ 0x60;
    for (let y = 0; y < 0x28; y++) {
      for (let row = 0; row < 0xc0; row++) { const off = (m[0x08c0 + row] << 8) + m[0x0800 + row] + y; m[to + off] = m[from + off]; }
      if (this.key === 0x9b) { this.strobe(); this.paused = true; }
      this.cycles += 192 * 53 + 27 + waitCycles(0x30);
      this.flips.push({ at: this.cycles, show: m[7] ^ 0x60, page: m.slice(to, to + 0x2000) });
    }
    m[0x12] = m[0x0800 + 0xbf]; m[0x13] = u8(m[0x08c0 + 0xbf] + m[0x80]);
  },

  r7C53() { const m = this.m; m[0x31] = m[0x50]; m[0x8a] = 7; m[0x10] = 0; m[0x50] = 0; m[0x36]++; m[0x37] = m[0x44]; m[0x9e] = m[0xe5]; m[0x9f] = m[0xe4]; m[0x56] = 0x0e; m[0x44] = 3; },
  r7A57() { const m = this.m; for (let x = 15; x >= 0; x--) m[0x0f00 + x] = m[0x70 + x]; },
  r7A62() { const m = this.m; for (let x = 15; x >= 0; x--) m[0x70 + x] = m[0x0f00 + x]; },
  // $7A6D: one frame of the opponent walking in the hall
  r7A6D() {
    const m = this.m;
    this.r7009(); this.r6780(); this.a = 0; this.r6540(); this.r6400(); this.r7006();
    m[0xe5] = m[0xe4]; m[0xe4] = m[0x72];
    m[0x0f] = 1; this.x = 1; this.r780C(); this.x = 4; this.r780C();
    this.r7ADC(); this.r7AC1(); this.r6783(); this.r7803(); this.a = 0x1e; this.r7AB3();
    this.sceneFrame();
  },
  // $79A0: the whole scene
  r79A0() {
    const m = this.m;
    this.r7C53(); m[0x3e]++; m[0x84] = 5; m[0x8f] = 6; m[0x83] = 1;
    this.r7800(); this.r7803(); this.r7AB1(); this.r7D03(); this.r7606(); this.r1000();
    m[0x56] = 0x10; this.r7CDC(); this.r7CDC(); this.r7A57();
    m[0x70] = 7; m[0x71] = 7; m[0x72] = 0xfb; m[0xe4] = 0xfb; m[0xe5] = 0xfb; m[0x79] = 1; m[0x7c] = 1; m[0x7f] = 0xff;
    for (;;) { this.r7A6D(); if ((m[0x72] & 0x80) || m[0x72] < 3) continue; m[0x79] = 0; if (m[0x70] === 7) break; }   // he walks in and stops
    this.r7A6D(); m[0x8a]++; this.r7CDC(); m[0x8a]++; this.r7CDC(); this.r100C(); this.r7CDC();                       // Akuma points
    this.r7009(); this.a = 0xc2; this.r6540(); this.r7006();
    do this.r7A6D(); while (m[0x70] !== 7);                                                                           // he bows
    m[0x79] = 1; m[0x7c] = 0xff;
    do this.r7A6D(); while (!(m[0x72] & 0x80));                                                                       // and goes
    m[0x8a]--; this.r7D03(); this.r7A6D(); m[0x8a]--;
    do { this.r7D03(); this.r7A6D(); } while (m[0x72] >= 0xfb);
    this.r7A62(); m[0x3e]--;
    m[0x5f] = 2; m[0x54] = 2; m[0x36] = 0; m[0x50] = m[0x31]; m[0x44] = m[0x37]; m[0xe4] = m[0x9f]; m[0xe5] = m[0x9e];
  },
});
