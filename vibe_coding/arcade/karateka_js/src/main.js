// The main loop: one call of rB7CA followed by r7600 is one frame of the game.
import { extend, u8 } from './core.js';
import { waitCycles } from './sound.js';

export class GameOver extends Error { constructor() { super('game over'); } }      // the original jumps to $0203
export class NextLevel extends Error { constructor() { super('next level'); } }    // the original jumps to $0200

extend({
  // $BB41 / $BB52: full strength for the opponent / the player
  rBB41() { const m = this.m; m[0xb7] = m[0xb1]; m[0xb3] = 6; m[0xb5] = 0x26; m[0x5b] = 0; },
  rBB52() { const m = this.m; m[0xb6] = m[0xb0]; m[0xb2] = 1; m[0xb4] = 0; m[0x5c] = 0; },
  // $BB63: the temper of opponent number $3E: how readily he kicks, punches and comes forward
  rBB63() { const m = this.m, x = this.x = m[0x3e]; m[0xd7] = m[0xbb75 + x]; m[0xd8] = m[0xbb7b + x]; m[0xd9] = m[0xbb81 + x]; },
  rBB1D() { const m = this.m; m[0xba] = this.a; m[0x4e] = 1; },                  // ask for the recorded cry, at speed A
  // $B600: one of the pictures of the opening of a level
  rB600() { const m = this.m, x = this.x; m[3] = m[0xb603 + x]; m[4] = m[0xb623 + x]; m[6] = m[0xb663 + x]; m[5] = m[0xb643 + x]; this.r1903(); },
  rBB31() { const m = this.m; m[0x0f] = 1; m[0x10] = 0; this.x++; this.rB600(); this.x++; this.rB600(); },

  // $BC4A: a piece of the gate
  rBC4A() { const m = this.m, x = this.x; m[3] = m[0xbc06 + x]; m[4] = m[0xbc17 + x]; m[6] = m[0xbc39 + x]; m[5] = u8(m[0x52] + m[0xbc28 + x]); this.r1903(); },
  // $BC00: the gate at the end of the road, behind the fighters
  rBC00() {
    const m = this.m, piece = (x) => { this.x = x; this.rBC4A(); };
    if (m[0x4c] === 0 || m[0x52] >= 0x2c) return;
    const inside = (v, lo) => v < 0x2c && v >= lo;
    let x = inside(m[0x72], 0x16) || inside(m[0x62], 0x13) ? 1 : 0;
    m[0x10] = m[0x50]; m[0x0f] = 1;
    if (m[0x54] === 0 && m[0x52] === m[0xe1] && m[0x50] === m[0xae]) { m[0x3f] = x; if (x === 0) return; }
    else { m[0x3f] = 1; for (x = 1; x < 8; x++) { m[0x80] = x; piece(x); } piece(0x0b); }
    piece(0x0a);
    do { m[6] = u8(m[6] + 2); this.r1903(); } while (m[6] < 0x95);
    piece(0x0d); piece(0x0e); piece(0x0f); piece(0x10);
  },
  // $BC03: the part of the gate in front of the fighters
  rBC03() {
    const m = this.m;
    if (m[0x52] >= 0x2c || m[0x4c] === 0 || m[0x3f] === 0) return;
    m[0x0f] = 1; this.x = 0; this.rBC4A();
    do { m[6] = u8(m[6] + 2); this.r1903(); } while (m[6] < 0x93);
    this.x = 0x0c; this.rBC4A();
  },

  // $BA91: the player's turn
  rBA91() {
    const m = this.m;
    m[0x34] = 0; this.r700F();
    let asked = false;
    if (m[0x5e] === 0) {
      if (m[0x88] !== 0 || m[0x70] !== 0x13) { m[0x29] = 0; this.a = 0xd1; this.r6540(); asked = true; }
      else if (m[0x21] === 0x1c || m[0x21] === 0x1d) { m[0x21] = 6; m[0x46] = 0; m[0x47] = 0; }
    }
    if (!asked) {
      let walkOn = false;
      if (m[0x22] >= 0x24) {                               // at the right edge of the screen
        walkOn = true;
        if (m[0x4c] !== 0) { if (m[0x5e] !== 0) { if (m[0x72] < 0x2e) walkOn = false; else m[0x72] = 0x3c; } if (walkOn) m[0x82] = 0x64; }
      }
      if (walkOn) { m[0x29] = 1; this.a = 0; this.r6540(); } else this.r6E00();
    }
    this.r6400();
    if (m[0x25] === 0x15) { this.a = 0x1e; this.rBB1D(); }
    if (m[0x4c] === 0 && m[0x44] === 0 && u8(0x30 - m[0x52] + m[0x22]) < 0x0b) m[0xa2] = 1;   // he stepped back off the cliff
    this.r700C();
  },

  // $B7CA: one frame
  rB7CA() {
    const m = this.m;
    if (m[0xa1] !== 0 && m[0xa1] <= 0x30) {
      if (m[0xa1] < 0x30) return this.rB874();
      m[0xb6] = 0; m[0xb4] = 0; m[0x5d] = 0; m[0x42] = 1;
      this.rB874(); this.r7600(); this.rB874(); this.r7600(); this.rB874(); this.r7600();
      this.r1003(); throw new GameOver();
    }
    if (m[0x60] === 0x13 && m[0x90] !== 0) {               // the player lies beaten
      const p = m[0x70];
      if (p === 0x13 || p === 7 || (m[0x4d] !== 0 && (m[0x4d] === 5 || p === 6))) { this.r1003(); throw new GameOver(); }
    }
    if (m[0x5d] !== 0) {
      let skip = false;
      if (m[0x82] !== 0) {                                 // the opponent is beaten: the next one is on his way
        if (m[0x30] === 0) skip = true;
        else if (m[0x82] === 0x14) {
          if (m[0x3e] < 5) { m[0x3e]++; this.rBB63(); }
          this.rBB41(); this.rB400(); m[0x5e] = 1; m[0x46] = 0; m[0x47] = 0; m[0x89] = 0x19;
        } else m[0x82]++;
      }
      if (!skip && m[0x89] === 0 && m[0x5e] !== 0) { this.rB403(); m[0x46] = 0; m[0x47] = 0; m[0x90] = u8(m[0x92] + 4); m[0x89] = 0x19; m[0x8b] = 0x0f; }
    }
    this.rB874();
  },

  // $B874: the frame proper: the two fighters think and move, blows are judged, everything is drawn
  rB874() {
    const m = this.m;
    this.r6D03();
    if (m[0x70] === 0x13 && m[0x60] === 6 && m[0x82] === 0 && m[0x5d] !== 0) {   // the opponent has just gone down
      m[0x88] = 1; m[0x82] = 1; m[0xb8] = 2;
      if (m[0xb0] >= 0x0a) { m[0xb0]--; m[0xb1]++; if (m[0xb0] < m[0xb6]) this.rBB52(); }
    }
    let thinking = m[0xa1] === 0;
    if (thinking && m[0x99] !== 0) thinking = false;
    const opening = m[0xa1] === 0 && m[0x99] !== 0;
    if (!opening) {
      if (m[0xa1] === 0) {
        if (m[0xa2] !== 0) { m[0xa1] = 0x1f; m[0x5d] = 0; m[0x60] = 0x13; }
        else {
          this.rBA91();
          if (m[0x4c] === 0 && m[0x4b] === 0 && u8(m[0x72] - m[0x62]) >= 0x3c && u8(0x30 - m[0x52] + m[0x72]) >= 0x50) {
            if (m[0x69] === 1) m[0x8b] = 0x0f; else if (--m[0x8b] === 0) { m[0x89] = 1; m[0x8b] = 0x0f; }
            m[0x89]--;
          }
        }
      }
      if (m[0x4d] < 2 && (m[7] === 0x20 || m[0x75] >= 0x0e || m[0x77] === 0)) {  // the opponent's turn
        m[0x34] = 0xfe; this.r7009(); this.r6B00(); this.r6400(); this.r7006();
      }
      this.r6D03();
      m[0x40] = 0; m[0x41] = 0;
      if (m[0x5e] !== 0 && m[0x5d] !== 0) {
        this.a = m[0x70]; this.r7012(); m[0xa3] = 0; this.y = m[0x6b]; this.r7366(); m[0x41] = this.a;
        if (m[7] === 0x20) { this.a = m[0x60]; this.r7012(); m[0xa3] = 1; this.y = m[0x7b]; this.r7366(); m[0x40] = this.a; }
        this.r701E();
      }
      this.r0B0F();
      if (m[0x5d] === 0) m[0x96] = 1;
      this.rB260();
      if (m[0xaf] !== 0) { m[0xaf]--; if (m[0x30] === 0 && m[0x82] !== 0) { this.r79A0(); m[0x30]++; return this.rB7CA(); } }
      if (m[0xd0] === 2) { if (m[7] === 0x40) throw new NextLevel(); if (--m[0xd0] === 0) throw new NextLevel(); }
      this.r7018(); this.r701B();
    }
    // draw
    this.r6D03(); this.r678C(); this.r6780(); this.r6D03();
    this.rAD12(); this.rB0C3(); this.rAD0C(); this.r7603(); this.rB406(); this.r0B00(); this.r6D03();
    this.rAD03(); this.r6D03(); this.rAD06(); this.rBC00(); this.r6D03(); this.rB0C0();
    const opponentInView = () => (m[0x72] >= 0xfc || m[0x72] < 0x2e) && (m[0x4b] === m[0x4c] || (!(m[0x52] & 0x80) && m[0x52] >= 0x31)) && (m[0x44] !== 0 || !(m[0x72] & 0x80));
    let shadows = true;
    if (m[0xa1] === 0) {
      if (m[0x99] !== 0) { if (m[0x99] < 0x17) shadows = false; else m[0x99] = 0; }
      if (shadows) {
        m[0x10] = m[0x50]; this.r6789();
        if (!(m[0x72] >= 0xfc || m[0x72] < 0x2e) || !(m[0x4b] === m[0x4c] || (!(m[0x52] & 0x80) && m[0x52] >= 0x31))) shadows = false;
      }
    }
    if (shadows && (m[0x44] !== 0 || !(m[0x72] & 0x80))) this.r6786();
    this.r0B03(); this.r6D03();
    if (m[0x99] !== 0) { this.x = m[0x99]; this.rBB31(); m[0x99] = this.x; }
    else if (m[0xa1] !== 0) { this.rAC30(); this.cycles += waitCycles(0x14); }
    else { this.r700F(); this.r6783(); }
    if (opponentInView()) {
      if (m[0x70] === 0x13) this.r6786();
      this.r7009(); this.r6783(); this.r0B06(); this.x = m[0x40]; this.r74C0();
    }
    this.r6D03(); this.r700F(); this.x = m[0x41]; this.r74C0();
    m[0x0f] = 1; this.rAD09(); m[0x10] = m[0x50]; this.rBC03(); this.rB409();
  },
});
