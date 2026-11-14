// Blows: does an attack land, and what happens to the one it lands on.
import { extend, u8 } from './core.js';

extend({
  // $7000 / $7003: a fresh player at the left, a fresh opponent at the right, both standing upright
  r7000() { const m = this.m; m[0x67] = m[0x68] = m[0x65] = m[0x66] = m[0x69] = 0; m[0x60] = m[0x61] = 7; m[0x62] = 0x0b; m[0x6b] = 0; m[0x6d] = 0; m[0x6f] = 0xff; m[0x6a] = 0xff; m[0x6c] = 1; },
  r7003() { const m = this.m; m[0x77] = m[0x78] = m[0x75] = m[0x76] = m[0x79] = m[0x7a] = 0; m[0x70] = m[0x71] = 7; m[0x72] = 0x30; m[0x7b] = 0; m[0x7d] = 0; m[0x4b] = 0; m[0x7f] = 0xff; m[0x7c] = 0xff; },
  r7006() { this.r7071(); }, r7009() { this.r707D(); }, r700C() { this.r708D(); }, r700F() { this.r7099(); },
  r7018() { this.r70F0(); }, r701B() { this.r710D(); }, r701E() { this.r7207(); },

  // $70F0 / $710D: has the player / the opponent walked off the screen, and on which side
  r70F0() { const m = this.m, a = u8(0x30 - m[0x52] + m[0x62]); if (a < 5 || a >= 0xfb) m[0x4c] = a & 0x80 ? 0 : 1; this.a = a; },
  r710D() { const m = this.m, a = u8(0x30 - m[0x52] + m[0x72]); if (a < 5 || a >= 0xfb) m[0x4b] = a & 0x80 ? 0 : 1; this.a = a; },

  // $7366: judge an attack. Y = the attack (0-2 kicks high/middle/low, 3-5 and 6-8 punches, $FF none),
  // $32 = the class of the defender's pose, $33 = the distance, $A3 = 1 when the opponent attacks.
  // Returns in A: 0 a miss, 1 parried, 4 too close (the two clash), anything else a hit.
  r7366() {
    const m = this.m, y = this.y;
    if (m[0x5e] === 0) {                                   // no opponent: the only thing to hit is the gate's lever
      this.a = 0;
      if (m[0xd0] < 3 || m[0x62] < 0x1d || m[0x62] >= 0x20 || m[0x84] === 0 || y === 0xff || y >= 6) return;
      if (y >= 3) { m[0x43] = 1; return; }
      m[0x4e] = 1; m[0x43] = 1; m[0xba] = 0x1e; m[0x84] = 0; m[0x83] = 2; m[0xc5]++; this.a = 0xff; return;
    }
    if (y !== 0xff && y >= 6) {                            // a punch reaches at exactly one distance
      const k = m[0x731a + m[0x32]], p = m[0x60] < 0x17 ? 0 : m[0x60] - 0x17;
      this.a = u8(m[0x7446 + k] + m[0x7450 + p]) === m[0x33] ? 1 : 0; return;
    }
    const k = m[0x80] = m[0x731a + m[0x32]];
    let a = m[0x7300 + m[0x32]];                          // the distance at which this pose is hit squarely
    const square = () => { this.a = y === 0xff ? 0 : y <= 2 ? 3 : y <= 4 ? m[0x7348 + k] : m[0x7352 + k]; };
    if (a === m[0x33]) return square();
    if (a > m[0x33]) {                                     // nearer than that
      if (m[0xa3] !== 0) { this.a = 4; return; }
      if (m[0x33] < 5 && m[0x60] !== 7) { m[0x62] = u8(m[0x62] - 2); if (m[0x6f] !== 0) { this.a = 4; return; } a = 0; }
      else if (m[0x33] < 5) a = 7;
    }
    if (y === 0xff) { this.a = 0; return; }
    a = u8(a + 2);                                         // one step further away: the tip of the foot
    if (a === m[0x33]) { this.a = y === 0 || y === 2 ? m[0x7334 + k] : y === 1 ? m[0x733e + k] : m[0x735c + k]; return; }
    if (a < m[0x33]) { this.a = 0; return; }
    square();
  },

  // $701E: act on the two verdicts ($40 the opponent's attack on the player, $41 the player's on him)
  r7207() {
    const m = this.m;
    let y = m[0x40];
    if (!(y & 0x80) && m[0x70] === 0x13 && m[0x4d] !== 5) return;
    if (y !== 0) {
      m[0x42] = 0; this.r7099();
      let done = false;
      if (y !== 4) {
        this.r0B09(); m[0x42] = 1;
        if (m[0x5d] === 0) {                               // the player is beaten
          if (m[0x4d] === 1) { m[0xba] = 0x28; m[0x4e] = 1; }
          this.r7145();
          if (m[0x4d] !== 1 && y === 0x88) m[0x22] = 0x1d;
          done = true;
        }
      }
      if (!done) {
        if (y === 0x88 && m[0x2f] !== 0) { m[0x22] = 0x1e; m[0x27] = 1; m[0x28] = 1; m[0x25] = 0x4c; m[0x26] = 0x34; }
        else { this.a = m[0x40]; this.y = y; this.r72BE(); }
      }
      this.r6400(); this.r70F0(); this.r708D();
    }
    y = m[0x41]; if (y === 0) return;
    m[0x43] = 0; this.r707D();
    if (y !== 4) {
      this.r0B0C(); m[0x43] = 1;
      if (m[0x5e] === 0) { m[0xba] = 0x1e; m[0x4e] = 1; this.r7145(); this.r6400(); this.r710D(); this.r7071(); return; }
    }
    if (m[0x5a] !== 0) return;
    this.a = m[0x41]; this.y = y; this.r72BE();
    this.r6400(); this.r710D(); this.r7071();
  },
  // $7145: knocked down
  r7145() { this.a = 0xc6; this.r6540(); },
  // $72BE: the one who was hit reacts (A = the verdict, the working copy = him)
  r72BE() {
    const m = this.m, a = this.a;
    if (a === 1) return;
    if (a === 0xff) return this.fall();
    const p = m[0x20], recoil = (i) => { m[0x25] = i; m[0x27] = 0xff; }, shove = () => { m[0x22] = u8(m[0x22] - m[0x2c]); };
    const armsUp = () => { m[0x26] = 0x39; m[0x28] = 1; };
    if (p === 0) { armsUp(); return recoil(0x58); }
    if (p === 1) { armsUp(); return recoil(0x59); }
    if (p === 5 || p === 9 || p === 0x0a) { armsUp(); m[0x2f] = 0; return recoil(0x5d); }
    if (p < 5 || (p >= 0x0e && p <= 0x16)) { m[0x25]--; m[0x26]--; m[0x24] = 0x0a; return; }   // caught in the middle of a kick: it is taken back
    if (p === 6 || p === 0x1f || p === 0x20) return recoil(0x5d);
    if (p < 9 || p < 0x0e || p >= 0x22) { if (m[0x62] === m[0x22]) this.fall(); return; }        // caught standing upright: he goes down
    if (p === 0x21) { m[0x25] = 0x4f; m[0x27] = 1; m[0x22] = u8(m[0x22] - 2 * m[0x2c]); return; }
    if (p < 0x19) { m[0x25] = 0x4f; m[0x27] = 1; return; }
    if (p === 0x19) return recoil(0x57);
    if (p === 0x1a) { shove(); return recoil(0x57); }
    if (p === 0x1d) { shove(); return recoil(0x59); }
    if (p === 0x1e) { shove(); shove(); return recoil(0x59); }
    recoil(0x58);
  },
  // $7131: the player is killed outright
  fall() { const m = this.m; m[0xb6] = 0; m[0x5d] = 0; if (this.y === 4) { m[0x42] = 1; m[0x40]--; m[0x7a] = 0; } this.r7145(); },

  // $74C0: the flash where a blow lands (X = which kind of blow)
  r74C0() {
    const m = this.m, x = this.x;
    const star = (col, a, b, c, d) => {
      m[5] = u8(col); m[0x10] = m[0x50]; m[3] = a; m[4] = b; m[0x0f] = 0xff; this.r1903();
      m[6] = u8(m[6] + 3); m[3] = c; if (d !== undefined) m[4] = d; m[0x0f] = 0; this.r1903();
    };
    if (m[0x97] === 2) { m[0x97] = 0; m[6] = u8(m[0x93] + 2); return star(m[0x91] - 3, 0xab, 0x93, 0x2a, 0x94); }
    if (m[0x97] === 1) { m[0x97] = 0; m[6] = u8(m[0x93] + 2); return star(m[0x91], 0xab, 0x93, 0x2a, 0x94); }
    const y = u8(m[0x2a] + 1), side = () => (m[0x2c] & 0x80 ? m[0x22] - 8 : m[0x22] + 4);
    if (x === 1) { m[6] = m[0x74fa + y]; let col = side(); if (m[0x20] === 0x1a || m[0x20] === 0x1d) col -= m[0x2c]; return star(col, 0x90, 0x94, 0xe6); }
    if (x === 2 || x === 3) { m[6] = m[(x === 2 ? 0x74f7 : 0x74fd) + y]; return star(side(), 0xab, 0x93, 0x2a, 0x94); }
    if (m[0x9a] === 0xfe) { m[6] = 0x7b; star(0x13, 0xab, 0x93, 0x2a, 0x94); }
  },
});
