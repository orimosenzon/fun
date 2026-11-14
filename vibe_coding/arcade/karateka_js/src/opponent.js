// The opponent's mind. Once per frame he decides on one command, by distance and by chance.
// The chances are thresholds on a random byte: tables at $BFE1-$BFF2 indexed by the distance,
// and three per-opponent values at $D7-$D9 (how readily he kicks, punches, comes forward).
import { extend, u8 } from './core.js';

extend({
  // $6C95: the game's random numbers: seed = seed * 17 + $A1
  r6C95() { const m = this.m; this.a = m[0x48] = u8(m[0x48] * 17 + 0xa1); },
  // $6C71: choose a height for the attack: high, middle or low
  r6C71() {
    const m = this.m; this.r6C95(); const r = this.a; this.x = m[0x4d];
    this.a = m[0x2a] = this.x === 1 ? (r >= 0x80 ? 0xff : 0) : r < 0x55 ? 0xff : r < 0xaa ? 0 : 1;
  },
  // $6B00: decide what the opponent does this frame
  r6B00() {
    const m = this.m, cmd = (a) => { this.a = a; this.r6540(); };
    const idle = () => { m[0x2a] = 0; m[0x29] = 0; cmd(0); };
    const rnd = () => { this.r6C95(); return this.a; };
    if (m[0x60] === 0x13) {                                  // the player is down
      if (m[0x4d] !== 0) return idle();
      this.x = 0; m[0x29] = 0; return cmd(m[0x2f] !== 0 ? m[0x2f] : 0x9b);
    }
    if (m[0x2f] !== 0) {                                     // he is standing upright
      if (m[0x30] !== 0) {                                   // walking in from the right
        let a = m[0x22]; if (m[0x62] >= 0x12) a = u8(a - 6);
        const arrived = a < 0x2d && (m[0x4b] === m[0x4c] || (!(m[0x52] & 0x80) && m[0x52] >= 0x31));
        if (arrived) { m[0x29] = 0; return cmd(0x9b); }
        if (m[0xd0] >= 3) return;
        m[0x29] = 1; return cmd(1);
      }
      if (m[0x22] >= 0x28) return;
      if (m[0x33] < 0x0d || m[0x6f] === 0) return cmd(0x9b);  // the player is near, or ready to fight: take the stance
      if (m[0x65] !== 0x1a) return;
      return cmd(0xc2);                                      // the player bows: bow back
    }
    if (m[0x5d] === 0) return idle();
    const forward = () => {
      m[0x29] = 1; m[0x2a] = 1; this.a = 1; limit();
    };
    const limit = () => {                                    // do not walk off the left of the scene
      if (m[0x44] === 0) { this.y = m[0xd0]; this.x = m[0x6c6c + this.y]; if (this.x >= m[0x22]) { this.x = 0; m[0x29] = 0; } } else this.y = m[0x44];
      this.r6540();
    };
    const other = () => {                                    // nothing decided: stand, or finish a step
      if (m[0x25] !== 3) return idle();
      this.a = m[0x29] = rnd() < m[0xbff2 + this.x] ? 1 : 0; limit();
    };
    const kick = () => { if (rnd() >= m[0xd7]) return other(); this.r6C71(); cmd(0xc5); };
    const maybeForward = () => { if (rnd() >= m[0xd9]) return other(); if (m[0x20] === 6) return forward(); other(); };
    if (m[0x6f] !== 0) return m[0x33] < 8 ? kick() : forward();   // the player is not in his stance
    this.x = m[0x33];
    if (this.x >= 0x0c) return forward();
    if (this.x < 7) this.x = 7;
    const x = this.x;
    if (m[0x4d] !== 0 && m[0x22] >= 0x20 && m[0x22] < 0x27) {
      const r = rnd();
      if (r >= m[0xbfe1 + x]) return other();
      if (r < m[0xbfeb + x]) return maybeForward();
      m[0x2a] = 0; return cmd(0xc5);
    }
    const r = rnd();
    if (r >= m[0xbfe1 + x]) return other();
    if (r >= m[0xbfe6 + x]) return kick();
    if (r < m[0xbfeb + x]) return maybeForward();
    if (rnd() >= m[0xd8]) return other();                    // punch
    this.r6C71();
    if (m[0x26] === 0x0d && m[0x4d] !== 0) m[0x2a] = 0;
    if (m[0x20] !== 0x16 && (m[0x20] !== 6 || m[0x21] !== 6)) return other();
    m[0x29] = 0; cmd(0xd7);
  },
  // $6D03: the joystick's buttons, when the game is played with one ($C4 = 0)
  r6D03() {
    const m = this.m; if (m[0xc4] !== 0) return;
    for (const [flag, n] of [[0x46, m[0xc3]], [0x47, m[0xc3] ^ 1]]) {
      if (m[flag] === 0xff) continue;
      if (!this.button(n)) m[flag] = 0; else if (!(m[flag] & 0x80)) m[flag] = 0xff;
    }
  },
  button() { return false; },   // no joystick here
});
