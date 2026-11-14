// A fighter's movement. Every movement is a script: a list of poses in the table at $6300
// (legs) and $63BE (upper body), with a horizontal step per frame at $6360. The fighter's
// record holds an index into each list and a direction (+1, -1 or 0 = standing still).
//
// Record fields used here (work copy at $20): $20 legs pose, $21 upper body pose, $22 column,
// $24 kicks in a row, $25/$26 script index legs/upper, $27/$28 script direction legs/upper,
// $29 wish to move (+1 forward, -1 back), $2A height of the attack (-1 high, 0 middle, 1 low),
// $2B the attack that lands this frame ($FF none), $2C facing (+1 player, -1 opponent),
// $2F 0 = fighting stance, otherwise standing upright.
import { extend, u8 } from './core.js';

const POSES = 0x6300, STEPS = 0x6360, UPPER = 0x63be;

extend({
  // $6400: one frame of the fighter's movement scripts
  r6400() {
    const m = this.m;
    for (;;) {
      m[0x2b] = 0xff;
      if (m[0x27] !== 0) {
        const x = u8(m[0x25] + m[0x27]); m[0x25] = x;
        let d = m[0x27] & 0x80 ? u8(-m[STEPS + u8(x + 1)]) : m[STEPS + x];
        if (m[0x2c] & 0x80) d = u8(-d);
        m[0x22] = u8(m[0x22] + d);
        const p = m[POSES + x];
        if (p < 0x80) m[0x20] = p;
        else if (p === 0xff) m[0x27] = 0;
        else if (p === 0x80) {              // the kick goes out
          m[0x45]++;
          const h = m[0x2a], pose = h & 0x80 ? 2 : h === 1 ? 4 : 3;
          m[0x20] = pose; m[0x21] = pose; m[0x26]++; m[0x2b] = pose - 2; return;
        } else if (p === 0x81) {            // end of a step: keep walking, or settle
          if (m[0x29] !== 0 && !(m[0x29] & 0x80) && m[0x33] >= 8) { m[0x25] = 0x39; m[0x26] = 0x39; m[0x27] = 1; m[0x28] = 1; }
          else if (m[0x29] & 0x80) { m[0x25] = 5; m[0x26] = 5; m[0x27] = 1; m[0x28] = 1; }
          else { m[0x25] = 0x3a; m[0x27] = 0xff; m[0x28] = 1; m[0x26] = 0x39; m[0x22] = u8(m[0x22] + 2 * m[0x2c]); }
          continue;
        } else if (p === 0x82) { m[0x26] = 0x39; m[0x27] = 0xff; m[0x25] = 0x30; continue; }
        else if (p === 0x83) { m[0x25] = 0; m[0x26] = 0; continue; }
      }
      break;
    }
    if (m[0x28] === 0) return;
    const x = u8(m[0x26] + m[0x28]); m[0x26] = x;
    const p = m[UPPER + x];
    if (p < 0x80) m[0x21] = p;
    else if (p === 0xff) m[0x28] = 0;
    else if (p === 0x86) { m[0x45]++; const h = m[0x2a], k = h & 0x80 ? 0 : h === 1 ? 2 : 1; m[0x21] = 0x19 + k; m[0x2b] = 6 + k; }   // the punch goes out
    else if (p === 0x84) { m[0x45]++; const h = m[0x2a], k = h & 0x80 ? 0 : h === 1 ? 2 : 1; m[0x21] = 0x16 + k; m[0x2b] = 3 + k; }
  },

  // $6540: a command for the fighter (A): $C5 kick, $D7 punch, $D1 raise or lower the guard, $C6 step in,
  // $9B change between standing and the fighting stance, $C2 bow. Anything else: just keep moving.
  r6540() {
    const m = this.m, a = this.a, fighting = m[0x2f] === 0;
    if (fighting && a === 0xd1) {
      const u = m[0x21];
      if (m[0x2a] & 0x80) { if (u === 6 || u === 0x1d) m[0x21] = 0x1c; } else if (u === 6 || u === 0x1c) m[0x21] = 0x1d;
      return this.move();
    }
    if (fighting && a === 0xd7) {
      if (m[0x24] >= 5) { m[0x24] = 0; return this.move(); }
      if (m[0x20] === 0x16 && (m[0x26] === 0x39 || m[0x26] === 0x0d)) { m[0x24] = 1; m[0x26] = m[0x26] === 0x39 ? 0x0b : 0x37; m[0x25] = 0x0b; return this.r674C(); }
      if (m[0x20] === 0x16) m[0x24] = 1;
      const u = m[0x21];
      if (u !== 6 && u !== 0x1c && u !== 0x1d) return this.move();
      if (m[0x20] === 6) { m[0x24] = 0; m[0x27] = 1; m[0x28] = 1; m[0x25] = 0x0a; m[0x26] = 0x0a; return this.r674C(); }
      m[0x24] = 1; m[0x28] = 1; m[0x26] = 0x38; this.r674C(); return this.move();
    }
    if (fighting && a === 0xc5) {
      const p = m[0x20];
      if (p === 6) { m[0x25] = 7; m[0x26] = 7; m[0x27] = 1; m[0x28] = 1; m[0x24] = 0; return this.r6741(); }
      if (p === 1) { if (m[0x24] >= 3) return this.move(); m[0x24]++; m[0x25] = 1; m[0x26] = 1; return this.r6741(); }
      if (p !== 0x1b || m[0x27] & 0x80) return this.move();
      m[0x25] = 0xff; m[0x26] = 0xff; m[0x27] = 1; m[0x28] = 1; m[0x24] = 0; return this.r6741();
    }
    if (a === 0xc6) {
      const p = m[0x20];
      if (p >= 0x0e && p < 0x14) return;
      const odd = (m[0x22] ^ m[0x50]) & 1;
      if (m[0x22] === m[0x62] ? !odd : odd) m[0x22] = u8(m[0x22] - m[0x2c]);
      m[0x25] = 0x20; m[0x26] = 0x20; m[0x27] = 1; m[0x28] = 1; return this.r674C();
    }
    if (a === 0x9b) {
      if (!fighting) { if (m[0x20] !== 7 || m[0x21] !== 7) return this.move(); m[0x27] = 1; m[0x28] = 1; m[0x25] = 0x11; m[0x26] = 0x11; }
      else { if (m[0x20] !== 6) return this.move(); m[0x27] = 0xff; m[0x28] = 0xff; m[0x25] = 0x15; m[0x26] = 0x15; }
      m[0x2f] ^= 0xff; this.r6741(); return this.move();
    }
    if (!fighting && a === 0xc2) {
      if (m[0x20] !== 7 || m[0x21] !== 7) return this.move();
      m[0x25] = 0x16; m[0x26] = 0x16; m[0x27] = 1; m[0x28] = 1; return this.r674C();
    }
    this.move();
  },
  // $6572: act on the wish to move
  move() {
    const m = this.m;
    if (m[0x2f] !== 0) {                       // standing upright: walk or run
      const p = m[0x20];
      if (p === 7) { if (m[0x29] !== 0 && !(m[0x29] & 0x80)) { m[0x27] = 1; m[0x28] = 1; m[0x25] = 0x3f; m[0x26] = 0x2a; } }
      else if (p === 0x23) { if (m[0x29] !== 1) { m[0x25] = 0x49; m[0x26] = 0x34; } }
      else if (m[0x25] === 0x49) { m[0x25] = 0x41; m[0x26] = 0x2c; }
      return;
    }
    if (m[0x29] === 0) return;
    const p = m[0x20];
    if (p === 6) { m[0x27] = m[0x29]; m[0x25] = m[0x29] & 0x80 ? 0x3d : 0x37; return; }
    if (p < 0x17) return;
    const i = m[0x25];
    if (i >= 0x3f) return;
    if (i === 0x39) { m[0x25] = 0x2c; return; }
    if (i === 0x3b) { m[0x25] = 0x33; return; }
    m[0x27] = m[0x29];
  },
  // $6741 / $674C: the player (and not the opponent) has begun something: note it for the sound
  r6741() { const m = this.m; if (m[0x62] === m[0x22]) m[0x46] = 0x80; },
  r674C() { const m = this.m; if (m[0x62] === m[0x22]) m[0x47] = 0x80; },
});
