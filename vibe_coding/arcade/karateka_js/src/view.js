// The view follows the player. $53 is its speed in pixels per frame (negative = the scene moves
// left under the fighters), $50 the fine position (0-6 pixels), $52 the column; when a whole
// column has gone by, everything on the screen moves one column: both fighters and the scenery.
import { extend, u8 } from './core.js';

extend({
  // $B3DF: jump the view by A columns
  rB3DF() { const m = this.m, a = this.a; m[0x80] = a; m[0x62] = u8(m[0x62] + a); m[0x72] = u8(m[0x72] + a); m[0x91] = u8(m[0x91] + a); },
  // $B34A: move the scene A pixels (A negative)
  rB34A() {
    const m = this.m; let a = u8(this.a + m[0x50]);
    if (!(a & 0x80)) { m[0x50] = a; return; }
    m[0x50] = u8(a + 7); m[0x72]--; m[0x62]--; m[0x52]--; m[0x91]--;
    a = u8(m[0x51] - 1); if (a === 0xf7) a = 0x27; m[0x51] = a;
    if (m[0x52] === 0x1a) { m[0x52]++; m[0x51]++; m[0x62]++; m[0x72]++; m[0x50] = 0; m[0x53] = 0; }
  },
  // $B260: where should the view be going
  rB260() {
    const m = this.m, stop = () => { m[0x53] = 0; };
    let a = m[0x62];
    const slowDown = () => {                               // towards +4, the speed for following a retreat
      let v = m[0x53];
      if (v !== 4) {
        if (v < 4 || (v - 4) & 0x80) v = u8(v + 2);
        else { v -= 2; if (v === 0) { m[0x53] = 0; return; } }
        m[0x53] = v;
      }
      shift(v);
    };
    const shift = (v) => {
      for (;;) {
        if (v & 0x80) {                                    // the scene moves left
          v = m[0x53];
          if (!(v & 0x80)) continue;
          if (v < 0xfa) { this.a = u8(v + 6); this.rB34A(); v = 0xfa; }
          this.a = v; return this.rB34A();
        }
        break;
      }
      let p = u8(v + m[0x50]);
      if (p >= 7) {
        m[0x50] = p - 7; m[0x62]++; m[0x72]++; m[0x52]++; m[0x91]++;
        let c = u8(m[0x51] + 1); if (c === 0x28) c = 0xf8; m[0x51] = c;
      } else m[0x50] = p;
      if (m[0x44] === 0 && m[0x52] === 0x30) { m[0x50] = 1; return stop(); }
      if (m[0x52] === 0x0b) { m[0x50] = 0; return stop(); }
    };
    if ((a & 0x80) || a < 3) {                             // the player is at the left edge
      if (m[0x44] !== 0 && m[0x44] !== m[0x4c]) { m[0x44]--; m[0x54] = 0; m[0x52] = 0x1b; this.a = 0x1a; this.rB3DF(); m[0x51] = u8(m[0x51] + 2); return; }
      m[0x53] = 1; return slowDown();
    }
    if (m[0x44] !== 0) { if (a >= 0x2a) { m[0xd0] = 2; return; } }
    else if (a >= 0x21) {
      if (m[0x30] !== 0) { m[0x44] = 1; m[0x54] = 0; m[0x52] = 0x0b; m[0x50] = 0; this.a = 0xe6; this.rB3DF(); m[0x51] = u8(m[0x51] - 2); return; }
      if (a >= 0x2a) { m[0xaf] = 1; return; }
    }
    if (m[0x52] === 0x1b && m[0x50] === 0) { if (m[0x62] >= 0x0f) return stop(); }
    else if (m[0x52] === 0x0b && m[0x50] === 0) { if (m[0x62] < 0x0f) return stop(); }
    a = m[0x62];
    if (a === 0x0f) return stop();
    if (a < 0x0f) return slowDown();
    // the player is right of the middle: the scene moves left, faster when he runs
    let want = 0xfc;
    if (m[0x6f] !== 0) { want = 0xfa; if (m[0x65] >= 0x42 && m[0x65] < 0x4a) want = 0xf6; }
    if (want !== m[0x53]) {
      if (want < m[0x53]) m[0x53] = u8(want - 2);
      else if (!(m[0x53] & 0x80)) m[0x53] = u8(m[0x53] - 2);
      else { m[0x53] = u8(m[0x53] + 2); if (m[0x53] === 0) return; }
    }
    shift(0x80);
  },
});
