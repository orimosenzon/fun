// The bottom of the screen: each fighter's strength as a row of arrowheads, the player's
// from the left and the opponent's from the right. Strength comes back slowly with time.
import { extend, u8 } from './core.js';

extend({
  // $0B00: clear the strip under the scene
  r0B00() { const m = this.m; m[5] = 0; m[9] = 0x28; m[6] = 0xb9; m[8] = 0xc0; m[2] = 0x80; m[0x11] = 0x80; this.r0A00(); },
  // $0B03: the player's arrowheads. With one or two left they blink (drawn on one page only)
  r0B03() {
    const m = this.m; if (m[0xb6] === 0 || (m[0xb6] < 3 && m[7] === 0x20)) return;
    m[0x0f] = 1; m[5] = 0; m[0x10] = 1; m[6] = 0xb9; m[3] = 0x12; m[4] = 0x0b;
    for (let x = 0; ; ) { this.x = x; this.r1903(); let s = m[0x10] + 3; if (s >= 7) { s -= 7; m[5]++; } m[0x10] = s; m[5]++; if (++x >= m[0xb6]) break; }
  },
  // $0B06: the opponent's
  r0B06() {
    const m = this.m; if (m[0xb7] === 0 || (m[0xb7] < 3 && m[7] === 0x20)) return;
    m[0x0f] = 1; m[5] = 0x26; m[0x10] = 6; m[6] = 0xb9; m[3] = 0x12; m[4] = 0x0b;
    for (let x = 0; ; ) { this.x = x; this.r1909(); let s = m[0x10] - 3; if (s < 0) { s += 7; m[5] = u8(m[5] - 1); } m[0x10] = s; m[5] = u8(m[5] - 1); if (++x >= m[0xb7]) break; }
  },
  // $0B09 / $0B0C: a blow costs the player / the opponent one point; at none he is beaten
  r0B09() { const m = this.m; if (m[0xb6] === 0) return; m[0x5b] = 0; if (--m[0xb6] === 0) m[0x5d] = 0; },
  r0B0C() { const m = this.m; if (m[0xb7] === 0) return; m[0x5c] = 0; if (--m[0xb7] === 0) m[0x5e] = 0; },
  // $0B0F: strength returning: every $B8 frames a point to the player, every $B9 to the opponent
  r0B0F() {
    const m = this.m;
    if (++m[0x5b] >= m[0xb8]) { if (m[0x5d] !== 0) { if (m[0xb6] === m[0xb0]) m[0xb8] = 0x1e; else m[0xb6]++; } m[0x5b] = 0; }
    if (++m[0x5c] >= m[0xb9]) { if (m[0x5e] !== 0 && m[0xb7] !== m[0xb1]) m[0xb7]++; m[0x5c] = 0; }
  },
});
