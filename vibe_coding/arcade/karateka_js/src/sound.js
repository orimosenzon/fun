// Sound. The Apple II has a speaker that can only be flipped, and the original made every
// sound by flipping it in timed loops. These routines do the same sums and write down when
// each flip falls (this.clicks, in processor cycles); the page turns them into audio.
// While a sound plays the original game stands still, so the clock moves on here too.
import { extend, u8 } from './core.js';

// the Apple monitor's WAIT with A = n, the delay the original's loops are built on (with its JSR)
export function waitCycles(n) {
  let a = n & 0xff, carry = 1, cyc = 2;
  do {
    cyc += 3; let t = a;
    do { t = t - 1 - (1 - carry); carry = t >= 0 ? 1 : 0; t &= 0xff; cyc += t ? 5 : 4; } while (t);
    cyc += 4; a = a - 1 - (1 - carry); carry = a >= 0 ? 1 : 0; a &= 0xff; cyc += a ? 5 : 4;
  } while (a);
  return cyc + 12;
}

extend({
  click() { this.clicks.push(this.cycles); },
  // $0C40: a blow lands: a falling tone
  r0C40() { this.cycles += 10; for (let y = 2; y !== 0x28; y++) { this.cycles += 4; this.click(); this.cycles += 2 + waitCycles(y) + 7; } },
  // $0C43 / $0C46: the two sounds of a blow on the opponent
  r0C43() { this.cycles += 10; for (let y = 2; y !== 0x3a; y += 2) { this.cycles += 4; this.click(); this.cycles += 2 + waitCycles(y) + 9; } },
  r0C46() { this.cycles += 10; for (let y = 2; y !== 0x24; y += 2) { this.cycles += 4; this.click(); this.cycles += 2 + waitCycles(y) + 9; } },
  // $0C49: the swish of a kick or a punch
  r0C49() { this.cycles += 10; for (let y = 10; y; y--) { this.cycles += 2 + waitCycles(5) + 4; this.click(); this.cycles += 5; } },
  // $0C4C: the beep that answers a control key
  r0C4C() { this.cycles += 10; for (let y = 16; y; y--) { this.cycles += 2 + waitCycles(10) + 4; this.click(); this.cycles += 5; } },
  // $0C4F / $0C52: footsteps
  r0C4F() { this.cycles += 10; for (let a = 0x10; ; ) { this.cycles += 4; this.click(); this.cycles += 2 + waitCycles(a) + 9; a >>= 1; if (a === 1) break; } },
  r0C52() { this.cycles += 10; for (let y = 3; y; y--) { this.cycles += 2 + waitCycles(0x10) + 4; this.click(); this.cycles += 5; } },
  // $0DC0: the recorded cry, stored one bit per sample: the speaker flips wherever the bit changes
  r0DC0() {
    const m = this.m;
    for (let x = 3; x; x--) m[0x9c + x] = m[0x0dfa + x];
    this.cycles += 49;
    for (;;) {
      let a = m[m[0x9c] | (m[0x9d] << 8)]; this.cycles += 10;
      for (let bit = 8; bit; bit--) {
        this.cycles += 3 + 5 * (m[0xba] || 256) - 1 + 3;
        if ((a ^ m[0x9b]) & 0x80) { m[0x9b] = a; this.cycles += 12; this.click(); } else this.cycles += 9;
        a = u8(a << 1); this.cycles += 7;
      }
      const more = (m[0x9c] | (m[0x9d] << 8)) < (m[0x9e] | (m[0x9f] << 8));
      m[0x9c] = u8(m[0x9c] + 1); if (m[0x9c] === 0) m[0x9d] = u8(m[0x9d] + 1);
      this.cycles += 22;
      if (!more) return;
    }
  },
});
