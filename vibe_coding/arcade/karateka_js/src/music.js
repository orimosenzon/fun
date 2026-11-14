// Music. A tune is a list of notes; each note is played by one tight loop that runs two
// counters side by side and flips the speaker whenever either runs out: two voices from a
// one-bit speaker, the second a shade off the first, which gives the tunes their shimmer.
// The loop is reproduced here pass by pass, with the time each pass took on the original.
import { extend, u8 } from './core.js';

extend({
  // $1000-$100C: the five tunes of this part of the game
  r1000() { this.tune(0x10e4); }, r1003() { this.tune(0x108b); }, r1006() { this.tune(0x10bf); }, r1009() { this.tune(0x1040); }, r100C() { this.tune(0x10f4); },
  tune(at) {
    const m = this.m; m[0xf7] = at & 0xff; m[0xf8] = at >> 8; this.a = m[0xf7]; this.x = m[0xf8];
    this.cycles += 27;
    if (m[0x4f] & m[0x86]) { const from = this.cycles; this.r0D00(); this.deaf.push([from, this.cycles]); }
  },
  // $0D00: play the tune at ($F7). While it plays the game does nothing else, and keys are swallowed
  r0D00() {
    const m = this.m, at = m[0xf7] | (m[0xf8] << 8);
    m[0x0d5e] = m[at]; m[0x0d88] = m[at + 1]; m[0xf6] = 2; this.cycles += 27;
    const far = (y) => ((at & 0xff) + y > 0xff ? 1 : 0);   // reading across a page boundary costs the processor a cycle
    for (;;) {
      this.cycles += 7;
      if (this.key & 0x80) { this.r7603(); this.strobe(); this.cycles += 20; if (!(m[0x86] & m[0x4f])) return; }
      let y = m[0xf6]; const a = m[at + y]; this.cycles += 12 + far(y);
      if (a === 0xff) { this.cycles += 7; return; }
      if (a === 0xfe) { m[0x0d79] = 0xea; m[0x0d7a] = 0xea; m[0xf6] = u8(m[0xf6] + 1); this.cycles += 25; continue; }        // the second voice runs free
      if (a === 0xfd) { m[0x0d79] = 0xa0; m[0x0d7a] = m[at + y + 1]; m[0xf6] = u8(m[0xf6] + 2); this.cycles += 42 + far(y + 1); continue; } // ... or restarts with the first
      m[0xfd] = a; const len = m[at + y + 1], pitch = m[at + y + 2];
      m[0xfa] = m[0xfb] = m[0xf9] = len; m[0xfc] = pitch;
      this.cycles += 45 + far(y + 1) + far(y + 2);
      let x = pitch; y = pitch;
      const silent = !!(len & 0x80), second = m[0x0d88] === 0xc0, sync = m[0x0d79] === 0xa0;
      do {
        m[0xfe] = m[0x0d5e]; this.cycles += 5;
        do {
          m[0xff] = 0xff; this.cycles += 5;
          do {
            x = u8(x - 1);
            if (x !== 0) this.cycles += 5;
            else if (--m[0xfa] !== 0) this.cycles += 12;
            else {
              this.cycles += 14; if (silent) this.cycles += 3; else { this.cycles += 6; this.click(); }
              x = m[0xfc]; m[0xfa] = m[0xf9]; this.cycles += 9;
              if (sync) { y = m[0x0d7a]; this.cycles += 2; } else this.cycles += 4;
            }
            for (;;) {
              y = u8(y - 1);
              if (y !== 0) { this.cycles += 5; break; }
              if (--m[0xfb] !== 0) { this.cycles += 12; continue; }
              this.cycles += 14; if (silent) this.cycles += 3; else { this.cycles += 6; if (second) this.click(); }
              y = u8(m[0xfc] - 1); m[0xfb] = m[0xf9]; this.cycles += 11;
              break;
            }
            m[0xff]--; this.cycles += m[0xff] ? 8 : 7;
          } while (m[0xff] !== 0);
          m[0xfe]--; this.cycles += m[0xfe] ? 8 : 7;
        } while (m[0xfe] !== 0);
        m[0xfd]--; this.cycles += m[0xfd] ? 8 : 7;
      } while (m[0xfd] !== 0);
      m[0xf6] = u8(m[0xf6] + 3); this.cycles += 18;
    }
  },
});
