// Between fights: the next opponent sets out from the fortress and arrives.
import { extend, u8 } from './core.js';

extend({
  rB406() { const m = this.m; if (m[0x5f] !== 0) { m[0x5f]--; this.rAD0F(); } },   // repaint the backdrop on both pages
  // $B409: remember where everything was, for the wiping and repainting of the next frames
  rB409() {
    const m = this.m;
    m[0xac] = m[0xab]; m[0xab] = m[0x51]; m[0xae] = m[0xad]; m[0xad] = m[0x50];
    m[0xe1] = m[0xe0]; m[0xe0] = m[0x52]; m[0xe3] = m[0xe2]; m[0xe2] = m[0x62]; m[0xe5] = m[0xe4]; m[0xe4] = m[0x72];
    if (m[0x54] !== 0) m[0x54]--;
  },
  rB5E9() { this.m[0x54] = 2; },
  // $B5A3: a new opponent, off the screen to the right
  rB5A3() { const m = this.m; this.r7003(); m[0x30] = 1; m[0x72] = u8(m[0x52] + 0x14); m[0x7f] = 0xff; m[0x4b] = 1; },
  rB58A() { const m = this.m; m[0x75] = 0x43; m[0x76] = 0x2e; m[0x70] = 0x23; m[0x71] = 0x1e; m[0x77] = m[0x78] = m[0x79] = 1; },   // running
  rB569() {
    const m = this.m; let a = u8(this.a + m[0x50]);
    if (a < 7) { m[0x50] = a; return; }
    m[0x50] = a - 7; m[0x72]++; m[0x37]--;
    a = u8(m[0x51] + 1); if (a === 0x28) a = 0xf8; m[0x51] = a;
  },
  rB562() { this.a = 6; this.rB569(); this.a = 4; this.rB569(); },

  // $B400: cut to the fortress gate: the next opponent runs out towards the player
  rB400() {
    const m = this.m;
    m[0x31] = m[0x50]; m[0x36]++; m[0x82] = 0; m[0x50] = 0; m[0x10] = 0;
    const view = m[0x51]; m[0x85] = 0x0e; this.rB5A3();
    if (m[0x4c] === 0) {
      this.rB58A(); m[0x51] = 0x17; m[0x38] = m[0x52]; m[0x4c] = 1; m[0x52] = 0x1b; m[0x72] = 0x27; this.rB5E9();
      do {
        this.r7009(); this.r6540(); this.r6400(); this.r7006();
        this.rAD12(); this.r7009(); this.r6780(); this.r678C(); this.rAD0C(); this.rB0C3(); this.rAD03(); this.rBC00();
        this.r7009(); this.r0B00(); this.r0B06(); this.r6786(); this.r6783(); m[0x10] = 0; this.rBC03();
        this.r7600(); this.r7603(); this.rB409();
      } while (u8((m[0x57] !== 0 ? 5 : m[0x3e]) * 4 + 4) < m[0x72]);
      m[0x4c] = 0; m[0x52] = m[0x38]; this.rB5A3(); m[0x72] = u8(m[0x72] - 0x2c);
    }
    m[0x51] = view; m[0x50] = m[0x31]; m[0x36] = 0; this.rB5E9();
  },
  // $B403: cut to the opponent on his way, running along the fence
  rB403() {
    const m = this.m;
    m[0x31] = m[0x50]; m[0x8d] = m[0x50] & 1;
    const view = m[0x51]; m[0x37] = m[0x72]; m[0x8c] = m[0x72] & 1;
    m[0x51] = 0x0f + m[0x8d]; m[0x36]++; this.rB5E9();
    m[0x89] = m[0x85]; const next = u8(m[0x85] - 2); if (next >= 5) m[0x85] = next;
    m[0x72] = 0x18 + m[0x8c];
    for (;;) {
      this.r7009(); this.r6540(); this.r6400(); this.r7006(); this.rB562();
      this.rAD12(); this.rAD0C(); this.rB0C3(); this.r6780(); this.r678C(); this.rAD03();
      this.r7009(); this.r0B00(); this.r0B06(); this.r6786(); this.r6783();
      this.r7600(); this.r7603(); this.rB409();
      if (m[0x89] !== 0 && --m[0x89] !== 0) continue;
      if ((m[0x50] & 1) === m[0x8d] && (m[0x72] & 1) === m[0x8c]) break;
    }
    m[0x50] = m[0x31]; m[0x72] = u8((m[0x37] & 0xfe) + m[0x8c]); m[0x51] = view; m[0x36] = 0; this.rB5E9();
  },
});
