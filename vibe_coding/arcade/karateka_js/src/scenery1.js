// The first scene: the cliff above the sea, the mountain behind, the fence along the road
// and the gateposts at its end. The scenery is redrawn only where the view has moved since
// the frame before last (the same page): $52/$50 are the view's column and fine position,
// $E1/$AE what they were then, $54 a countdown that forces a full repaint.
import { extend, u8 } from './core.js';

extend({
  viewMoved(col = 0x52, was = 0xe1) { const m = this.m; return m[0x54] !== 0 || m[col] !== m[was] || m[0x50] !== m[0xae]; },
  fillRows(y, y2, alt) { const m = this.m; m[6] = y; m[8] = y2; if (alt) this.r0A03(); else this.r0A00(); },

  // $AC30: the player falling from the cliff, two pictures further on every frame
  rAC30() {
    const m = this.m; let x = m[0xa1]; if (x === 0) return;
    x++; if (x < 0x23) { this.x = x; this.rAC52(); x++; this.x = x; this.rAC52(); }
    m[0xa1] = x; this.x = x;
  },
  rAC52() {
    const m = this.m, x = this.x;
    m[3] = m[0xac13 + x]; m[4] = 0x1e; m[6] = m[0xac1b + x]; m[5] = u8(m[0xac17 + x] + m[0x52] - 0x30); m[0x0f] = 1;
    let s = u8(m[0x50] - 1); if (s & 0x80) { m[5] = u8(m[5] - 1); s = 6; }
    m[0x10] = s; this.r1903();
  },

  // $AD15: get ready to draw the horizon line
  rAD15() { const m = this.m; m[0x10] = 0; m[6] = 0x68; m[3] = 0x11; m[4] = 0xaa; m[0x0f] = 1; },
  // $AD18 / $AD1F: the horizon across the screen / from the current column to the right edge
  rAD18() { this.m[5] = 0; this.rAD15(); this.rAD1F(); },
  rAD1F() { const m = this.m; for (;;) { this.r1903(); if (m[5] === 0x26) return; m[5] = u8(m[5] + 2); } },
  // the mountain
  mountain() { const m = this.m; m[3] = 0x48; m[4] = 0xa9; m[5] = 0x12; m[6] = 0x51; m[0x0f] = 1; this.r1903(); m[3] = 0x76; m[5] = 0x10; m[6] = 0x5c; this.r1903(); },
  // $AD0F: the whole backdrop: sky, sea, the ground, the horizon and the mountain
  rAD0F() {
    const m = this.m;
    m[5] = 0; m[9] = 0x28; m[2] = 0xd5; m[0x11] = 0xaa; this.fillRows(0, 0x68);
    m[0x13] = 0xd5; m[2] = 0xd5; m[0x12] = 0xaa; m[0x11] = 0xaa; this.fillRows(0x99, 0xb5, true);
    m[2] = 0x80; m[0x11] = 0x80; this.fillRows(0x6f, 0x99); this.fillRows(0xb5, 0xc0);
    this.rAD18(); this.mountain();
  },
  // $AD12 / $AD03: the fence posts: wipe them where they were, draw them where they are
  rAD12() { const m = this.m; if (!this.viewMoved(0x51, 0xac)) return; m[3] = 0x31; this.posts(m[0xac], m[0xae]); },
  rAD03() {
    const m = this.m; if (!this.viewMoved(0x51, 0xac)) return;
    m[3] = 0xb8; m[4] = 0xa9; m[5] = 0x0f; m[6] = 0x64; m[0x10] = 0; this.r1903(); m[3] = 0xe2; m[5] = 0x0c; m[6] = 0x6c; this.r1903();   // the tree
    m[3] = 0x23; this.posts(m[0x51], m[0x50]);
  },
  posts(x, y) {
    const m = this.m;
    m[4] = 0xaa; m[6] = 0x64;
    let s = y + 1; if (s >= 7) { x = u8(x + 1); s = 0; }
    m[0x10] = s; m[0x0f] = 1;
    let a = u8(x + 1);
    for (let n = 4; ; ) {
      m[5] = a;
      if (a >= 0x15 && a < 0x95) this.r1909(); else this.r1903();
      if (--n === 0) return;
      a = u8(m[5] + 0x0c); if (a >= 0x28) a = u8(a - 0x30);
    }
  },

  // $AE3F: a piece of the gateposts; $AE65 draws it, mirrored once the view has passed the gate
  rAE3F() {
    const m = this.m, x = this.x, v = m[0x52], right = !(v & 0x80) && v >= 0x14;
    m[3] = m[0xadf7 + x]; m[4] = m[0xae09 + x]; m[6] = m[0xae2d + x];
    m[5] = right ? u8(v + m[0xae1b + x]) : u8(v - m[0xae1b + x]);
    this.rAE65();
  },
  rAE65() { const m = this.m, v = m[0x52]; if (!(v & 0x80) && v >= 0x14) this.r1903(); else this.r190C(); },
  // $AD06: the gateposts, behind the fighters
  rAD06() {
    const m = this.m, piece = (x) => { this.x = x; this.rAE3F(); }, column = (to) => { do { m[6] = u8(m[6] + 2); this.rAE65(); } while (m[6] < to); };
    if (m[0x4c] !== 0) return;
    const v = m[0x52]; let x;
    if (v >= 0xfe || v < 0x14) x = m[0x72] < 0x13 || m[0x62] < 0x10 ? 1 : 0;
    else { if (v >= 0x29) return; x = (m[0x72] < 0x2c && m[0x72] >= 0x16) || m[0x62] >= 0x13 ? 1 : 0; }
    m[0x10] = m[0x50]; m[0x0f] = 1;
    if (!this.viewMoved()) { m[0x3f] = x; if (x === 0) return; }
    else { m[0x3f] = 1; for (const k of [5, 8, 9, 0x0a, 0x0b, 0x0c, 0x0d]) piece(k); column(0x66); piece(0x0e); piece(0x0f); }
    piece(1); column(0x97); for (const k of [2, 3, 4, 6, 7]) piece(k); column(0x66); piece(0x11);
  },
  // $AD09: the post that stands in front of the fighters
  rAD09() {
    const m = this.m; if (m[0x4c] !== 0 || m[0x3f] === 0) return;
    const v = m[0x52]; if (v < 0xf4 && v >= 0x29) return;
    m[0x10] = m[0x50]; this.x = 0; this.rAE3F();
    do { m[6] = u8(m[6] + 2); this.rAE65(); } while (m[6] < 0xaa);
  },
  // $AD0C: repaint the part of the backdrop the gateposts stood on
  rAD0C() {
    const m = this.m; if (!this.viewMoved()) return;
    const was = m[0xe1];
    if (was >= 0xfe || was < 0x14) {
      m[5] = 0; this.rAD15(); do { this.r1903(); m[5] = u8(m[5] + 2); } while (m[5] < 0x0c);
      m[5] = 0; m[9] = 0x0b;
    } else {
      if (was >= 0x28) return;
      m[2] = 0xaa; m[5] = 0x1b; m[9] = 0x1c; this.fillRows(0x54, 0x56);
      this.rAD15(); m[5] = 0x1c; this.rAD1F(); m[5] = 0x1c; m[9] = 0x28;
    }
    m[2] = 0xd5; m[0x11] = 0xaa; this.fillRows(0x1e, 0x68);
    m[0x13] = 0xd5; m[0x12] = 0xaa; this.fillRows(0x99, 0xb2, true);
    m[2] = 0x80; m[0x11] = 0x80; this.fillRows(0x70, 0x99);
  },

  // $B1AE: a piece of the cliff
  rB1AE() { const m = this.m, x = this.x; m[3] = m[0xb19a + x]; m[4] = m[0xb19f + x]; m[6] = m[0xb1a9 + x]; m[5] = u8(m[0xb1a4 + x] + m[0x80]); },
  // $B0C3: repaint the backdrop where the cliff's edge was
  rB0C3() {
    const m = this.m; if (m[0x44] !== 0 || !this.viewMoved()) return;
    if (m[0xe1] < 0x23) return;
    this.rAD15(); m[5] = 0; do { this.r1903(); m[5] = u8(m[5] + 2); } while (m[5] < 0x0c);
    if (m[0x54] !== 0) { m[5] = 0; m[9] = 0x0c; }
    else { let a = u8(m[0xe1] - 0x27) & 0xfe; if (a & 0x80) a = 0; m[5] = a; m[9] = a + 4; }
    m[2] = 0x80; m[0x11] = 0x80; this.fillRows(0x70, 0x99);
    m[2] = 0xd5; m[0x13] = 0xd5; m[0x11] = 0xaa; m[0x12] = 0xaa; this.fillRows(0x99, 0xb5, true);
  },
  // $B0C0: the cliff's edge at the left of the scene, and the sea beyond it
  rB0C0() {
    const m = this.m, piece = (x) => { this.x = x; this.rB1AE(); this.r1903(); };
    if (m[0x44] !== 0 || m[0x52] < 0x23) return;
    if (m[0x50] === m[0xae] && m[0x52] === m[0xe1] && m[0x62] >= 0x10 && m[0x72] >= 0x13 && m[0x54] === 0) return;
    m[0x80] = 0; m[0x10] = 0; m[0x0f] = 1; piece(1);
    m[0x80] = u8(m[0x52] - 0x30);
    let s = u8(m[0x50] - 1); if (s & 0x80) { m[0x80] = u8(m[0x80] - 1); s = 6; }
    m[0x10] = s; m[5] = 0;
    let w = u8(0x0b + m[0x80]);
    if (w !== 0 && !(w & 0x80)) {
      m[9] = w; m[0x12] = 0xd5; m[0x13] = 0xaa; m[2] = 0x80; m[0x11] = 0x80; this.fillRows(0x74, 0x98, true);
      w = u8(7 + m[0x80]); if (w !== 0 && !(w & 0x80)) { m[9] = w; this.fillRows(0x9a, 0xb5, true); }
      m[9] = u8(0x0b + m[0x80]); m[2] = 0xd5; m[0x11] = 0xaa; this.fillRows(0x68, 0x71);
    }
    piece(0); piece(2); piece(3);
    do { m[6] = u8(m[6] + 2); this.r1903(); } while (m[6] < 0x97);
    m[0x0f] = 0xff; piece(4);
  },
});
