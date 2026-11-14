// Drawing a fighter. He is put together from separate shapes, looked up by pose:
// the legs, the body, the head, and where needed a mask that first clears the place
// behind them. The player is drawn as stored; the opponent faces left and is drawn mirrored.
// Tables (one entry per pose): $6000/$602A legs shape, $6054 column of the legs, $607E their row,
// $60A8/$60D2/$60FC/$6126 the shadow, $6150/$617A column and bit shift of the body,
// $61A4 which head, $61DA/$6200 column and row of the head, $6226... the body shape.
import { extend, u8 } from './core.js';

extend({
  // $693F: the row for the next shape, counted from the top of the fighters' strip
  r693F() { this.a = this.m[6] = u8(this.a + 0x72); },
  // $6945: the fighter's shadow on the floor
  r6945() { const m = this.m, x = this.x; m[0x0f] = 0xff; m[3] = m[0x60a8 + x]; m[4] = m[0x60d2 + x]; this.a = m[0x6126 + x]; this.r693F(); },
  // $6959: the legs
  r6959() { const m = this.m, x = this.x; m[0x0f] = 0; m[3] = m[0x6000 + x]; m[4] = m[0x602a + x]; m[5] = m[0x23]; this.a = m[0x607e + x]; this.r693F(); },
  // $6971 / $697C: draw facing right / facing left; a few poses are stored the other way round
  r6971() { if (this.m[0x62be + this.x]) this.r1909(); else this.r1903(); },
  r697C() { if (this.m[0x62be + this.x]) this.r1906(); else this.r190C(); },
  // $6987: which head. The opponents each have their own set of heads ($3E = which opponent)
  r6987() {
    const m = this.m; this.x = m[0x21];
    const k = m[0x61a4 + this.x]; this.y = k >= 7 ? k - 6 : 0;
    if (m[0x36] === 0 && !(m[0x2c] & 0x80)) { m[3] = m[0x61cd + this.y]; this.a = m[0x61ca + this.y]; if (this.a) { m[4] = this.a; return; } }
    m[6] = u8(m[6] - 2); m[5] = u8(m[5] - m[0x2c]);
    this.r6A6C(); if (this.a) { m[4] = this.a; return; }
    this.r69BD();
  },
  // $69BD: the body's upper part, and for the plain stance the fine position of the head
  r69BD() {
    const m = this.m; this.x = m[0x21];
    const k = m[0x61a4 + this.x];
    if (k >= 7) this.y = k - 6; else { this.a = k; this.r6AC5(); this.y = 0; }
    m[3] = m[0x61d9 + this.y]; this.a = m[4] = m[0x61d6 + this.y];
  },
  // $6AC5: add A pixels to the shift, carrying into the column
  r6AC5() {
    const m = this.m; let a = this.a;
    if (!(m[0x2c] & 0x80)) { a = u8(a + m[0x10]); if (a >= 7) { a -= 7; m[0x23]++; } }
    else { a = u8(m[0x10] - a); if (a & 0x80) { a = u8(a + 7); m[0x23]--; } }
    m[0x10] = a; this.a = a;
  },
  // $6A6C: the opponent's head, by which opponent this is
  r6A6C() {
    const m = this.m, y = this.y, T = [[0x61d3, 0x61d0], [0x62eb, 0x62e8], [0x62f1, 0x62ee], [0x62f7, 0x62f4], [0x62fd, 0x62fa], [0x6aed, 0x6aea], [0x6af3, 0x6af0]];
    const [lo, hi] = T[m[0x3e] >= 1 && m[0x3e] <= 6 ? m[0x3e] : 0];
    m[3] = m[lo + y]; this.a = m[hi + y];
  },
  // $69DC: the mask that clears the place behind the arms, for the poses that need one
  r69DC() {
    const m = this.m, x = this.x = m[0x21];
    if (x === 6 || (x >= 0x14 && x < 0x1e)) { m[3] = 0x2a; this.a = 0x0f; this.r693F(); m[4] = 0x9a; m[0x0f] = 0xff; m[5] = u8(m[0x2c] + m[0x80]); this.x = 0x64; }
    else if (x === 7 || x === 0x27 || (x >= 0x0a && x < 0x0e)) { m[3] = 0x18; this.a = 0x16; this.r693F(); m[4] = 0x9a; m[0x0f] = 0xff; m[5] = m[0x80]; this.x = 0x64; }
  },
  // $6786 / $6789: the shadows of the opponent and of the player
  r6786() { const m = this.m; m[0x10] = m[0x50]; this.x = m[0x70]; m[0x23] = u8(m[0x72] - m[0x6054 + this.x]); this.r6945(); if (m[4] === 0) return; m[5] = u8(m[0x23] - m[0x60fc + this.x]); this.r697C(); },
  r6789() { const m = this.m; this.x = m[0x60]; m[0x23] = u8(m[0x62] + m[0x6054 + this.x]); this.r6945(); if (m[4] === 0) return; m[5] = u8(m[0x23] + m[0x60fc + this.x]); this.r6971(); },

  // $6783: draw the fighter in the working copy
  r6783() {
    const m = this.m; this.x = m[0x20]; m[0x10] = m[0x50];
    const left = !!(m[0x2c] & 0x80), s = left ? -1 : 1;
    if (left && m[0x4d] !== 0 && m[0x4d] !== 5) return this.rBC00();
    const draw = () => (left ? this.r190C() : this.r1903());
    let x = this.x;
    m[0x23] = u8(m[0x22] + s * m[0x6054 + x]);
    m[0x80] = u8(m[0x22] + s * m[0x6150 + x]);
    let sh = u8(m[0x50] + s * m[0x617a + x]);
    if (!left) { if (sh >= 7) { sh -= 7; m[0x80]++; } } else if (sh & 0x80) { sh = u8(sh + 7); m[0x80]--; }
    m[0x81] = sh; m[0x10] = sh;
    this.r69DC(); if (this.x === 0x64) draw();
    this.x = m[0x20]; m[0x10] = m[0x50];
    this.r6959(); if (left) this.r697C(); else this.r6971();
    m[0x23] = m[0x80]; m[0x10] = m[0x81];
    this.r69BD(); x = this.x;
    const col = m[5] = u8(m[0x23] + s * m[0x61da + x]);
    this.a = m[0x6200 + x]; this.r693F();
    if (m[4] !== 0) { m[0x0f] = 0xff; draw(); }
    if (left) m[5] = col;
    m[0x0f] = 0; this.r6987(); draw(); x = this.x;
    m[0x23] = m[0x80]; m[0x10] = m[0x81];
    m[3] = m[0x6226 + x]; m[4] = m[0x624c + x];
    m[5] = u8(m[0x23] + s * m[0x6272 + x]);
    this.a = m[0x6298 + x]; this.r693F(); draw();
  },

  // $6780 / $678C: wipe the strip where a fighter stood ($E5 the opponent's last column, $E3 the player's)
  r6780() { this.wipe(u8(this.m[0xe5] - 6)); },
  r678C() { this.wipe(u8(this.m[0xe3] - 3)); },
  wipe(a) {
    const m = this.m;
    if (a & 0x80) a = 0; else if (a >= 0x28) return;
    m[5] = a; a += 0x0b; if (a >= 0x29) a = 0x28;
    m[9] = a; m[6] = 0x72; m[8] = 0x99; m[2] = 0x80; m[0x11] = 0x80; this.r0A00();
    m[5] &= 0xfe; m[6] = 0x99; m[8] = 0xa4; m[0x11] = 0xaa; m[0x12] = 0xaa; m[2] = 0xd5; m[0x13] = 0xd5;
    if (m[0x44] === 6) { m[2] = 0x80; m[0x11] = 0x80; return this.r0A00(); }
    if (m[0x44] >= 2 || m[0xd0] >= 3) { m[0x12] = 0x80; m[0x13] = 0x80; }
    else if (m[0xd0] === 2) { m[2] = 0x80; m[0x11] = 0x80; }
    this.r0A03();
  },
});
