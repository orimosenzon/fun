// The drawing primitives, as the game's routines call them. The work is done in gfx.js;
// parameters are where the original kept them: $03/$04 the shape, $05 column, $06 row,
// $07 page, $0F mode (0 = OR, negative = AND, positive = replace), $10 shift within the byte.
//
// The original ran as fast as its processor went, so how long a frame lasted depended on how
// much was drawn in it. Each call adds an estimate of the cycles the original spent on it
// (data/timing.js); the sounds are timed exactly in sound.js and music.js.
import { extend } from './core.js';
import { SHAPE, FILL, CLEAR, OWN_FIGHTER } from './data/timing.js';

extend({
  shape(mirrored, right, draw) {
    const m = this.m, p = m[3] | (m[4] << 8), wide = m[p + 1];
    let x = m[5]; if (right) x = (x - wide + 1) & 0xff;
    const c = SHAPE[mirrored][m[0x0f] === 0 ? 0 : m[0x0f] & 0x80 ? 1 : 2], shift = m[0x10];
    draw();
    if (x < 0x80 ? x >= 40 : x + wide <= 256) { this.cycles += c[5]; return; }
    const rows = m[0x0d], bytes = rows * m[0x0e];
    this.cycles += (c[0] + c[1] * rows + c[2] * bytes + (shift ? (c[3] + c[4] * shift) * bytes : 0)) * (p >= 0xc000 ? OWN_FIGHTER : 1);
  },
  box(alt) {
    const m = this.m, c = FILL[alt ? 1 : 0];
    this.g.fill(alt);
    const rows = (m[0x0d] + 1) & 0xff, cols = (m[0x0e] + 1) & 0xff;
    this.cycles += c[0] + c[1] * rows + c[2] * rows * cols;
  },
  r1900() { this.cycles += CLEAR; this.g.clear(); },                          // clear the page
  r1903() { this.shape(0, false, () => this.g.blit()); },                     // shape, left edge at the column
  r1906() { this.shape(0, true, () => this.g.blitRight()); },                 // shape, right edge at the column
  r1909() { this.shape(1, false, () => this.g.blitMirror()); },               // mirrored, left edge at the column
  r190C() { this.shape(1, true, () => this.g.blitMirrorRight()); },           // mirrored, right edge at the column
  r0A00() { this.box(false); },                                               // fill a rectangle with a pattern
  r0A03() { this.box(true); },
});
