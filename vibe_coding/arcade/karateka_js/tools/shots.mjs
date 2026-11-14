// Development only: screenshots of a bot's game, from the reference memory or from ours.
// usage: node tools/shots.mjs out.png [--ours] [--both] [--seed N] [--cheat] --at "10 200 900"   (pass numbers of the main loop)
//        ... --flips "A B" instead: the A-th to B-th picture shown on the screen (cut scenes show many per pass)
import fs from 'node:fs';
import { Game } from '../src/index.js';
import { memoryFor, botPlay } from './lib.mjs';
import { writePNG } from '../../karateka/tools/emu/apple2.mjs';
import { hgrToIndexed, PALETTE } from '../../karateka/src/vm/machine.js';
const args = process.argv.slice(2), opt = (k, d) => { const i = args.indexOf('--' + k); return i < 0 ? d : args[i + 1]; };
const at = opt('at', '').split(' ').filter(Boolean).map(Number), fl = opt('flips', '').split(' ').filter(Boolean).map(Number);
const step = +opt('every', 1);
// --log: also print what was drawn in each pass that is captured (shape address, size, column, row, shift, mode)
import { Gfx } from '../src/gfx.js';
let log = [];
if (args.includes('--log')) {
  const h = (v) => v.toString(16);
  for (const k of ['blit', 'blitRight', 'blitMirror', 'blitMirrorRight']) { const f = Gfx.prototype[k]; Gfx.prototype[k] = function () { const m = this.mem, p = m[3] | (m[4] << 8); log.push(h(p) + ' ' + m[p] + 'x' + m[p + 1] + ' ' + k + ' col ' + (m[5] << 24 >> 24) + ' row ' + m[6] + ' shift ' + m[0x10] + ' ' + (m[0x0f] === 0 ? 'or' : m[0x0f] & 0x80 ? 'and' : 'set') + ' page ' + h(m[7])); return f.call(this); }; }
  const f = Gfx.prototype.fill; Gfx.prototype.fill = function (alt) { const m = this.mem; log.push('fill cols ' + m[5] + '-' + m[9] + ' rows ' + m[6] + '-' + m[8] + ' pattern ' + h(m[2]) + ' ' + h(m[0x11]) + (alt ? ' / ' + h(m[0x12]) + ' ' + h(m[0x13]) : '') + ' page ' + h(m[7])); return f.call(this, alt); };
}
function play(ours) {
  const g = new Game(memoryFor(ours)), shots = []; let count = 0;
  botPlay(g, { seed: +opt('seed', 2), cheat: args.includes('--cheat'), frames: Math.max(0, ...at) + 1 || 9000, each: (g, n) => {
    for (const f of g.flips) { if (fl.length && count >= fl[0] && count <= fl[1] && (count - fl[0]) % step === 0) shots.push(f.page ? f.page.slice() : g.m.slice(f.show << 8, (f.show << 8) + 0x2000)); count++; }
    if (at.includes(n) && log.length) console.log('--- pass ' + n + '\n' + log.join('\n'));
    log = [];
    if (at.includes(n)) { const f = g.flips.at(-1), base = (f ? f.show : 0x20) << 8; shots.push(g.m.slice(base, base + 0x2000)); }
  } });
  return shots;
}
const sets = args.includes('--both') ? [play(false), play(true)] : [play(args.includes('--ours'))];
const n = Math.min(...sets.map((s) => s.length)), cols = sets.length, rgb = new Uint8Array(280 * cols * 192 * n * 3);
for (let k = 0; k < n; k++) sets.forEach((s, c) => { const idx = hgrToIndexed(s[k]); for (let y = 0; y < 192; y++) for (let x = 0; x < 280; x++) rgb.set(PALETTE[idx[y * 280 + x]], ((k * 192 + y) * 280 * cols + c * 280 + x) * 3); });
writePNG(args[0], 280 * cols, 192 * n, rgb, +opt('scale', 2));
console.log(n, 'pictures');
