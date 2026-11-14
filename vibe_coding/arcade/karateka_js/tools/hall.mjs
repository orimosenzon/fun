// Development only: pictures from inside one pass of the main loop (the hall scene shows many).
// usage: node tools/hall.mjs out.png [--ours|--both] [--pass 499] [--pick "40 60 90"] [--scale 2]
import { Game } from '../src/index.js';
import { memoryFor, botPlay } from './lib.mjs';
import { writePNG } from '../../karateka/tools/emu/apple2.mjs';
import { hgrToIndexed, PALETTE } from '../../karateka/src/vm/machine.js';
const args = process.argv.slice(2), opt = (k, d) => { const i = args.indexOf('--' + k); return i < 0 ? d : args[i + 1]; };
const pass = +opt('pass', 499), pick = opt('pick', '').split(' ').filter(Boolean).map(Number);
function play(ours) {
  const g = new Game(memoryFor(ours)); let shots = [];
  botPlay(g, { seed: +opt('seed', 2), cheat: true, frames: pass + 1, each: (g, n) => { if (n === pass) shots = g.flips.map((f) => f.page ? f.page.slice() : g.m.slice(f.show << 8, (f.show << 8) + 0x2000)); } });
  console.log(shots.length, 'pictures in the pass');
  return pick.length ? pick.map((i) => shots[Math.min(i, shots.length - 1)]) : shots.filter((_, i) => i % Math.ceil(shots.length / 6) === 0);
}
const sets = args.includes('--both') ? [play(false), play(true)] : [play(args.includes('--ours'))];
const n = sets[0].length, cols = sets.length, rgb = new Uint8Array(280 * cols * 192 * n * 3);
for (let k = 0; k < n; k++) sets.forEach((s, c) => { const idx = hgrToIndexed(s[k]); for (let y = 0; y < 192; y++) for (let x = 0; x < 280; x++) rgb.set(PALETTE[idx[y * 280 + x]], ((k * 192 + y) * 280 * cols + c * 280 + x) * 3); });
writePNG(args[0], 280 * cols, 192 * n, rgb, +opt('scale', 2));
