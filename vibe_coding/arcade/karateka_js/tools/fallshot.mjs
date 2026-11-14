// Development only: the hero walks back off the cliff; pictures of the fall and of the closing words.
// usage: node tools/fallshot.mjs out.png [--ours]
import { Game } from '../src/index.js';
import { memoryFor } from './lib.mjs';
import { writePNG } from '../../karateka/tools/emu/apple2.mjs';
import { hgrToIndexed, PALETTE } from '../../karateka/src/vm/machine.js';
const sets = (process.argv.includes('--both') ? [false, true] : [process.argv.includes('--ours')]).map((ours) => {
  const g = new Game(memoryFor(ours)), shots = []; g.rB700(); let res = 'frame';
  const snap = () => { const f = g.flips.at(-1); shots.push(f.page ? f.page.slice() : g.m.slice(f.show << 8, (f.show << 8) + 0x2000)); };
  for (let n = 0; n < 400 && res === 'frame'; n++) { if (n === 30) g.key = 0xa0; else if (n > 34 && n % 3 === 0) g.key = 0x88; res = g.step(); if (g.m[0xa1]) snap(); g.flips.length = 0; }
  console.log('ended with', res, shots.length, 'pictures of the fall');
  g.gameOver(); snap();
  return [shots[0], shots[1], shots[3], shots.at(-1)].filter(Boolean);
});
const n = sets[0].length, cols = sets.length, rgb = new Uint8Array(280 * cols * 192 * n * 3);
for (let k = 0; k < n; k++) sets.forEach((s, c) => { const idx = hgrToIndexed(s[k]); for (let y = 0; y < 192; y++) for (let x = 0; x < 280; x++) rgb.set(PALETTE[idx[y * 280 + x]], ((k * 192 + y) * 280 * cols + c * 280 + x) * 3); });
writePNG(process.argv[2], 280 * cols, 192 * n, rgb, 2);
