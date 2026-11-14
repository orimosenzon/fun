// Development only: a sheet of fighter poses, drawn by the game's own routine ($6783) from the
// reference memory or from ours, to compare the two by eye.
// usage: node tools/poses.mjs out.png [--ours] [--pairs]  [--from N --to N] [--left] [--who N]
//   default: pose n for both the legs and the body, n = 0..37 (legs up to 41)
//   --pairs: the combinations of legs and body that occur in play (collected once into re/pairs.json)
import fs from 'node:fs';
import { Game } from '../src/index.js';
import { memoryFor, botPlay } from './lib.mjs';
import { writePNG } from '../../karateka/tools/emu/apple2.mjs';
import { hgrToIndexed, PALETTE } from '../../karateka/src/vm/machine.js';
const args = process.argv.slice(2), opt = (k, d) => { const i = args.indexOf('--' + k); return i < 0 ? d : args[i + 1]; };
const pairsFile = new URL('../re/pairs.json', import.meta.url);

export function collectPairs() {
  if (fs.existsSync(pairsFile)) return JSON.parse(fs.readFileSync(pairsFile));
  const seen = {}, f = Game.prototype.r6783;
  Game.prototype.r6783 = function () { const m = this.m, k = m[0x20] + ',' + m[0x21]; seen[k] = (seen[k] || 0) + 1; return f.call(this); };
  for (let seed = 1; seed <= 8; seed++) for (const cheat of [true, false]) botPlay(new Game(memoryFor(false)), { seed, cheat, frames: 6000 });
  Game.prototype.r6783 = f;
  const list = Object.entries(seen).map(([k, n]) => [...k.split(',').map(Number), n]).sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  fs.writeFileSync(pairsFile, JSON.stringify(list));
  return list;
}

export function sheet(ours, list, { left = false, who = 0, per = 4 } = {}) {
  const T = 0x60, B = 0xa6, bands = [];
  for (let i = 0; i < list.length; i += per) {
    const g = new Game(memoryFor(ours)), m = g.m;
    m[7] = 0x20; m.fill(0x80, 0x2000, 0x4000);
    list.slice(i, i + per).forEach(([legs, body], k) => {
      m[0x20] = legs; m[0x21] = body; m[0x22] = (left ? 8 : 2) + k * 10; m[0x2c] = left ? 0xff : 0; m[0x50] = 0; m[0x36] = 0; m[0x3e] = who; m[0x4d] = 0;
      g.r6783();
    });
    const idx = hgrToIndexed(m.subarray(0x2000, 0x4000));
    for (let x = 0; x < 280; x++) { if (!idx[0xa0 * 280 + x]) idx[0xa0 * 280 + x] = 6; }
    for (let k = 1; k < per; k++) for (let y = T; y < B; y += 2) idx[y * 280 + k * 70 - 1] = 6;
    bands.push(idx.subarray(T * 280, B * 280));
  }
  return { bands, h: B - T };
}

if (import.meta.url === `file://${process.argv[1]}`) {
  const out = args[0];
  let list = args.includes('--pairs') ? collectPairs().map(([a, b]) => [a, b]) : Array.from({ length: 42 }, (_, n) => [n, Math.min(n, 37)]);
  if (opt('list', '')) list = opt('list').split(' ').map((s) => s.split('/').map(Number));
  list = list.slice(+opt('from', 0), +opt('to', 999));
  const both = args.includes('--both');
  const sets = both ? [sheet(false, list, { left: args.includes('--left'), who: +opt('who', 0), per: 2 }), sheet(true, list, { left: args.includes('--left'), who: +opt('who', 0), per: 2 })] : [sheet(args.includes('--ours'), list, { left: args.includes('--left'), who: +opt('who', 0) })];
  const h = sets[0].h, n = sets[0].bands.length, pal = [...PALETTE, [90, 90, 90]];
  const rgb = new Uint8Array(280 * h * n * 3);
  for (let k = 0; k < n; k++) for (let i = 0; i < 280 * h; i++) {
    let v = sets[0].bands[k][i];
    if (both) { const x = i % 280; v = x < 140 ? sets[0].bands[k][i] : sets[1].bands[k][i - 140]; }
    rgb.set(pal[v], (k * 280 * h + i) * 3);
  }
  writePNG(out, 280, h * n, rgb, +opt('scale', 3));
  console.log(list.map((p, i) => (i % 4 === 0 ? '\n' : '') + p.join('/')).join('  '));
}
