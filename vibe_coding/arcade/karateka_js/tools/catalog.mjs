// Development only: a catalogue of every picture the game draws in a scene: where the shape sits in
// memory, its size, how it is drawn. Written to re/catalog.json; `--sheet out.png` also paints the
// shapes of the given memory (the disk's or ours) side by side for a look.
// usage: node tools/catalog.mjs [--seeds N] [--frames N]
//        node tools/catalog.mjs --sheet out.png [--ours] [--from HEX --to HEX]
import fs from 'node:fs';
import { Game } from '../src/index.js';
import { Gfx } from '../src/gfx.js';
import { memoryFor, botPlay } from './lib.mjs';
const args = process.argv.slice(2), opt = (k, d) => { const i = args.indexOf('--' + k); return i < 0 ? d : args[i + 1]; };
const file = new URL('../re/catalog.json', import.meta.url);

if (!args.includes('--sheet')) {
  const cat = {};
  const header = Gfx.prototype.header;
  let ctx = '';
  Gfx.prototype.header = function () {
    const m = this.mem, p = m[3] | (m[4] << 8), e = cat[p] ??= { h: m[p], w: m[p + 1], n: 0, modes: {}, by: {}, at: [] };
    e.n++; e.modes[m[0x0f] === 0 ? 'or' : m[0x0f] & 0x80 ? 'and' : 'set'] = 1; e.by[ctx] = 1;
    if (e.at.length < 4) e.at.push([m[5], m[6], m[0x10]]);
    return header.call(this);
  };
  for (const k of Object.getOwnPropertyNames(Game.prototype)) if (/^r[0-9A-F]{4}$/.test(k)) {     // remember which routine asked for the shape
    const f = Game.prototype[k]; if (/^r(19|0A)/.test(k)) continue;
    Game.prototype[k] = function (...a) { const was = ctx; ctx = k; try { return f.apply(this, a); } finally { ctx = was; } };
  }
  for (let seed = 1; seed <= +opt('seeds', 6); seed++) for (const cheat of [true, false]) botPlay(new Game(memoryFor(false)), { seed, cheat, frames: +opt('frames', 6000), end: true });
  const out = Object.fromEntries(Object.entries(cat).sort((a, b) => a[0] - b[0]).map(([p, e]) => [(+p).toString(16), { ...e, modes: Object.keys(e.modes).join(','), by: Object.keys(e.by).join(',') }]));
  fs.writeFileSync(file, JSON.stringify(out, null, 0).replace(/\},"/g, '},\n"'));
  let bytes = 0; for (const e of Object.values(out)) bytes += e.h * e.w;
  console.log(Object.keys(out).length, 'shapes,', bytes, 'bytes of picture');
} else {
  const { writePNG } = await import('../../karateka/tools/emu/apple2.mjs');
  const { hgrToIndexed, PALETTE } = await import('../../karateka/src/vm/machine.js');
  const cat = JSON.parse(fs.readFileSync(file)), mem = memoryFor(args.includes('--ours'));
  const from = parseInt(opt('from', '0'), 16), to = parseInt(opt('to', 'ffff'), 16);
  const list = Object.entries(cat).map(([p, e]) => [parseInt(p, 16), e]).filter(([p]) => p >= from && p <= to);
  const pages = []; let page, g, x = 0, y = 0, rowH = 0; const where = [];
  const fresh = () => { page = new Uint8Array(0x10000); page.set(mem.subarray(0x800, 0xa00), 0x800); page.set(mem.subarray(0x1a34, 0x1a42), 0x1a34); page.fill(0x80, 0x2000, 0x4000); page[7] = 0x20; g = new Gfx(page); pages.push(page); x = 0; y = 0; rowH = 0; };
  fresh();
  for (const [p, e] of list) {
    if (e.w > 40 || e.h > 190 || !e.w || !e.h) continue;
    if (x + e.w > 40) { x = 0; y += rowH + 3; rowH = 0; }
    if (y + e.h > 190) fresh();
    page.set(mem.subarray(p, p + 2 + e.h * e.w), 0x6000); page[3] = 0; page[4] = 0x60; page[5] = x; page[6] = y; page[0x0f] = 1; page[0x10] = 0; g.blit();
    where.push(`${p.toString(16)} ${e.h}x${e.w} page ${pages.length} col ${x} row ${y} ${e.modes} ${e.by}`);
    x += e.w + 1; rowH = Math.max(rowH, e.h);
  }
  const rgb = new Uint8Array(280 * 192 * pages.length * 3), pal = [...PALETTE];
  pages.forEach((pg, k) => { const idx = hgrToIndexed(pg.subarray(0x2000, 0x4000)); for (let i = 0; i < 280 * 192; i++) rgb.set(pal[idx[i]], (k * 280 * 192 + i) * 3); });
  writePNG(opt('sheet'), 280, 192 * pages.length, rgb, +opt('scale', 3));
  console.log(where.join('\n'));
}
