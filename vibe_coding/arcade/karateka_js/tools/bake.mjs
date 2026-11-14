// Works out how a scene's memory is put together from the disk, by looking at the original at
// the moment the scene starts: which disk sector each page of memory is a copy of, the few bytes
// that differ, the variables in the first two pages, and which bytes are the original's program
// (those are wiped after loading: the rewritten game must never need them).
// Writes tools/ref/<name>.map.js, which holds numbers only: no byte of the disk is in this project.
// usage: node tools/bake.mjs scene1
import fs from 'node:fs';
import { boot, DISK } from './oracle.mjs';
import { OPTABLE, MODES as M } from '../../karateka/src/vm/cpu6502.js';
const name = process.argv[2] || 'scene1';
const dsk = fs.readFileSync(DISK);
let ram, listing;
if (name === 'scene1') {
  const m = boot(), c = m.cpu; m.sec(20); m.press('SPACE');
  while (!(m.inGame && c.pc === 0xb700 && m.ram[0xb700] === 0x20 && m.ram[0xb701] === 0x00 && m.ram[0xb702] === 0x70)) m.step();
  ram = m.ram.slice(0, 0x10000); listing = 're/game.lst';
} else { ram = new Uint8Array(fs.readFileSync(`re/${name}.start.ram`)); listing = `re/${name}.lst`; }
const idx = new Map();
for (let s = 0; s < dsk.length / 256; s++) { const k = dsk.subarray(s * 256, s * 256 + 256).toString('latin1'); if (!idx.has(k)) idx.set(k, s); }
const pages = [], patches = [], odd = [];
for (let p = 0x02; p < 0xc0; p++) {
  if (p >= 0x20 && p < 0x60) continue;
  const page = Buffer.from(ram.subarray(p * 256, p * 256 + 256));
  let s = idx.get(page.toString('latin1'));
  if (s === undefined) {
    let best = -1, bd = 257; for (let q = 0; q < dsk.length / 256; q++) { let d = 0; for (let i = 0; i < 256 && d < bd; i++) if (dsk[q * 256 + i] !== page[i]) d++; if (d < bd) { bd = d; best = q; } }
    if (bd > 64) { odd.push(p.toString(16) + ':' + bd); continue; }
    s = best; for (let i = 0; i < 256; i++) if (dsk[s * 256 + i] !== page[i]) patches.push([p * 256 + i, page[i]]);
  }
  pages.push([p, s]);
}
// the original's program: every instruction address in the study listing
const code = new Uint8Array(0x10000), LEN = [1, 1, 2, 2, 2, 2, 3, 3, 3, 2, 2, 3, 2];
for (const l of fs.readFileSync(listing, 'utf8').split('\n')) { const k = /^([0-9A-F]{4})  /.exec(l); if (k) { const a = parseInt(k[1], 16), n = LEN[OPTABLE[ram[a]].mode]; for (let i = 0; i < n; i++) code[a + i] = 1; } }
for (const a of [0x0d5e, 0x0d79, 0x0d7a, 0x0d88]) code[a] = 0;   // four bytes the tune player keeps its settings in
const wipe = []; for (let a = 0x200; a < 0xc000; a++) if (code[a] && !code[a - 1]) { let e = a; while (code[e]) e++; wipe.push([a, e - a]); }
const screens = new Set(ram.subarray(0x2000, 0x6000));
const low = [...ram.subarray(0, 0x200)];
fs.writeFileSync(new URL(`./ref/${name}.map.js`, import.meta.url), `// Made by tools/bake.mjs. How the memory of this scene is assembled from the player's disk.
// pages: [memory page, disk sector] (sector = track * 16 + sector, as stored in a .dsk file)
// patches: [address, value] where memory differs from the disk when the scene starts
// low: the first two pages (the game's variables) at that moment
// wipe: [address, length] of the original's program, cleared after loading
export default {
  pages: ${JSON.stringify(pages)},
  patches: ${JSON.stringify(patches)},
  low: ${JSON.stringify(low)},
  wipe: ${JSON.stringify(wipe)},
};
`);
console.log(name, 'pages', pages.length, 'patches', patches.length, 'not from the disk:', odd.join(' ') || 'none', 'wipe ranges', wipe.length, 'bytes', wipe.reduce((a, w) => a + w[1], 0), 'screen bytes', [...screens].map((x) => x.toString(16)).join(','));
