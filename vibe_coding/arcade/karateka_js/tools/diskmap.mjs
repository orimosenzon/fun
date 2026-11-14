// Where on the disk does each page of the running game come from? (study tool)
import fs from 'node:fs';
import { startGame, DISK } from './oracle.mjs';
const dsk = fs.readFileSync(DISK), m = startGame(), ram = m.ram;
const idx = new Map();
for (let s = 0; s < dsk.length / 256; s++) { const k = dsk.subarray(s * 256, s * 256 + 256).toString('latin1'); if (!idx.has(k)) idx.set(k, s); }
const out = []; let line = '';
for (let p = 0x02; p < 0xc0; p++) {
  const page = Buffer.from(ram.subarray(p * 256, p * 256 + 256));
  let s = idx.get(page.toString('latin1'));
  let how = s === undefined ? '' : 'exact';
  if (s === undefined) { // closest sector
    let best = -1, bd = 257; for (let q = 0; q < dsk.length / 256; q++) { let d = 0; for (let i = 0; i < 256 && d < bd; i++) if (dsk[q * 256 + i] !== page[i]) d++; if (d < bd) { bd = d; best = q; } }
    if (bd < 128) { s = best; how = 'diff' + bd; }
  }
  out.push([p, s, how]);
}
for (const [p, s, how] of out) console.log(p.toString(16).padStart(2, '0'), s === undefined ? '--' : `T${(s >> 4).toString(16)} S${(s & 15).toString(16)}`, how);
