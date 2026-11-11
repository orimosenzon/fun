// Which code draws to the hi-res pages, and where does it read its data from?
import fs from 'node:fs';
import path from 'node:path';
import { Apple2 } from './apple2.mjs';
const root = path.resolve(import.meta.dirname, '../../reference');
const [t0, t1] = [+(process.argv[2] || 108), +(process.argv[3] || 112)];
const m = new Apple2({ romDir: path.join(root, 'roms'), disk: path.join(root, 'disks', 'Karateka_1984_Broderbund_a3.dsk') });
m.cpu.pc = 0xc600;
m.runSeconds(t0);
fs.writeFileSync(path.join(root, `ram_${t0}.bin`), m.ram);
const c = m.cpu;
const writers = new Map(), exec = new Uint32Array(0x10000), readBy = new Map();
let curPC = 0;
m.onWrite = (a) => { if (a >= 0x2000 && a < 0x6000) writers.set(curPC, (writers.get(curPC) || 0) + 1); };
const origRead = c.read;
const dataReads = new Uint32Array(0x10000);
c.read = (a) => { if (a !== curPC && (a < curPC || a > curPC + 2)) dataReads[a]++; return origRead(a); };
const end = c.cycles + (t1 - t0) * 1020484;
while (c.cycles < end) { curPC = c.pc; exec[curPC]++; c.step(); }
const ranges = (arr, min = 1) => { const out = []; let s = -1; for (let a = 0; a <= 0x10000; a++) { const on = a < 0x10000 && arr[a] >= min; if (on && s < 0) s = a; if (!on && s >= 0) { out.push(`${s.toString(16)}-${(a - 1).toString(16)}`); s = -1; } } return out; };
console.log('HGR writers (pc:count):', [...writers].sort((a, b) => b[1] - a[1]).slice(0, 30).map(([k, v]) => k.toString(16) + ':' + v).join(' '));
console.log('executed ranges:', ranges(exec).join(' '));
console.log('data read ranges (>=0x200, not hgr):', ranges(dataReads).filter((r) => { const s = parseInt(r, 16); return s >= 0x200 && (s < 0x2000 || s >= 0x6000); }).join(' '));
