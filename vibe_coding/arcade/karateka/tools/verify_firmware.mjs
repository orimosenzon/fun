// The ROM-less machine must show exactly the same sequence of pictures as the reference
// emulator with the real ROMs. Compared: the hash of the displayed page at every page flip.
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { boot, root, parseScript, applyKey } from './emu/play.mjs';
import { CPU_HZ } from './emu/machine.mjs';
import { GameMachine } from '../src/vm/firmware.js';
const secs = +(process.argv[2] || 150);
const script = parseScript(process.argv[3] || ''); // key times are seconds after the first picture appears
function flips(m, step, limitFlips) {
  const out = []; let last = m.video.page2, lastH = m.video.hires;
  const end = secs * CPU_HZ;
  let ei = 0, t0 = -1;
  while (m.cpu.cycles < end && out.length < limitFlips) {
    if (t0 >= 0 && ei < script.length && m.cpu.cycles >= t0 + script[ei].t * CPU_HZ) applyKey(m, script[ei++].k);
    step();
    if (t0 < 0 && m.video.hires && !m.video.text) t0 = m.cpu.cycles;
    if (m.video.page2 !== last || m.video.hires !== lastH) { last = m.video.page2; lastH = m.video.hires; out.push([crypto.createHash('md5').update(m.hgrPage()).digest('hex').slice(0, 8), m.cpu.cycles]); }
  }
  return out;
}
const ref = boot(); const a = flips(ref, () => ref.cpu.step(), 1e9);
const g = new GameMachine(new Uint8Array(fs.readFileSync(path.join(root, 'disks', 'Karateka_1984_Broderbund_a3.dsk'))));
let b; try { b = flips(g, () => g.step(), a.length); } catch (e) { console.log('ERROR', e.message, 'pc', g.cpu.pc.toString(16)); process.exit(1); }
let same = 0; while (same < a.length && same < b.length && a[same][0] === b[same][0]) same++;
console.log(`reference flips ${a.length}, rom-less flips ${b.length}, identical prefix ${same}`);
if (same > 2) { const d = []; for (let i = 1; i < same; i++) d.push((b[i][1] - b[i - 1][1]) - (a[i][1] - a[i - 1][1])); const nz = d.filter((x) => x !== 0); console.log('frame-to-frame timing differences (cycles): nonzero', nz.length, 'of', d.length, 'max', Math.max(0, ...nz.map(Math.abs))); }
console.log('first flip at: reference', (a[0][1] / CPU_HZ).toFixed(2) + 's', 'rom-less', (b[0]?.[1] / CPU_HZ).toFixed(2) + 's', 'illegal ops', g.cpu.illegal);
process.exit(same === a.length ? 0 : 1);
