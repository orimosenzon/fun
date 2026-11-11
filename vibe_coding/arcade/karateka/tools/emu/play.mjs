// Scripted play of the original. Usage: node play.mjs "<script>" total every [prefix]
// script: comma separated  time:key  (key = single char, or SPACE/LEFT/RIGHT/UP/DOWN/RET/ESC, or B0+/B0- for button 0)
import fs from 'node:fs';
import path from 'node:path';
import { Apple2, savePagePNG, CPU_HZ } from './apple2.mjs';
export const root = path.resolve(import.meta.dirname, '../../reference');
const KEYS = { SPACE: 0x20, LEFT: 0x08, RIGHT: 0x15, UP: 0x0b, DOWN: 0x0a, RET: 0x0d, ESC: 0x1b };
export function boot() {
  const m = new Apple2({ romDir: path.join(root, 'roms'), disk: path.join(root, 'disks', 'Karateka_1984_Broderbund_a3.dsk') });
  m.cpu.pc = 0xc600; return m;
}
export function parseScript(s) {
  return s.split(',').filter(Boolean).map((e) => { const i = e.indexOf(':'); return { t: +e.slice(0, i), k: e.slice(i + 1) }; }).sort((a, b) => a.t - b.t);
}
export function applyKey(m, k) {
  if (k === 'B0+') m.buttons[0] = 1; else if (k === 'B0-') m.buttons[0] = 0;
  else m.pressKey(KEYS[k] ?? k.charCodeAt(0));
}
if (process.argv[1] === import.meta.filename) {
  const [script, total, every, prefix = 'play'] = [process.argv[2], +process.argv[3], +process.argv[4], process.argv[5]];
  const ev = parseScript(script); const m = boot();
  fs.mkdirSync(path.join(root, 'shots'), { recursive: true });
  let next = +(process.argv[6] || every);
  const marks = [...ev.map((e) => ({ t: e.t, e })), ...Array.from({ length: Math.ceil(total / every) + 1 }, (_, i) => ({ t: next + i * every }))].filter((x) => x.t <= total).sort((a, b) => a.t - b.t);
  for (const mk of marks) {
    m.runTo(Math.round(mk.t * CPU_HZ));
    if (mk.e) applyKey(m, mk.e.k);
    else savePagePNG(path.join(root, 'shots', `${prefix}_${mk.t.toFixed(1).padStart(6, '0')}.png`), m.hgrPage(), 1);
  }
  console.log('done pc=' + m.cpu.pc.toString(16), 'spk', m.speaker.length);
}
