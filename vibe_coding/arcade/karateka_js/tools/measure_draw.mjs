// How long does each of the original's drawing calls take? Plays the original with the test bot and
// times every call of the drawing routines, with what was asked of them. (study tool)
import fs from 'node:fs';
import { boot } from './oracle.mjs';
const m = boot(), c = m.cpu, ram = m.ram;
m.sec(20); m.press('SPACE');
while (!(m.inGame && c.pc === 0xb700 && ram[0xb700] === 0x20 && ram[0xb702] === 0x70)) m.step();
ram[0xc4] = 1;
const ENTRY = { 0x1903: 'blit', 0x1906: 'blitR', 0x1909: 'mir', 0x190c: 'mirR', 0x0a00: 'fill', 0x0a03: 'fill2', 0x1900: 'clear', 0x0a06: 'edge' };
let rnd = 2 * 2654435761 >>> 0; const rand = (n) => { rnd = (Math.imul(rnd, 1664525) + 1013904223) >>> 0; return (rnd >>> 8) % n; };
const POOL = ['RIGHT', 'RIGHT', 'RIGHT', 'RIGHT', 'W', 'S', 'X', 'Q', 'A', 'Z', 'W', 'S', 'RIGHT'], KEYS = { RIGHT: 0x15, SPACE: 0x20 };
const out = [], pend = []; let frames = 0;
while (frames < 3300 && !m.drive.motor) {
  const k = ENTRY[c.pc];
  if (k) pend.push({ k, sp: c.sp, ret: ((ram[0x101 + c.sp] | (ram[0x102 + c.sp] << 8)) + 1) & 0xffff, t: c.cycles, shift: ram[0x10], mode: ram[0x0f], x: ram[5], y: ram[6], y2: ram[8], x2: ram[9], ptr: ram[3] | (ram[4] << 8) });
  if (c.pc === 0xb7ab || c.pc === 0xb7c1) { frames++; if (ram[0x5d]) ram[0xb6] = ram[0xb0]; if (ram[0xb7] > 1) ram[0xb7] = 1;
    if (rand(3) === 0) { let key = POOL[rand(POOL.length)]; if (ram[0x5e] && ((ram[0x72] - ram[0x62]) & 255) < 0x18 && ram[0x6f]) key = 'SPACE'; m.pressKey(KEYS[key] ?? key.charCodeAt(0)); } }
  m.step();
  while (pend.length) { const f = pend[pend.length - 1]; if (c.pc === f.ret && c.sp === ((f.sp + 2) & 0xff)) { pend.pop(); out.push([f.k, c.cycles - f.t, ram[0x0d], ram[0x0e], f.shift, f.mode, f.x, f.y, f.y2, f.x2, f.ptr]); } else break; }
}
fs.writeFileSync('re/draw_costs.json', JSON.stringify(out));
const by = {}; for (const r of out) (by[r[0]] ??= []).push(r);
for (const [k, v] of Object.entries(by)) console.log(k, v.length, 'calls, cycles', Math.min(...v.map((r) => r[1])), '..', Math.max(...v.map((r) => r[1])));
