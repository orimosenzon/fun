// Who reads the disk after the game has started, with which call chain, and what changes in memory?
import { boot, parseScript, applyKey } from './play.mjs';
import { CPU_HZ } from './machine.mjs';
import { OPTABLE } from './cpu6502.mjs';
const [script, secs] = [process.argv[2] || '', +(process.argv[3] || 150)];
const m = boot(), c = m.cpu, z = m.ram;
const ev = parseScript(script); let ei = 0;
const h = (v) => v.toString(16);
let motor = false, before = null, t0 = 0, readers = new Set(), stacks = new Set(), romCalls = new Map(), romReads = new Map(), ioReads = new Map();
const origRead = c.read; let curPC = 0;
c.read = (a) => { if (a >= 0xc000 && curPC < 0xc000 && c.cycles > 8 * CPU_HZ) { const map = a >= 0xd000 ? romReads : ioReads; const k = h(a); if (!map.has(k)) map.set(k, new Set()); map.get(k).add(h(curPC)); } return origRead(a); };
const end = secs * CPU_HZ;
while (c.cycles < end) {
  if (ei < ev.length && c.cycles >= ev[ei].t * CPU_HZ) applyKey(m, ev[ei++].k);
  curPC = c.pc;
  if (curPC < 0xc000 && c.cycles > 8 * CPU_HZ) { const op = z[curPC]; if (op === 0x20 || op === 0x4c) { const t = z[curPC + 1] | (z[curPC + 2] << 8); if (t >= 0xc000) romCalls.set(h(t), (romCalls.get(h(t)) || new Set()).add(h(curPC))); } }
  if (m.drive.motor && !motor) {
    motor = true; before = z.slice(); t0 = c.cycles; readers = new Set();
    const st = []; for (let s = c.sp + 1; s < 0xff; s += 2) st.push(h((z[0x100 + s] | (z[0x101 + s] << 8)) + 1));
    console.log(`\nmotor ON t=${(c.cycles / CPU_HZ).toFixed(2)} pc=${h(curPC)} stack(ret addrs)=${st.join(',')} A=${h(c.a)} X=${h(c.x)} Y=${h(c.y)}`);
  }
  if (!m.drive.motor && motor) {
    motor = false;
    const r = []; let s = -1, p = -1;
    for (let i = 0x200; i < 0xc000; i++) if (z[i] !== before[i]) { if (s < 0) s = i; else if (i - p > 16) { r.push(h(s) + '-' + h(p)); s = i; } p = i; }
    if (s >= 0) r.push(h(s) + '-' + h(p));
    console.log(`motor OFF t=${(c.cycles / CPU_HZ).toFixed(2)} dur=${((c.cycles - t0) / CPU_HZ).toFixed(2)} track=${m.drive.halfTrack / 2} changed: ${r.join(' ')}`);
  }
  c.step();
}
const show = (map) => [...map].map(([k, v]) => `${k}<-[${[...v].slice(0, 6).join(',')}]`).join(' ');
console.log('\nROM calls:', show(romCalls));
console.log('ROM data reads:', show(romReads));
console.log('IO reads:', show(ioReads));
