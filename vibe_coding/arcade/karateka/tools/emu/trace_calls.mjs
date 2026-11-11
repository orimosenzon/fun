// Record every call to the original's drawing entry points during a scripted run.
// Usage: node trace_calls.mjs "<script>" seconds name  → reference/<name>.calls.json
import fs from 'node:fs';
import path from 'node:path';
import { boot, parseScript, applyKey, root } from './play.mjs';
import { CPU_HZ } from './apple2.mjs';
const [script, secs, name] = [process.argv[2] || '', +(process.argv[3] || 150), process.argv[4] || 'demo'];
const ENTRY = { 0x1903: 'blit', 0x1906: 'blitR', 0x1909: 'blitM', 0x190c: 'blitMR', 0x0a00: 'fill', 0x0a03: 'fill2', 0x0a06: 'edge', 0x1900: 'clear' };
const m = boot(), c = m.cpu, calls = [], shapes = {};
const ev = parseScript(script); let ei = 0;
const end = secs * CPU_HZ;
while (c.cycles < end) {
  if (ei < ev.length && c.cycles >= ev[ei].t * CPU_HZ) applyKey(m, ev[ei++].k);
  const e = ENTRY[c.pc];
  if (e) {
    const z = m.ram, ret = (z[0x101 + c.sp] | (z[0x102 + c.sp] << 8)) + 1;
    const rec = { t: +(c.cycles / CPU_HZ).toFixed(3), e, ret, x: z[5], y: z[6], page: z[7], shift: z[0x10], mode: z[0x0f] };
    if (e.startsWith('blit')) {
      rec.ptr = z[3] | (z[4] << 8);
      const k = rec.ptr.toString(16);
      rec.h = z[rec.ptr];
      if (shapes[k] && shapes[k].h < z[rec.ptr]) { shapes[k].h = z[rec.ptr]; shapes[k].data = Buffer.from(z.slice(rec.ptr, rec.ptr + 2 + z[rec.ptr] * z[rec.ptr + 1])).toString('hex'); }
      if (!shapes[k]) shapes[k] = { h: z[rec.ptr], w: z[rec.ptr + 1], data: Buffer.from(z.slice(rec.ptr, rec.ptr + 2 + z[rec.ptr] * z[rec.ptr + 1])).toString('hex'), n: 0, modes: {}, kinds: {} };
      shapes[k].n++; shapes[k].modes[rec.mode] = 1; shapes[k].kinds[e] = 1;
    } else if (e !== 'clear' && e !== 'edge') Object.assign(rec, { x2: z[9], y2: z[8], a: z[2], b: z[0x11], c: z[0x12], d: z[0x13] });
    calls.push(rec);
  }
  c.step();
}
fs.writeFileSync(path.join(root, `${name}.calls.json`), JSON.stringify({ calls, shapes }));
const byRet = {}; for (const r of calls) byRet[r.ret.toString(16)] = (byRet[r.ret.toString(16)] || 0) + 1;
console.log(name, 'calls', calls.length, 'shapes', Object.keys(shapes).length);
console.log('callers:', Object.entries(byRet).sort((a, b) => b[1] - a[1]).map(([k, v]) => k + ':' + v).join(' '));
const ks = Object.keys(shapes).map((k) => parseInt(k, 16)).sort((a, b) => a - b);
console.log('shape ptr range', ks[0].toString(16), ks[ks.length - 1].toString(16));
