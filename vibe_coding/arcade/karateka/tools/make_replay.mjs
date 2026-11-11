// Build a replay: run the original with a key script, record every drawing call, page
// flip and speaker click, and verify on the way that the ported renderer (src/gfx.js)
// produces exactly the page the original shows at every flip.
// Usage: node make_replay.mjs "<script>" seconds name  → data/replay_<name>.json
import fs from 'node:fs';
import path from 'node:path';
import { boot, parseScript, applyKey } from './emu/play.mjs';
import { CPU_HZ } from './emu/machine.mjs';
import { Gfx } from '../src/gfx.js';
const [script, secs, name] = [process.argv[2] || '', +(process.argv[3] || 150), process.argv[4] || 'demo'];
const ENTRY = { 0x1903: 'blit', 0x1906: 'blitRight', 0x1909: 'blitMirror', 0x190c: 'blitMirrorRight', 0x0a00: 'fill', 0x0a03: 'fill2', 0x0a06: 'clearRightEdge', 0x1900: 'clear', 0x0780: 'scrollUp' };
const NAMES = Object.values(ENTRY);
const m = boot(), c = m.cpu, z = m.ram;
const mem = new Uint8Array(0x10000), gfx = new Gfx(mem);
const events = [], stats = { calls: 0, flips: 0, patches: 0, data: 0 };
const b64 = (u8) => Buffer.from(u8).toString('base64');
let started = false, lastPage2 = false, lastHires = false;
const ev = parseScript(script); let ei = 0;
const syncRange = (a, n) => { // make sure the port sees the same source bytes
  let diff = false; for (let i = 0; i < n; i++) if (mem[a + i] !== z[a + i]) { diff = true; break; }
  if (diff) { mem.set(z.subarray(a, a + n), a); events.push([c.cycles, 'd', a, b64(z.subarray(a, a + n))]); stats.data++; }
};
const checkPage = () => {
  const base = m.video.page2 ? 0x4000 : 0x2000;
  for (let i = 0; i < 0x2000; i++) if (mem[base + i] !== z[base + i]) {
    mem.set(z.subarray(base, base + 0x2000), base); events.push([c.cycles, 'p', base, b64(z.subarray(base, base + 0x2000))]); stats.patches++; return;
  }
};
const end = secs * CPU_HZ;
while (c.cycles < end) {
  if (ei < ev.length && c.cycles >= ev[ei].t * CPU_HZ) applyKey(m, ev[ei++].k);
  if (!started && m.video.hires && !m.video.text) {
    started = true;
    for (const [a, n] of [[0x0800, 0x200], [0x1a34, 14]]) { mem.set(z.subarray(a, a + n), a); events.push([c.cycles, 'd', a, b64(z.subarray(a, a + n))]); }
    mem.set(z.subarray(0x2000, 0x6000), 0x2000); events.push([c.cycles, 'p', 0x2000, b64(z.subarray(0x2000, 0x4000))], [c.cycles, 'p', 0x4000, b64(z.subarray(0x4000, 0x6000))]);
  }
  const e = started && ENTRY[c.pc];
  if (e) {
    const zp = [z[2], z[3], z[4], z[5], z[6], z[7], z[8], z[9], z[0x0f], z[0x10], z[0x11], z[0x12], z[0x13], z[0xe4]];
    [mem[2], mem[3], mem[4], mem[5], mem[6], mem[7], mem[8], mem[9], mem[0x0f], mem[0x10], mem[0x11], mem[0x12], mem[0x13], mem[0xe4]] = zp;
    if (e.startsWith('blit')) { const p = z[3] | (z[4] << 8); syncRange(p, 2 + z[p] * z[p + 1]); }
    events.push([c.cycles, NAMES.indexOf(e), ...zp]);
    gfx[e === 'fill2' ? 'fill' : e](e === 'fill2');
    stats.calls++;
  }
  c.step();
  if (started && (m.video.page2 !== lastPage2)) { lastPage2 = m.video.page2; checkPage(); events.push([c.cycles, 'f', lastPage2 ? 1 : 0]); stats.flips++; }
}
const t0 = events[0][0];
for (const e of events) e[0] -= t0;
const spk = m.speaker.filter((t) => t >= t0).map((t) => t - t0);
for (let i = spk.length - 1; i > 0; i--) spk[i] -= spk[i - 1];
fs.mkdirSync(path.resolve(import.meta.dirname, '../data'), { recursive: true });
const out = path.resolve(import.meta.dirname, `../data/replay_${name}.json`);
fs.writeFileSync(out, JSON.stringify({ hz: CPU_HZ, names: NAMES, events, speaker: spk, length: end - t0 }));
console.log(name, JSON.stringify(stats), 'speaker clicks', spk.length, 'size', (fs.statSync(out).size / 1e6).toFixed(1) + 'MB');
console.log('patches (pages the port could not reproduce from draw calls) at seconds:', events.filter((e) => e[1] === 'p').map((e) => ((e[0] + t0) / CPU_HZ).toFixed(1)).join(' '));
