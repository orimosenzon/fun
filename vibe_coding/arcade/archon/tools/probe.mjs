// node tools/probe.mjs "<script>" <seconds>: run, then dump chip state.
import { startGame, KEYS, JOY, shot } from './node64.mjs';
const [script = '', secs = '20', png] = process.argv.slice(2);
const m = startGame();
const fps = m.model.hz / (m.model.lines * m.model.cpl);
const ev = [];
for (const item of script.split(',').filter(Boolean)) {
  const [t, a, b] = item.split(':'); const at = Math.round(parseFloat(t) * fps);
  if (a === 'J1' || a === 'J2') { let bits = 0; for (const ch of b || '') bits |= { U: 1, D: 2, L: 4, R: 8, F: 16 }[ch]; ev.push({ at, fn: () => { m.joy[a === 'J1' ? 0 : 1] = bits; } }); }
  else ev.push({ at, fn: () => m.setKey(KEYS[a], true) }, { at: at + 9, fn: () => m.setKey(KEYS[a], false) });
}
const pcs = new Map();
const total = Math.round(parseFloat(secs) * fps);
for (let f = 0; f <= total; f++) {
  for (const e of ev) if (e.at === f) e.fn();
  if (f > total - 60) { m.vic.frameReady = false; while (!m.vic.frameReady) { const pc = m.cpu.pc & 0xff00; pcs.set(pc, (pcs.get(pc) || 0) + 1); m.step(); } }
  else m.runFrame();
}
const h = (v, n = 2) => v.toString(16).padStart(n, '0');
const r = m.vic.reg;
console.log('sprites en', h(r[0x15]), 'msb', h(r[0x10]), 'mc', h(r[0x1c]), 'xexp', h(r[0x1d]), 'yexp', h(r[0x17]), 'prio', h(r[0x1b]));
for (let s = 0; s < 8; s++) console.log(' spr', s, 'x', r[s * 2] + ((r[0x10] >> s) & 1) * 256, 'y', r[s * 2 + 1], 'col', r[0x27 + s] & 15, 'ptr', h(m.ram[(((r[0x18] & 0xf0) << 6) | m.vic.bank) + 0x3f8 + s]));
console.log('d011', h(r[0x11]), 'd016', h(r[0x16]), 'd018', h(r[0x18]), 'bank', h(m.vic.bank, 4), 'irqmask', h(m.vic.irqMask), 'rastercmp', m.vic.rasterCompare);
console.log('cia1 cra', h(m.cia1.cra), 'la', m.cia1.la, 'mask', h(m.cia1.mask), 'ddra', h(m.cia1.ddra), 'ddrb', h(m.cia1.ddrb), '| cia2 mask', h(m.cia2.mask));
console.log('irq vec', h(m.ram[0x314] | (m.ram[0x315] << 8), 4), 'hw irq', h(m.read(0xfffe) | (m.read(0xffff) << 8), 4), 'port', h(m.port));
console.log('hot pages:', [...pcs].sort((a, b) => b[1] - a[1]).slice(0, 8).map(([p, n]) => h(p, 4) + ':' + n).join(' '));
if (png) shot(m, png);
