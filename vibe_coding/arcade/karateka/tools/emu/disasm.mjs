// Coverage-seeded recursive disassembly of the original.
// Usage: node disasm.mjs "<script>" seconds name   → reference/<name>.lst, <name>.ram, <name>.cov
import fs from 'node:fs';
import path from 'node:path';
import { boot, parseScript, applyKey, root } from './play.mjs';
import { CPU_HZ } from './apple2.mjs';
import { OPTABLE, MODES as M } from './cpu6502.mjs';
const [script, secs, name] = [process.argv[2] || '', +(process.argv[3] || 150), process.argv[4] || 'demo'];
const m = boot(), c = m.cpu, exec = new Uint32Array(0x10000);
const ev = parseScript(script); let ei = 0;
const end = secs * CPU_HZ;
while (c.cycles < end) {
  if (ei < ev.length && c.cycles >= ev[ei].t * CPU_HZ) applyKey(m, ev[ei++].k);
  exec[c.pc]++; c.step();
}
const ram = m.ram;
const h2 = (v) => v.toString(16).padStart(2, '0').toUpperCase(), h4 = (v) => v.toString(16).padStart(4, '0').toUpperCase();
const LEN = [1, 1, 2, 2, 2, 2, 3, 3, 3, 2, 2, 3, 2];
const inCode = (a) => a >= 0x200 && a < 0xc000 && !(a >= 0x2000 && a < 0x6000);
const isInstr = new Uint8Array(0x10000), label = new Uint8Array(0x10000);
const work = [];
for (let a = 0x200; a < 0xc000; a++) if (exec[a] && inCode(a)) work.push(a);
const target = (o, a) => o.mode === M.REL ? (a + 2 + (ram[a + 1] > 127 ? ram[a + 1] - 256 : ram[a + 1])) & 0xffff : ram[a + 1] | (ram[a + 2] << 8);
while (work.length) {
  let a = work.pop();
  while (inCode(a) && !isInstr[a]) {
    const o = OPTABLE[ram[a]]; if (!o) break;
    isInstr[a] = 1;
    if (o.mode === M.REL || o.name === 'JSR' || (o.name === 'JMP' && o.mode === M.ABS)) { const t = target(o, a); label[t] = Math.max(label[t], o.name === 'JSR' ? 2 : 1); if (inCode(t)) work.push(t); }
    if (o.name === 'RTS' || o.name === 'RTI' || o.name === 'JMP' || o.name === 'BRK') break;
    a += LEN[o.mode];
  }
}
const fmt = (o, a) => {
  const b1 = ram[a + 1], w = b1 | (ram[a + 2] << 8);
  switch (o.mode) {
    case M.IMP: return ''; case M.ACC: return 'A'; case M.IMM: return `#$${h2(b1)}`;
    case M.ZP: return `$${h2(b1)}`; case M.ZPX: return `$${h2(b1)},X`; case M.ZPY: return `$${h2(b1)},Y`;
    case M.ABS: return `$${h4(w)}`; case M.ABX: return `$${h4(w)},X`; case M.ABY: return `$${h4(w)},Y`;
    case M.IZX: return `($${h2(b1)},X)`; case M.IZY: return `($${h2(b1)}),Y`; case M.IND: return `($${h4(w)})`;
    case M.REL: return `$${h4(target(o, a))}`;
  }
};
const lines = []; let prevEnd = -1, n = 0, nx = 0;
for (let a = 0x200; a < 0xc000; a++) {
  if (!isInstr[a]) continue;
  const o = OPTABLE[ram[a]], len = LEN[o.mode];
  if (a !== prevEnd) lines.push('', `; ---- $${h4(a)}`);
  if (label[a] === 2) lines.push(`sub_${h4(a)}:`); else if (label[a] === 1) lines.push(`L${h4(a)}:`);
  lines.push(`${h4(a)} ${[...ram.slice(a, a + len)].map(h2).join('').padEnd(6)} ${o.name} ${fmt(o, a)}`.padEnd(30) + (exec[a] ? `;${exec[a]}` : ';-'));
  prevEnd = a + len; n++; if (exec[a]) nx++;
}
fs.writeFileSync(path.join(root, `${name}.lst`), lines.join('\n'));
fs.writeFileSync(path.join(root, `${name}.cov`), Buffer.from(exec.buffer));
fs.writeFileSync(path.join(root, `${name}.ram`), ram);
const blocks = []; let s = -1, last = -1;
for (let a = 0x200; a < 0xc000; a++) if (isInstr[a]) { if (s < 0 || a - last > 48) { if (s >= 0) blocks.push(`${h4(s)}-${h4(last)}`); s = a; } last = a; }
blocks.push(`${h4(s)}-${h4(last)}`);
console.log(`${name}: instructions ${n} (executed ${nx}), lines ${lines.length}`);
console.log('blocks:', blocks.join(' '));
console.log('zp/state:', 'pc', h4(c.pc));
