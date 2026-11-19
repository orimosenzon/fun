// node tools/dis.mjs <name>: recursive disassembly of reference/re/<name>.ram, seeded by
// the execution counts in reference/re/<name>.cov (both written by a lab script).
import fs from 'node:fs';
import { OPNAMES, OPMODES, MODES as M } from '../src/vm/cpu6510.js';
const name = process.argv[2] || 'combat';
const dir = new URL('../reference/re/', import.meta.url).pathname;
const ram = new Uint8Array(fs.readFileSync(`${dir}${name}.ram`));
const exec = new Uint32Array(new Uint8Array(fs.readFileSync(`${dir}${name}.cov`)).buffer);
const h2 = (v) => v.toString(16).padStart(2, '0').toUpperCase(), h4 = (v) => v.toString(16).padStart(4, '0').toUpperCase();
const LEN = [1, 1, 2, 2, 2, 2, 3, 3, 3, 2, 2, 3, 2];
const inCode = (a) => a >= 0x0800 && a < 0xd000;
const isInstr = new Uint8Array(0x10000), xref = new Map(), isSub = new Uint8Array(0x10000);
const work = [];
for (let a = 0; a < 0x10000; a++) if (exec[a] && inCode(a)) work.push(a);
const target = (op, a) => OPMODES[op] === M.REL ? (a + 2 + (ram[a + 1] > 127 ? ram[a + 1] - 256 : ram[a + 1])) & 0xffff : ram[a + 1] | (ram[a + 2] << 8);
const STOP = new Set(['RTS', 'RTI', 'JMP', 'BRK', 'KIL']);
while (work.length) {
  let a = work.pop();
  while (inCode(a) && !isInstr[a]) {
    const op = ram[a], nm = OPNAMES[op], mode = OPMODES[op];
    isInstr[a] = 1;
    if (mode === M.REL || nm === 'JSR' || (nm === 'JMP' && mode === M.ABS)) { const t = target(op, a); if (!xref.has(t)) xref.set(t, []); xref.get(t).push(a); if (nm === 'JSR') isSub[t] = 1; if (inCode(t)) work.push(t); }
    if (STOP.has(nm)) break;
    a += LEN[mode];
  }
}
const fmt = (op, a) => {
  const b1 = ram[a + 1], w = b1 | (ram[a + 2] << 8);
  switch (OPMODES[op]) {
    case M.IMP: return ''; case M.ACC: return 'A'; case M.IMM: return `#$${h2(b1)}`;
    case M.ZP: return `$${h2(b1)}`; case M.ZPX: return `$${h2(b1)},X`; case M.ZPY: return `$${h2(b1)},Y`;
    case M.ABS: return `$${h4(w)}`; case M.ABX: return `$${h4(w)},X`; case M.ABY: return `$${h4(w)},Y`;
    case M.IZX: return `($${h2(b1)},X)`; case M.IZY: return `($${h2(b1)}),Y`; case M.IND: return `($${h4(w)})`;
    case M.REL: return `$${h4(target(op, a))}`;
  }
  return '?';
};
const lines = []; let prevEnd = -1, n = 0;
for (let a = 0; a < 0x10000; a++) {
  if (!isInstr[a]) continue;
  const op = ram[a], len = LEN[OPMODES[op]];
  if (a !== prevEnd) lines.push('', `; ---- $${h4(a)}`);
  if (xref.has(a)) { const x = [...new Set(xref.get(a))].sort((p, q) => p - q); lines.push(`${isSub[a] ? 'sub' : 'L'}_${h4(a)}:    ; from ${x.slice(0, 8).map(h4).join(' ')}${x.length > 8 ? ' +' + (x.length - 8) : ''}`); }
  lines.push(`${h4(a)}  ${OPNAMES[op]} ${fmt(op, a)}`.padEnd(24) + (exec[a] ? `; ${exec[a]}` : ';-'));
  n++; prevEnd = a + len; for (let k = 1; k < len; k++) isInstr[a + k] = 0;
}
fs.writeFileSync(`${dir}${name}.lst`, lines.join('\n') + '\n');
console.log(n, 'instructions ->', `${dir}${name}.lst`);
