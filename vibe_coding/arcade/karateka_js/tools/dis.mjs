// Clean recursive disassembly of the original as it sits in memory during play, for study.
// usage: node tools/dis.mjs out.lst  (output goes under re/, which is not committed)
import fs from 'node:fs';
import { startGame, CPU_HZ } from './oracle.mjs';
import { OPTABLE, MODES as M } from '../../karateka/src/vm/cpu6502.js';
// with a second argument: disassemble a saved scene (re/<name>.ram + re/<name>.cov from tools/reach.mjs)
const out = process.argv[2] || 're/game.lst', saved = process.argv[3];
let ram, exec;
if (saved) { ram = new Uint8Array(fs.readFileSync(`re/${saved}.ram`)); exec = new Uint32Array(new Uint8Array(fs.readFileSync(`re/${saved}.cov`)).buffer); }
else {
  const m = startGame(), c = m.cpu; ram = m.ram; exec = new Uint32Array(0x10000);
  const script = 'K,SPACE,RIGHT,RIGHT,W,S,X,Q,A,Z,RIGHT,RIGHT,W,RIGHT,S,RIGHT,X,LEFT,Q,RIGHT,A,RIGHT,Z,B,SPACE,RIGHT,RIGHT,SPACE,W,W,S,S,RIGHT,RIGHT,X,X,RIGHT,W,RIGHT,S,RIGHT,X,RIGHT,W,RIGHT,S,RIGHT,W,SPACE,RIGHT,RIGHT,RIGHT,RIGHT,SPACE,W,S,X'.split(',');
  let next = c.cycles, i = 0;
  const end = c.cycles + 70 * CPU_HZ;
  while (c.cycles < end) { if (c.cycles >= next && i < script.length) { m.press(script[i++]); next = c.cycles + 0.6 * CPU_HZ; } if (c.pc < 0xc000) exec[c.pc]++; m.step(); }
}
const h2 = (v) => v.toString(16).padStart(2, '0').toUpperCase(), h4 = (v) => v.toString(16).padStart(4, '0').toUpperCase();
const LEN = [1, 1, 2, 2, 2, 2, 3, 3, 3, 2, 2, 3, 2];
const inCode = (a) => a >= 0x200 && a < 0xc000 && !(a >= 0x2000 && a < 0x6000);
const isInstr = new Uint8Array(0x10000), xref = new Map(), isSub = new Uint8Array(0x10000);
const work = [];
for (let a = 0x200; a < 0xc000; a++) if (exec[a] && inCode(a)) work.push(a);
const target = (o, a) => o.mode === M.REL ? (a + 2 + (ram[a + 1] > 127 ? ram[a + 1] - 256 : ram[a + 1])) & 0xffff : ram[a + 1] | (ram[a + 2] << 8);
while (work.length) {
  let a = work.pop();
  while (inCode(a) && !isInstr[a]) {
    const o = OPTABLE[ram[a]]; if (!o) break;
    isInstr[a] = 1;
    if (o.mode === M.REL || o.name === 'JSR' || (o.name === 'JMP' && o.mode === M.ABS)) { const t = target(o, a); if (!xref.has(t)) xref.set(t, []); xref.get(t).push(a); if (o.name === 'JSR') isSub[t] = 1; if (inCode(t)) work.push(t); }
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
  return '?';
};
const lines = []; let prevEnd = -1, n = 0;
for (let a = 0x200; a < 0xc000; a++) {
  if (!isInstr[a]) continue;
  const o = OPTABLE[ram[a]], len = LEN[o.mode];
  if (a !== prevEnd) lines.push('', `; ---- $${h4(a)}`);
  if (xref.has(a)) { const x = [...new Set(xref.get(a))].sort((p, q) => p - q); lines.push(`${isSub[a] ? 'sub' : 'L'}_${h4(a)}:    ; from ${x.slice(0, 8).map(h4).join(' ')}${x.length > 8 ? ' +' + (x.length - 8) : ''}`); }
  lines.push(`${h4(a)}  ${o.name} ${fmt(o, a)}`.padEnd(24) + (exec[a] ? `; ${exec[a]}` : ';-'));
  n++; prevEnd = a + len; for (let k = 1; k < len; k++) isInstr[a + k] = 0;
}
fs.writeFileSync(out, lines.join('\n') + '\n');
if (!saved) fs.writeFileSync(out.replace(/\.lst$/, '.ram'), ram);
console.log(n, 'instructions,', lines.length, 'lines ->', out);
