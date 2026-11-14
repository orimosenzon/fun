// Differential check of the rewritten routines against the original.
// The original is played with a stream of key presses; every time it enters one of the routines
// under test, its memory is copied, the JavaScript routine is run on the copy, and when the
// original returns the two memories must be identical.
// usage: node tools/check.mjs <addr|all> [...] [--secs N] [--max N] [--seed N] [--v]
import { startGame, CPU_HZ } from './oracle.mjs';
import { Game } from '../src/index.js';
const args = process.argv.slice(2), opt = (k, d) => { const i = args.indexOf('--' + k); return i < 0 ? d : +args.splice(i, 2)[1]; };
const secs = opt('secs', 60), max = opt('max', 400), seed = opt('seed', 1);
const verbose = args.includes('--v'); if (verbose) args.splice(args.indexOf('--v'), 1);
// --cheat: at the top of every frame the player is given full strength and the opponent is left with one point,
// so that a random player gets far into the game
const cheat = args.includes('--cheat'); if (cheat) args.splice(args.indexOf('--cheat'), 1);
const names = Object.getOwnPropertyNames(Game.prototype).filter((k) => /^r[0-9A-F]{4}$/.test(k));
const want = args.includes('all') || !args.length ? names : args.map((a) => 'r' + a.toUpperCase());
const table = new Map(want.map((k) => [parseInt(k.slice(1), 16), { name: k, seen: 0, ran: 0, bad: 0, soft: 0, first: null }]));
for (const k of want) if (!names.includes(k)) { console.log('no such routine', k); process.exit(2); }

const m = startGame(), c = m.cpu, ram = m.ram;
m.press('K'); m.sec(0.5);
let rnd = seed * 2654435761 >>> 0; const rand = (n) => { rnd = (Math.imul(rnd, 1664525) + 1013904223) >>> 0; return (rnd >>> 8) % n; };
const KEYS = ['RIGHT', 'RIGHT', 'RIGHT', 'LEFT', 'W', 'S', 'X', 'Q', 'A', 'Z', 'SPACE', 'B', 'RIGHT', 'W', 'S'];
const pending = [];
// routines that last many frames: no key is pressed while one of them runs, or no case of theirs would survive
const LONG = new Set([0x0d00, 0x1000, 0x1003, 0x1006, 0x1009, 0x100c, 0xb400, 0xb403, 0x79a0, 0x7a6d, 0x7adc, 0xb763]);
const TIMED = new Set(['r0D00', 'r0C40', 'r0C43', 'r0C46', 'r0C49', 'r0C4C', 'r0C4F', 'r0C52', 'r0DC0']);
// not compared: the stack, the drawing routines' scratch bytes ($00/$01 pointer, $0A, $14-$1D) and their self-modified code
const SKIP = (a) => (a >= 0x100 && a < 0x200) || a < 2 || a === 0x0a || (a >= 0x14 && a <= 0x1d) || (a >= 0x1900 && a < 0x1d00);
let epoch = 0;   // counts key presses: a key that arrives in the middle of a routine makes the case unusable
function finish(f) {
  if (f.epoch !== epoch) return;
  const t = table.get(f.addr); t.ran++;
  const g = new Game(f.mem); g.a = f.a; g.x = f.x; g.y = f.y; g.c = f.c; g.key = f.key;
  let err = null;
  try { g[t.name](); } catch (e) { err = e; }
  const diffs = [];
  if (!err) for (let i = 0; i < 0xc000 && diffs.length < 12; i++) if (f.mem[i] !== ram[i] && !SKIP(i)) diffs.push(`$${i.toString(16)}: ours ${f.mem[i].toString(16)} theirs ${ram[i].toString(16)} (was ${f.before[i].toString(16)})`);
  const regs = []; if (g.a !== c.a) regs.push(`A ours ${g.a} theirs ${c.a}`); if (g.x !== c.x) regs.push(`X ours ${g.x} theirs ${c.x}`); if (g.y !== c.y) regs.push(`Y ours ${g.y} theirs ${c.y}`); if (g.c !== c.c) regs.push(`C ours ${g.c} theirs ${c.c}`);
  if (g.key !== m.key) diffs.push(`key latch ours ${g.key} theirs ${m.key}`);
  if (TIMED.has(t.name) && !err) {   // sound: the speaker must flip at the same moments, and the routine take as long
    const theirs = m.speaker.slice(f.spk).map((x) => x - m.speaker[f.spk]), ours = g.clicks.map((x) => x - g.clicks[0]);
    if (theirs.length !== ours.length) diffs.push(`speaker flips: ours ${ours.length} theirs ${theirs.length}`);
    else { const k = ours.findIndex((v, i) => v !== theirs[i]); if (k >= 0) diffs.push(`speaker flip ${k} at +${ours[k]} cycles, theirs +${theirs[k]}`); }
    if (g.cycles !== c.cycles - f.cyc && !(f.key & 0x80)) diffs.push(`duration ours ${g.cycles} theirs ${c.cycles - f.cyc} cycles`);
  }
  if (err || diffs.length) { t.bad++; if (!t.first) t.first = { in: `A=${f.a.toString(16)} X=${f.x.toString(16)} Y=${f.y.toString(16)} C=${f.c}`, diffs, err: err && (err.stack || String(err)).split('\n').slice(0, 4).join('\n'), n: t.ran }; }
  else if (regs.length) { t.soft++; if (!t.softFirst) t.softFirst = regs.join(', ') + ` (in A=${f.a.toString(16)} X=${f.x.toString(16)} Y=${f.y.toString(16)})`; }
}
const end = c.cycles + secs * CPU_HZ; let nextKey = c.cycles;
while (c.cycles < end) {
  if (c.cycles >= nextKey && !pending.some((p) => LONG.has(p.addr))) { epoch++; m.press(KEYS[rand(KEYS.length)]); nextKey = c.cycles + (0.15 + rand(60) / 100) * CPU_HZ; }
  if (cheat && c.pc === 0xb7ca) { if (ram[0x5d]) ram[0xb6] = ram[0xb0]; if (ram[0xb7] > 1) ram[0xb7] = 1; }
  if (m.drive.motor) { console.log(`the original went to the disk after ${((c.cycles - (end - secs * CPU_HZ)) / CPU_HZ).toFixed(0)} s: stopping there`); break; }
  const t = table.get(c.pc);
  if (t && m.inGame) {
    t.seen++;
    if (t.ran + pending.filter((p) => p.addr === c.pc).length < max && (t.seen <= max / 2 || t.seen % 7 === 0)) {
      const sp = c.sp, ret = ((ram[0x101 + sp] | (ram[0x102 + sp] << 8)) + 1) & 0xffff;
      const before = ram.slice(0, 0x10000);
      pending.push({ addr: c.pc, sp, ret, before, mem: before.slice(), a: c.a, x: c.x, y: c.y, c: c.c, key: m.key, epoch, spk: m.speaker.length, cyc: c.cycles });
    }
  }
  m.step();
  while (pending.length) { const f = pending[pending.length - 1]; if (c.pc === f.ret && c.sp === ((f.sp + 2) & 0xff)) { pending.pop(); finish(f); } else break; }
}
let fail = 0;
for (const [addr, t] of [...table].sort((p, q) => p[0] - q[0])) {
  const status = t.bad ? 'FAIL' : t.ran ? (t.soft ? 'ok*  ' : 'ok   ') : 'never';
  if (t.bad) fail++;
  if (t.bad || verbose || !t.ran) console.log(`${status} ${t.name}  ${t.ran - t.bad}/${t.ran} cases` + (t.soft ? `  (registers differ in ${t.soft}: ${t.softFirst})` : ''));
  if (t.bad) { console.log(`     first failure (case ${t.first.n}, ${t.first.in}):`); if (t.first.err) console.log('     ' + t.first.err); for (const d of t.first.diffs) console.log('       ' + d); }
}
const ran = [...table.values()].filter((t) => t.ran).length;
console.log(`${ran - fail}/${table.size} routines identical to the original` + (fail ? `, ${fail} FAILING` : '') + (ran < table.size ? `, ${table.size - ran} never reached in this run` : ''));
process.exit(fail ? 1 : 0);
