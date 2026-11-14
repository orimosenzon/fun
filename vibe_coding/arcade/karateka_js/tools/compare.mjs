// The whole game against the original, frame by frame.
// Both start from the same memory at the moment the original enters its level set-up ($B700);
// from then on the rewritten game runs on its own. The same keys are pressed in both, always
// between two passes of the main loop, and after every pass the two memories (screens included)
// must be identical.
// usage: node tools/compare.mjs [--frames N] [--seed N] [--cheat] [--keys idle|random] [--timing]
import { boot, CPU_HZ } from './oracle.mjs';
import fs from 'node:fs';
import { Game } from '../src/index.js';
import { fromDisk, MAPS } from './lib.mjs';
import { DISK } from './oracle.mjs';
const args = process.argv.slice(2), opt = (k, d) => { const i = args.indexOf('--' + k); return i < 0 ? d : args[i + 1]; };
const frames = +opt('frames', 2000), seed = +opt('seed', 1), cheat = args.includes('--cheat'), mode = opt('keys', 'random'), timing = args.includes('--timing');
const SKIP0 = (a) => (a >= 0x100 && a < 0x200) || a < 2 || a === 0x0a || (a >= 0x14 && a <= 0x1d) || (a >= 0x1900 && a < 0x1d00);

const m = boot(), c = m.cpu, ram = m.ram;
m.sec(20); m.press('SPACE');
while (!(m.inGame && c.pc === 0xb700 && ram[0xb700] === 0x20 && ram[0xb701] === 0x00 && ram[0xb702] === 0x70)) m.step();   // the game's set-up, not the disk routines that sit there while it loads
m.ram[0xc4] = 1; m.keyboardMode = true;                          // keyboard, as the page sets it
// --standalone: ours starts from nothing but the disk image (and with the original's program wiped from its memory)
const standalone = args.includes('--standalone');
const g = new Game(standalone ? fromDisk(new Uint8Array(fs.readFileSync(DISK)), 1) : ram.slice(0, 0x10000)); g.key = m.key;
const wiped = new Uint8Array(0x10000); if (standalone) for (const [a, n] of MAPS[1].wipe) wiped.fill(1, a, a + n);
let screens = !standalone;   // a stand-alone game starts with black screens; they are compared once the original has painted over its own
g.rB700();
const topLevel = () => c.pc === 0xb7ab || c.pc === 0xb7c1;   // the two places where the original's main loop starts a pass
while (!topLevel()) m.step();

let rnd = seed * 2654435761 >>> 0; const rand = (n) => { rnd = (Math.imul(rnd, 1664525) + 1013904223) >>> 0; return (rnd >>> 8) % n; };
const KEYS = { RIGHT: 0x95, LEFT: 0x88, SPACE: 0xa0 };
const POOL = ['RIGHT', 'RIGHT', 'RIGHT', 'RIGHT', 'LEFT', 'W', 'S', 'X', 'Q', 'A', 'Z', 'SPACE', 'B', 'W', 'S', 'RIGHT'];
const u8 = (v) => v & 0xff;
// the 8 bytes at the end of every 128 in the screen memory are never shown
const SKIP = (a) => SKIP0(a) || wiped[a] || (a >= 0x2000 && a < 0x6000 && (!screens || (standalone && (a & 0x7f) >= 0x78)));
const diff = () => { const out = []; for (let i = 0; i < 0xc000 && out.length < 10; i++) if (g.m[i] !== ram[i] && !SKIP(i)) out.push(`$${i.toString(16)}: ours ${g.m[i].toString(16)} theirs ${ram[i].toString(16)}`); if (g.key !== m.key) out.push(`key latch ours ${g.key} theirs ${m.key}`); return out; };
let d = diff(); if (d.length) { console.log('different before the first frame:\n  ' + d.join('\n  ')); process.exit(1); }

let n = 0, pixels = 0, ended = '', lastCyc = c.cycles, lastOurs = g.cycles; const samples = [];
for (; n < frames; n++) {
  if (cheat) for (const z of [g.m, ram]) { if (z[0x5d]) z[0xb6] = z[0xb0]; if (z[0xb7] > 1) z[0xb7] = 1; }
  if (mode === 'random' && rand(3) === 0) {
    let k = POOL[rand(POOL.length)];
    const near = ram[0x5e] && u8(ram[0x72] - ram[0x62]) < 0x18, upright = ram[0x6f] !== 0;
    if (k === 'LEFT' && u8(0x30 - ram[0x52] + ram[0x62]) < 0x14) k = 'RIGHT';   // do not walk back off the cliff
    if (near && upright) k = 'SPACE';                                           // an opponent kills a player who is not in his stance
    else if (k === 'SPACE' && (near || rand(4))) k = 'RIGHT';
    else if (k === 'B' && near) k = 'W';
    const code = KEYS[k] ?? (k.charCodeAt(0) | 0x80); m.pressKey(code & 0x7f); g.key = code;
  }
  const res = g.step();
  m.step(); while (!topLevel() && !m.drive.motor && c.pc !== 0x0203 && c.pc !== 0x025e) m.step();
  if (res !== 'frame' || !topLevel()) { ended = `after ${n} frames ours says "${res}", the original ${topLevel() ? 'goes on' : c.pc === 0x025e || c.pc === 0x0203 ? 'is at its game-over screen' : m.drive.motor ? 'goes to the disk (next scene)' : 'is at $' + c.pc.toString(16)}`; break; }
  d = diff();
  if (d.length) { console.log(`frame ${n}: memory differs\n  ` + d.join('\n  ')); console.log('  state: level', ram[0xd0], 'opponent', ram[0x3e], 'player x', ram[0x62], 'view', ram[0x52]); process.exit(1); }
  if (!screens && n === 40) { screens = true; const e = diff(); if (e.length) { console.log('the screens differ after 40 frames:\n  ' + e.join('\n  ')); process.exit(1); } }
  if (timing) samples.push([c.cycles - lastCyc, g.cycles - lastOurs, g.flips.length, g.clicks.length]);
  lastCyc = c.cycles; lastOurs = g.cycles; g.flips.length = 0; g.clicks.length = 0;
}
console.log(`${n} frames identical to the original, byte for byte` + (ended ? `; ${ended}` : '') + `  (opponent ${ram[0x3e]}, player strength ${ram[0xb6]})`);
if (timing) {
  const tot = samples.reduce((a, s) => [a[0] + s[0], a[1] + s[1]], [0, 0]);
  console.log('time: the original', (tot[0] / CPU_HZ).toFixed(1), 's, ours', (tot[1] / CPU_HZ).toFixed(1), 's, ratio', (tot[1] / tot[0]).toFixed(4));
  const gap = samples.map((s) => s[0] - s[1]).sort((a, b) => a - b), plain = samples.filter((s) => s[2] === 1).map((s) => s[0] - s[1]).sort((a, b) => a - b);
  console.log('cycles per pass that ours does not account for: median', gap[gap.length >> 1], 'in ordinary passes', plain[plain.length >> 1], '10%..90%', plain[Math.floor(plain.length * 0.1)], plain[Math.floor(plain.length * 0.9)]);
  const err = samples.map((s) => Math.abs(s[1] - s[0]) / s[0]).sort((a, b) => a - b);
  console.log('error per pass: median', (err[err.length >> 1] * 100).toFixed(1) + '%', '90th percentile', (err[Math.floor(err.length * 0.9)] * 100).toFixed(1) + '%', 'worst', (err.at(-1) * 100).toFixed(1) + '%');
  if (opt('dump', '')) fs.writeFileSync(opt('dump', ''), JSON.stringify(samples));
}
