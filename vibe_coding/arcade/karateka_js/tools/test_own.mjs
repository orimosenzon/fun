// The game on its own data, with no disk anywhere: it must play the whole scene, and every decision
// in it (who stands where, in which pose, with how much strength left) must be the same as when the
// same keys are played on the reference memory. Only the pictures and the sounds are different.
// usage: node tools/test_own.mjs        (the comparison needs the reference disk; without it only the first part runs)
import fs from 'node:fs';
import { Game } from '../src/index.js';
import { buildMemory } from '../src/loader.js';
import { botPlay, memoryFor } from './lib.mjs';
let failed = 0;
const check = (ok, what) => { console.log((ok ? 'ok   ' : 'FAIL ') + what); if (!ok) failed++; };
const a = botPlay(new Game(buildMemory(1)), { seed: 2, cheat: true, frames: 9000 });
check(a.res === 'next', 'a player who cannot lose reaches the gate: ' + a.frames + ' passes');
const b = botPlay(new Game(buildMemory(1)), { seed: 1, frames: 9000, end: true });
check(b.res === 'over', 'a player who can lose is beaten: ' + b.frames + ' passes');
// not compared: where the drawing routine keeps its place ($23 $80 $81), the speaker's state ($9B), the tune player's ($F6 up),
// and four bytes the loader of the original left behind and the game never reads
const SCRATCH = [0x23, 0x80, 0x81, 0x9b, 0x3d, 0xd2, 0xdc, 0xe6];
let ref = true; try { memoryFor(false); } catch { ref = false; }
if (ref) for (const [seed, cheat] of [[2, true], [3, true], [1, false], [5, false]]) {
  const trace = (ours) => { const g = new Game(memoryFor(ours)), rows = []; const r = botPlay(g, { seed, cheat, frames: 9000, each: (g) => { const v = g.m.slice(0x20, 0xf6); for (const a of SCRATCH) v[a - 0x20] = 0; rows.push(Buffer.from(v).toString('latin1')); } }); return { rows, r, cycles: g.cycles }; };
  const x = trace(false), y = trace(true);
  let n = 0; while (n < x.rows.length && n < y.rows.length && x.rows[n] === y.rows[n]) n++;
  check(n === x.rows.length && n === y.rows.length && x.r.res === y.r.res, `seed ${seed}${cheat ? ' (cannot lose)' : ''}: the same game on both, ${n} of ${x.rows.length} passes, ends "${y.r.res}"; time ${(y.cycles / x.cycles).toFixed(3)} of the reference`);
}
process.exit(failed ? 1 : 0);
