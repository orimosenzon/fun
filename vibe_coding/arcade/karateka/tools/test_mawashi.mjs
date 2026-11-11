// The mawashi-geri extension, played on the real game: walk up to the first guard, kick with E,
// and check the pose sequence, the two held frames and the double damage.
// Also checks that a game in which E is never pressed is untouched by the extension.
import { startGame } from './lab.mjs';
import { installMawashi } from '../src/ext/mawashi.js';

let failed = 0;
const check = (ok, what) => { console.log((ok ? 'ok   ' : 'FAIL ') + what); if (!ok) failed++; };
const frameHash = (m) => { let h = 0; const p = m.hgrPage(); for (let i = 0; i < p.length; i++) h = (h * 31 + p[i]) | 0; return h; };

function play(withExt, keys, shots) {
  const m = startGame(); if (withExt) installMawashi(m);
  const r = m.ram, log = [], draws = []; let last = -1, lastFlip = m.video.page2, n = 0;
  m.sec(3); m.press('K'); m.sec(0.5); m.press('SPACE'); m.sec(1.5);
  const run = (s) => { const end = m.cpu.cycles + s * 1020484;
    while (m.cpu.cycles < end) { for (let i = 0; i < 500; i++) { if (m.cpu.pc === 0x6811 && r[0x2c] < 0x80) draws.push(r[0x20]); m.step(); }
      if (m.video.page2 !== lastFlip) { lastFlip = m.video.page2; log.push({ pose: r[0x60], hp: r[0xb7], hash: frameHash(m), c: m.cpu.cycles }); if (shots && [1, 3].includes(r[0x60]) && r[0x60] !== last) m.shot(`${shots}_${n++}_pose${r[0x60]}`); last = r[0x60]; } } };
  for (const k of keys) { if (typeof k === 'function') k(m, run); else if (k) m.press(k); run(0.4); }
  return { m, log, draws };
}
// walk towards the guard until he is within kicking distance, then run `kick`
const approach = (kick) => (m, run) => { for (let i = 0; i < 60 && !(m.ram[0x5e] && m.ram[0x72] - m.ram[0x62] <= 10); i++) { m.press('RIGHT'); run(0.4); } m.walked = true; for (let i = 0; i < 12; i++) { m.press(kick); run(1.2); if (m.ram[0x72] - m.ram[0x62] > 9) { m.press('RIGHT'); run(0.4); } } };

// 2. ordinary middle kick against the guard, for comparison
const normal = play(false, [approach('S')]);
// 3. the mawashi
const maw = play(true, [approach('E')], 'mawashi');
// 1. E never pressed: the extension must not change a frame or a cycle
{ const b = play(true, [approach('S')]), a = normal;
  check(a.log.length === b.log.length && a.log.every((f, i) => f.hash === b.log[i].hash && f.c === b.log[i].c), `no E pressed: ${a.log.length} frames identical, same cycle counts`); }
const seq = (log) => log.map((f) => f.pose.toString(16)).join(' ');
const runs = (log, pose) => { const out = []; let n = 0; for (const f of log) { if (f === pose) n++; else if (n) { out.push(n); n = 0; } } return out; };
const drops = (log) => { const out = []; for (let i = 1; i < log.length; i++) if (log[i].hp < log[i - 1].hp) out.push(log[i - 1].hp - log[i].hp); return out; };
console.log('middle kick, kick frames per kick:', runs(normal.draws, 3).join(','), ' damage per hit:', drops(normal.log).join(','));
console.log('mawashi,     kick frames per kick:', runs(maw.draws, 3).join(','), ' damage per hit:', drops(maw.log).join(','), ' state:', JSON.stringify(maw.m.mawashi));
check(maw.m.mawashi.kicks > 0, 'the game accepted the kick');
check(runs(maw.draws, 3).length > 0 && runs(maw.draws, 3).every((n) => n === 2), 'kick pose is held for two frames');
check(runs(normal.draws, 3).every((n) => n === 1), 'ordinary kick is one frame');
check(maw.m.mawashi.hits > 0 && drops(maw.log).some((d) => d === 2), 'a hit takes two points');
check(drops(normal.log).every((d) => d === 1), 'ordinary hit takes one point');
check(!maw.m.mawashi.active || maw.log.at(-1).pose !== 6, 'extension is idle after the kick');
if (process.argv[2]) console.log(maw.draws.slice(-80).map((x) => x.toString(16)).join(' '));
process.exit(failed ? 1 : 0);
