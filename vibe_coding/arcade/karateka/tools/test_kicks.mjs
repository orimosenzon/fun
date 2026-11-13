// Yoko-geri (Y) and ushiro-geri (U), Yaniv Waldman's idea, played on the real game against the first guard:
// the frames shown, how long each kick takes, what a hit costs the guard, and the side kick's longer reach.
import { startGame } from './lab.mjs';
import { installMawashi } from '../src/ext/mawashi.js';
import { installFlip } from '../src/ext/flip.js';

let failed = 0;
const check = (ok, what) => { console.log((ok ? 'ok   ' : 'FAIL ') + what); if (!ok) failed++; };

// fight the guard with one key from a given distance; returns what happened
function fight(key, gap, rounds = 12) {
  const m = startGame(); installMawashi(m); installFlip(m);
  const r = m.ram, draws = [], drops = []; let prev = -1, hitsFar = 0;
  const run = (s) => { const end = m.cpu.cycles + s * 1020484;
    while (m.cpu.cycles < end) { if (m.cpu.pc === 0x6811 && r[0x2c] < 0x80) draws.push(r[0x20]); m.step(); } };
  run(3); m.press('K'); run(0.5); m.press('SPACE'); run(3);
  const dist = () => r[0x72] - r[0x62];
  for (let i = 0; i < rounds && r[0x5e]; i++) {
    for (let j = 0; j < 30 && r[0x5e] && dist() !== gap; j++) { m.press(dist() > gap ? 'RIGHT' : 'LEFT'); run(0.4); }
    const hp = r[0xb7], g = dist(); m.press(key); run(1.6);
    if (r[0xb7] < hp) { drops.push(hp - r[0xb7]); if (g >= 11) hitsFar++; }
  }
  const runs = []; let n = 0; for (const p of draws) { if (p === 3) n++; else if (n) { runs.push(n); n = 0; } }
  const chambers = []; n = 0; for (const p of draws) { if (p === 1) n++; else if (n) { chambers.push(n); n = 0; } }
  return { m, drops, runs, chambers, hitsFar };
}

const yoko = fight('Y', 9), ushiro = fight('U', 9);
console.log('yoko:   kick frames', yoko.runs.join(','), ' damage', yoko.drops.join(','), ' kicks', yoko.m.mawashi.kicks);
console.log('ushiro: kick frames', ushiro.runs.join(','), ' chamber frames', ushiro.chambers.join(','), ' damage', ushiro.drops.join(','), ' kicks', ushiro.m.mawashi.kicks);
check(yoko.m.mawashi.kicks > 0 && yoko.m.mawashi.kind === 0x59, 'the game accepted the side kick');
check(yoko.runs.length > 0 && yoko.runs.every((n) => n === 2), 'side kick: the thrust is held for two frames');
check(yoko.drops.length > 0 && yoko.drops.every((d) => d === 1), 'side kick: a hit takes one point');
check(ushiro.m.mawashi.kicks > 0 && ushiro.m.mawashi.kind === 0x55, 'the game accepted the back kick');
check(ushiro.runs.some((n) => n === 2) && ushiro.runs.every((n) => n <= 2), 'back kick: the thrust is held for two frames (unless he is hit first)');
check(ushiro.chambers.some((n) => n >= 3), 'back kick: three frames of turning before the thrust');
check(ushiro.drops.length > 0 && ushiro.drops.some((d) => d === 3) && ushiro.drops.every((d) => d <= 3), 'back kick: a hit takes three points');

// reach: from one step further away the ordinary middle kick misses and the side kick lands
const farS = fight('S', 11, 8), farY = fight('Y', 11, 8);
console.log('from a step further: middle kick hits', farS.hitsFar, ' side kick hits', farY.hitsFar, ' judged far', farY.m.mawashi.far);
check(farS.hitsFar === 0, 'the ordinary kick does not reach from a step further');
check(farY.hitsFar > 0, 'the side kick does');
process.exit(failed ? 1 : 0);
