// The crane kick (C, Arik Shemesh's idea) on the real game: the leap, the scenery
// left as it was, and against the guard: the kick lands, takes two points, and nothing hits in the air.
import { startGame } from './lab.mjs';
import { installMawashi } from '../src/ext/mawashi.js';
import { installFlip } from '../src/ext/flip.js';
let failed = 0;
const check = (ok, what) => { console.log((ok ? 'ok   ' : 'FAIL ') + what); if (!ok) failed++; };
function game() {
  const m = startGame(); installMawashi(m); installFlip(m);
  m.run = (s) => { const end = m.cpu.cycles + s * 1020484; while (m.cpu.cycles < end) m.runTo(m.cpu.cycles + 1500); };
  m.run(3); m.press('K'); m.run(0.5); m.press('SPACE'); m.run(3);
  return m;
}
const m = game(), r = m.ram;
const sky = () => { let h = 0; for (const base of [0x2000, 0x4000]) for (let y = 0; y < 100; y++) { const b = base + (((y & 7) << 10) | (((y >> 3) & 7) << 7) | ((y >> 6) * 40)); for (let i = 4; i < 31; i++) h = (h * 31 + r[b + i]) | 0; } return h; };
const world = () => r[0x62] - r[0x52];
const crane = () => { const n = m.flip.cranes; for (let i = 0; i < 20 && m.flip.cranes === n; i++) { m.press('C'); m.run(0.3); } let painted = false, k = 0; while (m.flip.frame >= 0 && k++ < 400) { m.run(0.05); if (m.flip.under[32].length || m.flip.under[64].length) painted = true; } return painted; };

const before = sky(), w = world(), painted = crane();
m.run(1);
check(m.flip.cranes === 1 && m.flip.frame === -1 && m.flip.flips === 0, 'crane kick starts and ends');
check(world() - w >= 0 && world() - w <= 2, `it hops at most two columns forward (${world() - w} columns in the world)`);
check(sky() === before, 'it leaves the sky and the mountain untouched');

// to the guard
for (let i = 0; i < 60 && !(r[0x5e] && r[0x72] - r[0x62] <= 11); i++) { m.press('RIGHT'); m.run(0.4); }
let prevP = r[0xb6], prevE = r[0xb7], airHits = 0, minGap = 99; const drops = [];
m.onWrite = (a, v) => {
  if (a === 0xb6) { if (v < prevP && m.flip.inAir) airHits++; prevP = v; }
};
for (let i = 0; i < 16 && r[0x5e]; i++) {
  const e0 = r[0xb7], h0 = m.flip.craneHits;
  m.press('C'); for (let j = 0; j < 14; j++) { m.run(0.12); minGap = Math.min(minGap, r[0x72] - r[0x62]); }
  if (m.flip.craneHits > h0) drops.push(e0 - r[0xb7]);
  if (r[0x72] - r[0x62] > 11) { m.press('RIGHT'); m.run(0.4); }
}
console.log('  cranes', m.flip.cranes, 'hits', m.flip.craneHits, 'damage per hit', drops.join(','), 'dodged', m.flip.dodged, 'min gap', minGap, 'enemy alive', r[0x5e], 'health', r[0xb7], 'player', r[0xb6]);
check(m.flip.craneHits > 0, 'the kick lands on the guard');
check(drops.length > 0 && drops.every((d) => d === 2 || d === 1) && drops.some((d) => d === 2), 'a clean hit takes two points');
check(airHits === 0, 'no damage taken in the air');
check(minGap >= 5, 'never passes through the opponent');
process.exit(failed ? 1 : 0);
