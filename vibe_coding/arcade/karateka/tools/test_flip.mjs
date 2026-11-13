// The somersault extension on the real game: flips in the open, the floor routine, next to the guard,
// and under attack; also that nothing changes when F and D are never pressed.
import { startGame } from './lab.mjs';
import { installMawashi } from '../src/ext/mawashi.js';
import { installFlip } from '../src/ext/flip.js';
let failed = 0;
const check = (ok, what) => { console.log((ok ? 'ok   ' : 'FAIL ') + what); if (!ok) failed++; };
function game(ext) {
  const m = startGame(); if (ext) { installMawashi(m); installFlip(m); }
  const r = m.ram; m.frames = []; let lastFlip = m.video.page2;
  m.run = (s) => { const end = m.cpu.cycles + s * 1020484; while (m.cpu.cycles < end) { m.runTo(m.cpu.cycles + 1500);
    if (m.video.page2 !== lastFlip) { lastFlip = m.video.page2; let h = 0; const p = m.hgrPage(); for (let i = 0; i < p.length; i++) h = (h * 31 + p[i]) | 0; m.frames.push(h + ':' + m.cpu.cycles); if (m.shots && m.flip.frame >= 0) m.shot(`${m.shots}_${m.flip.flips}_${m.flip.frame}`); } } };
  m.run(3); m.press('K'); m.run(0.5); m.press('SPACE'); m.run(3);
  return m;
}
const script = (m) => { for (const k of ['RIGHT', 'RIGHT', 'S', 'RIGHT', 'W', 'RIGHT', 'RIGHT', 'X', 'A', 'RIGHT', 'RIGHT', 'RIGHT', 'S', 'S', 'Q']) { m.press(k); m.run(0.5); } };
{ const a = game(false), b = game(true); script(a); script(b);
  check(a.frames.length === b.frames.length && a.frames.every((f, i) => f === b.frames[i]), `F and D not pressed: ${a.frames.length} frames identical, same timing`); }

const m = game(true), r = m.ram; m.shots = 'flip';
const x0 = r[0x62], hp0 = r[0xb6];
const flip = (k) => { const n = m.flip.flips; for (let i = 0; i < 20 && m.flip.flips === n; i++) { m.press(k); m.run(0.4); } const xs = r[0x62]; while (m.flip.frame >= 0) m.run(0.05); return r[0x62] - xs; };
const world = () => r[0x62] - r[0x52];
let w = world(); flip('F');
check(m.flip.flips === 1 && m.flip.frame === -1, 'forward flip starts and ends');
check(world() - w > 0, `forward flip travels forward (${world() - w} columns in the world)`);
w = world(); flip('D');
check(m.flip.flips === 2 && m.flip.frame === -1, 'back flip starts and ends');
check(world() - w < 0, `back flip travels back (${world() - w} columns in the world)`);
// the floor routine: it goes high above the strip the game repaints, and must leave the scenery as it found it
for (const [k, name] of [['D', 'back'], ['F', 'front']]) {
  const s = game(true), q = s.ram; s.flip.always = true;
  const sky = () => { let h = 0; for (const base of [0x2000, 0x4000]) for (let y = 0; y < 100; y++) { const b = base + (((y & 7) << 10) | (((y >> 3) & 7) << 7) | ((y >> 6) * 40)); for (let i = 4; i < 31; i++) h = (h * 31 + q[b + i]) | 0; } return h; };   // the part of the sky the game does not scroll
  const before = sky(); let painted = false, n = 0;
  for (let i = 0; i < 20 && !s.flip.flips; i++) { s.press(k); s.run(0.3); }
  while (s.flip.frame >= 0 && n++ < 400) { s.run(0.05); if (sky() !== before) painted = true; }
  s.run(1);
  check(s.flip.shows === 1 && s.flip.frame === -1, `${name} floor routine starts and ends`);
  check(painted, `${name} floor routine rises above the fighters' strip`);
  check(sky() === before, `${name} floor routine leaves the sky and the mountain untouched`);
}
// to the guard: flip forward must not pass him, and attacks miss in the air
for (let i = 0; i < 60 && !(r[0x5e] && r[0x72] - r[0x62] <= 10); i++) { m.press('RIGHT'); m.run(0.4); }
let minGap = 99, hits = 0, ground = 0, hp = r[0xb6];
let prev = r[0xb6]; m.onWrite = (a, v) => { if (a !== 0xb6) return; const drop = v < prev; prev = v; if (drop) { if (m.flip.inAir) hits++; else ground++; } };
 console.log('  before the fight: alive', r[0x5d], 'health', hp, 'enemy', r[0x5e], r[0xb7]);
for (let i = 0; i < 14; i++) { m.press(i % 3 === 2 ? 'D' : 'F'); for (let j = 0; j < 12; j++) { m.run(0.12); minGap = Math.min(minGap, r[0x72] - r[0x62]); } }
console.log('  flips', m.flip.flips, 'dodged', m.flip.dodged, 'min gap', minGap, 'player health', hp0, '->', r[0xb6], 'alive', r[0x5d]);
console.log('  hits taken on the ground:', ground);
check(minGap >= 5, 'never passes through the opponent');
check(hits === 0, 'no damage taken in the air');
check(m.flip.flips > 6, 'flips keep working during a fight');
process.exit(failed ? 1 : 0);
