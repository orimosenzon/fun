// How hard is each wind level? Hands off in level flight, and the demo rider on a course.
// node tests/wind.mjs [event]
import { loadWorld, deg } from './util.mjs';
import { Sim } from '../js/sim/sim.js';
import { windFor, WIND_LEVELS } from '../js/sim/wind.js';
import { Game } from '../js/sim/game.js';
import { Autopilot } from '../js/sim/autopilot.js';
import { buildScenery } from '../js/sim/scenery.js';
import { yprOf, len } from '../js/sim/vmath.js';
loadWorld(); buildScenery();
console.log('hands off for 40 s (high over the lake):');
for (const V0 of [0, 30, 70]) for (let lv = 0; lv < 4; lv++) {
  let sy = 0, sz = 0, bank = 0, wmax = 0, amax = 0, n = 0, lost = 0;
  for (const seed of [1, 2, 3]) {
    const s = new Sim(windFor({ speed: 4, gust: 1, heading: 1.2 }, lv, seed)); s.reset([1500, 2600, 1500], 0, V0 || 0.01);
    for (let i = 0; i < 240; i++) s.tick({ turn: 0, pitch: 0, vt: V0, noFloor: true });
    const b = s.bike, y0 = b.p[1], z0 = b.p[2], x0 = b.p[0];
    for (let i = 0; i < 4800; i++) {
      s.tick({ turn: 0, pitch: 0, vt: V0, noFloor: true });
      const dz = V0 ? b.p[2] - z0 : Math.hypot(b.p[0] - x0, b.p[2] - z0);
      sy += (b.p[1] - y0) ** 2; sz = Math.max(sz, Math.abs(dz)); n++;
      bank = Math.max(bank, Math.abs(yprOf(b.q)[2])); wmax = Math.max(wmax, len(s.windW)); amax = Math.max(amax, Math.hypot(b.acc[0], b.acc[1], b.acc[2]));
      if (len(b.w) > 6.5) lost++;
    }
  }
  console.log(`  ${String(V0 * 3.6).padStart(3)} km/h ${WIND_LEVELS[lv].padEnd(7)} strongest wind ${(wmax * 3.6).toFixed(0).padStart(3)} km/h  pushed aside up to ${sz.toFixed(0).padStart(3)} m  height rms ${Math.sqrt(sy / n).toFixed(2)} m  bank up to ${deg(bank).toFixed(0)} deg  jolts up to ${(amax / 9.81).toFixed(2)} g ${lost ? 'LOST' : ''}`);
}
const id = process.argv[2] || 'lake';
console.log(`demo rider on "${id}":`);
for (let lv = 0; lv < 4; lv++) for (const seed of [11, 12]) {
  const g = new Game(); g.windLevel = lv; g.windSeed = seed; g.load(id);
  const ap = new Autopilot(g);
  while (g.state !== 'finished' && g.total < 500) { g.update(1 / 60, ap.update()); for (const e of g.events) if (e.type === 'respawn') ap.reset(); g.events.length = 0; }
  console.log(`  ${WIND_LEVELS[lv].padEnd(7)} seed ${seed}: ${g.state === 'finished' ? 'finished' : 'DNF'} in ${g.total.toFixed(1)} s, crashes ${g.crashes}, missed rings ${g.misses}`);
}
