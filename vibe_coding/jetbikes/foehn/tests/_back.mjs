// how long to get back to a missed ring? fly through a point at V0, then come back through it.
import { loadWorld } from './util.mjs';
import { Sim } from '../js/sim/sim.js';
import { yprOf, len } from '../js/sim/vmath.js';
loadWorld();
const wrap = (a) => Math.atan2(Math.sin(a), Math.cos(a));
const alt = +(process.argv[2] || 1100);
for (const strat of ['turn', 'brake+turn']) for (const V0 of [20, 30, 40, 55, 75]) {
  const s = new Sim(); s.reset([2500, alt, 1500], 0, V0);
  for (let i = 0; i < 360; i++) s.tick({ turn: 0, pitch: 0, vt: V0, noFloor: true });
  const b = s.bike, p0 = [...b.p]; let t = 0, vt = V0, far = 0, done = 0, lost = 0;
  while (t < 60) {
    const dx = p0[0] - b.p[0], dz = p0[2] - b.p[2], d = Math.hypot(dx, dz);
    far = Math.max(far, d); lost = Math.min(lost, b.p[1] - p0[1]);
    if (t > 2 && d < 8) { done = 1; break; }
    const V = len(b.v), hd = V > 3 ? Math.atan2(-b.v[2], b.v[0]) : yprOf(b.q)[0];
    const err = wrap(Math.atan2(-dz, dx) - hd);          // + = target to the left
    const turn = t < 0.3 ? 1 : Math.max(-1, Math.min(1, -err * 3));   // steer at the point (turn + = right)
    let brake = false;
    if (strat === 'brake+turn') { brake = Math.abs(err) > 1.0 && t > 0.3 || t < 0.3; if (brake) vt = Math.max(8, Math.min(vt, V) - 30 / 120); else vt = Math.min(V0, vt + 20 / 120); }
    s.tick({ turn, pitch: 0, vt, brake, noFloor: true }); t += 1 / 120;
  }
  console.log(`${strat.padEnd(10)} V0=${(V0 * 3.6).toFixed(0).padStart(3)} km/h  back in ${done ? t.toFixed(1) + 's' : 'NEVER'}  farthest=${far.toFixed(0)}m  height lost=${(-lost).toFixed(1)}m`);
}
