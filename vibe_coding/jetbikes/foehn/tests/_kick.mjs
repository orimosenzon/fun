// The side kick: hold the side key for one second in level flight. node tests/_kick.mjs
import { loadWorld, deg } from './util.mjs';
import { Sim } from '../js/sim/sim.js';
import { G } from '../js/sim/bike.js';
import { yprOf, len } from '../js/sim/vmath.js';
loadWorld();
for (const V0 of [0, 8, 20, 40, 60, 90]) {
  const s = new Sim(); s.reset([2500, 2400, 1500], 0, V0 || 0.01);
  for (let i = 0; i < 360; i++) s.tick({ turn: 0, pitch: 0, vt: V0, noFloor: true });
  const b = s.bike, z0 = b.p[2], y0 = b.p[1], h0 = b.heat; let aMax = 0, sd = 0, t50 = 0, bank = 0;
  for (let i = 0; i < 120; i++) { s.tick({ turn: 0, pitch: 0, strafe: 1, vt: V0, noFloor: true }); aMax = Math.max(aMax, b.acc[2]); sd = Math.max(sd, b.sd[0] + b.sd[1]); bank = Math.max(bank, Math.abs(yprOf(b.q)[2])); if (!t50 && b.acc[2] > 5) t50 = (i + 1) / 120; }
  const z1 = b.p[2] - z0, vz = b.v[2], heat = b.heat - h0, yaw1 = yprOf(b.q)[0];
  for (let i = 0; i < 240; i++) s.tick({ turn: 0, pitch: 0, strafe: 0, vt: V0, noFloor: true });
  console.log(`V0=${(V0 * 3.6).toFixed(0).padStart(3)} km/h  1 s of kick: ${z1.toFixed(1)} m aside, side speed ${vz.toFixed(1)} m/s, peak ${(aMax / G).toFixed(2)} g (5 m/s2 after ${t50.toFixed(2)} s), side jets ${sd.toFixed(0)} N, bank ${deg(bank).toFixed(0)} deg, heat +${(heat * 100).toFixed(0)}%, nose ${deg(yaw1).toFixed(1)} deg | 2 s later: course ${deg(Math.atan2(-b.v[2], b.v[0])).toFixed(1)} deg, nose ${deg(yprOf(b.q)[0]).toFixed(1)}, dy=${(b.p[1] - y0).toFixed(1)} V=${(len(b.v) * 3.6).toFixed(0)}`);
}
