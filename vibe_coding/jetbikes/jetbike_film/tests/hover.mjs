// step the sim manually through takeoff + hover + start of path, print pitch dynamics
import { JetBike, makeWind } from '../js/physics.js';
import { FlightComputer, GAINS } from '../js/control.js';
if (process.env.G) for (const [k, v] of Object.entries(JSON.parse(process.env.G))) { if (Array.isArray(v)) v.forEach((x, i) => GAINS[k][i] = x); else GAINS[k] = v; }
import { buildMission, TL } from '../js/mission.js';
import { massProps, FUEL0, G } from '../js/vehicle.js';
import { qrot, len } from '../js/vmath.js';
const mission = buildMission(massProps(FUEL0).cg[1] - 0.012);
const bike = new JetBike(); const fc = new FlightComputer(); const wind = makeWind();
const dt = 0.0005; let cmd = fc.cmd; const tEnd = +(process.argv[2] || 30);
TL.liftStart.forEach((_, i) => bike.lift[i].start()); bike.main.start();
for (let k = 0; bike.t < tEnd; k++) {
  const t = bike.t;
  if (k % 10 === 0) {
    const ref = mission.sample(t);
    if (t < TL.liftoff) { const u = Math.min(1, Math.max(0, (t - TL.spoolUp) / 2.4)); cmd = fc.update(dt * 10, bike, ref, 'ground', 0.93 * bike.mp.m * G * u); }
    else cmd = fc.update(dt * 10, bike, ref, 'flight');
  }
  bike.step(dt, cmd, wind);
  if (k % (+process.env.EV || 200) === 0 && t > 10) {
    const fwd = qrot(bike.q, [1, 0, 0]); const r = qrot(bike.q, [0, 0, 1]);
    console.log(`t=${t.toFixed(2)} y=${bike.p[1].toFixed(2)} x=${bike.p[0].toFixed(1)} pitch=${(Math.asin(fwd[1]) * 57.3).toFixed(1)} bank=${(Math.asin(-r[1]) * 57.3).toFixed(1)} w=${bike.w.map(v => v.toFixed(2))} e=${fc.telemetry.e?.map(v => v.toFixed(3))} T=${bike.lift.map(e => e.T.toFixed(0))} cmd=${cmd.lift.map(v => v.toFixed(0))} vanes=${cmd.vanes.map(v => v.map(a => (a * 57.3).toFixed(0)).join('/'))} main=${bike.main.T.toFixed(0)}`);
  }
}
