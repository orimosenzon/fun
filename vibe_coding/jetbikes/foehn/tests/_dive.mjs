// Dive response: full nose-down stick from level flight. node tests/_dive.mjs
import { loadWorld, deg } from './util.mjs';
import { Sim } from '../js/sim/sim.js';
import { FEEL } from '../js/sim/fcs.js';
import { yprOf, len } from '../js/sim/vmath.js';
loadWorld();
for (const a of process.argv.slice(2)) { const [k, v] = a.split('='); if (k in FEEL) FEEL[k] = +v; }
for (const V0 of [10, 30, 55, 75]) {
  const s = new Sim(); s.reset([2500, 2400, 1500], 0, V0);
  for (let i = 0; i < 480; i++) s.tick({ turn: 0, pitch: 0, vt: V0, noFloor: true });
  const b = s.bike, y0 = b.p[1];
  let t = 0, t50 = 0, t100 = 0, t05 = 0, sink = 0, nose = 0, noseAt05 = 0, vmax = 0;
  while (t < 8) {
    s.tick({ turn: 0, pitch: -1, vt: V0, noFloor: true }); t += 1 / 120;
    const dy = y0 - b.p[1];
    if (!t50 && dy > 50) t50 = t; if (!t100 && dy > 100) t100 = t;
    sink = Math.max(sink, -b.v[1]); nose = Math.min(nose, yprOf(b.q)[1]); vmax = Math.max(vmax, len(b.v));
    if (!t05 && t >= 0.5) { t05 = 1; noseAt05 = yprOf(b.q)[1]; }
  }
  console.log(`V0=${(V0 * 3.6).toFixed(0).padStart(3)} km/h  50m in ${t50.toFixed(2)}s  100m in ${t100.toFixed(2)}s  sinkMax=${sink.toFixed(1)} m/s  nose@0.5s=${deg(noseAt05).toFixed(0)}° noseMin=${deg(nose).toFixed(0)}°  Vmax=${(vmax*3.6).toFixed(0)} km/h`);
}
