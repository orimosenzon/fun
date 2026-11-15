// Turn response with FEEL overrides. node tests/_feel.mjs key=value ...   (e.g. rollRate=2 yawAcc=3)
import { loadWorld, deg } from './util.mjs';
import { Sim } from '../js/sim/sim.js';
import { G } from '../js/sim/bike.js';
import { FEEL } from '../js/sim/fcs.js';
import { yprOf, len } from '../js/sim/vmath.js';
loadWorld();
for (const a of process.argv.slice(2)) { const [k, v] = a.split('='); if (k in FEEL) FEEL[k] = +v; }
const wrap = (a) => Math.atan2(Math.sin(a), Math.cos(a));
for (const V0 of [0, 10, 20, 30, 40, 55, 75, 100]) {
  const s = new Sim(); s.reset([2500, 2400, 1500], 0, V0 || 0.01);
  for (let i = 0; i < 360; i++) s.tick({ turn: 0, pitch: 0, vt: V0, noFloor: true });
  const b = s.bike, p0 = [...b.p], h0 = Math.atan2(-b.v[2], b.v[0]), yaw0 = yprOf(b.q)[0];
  let ang = 0, last = V0 > 3 ? h0 : yaw0, t = 0, t30 = 0, t90 = 0, bankMax = 0, nMax = 0, zmin = 1e9, zmax = -1e9, xmin = 1e9, xmax = -1e9, tb = 0, sdMax = 0;
  while (Math.abs(ang) < Math.PI && t < 30) {
    s.tick({ turn: 1, pitch: 0, vt: V0, noFloor: true }); t += 1 / 120;
    const V = len(b.v), h = V > 3 ? Math.atan2(-b.v[2], b.v[0]) : yprOf(b.q)[0];
    ang += wrap(h - last); last = h;
    if (!t30 && Math.abs(ang) > Math.PI / 6) t30 = t;
    if (!t90 && Math.abs(ang) > Math.PI / 2) t90 = t;
    const bank = Math.abs(yprOf(b.q)[2]); if (!tb && bank > 0.8) tb = t;
    bankMax = Math.max(bankMax, bank); sdMax = Math.max(sdMax, Math.abs(b.sd[0]) + Math.abs(b.sd[1]));
    nMax = Math.max(nMax, Math.hypot(b.acc[0], b.acc[1] + G, b.acc[2]) / G);
    xmin = Math.min(xmin, b.p[0]); xmax = Math.max(xmax, b.p[0]); zmin = Math.min(zmin, b.p[2]); zmax = Math.max(zmax, b.p[2]);
  }
  console.log(`V0=${(V0 * 3.6).toFixed(0).padStart(3)} km/h  t30=${t30.toFixed(2)} t90=${t90.toFixed(2)} t180=${t.toFixed(2)}s  width=${Math.max(xmax - xmin, zmax - zmin).toFixed(0)}m  bank>46deg at ${tb.toFixed(2)}s bankMax=${deg(bankMax).toFixed(0)} nMax=${nMax.toFixed(2)} dy=${(b.p[1] - p0[1]).toFixed(1)} side=${sdMax.toFixed(0)}N`);
}
