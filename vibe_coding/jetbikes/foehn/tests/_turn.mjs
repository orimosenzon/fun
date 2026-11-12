import { loadWorld, deg } from './util.mjs';
import { Sim } from '../js/sim/sim.js';
import { G } from '../js/sim/bike.js';
import { yprOf, len } from '../js/sim/vmath.js';
loadWorld();
const wrap = (a) => Math.atan2(Math.sin(a), Math.cos(a));
const mode = process.argv[2] || 'hold';
for (const V0 of [0, 10, 20, 30, 40, 55, 75, 100]) {
  const s = new Sim(); s.reset([2500, +(process.argv[3] || 2400), 1500], 0, V0 || 0.01);
  for (let i = 0; i < 360; i++) s.tick({ turn: 0, pitch: 0, vt: V0, noFloor: true });
  const b = s.bike, p0 = [...b.p], h0 = Math.atan2(-b.v[2], b.v[0]), yaw0 = yprOf(b.q)[0];
  let ang = 0, last = V0 > 3 ? h0 : yaw0, t = 0, t90 = 0, bankMax = 0, nMax = 0, vmin = 1e9, xmin = 1e9, xmax = -1e9, zmin = 1e9, zmax = -1e9, satN = 0, n = 0;
  const inp = { turn: 1, pitch: 0, vt: V0, noFloor: true, brake: mode === 'brake' };
  while (Math.abs(ang) < Math.PI && t < 40) {
    s.tick(inp); t += 1 / 120; n++;
    const V = len(b.v), h = V > 3 ? Math.atan2(-b.v[2], b.v[0]) : yprOf(b.q)[0];
    ang += wrap(h - last); last = h;
    if (!t90 && Math.abs(ang) > Math.PI / 2) t90 = t;
    bankMax = Math.max(bankMax, Math.abs(yprOf(b.q)[2]));
    nMax = Math.max(nMax, Math.hypot(b.acc[0], b.acc[1] + G, b.acc[2]) / G);
    vmin = Math.min(vmin, V);
    xmin = Math.min(xmin, b.p[0]); xmax = Math.max(xmax, b.p[0]); zmin = Math.min(zmin, b.p[2]); zmax = Math.max(zmax, b.p[2]);
  }
  console.log(`V0=${(V0 * 3.6).toFixed(0).padStart(3)} km/h  t90=${t90.toFixed(1)}s t180=${t.toFixed(1)}s  width=${Math.max(xmax - xmin, zmax - zmin).toFixed(0)}m  bankMax=${deg(bankMax).toFixed(0)} nMax=${nMax.toFixed(2)} Vend=${(len(b.v) * 3.6).toFixed(0)} Vmin=${(vmin * 3.6).toFixed(0)} dy=${(b.p[1] - p0[1]).toFixed(1)} wb=${s.fcs.tele.wb.toFixed(2)}`);
}
