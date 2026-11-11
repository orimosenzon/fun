// circling in a thermal: node tests/thermal.mjs
import { initWorld, thermals, thermalCentre, windAt } from '../js/sim/world.js';
import { Bird } from '../js/sim/bird.js';
import { wrapPi, clamp } from '../js/sim/vmath.js';
initWorld();
const th = thermals[0];
for (const [roll, pitch, smart] of [[0.45, 0.4, 0], [0.45, 0.4, 1], [0.6, 0.5, 1], [0.3, 0.3, 1]]) {
  const b = new Bird(); const y0 = th.y0 + 150; const [cx, cz] = thermalCentre(th, y0);
  b.placeAir([cx - 30, y0, cz], 0, 13);
  const dt = 1 / 60; let dsum = 0, n = 0, dmax = 0;
  for (let t = 0; t < 120; t += dt) {
    const [x, z] = thermalCentre(th, b.p[1]); const dx = x - b.p[0], dz = z - b.p[2], d = Math.hypot(dx, dz);
    let r = roll;
    // a soaring pilot's trick: widen the turn while flying towards the core, tighten when flying away
    if (smart) { const toC = wrapPi(Math.atan2(-dz, dx) - Math.atan2(-b.v[2], b.v[0])); r = roll * clamp(1 - 0.9 * Math.cos(toC) * Math.min(1, d / 60), 0.15, 1.6); }
    b.step(dt, { pitch, roll: r, fwd: 0, flap: false, tuck: false, jump: false }, t);
    dsum += d; n++; dmax = Math.max(dmax, d);
  }
  console.log(`roll=${roll} pitch=${pitch} smart=${smart}: climbed ${(b.p[1] - y0).toFixed(0)} m in 120 s (${((b.p[1] - y0) / 120).toFixed(2)} m/s), mean dist from core ${(dsum / n).toFixed(0)} m, max ${dmax.toFixed(0)}, V=${(b.airspeed * 3.6).toFixed(0)} bank=${(b.bank * 57.3).toFixed(0)}`);
}
console.log('core lift at +150/+400/+800:', [150, 400, 800].map((h) => { const [x, z] = thermalCentre(th, th.y0 + h); return windAt(x, th.y0 + h, z, 0)[1].toFixed(1); }).join(' '), 'wind there', windAt(...((p) => [p[0], th.y0 + 400, p[1]])(thermalCentre(th, th.y0 + 400)), 0).map((v) => v.toFixed(1)).join(','));
