import { initWorld } from '../js/sim/world.js';
import { Bird } from '../js/sim/bird.js';
initWorld();
function run(tune, pitch = 0.2) {
  const b = new Bird(); b.windOn = false; b.placeAir([-3600, 900, -2500], 0, 13); Object.assign(b.tune, tune);
  const dt = 1 / 60; let n = 0, sv = 0, sp = 0, so = 0, sV = 0, pk = 0, gmin = 9, gmax = -9, sneg = 0;
  for (let t = 0; t < 25; t += dt) { b.step(dt, { pitch, roll: 0, fwd: 0, flap: true, tuck: false, jump: false }, t); if (t > 8) { n++; sv += b.v[1]; sp += Math.max(b.power, 0); sneg += Math.min(b.power, 0); so += b.om[2] ** 2; sV += b.airspeed; pk = Math.max(pk, b.power); gmin = Math.min(gmin, b.gload); gmax = Math.max(gmax, b.gload); } }
  return `climb=${(sv / n).toFixed(2)} V=${(sV / n * 3.6).toFixed(0)} P=${(sp / n).toFixed(0)} Pneg=${(sneg / n).toFixed(0)} pk=${pk.toFixed(0)} omz=${Math.sqrt(so / n).toFixed(2)} g=${gmin.toFixed(1)}..${gmax.toFixed(1)} st=${b.stamina.toFixed(2)}`;
}
const grid = JSON.parse(process.argv[2] || '[{}]');
for (const g of grid) console.log(JSON.stringify(g), run(g, g.pitch ?? 0.2));
