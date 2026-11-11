import { initWorld } from '../js/sim/world.js';
import { Bird } from '../js/sim/bird.js';
initWorld();
const pitch = +(process.argv[2] ?? 0.35), tr = process.argv[3];
const b = new Bird(); b.windOn = false; b.placeGround(250, 950, 0);
const dt = 1 / 60; let t = 0; const y0 = b.p[1];
const I = (o) => ({ pitch: 0, roll: 0, fwd: 1, flap: false, tuck: false, jump: false, ...o });
for (; t < 8 && b.mode === 'ground'; t += dt) b.step(dt, I({ flap: true }), t);
let k = 0, air = 0;
for (; t < 16; t += dt) { if (b.mode === 'air') { b.step(dt, I({ pitch, flap: true }), t); air += dt; } else if (air > 0) break;
  if (tr && k++ % 6 === 0) console.log(`t=${air.toFixed(2)} y=${(b.p[1] - y0).toFixed(2)} V=${b.airspeed.toFixed(1)} vy=${b.v[1].toFixed(2)} al=${(b.alpha * 57.3).toFixed(0)} pit=${(b.pitch * 57.3).toFixed(0)} fold=${b.fold.toFixed(2)} amp=${b.amp.toFixed(2)} P=${b.power.toFixed(0)} g=${b.gload.toFixed(2)} st=${b.stall.toFixed(2)} vi=${b.vi.toFixed(2)} legs=${b.legs.toFixed(1)} uP=${b.uP.toFixed(2)}`); }
console.log(`pitch=${pitch} mode=${b.mode} air=${air.toFixed(1)}s height=${(b.p[1] - y0).toFixed(1)} V=${(b.airspeed * 3.6).toFixed(0)}`, b.events.map((e) => e.type).join(','));
