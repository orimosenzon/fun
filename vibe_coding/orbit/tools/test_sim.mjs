import * as P from '../js/physics.js';
import { ROCKETS } from '../js/rockets.js';
const f = x => (x/1000).toFixed(1);
for (const id of Object.keys(ROCKETS)) {
  const R = ROCKETS[id];
  const inc = R.defaultInclination ?? R.site.lat;
  const alt = 200000;
  const t0 = Date.now();
  const kick = P.optimizeKick(R, R.defaultPayload, alt, inc);
  const res = P.simulateAscent({ rocket: R, payload: R.defaultPayload, targetAlt: alt, inclination: inc, kick });
  const dvLeft = P.upperStageDv(res, R.defaultPayload);
  console.log(`${id}: kick=${kick} ${res.result} t=${res.t.toFixed(0)}s peri=${f(res.peri)} apo=${f(res.apo)} v=${res.v.toFixed(0)} maxQ=${(res.maxQ/1000).toFixed(1)}kPa@${res.maxQt.toFixed(0)} maxG=${res.maxG.toFixed(2)} propLeft=${res.upperPropLeft.toFixed(0)} dvLeft=${dvLeft.toFixed(0)}  (${Date.now()-t0}ms)`);
  const L=res.loss; console.log(`   ideal=${L.ideal.toFixed(0)} grav=${L.gravity.toFixed(0)} drag=${L.drag.toFixed(0)} steer=${L.steering.toFixed(0)} v0=${res.v0.toFixed(0)}`);
  console.log('   events', res.events.map(e=>`${e.name}@${e.t.toFixed(0)}s/${f(e.alt)}km/${e.v.toFixed(0)}`).join(' '));
}
