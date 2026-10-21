import { initMountain } from './mountain.js';
import { initTwoCons } from './twocons.js';
import { initSaddle } from './saddle.js';
import { initPolytope } from './polytope.js';

export function init3D() {
  for (const init of [initMountain, initTwoCons, initSaddle, initPolytope]) {
    try { init(); } catch (e) { console.error(init.name, e); }
  }
}
