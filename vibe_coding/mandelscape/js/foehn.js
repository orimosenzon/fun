// Everything this project takes from FÖHN (../jetbikes/foehn), in one place: three.js, the F-5
// with its engine and flight computer, its model, its flames and its sound. They are imported,
// not copied, so the bike here is always the latest one. If FÖHN ever moves on in a way that
// breaks this, copy those files here and change only the paths below (tests/flight.mjs tells).
import * as THREE from '../../jetbikes/foehn/lib/three.module.min.js';
export { THREE };
export { T as FoehnTerrain } from '../../jetbikes/foehn/js/sim/terrain.js';
export { Sim } from '../../jetbikes/foehn/js/sim/sim.js';
export { G, PROBES } from '../../jetbikes/foehn/js/sim/bike.js';
export { V_MAX, V_MIN, FEEL } from '../../jetbikes/foehn/js/sim/fcs.js';
export { len, yprOf, qrot, clamp, lerp } from '../../jetbikes/foehn/js/sim/vmath.js';
