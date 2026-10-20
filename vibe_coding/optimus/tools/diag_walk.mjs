import { mj } from './load.mjs';
import { World, DT } from '../js/sim/world.js';
import { Robot } from '../js/ctrl/robot.js';
const w = new World(mj, {}); const m = w.model, d = w.data; const r = new Robot(w);
const f = (a, n = 3) => a.map(x => x.toFixed(n)).join(',');
let lastPhase = null, satMax = 0, satJ = {};
for (let i = 0; i < 6000 && !r.fallen; i++) {
  const t = i * DT;
  r.loco.setCommand(t > 0.5 ? Math.min(+(process.argv[2]||0.45), (t - 0.5) * 0.3) : 0, 0, 0);
  mj.mj_step1(m, d); r.control();
  r.wbc.joints.forEach((j, k) => { if (Math.abs(r.tau[k]) > 0.97 * j.torque) satJ[j.name] = (satJ[j.name] || 0) + 1; });
  mj.mj_step2(m, d);
  const ph = r.loco.phases[0];
  if (ph && ph !== lastPhase) {
    const v = r.loco.viz;
    if (ph.type === 'ss') console.log(t.toFixed(2), 'SS swing', ph.swing, 'to', f(ph.toPlanned.p.slice(0, 2)), 'stance', f(r.loco.feet[ph.stanceSide].p.slice(0, 2)), 'xi', f(v.xi || [0, 0]), 'xiRef', f(v.xiRef || [0, 0]), 'com', f(r.wbc.com().slice(0, 2)));
    else if (lastPhase?.type === 'ss') console.log(t.toFixed(2), '  landed', lastPhase.swing, 'adj', f(lastPhase.adj), 'final to', f(lastPhase.to.p.slice(0, 2)), 'actual', f(r.loco.feet[lastPhase.swing].p.slice(0, 2)));
    lastPhase = ph;
  }
}
console.log('saturation counts', satJ);
