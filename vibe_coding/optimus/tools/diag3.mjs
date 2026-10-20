import { mj } from './load.mjs';
import { World, DT } from '../js/sim/world.js';
import { Robot } from '../js/ctrl/robot.js';
const w = new World(mj, {}); const m = w.model, d = w.data; const r = new Robot(w);
const f = (a, n = 2) => Array.from(a).map(x => x.toFixed(n)).join(',');
const v = +(process.argv[2] || 0.45), from = +(process.argv[3] || 1.3), to = +(process.argv[4] || 1.8);
const pitchOf = (b) => Math.asin(-d.xmat[9 * b + 6]) * 57.3;
for (let i = 0; i < 2500 && !r.fallen; i++) {
  const t = i * DT;
  r.loco.setCommand(t > 0.5 ? v : 0, 0, 0);
  mj.mj_step1(m, d); r.control();
  const qddPlan = Array.from(r.wbc.qdd.slice(3, 6));
  mj.mj_step2(m, d);
  if (t > from && t < to && i % 10 === 0) {
    const ws = r.wbc.out.wrenches.map(x => x.s + ':Fz' + x.w[2].toFixed(0) + ' T' + f(x.w.slice(3, 5), 0)).join(' ');
    console.log(t.toFixed(2), 'pelv', pitchOf(1).toFixed(1), 'lfoot', pitchOf(r.wbc.ids.l_foot).toFixed(1), 'rfoot', pitchOf(r.wbc.ids.r_foot).toFixed(1), 'qddPlan(ang)', f(qddPlan, 1), 'qacc', f(d.qacc.slice(3, 6), 1), ws);
  }
}
