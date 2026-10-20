import { mj } from './load.mjs';
import { World, DT } from '../js/sim/world.js';
import { Robot } from '../js/ctrl/robot.js';
const w = new World(mj, {}); const m = w.model, d = w.data; const r = new Robot(w);
const f = (a, n = 1) => Array.from(a).map(x => x.toFixed(n)).join(',');
for (let i = 0; i < 400 && !r.fallen; i++) {
  mj.mj_step1(m, d); r.control();
  const qp = Array.from(r.wbc.qdd.slice(0, 6));
  mj.mj_step2(m, d);
  if (i % 20 === 0) {
    const ws = r.wbc.out.wrenches.map(x => x.s + ':F' + f(x.w.slice(0, 3), 0) + ' T' + f(x.w.slice(3), 1)).join(' | ');
    console.log((i * DT).toFixed(2), 'res', r.wbc.out.baseResidual.toFixed(1), 'qddB', f(qp, 2), 'qacc', f(d.qacc.slice(0, 6), 2), ws);
  }
}
