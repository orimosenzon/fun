import { mj } from './load.mjs';
import { World, DT } from '../js/sim/world.js';
import { Robot } from '../js/ctrl/robot.js';
const w = new World(mj, {}); const m = w.model, d = w.data; const r = new Robot(w);
const names = ['bx', 'by', 'bz', 'rx', 'ry', 'rz', ...r.wbc.joints.map(j => j.name)];
for (let i = 0; i < 60; i++) {
  mj.mj_step1(m, d); r.control();
  const plan = Float64Array.from(r.wbc.qdd);
  const tau = Float64Array.from(r.tau);
  mj.mj_step2(m, d);
  if (i === 50) {
    const rows = r.wbc.B.map((dof, k) => [names[k], plan[k], d.qacc[dof], tau[k - 6]]);
    rows.sort((a, b) => Math.abs(b[1] - b[2]) - Math.abs(a[1] - a[2]));
    for (const x of rows.slice(0, 12)) console.log(x[0].padEnd(18), 'plan', x[1].toFixed(2).padStart(8), 'actual', x[2].toFixed(2).padStart(8), 'tau', x[3]?.toFixed(1));
    console.log('ncon', d.ncon, 'efc', d.nefc);
    // כוחות המגע בפועל לעומת המתוכננים
    let Fz = 0; const c = d.contact; for (let k = 0; k < d.ncon; k++) { const res = new mj.DoubleBuffer(6); mj.mj_contactForce(m, d, k, res); Fz += res.GetView()[0]; res.delete(); }
    console.log('actual total normal', Fz.toFixed(1), 'planned', r.wbc.out.wrenches.reduce((s, x) => s + x.w[2], 0).toFixed(1));
  }
}
