import { mj } from './load.mjs';
import { World, DT } from '../js/sim/world.js';
import { Robot } from '../js/ctrl/robot.js';
import { SCENES } from '../js/sim/scenes.js';
import { TaskRunner } from '../js/tasks/tasks.js';
const world = new World(mj, SCENES.lab.build());
const app = { world, robot: new Robot(world) }; app.tasks = new TaskRunner(app); app.tasks.start('cells');
const m = world.model, d = world.data, r = app.robot;
const f = (a, n = 3) => Array.from(a).map(x => x.toFixed(n)).join(',');
const from = +(process.argv[2] || 6.9), to = +(process.argv[3] || 10.6);
const cb = world.bodyId('cell1');
const handBodies = new Set(); for (let b = 0; b < m.nbody; b++) { const n = world.names.body[b] || ''; if (n.startsWith('r_') && (n.includes('hand') || /thumb|index|middle|ring|little/.test(n))) handBodies.add(b); }
for (let i = 0; i < to / DT; i++) {
  mj.mj_step1(m, d); app.tasks.update(); r.control(); mj.mj_step2(m, d);
  const t = i * DT;
  if (t > from && i % 25 === 0) {
    let hc = 0, other = []; const cs = d.contact;
    for (let k = 0; k < d.ncon; k++) { const c = cs.get(k); const b1 = m.geom_bodyid[c.geom1], b2 = m.geom_bodyid[c.geom2]; if (b1 === cb || b2 === cb) { const o = b1 === cb ? b2 : b1; if (handBodies.has(o)) hc++; else other.push(world.names.body[o] || 'world'); } }
    cs.delete();
    const g = r.wbc.sitePos(r.wbc.ids.r_grip), cp = Array.from(d.xpos.subarray(3 * cb, 3 * cb + 3));
    console.log(t.toFixed(2), app.tasks.status.slice(0, 18).padEnd(18), 'grip', f(g), 'tgt', r.handTargets.r ? f(r.handTargets.r.pos) : '-', 'cell', f(cp), 'handCon', hc, 'other', [...new Set(other)].join('|'), 'close', r.hands.state.r.close.toFixed(2));
  }
}
