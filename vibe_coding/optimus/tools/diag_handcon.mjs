import { mj } from './load.mjs';
import { World, DT } from '../js/sim/world.js';
import { Robot } from '../js/ctrl/robot.js';
import { SCENES } from '../js/sim/scenes.js';
import { TaskRunner } from '../js/tasks/tasks.js';
const world = new World(mj, SCENES.lab.build());
const app = { world, robot: new Robot(world) }; app.tasks = new TaskRunner(app); app.tasks.start(process.argv[4] || 'cells');
const m = world.model, d = world.data, r = app.robot;
const from = +(process.argv[2] || 8.5), to = +(process.argv[3] || 9.3);
const gname = (g) => mj.mj_id2name(m, mj.mjtObj.mjOBJ_GEOM.value, g) || ('geom' + g + '@' + world.names.body[m.geom_bodyid[g]]);
for (let i = 0; i < to / DT; i++) {
  mj.mj_step1(m, d); app.tasks.update(); r.control(); mj.mj_step2(m, d);
  const t = i * DT;
  if (t > from && i % 50 === 0) {
    const cs = d.contact, out = new Set();
    for (let k = 0; k < d.ncon; k++) { const c = cs.get(k); const n1 = gname(c.geom1), n2 = gname(c.geom2); if (!/floor/.test(n1 + n2)) out.add(n1 + '~' + n2); }
    cs.delete();
    console.log(t.toFixed(2), [...out].join('  '));
  }
}
