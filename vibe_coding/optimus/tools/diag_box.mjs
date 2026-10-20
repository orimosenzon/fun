import { mj } from './load.mjs';
import { World, DT } from '../js/sim/world.js';
import { Robot } from '../js/ctrl/robot.js';
import { SCENES } from '../js/sim/scenes.js';
import { TaskRunner } from '../js/tasks/tasks.js';
const world = new World(mj, SCENES.lab.build());
const app = { world, robot: new Robot(world) }; app.tasks = new TaskRunner(app); app.tasks.start('box');
const m = world.model, d = world.data, r = app.robot;
const f = (a) => Array.from(a).map(x => x.toFixed(3)).join(',');
const from = +(process.argv[2] || 3), to = +(process.argv[3] || 7.5);
const palmL = world.id('mjOBJ_GEOM', 'l_palm'), palmR = world.id('mjOBJ_GEOM', 'r_palm');
for (let i = 0; i < to / DT; i++) {
  mj.mj_step1(m, d); app.tasks.update(); r.control(); mj.mj_step2(m, d);
  const t = i * DT;
  if (t > from && i % 50 === 0) {
    const b = world.bodyId('box1');
    const gl = r.wbc.sitePos(r.wbc.ids.l_grip), gr = r.wbc.sitePos(r.wbc.ids.r_grip);
    const pl = Array.from(d.geom_xpos.slice(3 * palmL, 3 * palmL + 3)), pr = Array.from(d.geom_xpos.slice(3 * palmR, 3 * palmR + 3));
    let nc = 0; const cs = d.contact; for (let k = 0; k < d.ncon; k++) { const c = cs.get(k); const gs = [c.geom1, c.geom2]; if (gs.some(g => m.geom_bodyid[g] === b) && gs.some(g => m.geom_bodyid[g] !== b && g !== 0 && !String(mj.mj_id2name(m, mj.mjtObj.mjOBJ_GEOM.value, g)).startsWith('table'))) nc++; } cs.delete();
    console.log(t.toFixed(1), app.tasks.status.slice(0, 12).padEnd(12), 'box', f(d.xpos.slice(3 * b, 3 * b + 3)), 'gripL', f(gl), 'tgtL', r.handTargets.l ? f(r.handTargets.l.pos) : '-', 'palmL', f(pl), 'palmR', f(pr), 'handCon', nc);
  }
}
