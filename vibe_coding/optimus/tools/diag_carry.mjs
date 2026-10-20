import { mj } from './load.mjs';
import { World, DT } from '../js/sim/world.js';
import { Robot } from '../js/ctrl/robot.js';
import { SCENES } from '../js/sim/scenes.js';
import { TaskRunner } from '../js/tasks/tasks.js';
const world = new World(mj, SCENES.lab.build());
const app = { world, robot: new Robot(world) }; app.tasks = new TaskRunner(app); app.tasks.start('box');
const m = world.model, d = world.data, r = app.robot;
const f = (a, n = 3) => Array.from(a).map(x => x.toFixed(n)).join(',');
const from = +(process.argv[2] || 7.4), to = +(process.argv[3] || 17.5);
const b = world.bodyId('box1');
for (let i = 0; i < to / DT; i++) {
  mj.mj_step1(m, d); app.tasks.update(); r.control(); mj.mj_step2(m, d);
  const t = i * DT;
  if (t > from && i % 100 === 0) {
    const gl = r.wbc.sitePos(r.wbc.ids.l_grip), gr = r.wbc.sitePos(r.wbc.ids.r_grip);
    const bp = d.xpos.subarray(3 * b, 3 * b + 3);
    const mid = [(gl[0] + gr[0]) / 2, (gl[1] + gr[1]) / 2, (gl[2] + gr[2]) / 2];
    const sep = Math.hypot(gl[0] - gr[0], gl[1] - gr[1]);
    console.log(t.toFixed(1), app.tasks.status.slice(0, 14).padEnd(14), 'box-handsMid', f([bp[0] - mid[0], bp[1] - mid[1], bp[2] - mid[2]]), 'sep', sep.toFixed(3), 'tgtL', r.handTargets.l ? f(r.handTargets.l.pos, 2) : '-', 'gripL', f(gl, 2), 'mode', r.loco.mode, 'cmd', f(r.loco.cmd, 2));
  }
}
