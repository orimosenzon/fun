// בדיקת משימה: node tools/test_tasks.mjs <task> [seconds]
import { mj } from './load.mjs';
import { World, DT } from '../js/sim/world.js';
import { Robot } from '../js/ctrl/robot.js';
import { SCENES } from '../js/sim/scenes.js';
import { TaskRunner } from '../js/tasks/tasks.js';
const key = process.argv[2] || 'box', T = +(process.argv[3] || 30);
const def = SCENES.lab.build();
const world = new World(mj, def);
const app = { world, robot: new Robot(world) };
app.tasks = new TaskRunner(app);
app.tasks.start(key);
const m = world.model, d = world.data;
let lastStatus = '', ms = 0;
const N = Math.round(T / DT);
for (let i = 0; i < N; i++) {
  mj.mj_step1(m, d);
  const t0 = performance.now();
  app.tasks.update(); app.robot.control();
  ms += performance.now() - t0;
  d.xfrc_applied.fill(0);
  if (app.push && app.push.t > 0) { for (let k = 0; k < 3; k++) d.xfrc_applied[6 * app.push.body + k] = app.push.F[k]; app.push.t -= DT; }
  mj.mj_step2(m, d); world.t += DT;
  const st = app.tasks.status;
  if (st !== lastStatus) { console.log((i * DT).toFixed(2), st); lastStatus = st; }
  if (!app.tasks.gen) break;
}
const b = world.bodyId(key === 'cells' ? 'cell1' : 'box1');
console.log('end', (world.t).toFixed(1), 's; box at', Array.from(d.xpos.slice(3 * b, 3 * b + 3)).map(x => x.toFixed(2)).join(','), 'fallen', app.robot.fallen, 'done', app.tasks.ctx?.done, 'failed', app.tasks.ctx?.failed, 'ctrl ms/step', (ms / (world.t / DT)).toFixed(2));
