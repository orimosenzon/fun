// בדיקה ממוקדת לאחיזת תא: node tools/test_cell.mjs <index 0..2> [standX]
import { mj } from './load.mjs';
import { World, DT } from '../js/sim/world.js';
import { Robot } from '../js/ctrl/robot.js';
import { SCENES, STATIONS } from '../js/sim/scenes.js';
import { TaskRunner } from '../js/tasks/tasks.js';
const k = +(process.argv[2] || 0);
const def = SCENES.lab.build(); const C = STATIONS.cells;
def.start = { x: C.pickStand[0] - 0.056 * 0, y: C.pickStand[1] - 0.056, yaw: Math.PI / 2 };
// התאים הקודמים כבר הועברו: מוציאים אותם מהשולחן
const world = new World(mj, def);
const m = world.model, d = world.data;
for (let i = 0; i < k; i++) { const b = world.bodyId(`cell${i + 1}`), a = m.jnt_qposadr[m.body_jntadr[b]]; d.qpos[a] = -3 - i; d.qpos[a + 1] = -3; d.qpos[a + 2] = 0.05; }
mj.mj_forward(m, d);
const app = { world, robot: new Robot(world) }; app.tasks = new TaskRunner(app);
app.tasks.start('cells', { noWalk: true, first: k, only: true });
const r = app.robot, cb = world.bodyId(`cell${k + 1}`);
let last = '';
const f = (a) => Array.from(a).map(x => x.toFixed(3)).join(',');
for (let i = 0; i < 40 / DT && app.tasks.gen; i++) {
  mj.mj_step1(m, d); app.tasks.update(); r.control(); mj.mj_step2(m, d); world.t += DT;
  const st = app.tasks.status;
  if (st !== last) { const g = r.wbc.sitePos(r.wbc.ids.r_grip); console.log((i * DT).toFixed(2), st.padEnd(26), 'cell', f(d.xpos.subarray(3 * cb, 3 * cb + 3)), 'grip', f(g)); last = st; }
}
console.log('final cell', f(d.xpos.subarray(3 * cb, 3 * cb + 3)), 'upright', d.xmat[9 * cb + 8].toFixed(2), 'fallen', r.fallen);
