// אמינות משימה: מריצים כמה פעמים עם נקודת התחלה מוזזת מעט. node tools/test_robust.mjs <task> [runs]
import { mj } from './load.mjs';
import { World, DT } from '../js/sim/world.js';
import { Robot } from '../js/ctrl/robot.js';
import { SCENES } from '../js/sim/scenes.js';
import { TaskRunner } from '../js/tasks/tasks.js';
const key = process.argv[2] || 'cells', runs = +(process.argv[3] || 5);
let ok = 0;
for (let k = 0; k < runs; k++) {
  const def = SCENES.lab.build();
  const rnd = (a) => (Math.sin(k * 12.9898 + a * 78.233) * 43758.5453 % 1) * 2 - 1;
  def.start = { x: 0.05 * rnd(1), y: 0.05 * rnd(2), yaw: 0.08 * rnd(3) };
  const world = new World(mj, def); const m = world.model, d = world.data;
  const app = { world, robot: new Robot(world) }; app.tasks = new TaskRunner(app); app.tasks.start(key);
  let t = 0;
  for (; t < 120 && app.tasks.gen; t += DT) { mj.mj_step1(m, d); app.tasks.update(); app.robot.control(); mj.mj_step2(m, d); }
  const done = !!app.tasks.ctx?.done;
  ok += done;
  console.log(`run ${k}: start ${Object.values(def.start).map(v => v.toFixed(3)).join(',')} -> ${app.tasks.status} (${t.toFixed(1)} s) ${done ? 'OK' : 'FAIL'}`);
  world.dispose();
}
console.log(`${ok}/${runs} succeeded`);
