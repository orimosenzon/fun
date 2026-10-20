// מקליט נקודות עבודה (מהירות, מומנט) של מפרקים מהסימולטור, לגרף מומנט-מהירות במרכז המקצועי
import { mj } from './load.mjs';
import { World, DT } from '../js/sim/world.js';
import { Robot } from '../js/ctrl/robot.js';
import { SCENES } from '../js/sim/scenes.js';
import { TaskRunner } from '../js/tasks/tasks.js';
import { writeFileSync } from 'fs';
const joints = ['l_knee', 'l_hip_pitch', 'l_ankle_pitch', 'l_hip_roll'];
const out = {};
function run(label, setup, T, scene = 'empty') {
  const world = new World(mj, SCENES[scene].build()); const m = world.model, d = world.data;
  const app = { world, robot: new Robot(world) }; app.tasks = new TaskRunner(app);
  const r = app.robot; setup(app);
  const pts = Object.fromEntries(joints.map(j => [j, []]));
  let Psum = 0, n = 0;
  for (let i = 0; i < T / DT; i++) {
    if (app.cmd) r.loco.setCommand(...app.cmd(i * DT));
    mj.mj_step1(m, d); app.tasks.update(); r.control(); mj.mj_step2(m, d);
    if (i * DT > 1 && i % 10 === 0) {
      for (const j of joints) { const k = r.jidx[j], jj = r.wbc.joints[k]; pts[j].push([+Math.abs(d.qvel[jj.dof]).toFixed(2), +Math.abs(r.tau[k]).toFixed(1)]); }
      Psum += r.power.P; n++;
    }
  }
  out[label] = { pts, power: Math.round(Psum / n) };
  console.log(label, 'avg power', Math.round(Psum / n), 'W', r.fallen ? 'FALLEN' : '');
}
run('walk', (a) => { a.cmd = (t) => t > 0.5 ? [0.45, 0, 0] : [0, 0, 0]; }, 8);
run('walkFast', (a) => { a.cmd = (t) => t > 0.5 ? [Math.min(0.62, 0.3 + t * 0.08), 0, 0] : [0, 0, 0]; }, 8);
run('squat', (a) => { a.tasks.start('squat'); }, 14, 'lab');
run('stand', () => {}, 4);
writeFileSync(new URL('../js/academy/oppoints.js', import.meta.url), '// נקודות עבודה שהוקלטו מהסימולטור (tools/record_oppoints.mjs). [מהירות רד\'/ש\', מומנט נ"מ]\nexport const OPPOINTS = ' + JSON.stringify(out) + ';\n');
