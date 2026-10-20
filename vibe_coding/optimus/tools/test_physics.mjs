// חבילת הבדיקות: הליכה, דחיפות, משימות והספק. node tools/test_physics.mjs [walk|push|tasks|power]
import { mj } from './load.mjs';
import { World, DT } from '../js/sim/world.js';
import { Robot } from '../js/ctrl/robot.js';
import { SCENES } from '../js/sim/scenes.js';
import { TaskRunner } from '../js/tasks/tasks.js';

const only = process.argv[2];
const results = [];
const check = (name, ok, info) => { results.push({ name, ok, info }); console.log(`${ok ? '✓' : '✗'} ${name.padEnd(34)} ${info}`); };

function sim(scene, setup, T, perStep) {
  const def = SCENES[scene].build();
  const world = new World(mj, def); const m = world.model, d = world.data;
  const app = { world, robot: new Robot(world) }; app.tasks = new TaskRunner(app);
  setup?.(app);
  let t = 0;
  for (; t < T; t += DT) {
    perStep?.(app, t);
    mj.mj_step1(m, d); app.tasks.update(); app.robot.control();
    d.xfrc_applied.fill(0);
    if (app.push && app.push.t > 0) { for (let k = 0; k < 3; k++) d.xfrc_applied[6 * app.push.body + k] = app.push.F[k]; app.push.t -= DT; }
    mj.mj_step2(m, d); world.t += DT;
    if (app.robot.fallen) break;
    if (app.tasks.active && !app.tasks.gen && t > 1) break;
  }
  const out = { app, t, x: d.qpos[0], y: d.qpos[1], fallen: app.robot.fallen, P: app.robot.power.Pavg, steps: app.robot.loco.stepCount };
  world.dispose?.();
  return out;
}

if (!only || only === 'walk') {
  for (const v of [0.3, 0.45, 0.62]) {
    const r = sim('empty', null, 10, (a, t) => a.robot.loco.setCommand(t > 0.5 ? v : 0, 0, 0));
    const speed = r.x / 9.5;
    check(`הליכה קדימה ${v} מ'/ש'`, !r.fallen && speed > v * 0.75, `${r.x.toFixed(2)} מ' ב-10 ש', ${r.steps} צעדים, ${Math.round(r.P)} ואט`);
  }
  {
    const r = sim('empty', null, 9, (a, t) => a.robot.loco.setCommand(t > 0.5 && t < 6 ? 0.2 : 0, 0, t > 0.5 && t < 6 ? 0.6 : 0));
    check('הליכה בקשת', !r.fallen, `סיום ב-(${r.x.toFixed(2)}, ${r.y.toFixed(2)})`);
  }
  {
    const r = sim('empty', null, 6, (a, t) => a.robot.loco.setCommand(0, 0, t > 0.5 && t < 5 ? 0.7 : 0));
    check('סיבוב במקום', !r.fallen, `${r.steps} צעדים`);
  }
  {
    const r = sim('empty', null, 8, (a, t) => a.robot.loco.setCommand(0, t > 0.5 && t < 5 ? 0.15 : 0, 0));
    check('הליכה הצידה', !r.fallen && r.y > 0.4, `${r.y.toFixed(2)} מ' שמאלה`);
  }
  {
    const r = sim('empty', null, 8, (a, t) => a.robot.loco.setCommand(t > 0.5 && t < 5 ? -0.25 : 0, 0, 0));
    check('הליכה אחורה', !r.fallen && r.x < -0.6, `${r.x.toFixed(2)} מ'`);
  }
}

if (!only || only === 'push') {
  const pushes = [[30, 0, 'קדימה'], [45, 0, 'קדימה'], [-25, 0, 'אחורה'], [0, 20, 'שמאלה'], [0, -20, 'ימינה'], [20, 20, 'באלכסון']];
  for (const [jx, jy, dir] of pushes) {
    const r = sim('empty', null, 5, (a, t) => { if (Math.abs(t - 1) < DT / 2) a.push = { body: a.robot.wbc.ids.torso, F: [jx / 0.1, jy / 0.1, 0], t: 0.1 }; });
    check(`דחיפה ${Math.hypot(jx, jy).toFixed(0)} נ"ש ${dir}`, !r.fallen, r.steps ? `${r.steps} צעדי התאוששות` : 'בלי צעד (אסטרטגיית קרסול)');
  }
}

if (!only || only === 'power') {
  const r = sim('empty', null, 4);
  check('הספק בעמידה (AI Day: ~100 ואט)', Math.abs(r.P - 100) < 25, `${Math.round(r.P)} ואט`);
  const w = sim('empty', null, 9, (a, t) => a.robot.loco.setCommand(t > 0.5 ? 0.6 : 0, 0, 0));
  check('הספק בהליכה מהירה (AI Day: ~500)', w.P > 250 && w.P < 650, `${Math.round(w.P)} ואט`);
}

if (!only || only === 'tasks') {
  for (const [key, T] of [['wave', 12], ['squat', 30], ['dance', 35], ['tour', 80], ['box', 40], ['cells', 110]]) {
    const r = sim('lab', (a) => a.tasks.start(key), T);
    const ctx = r.app.tasks.ctx;
    check(`משימה: ${key}`, !r.fallen && !!ctx?.done, `${ctx?.status || ''} (${r.t.toFixed(1)} ש')`);
  }
}

const bad = results.filter(r => !r.ok);
console.log(`\n${results.length - bad.length}/${results.length} עברו`);
process.exit(bad.length ? 1 : 0);
