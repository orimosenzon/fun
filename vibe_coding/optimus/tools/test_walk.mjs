// בדיקות הליכה: node tools/test_walk.mjs [scenario]
import { mj } from './load.mjs';
import { World, DT } from '../js/sim/world.js';
import { Robot } from '../js/ctrl/robot.js';
const which = process.argv[2] || 'forward';
const w = new World(mj, {});
const m = w.model, d = w.data;
const r = new Robot(w);
const f3 = (a) => a.map(x => x.toFixed(2)).join(',');
const script = {
  stand: (t) => [0, 0, 0],
  forward: (t) => t > 0.5 && t < 6.5 ? [0.45, 0, 0] : [0, 0, 0],
  fast: (t) => t > 0.5 && t < 8 ? [Math.min(0.62, 0.2 + t * 0.1), 0, 0] : [0, 0, 0],
  turn: (t) => t > 0.5 && t < 6 ? [0.2, 0, 0.6] : [0, 0, 0],
  spin: (t) => t > 0.5 && t < 5 ? [0, 0, 0.7] : [0, 0, 0],
  side: (t) => t > 0.5 && t < 5 ? [0, 0.15, 0] : [0, 0, 0],
  back: (t) => t > 0.5 && t < 5 ? [-0.25, 0, 0] : [0, 0, 0],
  push: (t) => [0, 0, 0],
};
const pushes = which === 'push' ? [[1.5, [300, 0]], [4, [0, 260]], [6.5, [-250, 0]], [9, [0, -300]]] : [];
let T = which === 'push' ? 11 : 9, ms = 0, Psum = 0, Pn = 0, walkP = 0, walkN = 0;
const N = Math.round(T / DT);
let maxTau = {};
for (let i = 0; i < N; i++) {
  const t = i * DT;
  r.loco.setCommand(...script[which](t));
  mj.mj_step1(m, d);
  const t0 = performance.now();
  r.control();
  ms += performance.now() - t0;
  d.xfrc_applied.fill(0);
  for (const [tp, F] of pushes) if (t >= tp && t < tp + 0.1) { d.xfrc_applied[6 * r.wbc.ids.torso] = F[0]; d.xfrc_applied[6 * r.wbc.ids.torso + 1] = F[1]; }
  mj.mj_step2(m, d);
  r.wbc.joints.forEach((j, k) => { maxTau[j.group] = Math.max(maxTau[j.group] || 0, Math.abs(r.tau[k]) / j.torque); });
  if (t > 0.3) { Psum += r.power.P; Pn++; if (r.loco.mode === 'walk') { walkP += r.power.P; walkN++; } }
  if (i % 250 === 0) {
    const tm = r.telemetry();
    console.log(t.toFixed(1), r.loco.mode.padEnd(5), 'pos', f3([d.qpos[0], d.qpos[1], d.qpos[2]]), 'yaw', (r.loco.heading * 57.3).toFixed(0), 'v', tm.speed.toFixed(2), 'P', tm.power.toFixed(0), 'steps', r.loco.stepCount, r.fallen ? 'FALLEN' : '');
  }
  if (r.fallen) { console.log('FELL at', t.toFixed(2)); break; }
}
console.log('ctrl ms/step', (ms / N).toFixed(3), 'avgP', (Psum / Pn).toFixed(0), 'walkP', walkN ? (walkP / walkN).toFixed(0) : '-', 'maxTau%', Object.entries(maxTau).map(([k, v]) => k + ':' + (v * 100).toFixed(0)).join(' '));
