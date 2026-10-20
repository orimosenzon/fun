import { mj } from './load.mjs';
import { World, DT } from '../js/sim/world.js';
import { Robot } from '../js/ctrl/robot.js';
import { rotZYX } from '../js/ctrl/linalg.js';
const RH = rotZYX(0, -Math.PI / 2, 0);
const tests = [
  [[0.30, -0.20, 0.84], RH, 0.12], [[0.25, -0.20, 0.84], RH, 0.12], [[0.30, -0.20, 0.95], RH, 0.12], [[0.30, -0.20, 0.84], null, 0.12], [[0.30, -0.20, 0.84], RH, 0.3],
];
for (const [pos, R, lean] of tests) {
  const w = new World(mj, {}); const r = new Robot(w); const m = w.model, d = w.data;
  r.lean = lean; r.crouch = 0.03;
  for (let i = 0; i < 1500; i++) {
    r.handTargets.r = { pos, R, w: 120, kp: 220, wr: 30 };
    mj.mj_step1(m, d); r.control(); mj.mj_step2(m, d);
  }
  const g = r.wbc.sitePos(r.wbc.ids.r_grip);
  const err = Math.hypot(g[0] - pos[0], g[1] - pos[1], g[2] - pos[2]);
  const names = ['shoulder_pitch', 'shoulder_roll', 'shoulder_yaw', 'elbow', 'forearm_roll', 'wrist_flex', 'wrist_dev'];
  const q = names.map(n => r.getJointQ('r_' + n).toFixed(2)).join(' ');
  const lim = names.map(n => { const j = r.wbc.joints[r.jidx['r_' + n]]; const v = r.getJointQ('r_' + n); return (v <= j.range[0] + 0.02 || v >= j.range[1] - 0.02) ? n : null; }).filter(Boolean).join(',');
  const Rh = r.wbc.bodyMat(r.wbc.ids.r_hand);
  console.log(JSON.stringify(pos), R ? 'RH' : '-', 'lean', lean, 'grip', g.map(x => x.toFixed(3)).join(','), 'err', err.toFixed(3), 'q', q, 'atLimit', lim, 'fingersDir', [-Rh[2], -Rh[5], -Rh[8]].map(x => x.toFixed(2)).join(','));
}
