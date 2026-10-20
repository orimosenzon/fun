import { mj } from './load.mjs';
import { World, DT } from '../js/sim/world.js';
import { Robot } from '../js/ctrl/robot.js';
import { rotZYX } from '../js/ctrl/linalg.js';
const tests = [
  [[0.35, 0.22, 0.85], null], [[0.35, 0.22, 0.85], rotZYX(0)], [[0.3, 0.18, 0.80], rotZYX(0, -0.25)],
  [[0.25, 0.25, 1.1], null], [[0.4, 0.2, 1.3], null], [[0.3, -0.3, 0.8], null],
];
for (const [pos, R] of tests) {
  const w = new World(mj, {}); const r = new Robot(w); const m = w.model, d = w.data;
  r.lean = 0.2; r.crouch = 0.05;
  for (let i = 0; i < 1500; i++) {
    const side = pos[1] > 0 ? 'l' : 'r';
    r.handTargets[side] = { pos, R, w: 90, kp: 180, wr: 25 };
    mj.mj_step1(m, d); r.control(); mj.mj_step2(m, d);
    if (r.fallen) break;
  }
  const side = pos[1] > 0 ? 'l' : 'r';
  const g = r.wbc.sitePos(r.wbc.ids[side + '_grip']);
  const err = Math.hypot(g[0] - pos[0], g[1] - pos[1], g[2] - pos[2]);
  const q = ['shoulder_pitch', 'shoulder_roll', 'shoulder_yaw', 'elbow', 'forearm_roll', 'wrist_flex', 'wrist_dev'].map(n => r.getJointQ(side + '_' + n).toFixed(2)).join(' ');
  const tq = ['shoulder_pitch', 'shoulder_roll', 'elbow'].map(n => r.tau[r.jidx[side + '_' + n]].toFixed(0)).join(' ');
  console.log(JSON.stringify(pos), R ? 'R' : '-', 'grip', g.map(x => x.toFixed(3)).join(','), 'err', err.toFixed(3), 'q', q, 'tau', tq, r.fallen ? 'FALL' : '');
}
