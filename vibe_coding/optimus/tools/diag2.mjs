// מעקב מפורט אחרי צעד: מומנטים, ZMP, ξ בכל 20 מ"ש
import { mj } from './load.mjs';
import { World, DT } from '../js/sim/world.js';
import { Robot } from '../js/ctrl/robot.js';
const w = new World(mj, {}); const m = w.model, d = w.data; const r = new Robot(w);
const f = (a, n = 2) => Array.from(a).map(x => x.toFixed(n)).join(',');
const v = +(process.argv[2] || 0.45), from = +(process.argv[3] || 1.2), to = +(process.argv[4] || 2.3);
const names = ['l_hip_roll', 'l_hip_pitch', 'l_knee', 'l_ankle_pitch', 'r_hip_roll', 'r_hip_pitch', 'r_knee', 'r_ankle_pitch', 'waist_pitch'];
for (let i = 0; i < 6000 && !r.fallen; i++) {
  const t = i * DT;
  r.loco.setCommand(t > 0.5 ? (process.env.WZ ? 0.2 : Math.min(v, (t - 0.5) * 0.3)) : 0, 0, t > 0.5 && process.env.WZ ? +process.env.WZ : 0);
  mj.mj_step1(m, d); r.control(); mj.mj_step2(m, d);
  if (t > from && t < to && i % 10 === 0) {
    const z = r.loco.viz, ph = r.loco.phases[0];
    const tau = names.map(n => r.tau[r.jidx[n]].toFixed(0).padStart(4)).join(' ');
    console.log(t.toFixed(2), (ph?.type || '-') + (ph?.swing || ' '), 'xi-ref', f([z.xi[0] - z.xiRef[0], z.xi[1] - z.xiRef[1]]), 'sat', "-", 'com', f(r.wbc.com(), 2), 'pitch', (Math.asin(-d.xmat[9 * 1 + 6]) * 57.3).toFixed(1), 'τ', tau);
  }
}
