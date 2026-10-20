import { mj } from './load.mjs';
import { World, DT } from '../js/sim/world.js';
import { Robot } from '../js/ctrl/robot.js';
const w = new World(mj, {}); const m = w.model, d = w.data; const r = new Robot(w);
const f = (a, n = 2) => Array.from(a).map(x => x.toFixed(n)).join(',');
const F = [+(process.argv[2] || 300), +(process.argv[3] || 0)];
let last = null;
for (let i = 0; i < 2000 && !r.fallen; i++) {
  const t = i * DT;
  mj.mj_step1(m, d); r.control();
  d.xfrc_applied.fill(0);
  if (t >= 1 && t < 1.1) { d.xfrc_applied[6 * r.wbc.ids.torso] = F[0]; d.xfrc_applied[6 * r.wbc.ids.torso + 1] = F[1]; }
  mj.mj_step2(m, d);
  const ph = r.loco.phases[0];
  if (t > 0.98 && i % 15 === 0) { const z = r.loco.viz; console.log(t.toFixed(2), r.loco.mode, (ph?.type || '') + (ph?.swing || ''), 'xi', f(z.xi), 'xiRef', f(z.xiRef), 'sat', "-", 'to', ph?.to ? f(ph.to.p.slice(0, 2)) : '', 'adj', ph?.adj ? f(ph.adj) : ''); }
}
console.log(r.fallen ? 'FELL' : 'OK');
