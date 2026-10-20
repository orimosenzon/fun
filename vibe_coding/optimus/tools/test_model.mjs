import { mj } from './load.mjs';
import { World, buildXML } from '../js/sim/world.js';
import { totalMass } from '../js/robot/spec.js';
let w;
try { w = new World(mj, {}); } catch (e) { console.log('ERR', e.message || e); const xml = buildXML({}); console.log(xml.slice(0, 400)); process.exit(1); }
const m = w.model, d = w.data;
console.log('nq', m.nq, 'nv', m.nv, 'nu', m.nu, 'nbody', m.nbody, 'ngeom', m.ngeom);
let mass = 0; for (let i = 1; i < m.nbody; i++) mass += m.body_mass[i];
console.log('mass model', mass.toFixed(2), 'spec', totalMass().toFixed(2));
console.log('pelvis z', d.qpos[2].toFixed(3));
// מיקום כפות הרגליים
for (const s of ['l_sole', 'r_sole']) { const id = w.siteId(s); console.log(s, Array.from(d.site_xpos.slice(3*id, 3*id+3)).map(x => x.toFixed(3)).join(' ')); }
console.log('subtree com', Array.from(d.subtree_com.slice(3, 6)).map(x => x.toFixed(3)).join(' '));
for (const s of ['l_grip', 'r_grip', 'l_palm_c']) { const id = w.siteId(s); console.log(s, Array.from(d.site_xpos.slice(3*id, 3*id+3)).map(x => x.toFixed(3)).join(' ')); }
console.log('ncon', d.ncon);
const t0 = performance.now(); for (let i = 0; i < 500; i++) w.step(); const ms = performance.now() - t0;
console.log('500 steps (1s sim) with no control:', ms.toFixed(0), 'ms; pelvis z', d.qpos[2].toFixed(3), 'ncon', d.ncon);
