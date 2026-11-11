// node tests/scen.mjs "<js expr of t returning input>" tEnd [logFrom logStep] [startSpeed startAlt]
import { loadWorld, deg } from './util.mjs';
import { Sim } from '../js/sim/sim.js';
import { yprOf } from '../js/sim/vmath.js';
const meta = loadWorld();
const sim = new Sim();
const pad = meta.pads.base;
const fn = new Function('t', 'b', 'return ' + process.argv[2]);
const tEnd = +process.argv[3], from = +(process.argv[4] || 0), stepL = +(process.argv[5] || 1);
const v0 = +(process.argv[6] || 0), alt = +(process.argv[7] || 300);
if (v0 > 0) sim.reset([pad[0], pad[1] + alt, pad[2]], 0, v0); else sim.reset([pad[0], pad[1], pad[2]], 0);
const b = sim.bike;
let next = from;
while (sim.t < tEnd && !b.crash) {
  sim.tick({ noFloor: !process.env.FLOOR, ...fn(sim.t, b) });
  if (sim.t >= next) {
    const [y, p, r] = yprOf(b.q), te = sim.fcs.tele;
    console.log(`t=${sim.t.toFixed(2)} agl=${b.agl.toFixed(1)} V=${b.airspeed.toFixed(1)} vz=${b.v[1].toFixed(1)}/${te.vzDes.toFixed(1)} yaw=${deg(y).toFixed(0)} pit=${deg(p).toFixed(1)}/${deg(te.thetaDes).toFixed(1)} roll=${deg(r).toFixed(1)}/${deg(te.phiDes).toFixed(1)} w=${b.w.map(v=>v.toFixed(2))} tilt=${deg(b.tilt).toFixed(0)} L=${b.L.map(v=>v.toFixed(0)).join('/')} R=${b.R.toFixed(0)} va=${b.va.map(v=>deg(v).toFixed(0)).join('/')} rv=${deg(b.rp).toFixed(0)}/${deg(b.ry).toFixed(0)} dae=${deg(b.da).toFixed(0)}/${deg(b.de).toFixed(0)}/${deg(b.dr).toFixed(0)} g=${(Math.hypot(b.acc[0],b.acc[1]+9.81,b.acc[2])/9.81).toFixed(2)} st=${b.stall.toFixed(1)} sat=${te.sat}`);
    next += stepL;
  }
}
console.log('crash', b.crash, 'fuel', b.fuel.toFixed(1));
