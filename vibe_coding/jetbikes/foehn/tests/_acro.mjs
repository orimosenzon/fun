// Exploring the aerobatic law. node tests/_acro.mjs [loop|roll|knife|flip] [V] [key=value ...]
import { loadWorld, deg } from './util.mjs';
import { Sim } from '../js/sim/sim.js';
import { G } from '../js/sim/bike.js';
import { FEEL, ACRO } from '../js/sim/fcs.js';
import { yprOf, len, qrot } from '../js/sim/vmath.js';
loadWorld();
const what = process.argv[2] || 'loop', V0 = +(process.argv[3] || 55);
for (const a of process.argv.slice(4)) { const [k, v] = a.split('='); if (k in FEEL) FEEL[k] = +v; if (k in ACRO) ACRO[k] = +v; }
const s = new Sim(); s.reset([2500, 2400, 1500], 0, V0 || 0.01);
const b = s.bike, f = s.fcs;
const base = { turn: 0, pitch: 0, strafe: 0, vt: V0, noFloor: true };
for (let i = 0; i < 360; i++) s.tick(base);
f.setAcro(true);
const y0 = b.p[1], x0 = b.p[0], z0 = b.p[2];
let ang = 0, t = 0, gmax = 0, ymax = -1e9, ymin = 1e9, vmin = 1e9;
let emax = 0; const line = () => { const up = qrot(b.q, [0, 1, 0]), [yw, pt, rl] = yprOf(b.q), vI = [0, 0, 0]; console.log(`t=${t.toFixed(2)} V=${len(b.v).toFixed(0)} dy=${(b.p[1] - y0).toFixed(1)} vz=${b.v[1].toFixed(1)} pitch=${deg(pt).toFixed(0)} roll=${deg(rl).toFixed(0)} yaw=${deg(yw).toFixed(0)} upY=${up[1].toFixed(2)} w=${b.w.map((x) => x.toFixed(2))} g=${(Math.hypot(b.acc[0], b.acc[1] + G, b.acc[2]) / G).toFixed(1)} vB=${b.vAirB.map((x) => x.toFixed(1))} L=${b.lift.toFixed(0)} R=${b.R.toFixed(0)} sd=${b.sd.map((x) => x.toFixed(0))} N=${b.eng.N.toFixed(2)} heat=${b.heat.toFixed(2)} err=${deg(f.tele.attErr).toFixed(0)} sat=${f.tele.sat}`); };
const fly = (secs, inp, until) => { const n = Math.round(secs * 120); for (let i = 0; i < n; i++) { s.tick({ ...base, ...inp }); t += 1 / 120; ang += (what === 'roll' || what === 'knife' || what === 'inv' ? b.w[0] : b.w[2]) / 120; gmax = Math.max(gmax, Math.hypot(b.acc[0], b.acc[1] + G, b.acc[2]) / G); ymax = Math.max(ymax, b.p[1] - y0); ymin = Math.min(ymin, b.p[1] - y0); vmin = Math.min(vmin, len(b.v)); emax = Math.max(emax, f.tele.attErr); if (i % 30 === 0) line(); if (until && until()) break; } };
if (what === 'loop') { fly(30, { pitch: 1 }, () => Math.abs(ang) >= 2 * Math.PI); console.log('--- released'); fly(2, {}); }
if (what === 'roll') { fly(10, { turn: 1 }, () => Math.abs(ang) >= 2 * Math.PI - 0.4); console.log('--- released'); fly(3, {}); }
if (what === 'knife') { fly(5, { turn: 1 }, () => Math.abs(ang) >= Math.PI / 2 - 0.1); console.log('--- released'); what === 'knife' && fly(5, {}); }
if (what === 'inv') { fly(5, { turn: 1 }, () => Math.abs(ang) >= Math.PI - 0.3); console.log('--- released'); fly(4, {}); }
if (what === 'vert') { fly(5, { pitch: 1 }, () => Math.abs(ang) >= Math.PI / 2 - 0.25); console.log('--- released'); fly(4, {}); console.log('--- roll'); fly(1.4, { turn: 1 }); fly(1.5, {}); }
if (what === 'flip') { fly(30, { pitch: 1 }, () => Math.abs(ang) >= 2 * Math.PI); console.log('--- released'); fly(2, {}); }
console.log(`${what} from ${(V0 * 3.6).toFixed(0)} km/h: ${t.toFixed(2)} s, gmax=${gmax.toFixed(2)}, height ${ymin.toFixed(1)}..${ymax.toFixed(1)} m, end dy=${(b.p[1] - y0).toFixed(1)}, vmin=${vmin.toFixed(1)}, moved ${Math.hypot(b.p[0] - x0, b.p[2] - z0).toFixed(0)} m, crash=${b.crash} errMax=${deg(emax).toFixed(0)} heat=${b.heat.toFixed(2)} yaw=${deg(yprOf(b.q)[0]).toFixed(1)} roll=${deg(yprOf(b.q)[2]).toFixed(1)} vz=${b.v[1].toFixed(1)}`);
