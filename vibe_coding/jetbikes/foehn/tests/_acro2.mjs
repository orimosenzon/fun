// roll stop: hold the roll key for T seconds, then let go. node tests/_acro2.mjs T [key=value]
import { loadWorld, deg } from './util.mjs';
import { Sim } from '../js/sim/sim.js';
import { FEEL, ACRO } from '../js/sim/fcs.js';
import { yprOf } from '../js/sim/vmath.js';
loadWorld();
const T = +(process.argv[2] || 0.5);
for (const a of process.argv.slice(3)) { const [k, v] = a.split('='); if (k in FEEL) FEEL[k] = +v; if (k in ACRO) ACRO[k] = +v; }
const s = new Sim(); s.reset([2500, 2400, 1500], 0, 55);
const base = { turn: 0, pitch: 0, strafe: 0, vt: 55, noFloor: true };
for (let i = 0; i < 360; i++) s.tick(base);
s.fcs.setAcro(true);
let ang = 0, out = [], satMax = 0;
for (let i = 0; i < 240; i++) { s.tick({ ...base, turn: i < T * 120 ? 1 : 0 }); ang += s.bike.w[0] / 120; satMax = Math.max(satMax, s.fcs.tele.sat); if (i % 6 === 5) out.push(`${((i + 1) / 120).toFixed(2)}:${deg(ang).toFixed(0)}`); }
console.log(out.join(' '), `| dy=${(s.bike.p[1] - 2400).toFixed(1)}`);
