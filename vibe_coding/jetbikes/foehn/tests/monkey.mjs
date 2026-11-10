// Random-input stress test: the flight computer must never lose the bike, whatever the rider does.
import { loadWorld, deg } from './util.mjs';
import { Sim } from '../js/sim/sim.js';
import { makeWind } from '../js/sim/wind.js';
import { yprOf, len } from '../js/sim/vmath.js';
import { mulberry32 } from '../js/sim/noise.js';
const meta = loadWorld();
const seeds = +(process.argv[2] || 6), dur = +(process.argv[3] || 120);
let bad = 0;
for (let seed = 1; seed <= seeds; seed++) {
  const r = mulberry32(seed * 977);
  const sim = new Sim(makeWind({ speed: 3 + 6 * r(), gust: 0.5 + 1.5 * r(), seed }));
  sim.reset([2500, 3000, 1500], r() * 6.28, r() < 0.5 ? 0.01 : 60);   // high above the lake
  const b = sim.bike;
  let inp = { turn: 0, pitch: 0, vt: 40, strafe: 0, boost: false, brake: false, noFloor: true }, next = 0;
  const hist = [];
  let bad_t = 0;
  let maxErr = 0, maxG = 0, maxW = 0, lostAt = null, maxSt = 0;
  while (sim.t < dur) {
    if (sim.t >= next) {
      const pick = () => [-1, -1, 0, 0, 0, 1, 1][(r() * 7) | 0];
      inp = { turn: pick(), pitch: pick() * (b.p[1] < 2200 ? -0.2 : 1) , strafe: r() < 0.2 ? pick() : 0, vt: [0, 0, 20, 45, 70, 95, 95][(r() * 7) | 0], boost: r() < 0.25, brake: r() < 0.15, noFloor: true };
      if (b.p[1] < 1900) inp.pitch = 1; if (b.p[1] > 3600) inp.pitch = -1;
      next = sim.t + 0.2 + r() * 2.5;
    }
    sim.tick(inp);
    if (process.env.DUMP && seed === +process.env.DUMP && !lostAt) {
      hist.push(`t=${sim.t.toFixed(2)} in=${inp.turn},${inp.pitch},vt${inp.vt},s${inp.strafe},${inp.boost ? 'B' : ''}${inp.brake ? 'K' : ''} V=${b.airspeed.toFixed(0)} vB=${b.vAirB.map(v => v.toFixed(0))} vz=${b.v[1].toFixed(0)} ypr=${yprOf(b.q).map(v => deg(v).toFixed(0))} des=${deg(sim.fcs.tele.thetaDes).toFixed(0)},${deg(sim.fcs.tele.phiDes).toFixed(0)} err=${deg(sim.fcs.tele.attErr).toFixed(0)} wb=${sim.fcs.tele.wb.toFixed(2)} a=${sim.fcs.tele.aCmd.map(v=>v.toFixed(0))} w=${b.w.map(v => v.toFixed(1))} tilt=${deg(b.tilt).toFixed(0)} T=${b.eng.map(e => e.T.toFixed(0))} va=${b.va.map(v => deg(v).toFixed(0))} vb=${b.vb.map(v => deg(v).toFixed(0))} s=${deg(b.da).toFixed(0)},${deg(b.de).toFixed(0)},${deg(b.dr).toFixed(0)} st=${b.stall.toFixed(1)} sat=${sim.fcs.tele.sat}\n      tau=${sim.fcs.tele.tau.map(v=>v.toFixed(0))} tq=${sim.fcs.tele.tq.map(v=>v.toFixed(0))} F=${sim.fcs.tele.F.map(v=>v.toFixed(0))} cmdT=${sim.fcs.cmd.T.map(v=>v.toFixed(0))} gfx=${b.gfx.map(v=>v.toFixed(2))} N=${b.eng.map(e=>e.N.toFixed(2))}`);
    }
    const te = sim.fcs.tele;
    const err = te.attErr || 0;
    maxErr = Math.max(maxErr, err); maxW = Math.max(maxW, len(b.w)); maxSt = Math.max(maxSt, b.stall);
    maxG = Math.max(maxG, Math.hypot(b.acc[0], b.acc[1] + 9.81, b.acc[2]) / 9.81);
    bad_t = err > 1.0 ? bad_t + 1 / 120 : 0;
    if (!lostAt && (bad_t > 2 || len(b.w) > 5 || b.crash)) lostAt = sim.t;
    if (b.p[1] < 2300) b.p[1] += 900;
    // keep it over the lake
    if (Math.hypot(b.p[0] - 2500, b.p[2] - 1500) > 2500) { b.p[0] = 2500; b.p[2] = 1500; }
  }
  if (lostAt) bad++;
  if (hist.length) console.log(hist.filter((_, i) => i % 18 === 0).slice(-(+process.env.N || 14)).join('\n'));
  console.log(`seed ${seed}: maxAttErr=${deg(maxErr).toFixed(0)}deg maxRate=${deg(maxW).toFixed(0)}deg/s maxG=${maxG.toFixed(1)} stall=${maxSt.toFixed(1)} ${lostAt ? 'LOST CONTROL at t=' + lostAt.toFixed(1) : 'ok'} fuel=${b.fuel.toFixed(0)}`);
}
console.log(bad ? `FAIL: ${bad} runs lost control` : 'PASS');
process.exit(bad ? 1 : 0);
