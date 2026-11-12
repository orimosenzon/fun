// Physics and flight-control checks. node tests/physics.mjs
import { loadWorld, deg } from './util.mjs';
import { Sim } from '../js/sim/sim.js';
import { G, aeroWrench, mkW } from '../js/sim/bike.js';
import { yprOf, len } from '../js/sim/vmath.js';
import { makeWind } from '../js/sim/wind.js';
const meta = loadWorld();
let fail = 0;
const ok = (name, cond, info = '') => { console.log(`${cond ? 'ok  ' : 'FAIL'} ${name} ${info}`); if (!cond) fail++; };
const run = (sim, secs, inp, each) => { const n = Math.round(secs * 120); for (let i = 0; i < n; i++) { sim.tick(typeof inp === 'function' ? inp(sim.t) : inp); if (each) each(); } };
const air = (v = 0) => { const s = new Sim(); s.reset([2500, 2400, 1500], 0, v || 0.01); return s; };

{ // 1. hover: thrust carries exactly the weight
  const s = air(); run(s, 8, { turn: 0, pitch: 0, vt: 0, noFloor: true });
  const b = s.bike, lift = b.lift, W = b.mp.m * G;
  ok('hover holds height', Math.abs(b.v[1]) < 0.1 && len(b.v) < 0.5, `vz=${b.v[1].toFixed(3)} V=${len(b.v).toFixed(2)}`);
  ok('hover thrust = weight', Math.abs(lift / W - 1) < 0.02, `lift pipes/W=${(lift / W).toFixed(3)} rear=${b.R.toFixed(0)} N`);
}
{ // 2. cruise: no wing, so the lift nozzles still carry the whole weight; height is held
  const s = air(75); let y0;
  run(s, 6, { turn: 0, pitch: 0, vt: 75, noFloor: true }); y0 = s.bike.p[1];
  run(s, 6, { turn: 0, pitch: 0, vt: 75, noFloor: true });
  const b = s.bike, A = aeroWrench(b.vAirB, b.w, b.rho, b, b.mp.cg, mkW()), W = b.mp.m * G;
  ok('cruise holds height and speed', Math.abs(b.p[1] - y0) < 1.5 && Math.abs(len(b.v) - 75) < 2, `dy=${(b.p[1] - y0).toFixed(2)} V=${len(b.v).toFixed(1)}`);
  ok('cruise is thrust-borne', Math.abs(A.F[1]) / W < 0.1 && Math.abs(b.lift / W - 1) < 0.12, `aero lift/W=${(A.F[1] / W).toFixed(2)} lift nozzles/W=${(b.lift / W).toFixed(2)} pitch=${deg(yprOf(b.q)[1]).toFixed(1)}`);
}
{ // 3. hard turn: load factor matches the bank, no height lost
  const s = air(70); run(s, 3, { turn: 0, pitch: 0, vt: 70, noFloor: true });
  const y0 = s.bike.p[1]; let gmax = 0;
  run(s, 6, { turn: 0.85, pitch: 0, vt: 70, noFloor: true }, () => { gmax = Math.max(gmax, Math.hypot(s.bike.acc[0], s.bike.acc[1] + G, s.bike.acc[2]) / G); });
  const roll = yprOf(s.bike.q)[2];
  ok('turn: bank and load factor agree', Math.abs(1 / Math.cos(roll) - gmax) < 0.5 && gmax > 2.0, `bank=${deg(roll).toFixed(0)} n=${gmax.toFixed(2)} 1/cos=${(1 / Math.cos(roll)).toFixed(2)}`);
  ok('turn holds height', Math.abs(s.bike.p[1] - y0) < 6, `dy=${(s.bike.p[1] - y0).toFixed(1)}`);
}
{ // 4. braking from cruise to the hover
  const s = air(75); run(s, 2, { turn: 0, pitch: 0, vt: 75, noFloor: true });
  let t = 0; run(s, 14, { turn: 0, pitch: 0, vt: 0, brake: true, noFloor: true }, () => { if (len(s.bike.v) > 5) t = s.t - 2; });
  ok('brake 270 km/h -> hover', t < 9, `${t.toFixed(1)} s`);
}
{ // 5. take-off and automatic flare: touch-down is gentle
  const s = new Sim(), p = meta.pads.base; s.reset([p[0], p[1], p[2]], 0);
  run(s, 4, (t) => ({ turn: 0, pitch: t > 0.5 ? 1 : 0, vt: 0 }));
  ok('lifts off', s.bike.agl > 8, `agl=${s.bike.agl.toFixed(1)}`);
  let vTouch = null; run(s, 25, { turn: 0, pitch: -1, vt: 0 }, () => { if (vTouch === null && s.bike.contactN > 0) vTouch = s.bike.v[1]; });
  ok('lands softly, no crash', vTouch !== null && vTouch > -1.6 && !s.bike.crash, `touch-down ${vTouch && vTouch.toFixed(2)} m/s`);
  ok('idles on the ground', s.bike.thrust < 400, `T=${s.bike.thrust.toFixed(0)} N`);
}
{ // 6. U-turn at 200 km/h: tight, and no height lost
  const s = air(55); run(s, 3, { turn: 0, pitch: 0, vt: 55, noFloor: true });
  const b = s.bike, y0 = b.p[1], h0 = Math.atan2(-b.v[2], b.v[0]), z0 = b.p[2]; let t = 0, w = 0;
  while (t < 20) { s.tick({ turn: 1, pitch: 0, vt: 55, noFloor: true }); t += 1 / 120; w = Math.max(w, Math.abs(b.p[2] - z0)); if (Math.cos(Math.atan2(-b.v[2], b.v[0]) - h0) < -0.999) break; }
  ok('U-turn from 200 km/h', t < 5 && w < 120 && Math.abs(b.p[1] - y0) < 4, `${t.toFixed(1)} s, ${w.toFixed(0)} m wide, dy=${(b.p[1] - y0).toFixed(1)}`);
}
{ // 7. hover in gusty wind stays put; the run is deterministic
  const mk = () => { const s = new Sim(makeWind({ speed: 8, gust: 1.5, seed: 3 })); s.reset([2500, 2400, 1500], 0, 0.01); run(s, 20, { turn: 0, pitch: 0, vt: 0, noFloor: true }); return s.bike; };
  const a = mk(), b = mk();
  ok('holds position in 8 m/s gusty wind', Math.hypot(a.p[0] - 2500, a.p[2] - 1500) < 25 && Math.abs(a.p[1] - 2400) < 4, `drift=${Math.hypot(a.p[0] - 2500, a.p[2] - 1500).toFixed(1)} m dy=${(a.p[1] - 2400).toFixed(1)}`);
  ok('deterministic', a.p[0] === b.p[0] && a.q[1] === b.q[1]);
}
{ // 8. thin air: the engine still breathes, so it pulls less up high
  const m = (alt) => { const s = new Sim(); s.reset([2500, alt, 1500], 0, 40); let T = 0; run(s, 4, { turn: 1, pitch: 1, vt: 95, boost: true, noFloor: true }, () => { T = Math.max(T, s.bike.thrust); }); return T; };
  const lo = m(1500), hi = m(3600);
  ok('thrust falls with altitude', hi < 0.9 * lo, `everything asked at once: ${(lo / 1000).toFixed(1)} kN at 1500 m, ${(hi / 1000).toFixed(1)} kN at 3600 m`);
}
console.log(fail ? `${fail} FAILED` : 'ALL PASS'); process.exit(fail ? 1 : 0);
