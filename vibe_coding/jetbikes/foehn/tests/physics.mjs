// Physics and flight-control checks. node tests/physics.mjs
import { loadWorld, deg } from './util.mjs';
import { Sim } from '../js/sim/sim.js';
import { G, aeroWrench, mkW, WING, massProps, SURF } from '../js/sim/bike.js';
import { yprOf, len } from '../js/sim/vmath.js';
import { makeWind } from '../js/sim/wind.js';
const meta = loadWorld();
let fail = 0;
const ok = (name, cond, info = '') => { console.log(`${cond ? 'ok  ' : 'FAIL'} ${name} ${info}`); if (!cond) fail++; };
const run = (sim, secs, inp, each) => { const n = Math.round(secs * 120); for (let i = 0; i < n; i++) { sim.tick(typeof inp === 'function' ? inp(sim.t) : inp); if (each) each(); } };
const air = (v = 0) => { const s = new Sim(); s.reset([2500, 2400, 1500], 0, v || 0.01); return s; };

{ // 1. hover: thrust carries exactly the weight
  const s = air(); run(s, 8, { turn: 0, pitch: 0, vt: 0, noFloor: true });
  const b = s.bike, lift = b.thrust * Math.sin(b.tilt), W = b.mp.m * G;
  ok('hover holds height', Math.abs(b.v[1]) < 0.1 && len(b.v) < 0.5, `vz=${b.v[1].toFixed(3)} V=${len(b.v).toFixed(2)}`);
  ok('hover thrust = weight', Math.abs(lift / W - 1) < 0.02, `T sin(tilt)/W=${(lift / W).toFixed(3)}`);
}
{ // 2. cruise: the wing carries most of the weight, height is held
  const s = air(75); let y0;
  run(s, 6, { turn: 0, pitch: 0, vt: 75, noFloor: true }); y0 = s.bike.p[1];
  run(s, 6, { turn: 0, pitch: 0, vt: 75, noFloor: true });
  const b = s.bike, A = aeroWrench(b.vAirB, b.w, b.rho, b, b.mp.cg, mkW());
  ok('cruise holds height', Math.abs(b.p[1] - y0) < 1.5, `dy=${(b.p[1] - y0).toFixed(2)}`);
  ok('cruise is wing-borne', A.F[1] / (b.mp.m * G) > 0.8 && b.tilt < 0.1, `aero lift/W=${(A.F[1] / (b.mp.m * G)).toFixed(2)} tilt=${deg(b.tilt).toFixed(0)}`);
}
{ // 3. hard turn: load factor matches the bank, no height lost
  const s = air(70); run(s, 3, { turn: 0, pitch: 0, vt: 70, noFloor: true });
  const y0 = s.bike.p[1]; let gmax = 0;
  run(s, 6, { turn: 1, pitch: 0, vt: 70, noFloor: true }, () => { gmax = Math.max(gmax, Math.hypot(s.bike.acc[0], s.bike.acc[1] + G, s.bike.acc[2]) / G); });
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
{ // 6. strip aerodynamics: lift slope below the stall, lift loss beyond it
  const cg = massProps(60).cg, w = [0, 0, 0], ctl = { da: 0, de: 0, dr: 0, brake: 0 }, V = 60, rho = 1.1;
  const L = (a) => { const o = aeroWrench([V * Math.cos(a), -V * Math.sin(a), 0], w, rho, ctl, cg, mkW()); return o.F[1] * Math.cos(a) + o.F[0] * Math.sin(a); };   // wind-axis lift
  const slope = (L(0.04) - L(0.0)) / 0.04 / (0.5 * rho * V * V), ref = SURF.filter((s) => s.n[1]).reduce((a, s) => a + s.S * s.cla, 0);
  ok('lift slope matches finite-wing theory', Math.abs(slope / ref - 1) < 0.08, `dCL*S/da=${slope.toFixed(2)} theory=${ref.toFixed(2)}`);
  const post = (L(0.36) - L(0.26)) / 0.1 / (0.5 * rho * V * V);
  ok('lift curve breaks at the stall', post < 0.35 * slope, `slope beyond the stall=${post.toFixed(2)}`);
}
{ // 7. hover in gusty wind stays put; the run is deterministic
  const mk = () => { const s = new Sim(makeWind({ speed: 8, gust: 1.5, seed: 3 })); s.reset([2500, 2400, 1500], 0, 0.01); run(s, 20, { turn: 0, pitch: 0, vt: 0, noFloor: true }); return s.bike; };
  const a = mk(), b = mk();
  ok('holds position in 8 m/s gusty wind', Math.hypot(a.p[0] - 2500, a.p[2] - 1500) < 25 && Math.abs(a.p[1] - 2400) < 4, `drift=${Math.hypot(a.p[0] - 2500, a.p[2] - 1500).toFixed(1)} m dy=${(a.p[1] - 2400).toFixed(1)}`);
  ok('deterministic', a.p[0] === b.p[0] && a.q[1] === b.q[1]);
}
{ // 8. thin air: the hover margin shrinks with altitude
  const m = (alt) => { const s = new Sim(); s.reset([2500, alt, 1500], 0, 0.01); run(s, 5, { turn: 0, pitch: 1, vt: 0, noFloor: true }); return s.bike.v[1]; };
  const lo = m(1500), hi = m(3600);
  ok('climb rate falls with altitude', hi < lo, `1500 m: ${lo.toFixed(1)} m/s, 3600 m: ${hi.toFixed(1)} m/s`);
}
console.log(fail ? `${fail} FAILED` : 'ALL PASS'); process.exit(fail ? 1 : 0);
