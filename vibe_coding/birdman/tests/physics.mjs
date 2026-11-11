// Flight-physics checks in still air. node tests/physics.mjs [-v]
import { initWorld, groundAt } from '../js/sim/world.js';
import { Bird, P_MAX } from '../js/sim/bird.js';
import { MASS, G, WING_AREA, liftDrag } from '../js/sim/aero.js';
initWorld();
const verbose = process.argv.includes('-v');
const deg = (r) => (r * 180) / Math.PI;
let fails = 0;
const check = (name, ok, info) => { console.log(`${ok ? 'ok  ' : 'FAIL'} ${name}  ${info}`); if (!ok) fails++; };
const START = [-3600, 900, -2500]; // over open sea

function fly(inp, T, opt = {}) {
  const b = new Bird(); b.windOn = false;
  b.placeAir(opt.p || START, opt.heading ?? 0, opt.speed ?? 13);
  if (opt.setup) opt.setup(b);
  const dt = 1 / 60, log = [];
  for (let t = 0; t < T && b.mode === 'air'; t += dt) {
    const i = typeof inp === 'function' ? inp(t, b) : inp;
    b.step(dt, { pitch: 0, roll: 0, fwd: 0, flap: false, tuck: false, jump: false, ...i }, t);
    log.push({ t, V: b.airspeed, vy: b.v[1], y: b.p[1], al: b.alpha, bank: b.bank, pitch: b.pitch, P: b.power, g: b.gload, x: b.p[0], z: b.p[2], st: b.stamina, om: [...b.om] });
    if (verbose && opt.trace == process.env.TRACE && Math.round(t * 60) % 30 === 0) console.log(`   t=${t.toFixed(1)} V=${b.airspeed.toFixed(1)} vy=${b.v[1].toFixed(2)} al=${deg(b.alpha).toFixed(1)} pitch=${deg(b.pitch).toFixed(1)} bank=${deg(b.bank).toFixed(1)} uP=${b.uP.toFixed(2)} uR=${b.uR.toFixed(2)} om=${b.om.map((x) => x.toFixed(2))} beta=${deg(b.beta).toFixed(1)} st=${b.stall.toFixed(2)} P=${b.power.toFixed(0)} g=${b.gload.toFixed(2)} y=${b.p[1].toFixed(0)}`);
  }
  return { b, log };
}
const avg = (log, f, from) => { const l = log.filter((r) => r.t >= from); return l.reduce((s, r) => s + f(r), 0) / l.length; };
const nan = (b) => !isFinite(b.p[0] + b.p[1] + b.p[2] + b.v[0] + b.q[0]);

console.log(`wing area ${WING_AREA.toFixed(1)} m2, wing loading ${(MASS * G / WING_AREA).toFixed(0)} N/m2, CLmax ~${Math.max(...Array.from({ length: 40 }, (_, i) => liftDrag(i * 0.01)[0])).toFixed(2)}`);

{ // 1. hands-off glide
  const { b, log } = fly({}, 40, { trace: 1 });
  const V = avg(log, (r) => r.V, 25), vy = avg(log, (r) => r.vy, 25), ld = Math.sqrt(V * V - vy * vy) / -vy;
  const wob = Math.max(...log.filter((r) => r.t > 25).map((r) => Math.abs(r.vy - vy)));
  check('glide trim', V > 10 && V < 15 && ld > 9 && ld < 16 && wob < 0.3 && !nan(b), `V=${(V * 3.6).toFixed(0)} km/h sink=${(-vy).toFixed(2)} m/s L/D=${ld.toFixed(1)} alpha=${deg(avg(log, (r) => r.al, 25)).toFixed(1)} wobble=${wob.toFixed(2)}`);
}
{ // 2. full pull: slow flight, no departure
  const { b, log } = fly({ pitch: 1 }, 30, { trace: 2 });
  const V = avg(log, (r) => r.V, 20), vy = avg(log, (r) => r.vy, 20);
  check('slow flight', V > 6.5 && V < 10.5 && vy > -3 && Math.abs(b.bank) < 0.3 && !nan(b), `V=${(V * 3.6).toFixed(0)} km/h sink=${(-vy).toFixed(2)} alpha=${deg(avg(log, (r) => r.al, 20)).toFixed(1)}`);
}
{ // 3. push: fast glide
  const { b, log } = fly({ pitch: -0.5 }, 25);
  const V = avg(log, (r) => r.V, 18);
  check('fast glide', V > 18 && V < 45 && !nan(b), `V=${(V * 3.6).toFixed(0)} km/h sink=${(-avg(log, (r) => r.vy, 18)).toFixed(1)}`);
}
{ // 4. stoop: tuck and fall like a falcon
  const { b, log } = fly({ tuck: true, pitch: -1 }, 22, { p: [-3600, 2600, -2500] });
  const V = log[log.length - 1].V;
  check('stoop terminal speed', V > 55 && V < 100 && !nan(b), `V=${(V * 3.6).toFixed(0)} km/h after ${log[log.length - 1].t.toFixed(0)} s, lost ${(2600 - b.p[1]).toFixed(0)} m`);
}
{ // 5. stoop then spread: pull-out g and height used
  let tSpread = 9, y0 = 0;
  const { b, log } = fly((t, bb) => { if (t < tSpread) return { tuck: true, pitch: -1 }; if (!y0) y0 = bb.p[1]; return { pitch: 0.7 }; }, 20, { p: [-3600, 2000, -2500] });
  const gmax = Math.max(...log.map((r) => r.g)), ymin = Math.min(...log.map((r) => r.y));
  check('pull-out', gmax > 2.5 && gmax < 9 && !nan(b) && b.mode === 'air', `Vdive=${(log.find((r) => r.t >= tSpread).V * 3.6).toFixed(0)} km/h gmax=${gmax.toFixed(1)} height used=${(y0 - ymin).toFixed(0)} m, end V=${(b.airspeed * 3.6).toFixed(0)}`);
}
{ // 6. flapping: climb and power
  const { b, log } = fly({ flap: true, pitch: 0.2 }, 30, { trace: 6 });
  const vy = avg(log, (r) => r.vy, 8), V = avg(log, (r) => r.V, 8), P = avg(log, (r) => Math.max(r.P, 0), 8), pk = Math.max(...log.map((r) => r.P));
  check('flapping climb', vy > 1.2 && vy < 6 && P < P_MAX * 1.05 && !nan(b), `climb=${vy.toFixed(2)} m/s V=${(V * 3.6).toFixed(0)} km/h mean power=${P.toFixed(0)} W peak=${pk.toFixed(0)} W stamina left=${b.stamina.toFixed(2)}`);
}
{ // 7. roll response and turn
  const { b, log } = fly({ roll: 1, pitch: 0.3 }, 14, { trace: 7 });
  const t45 = (log.find((r) => r.bank > 0.78) || {}).t, bank = avg(log, (r) => r.bank, 8);
  const h0 = Math.atan2(-b.v[2], b.v[0]);
  const l1 = log[log.length - 1], l0 = log[log.length - 121];
  const tr = Math.abs(Math.atan2(Math.sin((l1.om[1])), 1)); // placeholder
  const R = (() => { const a = log.filter((r) => r.t > 8); const xs = a.map((r) => r.x), zs = a.map((r) => r.z); return (Math.max(...xs) - Math.min(...xs) + Math.max(...zs) - Math.min(...zs)) / 4; })();
  check('roll and turn', t45 < 2.5 && bank > 0.9 && bank < 1.4 && R < 60 && !nan(b), `45deg in ${t45?.toFixed(2)} s, holds ${deg(bank).toFixed(0)} deg, turn radius ~${R.toFixed(0)} m, V=${(b.airspeed * 3.6).toFixed(0)}`);
}
{ // 8. roll reversal
  const { b, log } = fly((t) => ({ roll: t < 4 ? 1 : -1 }), 8);
  check('roll reversal', b.bank < -0.5 && !nan(b), `bank at end ${deg(b.bank).toFixed(0)}`);
}
{ // 9. jump off the cliff, spread late
  const { b, log } = fly((t) => ({ flap: t > 2.2 && t < 2.4 }), 14, { setup: (bb) => { bb.fold = 0; bb.latch = false; bb.L.fold = bb.R.fold = 0; bb.v = [6, 3, 0]; }, p: [-3600, 300, -2500] });
  const ymin = Math.min(...log.map((r) => r.y)), gmax = Math.max(...log.map((r) => r.g));
  check('late spread after a jump', ymin > 200 && Math.abs(log[log.length - 1].vy) < 2.5 && !nan(b), `fell ${(300 - ymin).toFixed(0)} m, gmax=${gmax.toFixed(1)}, end V=${(b.airspeed * 3.6).toFixed(0)} vy=${b.v[1].toFixed(1)}`);
}
{ // 10. take off from flat ground: a flapping run, then climb away
  const b = new Bird(); b.windOn = false; b.placeGround(250, 950, 0);
  const dt = 1 / 60; let t = 0, y0 = b.p[1], tOff = 0, ymin = 1e9;
  for (; t < 16 && b.mode !== 'crash'; t += dt) { b.step(dt, { pitch: b.mode === 'air' ? 0.3 : 0, roll: 0, fwd: 1, flap: true, tuck: false, jump: false }, t); if (b.mode === 'air') { if (!tOff) tOff = t; ymin = Math.min(ymin, b.p[1] - groundAt(b.p[0], b.p[2])); } }
  check('ground take-off', b.mode === 'air' && b.p[1] - y0 > 8 && ymin > 0.5, `mode=${b.mode} lift-off after ${tOff.toFixed(1)} s, lowest ${ymin.toFixed(1)} m, height gained=${(b.p[1] - y0).toFixed(1)} m at ${t.toFixed(0)} s, V=${(b.airspeed * 3.6).toFixed(0)}`);
}
{ // 11. landing: glide down to the plain and flare
  const b = new Bird(); b.windOn = false; b.placeAir([-300, 26 + 40, 950], 0, 12);
  const dt = 1 / 60; let t = 0, ev = null;
  for (; t < 40 && b.mode === 'air'; t += dt) { b.step(dt, { pitch: b.agl < 3.5 ? 1 : b.agl < 12 ? 0.3 : 0, roll: 0, fwd: 0, flap: false, tuck: false, jump: false }, t); }
  ev = b.events.find((e) => e.type === 'land' || e.type === 'crash');
  check('flare and land', b.mode === 'ground' && ev.type === 'land', `${ev?.type} vn=${ev?.vn?.toFixed(1)} vh=${ev?.vh?.toFixed(1)} quality=${ev?.q?.toFixed(2)}`);
}
{ // 12. energy: nothing blows up at speed
  const { b, log } = fly((t) => ({ tuck: t < 12, pitch: t < 12 ? -1 : Math.sin(t * 3), roll: Math.sin(t * 2.2) }), 30, { p: [-3600, 2600, -2500] });
  check('abuse at speed', !nan(b) && Math.max(...log.map((r) => Math.abs(r.g))) < 14, `gmax=${Math.max(...log.map((r) => Math.abs(r.g))).toFixed(1)} mode=${b.mode}`);
}
console.log(fails ? `${fails} FAILED` : 'all passed');
process.exit(fails ? 1 : 0);
