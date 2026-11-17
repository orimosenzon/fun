// Aerobatic mode: the figures, the notches, the recovery and the safety net. node tests/acro.mjs
import { loadWorld, deg } from './util.mjs';
import { Sim } from '../js/sim/sim.js';
import { Game } from '../js/sim/game.js';
import { buildScenery } from '../js/sim/scenery.js';
import { G } from '../js/sim/bike.js';
import { T, surfaceAt } from '../js/sim/terrain.js';
import { mulberry32 } from '../js/sim/noise.js';
import { yprOf, len, qrot, qmul, qaxis } from '../js/sim/vmath.js';
const meta = loadWorld(); buildScenery();
let fail = 0;
const ok = (name, cond, info = '') => { console.log(`${cond ? 'ok  ' : 'FAIL'} ${name} ${info}`); if (!cond) fail++; };
const wrap = (a) => Math.atan2(Math.sin(a), Math.cos(a));
// a bike in level flight at V (m/s), high over the lake, already in aerobatic mode
function start(V, acro = true) {
  const s = new Sim(); s.reset([2500, 2400, 1500], 0, V || 0.01);
  const o = { s, b: s.bike, f: s.fcs, V, t: 0, pitch: 0, roll: 0, gmax: 0, vmin: 1e9, ymin: 1e9, ymax: -1e9, emax: 0 };
  o.inp = (x = {}) => ({ turn: 0, pitch: 0, strafe: 0, vt: V, ...x });
  // fly for secs with the keys x held, or until the predicate says stop
  o.fly = (secs, x, until) => {
    for (let i = 0, n = Math.round(secs * 120); i < n; i++) {
      s.tick(o.inp(x)); o.t += 1 / 120; o.pitch += o.b.w[2] / 120; o.roll += o.b.w[0] / 120;
      o.gmax = Math.max(o.gmax, Math.hypot(o.b.acc[0], o.b.acc[1] + G, o.b.acc[2]) / G); o.vmin = Math.min(o.vmin, len(o.b.v));
      o.ymin = Math.min(o.ymin, o.b.p[1] - o.y0); o.ymax = Math.max(o.ymax, o.b.p[1] - o.y0); if (o.f.acro) o.emax = Math.max(o.emax, o.f.tele.attErr);
      if (until && until()) return true;
    }
    return !until;
  };
  for (let i = 0; i < 360; i++) s.tick(o.inp({ noFloor: true }));
  o.y0 = o.b.p[1]; o.p0 = o.b.p.slice(); o.ymin = o.ymax = 0;
  if (acro) s.fcs.setAcro(true);
  return o;
}
const up = (o) => qrot(o.b.q, [0, 1, 0])[1], nose = (o) => qrot(o.b.q, [1, 0, 0]);
const head = (o) => deg(Math.atan2(-nose(o)[2], nose(o)[0]));

{ // 1. the ordinary law is untouched, and aerobatic mode is refused on the ground
  const s = new Sim(), pad = meta.pads.base; s.reset([pad[0], pad[1], pad[2]], 0);
  for (let i = 0; i < 120; i++) s.tick({ turn: 0, pitch: 0, vt: 0 });
  ok('refused on the ground', s.fcs.setAcro(true) === false && !s.fcs.acro);
}
{ // 2. a loop: hold the pull all the way round
  for (const V of [30, 55, 80]) {
    const o = start(V);
    const done = o.fly(20, { pitch: 1 }, () => o.pitch >= 2 * Math.PI - 0.12); const tLoop = o.t; o.fly(2, {});
    ok(`loop from ${(V * 3.6).toFixed(0)} km/h`, done && tLoop < 12 && o.gmax < 4.3 && up(o) > 0.995 && Math.abs(head(o)) < 3 && Math.abs(o.b.p[2] - o.p0[2]) < 4 && !o.b.crash && o.f.acro,
      `${tLoop.toFixed(1)} s, ${(o.ymax - o.ymin).toFixed(0)} m high, ${o.gmax.toFixed(1)} g, slowest ${(o.vmin * 3.6).toFixed(0)} km/h, ends ${(o.b.p[1] - o.y0).toFixed(0)} m up, heading ${head(o).toFixed(1)}, pitch ${deg(yprOf(o.b.q)[1]).toFixed(1)}`);
  }
}
{ // 3. a roll: hold the key, let go 35 degrees early, and the notch finishes it exactly
  for (const dir of [1, -1]) {
    const o = start(55);
    o.fly(5, { turn: dir }, () => Math.abs(o.roll) >= 2 * Math.PI - 0.6); const tRoll = o.t; o.fly(1.5, {});
    const [, pt, rl] = yprOf(o.b.q);
    ok(`roll ${dir > 0 ? 'right' : 'left'}`, tRoll < 2.2 && Math.abs(deg(rl)) < 1 && Math.abs(deg(pt)) < 1.5 && o.ymin > -9 && Math.abs(head(o)) < 4 && o.b.heat < 0.3,
      `${tRoll.toFixed(2)} s, ends at roll ${deg(rl).toFixed(1)} pitch ${deg(pt).toFixed(1)} heading ${head(o).toFixed(1)}, lowest ${o.ymin.toFixed(1)} m, heat ${o.b.heat.toFixed(2)}`);
  }
}
{ // 4. a four-point roll: every quarter stops within a few degrees of its point
  const o = start(55); let worst = 0;
  for (let k = 1; k <= 4; k++) { o.fly(3, { turn: 1 }, () => o.roll >= k * Math.PI / 2 - 0.55); o.fly(0.7, {}); worst = Math.max(worst, Math.abs(deg(wrap(yprOf(o.b.q)[2] - k * Math.PI / 2)))); }
  ok('four-point roll', worst < 4, `worst stop ${worst.toFixed(1)} deg off its point, ${(o.b.p[1] - o.y0).toFixed(0)} m lost, heat ${o.b.heat.toFixed(2)}`);
}
{ // 5. released, the attitude is held (away from the notches)
  const o = start(55);
  o.fly(0.35, { turn: 1 }); o.fly(0.5, {}); o.fly(0.45, { pitch: 1 }); o.fly(0.6, {});
  const a = yprOf(o.b.q); o.fly(1.5, {}); const c = yprOf(o.b.q);
  ok('holds a banked climb', Math.abs(deg(c[2] - a[2])) < 3 && Math.abs(deg(c[1] - a[1])) < 3 && Math.abs(deg(a[2])) > 12, `roll ${deg(a[2]).toFixed(0)} -> ${deg(c[2]).toFixed(0)}, pitch ${deg(a[1]).toFixed(0)} -> ${deg(c[1]).toFixed(0)}`);
}
{ // 6. Immelmann: half a loop, then half a roll on top. The way back, and higher
  const o = start(60);
  o.fly(10, { pitch: 1 }, () => o.pitch >= Math.PI - 0.2); o.fly(0.5, {});
  const inv = up(o); o.roll = 0;
  o.fly(3, { turn: 1 }, () => o.roll >= Math.PI - 0.6); o.fly(1.2, {});
  ok('Immelmann', inv < -0.97 && Math.abs(yprOf(o.b.q)[2]) < 0.03 && Math.abs(wrap((head(o) - 180) * Math.PI / 180)) < 0.17 && o.b.p[1] - o.y0 > 30, `on its back up=${inv.toFixed(2)}, then upright up=${up(o).toFixed(2)}, heading ${head(o).toFixed(0)}, ${(o.b.p[1] - o.y0).toFixed(0)} m higher, ${(len(o.b.v) * 3.6).toFixed(0)} km/h`);
}
{ // 7. split-S: half a roll, then pull through
  const o = start(45);
  o.fly(3, { turn: 1 }, () => o.roll >= Math.PI - 0.6); o.fly(0.3, {}); o.pitch = 0;
  o.fly(10, { pitch: 1 }, () => o.pitch >= Math.PI - 0.15); o.fly(1.5, {});
  ok('split-S', Math.abs(yprOf(o.b.q)[2]) < 0.03 && Math.abs(wrap((head(o) - 180) * Math.PI / 180)) < 0.17 && o.b.p[1] - o.y0 < -20, `upright up=${up(o).toFixed(2)}, heading ${head(o).toFixed(0)}, ${(o.y0 - o.b.p[1]).toFixed(0)} m lower, ${o.gmax.toFixed(1)} g`);
}
{ // 8. straight up: the notch holds the vertical, and a roll on the way up stays on the line
  const o = start(70);
  o.fly(6, { pitch: 1 }, () => o.pitch >= Math.PI / 2 - 0.08); o.fly(1.0, {});
  const n0 = nose(o), x0 = o.b.p[0], z0 = o.b.p[2], y0 = o.b.p[1]; o.roll = 0;
  o.fly(3, { turn: 1 }, () => o.roll >= 2 * Math.PI - 0.6); o.fly(0.8, {});
  const n1 = nose(o), dy = o.b.p[1] - y0, off = Math.hypot(o.b.p[0] - x0, o.b.p[2] - z0);
  ok('vertical line and vertical roll', n0[1] > 0.998 && n1[1] > 0.995 && off < 0.12 * dy && dy > 60, `nose up ${n0[1].toFixed(4)} -> ${n1[1].toFixed(4)}, climbed ${dy.toFixed(0)} m, ${off.toFixed(1)} m off the line, ${(len(o.b.v) * 3.6).toFixed(0)} km/h`);
}
{ // 9. a back-flip from the hover
  const o = start(0);
  const done = o.fly(10, { pitch: 1 }, () => o.pitch >= 2 * Math.PI - 0.1); const tf = o.t; o.fly(2.5, {});
  ok('back-flip from the hover', done && tf < 5 && up(o) > 0.99 && o.ymin > -35 && len(o.b.v) < 6, `${tf.toFixed(1)} s, ${(-o.ymin).toFixed(0)} m lost, ends at ${len(o.b.v).toFixed(1)} m/s`);
}
{ // 10. on its back nothing holds the bike up: it falls, and the nose goes down with the path
  const o = start(55);
  o.fly(3, { turn: 1 }, () => o.roll >= Math.PI - 0.6); o.fly(0.5, {}); const y1 = o.b.p[1];
  o.fly(2, {});
  ok('falls on its back (honest)', up(o) < -0.9 && y1 - o.b.p[1] > 12 && o.f.acro, `${(y1 - o.b.p[1]).toFixed(0)} m lost in 2 s, up=${up(o).toFixed(2)}`);
}
{ // 11. the rider leaves aerobatic mode in any attitude: the computer brings the bike back
  const cases = [['on its back', [1, 0, 0], Math.PI, 0], ['straight down', [0, 0, 1], -1.5, 0], ['straight up', [0, 0, 1], 1.5, 0], ['on its side', [1, 0, 0], 1.57, 0], ['on its back, nose down', [1, 0, 0], Math.PI, -0.8], ['a steep climb', [1, 0, 0], 0, 0.9], ['a steep climb on its back', [1, 0, 0], Math.PI, -0.9], ['on its side, nose up', [1, 0, 0], 1.57, 0.9]];
  for (const [name, ax, ang, pit] of cases) {
    const o = start(55, false);
    o.b.q = qmul(qmul(o.b.q, qaxis([0, 0, 1], pit)), qaxis(ax, ang)); o.b.v = qrot(o.b.q, [55, 0, 0]);
    o.f.setAcro(true); o.fly(0.2, {}); o.f.setAcro(false, 'rider');
    const y1 = o.b.p[1]; let tUp = null, low = 0;
    o.fly(8, {}, () => { low = Math.min(low, o.b.p[1] - y1); if (tUp === null && !o.f.acro && up(o) > 0.97 && Math.abs(o.b.v[1]) < 3) tUp = o.t - 0.2; return false; });
    ok(`recovery from ${name}`, tUp !== null && tUp < (pit > 0 ? 9 : 6) && low > -130 && !o.f.acro && up(o) > 0.98, `${tUp === null ? 'never' : tUp.toFixed(1) + ' s'}, ${(-low).toFixed(0)} m lost`);
  }
}
{ // 12. the safety net: a loop begun too low is taken away from the rider, and the bike is not lost
  const g = new Game(); g.load('free'); g.update(0.02, {}); g.events.length = 0;
  const L = T.lakes[0], x = (L.x0 + L.x1) / 2, z = (L.z0 + L.z1) / 2;
  // (on its back and let go; half a roll and a pull that is given up half way; the nose pushed down and held)
  for (const [name, h, keys] of [['left on its back at 40 m', 40, (t) => (t < 0.8 ? { turn: 1 } : {})], ['split-S given up, from 90 m', 90, (t) => (t < 0.8 ? { turn: 1 } : t < 2.6 ? { pitch: 1 } : {})], ['pushed into a dive from 150 m', 150, () => ({ pitch: -1 })]]) {
    g.sim.reset([x, L.level + h, z], 0, 55); g.state = 'fly'; g.prevP = g.bike.p.slice(); g.update(1 / 60, { vt: 55 }); g.toggleAcro(true);
    let net = null, low = 1e9, t = 0, events = [];
    while (t < 25 && g.state === 'fly') {
      const k = keys(t);
      g.update(1 / 60, { turn: 0, pitch: 0, strafe: 0, vt: 55, ...(g.acro ? k : {}) }); t += 1 / 60; low = Math.min(low, g.bike.agl);
      for (const e of g.events) { if (e.type === "acro" && e.why === "net") net = t; if (e.type === "crash") events.push(e.reason); }
      if (net !== null && t > net + 6) break;
      g.events.length = 0;
    }
    ok(`safety net: ${name}`, net !== null && !events.length && low > 1.5, `took over at ${net === null ? 'never' : net.toFixed(1) + ' s'}, lowest ${low.toFixed(1)} m ${events.join(',')}`);
  }
}
{ // 13. figures are counted, and a flat circle is not a loop
  const g = new Game(); g.load('free'); g.update(0.02, {}); g.events.length = 0;
  g.sim.reset([2500, 2400, 1500], 0, 55); g.state = 'fly'; g.prevP = g.bike.p.slice(); g.update(1 / 60, { vt: 55 }); g.toggleAcro(true);
  const tricks = []; let t = 0;
  const fly = (secs, k) => { for (let i = 0; i < secs * 60; i++) { g.update(1 / 60, { turn: 0, pitch: 0, strafe: 0, vt: 55, ...k }); t += 1 / 60; for (const e of g.events) if (e.type === 'trick') tricks.push(e.kind); g.events.length = 0; } };
  fly(7.2, { pitch: 1 }); fly(1.5, {}); fly(1.75, { turn: 1 }); fly(1.5, {});
  const a = tricks.join(',');
  fly(0.42, { turn: 1 }); fly(0.6, {}); fly(5.5, { pitch: 1 }); fly(1, {});          // on its side, and round in a flat circle
  ok('figures counted', a === 'loop,roll' && tricks.join(',') === a && g.score >= 650, `${tricks.join(',')} score ${Math.round(g.score)}`);
}
{ // 14. random aerobatics low over the lake with the net on: the bike is never lost
  let crashes = 0, nets = 0, mins = 0;
  const L = T.lakes[0], x = (L.x0 + L.x1) / 2, z = (L.z0 + L.z1) / 2;
  for (let seed = 1; seed <= 16; seed++) {
    const r = mulberry32(seed * 7919), g = new Game(); g.load('free'); g.update(0.02, {}); g.events.length = 0;
    const put = () => { g.sim.reset([x, L.level + 60 + r() * 240, z], r() * 6.28, 25 + r() * 70); g.state = 'fly'; g.prevP = g.bike.p.slice(); };
    put();
    let ctl = {}, next = 0, t = 0, back = 0;
    while (t < 45 && g.state === 'fly') {
      if (!g.acro && !g.sim.fcs.rec && t > back) g.toggleAcro(true);
      const was = g.acro;
      if (t >= next) {
        const pick = () => [-1, -1, 0, 0, 1, 1, 1][(r() * 7) | 0];
        ctl = { turn: r() < 0.6 ? pick() : 0, pitch: r() < 0.7 ? pick() : 0, strafe: r() < 0.15 ? pick() : 0, vt: [20, 40, 55, 70, 95][(r() * 5) | 0], boost: r() < 0.2 };
        if (Math.hypot(g.bike.p[0] - x, g.bike.p[2] - z) > 300 || g.bike.agl > 500) put();
        next = t + 0.3 + r() * 2.2;
      }
      g.update(1 / 60, ctl); t += 1 / 60; if (g.acro) mins += 1 / 3600;
      for (const e of g.events) { if (e.type === 'acro' && e.why === 'net') { nets++; back = t + 2.5; } if (e.type === 'crash') { crashes++; console.log(`     seed ${seed} t=${t.toFixed(1)} crash ${e.reason}, ${was ? 'in aerobatic mode' : (t - back + 2.5).toFixed(1) + ' s after the net'} V=${len(e.v).toFixed(0)} vz=${e.v[1].toFixed(0)}`); } }
      g.events.length = 0;
    }
  }
  ok('random aerobatics with the net', crashes === 0 && nets > 20, `${mins.toFixed(1)} min, the net took over ${nets} times, ${crashes} crashes`);
}
console.log(fail ? `${fail} FAILED` : 'ALL PASS');
process.exit(fail ? 1 : 0);
