// Experiment: how does a landing go for a rider with a keyboard? node tests/_land.mjs
import { loadWorld } from './util.mjs';
import { Game } from '../js/sim/game.js';
import { buildScenery } from '../js/sim/scenery.js';
import { len } from '../js/sim/vmath.js';
import { groundAt } from '../js/sim/terrain.js';
const ANG = +(process.argv[2] || 0) * Math.PI / 180;
loadWorld(); buildScenery();
function run(name, V0, H0, policy, evId = 'first') {
  const g = new Game(); g.load(evId);
  while (g.state !== 'fly') { g.update(1 / 60, {}); g.events.length = 0; }
  g.cp = g.cps.findIndex((c) => c.type === 'land'); const c = g.cps[g.cp];
  const ux = Math.cos(ANG), uz = -Math.sin(ANG), sx = c.p[0] - 600 * ux, sz = c.p[2] - 600 * uz;
  g.respawnPose = { p: [sx, Math.max(c.p[1], groundAt(sx, sz)) + H0, sz], yaw: ANG, speed: V0, pitch: 0 }; g.respawn(); g.vt = V0;
  while (g.state !== 'fly') { g.update(1 / 60, { vt: V0 }); g.events.length = 0; }
  const b = g.bike; let t = 0, out = 'timeout', tBrake = -1, vzTouch = 0, minAgl = 99, log = [];
  const st = {};
  while (t < 90) {
    const d = (c.p[0] - b.p[0]) * ux + (c.p[2] - b.p[2]) * uz, ctl = policy({ d, V: len(b.v), agl: b.agl, vz: b.v[1], t, st, b, off: Math.hypot(c.p[0] - b.p[0], c.p[2] - b.p[2]) });
    if (ctl.brake && tBrake < 0) tBrake = t;
    const was = b.contactN;
    g.update(1 / 60, ctl); t += 1 / 60;
    if (!was && b.contactN) vzTouch = Math.min(vzTouch, b.v[1]);
    minAgl = Math.min(minAgl, b.agl);
    for (const e of g.events) { if (e.type === 'crash') out = 'CRASH ' + e.reason; if (e.type === 'land') out = 'landed'; if (e.type === 'warn') log.push(e.text); }
    g.events.length = 0;
    if (out !== 'timeout') break;
  }
  console.log(`${name.padEnd(46)} ${out.padEnd(14)} t=${t.toFixed(1)}s from brake=${tBrake < 0 ? '-' : (t - tBrake).toFixed(1)}s  off-centre=${Math.hypot(c.p[0] - b.p[0], c.p[2] - b.p[2]).toFixed(1)}m (r=${c.r})  agl=${b.agl.toFixed(1)} V=${len(b.v).toFixed(1)} vt=${g.vt.toFixed(1)} ${[...new Set(log)].join(',')}`);
}
const K = { throttle: 0, pitch: 0, turn: 0, strafe: 0 };
for (const V0 of [25, 45, 70]) for (const D of [60, 120, 200, 300]) {
  run(`V=${V0} brake+descend held from ${D}m`, V0, 40, ({ d }) => (d < D ? { ...K, brake: true, pitch: -1 } : { ...K }));
}
for (const V0 of [25, 45]) for (const D of [120, 200]) {
  run(`V=${V0} brake at ${D}m, descend when stopped`, V0, 40, ({ d, V }) => (d < D ? { ...K, brake: true, pitch: V < 3 ? -1 : 0 } : { ...K }));
  run(`V=${V0} descend early (no brake till ${D}m)`, V0, 40, ({ d }) => ({ ...K, pitch: -1, brake: d < D }));
}
for (const V0 of [25, 45, 60, 75, 90]) run(`V=${V0} descend held, no brake`, V0, 40, () => ({ ...K, pitch: -1 }));
// overshoot: stop 40 m past, then come back with S (reverse)
run('V=45 overshoot, reverse with S, descend', 45, 40, ({ d, V, st }) => { if (d < 60) st.b = 1; if (!st.b) return { ...K }; if (!st.stopped) { if (V < 1) st.stopped = 1; return { ...K, brake: true }; } return Math.abs(d) > 8 ? { ...K, throttle: d > 0 ? 1 : -1 } : { ...K, brake: true, pitch: -1 }; });
if (process.argv[3] === 'trace') {
  const { yprOf } = await import('../js/sim/vmath.js');
  const tr = (label, V0, pol) => { console.log('\n' + label); let n = 0; run(label, V0, 40, (o) => { const c = pol(o); if (c.pitch && n++ % 12 === 0) { const [, pi, ro] = yprOf(o.b.q); console.log(`  t=${o.t.toFixed(2)} agl=${o.agl.toFixed(2)} V=${o.V.toFixed(1)} Va=${o.b.airspeed.toFixed(1)} vz=${o.vz.toFixed(2)} pitch=${(pi * 57.3).toFixed(1)} roll=${(ro * 57.3).toFixed(1)} contact=${(o.b.contactN / 1000).toFixed(1)}kN scrape=${(o.b.scrape || 0).toFixed(2)}`); } return c; }); };
  tr('A: brake+descend from 60 m at 45', 45, ({ d }) => (d < 60 ? { ...K, brake: true, pitch: -1 } : { ...K }));
  tr('B: descend held at 25 m/s', 25, () => ({ ...K, pitch: -1 }));
}
if (process.argv[3] === 'touch') {
  const { yprOf } = await import('../js/sim/vmath.js'); const { normalAt } = await import('../js/sim/terrain.js');
  let G_ = null, n = 0;
  run('touch', 45, 40, ({ d, b, agl, V, vz, t }) => { if (agl < 0.5 && n++ % 4 === 0) { const [ya, pi, ro] = yprOf(b.q); console.log(`t=${t.toFixed(2)} agl=${agl.toFixed(2)} v=${b.v.map((x) => x.toFixed(2))} pitch=${(pi * 57.3).toFixed(1)} roll=${(ro * 57.3).toFixed(1)} w=${b.w.map((x) => x.toFixed(2))} N=${(b.contactN / 1000).toFixed(1)} skid=${b.skidF.map((f) => (f / 1000).toFixed(1)).join('/')} thrust=${(b.thrust / 1000).toFixed(2)} sd=${b.sd.map((x) => x.toFixed(0)).join('/')} nrm=${normalAt(b.p[0], b.p[2]).map((x) => x.toFixed(2))} tilt=${b.tilt.toFixed(2)}`); } return d < 120 ? { ...K, brake: true, pitch: V < 3 ? -1 : 0 } : { ...K }; });
}
