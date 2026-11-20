// The second ring: the shaft, the halls and the tunnel are where they should be, the line of the
// event runs through free air, and a rider who simply dives down the shaft arrives in one piece.
//   node tests/tunnel.mjs
import { loadWorld } from './util.mjs';
import { T, groundAt, waterAt } from '../js/sim/terrain.js';
import { buildScenery, hitScenery } from '../js/sim/scenery.js';
import { RING, fromRing, toRing, doorAt, rimY, underAt, hitTunnel, inCavity } from '../js/sim/tunnel.js';
import { HOLE, LAKE2, ISLE, HAMLET } from '../js/sim/ringSite.js';
import { buildEvent, eventById } from '../js/sim/courses.js';
import { Game } from '../js/sim/game.js';
import { len } from '../js/sim/vmath.js';
loadWorld(); buildScenery();
let fail = 0;
const ok = (name, cond, info = '') => { console.log(`${cond ? 'ok  ' : 'FAIL'} ${name} ${info}`); if (!cond) fail++; };
const rim = rimY(), F = RING.floor;
// ---- the place ----
let flat = 0; for (let a = 0; a < 6.28; a += 0.4) for (const r of [12, 20, 28]) flat = Math.max(flat, Math.abs(groundAt(HOLE.x + Math.cos(a) * r, HOLE.z + Math.sin(a) * r) - rim));
ok('the top of the island is level round the hole', flat < 0.07, `max deviation ${flat.toFixed(3)} m, rim at ${rim.toFixed(1)} m`);
ok('the island stands in water', waterAt(ISLE.x + 150, ISLE.z) > groundAt(ISLE.x + 150, ISLE.z) && waterAt(ISLE.x - 150, ISLE.z) > groundAt(ISLE.x - 150, ISLE.z), `level ${LAKE2.level}`);
ok('the hole is open down to the floor of the ring', groundAt(HOLE.x, HOLE.z) === F && groundAt(HOLE.x + RING.shaftR + 0.5, HOLE.z) > rim - 0.1, '');
ok('and there is no water in it', waterAt(HOLE.x, HOLE.z) === -Infinity, '');
ok('the hamlet stands on dry, gentle ground', waterAt(HAMLET.x, HAMLET.z) < groundAt(HAMLET.x, HAMLET.z), `${groundAt(HAMLET.x, HAMLET.z).toFixed(0)} m`);
let lowest = 1e9; for (let s = 60; s < RING.len - 60; s += 25) { const p = fromRing(s, 0, 0); T.under = null; lowest = Math.min(lowest, groundAt(p[0], p[2]) - (F + RING.ys + RING.hw)); }
ok('the whole ring lies under the ground', lowest > 20, `least cover ${lowest.toFixed(0)} m`);
const far = fromRing(RING.len / 2, 0, 0); ok('and inside the map', [0, RING.len / 4, RING.len / 2, -RING.len / 4].every((s) => { const p = fromRing(s, 0, 0); return Math.abs(p[0]) < T.half - 330 && Math.abs(p[2]) < T.half - 330; }), `far side at ${far[0].toFixed(0)}, ${far[2].toFixed(0)}`);
const rt = toRing(...[fromRing(1234, 2.5, 0)].map((p) => [p[0], p[2]])[0]); ok('ring coordinates go there and back', Math.abs(rt[0] - 1234) < 1e-6 && Math.abs(rt[1] - 2.5) < 1e-6, '');
// ---- solid and free ----
const R = (s, u, v) => fromRing(s, u, v);
const free = [['the shaft, half way down', [HOLE.x, (rim + F) / 2, HOLE.z]], ['the hall under it', R(10, -12, 15)], ['the tunnel', R(300, 0, 6)], ['under the crown', R(300, 0, 11.4)], ['beside the beam line', R(300, 4.6, 1)],
  ['the far side of the ring', R(RING.len / 2, 0, 5)], ['the hall of the steel door', R(RING.vault.s, -18, 22)], ['the opening of the first door', R(doorAt(1).s, doorAt(1).u, doorAt(1).v)]];
for (const [name, p] of free) ok(`free:  ${name}`, !hitTunnel(p, 0.35) && underAt(p), '');
const solid = [['rock beside the shaft', [HOLE.x + 14, rim - 40, HOLE.z]], ['the wall of the shaft', [HOLE.x + RING.shaftR + 0.2, rim - 40, HOLE.z]], ['the curb round its mouth', [HOLE.x + RING.shaftR + 0.7, rim + 0.3, HOLE.z]],
  ['the wall of the tunnel', R(300, 8.3, 2)], ['above its crown', R(300, 0, 12.6)], ['the beam line', R(300, 6.5, 1.4)], ['the first door, beside its opening', R(doorAt(1).s, -doorAt(1).u * 3, 2)], ['rock between hall and tunnel', R(20, 12, 5).map((v, i) => (i === 1 ? v + 20 : v))]];
for (const [name, p] of solid) ok(`solid: ${name}`, hitTunnel(p, 0), '');
ok('nothing of it is in the way above the ground', !hitTunnel([HOLE.x + 15, rim + 0.5, HOLE.z]) && !hitTunnel([HOLE.x, rim + 3, HOLE.z]) && !underAt([HOLE.x, rim + 3, HOLE.z]) && !underAt([0, 900, 0]), '');
ok('63 places for a door, 61 doors', Array.from({ length: RING.nb }, (_, k) => doorAt(k)).filter(Boolean).length === 61, `one every ${RING.sp.toFixed(0)} m`);
// ---- the line of the event ----
const b = buildEvent(eventById('ring'));
let worst = 9, at = null;
for (const q of b.path) {
  if (!q.under || q[1] > rim - 2) continue;
  let c = 5;
  for (const r of [4, 3, 2, 1]) for (let a = 0; a < 8; a++) if (hitTunnel([q[0] + Math.cos(a * 0.785) * r, q[1], q[2] + Math.sin(a * 0.785) * r], 0) || hitTunnel([q[0], q[1] + Math.sin(a * 0.785) * r, q[2]], 0)) c = Math.min(c, r - 1);
  if (c < worst) { worst = c; at = q; }
}
ok('under the earth the line keeps 3 m from every wall', worst >= 3, `tightest ${worst} m at s=${toRing(at[0], at[2])[0].toFixed(0)}`);
ok('the rings of the tunnel stand in the openings of the doors', b.real.filter((c) => c.type === 'ring' && c.under).every((c) => { for (let a = 0; a < 6.28; a += 0.3) { const sx = -c.n[2], sz = c.n[0], l = Math.hypot(sx, sz); if (hitTunnel([c.p[0] + sx / l * Math.cos(a) * c.r, c.p[1] + Math.sin(a) * c.r, c.p[2] + sz / l * Math.cos(a) * c.r], 0.2)) return false; } return true; }), `${b.real.filter((c) => c.under && c.type === 'ring').length} of them`);
T.under = null;
ok('no tree on the top of the island, none in the hole', !hitScenery([HOLE.x + 14, rim + 2, HOLE.z]) && !hitScenery([HOLE.x, rim + 2, HOLE.z]), '');
// ---- a rider: hover over the hole, push the stick and keep it there ----
for (const [name, pitch] of [['gently', -0.5], ['the stick right forward', -1]]) {
  const g = new Game(); g.windLevel = 0; g.load('ring'); g.state = 'fly'; g.cp = 10; g.events.length = 0;
  g.sim.reset([HOLE.x, rim + 14, HOLE.z], 0, 0); g.prevP = g.bike.p.slice();
  let t = 0, vmax = 0, hit = null, landed = false;
  while (t < 60 && !hit && !landed) {
    g.update(1 / 60, { turn: 0, pitch, throttle: 0 }); t += 1 / 60; vmax = Math.max(vmax, -g.bike.v[1]);
    for (const e of g.events) { if (e.type === 'crash') hit = e.reason; if (e.type === 'land') landed = true; }
    g.events.length = 0;
  }
  ok(`down the shaft, ${name}`, landed && !hit, hit ? `crash: ${hit} at ${(g.bike.p[1] - F).toFixed(1)} m above the floor` : `${t.toFixed(1)} s, fastest ${vmax.toFixed(0)} m/s, lands at ${len(g.bike.v).toFixed(1)} m/s`);
  ok('  and the bike is under the earth there', T.under === F, '');
}
T.under = null;
console.log(fail ? `${fail} FAILED` : 'ALL PASS'); process.exit(fail ? 1 : 0);
