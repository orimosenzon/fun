// The fortress: the solids are where they should be, and the course runs through free air.
//   node tests/castle.mjs
import { loadWorld } from './util.mjs';
import { groundAt } from '../js/sim/terrain.js';
import { buildScenery, hitScenery } from '../js/sim/scenery.js';
import { C, hitCastle, castleY } from '../js/sim/castle.js';
import { SITE } from '../js/sim/castleSite.js';
import { buildEvent, eventById } from '../js/sim/courses.js';
loadWorld(); buildScenery();
let fail = 0;
const ok = (name, cond, info = '') => { console.log(`${cond ? 'ok  ' : 'FAIL'} ${name} ${info}`); if (!cond) fail++; };
const y0 = castleY(), W = (x, y, z) => [SITE.x + x, y0 + y, SITE.z + z];
console.log(`${C.solids.length} solids, ${C.deco.length} ornaments, courtyard at ${y0.toFixed(1)} m`);
// the plateau is flat, the moat is there
let flat = 0; for (let x = -160; x <= 160; x += 20) for (let z = -110; z <= 110; z += 20) flat = Math.max(flat, Math.abs(groundAt(SITE.x + x, SITE.z + z) - y0));
ok('the courtyard floor is level', flat < 0.01, `max deviation ${flat.toFixed(3)} m`);
ok('the moat is 22 m deep under the bridge', Math.abs(groundAt(SITE.x - 192, SITE.z) - (y0 - 22)) < 0.1, `${(groundAt(SITE.x - 192, SITE.z) - y0).toFixed(1)}`);
// solid where it should be
const solid = [['inside the keep', 61, 40, -20], ['curtain wall', 0, 10, -100], ['gatehouse above the tunnel', -150, 25, 0], ['pier of the hall', 25, 6, 44], ['vault of the hall above the nave', 60, 22.5, 56],
  ['corner tower', 150, 20, 105], ['its roof', 150, 60, 105], ['keystone of the bridge', -192, -1.5, 0], ['pillar of the east arcade', 124, 4, -49.5], ['roof of the hall', 60, 30, 56], ['spire of the keep', 61, 110, -4]];
for (const [name, x, y, z] of solid) ok(`solid: ${name}`, hitCastle(W(x, y, z), 0), '');
const air = [['gate tunnel', -150, 7, 0], ['nave of the hall', 60, 10, 56], ['window of the hall', 19.5, 8, 44], ['outer ward', -90, 10, 20], ['east gallery', 133, 8, -20], ['arcade opening', 124, 4, -44.25],
  ['passage of the keep', 61, 7, -4], ['alley under a flying arch', 61.5, 8, 28], ['postern', -5, 6, 100], ['under the bridge', -192, -12, 0], ['above the roof of the hall, beside the ridge', 60, 40, 44], ['just above the battlements', 0, 28, -100]];
for (const [name, x, y, z] of air) ok(`free:  ${name}`, !hitCastle(W(x, y, z), 0.35), '');
// the course
const b = buildEvent(eventById('castle'));
let worst = 1e9, at = null, low = 1e9;
for (const q of b.path) {
  let c = 6;
  for (const r of [5, 4, 3, 2, 1]) { let hit = false; for (let a = 0; a < 16 && !hit; a++) for (const e of [-1, 0, 1]) { const u = a * 0.3927; if (hitCastle([q[0] + Math.cos(u) * r * Math.cos(e), q[1] + Math.sin(e) * r, q[2] + Math.sin(u) * r * Math.cos(e)], 0)) { hit = true; break; } } if (hit) c = r - 1; }
  if (c < worst) { worst = c; at = q; }
  low = Math.min(low, q[1] - groundAt(q[0], q[2]));
}
ok('the line keeps at least 4 m from every wall', worst >= 4, `tightest ${worst} m at castle (${(at[0] - SITE.x).toFixed(0)}, ${(at[1] - y0).toFixed(1)}, ${(at[2] - SITE.z).toFixed(0)})`);
ok('and at least 5 m above the ground', low >= 5, `lowest ${low.toFixed(1)} m`);
console.log(`course: ${b.real.length} rings, ${(b.length).toFixed(0)} m`);
b.real.forEach((c, i) => {
  let clear = true;
  for (let a = 0; a < 24; a++) for (const f of [0.5, 1]) { const u = a * 0.2618, side = [-c.n[2], 0, c.n[0]], sl = Math.hypot(side[0], side[2]) || 1; const rr = (c.r + 0.3) * f; if (hitScenery([c.p[0] + side[0] / sl * Math.cos(u) * rr, c.p[1] + Math.sin(u) * rr, c.p[2] + side[2] / sl * Math.cos(u) * rr], 0)) clear = false; if (c.p[1] - c.r - groundAt(c.p[0], c.p[2]) < 1.5) clear = false; }
  ok(`ring ${i + 1} stands free (r=${c.r})`, clear, `at castle (${(c.p[0] - SITE.x).toFixed(0)}, ${(c.p[1] - y0).toFixed(1)}, ${(c.p[2] - SITE.z).toFixed(0)}) n=(${c.n.map((v) => v.toFixed(2))})`);
});
console.log(fail ? `${fail} FAILED` : 'ALL PASS'); process.exit(fail ? 1 : 0);
