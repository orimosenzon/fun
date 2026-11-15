// Experiment: straight-down landings on ground of every slope. node tests/_touch.mjs [hold]
import { loadWorld } from './util.mjs';
import { Game } from '../js/sim/game.js';
import { buildScenery } from '../js/sim/scenery.js';
import { len, yprOf } from '../js/sim/vmath.js';
import { groundAt, normalAt, waterAt } from '../js/sim/terrain.js';
const meta = loadWorld(); buildScenery();
let s = 7; const rnd = () => ((s = (s * 1664525 + 1013904223) >>> 0) / 4294967296);
const bins = {}; const fails = [];
const g = new Game(); g.load('free'); g.fcsObstacle;
while (g.state !== 'fly') { g.update(1 / 60, {}); g.events.length = 0; }
const pads = Object.values(meta.pads);
for (let k = 0; k < 400; k++) {
  const pd = pads[k % pads.length], a = rnd() * 7, r = 20 + rnd() * 500, x = pd[0] + Math.cos(a) * r, z = pd[2] + Math.sin(a) * r, n = normalAt(x, z), slope = Math.acos(n[1]) * 57.3;
  if (slope > 16 || waterAt(x, z) > groundAt(x, z)) continue;
  if (g.sim.fcs.obstacle && g.sim.fcs.obstacle(x, z) > groundAt(x, z) + 0.5) continue;      // a tree or a roof
  g.respawnPose = { p: [x, groundAt(x, z) + 5, z], yaw: rnd() * 6.28, speed: 0.01, pitch: 0 }; g.respawn(); g.vt = 0;
  let guard = 0; while (g.state !== 'fly' && guard++ < 600) { g.update(1 / 60, {}); g.events.length = 0; }
  const b = g.bike; let out = 'air', maxRoll = 0, vzT = 0, touched = false;
  for (let i = 0; i < 60 * 14; i++) {
    const rel = process.argv[2] === 'release' && touched;
    g.update(1 / 60, { throttle: 0, turn: 0, strafe: 0, pitch: rel ? 0 : -1 });
    if (!touched && b.contactN > 0) { touched = true; vzT = b.v[1]; }
    const [, pi, ro] = yprOf(b.q); maxRoll = Math.max(maxRoll, Math.abs(ro) * 57.3, Math.abs(pi) * 57.3);
    const cr = g.events.find((e) => e.type === 'crash'); g.events.length = 0;
    if (cr) { out = 'CRASH ' + cr.reason; break; }
    if (touched && g.sim.fcs.idle && len(b.v) < 0.05 && len(b.w) < 0.05) { out = 'ok'; break; }
  }
  if (out === 'air' && touched) out = len(b.v) > 0.3 ? 'sliding' : 'restless';
  const bin = Math.floor(slope / 2) * 2; (bins[bin] = bins[bin] || { n: 0, ok: 0, other: {} }).n++;
  if (out === 'ok') bins[bin].ok++; else { bins[bin].other[out] = (bins[bin].other[out] || 0) + 1; if (fails.length < 6) fails.push(`${out} slope=${slope.toFixed(1)} vzTouch=${vzT.toFixed(2)} maxTilt=${maxRoll.toFixed(0)} at ${x.toFixed(0)},${z.toFixed(0)}`); }
}
for (const k of Object.keys(bins).sort((a, b) => a - b)) console.log(`slope ${String(k).padStart(2)}-${String(+k + 2).padEnd(2)} deg: ${bins[k].ok}/${bins[k].n} ok  ${JSON.stringify(bins[k].other)}`);
console.log(fails.join('\n'));
