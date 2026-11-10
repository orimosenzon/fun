import { loadWorld, deg } from './util.mjs';
import { Game } from '../js/sim/game.js';
import { Autopilot } from '../js/sim/autopilot.js';
import { buildScenery } from '../js/sim/scenery.js';
import { len, yprOf } from '../js/sim/vmath.js';
loadWorld(); buildScenery();
const [id, t0, t1, step] = [process.argv[2], +process.argv[3], +process.argv[4], +(process.argv[5] || 0.5)];
const g = new Game(); g.load(id);
const ap = new Autopilot(g);
let next = t0;
console.log('path0', g.build.path.slice(0, 12).filter((_, i) => i % 3 === 0).map((q) => q.map((v) => v.toFixed(0)).join(',')).join(' | '));
while (g.total < t1 && g.state !== 'finished') {
  const c = ap.update();
  g.update(1 / 60, c);
  for (const e of g.events) { if (e.type === 'crash') console.log('CRASH', e.reason); if (e.type === 'respawn') ap.reset(); if (e.type === 'miss' || e.type === 'gate' || e.type === 'land') console.log(e.type, e.index); }
  g.events.length = 0;
  if (g.total >= next) {
    const b = g.bike, [y, p, r] = yprOf(b.q), q = g.build.path[ap.i];
    console.log(`t=${g.total.toFixed(1)} p=${b.p.map((v) => v.toFixed(0))} agl=${b.agl.toFixed(1)} V=${len(b.v).toFixed(0)} vz=${b.v[1].toFixed(1)} ypr=${deg(y).toFixed(0)},${deg(p).toFixed(0)},${deg(r).toFixed(0)} ctl=t${c.turn.toFixed(2)} p${c.pitch.toFixed(2)} vt${c.vt.toFixed(0)} s${c.strafe.toFixed(2)}${c.boost ? ' B' : ''}${c.brake ? ' K' : ''} i=${ap.i} path=${q.map((v) => v.toFixed(0))} cp=${g.cp} T=${b.thrust.toFixed(0)} tilt=${deg(b.tilt).toFixed(0)}`);
    next += step;
  }
}
