// Flies every event headless with the demo rider. node tests/events.mjs [id] [-v]
import { loadWorld } from './util.mjs';
import { Game } from '../js/sim/game.js';
import { Autopilot } from '../js/sim/autopilot.js';
import { buildScenery, S } from '../js/sim/scenery.js';
import { EVENTS } from '../js/sim/courses.js';
import { len } from '../js/sim/vmath.js';
loadWorld();
const t0 = Date.now();
buildScenery();
console.log(`scenery: ${S.count} trees, ${S.props.length} buildings, ${Date.now() - t0} ms`);
const only = process.argv[2] && !process.argv[2].startsWith('-') ? process.argv[2] : null, verbose = process.argv.includes('-v');
let fail = 0;
for (const ev of EVENTS) {
  if (only && ev.id !== only) continue;
  const g = new Game(); g.load(ev.id);
  const ap = new Autopilot(g);
  const log = [];
  let vmax = 0, tReal = Date.now();
  while (g.state !== 'finished' && g.total < 700) {
    g.update(1 / 60, ap.update());
    vmax = Math.max(vmax, len(g.bike.v));
    for (const e of g.events) {
      if (e.type === 'crash') log.push(`  crash(${e.reason}) t=${g.total.toFixed(1)} cp=${g.cp} at ${e.p.map((v) => v.toFixed(0))} agl=${g.bike.agl.toFixed(1)} V=${len(e.v).toFixed(0)}`);
      if (e.type === 'miss') log.push(`  miss cp ${e.index} t=${g.total.toFixed(1)} off=${e.off.toFixed(0)}m dy=${e.dy.toFixed(0)} V=${len(g.bike.v).toFixed(0)}`);
      if (verbose && (e.type === 'gate' || e.type === 'land')) log.push(`  ${e.type} ${e.index} t=${e.time.toFixed(1)} V=${len(g.bike.v).toFixed(0)}`);
      if (e.type === 'respawn') ap.reset();
    }
    g.events.length = 0;
  }
  const ok = g.state === 'finished';
  if (!ok || g.crashes > 2) fail++;
  console.log(`${ev.id.padEnd(8)} ${ok ? 'FINISH' : 'DNF   '} time=${g.total.toFixed(1)}s len=${(g.build.length / 1000).toFixed(1)}km cps=${g.cp}/${g.cps.length} crashes=${g.crashes} misses=${g.misses} vmax=${(vmax * 3.6).toFixed(0)}km/h fuel=${g.bike.fuel.toFixed(0)}kg medals=${ev.medals} (${((Date.now() - tReal) / 1000).toFixed(1)}s cpu)`);
  if (log.length) console.log(log.slice(0, verbose ? 60 : 8).join('\n'));
}
process.exit(fail ? 1 : 0);
