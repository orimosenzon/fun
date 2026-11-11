// Can every chapter be finished? A simple autopilot flies them with the same inputs a player has.
//   node tests/missions.mjs [chapter] [-v]
import { initWorld } from '../js/sim/world.js';
import { Bird } from '../js/sim/bird.js';
import { Game } from '../js/story.js';
import { pilot } from '../js/autopilot.js';
initWorld();
const verbose = process.argv.includes('-v'), only = process.argv[2] && process.argv[2] !== '-v' ? +process.argv[2] : null;
const ui = { markers() {}, title() {}, objective() {}, flash(t) { if (verbose) console.log('   !', t); }, chime() {}, respawned() {}, chapterDone() {} };
let fails = 0;

const b = new Bird(), g = new Game(b, ui);
for (let ci = 0; ci < g.chapters.length - 1; ci++) {
  if (only != null && only !== ci) continue;
  g.start(ci);
  const dt = 1 / 60; let t = 0, respawns = 0, lastSi = -1, minSt = 1;
  for (; t < 900 && g.state === 'play'; t += dt) {
    const inp = pilot(b, g, t);
    g.update(dt);
    for (const e of b.events) if (e.type === 'crash' || e.type === 'splash') { respawns++; if (verbose) console.log(`   ${e.type} at step ${g.si} t=${t.toFixed(0)} pos=${b.p.map((x) => x.toFixed(0))}`); }
    b.events.length = 0;
    b.step(dt * g.timeScale, inp, t);
    minSt = Math.min(minSt, b.stamina);
    if (verbose && g.si !== lastSi) { lastSi = g.si; console.log(`   step ${g.si} at t=${t.toFixed(0)} y=${b.p[1].toFixed(0)} V=${(b.airspeed * 3.6).toFixed(0)} st=${b.stamina.toFixed(2)}`); }
    if (verbose && Math.round(t * 60) % 600 === 0) console.log(`     t=${t.toFixed(0)} ${b.mode} pos=${b.p.map((x) => x.toFixed(0))} V=${(b.airspeed * 3.6).toFixed(0)} vy=${b.climb.toFixed(1)} st=${b.stamina.toFixed(2)} si=${g.si}`);
    if (respawns > 6) break;
  }
  const ok = g.state === 'done';
  if (!ok) fails++;
  console.log(`${ok ? 'ok  ' : 'FAIL'} ${ci + 1}. ${g.chapters[ci].title}: ${ok ? 'finished' : `stuck at step ${g.si}/${g.chapter.steps.length}`} in ${t.toFixed(0)} s, respawns ${respawns}, lowest stamina ${minSt.toFixed(2)}`);
}
process.exit(fails ? 1 : 0);
