// node tests/flight.mjs : the F-5 of FÖHN over the Mandelbrot land, and the logarithmic altimeter.
import { Sim, FoehnTerrain, G } from '../js/foehn.js';
import { Field, Tone } from '../js/core/field.js';
import { World, K_LO, K_HI, S_MAX, VIEW_M } from '../js/core/world.js';
import { bp, bpToD } from '../js/core/fixed.js';

let fail = 0;
const ok = (name, cond, info = '') => { console.log((cond ? 'ok   ' : 'FAIL ') + name + ' ' + info); if (!cond) fail++; };

const field = new Field(), tone = new Tone(), world = new World(field, tone, { quality: 1, cap: 100000 });
FoehnTerrain.ext = { groundAt: (x, z) => world.groundAt(x, z), normalAt: (x, z) => world.normalAt(x, z), waterAt: () => 0 };
const sim = new Sim(), b = sim.bike;

// fly for T seconds with a control function; returns a small log
function fly(T, ctl) {
  const log = { minY: 1e9, maxY: -1e9, crash: null, nan: false, contact: 0 };
  for (let t = 0; t < T; t += 1 / 60) {
    sim.advance(1 / 60, ctl(t));
    if (b.crash) { log.crash = b.crash; break; }
    world.follow(b.p, b.contactN > 0, 1 / 60);
    world.ensureRef(); world.sample(4); world.plan(); for (const L of world.levels) if (L.job) world.finish(L.job);
    log.minY = Math.min(log.minY, b.p[1]); log.maxY = Math.max(log.maxY, b.p[1]);
    if (b.contactN > 0) log.contact++;
    if (![...b.p, ...b.v, ...b.q, world.S, world.ox, world.oy].every(Number.isFinite)) { log.nan = true; break; }
  }
  return log;
}

// ---- over the open lake (inside the main cardioid): nose down, and the world grows ----
{
  world.place(bp('-0.15', '0.0', 128), S_MAX);
  sim.reset([0, 90, 0], 0, 30);
  const S0 = world.S;
  const a = fly(25, () => ({ turn: 0, pitch: -1, vt: 40 }));
  ok('dive over the lake: no crash', !a.crash && !a.nan, a.crash || '');
  ok('the bike shrinks instead of reaching the water', world.S < S0 * 1e-3 && a.minY > K_LO - 3, `scale fell ${(S0 / world.S).toExponential(2)} times in 25 s, lowest ${a.minY.toFixed(1)} m`);
  const [x, y] = bpToD(world.origin());
  ok('and stays where it was on the plane', Math.abs(x + 0.15) < 0.6 && Math.abs(y) < 0.3, `${x.toFixed(4)}, ${y.toFixed(4)}`);
  const S1 = world.S;
  const c = fly(60, () => ({ turn: 0, pitch: 1, vt: 40, boost: true }));
  ok('climbing brings it back to full size', world.S === S_MAX && c.maxY > K_HI && !c.crash && !c.nan, `from ${(S0 / S1).toExponential(1)} to ${(S0 / world.S).toFixed(1)}, ${c.maxY.toFixed(0)} m`);
}

// ---- a very long dive: to the bottom of the numbers ----
{
  world.place(bp('-0.15', '0.0', 128), S_MAX);
  sim.reset([0, 60, 0], 0, 30);
  let t = 0; const a = fly(160, () => ({ turn: 0, pitch: -1, vt: 30 }));
  ok('dive of 160 s', !a.nan && world.S < 1e-25, `scale ${world.S.toExponential(2)}, ${field.ref.center.bits} bits, ${a.crash || 'still flying'}`);
}

// ---- over land: the ground is where the field says, and the flight computer keeps the bike off it ----
{
  world.place(bp('0.62', '0.62', 128), S_MAX);
  const g = world.groundAt(0, 0);
  ok('land outside the set', g > 20 && g < 400, `${g.toFixed(1)} m above the lake`);
  const n = world.normalAt(0, 0);
  ok('its slope is about the slope asked for', n[1] > 0.7 && n[1] < 0.999, `normal y ${n[1].toFixed(3)} (slope ${world.slope})`);
  // from the middle of the lake straight at the eastern shore and up the land behind it, hands off the stick
  world.place(bp('-0.15', '0.0', 128), S_MAX);
  sim.reset([0, 60, 0], 0, 50);
  let low = 1e9, top = 0;
  const a = fly(40, (t) => { if (t > 1) low = Math.min(low, b.agl); top = Math.max(top, world.groundAt(0, 0)); return { turn: 0, pitch: 0, vt: 55 }; });
  ok('flies up the shore without touching it', !a.crash && !a.nan && low > 1 && top > 60, `${a.crash || ''} lowest ${low.toFixed(1)} m over ground, ground rose to ${top.toFixed(0)} m, bike at ${b.p[1].toFixed(0)} m`);
}

// ---- a deep place: the levels, the lattice and the budget all follow ----
{
  const s = 1.2e-13;
  world.place(bp('-0.743643887037158704752191506114774', '0.131825904205311970493132056385139', 256), s / VIEW_M);
  for (let i = 0; i < 40; i++) { world.sample(8); world.ensureRef(); world.plan(); for (const L of world.levels) if (L.job) world.finish(L.job); }
  const g = world.groundAt(0, 0), g2 = world.groundAt(30, 0), g3 = world.groundAt(0, -30);
  ok('deep place: heights are in metres of the bike', [g, g2, g3].every((v) => v >= -3 && v < 2000), `${g.toFixed(1)}, ${g2.toFixed(1)}, ${g3.toFixed(1)} m; ${world.bud.iter} iterations, ${world.levels.length} levels, finest 2^-${world.l0}`);
  ok('levels are nested fine to coarse', world.levels.every((L, i) => i === 0 || L.l === world.levels[i - 1].l - 1) && world.levels.every((L) => L.valid));
}

console.log(fail ? `${fail} FAILED` : 'ALL PASS');
process.exit(fail ? 1 : 0);
