// The rings count only in their order and in their direction. node tests/rings.mjs
import { loadWorld } from './util.mjs';
import { Game } from '../js/sim/game.js';
import { buildScenery } from '../js/sim/scenery.js';
loadWorld(); buildScenery();
let fail = 0;
const ok = (name, cond, info = '') => { console.log(`${cond ? 'ok  ' : 'FAIL'} ${name} ${info}`); if (!cond) fail++; };
const g = new Game(); g.load('lake'); g.state = 'fly';
const ring = (i) => g.cps[i];
// carry the bike across the plane of ring i, off its centre by `off` metres (sideways), forward or backward
const cross = (i, off, fwd) => {
  const c = ring(i), n = c.n, side = [-n[2], 0, n[0]], sl = Math.hypot(side[0], side[2]) || 1, s = fwd ? 1 : -1;
  const at = (d) => [c.p[0] + n[0] * d + side[0] / sl * off, c.p[1] + n[1] * d, c.p[2] + n[2] * d + side[2] / sl * off];
  g.events.length = 0; g.prevP = at(-2 * s); g.bike.p = at(2 * s); g.bike.crash = null; g.onTick();
  return g.events.map((e) => e.type).filter((t) => t !== 'warn').join(',');
};
ok('ring 0 flown through: next is ring 1', cross(0, 0, true) === 'gate' && g.cp === 1, `cp=${g.cp}`);
let ev = cross(1, ring(1).r + 15, true);
ok('ring 1 missed: it stays the next one, no time penalty', ev === 'miss' && g.cp === 1 && g.penalty === 0, `events=${ev} cp=${g.cp}`);
ev = cross(1, 0, false);
ok('ring 1 flown through backwards: does not count', ev === 'wrongway' && g.cp === 1, `events=${ev} cp=${g.cp}`);
ev = cross(0, 0, true);
ok('ring 0 again: nothing happens, the order does not go back', ev === '' && g.cp === 1, `events=${ev} cp=${g.cp}`);
ev = cross(2, 0, true);
ok('ring 2 before ring 1: does not count', ev === '' && g.cp === 1, `events=${ev} cp=${g.cp}`);
ev = cross(1, 0, true);
ok('ring 1 the right way: next is ring 2', ev === 'gate' && g.cp === 2, `events=${ev} cp=${g.cp}`);
ev = cross(2, ring(2).r, false);
ok('the rim is solid from behind too', g.state === 'crashed', `events=${ev} state=${g.state}`);
console.log(fail ? `${fail} FAILED` : 'ALL PASS'); process.exit(fail ? 1 : 0);
