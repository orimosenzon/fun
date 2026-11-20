/* The F-5 over flat ground, flown in Node: the seam between FÖHN's bike and
 * the flight mode's controls (web/f5_core.js), without a browser.
 *
 * The bike itself is tested where it lives (jetbikes/foehn/tests). What is
 * checked here is what this app asks of it: that it sits still when parked,
 * leaves the ground on E and on W, flies the way the compass says, turns the
 * way the stick says, stops on the brake, steps sideways on Z and X, crashes
 * and comes back, and answers to the names explore.js reads.
 *
 *     node tests/test_f5.mjs
 */
import { F5, yawFor, V_MAX } from '../web/f5_core.js';

let fails = 0;
const check = (name, ok, detail = '') => {
  console.log((ok ? '  ok   ' : '  FAIL ') + name + (ok ? '' : '   ' + detail));
  if (!ok) fails++;
};
const fly = (script, secs, f = new F5(), bearing = 0) => {
  if (!f._placed) { f.place(0, 0, yawFor(bearing)); f._placed = true; }
  let crashed = false;
  for (let i = 0; i < secs * 60; i++) {
    script(f, i / 60);
    f.step(1 / 60);
    if (f.consumeCrashEvent()) crashed = true;
  }
  f.crashedOnce = crashed;
  return f;
};
const kmh = (f) => Math.hypot(f.vel.x, f.vel.z) * 3.6;

let f = fly(() => {}, 5);
check('parked: stays on the ground and does not creep', f.pos.y < 0.05 && Math.hypot(f.pos.x, f.pos.z) < 0.2 && f.grounded,
  `y=${f.pos.y} x=${f.pos.x} z=${f.pos.z}`);

f = fly((b, t) => { b.vert = t < 3 ? 1 : 0; }, 7);
check('E lifts it, and it holds the height when E is let go', f.pos.y > 8 && Math.abs(f.vel.y) < 1 && kmh(f) < 12,
  `y=${f.pos.y} vy=${f.vel.y} v=${kmh(f)}`);

f = fly((b) => { b.throttle = 0.25; }, 8);
check('W alone takes off and flies north (bearing 0 is -z)', f.pos.z < -100 && Math.abs(f.pos.x) < 30 && f.pos.y > 1,
  `x=${f.pos.x} y=${f.pos.y} z=${f.pos.z}`);
check('a quarter of the lever is a quarter of the top speed', Math.abs(kmh(f) - 0.25 * V_MAX * 3.6) < 12, `v=${kmh(f)}`);

f = fly((b) => { b.throttle = 0.25; }, 8, new F5(), 90);
check('bearing 90 flies east (+x)', f.pos.x > 100 && Math.abs(f.pos.z) < 30, `x=${f.pos.x} z=${f.pos.z}`);

f = fly((b, t) => { b.throttle = 0.25; b.steer = t > 4 ? 1 : 0; }, 5);
check('stick right turns right (north to east)', f.vel.x > 5, `vx=${f.vel.x} vz=${f.vel.z}`);
f = fly((b, t) => { b.throttle = 0.25; b.steer = t > 4 ? -1 : 0; }, 5);
check('stick left turns left', f.vel.x < -5, `vx=${f.vel.x}`);

f = fly((b, t) => { b.throttle = t < 5 ? 0.5 : 0; b.brake = t > 5; }, 14);
check('lever closed and S: stops in the air, does not fall', kmh(f) < 12 && f.pos.y > 2 && !f.crashedOnce, `v=${kmh(f)} y=${f.pos.y}`);

f = fly((b, t) => { b.vert = t < 2 ? 1 : 0; if (Math.abs(t - 4) < 0.009) b.firePulse('R'); }, 7);
check('X kicks it to the right (east, facing north)', f.pos.x > 1.5, `x=${f.pos.x}`);
f = fly((b, t) => { b.vert = t < 2 ? 1 : 0; if (Math.abs(t - 4) < 0.009) b.firePulse('L'); }, 7);
check('Z kicks it to the left', f.pos.x < -1.5, `x=${f.pos.x}`);

f = fly((b, t) => { b.throttle = 1; b.pitch = t > 4 ? -1 : 0; }, 9);
check('stick held down at full speed: the computer keeps it off the ground', !f.crashedOnce && f.pos.y > 0.5, `y=${f.pos.y}`);
// The computer will not fly it into the ground, so the test throws it there.
f = fly((b, t) => { b.vert = t < 2 ? 1 : 0; if (Math.abs(t - 3) < 0.009) { b.b.v[1] = -45; } }, 5);
check('thrown at the ground: a crash, reported once', f.crashedOnce, `y=${f.pos.y}`);
fly(() => { f.throttle = 0; f.pitch = 0; }, 4, f);
check('and after it, back on the ground and whole', !f.crashed && f.pos.y < 0.3 && f.alive, `y=${f.pos.y} crashed=${f.crashed}`);

f = fly((b) => { b.throttle = 0.5; b.input.boostRear = true; }, 6);
const names = ['spoolRear', 'spoolLift', 'collective', 'ab', 'jetRear', 'jetLift'];
check('the instruments read finite numbers in range',
  names.every((k) => Number.isFinite(f[k]) && f[k] >= 0) && f.spoolRear <= 1 && f.spoolLift <= 1.25 && f.ab <= 1 && f.assist === true,
  names.map((k) => `${k}=${f[k]}`).join(' '));
check('the wind is a breeze, not a gale', f.wind.speed > 0.3 && f.wind.speed < 8, `wind=${f.wind.speed}`);

console.log(fails ? `\n${fails} FAILED` : '\nall passed');
process.exit(fails ? 1 : 0);
