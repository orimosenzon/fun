/* The flying car over flat ground, flown in Node: web/car_core.js, which is
 * FÖHN's F-5 (web/f5_core.js, tests/test_f5.mjs) with a car's manners.
 *
 * What is checked is what is the car's own: that it still does everything
 * the flight mode asks of a bike, that it banks less and more slowly than
 * the bike does, and that flying the car leaves the bike's dials where they
 * were, because both are flown by the same module.
 *
 *     node tests/test_car.mjs
 */
import { CarFlight, CAR_FEEL, yawFor, V_MAX } from '../web/car_core.js';
import { F5 } from '../web/f5_core.js';
import { FEEL, FEEL_DEFAULT } from '../../jetbikes/foehn/js/sim/fcs.js';
import { qrot } from '../../jetbikes/foehn/js/sim/vmath.js';

let fails = 0;
const check = (name, ok, detail = '') => {
  console.log((ok ? '  ok   ' : '  FAIL ') + name + (ok ? '' : '   ' + detail));
  if (!ok) fails++;
};
const bank = (f) => Math.asin(Math.max(-1, Math.min(1, qrot(f.b.q, [0, 0, -1])[1]))) * 180 / Math.PI;
const fly = (script, secs, f = new CarFlight(), bearing = 0) => {
  if (!f._placed) { f.place(0, 0, yawFor(bearing)); f._placed = true; }
  let crashed = false, maxBank = 0;
  for (let i = 0; i < secs * 60; i++) {
    script(f, i / 60);
    f.step(1 / 60);
    if (f.consumeCrashEvent()) crashed = true;
    maxBank = Math.max(maxBank, Math.abs(bank(f)));
  }
  f.crashedOnce = crashed;
  f.maxBank = maxBank;
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
check('W alone takes off and flies north', f.pos.z < -100 && Math.abs(f.pos.x) < 30 && f.pos.y > 1,
  `x=${f.pos.x} y=${f.pos.y} z=${f.pos.z}`);
check('a quarter of the lever is a quarter of the top speed', Math.abs(kmh(f) - 0.25 * V_MAX * 3.6) < 12, `v=${kmh(f)}`);

const turn = (Flight) => fly((b, t) => { b.throttle = 0.3; b.vert = t < 2 ? 1 : 0; b.steer = t > 5 ? 1 : 0; }, 9, new Flight());
const car = turn(CarFlight), bike = turn(F5);
check('stick right turns the car right', car.vel.x > 5, `vx=${car.vel.x}`);
check(`the car banks no further than its limit (${(CAR_FEEL.bank * 180 / Math.PI).toFixed(0)} degrees, and a little)`,
  car.maxBank > 20 && car.maxBank < CAR_FEEL.bank * 180 / Math.PI + 6, `max=${car.maxBank}`);
check('and less than the bike in the same turn', car.maxBank < bike.maxBank - 8, `car=${car.maxBank} bike=${bike.maxBank}`);
check('it holds its height through the turn', car.pos.y > 3 && !car.crashedOnce, `y=${car.pos.y}`);

check('flying the car leaves the bike\'s dials alone', Object.keys(FEEL_DEFAULT).every((k) => FEEL[k] === FEEL_DEFAULT[k]),
  JSON.stringify(Object.keys(FEEL_DEFAULT).filter((k) => FEEL[k] !== FEEL_DEFAULT[k])));

f = fly((b, t) => { b.throttle = t < 5 ? 0.5 : 0; b.brake = t > 5; }, 14);
check('lever closed and S: stops in the air, does not fall', kmh(f) < 12 && f.pos.y > 2 && !f.crashedOnce, `v=${kmh(f)} y=${f.pos.y}`);

f = fly((b, t) => { b.vert = t < 2 ? 1 : 0; if (Math.abs(t - 4) < 0.009) b.firePulse('R'); }, 7);
check('X steps it to the right', f.pos.x > 1, `x=${f.pos.x}`);

f = fly((b, t) => { b.throttle = 1; b.pitch = t > 4 ? -1 : 0; }, 9);
check('stick held down at full speed: the computer keeps it off the ground', !f.crashedOnce && f.pos.y > 0.5, `y=${f.pos.y}`);
// The computer will not fly it into the ground, so the test throws it there.
f = fly((b, t) => { b.vert = t < 2 ? 1 : 0; if (Math.abs(t - 3) < 0.009) { b.b.v[1] = -45; } }, 5);
check('thrown at the ground: a crash, reported once', f.crashedOnce, `y=${f.pos.y}`);
fly(() => { f.throttle = 0; f.pitch = 0; }, 4, f);
check('and after it, back on the ground and whole', !f.crashed && f.pos.y < 0.3 && f.alive, `y=${f.pos.y} crashed=${f.crashed}`);

console.log(fails ? `\n${fails} FAILED` : '\nall passed');
process.exit(fails ? 1 : 0);
