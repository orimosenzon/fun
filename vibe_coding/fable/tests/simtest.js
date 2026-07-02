// Headless physics verification for fable.
import { BikePhysics, TUNE } from './physics.js';
import { SPAWN, terrainHeight } from './world.js';
import * as THREE from 'three';

const STEP = 1 / 120;
const clampN = (v, a, b) => Math.min(b, Math.max(a, v));
let failures = 0;
function check(name, cond, detail) {
  console.log((cond ? 'PASS' : 'FAIL') + '  ' + name + (detail ? '   [' + detail + ']' : ''));
  if (!cond) failures++;
}
function run(phys, seconds, perStep) {
  const n = Math.round(seconds / STEP);
  for (let i = 0; i < n; i++) { if (perStep) perStep(i * STEP); phys.step(STEP); }
}

console.log('SPAWN:', SPAWN.toArray().map(v => v.toFixed(1)).join(', '));

// 1. Rest on pad: no input, should settle and stay put.
{
  const p = new BikePhysics();
  p.collective = 0;
  run(p, 5);
  const agl = p.pos.y - terrainHeight(p.pos.x, p.pos.z);
  check('rests on pad', !p.crashed && agl > 0.1 && agl < 1 && p.vel.length() < 0.5,
    'agl=' + agl.toFixed(2) + ' v=' + p.vel.length().toFixed(2));
}

// 2. Hover: collective at hover point should climb then roughly hold.
{
  const p = new BikePhysics();
  run(p, 8, () => { p.collective = 0.75; });
  const upDot = new THREE.Vector3(0, 1, 0).applyQuaternion(p.quat).y;
  check('climbs with high collective', !p.crashed && p.pos.y > SPAWN.y + 5, 'y=' + p.pos.y.toFixed(1));
  check('stays roughly upright in climb', upDot > 0.9, 'upDot=' + upDot.toFixed(3));
}

// 3. Level flight at full throttle (player-style altitude hold) reaches high speed.
{
  const p = new BikePhysics();
  // climb above the highest peaks, then cruise at the hover-balance collective
  run(p, 16, () => { p.collective = p.pos.y < SPAWN.y + 300 ? 0.9 : 0.53; });
  run(p, 25, () => { p.throttle = 1; p.collective = 0.53; });
  const v = p.vel.length();
  check('reaches high speed in level flight', !p.crashed && v > 60,
    'v=' + v.toFixed(1) + ' m/s, y=' + p.pos.y.toFixed(1) + ' crashed=' + p.crashed);
  check('top speed sane (<130 m/s)', v < 130, 'v=' + v.toFixed(1));
  check('moves along +Z', p.pos.z - SPAWN.z > 500, 'dz=' + (p.pos.z - SPAWN.z).toFixed(0));
}

// 4. Steering right should bank right and curve the path right (x increases).
{
  const p = new BikePhysics();
  run(p, 6, () => { p.collective = 0.85; p.throttle = 0.8; });
  const zBefore = p.pos.z;
  run(p, 6, () => { p.steer = 1; p.collective = 0.95; });
  const right = new THREE.Vector3(1, 0, 0).applyQuaternion(p.quat);
  check('banks right when steering right', right.y < -0.3, 'right.y=' + right.y.toFixed(2));
  check('turns right (x grows)', p.pos.x > 30, 'x=' + p.pos.x.toFixed(0) + ' z=' + p.pos.z.toFixed(0));
  check('no crash during turn', !p.crashed);
}

// 5. Z boost = raw roll torque: a short tap must spin the bike rolling right
// (negative body-frame roll rate). Long holds do full barrel rolls by design.
{
  const p = new BikePhysics();
  run(p, 6, () => { p.collective = 0.75; });
  run(p, 0.25, () => { p.input.boostL = true; });
  check('Z boost rolls right (roll rate)', p.angVel.z < -1, 'wz=' + p.angVel.z.toFixed(2));
  const p2 = new BikePhysics();
  run(p2, 6, () => { p2.collective = 0.75; });
  run(p2, 0.25, () => { p2.input.boostR = true; });
  check('X boost rolls left (roll rate)', p2.angVel.z > 1, 'wz=' + p2.angVel.z.toFixed(2));
}

// 6. S boost adds lift along body up: from hover, vertical speed should jump.
{
  const p = new BikePhysics();
  run(p, 6, () => { p.collective = 0.72; });
  const vy0 = p.vel.y;
  run(p, 1.5, () => { p.input.boostC = true; });
  check('S boost adds lift', p.vel.y > vy0 + 3, 'vy ' + vy0.toFixed(1) + ' -> ' + p.vel.y.toFixed(1));
}

// 7. Free fall from height must crash.
{
  const p = new BikePhysics();
  p.pos.y = SPAWN.y + 60;
  p.collective = 0;
  let crashMsg = null;
  const n = Math.round(6 / STEP);
  for (let i = 0; i < n; i++) { p.step(STEP); const e = p.consumeCrashEvent(); if (e) crashMsg = e; }
  check('free fall crashes', crashMsg !== null, 'msg=' + crashMsg);
}

// 8. Crash auto-respawns at pad (horizontal position; may drift up on ground effect).
{
  const p = new BikePhysics();
  p.pos.y = SPAWN.y + 60;
  p.collective = 0;
  run(p, 12);
  const dHoriz = Math.hypot(p.pos.x - SPAWN.x, p.pos.z - SPAWN.z);
  check('respawns after crash', !p.crashed && dHoriz < 2 && Math.abs(p.pos.y - SPAWN.y) < 8,
    'dH=' + dHoriz.toFixed(2) + ' y=' + p.pos.y.toFixed(1));
}

// 9. Long stress flight: random-ish inputs, physics must stay finite.
{
  const p = new BikePhysics();
  let t = 0;
  run(p, 60, () => {
    t += STEP;
    p.throttle = 0.5 + 0.5 * Math.sin(t * 0.3);
    p.collective = 0.7 + 0.4 * Math.sin(t * 0.17 + 1);
    p.steer = Math.sin(t * 0.5);
    p.input.boostRear = Math.sin(t * 0.9) > 0.7;
    p.input.boostC = Math.sin(t * 1.3) > 0.8;
  });
  const finite = [p.pos, p.vel, p.angVel].every(v => isFinite(v.x) && isFinite(v.y) && isFinite(v.z)) && isFinite(p.quat.w);
  check('60s stress flight stays finite', finite,
    'pos=(' + p.pos.toArray().map(v => v.toFixed(0)).join(',') + ') v=' + p.vel.length().toFixed(1) + ' crashed=' + p.crashed);
}

console.log(failures === 0 ? '\nALL TESTS PASSED' : '\n' + failures + ' FAILURES');
process.exit(failures ? 1 : 0);
