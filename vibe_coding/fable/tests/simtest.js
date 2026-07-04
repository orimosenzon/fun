// Headless physics verification for fable.
import { BikePhysics, TUNE } from './physics.js';
import { SPAWN, terrainHeight } from './world.js';
import { Wind } from './wind.js';
import * as THREE from 'three';

// legacy tests assume calm air; wind-specific tests switch this on locally
TUNE.windSpeed = 0;

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

// 3. Level flight at full throttle (player-style altitude hold) reaches high
// speed. Peak speed is tracked because the stronger jet now covers enough
// ground in 25 s to reach the world-rim mountains and scrub off speed there.
{
  const p = new BikePhysics();
  // climb above the highest peaks, then cruise at the hover-balance collective
  run(p, 16, () => { p.collective = p.pos.y < SPAWN.y + 300 ? 0.9 : 0.53; });
  let vPeak = 0;
  run(p, 20, () => { p.throttle = 1; p.collective = 0.53; vPeak = Math.max(vPeak, p.vel.length()); });
  check('reaches high speed in level flight', !p.crashed && vPeak > 60,
    'vPeak=' + vPeak.toFixed(1) + ' m/s, y=' + p.pos.y.toFixed(1) + ' crashed=' + p.crashed);
  check('top speed sane (<135 m/s)', vPeak < 135, 'vPeak=' + vPeak.toFixed(1));
  check('moves along +Z', p.pos.z - SPAWN.z > 500, 'dz=' + (p.pos.z - SPAWN.z).toFixed(0));
}

// 4. Steering right (screen-right). Body +X is the rider's LEFT in this frame,
// so a screen-right turn lifts +X (x-axis .y > 0) and curves toward world -X.
{
  const p = new BikePhysics();
  run(p, 6, () => { p.collective = 0.85; p.throttle = 0.8; });
  const xBefore = p.pos.x;
  run(p, 6, () => { p.steer = 1; p.collective = 0.95; });
  const xAxis = new THREE.Vector3(1, 0, 0).applyQuaternion(p.quat);
  check('banks screen-right when steering right', xAxis.y > 0.3, 'xAxis.y=' + xAxis.y.toFixed(2));
  check('turns screen-right (x shrinks)', p.pos.x < xBefore - 30,
    'dx=' + (p.pos.x - xBefore).toFixed(0) + ' z=' + p.pos.z.toFixed(0));
  check('no crash during turn', !p.crashed);
}

// 5. RCS pulses: one press = one fixed impulse bit (F * dur at the nozzle arm),
// exact regardless of integration step size, with a refractory gap that
// bounds how fast presses can stack.
{
  const expectedW = TUNE.pulseSide * TUNE.pulseDur * 0.75 / 48; // torque impulse / Iz
  const damp0 = TUNE.rotDamp;
  TUNE.rotDamp = 0;   // isolate the impulse from spin damping
  const p = new BikePhysics();
  run(p, 6, () => { p.collective = 0.75; });
  p.assist = false;
  p.firePulse('L');
  run(p, 0.3);
  check('L pulse rolls screen-left by the impulse bit', Math.abs(p.angVel.z + expectedW) < expectedW * 0.12,
    'wz=' + p.angVel.z.toFixed(3) + ' expected=' + (-expectedW).toFixed(3));

  // same impulse at 3x finer steps -> same roll rate (framerate independence)
  const p2 = new BikePhysics();
  run(p2, 6, () => { p2.collective = 0.75; });
  p2.assist = false;
  p2.firePulse('L');
  const n = Math.round(0.3 / (STEP / 3));
  for (let i = 0; i < n; i++) p2.step(STEP / 3);
  check('impulse bit is framerate-independent',
    Math.abs(p2.angVel.z - p.angVel.z) < Math.abs(p.angVel.z) * 0.05,
    'coarse=' + p.angVel.z.toFixed(3) + ' fine=' + p2.angVel.z.toFixed(3));

  // mashing: at most ceil(T / (dur+gap)) pulses can fire in T seconds
  const p3 = new BikePhysics();
  let fired = 0;
  run(p3, 0.5, () => { if (p3.firePulse('R')) fired++; });
  const maxPulses = Math.ceil(0.5 / (TUNE.pulseDur + TUNE.pulseGap));
  check('pulse spam is rate-limited', fired > 0 && fired <= maxPulses, 'fired=' + fired + ' max=' + maxPulses);
  TUNE.rotDamp = damp0;

  const p4 = new BikePhysics();
  run(p4, 6, () => { p4.collective = 0.75; });
  p4.assist = false;
  p4.firePulse('R');
  run(p4, 0.3);
  check('R pulse rolls screen-right', p4.angVel.z > expectedW * 0.6, 'wz=' + p4.angVel.z.toFixed(3));
}

// 5b. Pitch command: nose-up must raise the nose well past 30 deg (expanded
// range) and convert speed to climb; nose-down must drop the nose.
{
  const p = new BikePhysics();
  run(p, 6, () => { p.collective = 0.85; p.throttle = 1; });
  run(p, 8, () => { p.collective = 0.6; p.throttle = 1; });
  const y0 = p.pos.y, vy0 = p.vel.y;
  run(p, 3, () => { p.pitch = 1; });
  const fwdUp = new THREE.Vector3(0, 0, 1).applyQuaternion(p.quat).y;
  check('full pitch up reaches a steep nose angle (>33 deg)', fwdUp > 0.55, 'fwd.y=' + fwdUp.toFixed(2));
  check('pitch up climbs', p.vel.y > vy0 + 2 || p.pos.y > y0 + 10,
    'vy ' + vy0.toFixed(1) + ' -> ' + p.vel.y.toFixed(1) + ', dy=' + (p.pos.y - y0).toFixed(1));
  run(p, 2.5, () => { p.pitch = -1; });
  const fwdDown = new THREE.Vector3(0, 0, 1).applyQuaternion(p.quat).y;
  check('pitch down drops the nose', fwdDown < -0.1, 'fwd.y=' + fwdDown.toFixed(2));
}

// 5c. Fly-by-wire steering: steer commands a turn RATE, so the yaw rate at a
// given stick input must not grow with speed (coordinated turn: w = g*tan(phi)/V).
function headingOf(p) {
  const f = new THREE.Vector3(0, 0, 1).applyQuaternion(p.quat);
  return Math.atan2(f.x, f.z);
}
function turnRateAtThrottle(thr) {
  const p = new BikePhysics();
  run(p, 6, () => { p.collective = 0.85; p.throttle = thr; });
  run(p, 3, () => { p.throttle = thr; p.collective = 0.85; p.steer = 0.6; }); // settle into the turn
  const h0 = headingOf(p);
  const v0 = Math.hypot(p.vel.x, p.vel.z);
  run(p, 3, () => { p.throttle = thr; p.collective = 0.85; p.steer = 0.6; });
  const h1 = headingOf(p);
  let d = h1 - h0;
  while (d > Math.PI) d -= 2 * Math.PI;
  while (d < -Math.PI) d += 2 * Math.PI;
  return { rate: Math.abs(d) / 3, speed: (v0 + Math.hypot(p.vel.x, p.vel.z)) / 2, sign: Math.sign(d) };
}
{
  const slow = turnRateAtThrottle(0.4);
  const fast = turnRateAtThrottle(1.0);
  check('steer turns screen-right (heading decreases)', slow.sign < 0, 'sign=' + slow.sign);
  check('turn rate roughly speed-independent (fly-by-wire)',
    fast.speed > slow.speed + 8 && fast.rate < slow.rate * 1.45 && fast.rate > slow.rate * 0.35,
    'slow: ' + slow.rate.toFixed(2) + ' rad/s @ ' + slow.speed.toFixed(0) + ' m/s, fast: '
    + fast.rate.toFixed(2) + ' rad/s @ ' + fast.speed.toFixed(0) + ' m/s');
}

// 5d. Hover yaw via thrust vectoring: with some rear thrust the vectored
// nozzle turns the nose even at near-zero speed; with no thrust it barely does.
{
  const p = new BikePhysics();
  run(p, 6, () => { p.collective = 0.72; });
  const h0 = headingOf(p);
  run(p, 4, () => { p.collective = 0.72; p.throttle = 0.3; p.steer = 1; });
  let d1 = headingOf(p) - h0;
  while (d1 > Math.PI) d1 -= 2 * Math.PI;
  while (d1 < -Math.PI) d1 += 2 * Math.PI;
  const p2 = new BikePhysics();
  run(p2, 6, () => { p2.collective = 0.72; });
  const g0 = headingOf(p2);
  run(p2, 4, () => { p2.collective = 0.72; p2.throttle = 0; p2.steer = 1; });
  let d2 = headingOf(p2) - g0;
  while (d2 > Math.PI) d2 -= 2 * Math.PI;
  while (d2 < -Math.PI) d2 += 2 * Math.PI;
  check('vectored nozzle yaws the nose in hover (with thrust)', d1 < -0.25, 'dh=' + d1.toFixed(2));
  check('no thrust -> much weaker hover yaw', Math.abs(d2) < Math.abs(d1) * 0.5,
    'with=' + d1.toFixed(2) + ' without=' + d2.toFixed(2));
}

// 6. Center pulse adds lift along body up: from hover, one pulse kicks the
// vertical speed by ~F*dur/m.
{
  const p = new BikePhysics();
  run(p, 6, () => { p.collective = 0.72; });
  const vy0 = p.vel.y;
  p.firePulse('C');
  run(p, 0.3);
  const dv = TUNE.pulseCenter * TUNE.pulseDur / TUNE.mass;
  check('center pulse kicks vertical speed', p.vel.y > vy0 + dv * 0.5,
    'vy ' + vy0.toFixed(2) + ' -> ' + p.vel.y.toFixed(2) + ' (bit=' + dv.toFixed(2) + ')');
}

// 6b. Turbine spool: a throttle step reaches ~63% after one time constant and
// converges without ever exceeding the command (first-order lag).
{
  const p = new BikePhysics();
  run(p, TUNE.spoolUp, () => { p.throttle = 1; });
  check('spool reaches ~63% after tau', Math.abs(p.spoolRear - 0.632) < 0.07,
    'spool=' + p.spoolRear.toFixed(3));
  let over = false;
  run(p, 5, () => { p.throttle = 1; if (p.spoolRear > 1.0001) over = true; });
  check('spool converges to command, never overshoots', p.spoolRear > 0.95 && !over,
    'spool=' + p.spoolRear.toFixed(3));
}

// 6c. Rotor gyroscopics: a nose-up pitch rate at full spool must precess into
// yaw (T.y = wx * h, wx < 0 nose-up => yaw right); without spool, almost none.
{
  function yawFromPitch(thr) {
    const p = new BikePhysics();
    p.pos.y = SPAWN.y + 2500;
    p.assist = false;
    p.collective = 0;
    run(p, 5, () => { p.vel.set(0, 0, 0); p.throttle = thr; }); // spool up in place
    let sum = 0, n = 0;
    run(p, 0.8, () => {
      p.vel.set(0, 0, 0);          // keep aero out of it
      p.throttle = thr;
      p.pitch = 1;                 // raw-mode nose-up torque
      sum += p.angVel.y; n++;
    });
    return sum / n;
  }
  const spooled = yawFromPitch(1);
  const idle = yawFromPitch(0);
  check('pitch-up at full spool precesses into yaw (right)', spooled < -0.01,
    'mean wy=' + spooled.toFixed(4));
  check('no spool -> (almost) no coupling', Math.abs(idle) < Math.abs(spooled) * 0.25,
    'idle wy=' + idle.toFixed(4) + ' spooled=' + spooled.toFixed(4));
}

// 6d. Conservation including the rotor: in torque-free tumble the WORLD-frame
// total angular momentum R*(L + h*z) stays constant even at full spool.
{
  const gust0 = TUNE.gust, damp0 = TUNE.rotDamp;
  TUNE.gust = 0;
  TUNE.rotDamp = 0;
  const p = new BikePhysics();
  p.pos.y = SPAWN.y + 3000;
  p.assist = false;
  p.collective = 0;
  run(p, 6, () => { p.vel.set(0, 0, 0); p.throttle = 1; }); // spool to full
  p.angMom.set(80, 40, 30);
  p.angVel.set(80 / p.I.x, 40 / p.I.y, 30 / p.I.z);
  const totalL = () => p.angMom.clone()
    .add(new THREE.Vector3(0, 0, TUNE.rotorH * p.spoolRear))
    .applyQuaternion(p.quat);
  const L0 = totalL();
  const steps = Math.round(0.5 / STEP);
  for (let i = 0; i < steps; i++) { p.vel.set(0, 0, 0); p.throttle = 1; p.step(STEP); }
  const L1 = totalL();
  const drift = L0.clone().normalize().angleTo(L1.clone().normalize());
  check('total momentum (frame+rotor) conserved in tumble',
    drift < 0.02 && Math.abs(L1.length() - L0.length()) / L0.length() < 0.015,
    'drift=' + (drift * 180 / Math.PI).toFixed(2) + 'deg |L| ' + L0.length().toFixed(1) + '->' + L1.length().toFixed(1));
  TUNE.gust = gust0;
  TUNE.rotDamp = damp0;
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
    if (Math.sin(t * 1.3) > 0.8) p.firePulse('C');
    if (Math.sin(t * 2.1) > 0.9) p.firePulse('L');
  });
  const finite = [p.pos, p.vel, p.angVel].every(v => isFinite(v.x) && isFinite(v.y) && isFinite(v.z)) && isFinite(p.quat.w);
  check('60s stress flight stays finite', finite,
    'pos=(' + p.pos.toArray().map(v => v.toFixed(0)).join(',') + ') v=' + p.vel.length().toFixed(1) + ' crashed=' + p.crashed);
}

// 10. Angular momentum conservation: torque-free tumbling in a vacuum-like
// setup must keep the WORLD-frame angular momentum vector constant.
{
  const gust0 = TUNE.gust, damp0 = TUNE.rotDamp;
  TUNE.gust = 0;                           // isolate the integrator from gusts
  TUNE.rotDamp = 0;                        // anisotropic damping would rotate L physically
  const p = new BikePhysics();
  p.pos.y = SPAWN.y + 3000;               // far from ground
  p.collective = 0;
  p.assist = false;                        // no controller torques
  p.vel.set(0, 0, 0);
  p.angMom.set(120, 260, 90);              // arbitrary tumble
  p.angVel.set(120 / p.I.x, 260 / p.I.y, 90 / p.I.z);
  // strip air: zero drag/damping isn't exposed, so measure over a short window
  // where aero torques are tiny (v=0 -> no weathervane; only spin damping).
  const L0 = p.angMom.clone().applyQuaternion(p.quat);
  const steps = Math.round(0.5 / STEP);
  for (let i = 0; i < steps; i++) {
    p.vel.set(0, 0, 0);                    // hold airspeed at zero
    p.step(STEP);
  }
  const L1 = p.angMom.clone().applyQuaternion(p.quat);
  // spin damping bleeds magnitude; direction drift is the integrator's error
  const drift = L0.clone().normalize().angleTo(L1.clone().normalize());
  check('angular momentum conserved (<0.5deg, <1% magnitude over 0.5s tumble)',
    drift < 0.009 && Math.abs(L1.length() - L0.length()) / L0.length() < 0.01,
    'drift=' + (drift * 180 / Math.PI).toFixed(3) + 'deg |L| ' + L0.length().toFixed(1) + '->' + L1.length().toFixed(1));
  TUNE.gust = gust0;
  TUNE.rotDamp = damp0;
}

// 11. Dryden turbulence: seeded gusts stay finite, near zero-mean, sane sigma.
{
  const w = new Wind(123, { speed: 6, dir: 0.7 });
  let sx = 0, s2 = 0, n = 0, bad = false, peak = 0;
  for (let i = 0; i < 120 * 80; i++) {
    w.update(1 / 120, 25, 300, -400, 60);
    const gx = w.vec.x - w.W.x, gy = w.vec.y, gz = w.vec.z - w.W.z;
    if (!isFinite(gx + gy + gz)) bad = true;
    peak = Math.max(peak, Math.abs(gx), Math.abs(gy), Math.abs(gz));
    sx += gx; s2 += gx * gx; n++;
  }
  const mean = sx / n, std = Math.sqrt(Math.max(0, s2 / n - mean * mean));
  check('Dryden gusts finite and near zero-mean', !bad && Math.abs(mean) < 0.5,
    'mean=' + mean.toFixed(3) + ' finite=' + !bad);
  check('Dryden gust intensity sane (0.05..3 m/s std, peak<8)', std > 0.05 && std < 3 && peak < 8,
    'std=' + std.toFixed(2) + ' peak=' + peak.toFixed(2));
}

// 12. Prevailing wind: a hovering bike drifts downwind (aero uses air-relative flow).
{
  TUNE.windSpeed = 5;
  const p = new BikePhysics();
  p.pos.y = SPAWN.y + 60;
  const x0 = p.pos.x, z0 = p.pos.z;
  run(p, 15, () => { p.collective = 0.62; });
  const Wd = p.wind.W.clone().normalize();
  const drift = (p.pos.x - x0) * Wd.x + (p.pos.z - z0) * Wd.z;
  check('hovering bike drifts downwind', !p.crashed && drift > 3, 'drift=' + drift.toFixed(1) + ' m');
  TUNE.windSpeed = 0;
}

console.log(failures === 0 ? '\nALL TESTS PASSED' : '\n' + failures + ' FAILURES');
process.exit(failures ? 1 : 0);
