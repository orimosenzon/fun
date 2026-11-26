/* The F-16's flight model, flown in Node: node tests/test_f16core.mjs */
import { F16, AC, KT, FT, air, liftCoef } from '../web/f16_core.js';
const RAD = Math.PI / 180;
let fails = 0;
const check = (name, ok, detail = '') => { console.log((ok ? '  ok   ' : '  FAIL ') + name + (ok ? '' : '   ' + detail)); if (!ok) fails++; };
const run = (a, secs, each) => { const dt = 1 / 60; for (let t = 0; t < secs; t += dt) { if (each) each(a, t); a.step(dt); if (a.crashed) break; } };

// 1. hands off, it holds its path
{
  const a = new F16(); a.reset({ alt: 1500, speed: 200 }); a.lever = 0.5;
  run(a, 20);
  check('hands off: wings level', Math.abs(a.roll) < 1, a.roll);
  check('hands off: path within 2 degrees of level', Math.abs(a.gamma) < 2, a.gamma);
  check('hands off: 1 g', Math.abs(a.nz - 1) < 0.08, a.nz);
  check('alpha a few degrees', a.alpha / RAD > 0.5 && a.alpha / RAD < 8, a.alpha / RAD);
}
// 2. top speeds
{
  const a = new F16(); a.reset({ alt: 150, speed: 250 }); a.lever = AC.MIL;
  a.ap = true; a.apAlt = 150; a.apRoll = 0;
  run(a, 150);
  check('mil power at sea level tops out near Mach 1', a.mach > 0.9 && a.mach < 1.04, a.mach);
  a.lever = 1; let t1 = null;
  run(a, 90, (x, t) => { if (t1 === null && x.mach >= 1.1) t1 = t; });
  check('burner at sea level: Mach 1.2, held by the limiter', a.mach > 1.15 && a.mach < 1.26, a.mach);
  check('cas near 800 kt', a.cas * KT > 760 && a.cas * KT < 830, a.cas * KT);
  const b = new F16(); b.reset({ alt: 12000, speed: 300 }); b.lever = 1; b.ap = true; b.apAlt = 12000; b.apRoll = 0;
  run(b, 400);
  check('burner at 12 km: about Mach 2', b.mach > 1.85 && b.mach < 2.1, b.mach);
}
// 3. acceleration and g
{
  const a = new F16(); a.reset({ alt: 300, speed: 150 }); a.lever = 1; a.ap = true; a.apAlt = 300; a.apRoll = 0;
  let t = null; run(a, 60, (x, tt) => { if (t === null && x.mach >= 0.95) t = tt; });
  check('300 kt to Mach 0.95 in the burner under 25 s', t !== null && t < 25, t);
  const b = new F16(); b.reset({ alt: 1500, speed: 250 }); b.lever = 1;
  let peak = 0; run(b, 4, (x) => { x.stickY = 1; peak = Math.max(peak, x.nz); });
  check('full aft stick at 485 kt: 9 g, not more', peak > 8.5 && peak < 9.4, peak);
  const c = new F16(); c.reset({ alt: 1500, speed: 100 }); c.lever = 1;
  let amax = 0, gmax = 0; run(c, 5, (x) => { x.stickY = 1; amax = Math.max(amax, x.alpha / RAD); gmax = Math.max(gmax, x.nz); });
  check('full aft stick at 195 kt: alpha limited to 25', amax > 22 && amax < 26.5, amax);
  check('and the g is what the wing has', gmax < 4.2, gmax);
}
// 4. a roll and a loop come out where they went in
{
  const a = new F16(); a.reset({ alt: 2000, speed: 220 }); a.lever = AC.MIL;
  let turned = 0, last = 0;
  run(a, 1.6, (x) => { x.stickX = 1; const d = ((x.roll - last + 540) % 360) - 180; turned += d; last = x.roll; });
  check('full stick roll: about 240 deg/s', turned / 1.6 > 180 && turned / 1.6 < 250, turned / 1.6);
  const b = new F16(); b.reset({ alt: 1500, speed: 250, bearing: 90 }); b.lever = 1;
  let top = 0, inv = false, done = null;
  run(b, 40, (x, t) => { x.stickY = done === null ? 0.5 : 0; top = Math.max(top, x.pos[1]); if (Math.abs(x.roll) > 150) inv = true; if (inv && done === null && Math.abs(x.roll) < 20 && x.pitch > -5) done = t; });
  check('a loop: over the top inverted and round again', inv && done !== null, `${inv} ${done}`);
  check('loop is a few thousand feet tall', top - 1500 > 600 && top - 1500 < 3500, top - 1500);
  check('loop comes out on the heading it went in on', Math.abs(((b.heading - 90 + 540) % 360) - 180) < 12, b.heading);
}
// 5. a coordinated turn
{
  const a = new F16(); a.reset({ alt: 1500, speed: 200, bearing: 0 }); a.lever = 0.62;
  run(a, 4, (x) => { x.stickX = x.roll < 45 ? 0.3 : 0; });
  const bank = a.roll;
  a.ap = true; a.apAlt = 1500; a.apRoll = bank * RAD;
  let maxB = 0;
  run(a, 14, (x) => { x.stickX = 0; maxB = Math.max(maxB, Math.abs(x.beta / RAD)); });
  const h0 = a.heading;
  run(a, 6);
  const rate = (((a.heading - h0) + 540) % 360 - 180) / 6;
  const expect = 9.81 * Math.tan(a.roll * RAD) / a.tas / RAD;
  check('banked ' + bank.toFixed(0) + ' deg: turn rate is g tan(bank) / v', Math.abs(rate - expect) < 0.25 * Math.abs(expect) + 0.3, `${rate} vs ${expect}`);
  check('no sideslip to speak of', maxB < 2.5, maxB);
  check('autopilot holds the height in the turn', Math.abs(a.pos[1] - 1500) < 40, a.pos[1]);
}
// 6. take-off, and a landing
{
  const a = new F16(); a.reset({ ground: true, bearing: 45 });
  check('parked at idle', a.grounded && a.vg === 0);
  run(a, 3); check('idle does not creep with brakes off much', a.vg < 3, a.vg);
  a.lever = 1; let rot = null, off = null, dist = 0;
  run(a, 40, (x, t) => { if (x.cas * KT > 140) x.stickY = 0.6; if (!x.grounded && off === null) { off = t; dist = Math.hypot(x.pos[0], x.pos[2]); } if (off !== null) x.stickY = Math.max(-0.2, Math.min(0.3, 0.04 * (10 - x.pitch))); });
  check('lifts off in the burner', off !== null, off);
  check('lift-off roll 300-900 m', dist > 300 && dist < 900, dist);
  check('climbing away', a.pos[1] > 100 && !a.crashed, `${a.pos[1]} ${a.crashWhy}`);
  // landing: a stabilised approach flown by a crude pilot
  const b = new F16(); b.reset({ alt: 150, speed: 80, bearing: 0 }); b.gearDown = true; b.gear = 1; b.fuel = 900;
  run(b, 120, (x) => {
    if (x.grounded) { x.lever = 0; x.brake = true; x.stickY = 0; return; }
    const vzT = x.pos[1] > 6 ? -3.2 : -0.8;
    x.stickY = Math.max(-0.4, Math.min(0.5, 0.06 * (vzT - x.vel[1])));
    x.lever = Math.max(0, Math.min(1, 0.45 + 0.03 * (74 - x.tas)));
  });
  check('a landing is a landing', b.grounded && !b.crashed, b.crashWhy + ' sink ' + b.sinkAtTouch);
  check('stops on the brakes', b.vg < 1, b.vg);
  const c = new F16(); c.reset({ alt: 60, speed: 120 }); c.lever = 0.3;
  run(c, 20, (x) => { x.stickY = -0.15; });
  check('gear up into the ground is a crash', c.crashed && c.crashWhy === 'gear', c.crashWhy);
}
// 7. slow flight: no hovering, no spin
{
  const a = new F16(); a.reset({ alt: 3000, speed: 150 }); a.lever = 0;
  let minV = 1e9; run(a, 40, (x) => { x.stickY = 1; minV = Math.min(minV, x.tas); });
  check('idle and stick held back: mushes down, wings level', Math.abs(a.roll) < 5 && a.pos[1] < 2900 && minV > 25 && !isNaN(a.tas), `${a.roll} ${a.pos[1]} ${minV}`);
  check('alpha held at the limit', a.alpha / RAD < 26.5, a.alpha / RAD);
  const b = new F16(); b.reset({ alt: 3000, speed: 120 }); b.q = (new F16()).q; b.lever = 0;
  // straight up, and out of speed
  const up = new F16(); up.reset({ alt: 3000, speed: 200 }); up.lever = 0;
  let pmax = -90, ok = true; run(up, 6, (x) => { x.stickY = 0.6; });
  let slow = 1e9; run(up, 28, (x) => { x.stickY = 0; slow = Math.min(slow, x.tas); ok = ok && isFinite(x.pos[1]) && isFinite(x.q[0]); });
  check('a zoom to nothing: out of speed at the top', slow < 30, slow);
  check('and it falls through nose down and flies again', ok && up.tas > 80 && up.pitch < -20 && Math.abs(up.alpha) < 20 * RAD && !up.crashed, `${up.tas} ${up.pitch}`);
}
// 8. fuel
{
  const a = new F16(); a.reset({ alt: 300, speed: 200 }); a.lever = 1; a.ap = true; a.apAlt = 300; a.apRoll = 0;
  run(a, 5); const pph = a.ff * 3600 * 2.2046;
  check('burner at sea level drinks 45-75 thousand pounds an hour', pph > 45000 && pph < 80000, pph);
  a.lever = 0.5; run(a, 15); const pph2 = a.ff * 3600 * 2.2046;
  check('cruise is a few thousand', pph2 > 1500 && pph2 < 9000, pph2);
}
console.log(fails ? `\n${fails} FAILED` : '\nall passed');
process.exit(fails ? 1 : 0);
