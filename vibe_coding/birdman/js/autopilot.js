// A simple pilot that flies the chapters with the same inputs a player has.
// Used by tests/missions.mjs to prove every chapter can be finished, and by ?auto=1 as a demo.
import { thermals, thermalCentre } from './sim/world.js';
import { clamp, wrapPi } from './sim/vmath.js';

export function pilot(b, g, t) {
  const s = g.step, inp = { pitch: 0, roll: 0, fwd: 0, flap: false, tuck: false, jump: false };
  if (!s) return inp;
  if (b.mode === 'ground') {
    if (s.type === 'land') return inp; // stand still to finish
    if (s.type === 'cond' && /רוץ|קפוץ/.test(s.text)) { inp.fwd = 1; return inp; }
    inp.flap = true; inp.fwd = 1; return inp; // take-off run
  }
  if (b.mode !== 'air') return inp;
  if (!b.latch) { inp.flap = t % 1 < 0.5; return inp; }
  let tx, ty, tz, circle = false;
  if (s.p) { [tx, ty, tz] = s.p; }
  else if (s.type === 'alt') {
    // nearest thermal: go to it, then circle
    let best = null, bd = 1e9;
    for (const th of thermals) { const [cx, cz] = thermalCentre(th, b.p[1]); const d = Math.hypot(cx - b.p[0], cz - b.p[2]); if (d < bd) { bd = d; best = [cx, cz, th]; } }
    tx = best[0]; tz = best[1]; ty = b.p[1]; circle = bd < best[2].R * 0.75;
    if (circle) { const toC = wrapPi(Math.atan2(-(tz - b.p[2]), tx - b.p[0]) - Math.atan2(-b.v[2], b.v[0])); inp.roll = 0.4 * clamp(1 - 0.9 * Math.cos(toC) * Math.min(1, bd / 60), 0.15, 1.6); inp.pitch = 0.35; inp.flap = b.agl < 90 && b.stamina > 0.2; return inp; }
    const e2 = wrapPi(Math.atan2(-(tz - b.p[2]), tx - b.p[0]) - Math.atan2(-b.v[2], b.v[0]));
    inp.roll = clamp(-e2 * 1.6, -0.7, 0.7) * (b.agl < 14 ? 0.3 : 1); inp.pitch = 0.2; inp.flap = b.agl < 110 && b.stamina > 0.2; return inp;
  } else if (s.type === 'speed') { inp.tuck = true; inp.pitch = -1; return inp; }
  else if (s.big) { inp.pitch = 0.7; return inp; } // the pull-out
  else { inp.flap = true; inp.pitch = 0.3; return inp; } // climbing away after take-off
  const dx = tx - b.p[0], dz = tz - b.p[2], dist = Math.hypot(dx, dz);
  const track = Math.atan2(-b.v[2], b.v[0]), want = Math.atan2(-dz, dx);
  const err = wrapPi(want - track); // + = target is to the left
  const lim = b.agl < 14 ? 0.3 : 0.85; // no steep turns near the ground
  inp.roll = clamp(-err * 1.6, -lim, lim);
  const dy = ty + (s.type === 'land' ? 0 : 0) - b.p[1];
  // height we will have when we get there on a plain glide
  const reach = dist / 9;
  if (s.type === 'land') {
    if (dist < 80 && b.agl > 25) { inp.roll = 0.6; inp.pitch = -0.2; return inp; } // too high over the spot: spiral down
    inp.pitch = b.agl < 3.5 ? 1 : b.agl < 12 ? 0.35 : dy + reach < -30 ? -0.5 : 0;
    if (dy + reach > 10 && b.agl > 15) inp.flap = true;
    if (b.agl < 12 && dist > 70) inp.flap = true;
  } else {
    if (dy > -reach + 6) { inp.flap = b.stamina > 0.12; inp.pitch = 0.25; } else if (dy < -reach - 30 && b.agl > 300 && b.airspeed < 45) { inp.tuck = true; inp.pitch = -0.5; inp.roll *= 0.3; } else if (dy < -reach - 12) inp.pitch = -0.6; else inp.pitch = clamp(dy / 30, -0.4, 0.3) ;
    if (Math.abs(err) > 1.2) inp.pitch = Math.max(inp.pitch, 0.3);
  }
  return inp;
}

