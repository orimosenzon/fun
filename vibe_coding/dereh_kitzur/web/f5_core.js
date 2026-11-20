/* The F-5 of FÖHN, flown over the map: the flight itself, with nothing drawn.
 *
 * The bike, its engine and its flight computer are FÖHN's own files
 * (../../jetbikes/foehn), imported and not copied, so the bike here is the
 * one that game has today. What is in this file is the seam: the flat ground
 * the map is, and the controls of the flight mode (explore.js) translated
 * into what FÖHN's flight computer is asked: a turn, a climb, a side-step and
 * a wanted speed.
 *
 * explore.js was written for the first bike (bike.js, fable's FLYING HOG) and
 * talks to its physics object directly: it writes throttle, steer, pitch,
 * vert and brake, and reads pos, vel and the engine's state for the
 * instruments and the sound. F5 below answers to the same names, so the two
 * bikes are interchangeable to it. The differences that cannot be hidden:
 *   - there is no manual mode. The F-5 has no wings and no wheels and is
 *     flown by its computer or not at all; `assist` is always true.
 *   - the three pulses are not reaction thrusters here. Z and X are a kick of
 *     the side nozzles (a side-step), Shift a short climb.
 *
 * The frame is the one bike.js uses, so explore.js converts nothing: x east,
 * y up, z south, metres from the take-off point. FÖHN's yaw turns from +x
 * towards -z, so a compass bearing b is a yaw of pi/2 - b.
 *
 * No three.js in here, on purpose: tests/test_f5.mjs flies this file in Node.
 */
import { T } from '../../jetbikes/foehn/js/sim/terrain.js';
import { Sim } from '../../jetbikes/foehn/js/sim/sim.js';
import { G, ENG } from '../../jetbikes/foehn/js/sim/bike.js';
import { V_MAX, V_MIN } from '../../jetbikes/foehn/js/sim/fcs.js';
import { makeWind } from '../../jetbikes/foehn/js/sim/wind.js';
import { len, qrot, clamp } from '../../jetbikes/foehn/js/sim/vmath.js';

export { V_MAX };

/** The map is drawn without relief in the flight mode, so the ground is flat
 *  and at zero, and there is no water to fall into. */
export const FLAT = { groundAt: () => 0, normalAt: () => [0, 1, 0], waterAt: () => -Infinity };

const KICK_S = 0.45;      // seconds a side kick (Z, X) lasts
const HOP_S = 0.5;        // and the climb on Shift
const CRASH_S = 2.0;      // seconds of wreck before the bike is back on the ground

const vec = () => ({ x: 0, y: 0, z: 0, length() { return Math.hypot(this.x, this.y, this.z); } });

export const yawFor = (bearingDeg) => Math.PI / 2 - bearingDeg * Math.PI / 180;

export class F5 {
  constructor() {
    T.ext = FLAT;
    // A light sea breeze from the west, gusting a little: enough for a hover
    // to drift and for the compass to have a wind to show.
    this.sim = new Sim(makeWind({ heading: 0, speed: 2.5, gust: 0.5, seed: 11 }));
    this.b = this.sim.bike;

    // what explore.js writes
    this.throttle = 0; this.brake = false; this.vert = 0; this.steer = 0; this.pitch = 0;
    this.collective = 0; this.input = { boostRear: false };
    this.autoLift = true;
    // what it reads
    this.pos = vec(); this.vel = vec(); this.angVel = vec();
    this.ab = 0; this.jetRear = 0; this.jetLift = 0; this.spoolRear = 0; this.spoolLift = 0;
    this.grounded = true; this.crashed = false;
    this.wind = { W: vec(), gustMag: 0, speed: 0 };
    this.pulses = { L: 0, R: 0, C: 0 };

    this.lean = 0;            // the rider's lean into the turn, for the model
    this.alive = true;
    this._crashT = 0; this._crashEvent = null;
    this.place(0, 0, 0);
  }

  get assist() { return true; }
  set assist(v) { /* always on: see the header */ }

  /** On the ground at (x, z), facing `yaw` (FÖHN's yaw: see yawFor). */
  place(x, z, yaw) {
    T.ext = FLAT;
    this.sim.reset([x, 0, z], yaw, 0);
    this.b.crash = null;
    this.crashed = false; this.alive = true; this._crashT = 0; this._crashEvent = null;
    this.pulses.L = this.pulses.R = this.pulses.C = 0;
    this.lean = 0;
    this._read();
  }

  yaw() { const n = qrot(this.b.q, [1, 0, 0]); return Math.atan2(-n[2], n[0]); }

  /** Z / X / Shift. True if the kick was taken (for its sound). */
  firePulse(which) {
    if (this.crashed || this.pulses[which] > 0) return false;
    this.pulses[which] = which === 'C' ? HOP_S : KICK_S;
    return true;
  }

  /** R: the horizon back, where the bike is, keeping its speed over the ground. */
  level() {
    const b = this.b, p = b.p.slice(), yaw = this.yaw();
    const sp = Math.hypot(b.v[0], b.v[2]);
    this.sim.reset([p[0], p[1], p[2]], yaw, Math.max(sp, 0.01));
    b.p[0] = p[0]; b.p[1] = Math.max(p[1], b.mp.cg[1]); b.p[2] = p[2];
    b.crash = null;
    this._read();
  }

  consumeCrashEvent() { const e = this._crashEvent; this._crashEvent = null; return e; }

  /** Advance by `dt` seconds of wall clock. */
  step(dt) {
    const b = this.b, P = this.pulses;
    if (this.crashed) {
      this._crashT -= dt;
      if (this._crashT <= 0) this.place(b.p[0], b.p[2], this.yaw());
      return;
    }
    for (const k of ['L', 'R', 'C']) P[k] = Math.max(0, P[k] - dt);

    const nose = qrot(b.q, [1, 0, 0]);
    const fwd = b.v[0] * nose[0] + b.v[1] * nose[1] + b.v[2] * nose[2];
    const boost = !!this.input.boostRear;
    const climb = clamp(this.pitch + this.vert + (P.C > 0 ? 1 : 0), -1, 1);

    // The lever is a wanted speed, as in FÖHN: explore.js squares it (its
    // `throttle`), which keeps the first half of the travel under 90 km/h,
    // the speeds a street is flown at.
    let vt = this.throttle * V_MAX;
    if (this.brake) vt = Math.max(0, Math.min(vt, fwd) - 30 * dt);
    if (boost && !b.abLock) vt = clamp(Math.max(vt, fwd + 3), V_MIN, V_MAX + 30);
    // Parked: on its skids, slow, and nobody asking for anything. Without
    // this the computer holds a hover a hand above the ground for ever.
    if (b.contactN > 0.25 * b.mp.m * G && len(b.v) < 4 && !boost && climb <= 0.05 && this.throttle < 0.002) vt = 0;

    this.sim.advance(dt, {
      turn: clamp(this.steer, -1, 1), pitch: climb,
      strafe: (P.R > 0 ? 1 : 0) - (P.L > 0 ? 1 : 0),
      vt, boost, brake: !!this.brake
    });
    this.lean += (clamp(this.steer, -1, 1) - this.lean) * Math.min(1, dt * 5);

    if (b.crash) {
      this.crashed = true; this.alive = false;
      this._crashT = CRASH_S;
      this._crashEvent = 'ground';
    }
    this._read();
  }

  _read() {
    const b = this.b, w = this.sim.windW || [0, 0, 0];
    this.pos.x = b.p[0]; this.pos.y = Math.max(0, b.p[1] - b.mp.cg[1]); this.pos.z = b.p[2];
    this.vel.x = b.v[0]; this.vel.y = b.v[1]; this.vel.z = b.v[2];
    this.angVel.x = b.w[0]; this.angVel.y = b.w[1]; this.angVel.z = b.w[2];
    this.grounded = b.contactN > 0;
    // The instruments and the voice of the first bike, fed from this one:
    // thrust leaving the rear nozzle and the four lift pipes, as shares.
    const lift = b.L[0] + b.L[1] + b.L[2] + b.L[3];
    const weight = b.mp.m * G;
    this.spoolRear = clamp(b.R / (ENG.tmax * 0.6), 0, 1);
    this.ab = clamp(b.abOut, 0, 1);
    this.spoolLift = clamp(lift / weight * 0.8, 0, 1.25);
    this.collective = this.spoolLift;
    this.jetRear = b.R / ENG.tmax * 8400 * 1.6;
    this.jetLift = lift / weight * 7800 * 0.7;
    this.wind.W.x = w[0]; this.wind.W.y = w[1]; this.wind.W.z = w[2];
    this.wind.speed = Math.hypot(w[0], w[2]);
    this.wind.gustMag = Math.abs(this.wind.speed - 2.5);
  }
}
