// Pilot-assist flight computer: turns stick / throttle input into commands for the five real
// turbojets and the eight nozzle vanes. Nothing here moves the bike directly; every force
// still goes through spool lag, thrust limits and the vane servos in physics.js.
//
// ASSIST (default):
//   stick left/right -> bank angle (up to 45 deg); the heading follows a coordinated turn
//                       (yaw rate = g tan(bank) / V), so the bike carves like an aircraft
//   stick fwd/back   -> pitch attitude. At speed the flight path follows the nose (dive /
//                       climb, trading speed for height); at hover it tilts the lift jets and
//                       the bike walks forward/back like a drone while altitude is held
//   rudder           -> yaw rate (vanes only)
//   climb / descend  -> vertical speed (+/- 8 m/s), otherwise altitude hold
//   throttle         -> cruise engine thrust; boost = emergency over-speed (runs hot)
// EXPERT: stick = angular rates, climb/descend moves a collective lever. No auto-level,
//   no altitude hold. You fly the attitude yourself.

import {
  add, sub, scale, madd, dot, cross, len, qmul, qconj, qrot, qinvrot, qToRotVec, qAxisAngle, m3v, clamp,
} from './vmath.js';
import { airDensity, surfaceAt } from './terrain.js';
import { G, LIFT, MAIN, MAIN_DIR, LIFT_TMAX, MAIN_TMAX, EXHAUST_V, AERO } from './vehicle.js';
import { FlightComputer } from './fcs_core.js';

export const ASSIST = {
  MAX_BANK: 58 * Math.PI / 180,
  MAX_PITCH: 26 * Math.PI / 180,
  RUDDER_RATE: 1.1,        // rad/s
  CLIMB: 8,                // m/s
  KA: [4.0, 2.6, 3.6],     // attitude gains (roll, yaw, pitch)
  KR: [8, 5.5, 7.5],       // rate gains
  KIA: [30, 20, 30],
};

const wrap = (a) => Math.atan2(Math.sin(a), Math.cos(a));

export class PilotAssist extends FlightComputer {
  constructor() {
    super(7);
    this.reset();
  }
  reset(yaw = 0) {
    this.yawDes = yaw; this.altRef = null; this.iVz = 0; this.iAtt = [0, 0, 0]; this.tauSlow = [0, 0, 0];
    this.lever = 0; this.tel = {};
  }

  update(dt, bike, inp) {
    const mp = bike.mp, m = mp.m;
    const { p, v, q, w } = bike;
    const sigma = airDensity(p[1]) / 1.225;
    const W = m * G;
    const fwd = qrot(q, [1, 0, 0]), up = qrot(q, [0, 1, 0]), mainW = qrot(q, MAIN_DIR);
    const yawNow = Math.atan2(-fwd[2], fwd[0]);
    const pitchNow = Math.asin(clamp(fwd[1], -1, 1));
    const onGround = bike.contactN > 0.25 * W;
    const Vh = Math.hypot(v[0], v[2]);
    const hdg = [Math.cos(yawNow), 0, -Math.sin(yawNow)];
    const vf = dot(v, hdg);
    const vB = qinvrot(q, v), V = len(vB);
    const rho = airDensity(p[1]);
    const liftSum = bike.lift.reduce((a, e) => a + e.Tcore, 0);

    // drag the computer knows about (it cannot see the wind), world frame
    const DB = [0, 0, 0];
    for (let k = 0; k < 3; k++) DB[k] = -0.5 * rho * AERO.cda[k] * V * vB[k];
    const Dw = qrot(q, add(DB, scale(vB, -(liftSum + bike.main.Tcore) / EXHAUST_V)));

    let collective, wDes, useAtt = true, qDes = null, wff = [0, 0, 0];
    const tmaxAll = 4 * LIFT_TMAX * sigma;

    if (inp.assist) {
      // bank limit: the lift jets must still carry W / cos(bank) with some margin
      const mainLift = bike.main.T * MAIN_DIR[1];
      const bankLim = Math.min(ASSIST.MAX_BANK, Math.acos(clamp((W - mainLift) / (0.85 * tmaxAll), 0, 1)));
      const sitting = onGround && inp.climb <= 0;
      const bank = sitting ? 0 : inp.stickX * bankLim;
      this.tel.bankLim = bankLim;
      const pitch = onGround && inp.climb <= 0 ? 0 : -inp.stickY * ASSIST.MAX_PITCH;
      const kc = clamp((Vh - 4) / 10, 0, 1);
      // right bank -> heading turns right -> yaw angle decreases
      let yawRate = -inp.rudder * ASSIST.RUDDER_RATE - kc * G * Math.tan(bank) / Math.max(Vh, 6);
      if (onGround && inp.climb <= 0) { this.yawDes = yawNow; yawRate = 0; }
      this.yawDes = yawNow + clamp(wrap(this.yawDes + yawRate * dt - yawNow), -0.5, 0.5);
      qDes = qmul(qmul(qAxisAngle([0, 1, 0], this.yawDes), qAxisAngle([0, 0, 1], pitch)), qAxisAngle([1, 0, 0], bank));
      wff = qinvrot(q, [0, yawRate, 0]);

      // vertical channel: follow the nose at speed, hold altitude when hands-off
      const kn = clamp((vf - 6) / 14, 0, 1);
      let vzCmd = inp.climb * ASSIST.CLIMB + kn * vf * Math.tan(pitchNow);
      // hands-off: hold altitude. Low down (< 40 m) hold height above the terrain instead,
      // looking ~1 s ahead along the flight path, like a helicopter's radar-altitude hold.
      const ground = Math.max(surfaceAt(p[0], p[2]), surfaceAt(p[0] + v[0], p[2] + v[2]), surfaceAt(p[0] + v[0] * 0.5, p[2] + v[2] * 0.5));
      if (Math.abs(inp.climb) < 0.05 && Math.abs(inp.stickY) < 0.08 && !onGround) {
        if (this.altRef === null) {
          const stop = p[1] + v[1] * Math.abs(v[1]) / 8;     // stop where a ~4 m/s^2 flare ends
          this.radar = stop - ground < 40;
          this.altRef = this.radar ? Math.max(3.5, stop - ground) : stop;
        }
        const yWant = this.radar ? ground + this.altRef : this.altRef;
        vzCmd += clamp(0.9 * (yWant - p[1]), -3, 5);
      } else this.altRef = null;
      this.tel.radar = this.altRef !== null && this.radar;
      vzCmd = clamp(vzCmd, -25, 12);
      // interlock: no lift-off until every lift engine is running
      const allRunning = bike.lift.every((e) => e.phase === 'run');
      if (onGround && !allRunning) vzCmd = 0;
      this.tel.ready = allRunning;
      if (onGround && vzCmd < 0.5) {
        collective = 0.12 * W;                       // sit on the pads at near idle
        this.iVz = 0; this.altRef = null;
      } else {
        const e = vzCmd - v[1];
        if (!onGround) this.iVz = clamp(this.iVz + m * 0.9 * e * dt, -0.3 * W, 0.3 * W);
        // vertical acceleration limited to about -0.55 g .. +0.6 g: the lift jets never drop
        // so low that the differential thrust needed for attitude control disappears
        const Fz = m * (G + clamp(2.4 * e, -0.55 * G, 0.6 * G)) + this.iVz;
        collective = Math.max((Fz - Dw[1] - bike.main.T * mainW[1]) / Math.max(up[1], 0.35), 0.42 * W);
      }
      this.tel.vzCmd = vzCmd;
    } else {
      // expert: rate commands and a collective lever
      useAtt = false;
      wDes = [inp.stickX * 1.9, -inp.rudder * 1.1, -inp.stickY * 1.4];
      this.lever = clamp(this.lever + inp.climb * 0.4 * dt, 0, 1);
      collective = this.lever * tmaxAll;
      this.yawDes = yawNow; this.altRef = null;
    }
    collective = clamp(collective, 0.08 * W, tmaxAll);

    // attitude -> rates -> torque
    if (useAtt) {
      const e = qToRotVec(qmul(qconj(q), qDes));
      wDes = [0, 0, 0];
      for (let k = 0; k < 3; k++) wDes[k] = ASSIST.KA[k] * e[k] + wff[k];
      const wl = len(wDes);
      if (wl > 2.2) wDes = scale(wDes, 2.2 / wl);
      const ki = onGround ? 0 : 1;
      for (let k = 0; k < 3; k++) this.iAtt[k] = clamp(this.iAtt[k] + e[k] * dt * ki, -1.5, 1.5);
    } else this.iAtt = [0, 0, 0];
    const alpha = [0, 0, 0];
    for (let k = 0; k < 3; k++) alpha[k] = ASSIST.KR[k] * (wDes[k] - w[k]);
    const Hrot = add(
      bike.lift.reduce((acc, eng, i) => madd(acc, [0, LIFT[i].spin, 0], eng.rotorI * eng.omega), [0, 0, 0]),
      scale(MAIN_DIR, bike.main.rotorI * bike.main.omega));
    let tau = add(m3v(mp.I, alpha), cross(w, add(m3v(mp.I, w), Hrot)));
    for (let k = 0; k < 3; k++) tau[k] += ASSIST.KIA[k] * this.iAtt[k];
    if (onGround) tau = scale(tau, 0.3);
    const cg = mp.cg;
    const arm = (d) => [d[0] - cg[0], d[1] - cg[1], d[2] - cg[2]];
    tau = sub(tau, cross(arm(MAIN.exit), scale(MAIN_DIR, bike.main.T)));
    for (let k = 0; k < 3; k++) {
      const f = [0, 0, 0]; f[k] = DB[k];
      tau = sub(tau, cross(AERO.cp[k], f));
    }
    for (let i = 0; i < 4; i++) tau = sub(tau, cross(arm(LIFT[i].intake), scale(vB, -bike.lift[i].Tcore / EXHAUST_V)));

    // allocation: slow torque share -> differential thrust, vanes fill the gap right now
    const kf = Math.min(1, dt / 0.25);
    for (let k = 0; k < 3; k++) this.tauSlow[k] += (tau[k] - this.tauSlow[k]) * kf;
    const T = this.allocThrust(collective, this.tauSlow, cg, LIFT_TMAX * sigma);
    const vanes = this.allocVanes(tau, bike.lift.map((e) => e.Tcore), cg);
    for (let i = 0; i < 4; i++) {
      const [a, b] = vanes[i];
      this.cmd.lift[i] = T[i] / (1 - 0.35 * (a * a + b * b));
      this.cmd.vanes[i] = [a, b];
    }
    bike.main.nMax = inp.boost ? 1.04 : 1.0;
    this.cmd.main = clamp(inp.throttle, 0, 1) * MAIN_TMAX * sigma * (inp.boost ? 1.15 : 1);
    this.tel.collective = collective;
    return this.cmd;
  }
}
