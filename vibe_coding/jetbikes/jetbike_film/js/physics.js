// Rigid-body flight dynamics of the jet motorcycle.
// 6 DOF, integrated at 2 kHz (semi-implicit Euler, quaternion attitude). Forces modelled:
//   gravity; 5 turbojets with spool dynamics (thrust ~ rpm^3.2, first-order lag + FADEC
//   acceleration limits, start/shutdown sequences); two-axis thrust-vectoring vanes on the
//   lift nozzles (servo lag + rate limit); intake momentum drag of every engine, applied at
//   its intake; anisotropic body drag with separate centres of pressure; rate damping;
//   gyroscopic coupling of the spinning rotors (and their reaction torque when spooling);
//   jet "suck-down" and hot-gas re-ingestion near the ground; air density vs altitude;
//   fuel burn changing mass, centre of gravity and inertia; spring-damper landing pads
//   with Coulomb friction; wind with frozen-turbulence gusts.

import {
  add, sub, scale, madd, dot, cross, len, norm, qmul, qnorm, qrot, qinvrot, m3v, m3inv, clamp,
} from './vmath.js';
import { heightAt, surfaceAt, normalAt, airDensity } from './terrain.js';
import {
  G, LIFT, MAIN, PADS, LIFT_TMAX, MAIN_TMAX, VANE_MAX, TSFC, EXHAUST_V, IDLE_N, ROTOR_I,
  MAIN_ROTOR_I, OMEGA_MAX, MAIN_OMEGA_MAX, FUEL0, AERO, massProps, thrustOfN, nOfThrust,
} from './vehicle.js';

const RHO0 = 1.225;

export class Turbojet {
  constructor(tmax, omegaMax, rotorI) {
    this.tmax = tmax; this.omegaMax = omegaMax; this.rotorI = rotorI;
    this.N = 0;           // spool speed / max
    this.phase = 'off';   // off | start | run | stop
    this.phaseT = 0;
    this.T = 0;           // delivered thrust (after ground effect), N
    this.Tcore = 0;       // thrust at the nozzle before ground effect
    this.dNdt = 0;
    this.fuelFlow = 0;
  }
  start() { if (this.phase === 'off') { this.phase = 'start'; this.phaseT = 0; } }
  stop() { if (this.phase !== 'off') { this.phase = 'stop'; this.phaseT = 0; } }
  update(dt, Tcmd, sigma) {
    const N0 = this.N;
    this.phaseT += dt;
    if (this.phase === 'start') {
      // electric starter to 14 %, ignition at 1.6 s, then fuel-limited acceleration to idle
      if (this.phaseT < 1.6) this.N = Math.min(0.14, this.N + 0.11 * dt);
      else this.N += (0.05 + 0.25 * this.N) * dt;
      if (this.N >= IDLE_N) { this.N = IDLE_N; this.phase = 'run'; }
    } else if (this.phase === 'run') {
      const Nc = clamp(nOfThrust(Tcmd, this.tmax, sigma), IDLE_N, 1.0);
      let d = (Nc - this.N) / 0.2;
      // FADEC acceleration schedule (surge margin) and deceleration limit (flame-out margin)
      d = clamp(d, -0.9, 0.25 + 0.45 * this.N);
      this.N = clamp(this.N + d * dt, IDLE_N, 1.0);
    } else {
      // fuel off: rotor spins down on bearing and pumping losses
      this.N = Math.max(0, this.N - (this.N * 0.35 + 0.004) * dt);
      if (this.phase === 'stop' && this.N < 0.25) this.phase = 'off';
    }
    this.dNdt = (this.N - N0) / dt;
    const lit = this.phase === 'run' || (this.phase === 'start' && this.phaseT > 1.6);
    this.Tcore = lit ? thrustOfN(this.N, this.tmax, sigma) : 0;
    this.fuelFlow = lit ? TSFC * this.Tcore + 0.004 : 0;
  }
  get omega() { return this.N * this.omegaMax; }
  get egt() { // exhaust gas temperature, K (for visuals / sound)
    const lit = this.phase === 'run' || (this.phase === 'start' && this.phaseT > 1.6);
    return lit ? 780 + 280 * Math.pow(this.N, 2) : 300 + 400 * Math.max(0, this.N - 0.1);
  }
}

// Frozen-turbulence wind field (Taylor hypothesis): mean wind + gust modes
export function makeWind(seed = 7) {
  const mean = [2.6, 0, 1.4];
  const U = Math.hypot(mean[0], mean[2]);
  const dir = [mean[0] / U, 0, mean[2] / U];
  let s = seed;
  const r = () => ((s = (s * 16807) % 2147483647) / 2147483647);
  const modes = [];
  for (let k = 0; k < 14; k++) {
    const f = 0.025 * Math.pow(1.45, k);          // Hz
    const amp = Math.pow(f / 0.025, -1 / 3);      // von Karman slope for log-spaced modes
    const ang = r() * Math.PI * 2;
    modes.push({ f, amp, ph: [r() * 6.28, r() * 6.28, r() * 6.28], lat: [Math.cos(ang), Math.sin(ang)] });
  }
  const norm0 = Math.sqrt(modes.reduce((a, m) => a + m.amp * m.amp, 0) / 2);
  const sig = [1.3, 0.7, 1.1];
  return function wind(p, t) {
    const agl = Math.max(0.2, p[1] - surfaceAt(p[0], p[2]));
    const prof = clamp(Math.log(agl / 0.05) / Math.log(40 / 0.05), 0.25, 1.25); // log boundary layer
    const w = [mean[0] * prof, 0, mean[2] * prof];
    const along = p[0] * dir[0] + p[2] * dir[2];
    const across = -p[0] * dir[2] + p[2] * dir[0];
    for (const m of modes) {
      const k = 2 * Math.PI * m.f / U;
      const arg = 2 * Math.PI * m.f * t - k * along + 0.6 * k * across * m.lat[0] + 0.4 * k * p[1] * m.lat[1];
      const a = m.amp / norm0;
      w[0] += sig[0] * prof * a * Math.sin(arg + m.ph[0]);
      w[1] += sig[1] * Math.min(1, agl / 15) * a * Math.sin(arg * 1.07 + m.ph[1]);
      w[2] += sig[2] * prof * a * Math.sin(arg * 0.93 + m.ph[2]);
    }
    return w;
  };
}

export class JetBike {
  constructor(pos = [0, 0, 0], yaw = 0) {
    this.fuel = FUEL0;
    this.mp = massProps(this.fuel);
    // start resting on the pads
    this.p = [pos[0], pos[1] + this.mp.cg[1] - 0.012, pos[2]];
    this.v = [0, 0, 0];
    this.q = [Math.cos(yaw / 2), 0, Math.sin(yaw / 2), 0];
    this.w = [0, 0, 0];
    this.lift = LIFT.map(() => new Turbojet(LIFT_TMAX, OMEGA_MAX, ROTOR_I));
    this.main = new Turbojet(MAIN_TMAX, MAIN_OMEGA_MAX, MAIN_ROTOR_I);
    this.vanes = LIFT.map(() => [0, 0]);   // [fore-aft, lateral] tilt of each lift jet, rad
    this.t = 0;
    this.contactN = 0;                     // total normal pad force, N
    this.padForces = PADS.map(() => 0);
    this.acc = [0, 0, 0];                  // world acceleration (for g-meter)
    this.groundFx = LIFT.map(() => 1);
  }

  // body-frame lever arm of a design-frame point
  arm(d) { const c = this.mp.cg; return [d[0] - c[0], d[1] - c[1], d[2] - c[2]]; }
  toWorld(d) { return add(this.p, qrot(this.q, this.arm(d))); }

  step(dt, cmd, wind) {
    // mass properties change slowly (fuel burn): refresh at 100 Hz
    if (!this._mpAge || this._mpAge-- <= 0) { this.mp = massProps(this.fuel); this.Iinv = m3inv(this.mp.I); this._mpAge = 20; }
    const mp = this.mp, Iinv = this.Iinv;
    // terrain under the bike, refreshed at 200 Hz (used for ground effect and contact culling)
    if (!this._gAge || this._gAge-- <= 0) { this.groundY = surfaceAt(this.p[0], this.p[2]); this._gAge = 10; }
    const rho = airDensity(this.p[1]);
    const sigma = rho / RHO0;
    const q = this.q, w = this.w;

    const vAirW = sub(this.v, wind(this.p, this.t));
    const vAirB = qinvrot(q, vAirW);
    let F = [0, 0, 0];      // body-frame force
    let Tq = [0, 0, 0];     // body-frame torque about CG
    let H = [0, 0, 0];      // rotor angular momentum, body frame
    let fuelFlow = 0;

    const applyAt = (f, r) => { F = add(F, f); Tq = add(Tq, cross(r, f)); };
    const ramDrag = (eng, rIntake) => {
      const mdot = eng.Tcore / EXHAUST_V;   // intake air mass flow
      const vLocal = add(vAirB, cross(w, rIntake));
      applyAt(scale(vLocal, -mdot), rIntake);
    };

    // lift engines
    for (let i = 0; i < 4; i++) {
      const e = this.lift[i], L = LIFT[i];
      e.update(dt, cmd.lift[i], sigma);
      // vane servos: 30 ms lag, 3 rad/s rate limit
      for (let k = 0; k < 2; k++) {
        const target = clamp(cmd.vanes[i][k], -VANE_MAX, VANE_MAX);
        const d = clamp((target - this.vanes[i][k]) / 0.03, -3, 3);
        this.vanes[i][k] += d * dt;
      }
      // ground effect: suck-down + hot gas re-ingestion
      const exitW = this.toWorld(L.exit);
      const hn = Math.max(0, exitW[1] - this.groundY);
      const gfx = 1 - 0.06 * Math.exp(-hn / 0.8) - 0.03 * Math.exp(-hn / 2.2);
      this.groundFx[i] = gfx;
      e.T = e.Tcore * gfx;
      const [a, b] = this.vanes[i];
      const vaneLoss = 1 - 0.35 * (a * a + b * b);        // turning losses at the vanes
      const dir = norm([Math.tan(a), 1, Math.tan(b)]);
      applyAt(scale(dir, e.T * vaneLoss), this.arm(L.exit));
      ramDrag(e, this.arm(L.intake));
      const spinAxis = [0, L.spin, 0];
      H = madd(H, spinAxis, e.rotorI * e.omega);
      Tq = madd(Tq, spinAxis, -e.rotorI * e.dNdt * e.omegaMax); // reaction on the casing
      fuelFlow += e.fuelFlow;
    }
    // cruise engine
    {
      const e = this.main;
      e.update(dt, cmd.main, sigma);
      e.T = e.Tcore;
      applyAt([e.T, 0, 0], this.arm(MAIN.exit));
      ramDrag(e, this.arm(MAIN.intake));
      H = madd(H, [1, 0, 0], e.rotorI * e.omega);
      Tq = madd(Tq, [1, 0, 0], -e.rotorI * e.dNdt * e.omegaMax);
      fuelFlow += e.fuelFlow;
    }

    // body aerodynamics
    const V = len(vAirB);
    for (let k = 0; k < 3; k++) {
      const f = [0, 0, 0];
      f[k] = -0.5 * rho * AERO.cda[k] * V * vAirB[k];
      applyAt(f, AERO.cp[k]);
    }
    for (let k = 0; k < 3; k++) Tq[k] -= AERO.rotDamp[k] * w[k] * sigma * (1 + V / 25);

    // world-frame force
    let Fw = qrot(q, F);
    Fw[1] -= mp.m * G;

    // landing pads
    this.contactN = 0;
    const nearGround = this.p[1] - this.groundY < 3;
    for (let i = 0; nearGround && i < PADS.length; i++) {
      const r = this.arm(PADS[i]);
      const pw = add(this.p, qrot(q, r));
      const pen = heightAt(pw[0], pw[2]) - pw[1];
      this.padForces[i] = 0;
      if (pen <= 0) continue;
      const n = normalAt(pw[0], pw[2]);
      const vp = add(this.v, qrot(q, cross(w, r)));
      const vn = dot(vp, n);
      const Fn = Math.max(0, 52000 * pen - 2900 * vn);
      const vt = madd(vp, n, -vn);
      const vtl = len(vt);
      const mu = 0.65 * Math.min(1, vtl / 0.03);
      const fW = add(scale(n, Fn), vtl > 1e-6 ? scale(vt, -mu * Fn / vtl) : [0, 0, 0]);
      Fw = add(Fw, fW);
      Tq = add(Tq, cross(r, qinvrot(q, fW)));
      this.padForces[i] = Fn;
      this.contactN += Fn;
    }

    // integrate
    this.acc = scale(Fw, 1 / mp.m);
    this.v = madd(this.v, this.acc, dt);
    this.p = madd(this.p, this.v, dt);
    const Iw = m3v(mp.I, w);
    const wdot = m3v(Iinv, sub(Tq, cross(w, add(Iw, H))));
    this.w = madd(w, wdot, dt);
    const dq = qmul(q, [0, this.w[0], this.w[1], this.w[2]]);
    this.q = qnorm([q[0] + 0.5 * dq[0] * dt, q[1] + 0.5 * dq[1] * dt, q[2] + 0.5 * dq[2] * dt, q[3] + 0.5 * dq[3] * dt]);
    this.fuel = Math.max(0, this.fuel - fuelFlow * dt);
    this.fuelFlow = fuelFlow;
    this.sigma = sigma;
    this.airspeed = V;
    this.t += dt;
  }
}
