// Flight computer (fly-by-wire). The rider sits on the bike; the machine is flown by this
// controller, running at 200 Hz on noisy sensor data:
//   position/velocity tracking (PID + reference acceleration feed-forward)
//   -> required total force -> split between cruise engine and lift system
//   -> desired attitude (lift vector direction + heading)
//   -> attitude/rate loops -> required torque
//   -> control allocation in two speeds: the turbojets spool too slowly (~0.2 s) for fast
//      attitude control, so differential thrust takes the low-passed torque demand and the
//      nozzle vanes (30 ms servos) supply whatever the engines' *actual* thrusts are
//      missing at this instant. Yaw comes from the vanes alone.

import {
  add, sub, scale, madd, dot, cross, len, norm, qmul, qconj, qrot, qinvrot, qToRotVec,
  qFromBasis, m3v, clamp, solve, rng,
} from './vmath.js';
import { airDensity } from './terrain.js';
import {
  G, LIFT, MAIN, LIFT_TMAX, MAIN_TMAX, VANE_MAX, EXHAUST_V, IDLE_N, THRUST_EXP, AERO,
} from './vehicle.js';

export const GAINS = {
  KP: [1.25, 2.6, 1.25], KD: [2.1, 3.0, 2.1], KI: [0.12, 0.45, 0.12],
  KA: [3.0, 2.2, 3.0],     // attitude P (rad/s per rad) about body x (roll), y (yaw), z (pitch)
  KR: [7, 5, 7],           // rate loop (1/s)
  KIA: [40, 25, 40],       // attitude integral (N m per rad s)
  TAU_SLOW: 0.3,           // s, torque share handed to differential thrust
};
const { KP, KD, KI, KA, KR, KIA } = GAINS;
const MAX_TILT = 38 * Math.PI / 180;

export class FlightComputer {
  constructor(seed = 99) {
    this.iPos = [0, 0, 0];
    this.iAtt = [0, 0, 0];
    this.rand = rng(seed);
    this.cmd = { lift: [0, 0, 0, 0], vanes: [[0, 0], [0, 0], [0, 0], [0, 0]], main: 0 };
    this.qDes = [1, 0, 0, 0];
    this.tauSlow = [0, 0, 0];
    this.telemetry = {};
  }

  gauss() { // Box-Muller
    const u = Math.max(1e-12, this.rand()), v = this.rand();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }
  noisy(v, s) { return [v[0] + s * this.gauss(), v[1] + s * this.gauss(), v[2] + s * this.gauss()]; }

  // mode: 'ground' (collective = arg), 'flight', 'landed' (collective = arg), 'off'
  update(dt, bike, ref, mode, collective = 0) {
    const mp = bike.mp, m = mp.m;
    const p = this.noisy(bike.p, 0.012);
    const v = this.noisy(bike.v, 0.015);
    const w = this.noisy(bike.w, 0.003);
    const q = bike.q;
    const sigma = airDensity(p[1]) / 1.225;
    const tmaxL = LIFT_TMAX * sigma;

    if (mode === 'off') {
      this.cmd.lift = [0, 0, 0, 0]; this.cmd.main = 0;
      this.cmd.vanes = this.cmd.vanes.map(() => [0, 0]);
      return this.cmd;
    }

    let Fw, yaw = ref.yaw, yawRate = ref.yawRate;
    const h = [Math.cos(yaw), 0, -Math.sin(yaw)];
    let TmDes = 0;
    const vB = qinvrot(q, v);
    const V = len(vB);
    const liftSum = bike.lift.reduce((a, e) => a + e.Tcore, 0);

    if (mode === 'flight') {
      const ep = sub(ref.p, p), ev = sub(ref.v, v);
      for (let k = 0; k < 3; k++) this.iPos[k] = clamp(this.iPos[k] + ep[k] * dt, -4, 4);
      let a = [0, 0, 0];
      for (let k = 0; k < 3; k++) a[k] = ref.a[k] + KP[k] * ep[k] + KD[k] * ev[k] + KI[k] * this.iPos[k];
      const ah = Math.hypot(a[0], a[2]);
      if (ah > 6.5) { a[0] *= 6.5 / ah; a[2] *= 6.5 / ah; }
      a[1] = clamp(a[1], -4.5, 5.5);
      // drag model known to the flight computer (it cannot see the wind)
      const rho = airDensity(p[1]);
      const DB = [0, 0, 0];
      for (let k = 0; k < 3; k++) DB[k] = -0.5 * rho * AERO.cda[k] * V * vB[k];
      const ram = scale(vB, -(liftSum + bike.main.Tcore) / EXHAUST_V);
      const Dw = qrot(q, add(DB, ram));
      Fw = sub(scale(add(a, [0, G, 0]), m), Dw);
      TmDes = clamp(dot(Fw, h), 0, MAIN_TMAX * sigma * 0.97);
    } else {
      this.iPos = [0, 0, 0];
      Fw = [0, collective, 0];
      TmDes = 0;
    }

    // lift vector = required force minus what the cruise engine actually delivers now
    const fwdW = qrot(q, [1, 0, 0]);
    const Rw = madd(Fw, fwdW, -bike.main.T);
    let up = norm(Rw);
    const tilt = Math.acos(clamp(up[1], -1, 1));
    if (tilt > MAX_TILT) {
      const hz = norm([up[0], 0, up[2]]);
      up = [hz[0] * Math.sin(MAX_TILT), Math.cos(MAX_TILT), hz[2] * Math.sin(MAX_TILT)];
    }
    if (mode !== 'flight') { up = [0, 1, 0]; yawRate = 0; }
    // desired attitude: body y along the lift vector, body x as close to the heading as possible
    const hd = [Math.cos(yaw), 0, -Math.sin(yaw)];
    const bx = norm(madd(hd, up, -dot(hd, up)));
    const bz = cross(bx, up);
    const qDes = (this.qDes = qFromBasis(bx, up, bz));

    // attitude -> rate -> torque
    const e = qToRotVec(qmul(qconj(q), qDes));
    const wff = qinvrot(q, [0, yawRate, 0]);
    let wDes = [0, 0, 0];
    for (let k = 0; k < 3; k++) wDes[k] = KA[k] * e[k] + wff[k];
    const wl = len(wDes);
    if (wl > 1.6) wDes = scale(wDes, 1.6 / wl);
    const onGround = bike.contactN > 200;
    const kiOn = mode === 'flight' && !onGround ? 1 : 0;
    for (let k = 0; k < 3; k++) this.iAtt[k] = clamp(this.iAtt[k] + e[k] * dt * kiOn, -2, 2);
    const alpha = [0, 0, 0];
    for (let k = 0; k < 3; k++) alpha[k] = KR[k] * (wDes[k] - w[k]);
    const Hrot = add(
      bike.lift.reduce((acc, eng, i) => madd(acc, [0, LIFT[i].spin, 0], eng.rotorI * eng.omega), [0, 0, 0]),
      [bike.main.rotorI * bike.main.omega, 0, 0]);
    let tau = add(m3v(mp.I, alpha), cross(w, add(m3v(mp.I, w), Hrot)));
    for (let k = 0; k < 3; k++) tau[k] += KIA[k] * this.iAtt[k];
    if (mode !== 'flight') tau = scale(tau, onGround ? 0.25 : 1);

    // subtract known moments: cruise engine thrust line and body drag centres of pressure
    const cg = mp.cg;
    const arm = (d) => [d[0] - cg[0], d[1] - cg[1], d[2] - cg[2]];
    tau = sub(tau, cross(arm(MAIN.exit), [bike.main.T, 0, 0]));
    if (mode === 'flight') {
      const rho = airDensity(p[1]);
      for (let k = 0; k < 3; k++) {
        const f = [0, 0, 0];
        f[k] = -0.5 * rho * AERO.cda[k] * V * vB[k];
        tau = sub(tau, cross(AERO.cp[k], f));
      }
      // intake momentum drag acts high up at the intakes
      for (let i = 0; i < 4; i++) {
        const r = arm(LIFT[i].intake);
        tau = sub(tau, cross(r, scale(vB, -bike.lift[i].Tcore / EXHAUST_V)));
      }
    }

    const FLb = qinvrot(q, Rw);
    const FyDes = mode === 'flight' ? Math.max(FLb[1], 0.3 * m * G) : collective;
    // slow torque share -> differential thrust; the vanes make up the difference right now
    const kf = Math.min(1, dt / GAINS.TAU_SLOW);
    for (let k = 0; k < 3; k++) this.tauSlow[k] += (tau[k] - this.tauSlow[k]) * kf;
    const T = this.allocThrust(FyDes, this.tauSlow, cg, tmaxL);
    const Tact = bike.lift.map((e) => e.Tcore);          // from measured rpm
    const vanes = this.allocVanes(tau, Tact, cg);
    for (let i = 0; i < 4; i++) {
      const [a, b] = vanes[i];
      this.cmd.lift[i] = T[i] / (1 - 0.35 * (a * a + b * b));
      this.cmd.vanes[i] = [a, b];
    }
    this.cmd.main = TmDes;
    this.telemetry = { Fw, TmDes, tau, tilt, e };
    return this.cmd;
  }

  // Four thrusts for collective force, roll and pitch torque (min-norm about equal split),
  // then shift the collective so the busiest engine fits its limit (torque is kept).
  allocThrust(Fy, tau, cg, tmax) {
    const r = LIFT.map((L) => [L.exit[0] - cg[0], L.exit[1] - cg[1], L.exit[2] - cg[2]]);
    // rows: sum T = Fy ; tau_x = -sum r_z T ; tau_z = sum r_x T
    const B = [[1, 1, 1, 1], r.map((v) => -v[2]), r.map((v) => v[0])];
    const T0 = [Fy / 4, Fy / 4, Fy / 4, Fy / 4];
    const res = [Fy - T0.reduce((a, v) => a + v, 0), tau[0] - dotRow(B[1], T0), tau[2] - dotRow(B[2], T0)];
    const BBt = B.map((ri) => B.map((rj) => dotRow(ri, rj)));
    const lam = solve(BBt, res);
    const T = T0.map((t0, i) => t0 + B[0][i] * lam[0] + B[1][i] * lam[1] + B[2][i] * lam[2]);
    const tmin = tmax * Math.pow(IDLE_N, THRUST_EXP) * 1.05;
    const over = Math.max(...T) - tmax * 0.985;
    if (over > 0) for (let i = 0; i < 4; i++) T[i] -= over;
    const under = tmin - Math.min(...T);
    if (under > 0) for (let i = 0; i < 4; i++) T[i] += under;
    return T.map((t) => clamp(t, tmin, tmax));
  }

  // Vane side forces (fore-aft fx, lateral fz per nozzle) for the torque the actual thrusts
  // do not deliver yet. Min-norm solution of 3 torque equations in 8 unknowns.
  allocVanes(tau, Tact, cg) {
    const r = LIFT.map((L) => [L.exit[0] - cg[0], L.exit[1] - cg[1], L.exit[2] - cg[2]]);
    let tT = [0, 0, 0];
    for (let i = 0; i < 4; i++) tT = add(tT, cross(r[i], [0, Tact[i], 0]));
    const res = sub(tau, tT);
    // unknown order: fx0, fz0, fx1, fz1, ...
    const D = [new Array(8).fill(0), new Array(8).fill(0), new Array(8).fill(0)];
    for (let i = 0; i < 4; i++) {
      const [x, y, z] = r[i];
      D[0][2 * i + 1] = y;                         // tau_x = y fz
      D[1][2 * i] = z; D[1][2 * i + 1] = -x;       // tau_y = z fx - x fz
      D[2][2 * i] = -y;                            // tau_z = -y fx
    }
    const DDt = D.map((ri) => D.map((rj) => dotRow(ri, rj)));
    const lam = solve(DDt, res);
    const out = [];
    for (let i = 0; i < 4; i++) {
      const fx = D[0][2 * i] * lam[0] + D[1][2 * i] * lam[1] + D[2][2 * i] * lam[2];
      const fz = D[0][2 * i + 1] * lam[0] + D[1][2 * i + 1] * lam[1] + D[2][2 * i + 1] * lam[2];
      const Ti = Math.max(Tact[i], 20);
      out.push([clamp(Math.atan2(fx, Ti), -VANE_MAX, VANE_MAX), clamp(Math.atan2(fz, Ti), -VANE_MAX, VANE_MAX)]);
    }
    return out;
  }
}

const dotRow = (a, b) => a.reduce((s, v, i) => s + v * b[i], 0);
