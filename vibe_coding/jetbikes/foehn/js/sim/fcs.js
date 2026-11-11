// Flight control system of the F-4: a "unified" control law in the spirit of modern STOVL
// fighters. The rider never commands an actuator. The stick asks for a velocity vector
// (turn rate, climb rate, forward speed, side-step) and the computer works out
//   1. the acceleration that takes the bike there,
//   2. the attitude (bank for the turn, angle of attack for wing lift) and the thrust vector
//      that produce this acceleration, given what the wing can carry at this airspeed,
//   3. how much gas the one engine must make and how the diverter shares it between the rear
//      nozzle and the lift pipes (height first: the rear nozzle gets what the lift does not need),
//   4. a control allocation: 13 effectors (the shares of the 4 lift pipes, their 4 vanes, two
//      axes of rear-nozzle vectoring, ailerons, elevator, rudder) solved every tick by bounded
//      weighted least squares for the wanted moment. The lift valves can only re-distribute the
//      gas there is, so their sum is a constraint of the problem.
// Every force still passes through the simulated engine, valves, servos and aerodynamics.
import { clamp, lerp, sstep, dot, cross, len, qmul, qconj, qnorm, qinvrot, qrot, qFromYPR, qFromBasis, yprOf, m3v, wrapPi, solve } from './vmath.js';
const dot4 = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
import { G, ENG, TILT_MAX, VANE_MAX, VEC_MAX, SHARE_MIN, SHARE_MAX, LIM, WING, aeroWrench, propWrench, ramWrench, liftFrac, mkW, newCmd, thrustOfN } from './bike.js';
import { surfaceAt } from './terrain.js';

const NU = 13, NR = 5;
export const FCS_HZ = 120;
export const V_MAX = 95, V_MIN = -8;

export class FCS {
  constructor(bike) {
    this.bike = bike;
    this.cmd = newCmd();
    this._w0 = mkW(); this._w1 = mkW(); this._a0 = mkW(); this._a1 = mkW(); this._r0 = mkW();
    this._n = { tilt: 0, L: [0, 0, 0, 0], va: [0, 0, 0, 0], R: 0, rp: 0, ry: 0 };
    this.J = new Float64Array(NR * NU);
    this.reset();
  }
  reset() {
    this.turn = 0; this.pitch = 0; this.strafe = 0;
    this.iV = 0; this.iZ = 0; this.yawRef = null; this.tiltCmd = this.bike.tilt;
    this.idle = true; this.obstacle = this.obstacle || null;
    this.tele = { phiDes: 0, thetaDes: 0, aCmd: [0, 0, 0], wb: 0, sat: 0, vzDes: 0, floor: false };
    const c = this.cmd; c.G = 0; c.s.fill(1); c.va.fill(0); c.rp = c.ry = 0; c.da = c.de = c.dr = c.brake = c.ab = 0; c.tilt = this.bike.tilt;
  }

  // inp: { turn (+right), pitch (+climb), strafe (+right), vt (m/s forward), boost, brake }
  update(dt, inp) {
    const b = this.bike, mp = b.mp, m = mp.m, cg = mp.cg, cmd = this.cmd, W = m * G;
    const k1 = 1 - Math.exp(-dt / 0.10);
    this.turn += (clamp(inp.turn, -1, 1) - this.turn) * k1;
    this.pitch += (clamp(inp.pitch, -1, 1) - this.pitch) * k1;
    this.strafe += (clamp(inp.strafe || 0, -1, 1) - this.strafe) * k1;

    const [yaw, pitch, roll] = yprOf(b.q);
    const fh = [Math.cos(yaw), 0, -Math.sin(yaw)], rh = [Math.sin(yaw), 0, Math.cos(yaw)];
    const v = b.v, Vf = dot(v, fh), Vr = dot(v, rh), Vu = v[1];
    const Vh = Math.hypot(v[0], v[2]), V = len(v), Va = b.airspeed;
    const qbar = 0.5 * b.rho * Va * Va;
    const wb = sstep(24, 52, b.vAirB[0]);                       // 0 jet-borne .. 1 wing-borne
    const onGround = b.contactN > 0.25 * W;

    // ----- ground idle -----
    const wantsUp = inp.pitch > 0.05 || Math.abs(inp.vt) > 0.5 || Math.abs(inp.strafe || 0) > 0.05 || inp.boost;
    if ((onGround || (b.contactN > 0.06 * W && this.pitch < 0.02)) && !wantsUp && V < 4) {
      this.idle = true;
      cmd.G = 0; cmd.s.fill(1); cmd.va.fill(0); cmd.rp = cmd.ry = 0; cmd.da = cmd.de = cmd.dr = 0; cmd.ab = 0; cmd.brake = 0; cmd.tilt = Math.PI / 2;
      this.iV = this.iZ = 0; this.yawRef = yaw; this.tiltCmd = Math.PI / 2;
      return cmd;
    }
    this.idle = false;
    if (this.yawRef === null) this.yawRef = yaw;

    // ----- 1. wanted acceleration -----
    const boost = !!inp.boost && !b.abLock && !inp.brake;
    const eV = inp.vt - Vf;
    this.iV = clamp(this.iV + 0.25 * eV * dt, -1.5, 1.5);
    let aF = boost ? 16 : clamp(0.85 * eV + this.iV, -12, 10);
    const phiLim = lerp(0.62, 1.24, wb);
    const rateMax = Math.min(1.0, G * Math.tan(phiLim) / Math.max(Math.abs(Vf), 6));
    const yawRateCmd = -this.turn * rateMax;
    const low = 1 - sstep(16, 30, Math.abs(Vf));
    let aR = -Vf * yawRateCmd + low * clamp(1.3 * (this.strafe * 7 - Vr), -5, 5);

    let vzDes = this.pitch * Math.max(8, 0.62 * V);
    // radar-altimeter floor: looks 0.6 s ahead; released when the rider deliberately descends slowly (landing)
    const landing = this.pitch < -0.2 && Va < 16;
    const ax = b.p[0] + v[0] * 0.6, az = b.p[2] + v[2] * 0.6;
    // the radar also sees tree tops and roofs (the game supplies them through this.obstacle)
    const obst = this.obstacle;
    const top = (x, z) => (obst ? Math.max(surfaceAt(x, z), obst(x, z) + 1.5) : surfaceAt(x, z));
    const ax2 = b.p[0] + v[0] * 1.3, az2 = b.p[2] + v[2] * 1.3;
    const hBike = b.p[1] - cg[1];
    const aglAhead = Math.min(hBike - top(ax, az), hBike - top(ax2, az2) + Math.max(0, 0.25 * Vh));
    const hHere = obst ? Math.min(b.agl, hBike - top(b.p[0], b.p[2])) : b.agl;
    const hmin = Math.min(hHere, aglAhead), floorH = 1.6 + 0.07 * Va;
    this.tele.floor = false;
    if (!landing && !inp.noFloor) {
      // never sink faster than can still be arrested above the floor; below it, climb
      const f = hmin < floorH ? 2.2 * (floorH - hmin) : -1.5 * (hmin - floorH) - 0.5;
      if (f > vzDes) { vzDes = f; this.tele.floor = true; }
    }
    if (onGround && this.pitch <= 0.05) vzDes = Math.min(vzDes, 0);
    // automatic landing flare: the lower, the slower
    if (!inp.noFloor && Va < 25) vzDes = Math.max(vzDes, -(0.6 + 0.48 * Math.max(0, landing ? b.agl : hmin)));
    const eZ = vzDes - Vu;
    this.iZ = clamp(this.iZ + 0.35 * eZ * dt, -2, 2);
    // jet-borne, thrust is also the attitude control power: never let it fall too low
    let aU = clamp(1.9 * eZ + this.iZ, -lerp(0.55, 1.0, wb) * G, 1.9 * G);

    // what can the airframe actually deliver? Protect height first, then the turn, then speed.
    const Tavail = thrustOfN(1, b.sigma) * ENG.ductEta;
    const liftCap = 0.9 * Tavail + qbar * WING.S * WING.clUse;
    const need = m * Math.hypot(aR, aU + G);
    if (need > liftCap) {
      const k = Math.sqrt(Math.max(0, liftCap * liftCap / (m * m) - (aU + G) * (aU + G)));
      if (Math.abs(aR) > k) aR = Math.sign(aR) * Math.max(k, 0.25 * Math.abs(aR));
    }
    // bank limit (also keeps roll reversals on the short way round, through wings-level)
    const aRmax = Math.tan(phiLim) * Math.max(aU + G, 0.35 * G);
    aR = clamp(aR, -aRmax, aRmax);
    // speed changes act along the flight path once wing-borne (along the heading when hovering)
    let th = fh;
    if (wb > 0 && V > 5) {
      th = [lerp(fh[0], v[0] / V, wb), lerp(0, v[1] / V, wb), lerp(fh[2], v[2] / V, wb)];
      const l = len(th) || 1; th = [th[0] / l, th[1] / l, th[2] / l];
    }
    const f = [aF * th[0] + aR * rh[0], aF * th[1] + aU + G, aF * th[2] + aR * rh[2]];   // specific force, world
    this.tele.aCmd = [aF, aU, aR]; this.tele.vzDes = vzDes; this.tele.wb = wb;

    // ----- 2. attitude -----
    const phiDes = clamp(Math.atan2(aR, Math.max(aU + G, 0.35 * G)), -phiLim, phiLim);
    // jet-borne: lean back to brake (flare), a touch forward to accelerate; thrust tilt does the rest
    // (and stay level close to the ground so that all four skids arrive together)
    const lvl = inp.noFloor ? 1 : clamp(0.12 + 0.11 * b.agl, 0, 1);
    const thLow = clamp(-0.5 * aF / G, -0.20, 0.42) * (1 - 0.6 * sstep(8, 30, Vf) * (aF > 0 ? 1 : 0)) * lvl;
    // wing-borne: angle of attack for the lift the manoeuvre needs
    // wing-borne: build the attitude straight from the flow. Nose = air velocity pitched up by
    // the angle of attack the manoeuvre needs, "up" = direction of the lift that is wanted.
    // This holds angle of attack and zero sideslip at any bank angle.
    const va = qrot(b.q, b.vAirB);
    const vhat = Va > 2 ? [va[0] / Va, va[1] / Va, va[2] / Va] : fh;
    const fv = dot(f, vhat);
    // direction of "up" uses a floored vertical part, so a push-over cannot roll the bike on its back
    const fa = [aR * rh[0], Math.max(aU + G, 0.35 * G), aR * rh[2]], fav = dot(fa, vhat);
    let fp = [fa[0] - fav * vhat[0], fa[1] - fav * vhat[1], fa[2] - fav * vhat[2]];
    let fpl = len(fp);
    if (fpl < 0.5) { fp = [-vhat[0] * vhat[1], 1 - vhat[1] * vhat[1], -vhat[2] * vhat[1]]; fpl = len(fp) || 1; }
    const lhat = [fp[0] / fpl, fp[1] / fpl, fp[2] / fpl];
    const Lreq = m * len([f[0] - fv * vhat[0], f[1] - fv * vhat[1], f[2] - fv * vhat[2]]);
    // the jets never go below a floor while the air is slow (their thrust is the control power
    // of the vanes); the wing is asked for the rest
    const kq0 = qbar / (qbar + 900);
    const Tmin = 0.42 * W * (1 - kq0);
    // braking: hand the weight from the wing to the jets, whose nozzles then point up and forward
    const kb = clamp((-aF - 2) / 5, 0, 1) * (1 - sstep(70, 90, Va));
    const TupFloor = Math.max(Tmin * Math.max(0, Math.sin(this.tiltCmd)), kb * 0.95 * W);
    const qSa = qbar * WING.S * WING.cla, Tfwd = b.R;
    const alphaDes = clamp((Lreq - TupFloor + qSa * WING.a0) / Math.max(qSa + Tfwd, 1), -0.10, WING.alphaMax);
    const ca = Math.cos(alphaDes), sa = Math.sin(alphaDes);
    const xw = [vhat[0] * ca + lhat[0] * sa, vhat[1] * ca + lhat[1] * sa, vhat[2] * ca + lhat[2] * sa];
    const yw = [lhat[0] * ca - vhat[0] * sa, lhat[1] * ca - vhat[1] * sa, lhat[2] * ca - vhat[2] * sa];
    const qWing = qFromBasis(xw, yw, cross(xw, yw));
    // jet-borne: classic heading / pitch / bank
    const gamma = Math.atan2(Vu, Math.max(Vh, 6));
    const wq = sstep(14, 32, Va), slack = (1 - wq) * 0.7;
    const thetaH = clamp(thLow, Math.max(-0.9, gamma - 0.11 - slack), Math.min(0.9, gamma + WING.alphaMax + slack));
    const beta = b.vAirB[2] / Math.max(Va, 12);
    const yawRateH = yawRateCmd - 2.2 * beta * sstep(8, 26, Va);
    this.yawRef = yaw + clamp(wrapPi(this.yawRef + yawRateH * dt - yaw), -0.3, 0.3);
    const qHover = qFromYPR(this.yawRef, thetaH, phiDes);
    // blend the two laws through the transition
    let qDes = qWing;
    if (wb < 0.999) {
      const s = dot4(qHover, qWing) < 0 ? -1 : 1;
      qDes = qnorm([lerp(qHover[0], s * qWing[0], wb), lerp(qHover[1], s * qWing[1], wb), lerp(qHover[2], s * qWing[2], wb), lerp(qHover[3], s * qWing[3], wb)]);
    }
    const yawRate = lerp(yawRateH, -aR / Math.max(Vh, 10), wb);
    const thetaDes = Math.asin(clamp(2 * (qDes[0] * qDes[3] + qDes[1] * qDes[2]), -1, 1));   // for telemetry
    let qe = qmul(qconj(b.q), qDes);
    if (qe[0] < 0) qe = [-qe[0], -qe[1], -qe[2], -qe[3]];
    this.tele.attErr = 2 * Math.acos(Math.min(1, qe[0]));
    const wff = qinvrot(b.q, [0, yawRate, 0]);
    // rate and angular-acceleration limits keep the demand inside what the effectors can give
    const wDes = [clamp(5.0 * 2 * qe[1], -2.0, 2.0) + wff[0], clamp(3.2 * 2 * qe[2], -1.2, 1.2) + wff[1], clamp(4.6 * 2 * qe[3], -1.3, 1.3) + wff[2]];
    const aL = [lerp(4.2, 10, kq0), lerp(1.2, 2.6, kq0), lerp(2.6, 5, kq0)];
    const wdot = [clamp(11 * (wDes[0] - b.w[0]), -aL[0], aL[0]), clamp(6.5 * (wDes[1] - b.w[1]), -aL[1], aL[1]), clamp(10 * (wDes[2] - b.w[2]), -aL[2], aL[2])];
    const Iw = m3v(mp.I, b.w), gyro = cross(b.w, Iw);
    const tauDes = m3v(mp.I, wdot); tauDes[0] += gyro[0]; tauDes[1] += gyro[1]; tauDes[2] += gyro[2];
    // the spinning spool: pitching the bike makes it want to yaw and the other way round
    const Hx = ENG.rotorI * ENG.omegaMax * b.eng.N; tauDes[1] += b.w[2] * Hx; tauDes[2] -= b.w[1] * Hx;
    this.tele.phiDes = phiDes; this.tele.thetaDes = thetaDes;

    // ----- 3. thrust vector: how much gas, and where it goes -----
    const surf = { da: b.da, de: b.de, dr: b.dr, brake: b.brake };
    const A0 = aeroWrench(b.vAirB, b.w, b.rho, surf, cg, this._a0);
    const R0 = ramWrench(b.vAirB, b.w, b.eng.Tdry, cg, this._r0);
    const Fb = qinvrot(b.q, [m * f[0], m * f[1], m * f[2]]);
    let Fx = Fb[0] - A0.F[0] - R0.F[0];
    // jet-borne: thrust fills in whatever lift is missing right now. Wing-borne: thrust only
    // carries what the wing cannot at the commanded angle of attack (so it does not chase
    // the short lag of the pitch loop).
    const FyNow = Fb[1] - A0.F[1] - R0.F[1];
    const FyWing = Math.max(0, Fb[1] - qSa * (alphaDes - WING.a0) * 1.04);
    // while the air is slow the lift pipes are the roll and pitch control: they never run dry
    const Fy = Math.max(lerp(FyNow, Math.min(FyWing, Math.max(FyNow, 0) + 0.5 * W), wb), Tmin * (1 - wb));
    // one engine: height comes first, the rear nozzle gets the gas the lift pipes do not need
    const ab = 1 + ENG.abGain * b.eng.ab, eta = ENG.ductEta, Gmax = thrustOfN(1, b.sigma);
    const cant = Math.cos(0.126);
    if (Fx > 0) Fx = Math.min(Fx, Math.max(0.12 * Gmax, Gmax - Math.max(Fy, 0) / eta) * ab * cant);
    const mag = Math.hypot(Fx, Fy);
    const tiltNew = clamp(Fy <= 0 ? (Fx > 0 ? 0 : TILT_MAX) : Math.atan2(Fy, Fx), 0, TILT_MAX);
    this.tiltCmd = lerp(this.tiltCmd, tiltNew, sstep(120, 700, mag));
    cmd.tilt = this.tiltCmd;
    cmd.ab = boost ? 1 : 0;
    cmd.brake = inp.brake || (eV < -10 && Va > 20) ? 1 : 0;
    // gas for the force along the direction the diverter stands in right now
    const tilt = b.tilt, ct = Math.cos(tilt), st = Math.sin(tilt), fl = liftFrac(tilt);
    const per = tilt < Math.PI / 2 ? (1 - fl) * ab * cant * ct + fl * eta * st : eta;   // force along (ct, st) per N of gas
    cmd.G = clamp((Fx * ct + Fy * st) / Math.max(per, 0.3), Tmin, Gmax);

    // ----- 4. allocation of the moment -----
    const u0 = this._u0 || (this._u0 = new Float64Array(NU));
    const lo = this._lo || (this._lo = new Float64Array(NU)), hi = this._hi || (this._hi = new Float64Array(NU));
    const n = this._n, Ltot = b.lift;
    n.tilt = tilt; n.R = b.R; n.rp = b.rp; n.ry = b.ry;
    for (let i = 0; i < 4; i++) {
      n.L[i] = b.L[i]; n.va[i] = b.va[i];
      u0[i] = b.L[i]; lo[i] = Math.min(SHARE_MIN * 1.05 * Ltot * b.gfx[i] - b.L[i], 0); hi[i] = Math.max(SHARE_MAX * 0.95 * Ltot * b.gfx[i] - b.L[i], 0);
      u0[4 + i] = b.va[i]; lo[4 + i] = -VANE_MAX * 0.95 - b.va[i]; hi[4 + i] = VANE_MAX * 0.95 - b.va[i];
    }
    u0[8] = b.rp; lo[8] = -VEC_MAX * 0.95 - b.rp; hi[8] = VEC_MAX * 0.95 - b.rp;
    u0[9] = b.ry; lo[9] = -VEC_MAX * 0.95 - b.ry; hi[9] = VEC_MAX * 0.95 - b.ry;
    // surfaces only work while their own flow is attached: keep each inside its linear range
    const vx = Math.max(b.vAirB[0], 5), aw = Math.atan2(-b.vAirB[1], vx);
    const daMax = clamp((WING.stall - 0.035 - Math.abs(aw - WING.a0)) / 0.45, 0.07, LIM.da);
    const at = Math.atan2(-(b.vAirB[1] - b.w[2] * 1.07), vx);      // local flow angle at the tailplane
    const deLo = clamp(at - 0.27, -LIM.de, LIM.de - 0.05), deHi = clamp(at + 0.27, -LIM.de + 0.05, LIM.de);
    u0[10] = b.da; lo[10] = Math.min(-daMax - b.da, 0); hi[10] = Math.max(daMax - b.da, 0);
    u0[11] = b.de; lo[11] = Math.min(deLo - b.de, 0); hi[11] = Math.max(deHi - b.de, 0);
    u0[12] = b.dr; lo[12] = -LIM.dr - b.dr; hi[12] = LIM.dr - b.dr;

    // Jacobian by finite differences through the same force model the physics uses.
    // rows: [sum of lift thrust, -, Mx, My, Mz]
    const J = this.J;
    const row = (w, k, s, col, sum) => { J[col] = sum; J[NU + col] = 0; J[2 * NU + col] = (w.Tq[0] - k.Tq[0]) / s; J[3 * NU + col] = (w.Tq[1] - k.Tq[1]) / s; J[4 * NU + col] = (w.Tq[2] - k.Tq[2]) / s; };
    const P0 = propWrench(n, cg, this._w0);
    for (let i = 0; i < 4; i++) {
      n.L[i] += 20; row(propWrench(n, cg, this._w1), P0, 20, i, 1); n.L[i] -= 20;
      n.va[i] += 0.01; row(propWrench(n, cg, this._w1), P0, 0.01, 4 + i, 0); n.va[i] -= 0.01;
    }
    n.rp += 0.01; row(propWrench(n, cg, this._w1), P0, 0.01, 8, 0); n.rp -= 0.01;
    n.ry += 0.01; row(propWrench(n, cg, this._w1), P0, 0.01, 9, 0); n.ry -= 0.01;
    const keys = ['da', 'de', 'dr'];
    for (let k = 0; k < 3; k++) {
      surf[keys[k]] += 0.02;
      row(aeroWrench(b.vAirB, b.w, b.rho, surf, cg, this._a1), A0, 0.02, 10 + k, 0);
      surf[keys[k]] -= 0.02;
    }
    // preferred positions: equal shares, vanes, nozzle and surfaces centred
    const pref = this._pref || (this._pref = new Float64Array(NU));
    for (let i = 0; i < 4; i++) { pref[i] = Ltot * 0.25 - u0[i]; pref[4 + i] = -u0[4 + i]; }
    for (let i = 8; i < NU; i++) pref[i] = -u0[i];
    const tq = [tauDes[0] - A0.Tq[0] - R0.Tq[0], tauDes[1] - A0.Tq[1] - R0.Tq[1], tauDes[2] - A0.Tq[2] - R0.Tq[2]];
    const e = [0, 0, tq[0] - P0.Tq[0], tq[1] - P0.Tq[1], tq[2] - P0.Tq[2]];
    const du = wls(J, e, WV, WU, pref, lo, hi, null);
    for (let i = 0; i < 4; i++) { cmd.s[i] = Math.max(1e-3, (u0[i] + du[i]) / b.gfx[i]); cmd.va[i] = u0[4 + i] + du[4 + i]; }
    cmd.rp = u0[8] + du[8]; cmd.ry = u0[9] + du[9];
    cmd.da = u0[10] + du[10]; cmd.de = u0[11] + du[11]; cmd.dr = u0[12] + du[12];
    this.tele.sat = wls.sat; this.tele.tau = tauDes; this.tele.tq = tq; this.tele.F = [Fx, Fy];
    return cmd;
  }
}

// row weights: [sum of lift (N), -, Mx, My, Mz (N m)]
const WV = [0.5, 0, 6 / 200, 2.5 / 200, 6 / 200];
// effector weights (cost of leaving the preferred position): lift thrust per N, angles per rad
const WU = [4e-4, 4e-4, 4e-4, 4e-4, 1.2, 1.2, 1.2, 1.2, 0.3, 0.3, 0.2, 0.2, 0.2];

const _A = new Float64Array(NU * NU), _b = new Float64Array(NU), _du = new Float64Array(NU), _fix = new Int8Array(NU), _res = new Float64Array(NR);
// Bounded weighted least squares by the redistributed pseudo-inverse: solve, clip what
// saturates, freeze it, solve again for the rest.
function wls(J, e, wv, wu, pref, lo, hi, mask) {
  const du = _du, fix = _fix;
  for (let i = 0; i < NU; i++) { du[i] = 0; fix[i] = mask && !mask[i] ? 1 : 0; }
  let sat = 0;
  for (let pass = 0; pass < 4; pass++) {
    // residual after the frozen effectors
    for (let r = 0; r < NR; r++) { let s = e[r]; for (let c = 0; c < NU; c++) if (fix[c]) s -= J[r * NU + c] * du[c]; _res[r] = s; }
    const idx = [];
    for (let c = 0; c < NU; c++) if (!fix[c]) idx.push(c);
    const n = idx.length;
    if (!n) break;
    for (let a = 0; a < n; a++) {
      const ca = idx[a];
      let rhs = wu[ca] * wu[ca] * pref[ca];
      for (let r = 0; r < NR; r++) rhs += J[r * NU + ca] * wv[r] * wv[r] * _res[r];
      _b[a] = rhs;
      for (let c = 0; c < n; c++) {
        const cc = idx[c];
        let s = a === c ? wu[ca] * wu[ca] : 0;
        for (let r = 0; r < NR; r++) s += J[r * NU + ca] * wv[r] * wv[r] * J[r * NU + cc];
        _A[a * n + c] = s;
      }
    }
    solve(_A, _b, n);
    let clipped = false;
    for (let a = 0; a < n; a++) {
      const c = idx[a];
      let x = _b[a];
      if (x < lo[c]) { x = lo[c]; fix[c] = 1; clipped = true; sat++; } else if (x > hi[c]) { x = hi[c]; fix[c] = 1; clipped = true; sat++; }
      du[c] = x;
    }
    if (!clipped) break;
  }
  wls.sat = sat;
  return du.slice();
}
