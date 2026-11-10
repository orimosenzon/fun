// FÖHN F-3 tilt-jet motorcycle: geometry, mass properties, engines, aerodynamics and the
// rigid-body integrator.
//
// Body / design frame: x forward, y up, z to the rider's RIGHT. Design-frame origin is on the
// ground under the middle of the bike (skid soles at y = 0); the physics works about the centre
// of gravity, which moves as fuel burns.
//
// What is modelled
//   * 6-DOF rigid body, 600 Hz, exact exponential-map attitude update, full inertia tensor,
//     gyroscopic coupling of the four rotors and the reaction of tilting/spooling them.
//   * Four small turbojets in pods that tilt together from 0 deg (thrust forward, wing-borne
//     flight) to 116 deg (thrust up and back: hover and braking), like a tilt-jet VTOL.
//     Spool dynamics with FADEC acceleration limits, thrust ~ N^3.2, density lapse, intake
//     momentum ("ram") drag applied at each intake, after-burner with a thermal budget, fuel burn.
//   * Two-axis exhaust vanes on every nozzle (fast, +-12 deg) for attitude control.
//   * Strip-theory aerodynamics: 4 wing strips, 2 all-moving tailplanes, 2 canard pylons, fin.
//     Every strip sees its own local airflow (v + w x r), so roll/pitch/yaw damping, adverse
//     yaw, weathercock stability and asymmetric stall all come out of the model. Lift is linear
//     up to the stall and blends into flat-plate behaviour beyond it; induced drag from the
//     aspect ratio. Ailerons, elevator, rudder and an air brake.
//   * Bluff-body drag of bike + rider per axis with separate centres of pressure.
//   * Jet suck-down near the ground, spring-damper skids with Coulomb friction, crash probes.
import { add, sub, scale, madd, dot, cross, len, qmul, qnorm, qrot, qinvrot, qexp, m3v, m3inv, clamp } from './vmath.js';
import { groundAt, normalAt, waterAt, airDensity } from './terrain.js';

export const G = 9.80665;
const RHO0 = 1.225, D2R = Math.PI / 180;

// ---------------- propulsion ----------------
export const ENG = {
  tmax: 1450,            // N, sea-level static, dry, per engine
  abGain: 0.48,          // after-burner adds 48 %
  exp: 3.2,              // thrust ~ N^3.2
  idleN: 0.36,
  ve: 560,               // m/s exhaust speed (intake momentum drag = mdot * V)
  tsfc: 4.2e-5,          // kg/(N s) dry
  tsfcAb: 1.6e-4,        // kg/(N s) for the after-burner part
  rotorI: 1.5e-3, omegaMax: 10000,   // kg m^2, rad/s
  podLen: 0.36,          // pivot -> nozzle exit / intake lip
};
export const PODS = [
  { name: 'FL', p: [0.95, 0.72, -0.62], spin: +1 },
  { name: 'FR', p: [0.95, 0.72, 0.62], spin: -1 },
  { name: 'RL', p: [-1.0, 0.98, -0.62], spin: -1 },
  { name: 'RR', p: [-1.0, 0.98, 0.62], spin: +1 },
];
export const TILT_MAX = 116 * D2R, VANE_MAX = 12 * D2R;
export const LIM = { da: 25 * D2R, de: 22 * D2R, dr: 25 * D2R };

export const thrustOfN = (N, sigma) => ENG.tmax * Math.pow(sigma, 0.85) * Math.pow(Math.max(N, 0), ENG.exp);
export const nOfThrust = (T, sigma) => Math.pow(Math.max(T, 0) / (ENG.tmax * Math.pow(sigma, 0.85)), 1 / ENG.exp);

export class Turbojet {
  constructor() { this.N = ENG.idleN; this.ab = 0; this.on = true; this.Tdry = 0; this.T = 0; this.dNdt = 0; this.flow = 0; }
  update(dt, Tcmd, abCmd, sigma) {
    const N0 = this.N;
    if (this.on) {
      this.ab += clamp((abCmd - this.ab) / 0.12, -6, 5) * dt;
      const Nc = clamp(nOfThrust(Tcmd / (1 + ENG.abGain * this.ab), sigma), ENG.idleN, 1);
      // FADEC schedule: surge margin limits acceleration, flame-out margin limits deceleration
      const d = clamp((Nc - this.N) / 0.16, -1.1, 0.3 + 0.55 * this.N);
      this.N = clamp(this.N + d * dt, ENG.idleN, 1);
    } else {
      this.ab = 0;
      this.N = Math.max(0, this.N - (this.N * 0.35 + 0.004) * dt);
    }
    this.dNdt = (this.N - N0) / dt;
    this.Tdry = this.on ? thrustOfN(this.N, sigma) : 0;
    this.T = this.Tdry * (1 + ENG.abGain * this.ab);
    this.flow = this.on ? ENG.tsfc * this.Tdry + ENG.tsfcAb * this.Tdry * ENG.abGain * this.ab + 0.003 : 0;
  }
}

// ---------------- mass ----------------
// [mass kg, centre (design frame), box size x,y,z]
const FIXED = [
  [46, [0.0, 0.74, 0], [2.0, 0.28, 0.30]],      // spine frame, wiring, landing skids
  [12, [0.45, 0.96, 0], [0.8, 0.35, 0.40]],     // fairing, tank shell
  [14, [-0.40, 0.88, 0], [0.6, 0.14, 0.30]],    // seat, battery, flight computers
  [14, [-0.12, 0.86, 0], [0.7, 0.06, 2.5]],     // wing
  [6, [-1.05, 1.02, 0], [0.4, 0.6, 0.9]],       // tailplanes + fin
  ...PODS.map((e) => [24, e.p, [0.72, 0.2, 0.2]]),
  [10, [0, 0.3, 0], [1.3, 0.5, 0.9]],           // pylons, skids
  [37, [-0.28, 1.33, 0], [0.34, 0.56, 0.40]],   // rider: torso
  [5.5, [-0.08, 1.74, 0], [0.26, 0.28, 0.24]],  // head + helmet
  [27, [0.02, 0.98, 0], [0.70, 0.30, 0.46]],    // legs
  [8.5, [0.18, 1.30, 0], [0.50, 0.14, 0.60]],   // arms
  [6, [-0.30, 1.02, 0], [0.30, 0.15, 0.36]],    // hips, boots
];
const TANK = { c: [0.28, 0.94, 0], s: [0.68, 0.3, 0.4] };
export const FUEL0 = 85;

export function massProps(fuel) {
  const items = [...FIXED, [fuel, TANK.c, TANK.s]];
  let m = 0;
  const cg = [0, 0, 0];
  for (const [mi, c] of items) { m += mi; cg[0] += mi * c[0]; cg[1] += mi * c[1]; cg[2] += mi * c[2]; }
  cg[0] /= m; cg[1] /= m; cg[2] /= m;
  const I = [0, 0, 0, 0, 0, 0, 0, 0, 0];
  for (const [mi, c, s] of items) {
    const [a, b, d] = s, x = c[0] - cg[0], y = c[1] - cg[1], z = c[2] - cg[2];
    I[0] += mi * (b * b + d * d) / 12 + mi * (y * y + z * z);
    I[4] += mi * (a * a + d * d) / 12 + mi * (x * x + z * z);
    I[8] += mi * (a * a + b * b) / 12 + mi * (x * x + y * y);
    I[1] -= mi * x * y; I[2] -= mi * x * z; I[5] -= mi * y * z;
  }
  I[3] = I[1]; I[6] = I[2]; I[7] = I[5];
  return { m, cg, I, Iinv: m3inv(I) };
}

// ---------------- aerodynamics ----------------
const claOf = (AR) => 2 * Math.PI * AR / (2 + Math.sqrt(4 + AR * AR));
const strip = (name, r, n, S, AR, a0, ctrl, gain, tau, stall = 16) => ({ name, r, n, c: [1, 0, 0], S, AR, cla: claOf(AR), a0: a0 * D2R, ctrl, gain, tau, cd0: 0.012, stall: stall * D2R });
export const SURF = [
  strip('wingLi', [-0.02, 0.86, -0.45], [0, 1, 0], 0.52, 3.1, -5.5, null, 0, 0, 17),
  strip('wingRi', [-0.02, 0.86, 0.45], [0, 1, 0], 0.52, 3.1, -5.5, null, 0, 0, 17),
  strip('wingLo', [-0.06, 0.86, -0.98], [0, 1, 0], 0.48, 3.1, -4.5, 'da', +1, 0.45, 17),
  strip('wingRo', [-0.06, 0.86, 0.98], [0, 1, 0], 0.48, 3.1, -4.5, 'da', -1, 0.45, 17),
  strip('tailL', [-1.08, 0.98, -0.34], [0, 1, 0], 0.17, 2.4, 0, 'de', -1, 1.0, 18),
  strip('tailR', [-1.08, 0.98, 0.34], [0, 1, 0], 0.17, 2.4, 0, 'de', -1, 1.0, 18),
  strip('canL', [0.95, 0.72, -0.40], [0, 1, 0], 0.08, 1.4, 0, null, 0, 0, 20),
  strip('canR', [0.95, 0.72, 0.40], [0, 1, 0], 0.08, 1.4, 0, null, 0, 0, 20),
  strip('fin', [-1.10, 1.30, 0], [0, 0, 1], 0.30, 1.6, 0, 'dr', +1, 0.5, 20),
];
export const WING = { S: 2.0, cla: claOf(3.1), a0: -5.0 * D2R, stall: 17 * D2R };
WING.alphaMax = WING.stall + WING.a0 - 4.5 * D2R;   // body angle of attack the computer will not exceed
WING.clUse = WING.cla * (WING.alphaMax - WING.a0);
// bluff body (bike + rider): drag area for flow along each body axis and its centre of pressure
export const BODY = { cda: [0.46, 1.25, 1.15], cp: [[0.05, 1.05, 0], [-0.08, 0.9, 0], [-0.16, 1.0, 0]], brakeCda: 0.75, brakeCp: [-0.55, 1.05, 0] };

// Forces of one strip in the local air flow. Returns [cl, cd] at effective angle of attack a.
function clcd(s, a) {
  const sa = Math.sin(a), ca = Math.cos(a);
  const st = 1 / (1 + Math.exp(-(Math.abs(a) - s.stall) / 0.045));   // 0 attached .. 1 fully stalled
  const clA = s.cla * a;
  const cl = (1 - st) * clA + st * 2.1 * sa * ca;
  const cd = (1 - st) * (s.cd0 + clA * clA / (Math.PI * 0.8 * s.AR)) + st * (s.cd0 + 1.95 * sa * sa);
  return [cl, cd];
}

// Aerodynamic force and moment about the CG (body frame).
// vB: air-relative velocity of the CG in body axes; w: body rates; ctl: {da,de,dr,brake}
export function aeroWrench(vB, w, rho, ctl, cg, out) {
  let Fx = 0, Fy = 0, Fz = 0, Tx = 0, Ty = 0, Tz = 0;
  let stall = 0;
  for (let i = 0; i < SURF.length; i++) {
    const s = SURF[i];
    const rx = s.r[0] - cg[0], ry = s.r[1] - cg[1], rz = s.r[2] - cg[2];
    const vx = vB[0] + w[1] * rz - w[2] * ry, vy = vB[1] + w[2] * rx - w[0] * rz, vz = vB[2] + w[0] * ry - w[1] * rx;
    const u = vx;                                   // chord axis is body x for every strip
    const wn = s.n[1] ? vy : vz;                    // normal axis is y (wings) or z (fin)
    const V2 = u * u + wn * wn;
    if (V2 < 0.01) continue;
    const V = Math.sqrt(V2);
    let a = Math.atan2(-wn, u) - s.a0;
    if (s.ctrl) a += s.gain * s.tau * ctl[s.ctrl];
    const [cl, cd] = clcd(s, a);
    if (i < 4) stall = Math.max(stall, (Math.abs(a) - s.stall) / 0.1);
    const qS = 0.5 * rho * V2 * s.S;
    // lift is perpendicular to the local flow (in the chord-normal plane), drag along it
    const lx = wn / V, ln = u / V;                  // lift dir = (u n - wn c)/V  ->  c: -wn/V, n: u/V
    const fc = qS * (-cl * lx - cd * u / V);
    const fn = qS * (cl * ln - cd * wn / V);
    const fx = fc, fy = s.n[1] ? fn : 0, fz = s.n[1] ? 0 : fn;
    Fx += fx; Fy += fy; Fz += fz;
    Tx += ry * fz - rz * fy; Ty += rz * fx - rx * fz; Tz += rx * fy - ry * fx;
  }
  // bluff body
  const V = Math.hypot(vB[0], vB[1], vB[2]);
  for (let k = 0; k < 3; k++) {
    const f = -0.5 * rho * BODY.cda[k] * V * vB[k];
    const r = BODY.cp[k], rx = r[0] - cg[0], ry = r[1] - cg[1], rz = r[2] - cg[2];
    const fx = k === 0 ? f : 0, fy = k === 1 ? f : 0, fz = k === 2 ? f : 0;
    Fx += fx; Fy += fy; Fz += fz;
    Tx += ry * fz - rz * fy; Ty += rz * fx - rx * fz; Tz += rx * fy - ry * fx;
  }
  if (ctl.brake > 0.001) {
    const k = -0.5 * rho * BODY.brakeCda * ctl.brake * V;
    const fx = k * vB[0], fy = k * vB[1], fz = k * vB[2];
    const r = BODY.brakeCp, rx = r[0] - cg[0], ry = r[1] - cg[1], rz = r[2] - cg[2];
    Fx += fx; Fy += fy; Fz += fz;
    Tx += ry * fz - rz * fy; Ty += rz * fx - rx * fz; Tz += rx * fy - ry * fx;
  }
  // rider/fairing rate damping not captured by the strips
  const kd = rho / RHO0 * (1 + V / 30);
  Tx -= 6 * kd * w[0]; Ty -= 9 * kd * w[1]; Tz -= 9 * kd * w[2];
  out.F[0] = Fx; out.F[1] = Fy; out.F[2] = Fz; out.Tq[0] = Tx; out.Tq[1] = Ty; out.Tq[2] = Tz;
  out.stall = stall;
  return out;
}

// Thrust force and moment of the four nozzles (body frame). T: thrust per engine (N),
// tilt: pod angle, va/vb: vane deflection in the tilt plane / sideways.
export function propWrench(tilt, T, va, vb, cg, out) {
  let Fx = 0, Fy = 0, Fz = 0, Tx = 0, Ty = 0, Tz = 0;
  const c0 = Math.cos(tilt), s0 = Math.sin(tilt);
  for (let i = 0; i < 4; i++) {
    const a = va[i], b = vb[i];
    const th = tilt + a, cb = Math.cos(b);
    const t = T[i] * (1 - 0.3 * (a * a + b * b));
    const fx = Math.cos(th) * cb * t, fy = Math.sin(th) * cb * t, fz = Math.sin(b) * t;
    const p = PODS[i].p;
    const rx = p[0] - ENG.podLen * c0 - cg[0], ry = p[1] - ENG.podLen * s0 - cg[1], rz = p[2] - cg[2];
    Fx += fx; Fy += fy; Fz += fz;
    Tx += ry * fz - rz * fy; Ty += rz * fx - rx * fz; Tz += rx * fy - ry * fx;
  }
  out.F[0] = Fx; out.F[1] = Fy; out.F[2] = Fz; out.Tq[0] = Tx; out.Tq[1] = Ty; out.Tq[2] = Tz;
  return out;
}

// Intake momentum drag: each engine swallows mdot of air and must bring it up to its own speed.
export function ramWrench(vB, w, tilt, Tdry, cg, out) {
  let Fx = 0, Fy = 0, Fz = 0, Tx = 0, Ty = 0, Tz = 0;
  const c0 = Math.cos(tilt), s0 = Math.sin(tilt);
  for (let i = 0; i < 4; i++) {
    const md = Tdry[i] / ENG.ve;
    const p = PODS[i].p;
    const rx = p[0] + ENG.podLen * c0 - cg[0], ry = p[1] + ENG.podLen * s0 - cg[1], rz = p[2] - cg[2];
    const fx = -md * (vB[0] + w[1] * rz - w[2] * ry), fy = -md * (vB[1] + w[2] * rx - w[0] * rz), fz = -md * (vB[2] + w[0] * ry - w[1] * rx);
    Fx += fx; Fy += fy; Fz += fz;
    Tx += ry * fz - rz * fy; Ty += rz * fx - rx * fz; Tz += rx * fy - ry * fx;
  }
  out.F[0] = Fx; out.F[1] = Fy; out.F[2] = Fz; out.Tq[0] = Tx; out.Tq[1] = Ty; out.Tq[2] = Tz;
  return out;
}

export const mkW = () => ({ F: [0, 0, 0], Tq: [0, 0, 0], stall: 0 });
export const SKIDS = [[0.58, 0, -0.42], [0.58, 0, 0.42], [-0.62, 0, -0.42], [-0.62, 0, 0.42]];
// points of the airframe and rider that must not touch anything (design frame)
export const PROBES = [
  [1.45, 0.92, 0], [-1.35, 1.0, 0], [-0.08, 1.9, 0], [-1.15, 1.6, 0],        // nose, tail, helmet, fin tip
  [-0.1, 0.86, -1.26], [-0.1, 0.86, 1.26],                                      // wing tips
  [1.3, 0.6, -0.62], [1.3, 0.6, 0.62], [-1.36, 0.86, -0.62], [-1.36, 0.86, 0.62], // pod ends
];
export const newCmd = () => ({ T: [0, 0, 0, 0], ab: 0, tilt: Math.PI / 2, va: [0, 0, 0, 0], vb: [0, 0, 0, 0], da: 0, de: 0, dr: 0, brake: 0 });

const servo = (x, target, lag, rate, dt) => x + clamp((target - x) / lag, -rate, rate) * dt;

export class Bike {
  constructor() {
    this.eng = PODS.map(() => new Turbojet());
    this._a = mkW(); this._p = mkW(); this._r = mkW();
    this.reset([0, 0, 0], 0);
  }

  // place the bike standing on its skids at ground point pos (or flying if speed > 0)
  reset(pos, yaw, speed = 0, fuel = FUEL0) {
    this.fuel = fuel;
    this.mp = massProps(fuel); this._mpAge = 0;
    this.p = [pos[0], pos[1] + (speed > 0 ? 0 : this.mp.cg[1] - 0.01), pos[2]];
    this.q = [Math.cos(yaw / 2), 0, Math.sin(yaw / 2), 0];
    this.v = [Math.cos(yaw) * speed, 0, -Math.sin(yaw) * speed];
    this.w = [0, 0, 0];
    this.tilt = speed > 40 ? 0.1 : Math.PI / 2;
    this.va = [0, 0, 0, 0]; this.vb = [0, 0, 0, 0];
    this.da = 0; this.de = 0; this.dr = 0; this.brake = 0;
    this.heat = 0; this.abLock = false;
    this.t = 0;
    this.Hprev = null;
    this.contactN = 0; this.skidF = [0, 0, 0, 0];
    this.acc = [0, 0, 0]; this.crash = null; this.scrape = 0;
    this.airspeed = speed; this.vAirB = [speed, 0, 0]; this.sigma = 1; this.rho = RHO0; this.stall = 0;
    this.gfx = [1, 1, 1, 1];
    this.agl = 0; this.groundY = pos[1]; this.overWater = false;
    for (const e of this.eng) { e.on = true; e.ab = 0; e.N = speed > 0 ? 0.85 : ENG.idleN; e.update(1e-3, 0, 0, 1); }
  }

  arm(d) { const c = this.mp.cg; return [d[0] - c[0], d[1] - c[1], d[2] - c[2]]; }
  toWorld(d) { return add(this.p, qrot(this.q, this.arm(d))); }
  get thrust() { return this.eng[0].T + this.eng[1].T + this.eng[2].T + this.eng[3].T; }

  step(dt, cmd, windW) {
    if (this._mpAge-- <= 0) { this.mp = massProps(this.fuel); this._mpAge = 30; }
    const mp = this.mp, cg = mp.cg, q = this.q, w = this.w;
    if (!(this._gAge-- > 0)) {
      const g = groundAt(this.p[0], this.p[2]), wl = waterAt(this.p[0], this.p[2]);
      this.overWater = wl > g; this.groundY = Math.max(g, wl); this._gAge = 4;
      this.agl = this.p[1] - cg[1] - this.groundY;
    }
    const rho = airDensity(this.p[1]), sigma = rho / RHO0;

    // --- actuators ---
    this.tilt = servo(this.tilt, clamp(cmd.tilt, 0, TILT_MAX), 0.08, 1.15, dt);
    for (let i = 0; i < 4; i++) {
      this.va[i] = servo(this.va[i], clamp(cmd.va[i], -VANE_MAX, VANE_MAX), 0.03, 3.2, dt);
      this.vb[i] = servo(this.vb[i], clamp(cmd.vb[i], -VANE_MAX, VANE_MAX), 0.03, 3.2, dt);
    }
    this.da = servo(this.da, clamp(cmd.da, -LIM.da, LIM.da), 0.04, 2.5, dt);
    this.de = servo(this.de, clamp(cmd.de, -LIM.de, LIM.de), 0.04, 2.5, dt);
    this.dr = servo(this.dr, clamp(cmd.dr, -LIM.dr, LIM.dr), 0.04, 2.5, dt);
    this.brake = servo(this.brake, clamp(cmd.brake, 0, 1), 0.12, 4, dt);

    // --- after-burner thermal budget: ~7 s of reheat, cools faster in a fast air stream ---
    const abWanted = cmd.ab > 0.5 && !this.abLock && this.fuel > 0 ? 1 : 0;
    const abNow = this.eng[0].ab;
    this.heat = clamp(this.heat + (abNow * 0.15 - (1 - abNow) * (0.055 + 0.06 * Math.min(1, this.airspeed / 80))) * dt, 0, 1);
    if (this.heat >= 1) this.abLock = true; else if (this.heat < 0.45) this.abLock = false;

    // --- engines ---
    const T = this._T || (this._T = [0, 0, 0, 0]), Td = this._Td || (this._Td = [0, 0, 0, 0]);
    let flow = 0;
    const c0 = Math.cos(this.tilt), s0 = Math.sin(this.tilt);
    const axis = [c0, s0, 0];
    let H = [0, 0, 0];
    for (let i = 0; i < 4; i++) {
      const e = this.eng[i];
      e.update(dt, cmd.T[i], abWanted, sigma);
      // ground effect: suck-down and hot-gas re-ingestion when the jet points at nearby ground
      const hn = Math.max(0, this.agl + PODS[i].p[1] - ENG.podLen * s0);
      this.gfx[i] = 1 - (0.05 * Math.exp(-hn / 0.8) + 0.03 * Math.exp(-hn / 2.4)) * Math.max(0, s0);
      T[i] = e.T * this.gfx[i]; Td[i] = e.Tdry;
      flow += e.flow;
      H = madd(H, axis, PODS[i].spin * ENG.rotorI * ENG.omegaMax * e.N);
    }

    // --- air data ---
    const vAirW = sub(this.v, windW);
    const vB = qinvrot(q, vAirW);
    this.vAirB = vB; this.airspeed = len(vB); this.sigma = sigma; this.rho = rho;

    // --- forces (body frame) ---
    const A = aeroWrench(vB, w, rho, this, cg, this._a);
    const P = propWrench(this.tilt, T, this.va, this.vb, cg, this._p);
    const R = ramWrench(vB, w, this.tilt, Td, cg, this._r);
    this.stall = this.airspeed > 18 ? A.stall : 0;
    const Fb = [A.F[0] + P.F[0] + R.F[0], A.F[1] + P.F[1] + R.F[1], A.F[2] + P.F[2] + R.F[2]];
    let Tq = [A.Tq[0] + P.Tq[0] + R.Tq[0], A.Tq[1] + P.Tq[1] + R.Tq[1], A.Tq[2] + P.Tq[2] + R.Tq[2]];
    // rotor momentum changing in the body frame (tilting pods, spooling) reacts on the airframe
    if (this.Hprev) Tq = madd(Tq, sub(H, this.Hprev), -1 / dt);
    this.Hprev = H;

    let Fw = qrot(q, Fb);
    Fw[1] -= mp.m * G;

    // --- ground contact ---
    this.contactN = 0;
    if (this.agl < 4) {
      for (let i = 0; i < 4; i++) {
        const r = this.arm(SKIDS[i]);
        const pw = add(this.p, qrot(q, r));
        const gy = groundAt(pw[0], pw[2]);
        const pen = gy - pw[1];
        this.skidF[i] = 0;
        if (waterAt(pw[0], pw[2]) - 0.15 > Math.max(pw[1], gy)) { this.crash = this.crash || 'water'; continue; }
        if (pen <= 0) continue;
        const n = normalAt(pw[0], pw[2]);
        const vp = add(this.v, qrot(q, cross(w, r)));
        const vn = dot(vp, n);
        if (vn < -6.5) this.crash = this.crash || 'hard';
        const Fn = Math.max(0, 60000 * pen - 3300 * vn);
        const vt = madd(vp, n, -vn), vtl = len(vt);
        if (vtl > 34) this.crash = this.crash || 'ground';
        const mu = 0.6 * Math.min(1, vtl / 0.04);
        const fW = vtl > 1e-6 ? madd(scale(n, Fn), vt, -mu * Fn / vtl) : scale(n, Fn);
        Fw = add(Fw, fW);
        Tq = add(Tq, cross(r, qinvrot(q, fW)));
        this.skidF[i] = Fn; this.contactN += Fn;
      }
      for (let i = 0; i < PROBES.length; i++) {
        const r = this.arm(PROBES[i]);
        const pw = add(this.p, qrot(q, r));
        const gy = groundAt(pw[0], pw[2]), wl = waterAt(pw[0], pw[2]);
        if (wl > gy && pw[1] < wl) { this.crash = this.crash || 'water'; continue; }
        const pen = gy - pw[1];
        if (pen <= 0) continue;
        const n = normalAt(pw[0], pw[2]);
        const vp = add(this.v, qrot(q, cross(w, r)));
        const vn = dot(vp, n);
        if (vn < -2.2 || len(vp) > 12 || pen > 0.35) { this.crash = this.crash || 'ground'; continue; }
        const fW = scale(n, Math.max(0, 30000 * pen - 2500 * vn));   // gentle scrape
        Fw = add(Fw, fW); Tq = add(Tq, cross(r, qinvrot(q, fW)));
        this.scrape += dt;
      }
    }

    // --- integrate ---
    this.acc = scale(Fw, 1 / mp.m);
    this.v = madd(this.v, this.acc, dt);
    this.p = madd(this.p, this.v, dt);
    const wdot = m3v(mp.Iinv, sub(Tq, cross(w, add(m3v(mp.I, w), H))));
    this.w = madd(w, wdot, dt);
    this.q = qnorm(qmul(q, qexp(scale(this.w, dt))));
    this.fuel = Math.max(0, this.fuel - flow * dt);
    if (this.fuel <= 0) for (const e of this.eng) e.on = false;
    this.fuelFlow = flow;
    this.agl = this.p[1] - cg[1] - this.groundY;
    this.t += dt;
  }
}
