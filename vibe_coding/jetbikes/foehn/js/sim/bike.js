// FÖHN F-4 "one big engine" jet motorcycle: geometry, mass properties, the engine and its five
// nozzles, aerodynamics and the rigid-body integrator.
//
// Body / design frame: x forward, y up, z to the rider's RIGHT. Design-frame origin is on the
// ground under the middle of the bike (skid soles at y = 0); the physics works about the centre
// of gravity, which moves as fuel burns.
//
// What is modelled
//   * 6-DOF rigid body, 600 Hz, exact exponential-map attitude update, full inertia tensor,
//     gyroscopic coupling of the single big rotor and the reaction of spooling it.
//   * ONE turbofan the rider sits on (the size of a business-jet engine). Its gas goes through a
//     diverter to five exits, like the direct-lift STOVL fighters:
//       - the rear nozzle: thrust forward, after-burner, two-axis vectoring (+-15 deg);
//       - four lift pipes under the engine (front/rear, left/right): thrust up. Each has a fast
//         butterfly valve (its share of the lift gas) and a fore-aft vane (+-12 deg). The four
//         elbows can also swing 26 deg forward together: that is the brake.
//     The spool is slow (FADEC acceleration limits, thrust ~ N^3.2, density lapse); the valves
//     are fast but can only re-distribute the gas the engine makes right now. Splitting the gas
//     costs thrust: half back and half down gives 0.71 of the force of all of it one way.
//     Intake momentum ("ram") drag at the nose intake, duct losses, a thermal budget for the
//     after-burner, fuel burn.
//   * Strip-theory aerodynamics: 4 wing strips, 2 all-moving tailplanes, 2 intake strakes, fin.
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
  tmax: 7000,            // N of gas thrust, sea-level static, dry, all of it through one nozzle
  abGain: 0.6,           // after-burner adds 60 % to whatever leaves through the rear nozzle
  ductEta: 0.94,         // the bends of the lift pipes cost 6 %
  exp: 3.2,              // thrust ~ N^3.2
  idleN: 0.36,
  ve: 560,               // m/s exhaust speed (intake momentum drag = mdot * V)
  tsfc: 3.3e-5,          // kg/(N s) dry (a turbofan)
  tsfcAb: 1.0e-4,        // kg/(N s) for the after-burner part
  rotorI: 0.014, omegaMax: 4600,   // kg m^2, rad/s: one big spool, its momentum lies along +x
  axisY: 0.70, radius: 0.27,
};
// lift pipe exits (design frame) and the rear nozzle. The rear nozzle is canted so that its
// neutral thrust line passes through the centre of gravity (the rider sits above the engine).
export const LIFT = [
  { name: 'FL', p: [0.68, 0.38, -0.33] },
  { name: 'FR', p: [0.68, 0.38, 0.33] },
  { name: 'RL', p: [-0.80, 0.38, -0.33] },
  { name: 'RR', p: [-0.80, 0.38, 0.33] },
];
export const REAR = { p: [-1.42, 0.70, 0], cant: 7.2 * D2R };
export const INTAKE = [1.34, 0.70, 0];
export const TILT_MAX = 116 * D2R, VANE_MAX = 12 * D2R, VEC_MAX = 15 * D2R;
export const SHARE_MIN = 0.10, SHARE_MAX = 0.45;
export const LIM = { da: 25 * D2R, de: 22 * D2R, dr: 25 * D2R };

export const thrustOfN = (N, sigma) => ENG.tmax * Math.pow(sigma, 0.85) * Math.pow(Math.max(N, 0), ENG.exp);
export const nOfThrust = (T, sigma) => Math.pow(Math.max(T, 0) / (ENG.tmax * Math.pow(sigma, 0.85)), 1 / ENG.exp);
// The diverter. "tilt" is the direction of the thrust vector (0 = forward, 90 deg = up). Below
// 90 deg the gas is shared between the rear nozzle and the lift pipes in the ratio cos : sin;
// above it everything goes down and the lift elbows swing forward. Returns the lift fraction.
export const liftFrac = (tilt) => { if (tilt >= Math.PI / 2) return 1; const c = Math.cos(tilt), s = Math.sin(tilt); return s / (c + s); };

export class Turbofan {
  constructor() { this.N = ENG.idleN; this.ab = 0; this.on = true; this.Tdry = 0; this.dNdt = 0; }
  // Gcmd: gas thrust wanted (N, before after-burner and duct losses)
  update(dt, Gcmd, abCmd, sigma) {
    const N0 = this.N;
    if (this.on) {
      this.ab += clamp((abCmd - this.ab) / 0.12, -6, 5) * dt;
      const Nc = clamp(nOfThrust(Gcmd, sigma), ENG.idleN, 1);
      // FADEC schedule: surge margin limits acceleration, flame-out margin limits deceleration
      const d = clamp((Nc - this.N) / 0.16, -1.1, 0.3 + 0.55 * this.N);
      this.N = clamp(this.N + d * dt, ENG.idleN, 1);
    } else {
      this.ab = 0;
      this.N = Math.max(0, this.N - (this.N * 0.35 + 0.004) * dt);
    }
    this.dNdt = (this.N - N0) / dt;
    this.Tdry = this.on ? thrustOfN(this.N, sigma) : 0;
  }
}

// ---------------- mass ----------------
// [mass kg, centre (design frame), box size x,y,z]
const FIXED = [
  [118, [-0.06, 0.70, 0], [2.2, 0.5, 0.5]],     // the engine: fan, core, jet pipe, after-burner
  [13, [-0.02, 0.50, 0], [1.6, 0.25, 0.7]],     // diverter, lift pipes and their valves
  [30, [0.0, 0.92, 0], [2.0, 0.16, 0.30]],      // spine frame, wiring
  [12, [0.50, 1.00, 0], [0.8, 0.30, 0.40]],     // fairing, intake lip
  [14, [-0.40, 0.98, 0], [0.6, 0.14, 0.30]],    // seat, battery, flight computers
  [14, [-0.12, 0.86, 0], [0.7, 0.06, 2.5]],     // wing
  [6, [-1.05, 1.08, 0], [0.4, 0.6, 0.9]],       // tailplanes + fin
  [9, [0, 0.2, 0], [1.3, 0.4, 0.9]],            // skids
  [37, [-0.28, 1.33, 0], [0.34, 0.56, 0.40]],   // rider: torso
  [5.5, [-0.08, 1.74, 0], [0.26, 0.28, 0.24]],  // head + helmet
  [27, [0.02, 0.92, 0], [0.70, 0.30, 0.66]],    // legs
  [8.5, [0.18, 1.30, 0], [0.50, 0.14, 0.60]],   // arms
  [6, [-0.30, 1.02, 0], [0.30, 0.15, 0.36]],    // hips, boots
];
const TANK = { c: [0.10, 0.84, 0], s: [0.9, 0.4, 0.66] };   // saddle tank wrapped over the engine
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
  strip('strakeL', [0.95, 0.72, -0.40], [0, 1, 0], 0.08, 1.4, 0, null, 0, 0, 20),
  strip('strakeR', [0.95, 0.72, 0.40], [0, 1, 0], 0.08, 1.4, 0, null, 0, 0, 20),
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

// Thrust force and moment of the five nozzles (body frame).
// n: { tilt, L[4] thrust of each lift pipe (N), va[4] their fore-aft vanes, R rear thrust (N),
//      rp, ry rear nozzle vectoring (pitch up / yaw) }
export function propWrench(n, cg, out) {
  let Fx = 0, Fy = 0, Fz = 0, Tx = 0, Ty = 0, Tz = 0;
  const sw = Math.max(n.tilt, Math.PI / 2);          // lift elbows: straight down, or swung forward to brake
  for (let i = 0; i < 4; i++) {
    const a = n.va[i], th = sw + a;
    const t = n.L[i] * (1 - 0.3 * a * a);
    const fx = Math.cos(th) * t, fy = Math.sin(th) * t;
    const p = LIFT[i].p, rx = p[0] - cg[0], ry = p[1] - cg[1], rz = p[2] - cg[2];
    Fx += fx; Fy += fy;
    Tx += -rz * fy; Ty += rz * fx; Tz += rx * fy - ry * fx;
  }
  {
    const a = REAR.cant + n.rp, cy = Math.cos(n.ry);
    const t = n.R * (1 - 0.3 * (n.rp * n.rp + n.ry * n.ry));
    const fx = Math.cos(a) * cy * t, fy = Math.sin(a) * cy * t, fz = Math.sin(n.ry) * t;
    const p = REAR.p, rx = p[0] - cg[0], ry = p[1] - cg[1], rz = p[2] - cg[2];
    Fx += fx; Fy += fy; Fz += fz;
    Tx += ry * fz - rz * fy; Ty += rz * fx - rx * fz; Tz += rx * fy - ry * fx;
  }
  out.F[0] = Fx; out.F[1] = Fy; out.F[2] = Fz; out.Tq[0] = Tx; out.Tq[1] = Ty; out.Tq[2] = Tz;
  return out;
}

// Intake momentum drag: the engine swallows mdot of air and must bring it up to its own speed.
export function ramWrench(vB, w, Tdry, cg, out) {
  const md = Tdry / ENG.ve;
  const rx = INTAKE[0] - cg[0], ry = INTAKE[1] - cg[1], rz = INTAKE[2] - cg[2];
  const fx = -md * (vB[0] + w[1] * rz - w[2] * ry), fy = -md * (vB[1] + w[2] * rx - w[0] * rz), fz = -md * (vB[2] + w[0] * ry - w[1] * rx);
  out.F[0] = fx; out.F[1] = fy; out.F[2] = fz;
  out.Tq[0] = ry * fz - rz * fy; out.Tq[1] = rz * fx - rx * fz; out.Tq[2] = rx * fy - ry * fx;
  return out;
}

export const mkW = () => ({ F: [0, 0, 0], Tq: [0, 0, 0], stall: 0 });
export const SKIDS = [[0.58, 0, -0.42], [0.58, 0, 0.42], [-0.62, 0, -0.42], [-0.62, 0, 0.42]];
// points of the airframe and rider that must not touch anything (design frame)
export const PROBES = [
  [1.42, 0.70, 0], [-1.55, 0.74, 0], [-0.08, 1.9, 0], [-1.15, 1.66, 0],       // intake lip, nozzle, helmet, fin tip
  [-0.1, 0.86, -1.26], [-0.1, 0.86, 1.26],                                      // wing tips
  [0.9, 0.42, 0], [-0.9, 0.42, 0],                                              // engine belly
];
export const newCmd = () => ({ G: 0, ab: 0, tilt: Math.PI / 2, s: [1, 1, 1, 1], va: [0, 0, 0, 0], rp: 0, ry: 0, da: 0, de: 0, dr: 0, brake: 0 });

const servo = (x, target, lag, rate, dt) => x + clamp((target - x) / lag, -rate, rate) * dt;

export class Bike {
  constructor() {
    this.eng = new Turbofan();
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
    this.va = [0, 0, 0, 0]; this.sh = [0.25, 0.25, 0.25, 0.25]; this.rp = 0; this.ry = 0;
    this.L = [0, 0, 0, 0]; this.R = 0; this.abOut = 0;          // thrust leaving each lift pipe and the rear nozzle right now (N)
    this.da = 0; this.de = 0; this.dr = 0; this.brake = 0;
    this.heat = 0; this.abLock = false;
    this.t = 0;
    this.Hprev = null;
    this.contactN = 0; this.skidF = [0, 0, 0, 0];
    this.acc = [0, 0, 0]; this.crash = null; this.scrape = 0;
    this.airspeed = speed; this.vAirB = [speed, 0, 0]; this.sigma = 1; this.rho = RHO0; this.stall = 0;
    this.gfx = [1, 1, 1, 1];
    this.agl = 0; this.groundY = pos[1]; this.overWater = false;
    const e = this.eng; e.on = true; e.ab = 0; e.N = speed > 0 ? 0.85 : ENG.idleN; e.update(1e-3, 0, 0, 1);
  }

  arm(d) { const c = this.mp.cg; return [d[0] - c[0], d[1] - c[1], d[2] - c[2]]; }
  toWorld(d) { return add(this.p, qrot(this.q, this.arm(d))); }
  get lift() { return this.L[0] + this.L[1] + this.L[2] + this.L[3]; }
  get thrust() { return this.lift + this.R; }

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
    this.tilt = servo(this.tilt, clamp(cmd.tilt, 0, TILT_MAX), 0.08, 1.7, dt);
    const cs = cmd.s, csum = Math.max(cs[0], 0) + Math.max(cs[1], 0) + Math.max(cs[2], 0) + Math.max(cs[3], 0);
    let ssum = 0;
    for (let i = 0; i < 4; i++) {
      this.va[i] = servo(this.va[i], clamp(cmd.va[i], -VANE_MAX, VANE_MAX), 0.03, 3.2, dt);
      const want = csum > 1e-6 ? clamp(Math.max(cs[i], 0) / csum, SHARE_MIN, SHARE_MAX) : 0.25;
      this.sh[i] = servo(this.sh[i], want, 0.035, 5, dt); ssum += this.sh[i];
    }
    for (let i = 0; i < 4; i++) this.sh[i] /= ssum;      // the valves only share out what there is
    this.rp = servo(this.rp, clamp(cmd.rp, -VEC_MAX, VEC_MAX), 0.04, 2.5, dt);
    this.ry = servo(this.ry, clamp(cmd.ry, -VEC_MAX, VEC_MAX), 0.04, 2.5, dt);
    this.da = servo(this.da, clamp(cmd.da, -LIM.da, LIM.da), 0.04, 2.5, dt);
    this.de = servo(this.de, clamp(cmd.de, -LIM.de, LIM.de), 0.04, 2.5, dt);
    this.dr = servo(this.dr, clamp(cmd.dr, -LIM.dr, LIM.dr), 0.04, 2.5, dt);
    this.brake = servo(this.brake, clamp(cmd.brake, 0, 1), 0.12, 4, dt);

    // --- after-burner thermal budget: ~7 s of reheat, cools faster in a fast air stream ---
    const e = this.eng;
    const abWanted = cmd.ab > 0.5 && !this.abLock && this.fuel > 0 ? 1 : 0;
    const abNow = e.ab;
    this.heat = clamp(this.heat + (abNow * 0.15 - (1 - abNow) * (0.055 + 0.06 * Math.min(1, this.airspeed / 80))) * dt, 0, 1);
    if (this.heat >= 1) this.abLock = true; else if (this.heat < 0.45) this.abLock = false;

    // --- the engine and where its gas goes ---
    e.update(dt, cmd.G, abWanted, sigma);
    const fl = liftFrac(this.tilt), s0 = Math.sin(Math.max(this.tilt, Math.PI / 2));
    const gasR = e.Tdry * (1 - fl), gasL = e.Tdry * fl * ENG.ductEta;
    this.R = gasR * (1 + ENG.abGain * e.ab);
    this.abOut = e.ab * clamp(gasR / (0.12 * ENG.tmax), 0, 1);   // how much flame there really is: reheat needs gas in the jet pipe
    for (let i = 0; i < 4; i++) {
      // ground effect: suck-down and hot-gas re-ingestion when the jet hits nearby ground
      const hn = Math.max(0, this.agl + LIFT[i].p[1]);
      this.gfx[i] = 1 - (0.05 * Math.exp(-hn / 0.8) + 0.03 * Math.exp(-hn / 2.4)) * s0;
      this.L[i] = gasL * this.sh[i] * this.gfx[i];
    }
    const flow = e.on ? ENG.tsfc * e.Tdry + ENG.tsfcAb * gasR * ENG.abGain * e.ab + 0.012 : 0;
    const H = [ENG.rotorI * ENG.omegaMax * e.N, 0, 0];

    // --- air data ---
    const vAirW = sub(this.v, windW);
    const vB = qinvrot(q, vAirW);
    this.vAirB = vB; this.airspeed = len(vB); this.sigma = sigma; this.rho = rho;

    // --- forces (body frame) ---
    const A = aeroWrench(vB, w, rho, this, cg, this._a);
    const P = propWrench(this, cg, this._p);
    const R = ramWrench(vB, w, e.Tdry, cg, this._r);
    this.stall = this.airspeed > 18 ? A.stall : 0;
    const Fb = [A.F[0] + P.F[0] + R.F[0], A.F[1] + P.F[1] + R.F[1], A.F[2] + P.F[2] + R.F[2]];
    let Tq = [A.Tq[0] + P.Tq[0] + R.Tq[0], A.Tq[1] + P.Tq[1] + R.Tq[1], A.Tq[2] + P.Tq[2] + R.Tq[2]];
    // the spool changing speed reacts on the airframe (a blip of throttle rolls the bike a touch)
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
    if (this.fuel <= 0) e.on = false;
    this.fuelFlow = flow;
    this.agl = this.p[1] - cg[1] - this.groundY;
    this.t += dt;
  }
}
