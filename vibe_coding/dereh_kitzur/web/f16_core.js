/* The F-16 over the map: the flight itself, with nothing drawn (26/11/2026).
 *
 * Until today the jet was a camera with an F-16's numbers painted on it: the
 * speed followed a button, the arrows moved the altitude, and letting go
 * stopped it in the air. Ori asked for the flight to be like a modern flight
 * simulator's. So this is an aeroplane: a mass with a wing, an engine and a
 * flight control computer, flown with a stick.
 *
 * What is modelled, from the public figures for the F-16C block 50:
 *
 *   - the air: the standard atmosphere, density and the speed of sound by
 *     height;
 *   - the wing: lift against angle of attack with a stall that rounds off at
 *     about 32 degrees, induced drag, the transonic drag rise, the gear, the
 *     speed brakes and the flaps that come down with the gear;
 *   - the engine: an F110-GE-129, 76 kN dry and 131 kN in afterburner at
 *     rest, thinning with height and gaining with ram, a compressor that
 *     takes seconds to spool, a burner that lights in stages, and fuel that
 *     is burned and weighed;
 *   - the flight control computer (FLCS), which is what makes an F-16 an
 *     F-16: the stick does not move surfaces, it asks for g. Fore and aft is
 *     a load factor between -3 and +9 with 1 g in the middle, so hands off
 *     the aircraft holds its path; side to side is a roll rate; and the
 *     computer will not let the wing past 25 degrees of angle of attack, so
 *     a stick held back at low speed gives all the lift there is and no
 *     more, and the aircraft mushes instead of spinning;
 *   - the ground: wheels, brakes, nosewheel steering, rotation, lift-off,
 *     and a touchdown that is either a landing or a crash.
 *
 * What is not: moments. The surfaces are not simulated; the computer's
 * result is. The nose is turned at the rate that holds the angle of attack
 * the computer wants, with an authority that fades with dynamic pressure, so
 * at the top of a loop flown too slowly the nose stops answering and falls
 * through, as it should, and comes back as the speed does.
 *
 * The frame is the one the other vehicles here use: x east, y up, z south,
 * metres from the take-off point. The aircraft's own axes are three.js's
 * camera's: forward is -z, right is +x, up is +y. A compass bearing b looks
 * along (sin b, 0, -cos b).
 *
 * No three.js in here, on purpose: tests/test_f16core.mjs flies this file in
 * Node.
 */

export const KT = 1.943844;        // m/s to knots
export const FT = 3.280840;        // metres to feet
export const LB = 2.204623;        // kg to pounds
const G0 = 9.80665;
const RAD = Math.PI / 180;

export const AC = {
  S: 27.87,                // wing area, m²
  EMPTY: 9200,             // kg: the airframe, the pilot, two missiles on the tips
  FUEL_MAX: 3200,          // kg of internal fuel, about 7,000 lb
  T_MIL: 76300,            // N, dry, at rest at sea level
  T_AB: 131000,            // and in full afterburner
  T_IDLE: 3200,
  MIL: 0.66,               // the lever's position at the military-power detent; past it is the burner
  CL0: 0.05,
  CLA: 3.61,               // lift slope per radian, 0.063 per degree
  CL_FLAP: 0.25,           // the trailing-edge flaps, down with the gear
  A_MAX: 25 * RAD,         // the computer's angle-of-attack limit
  A_MAX_GEAR: 21 * RAD,
  A_MIN: -10 * RAD,
  G_MAX: 9, G_MIN: -3,
  P_MAX: 4.2,              // rad/s of roll with the stick on the stop, 240 degrees a second
  Q_LIM: 103800,           // Pa: 800 knots of dynamic pressure, the airframe's limit low down
  M_LIM: 2.05,             // and the intake's limit up high
  CEILING: 15240,
  V_GEAR: 154,             // m/s, 300 kt: above this the gear should be up
  SINK_OK: 6,              // m/s of sink the gear takes; the book says 3
  THETA_MAX_G: 14 * RAD    // on the wheels, the tail touches past this
};

const clamp = (v, a, b) => (v < a ? a : v > b ? b : v);
const lerp = (a, b, t) => a + (b - a) * t;
const sstep = (a, b, x) => { const t = clamp((x - a) / (b - a), 0, 1); return t * t * (3 - 2 * t); };

/* ---------- the air ---------- */

/** The standard atmosphere: density (kg/m³) and the speed of sound (m/s). */
export function air(h) {
  h = clamp(h, -200, 25000);
  if (h < 11000) {
    const T = 288.15 - 0.0065 * h;
    return { rho: 1.225 * Math.pow(T / 288.15, 4.2559), a: Math.sqrt(401.87 * T) };
  }
  return { rho: 0.36392 * Math.exp(-(h - 11000) / 6341.6), a: 295.07 };
}

/* ---------- the wing ---------- */

/** Lift coefficient against angle of attack (radians). Straight to sixteen
 *  degrees, rounding off to a maximum at thirty-two, and past that a plate. */
export function liftCoef(alpha, flap = 0) {
  const s = alpha < 0 ? -1 : 1;
  const d = Math.abs(alpha) / RAD;
  let cl;
  if (d <= 16) cl = 0.063 * d;
  else if (d <= 32) cl = 1.008 + 0.063 * (d - 16) - 0.00197 * (d - 16) * (d - 16);
  else cl = lerp(1.512, 1.1 * Math.sin(2 * d * RAD), sstep(32, 50, d));
  cl *= s < 0 ? 0.8 : 1;
  const att = 1 - sstep(20, 40, d);          // camber and flaps mean nothing to a stalled wing
  return s * cl + (AC.CL0 + AC.CL_FLAP * flap) * att;
}

/** Zero-lift drag by Mach number: flat, the transonic rise, a slow fall. */
function cd0(M) {
  if (M < 0.85) return 0.0185;
  if (M < 1.08) return 0.0185 + 0.0265 * sstep(0.85, 1.08, M);
  return lerp(0.045, 0.035, clamp((M - 1.08) / 0.92, 0, 1));
}

export function dragCoef(cl, alpha, M, gear, brake, flap) {
  const k = 0.115 + 0.19 * sstep(0.9, 1.4, M);
  const d = Math.abs(alpha) / RAD;
  const plate = 1.9 * Math.pow(Math.sin(alpha), 2) * sstep(18, 40, d);
  return cd0(M) + k * cl * cl * (1 - 0.5 * sstep(25, 45, d)) + plate + 0.022 * gear + 0.045 * brake + 0.014 * flap;
}

/* ---------- the engine ---------- */

/** Thrust in newtons from the compressor's speed (0.62 idle to 1), the
 *  burner (0..1), the Mach number and the density ratio. */
function thrustOf(rpm, ab, M, sigma) {
  const lapse = Math.pow(sigma, 0.72);
  const dry = clamp((rpm - 0.62) / 0.38, 0, 1);
  const mil = AC.T_MIL * lapse * (1 - 0.25 * M + 0.12 * M * M);
  const full = AC.T_AB * lapse * (1 + 0.25 * M);
  const tDry = AC.T_IDLE * lapse + (mil - AC.T_IDLE * lapse) * Math.pow(dry, 2.2);
  return { dry: tDry, total: tDry + (full - mil) * ab };
}

/* ---------- small vector and quaternion kit ---------- */

const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const norm = (a) => { const l = Math.hypot(a[0], a[1], a[2]) || 1; return [a[0] / l, a[1] / l, a[2] / l]; };

/** Rotate v by quaternion q = [x, y, z, w]. */
export function qrot(q, v) {
  const [x, y, z, w] = q;
  const tx = 2 * (y * v[2] - z * v[1]), ty = 2 * (z * v[0] - x * v[2]), tz = 2 * (x * v[1] - y * v[0]);
  return [v[0] + w * tx + y * tz - z * ty, v[1] + w * ty + z * tx - x * tz, v[2] + w * tz + x * ty - y * tx];
}
export function qmul(a, b) {
  return [
    a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1],
    a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
    a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3],
    a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2]
  ];
}
export function qaxis(axis, ang) {
  const s = Math.sin(ang / 2);
  return [axis[0] * s, axis[1] * s, axis[2] * s, Math.cos(ang / 2)];
}
const qnorm = (q) => { const l = Math.hypot(q[0], q[1], q[2], q[3]) || 1; return [q[0] / l, q[1] / l, q[2] / l, q[3] / l]; };

/** The attitude that faces a compass bearing (degrees), nose `pitch` radians
 *  above the horizon, right wing `roll` radians down. */
export function attitude(bearingDeg, pitch = 0, roll = 0) {
  let q = qaxis([0, 1, 0], -bearingDeg * RAD);
  q = qmul(q, qaxis([1, 0, 0], pitch));
  q = qmul(q, qaxis([0, 0, -1], roll));
  return qnorm(q);
}

/* ---------- the aircraft ---------- */

export class F16 {
  constructor() {
    // What the pilot's hands and feet are doing.
    this.stickX = 0;          // -1..1, right is positive: a roll rate
    this.stickY = 0;          // -1..1, back (pull) is positive: g
    this.rudder = 0;          // -1..1, right pedal positive
    this.lever = 0;           // 0..1, the throttle; AC.MIL is the detent
    this.brake = false;       // wheel brakes
    this.gearDown = false;    // the handle
    this.boards = false;      // the speed-brake switch
    this.ap = false;          // autopilot: altitude hold and attitude hold
    this.apAlt = 0; this.apRoll = 0;
    this.events = [];         // what happened this step, for the caller to act on
    this.reset({});
  }

  /** In the air at `alt` metres and `speed` m/s on a bearing, or with
   *  `ground: true` standing on the wheels with the engine at idle. */
  reset({ x = 0, z = 0, alt = 600, bearing = 0, speed = 180, ground = false, fuel = AC.FUEL_MAX } = {}) {
    this.pos = [x, ground ? 0 : alt, z];
    const f = [Math.sin(bearing * RAD), 0, -Math.cos(bearing * RAD)];
    this.vel = ground ? [0, 0, 0] : [f[0] * speed, 0, f[2] * speed];
    this.q = attitude(bearing, ground ? 0 : 2.5 * RAD, 0);
    this.p = 0; this.qb = 0; this.rb = 0;      // body rates, rad/s: roll, pitch, yaw (nose right)
    this.grounded = ground;
    this.crashed = false; this.crashWhy = '';
    this.vg = 0; this.psi = bearing * RAD; this.theta = 0;
    this.gear = ground ? 1 : 0; this.gearDown = ground;
    this.sb = 0; this.boards = false;
    this.fuel = fuel;
    this.rpm = ground ? 0.62 : 0.9; this.abf = 0;
    this.lever = ground ? 0 : 0.5;
    this.nCmd = 1; this.nz = 1; this.alpha = 0; this.beta = 0;
    this.mach = 0; this.qbar = 0; this.tas = ground ? 0 : speed; this.thrust = 0; this.ff = 0;
    this.ap = false;
    this.stickX = this.stickY = this.rudder = 0; this.brake = false;
    this.t = 0;
    this.events.length = 0;
    this.limit = false;        // the computer is holding the stick: g or angle of attack
    this.over = false;         // at the airframe's speed limit, the engine held back
    this.refuels = 0;
    this._derive();
  }

  get mass() { return AC.EMPTY + this.fuel; }

  /** Advance by dt seconds, in steps no longer than 1/120 s. */
  step(dt) {
    this.events.length = 0;
    if (this.crashed) return;
    let n = Math.max(1, Math.ceil(dt / (1 / 120)));
    const h = dt / n;
    while (n--) {
      this._systems(h);
      if (this.grounded) this._roll(h); else this._fly(h);
      if (this.crashed) break;
    }
    this.t += dt;
    this._derive();
  }

  /* The engine, the gear, the boards, the fuel. */
  _systems(h) {
    const lev = clamp(this.lever, 0, 1);
    const rpmWant = 0.62 + 0.38 * clamp(lev / AC.MIL, 0, 1);
    // A big fan is slow off idle and quick near the top.
    const tau = rpmWant > this.rpm ? 0.7 + 2.6 * (1 - (this.rpm - 0.62) / 0.38) : 1.4;
    this.rpm += (rpmWant - this.rpm) * clamp(h / tau, 0, 1);
    const abWant = lev > AC.MIL && this.rpm > 0.96 ? (lev - AC.MIL) / (1 - AC.MIL) : 0;
    this.abf += clamp(abWant - this.abf, -h / 0.5, h / 0.9);

    const gw = this.gearDown ? 1 : 0;
    if (this.grounded) this.gear = 1;
    else if (this.gear !== gw) {
      this.gear = clamp(this.gear + Math.sign(gw - this.gear) * h / 2.6, 0, 1);
      if (this.gear === gw) this.events.push(gw ? 'gear-down' : 'gear-up');
    }
    this.sb = clamp(this.sb + ((this.boards ? 1 : 0) - this.sb > 0 ? 1 : -1) * h / 1.1, 0, 1);
    if (!this.boards && this.sb < 0.02) this.sb = 0;

    const atm = air(this.pos[1]);
    const sigma = atm.rho / 1.225;
    const M = this.tas / atm.a;
    const T = thrustOf(this.rpm, this.abf, M, sigma);
    // The airframe's limits: 800 knots of dynamic pressure low down, Mach
    // 2.05 up high. The engine's own schedule holds it there.
    const vLim = Math.min(Math.sqrt(2 * AC.Q_LIM / atm.rho), AC.M_LIM * atm.a);
    const hold = sstep(0.97, 1.02, this.tas / vLim);
    this.over = hold > 0.5;
    this.vLim = vLim;
    this.thrust = T.total * (1 - hold);
    // Fuel: 0.74 lb/lbf/h dry, and the burner drinks five times that for what it adds.
    this.ff = (T.dry * 0.0755 + (T.total - T.dry) * 0.36) / 3600 + 0.1;
    this.fuel -= this.ff * h;
    if (this.fuel < 140) {
      // A tanker that is always there. A flight over a town for the pleasure
      // of it should not end in a glide; the gauge still tells the truth
      // about what the burner costs.
      this.fuel = AC.FUEL_MAX;
      this.refuels++;
      this.events.push('refuel');
    }
    this._atm = atm;
  }

  /* In the air. */
  _fly(h) {
    const atm = this._atm, m = this.mass;
    const v = this.vel;
    const V = Math.hypot(v[0], v[1], v[2]);
    const f = qrot(this.q, [0, 0, -1]), r = qrot(this.q, [1, 0, 0]), u = qrot(this.q, [0, 1, 0]);
    const vh = V > 0.5 ? [v[0] / V, v[1] / V, v[2] / V] : f;
    const vf = dot(vh, f), vr = dot(vh, r), vu = dot(vh, u);
    const alpha = Math.atan2(-vu, vf);
    const beta = Math.atan2(vr, Math.hypot(vf, vu));
    const qbar = 0.5 * atm.rho * V * V;
    const M = V / atm.a;
    const flap = this.gear * (1 - sstep(120, 150, V));
    const cl = liftCoef(alpha, flap);
    const cd = dragCoef(cl, alpha, M, this.gear, this.sb, flap);
    let ld = cross(r, vh);
    const ll = Math.hypot(ld[0], ld[1], ld[2]);
    ld = ll > 0.05 ? [ld[0] / ll, ld[1] / ll, ld[2] / ll] : u;
    const qS = qbar * AC.S;
    const L = qS * cl, D = qS * cd, Y = -qS * 1.1 * beta;
    const T = this.thrust;
    const F = [
      ld[0] * L - vh[0] * D + r[0] * Y + f[0] * T,
      ld[1] * L - vh[1] * D + r[1] * Y + f[1] * T - m * G0,
      ld[2] * L - vh[2] * D + r[2] * Y + f[2] * T
    ];
    const acc = [F[0] / m, F[1] / m, F[2] / m];
    this.nz = (dot(acc, u) + G0 * u[1]) / G0;      // what the seat pushes with, along the spine

    // ---- the flight control computer ----
    let nWant, pWant;
    const sy = this.stickY, sx = this.stickX;
    if (this.ap && (Math.abs(sy) > 0.3 || Math.abs(sx) > 0.3)) { this.ap = false; this.events.push('ap-off'); }
    const roll = Math.atan2(-r[1], u[1]);
    if (this.ap) {
      const vzT = clamp(0.4 * (this.apAlt - this.pos[1]), -25, 25);
      nWant = 1 / Math.max(Math.cos(roll), 0.45) + clamp(0.22 * (vzT - v[1]), -0.8, 1.6);
      pWant = clamp(2.2 * (this.apRoll - roll), -0.6, 0.6);
      if (Math.abs(roll) > 70 * RAD || V < 70) { this.ap = false; this.events.push('ap-off'); }
    } else {
      nWant = sy >= 0 ? 1 + (AC.G_MAX - 1) * sy : 1 + (1 - AC.G_MIN) * sy;
      const pMax = AC.P_MAX * clamp(qbar / 9000, 0.2, 1) * (1 - 0.6 * sstep(15, 25, Math.abs(alpha) / RAD));
      pWant = sx * pMax;
    }
    this.nCmd += (nWant - this.nCmd) * clamp(h / 0.16, 0, 1);
    // The g asked for, as an angle of attack, inside the limiter.
    const aMax = lerp(AC.A_MAX, AC.A_MAX_GEAR, this.gear);
    const clReq = this.nCmd * m * G0 / Math.max(qS, 1);
    let aCmd = (clReq - AC.CL0 - AC.CL_FLAP * flap) / AC.CLA;
    this.limit = aCmd > aMax || (sy > 0.96 && this.nz > 8.6);
    aCmd = clamp(aCmd, AC.A_MIN, aMax);
    const auth = clamp(qbar / 2200, 0, 1);         // nothing to push against at no speed
    const Vk = Math.max(V, 30);
    let da = aCmd - alpha;
    if (da > Math.PI) da -= 2 * Math.PI; else if (da < -Math.PI) da += 2 * Math.PI;
    const qWant = clamp(auth * (dot(acc, ld) / Vk + 4.5 * da), -1.5, 1.5);
    const bCmd = -this.rudder * 8 * RAD * auth;
    const rWant = clamp(auth * (dot(acc, r) / Vk + 3.2 * (beta - bCmd)), -1.2, 1.2);
    this.qb += (qWant - this.qb) * clamp(h / 0.07, 0, 1);
    this.rb += (rWant - this.rb) * clamp(h / 0.07, 0, 1);
    this.p += (pWant - this.p) * clamp(h / 0.12, 0, 1);

    // ---- move ----
    // The roll is about the velocity, as the F-16's is at any angle of
    // attack worth the name: rolled about the nose, a wing at twenty
    // degrees would trade its lift for sideslip.
    const ax = V > 30 && vf > 0.5 ? vh : f;
    let q = this.q;
    q = qmul(qaxis(ax, this.p * h), q);
    q = qmul(qaxis(r, this.qb * h), q);
    q = qmul(qaxis(u, -this.rb * h), q);
    this.q = qnorm(q);
    v[0] += acc[0] * h; v[1] += acc[1] * h; v[2] += acc[2] * h;
    this.pos[0] += v[0] * h; this.pos[1] += v[1] * h; this.pos[2] += v[2] * h;
    this.alpha = alpha; this.beta = beta; this.qbar = qbar; this.mach = M; this.tas = V;

    if (this.pos[1] <= 0) this._touch();
  }

  /* The wheels meet the ground: a landing, or not. */
  _touch() {
    const v = this.vel;
    const f = qrot(this.q, [0, 0, -1]), r = qrot(this.q, [1, 0, 0]), u = qrot(this.q, [0, 1, 0]);
    const theta = Math.asin(clamp(f[1], -1, 1));
    const roll = Math.atan2(-r[1], u[1]);
    const why = this.gear < 0.99 ? 'gear'
      : v[1] < -AC.SINK_OK ? 'sink'
      : Math.abs(roll) > 12 * RAD ? 'roll'
      : theta < -3 * RAD || theta > 16 * RAD ? 'pitch'
      : Math.abs(this.beta) > 14 * RAD ? 'drift' : '';
    this.pos[1] = 0;
    if (why) {
      this.crashed = true; this.crashWhy = why;
      this.events.push('crash');
      return;
    }
    this.sinkAtTouch = -v[1];
    this.grounded = true;
    this.vg = Math.hypot(v[0], v[2]);
    this.psi = Math.atan2(v[0], -v[2]);
    this.theta = clamp(theta, 0, AC.THETA_MAX_G);
    this.p = this.qb = this.rb = 0;
    this.ap = false;
    this.events.push('touchdown');
  }

  /* On the wheels. */
  _roll(h) {
    const atm = this._atm, m = this.mass;
    const V = this.vg;
    const qbar = 0.5 * atm.rho * V * V, qS = qbar * AC.S;
    const flap = 1 - sstep(120, 150, V);
    const cl = liftCoef(this.theta, flap);
    const cd = dragCoef(cl, this.theta, V / atm.a, 1, this.sb, flap);
    const L = qS * cl, T = this.thrust;
    const N = Math.max(0, m * G0 - L - T * Math.sin(this.theta));
    const mu = 0.02 + (this.brake ? 0.42 : 0);
    let a = (T * Math.cos(this.theta) - qS * cd) / m;
    const fr = mu * N / m;
    // Friction opposes motion and holds a parked aircraft against idle thrust.
    if (V > 0.05) a -= fr; else if (a > 0) a = Math.max(0, a - fr - (this.brake ? 0 : 0)); else a = 0;
    this.vg = Math.max(0, V + a * h);

    // Rotation: the tail has authority from about a hundred knots.
    const rot = sstep(1400, 3600, qbar);
    const want = clamp(this.stickY, 0, 1) * 13 * RAD * rot;
    this.theta += clamp(want - this.theta, -7 * RAD * h, 9 * RAD * h);
    this.theta = clamp(this.theta, 0, AC.THETA_MAX_G);

    // Nosewheel steering, on the stick and the pedals, never more than the tyres hold.
    const steer = clamp(this.stickX * 0.6 + this.rudder, -1, 1);
    const rate = Math.min(32 * RAD, 0.45 * G0 / Math.max(this.vg, 1)) * clamp(this.vg / 3, 0, 1);
    this.psi += steer * rate * h;

    const f = [Math.sin(this.psi), 0, -Math.cos(this.psi)];
    this.vel = [f[0] * this.vg, 0, f[2] * this.vg];
    this.pos[0] += this.vel[0] * h; this.pos[2] += this.vel[2] * h; this.pos[1] = 0;
    this.q = attitude(this.psi / RAD, this.theta, 0);
    this.alpha = this.theta; this.beta = 0; this.qbar = qbar; this.mach = V / atm.a; this.tas = this.vg;
    this.nz = 1; this.nCmd = 1; this.limit = false;

    if (L * Math.cos(this.theta) + T * Math.sin(this.theta) > m * G0 * 1.005) {
      this.grounded = false;
      this.pos[1] = 0.02;
      this.events.push('liftoff');
    }
  }

  /* The numbers the instruments show. */
  _derive() {
    const f = qrot(this.q, [0, 0, -1]), r = qrot(this.q, [1, 0, 0]), u = qrot(this.q, [0, 1, 0]);
    const atm = air(this.pos[1]);
    this.fwd = f; this.right = r; this.up = u;
    this.heading = (Math.atan2(f[0], -f[2]) / RAD + 360) % 360;
    this.pitch = Math.asin(clamp(f[1], -1, 1)) / RAD;
    this.roll = Math.atan2(-r[1], u[1]) / RAD;
    const V = Math.hypot(this.vel[0], this.vel[1], this.vel[2]);
    this.tas = V;
    this.mach = V / atm.a;
    this.cas = V * Math.sqrt(atm.rho / 1.225);            // equivalent airspeed, near enough what the tape shows
    this.gamma = V > 1 ? Math.asin(clamp(this.vel[1] / V, -1, 1)) / RAD : 0;
    this.track = V > 1 ? (Math.atan2(this.vel[0], -this.vel[2]) / RAD + 360) % 360 : this.heading;
    this.vs = this.vel[1];
    // The engine's dials. The nozzle is wide at idle, closed at military
    // power and wide again in the burner.
    const dry = clamp((this.rpm - 0.62) / 0.38, 0, 1);
    this.noz = this.abf > 0.01 ? 0.25 + 0.7 * this.abf : lerp(0.8, 0.08, sstep(0.1, 0.85, dry));
    this.ftit = 330 + 470 * Math.pow(dry, 0.8) + 160 * this.abf;
    this.oil = 28 + 30 * dry;
    this.speedLimit = this.vLim || 400;
  }
}
