// The bird-man: a rigid body with two articulated wings, the reflexes that fly them, and legs for the ground.
// The player gives intentions (pull, bank, flap, tuck). The "instinct" layer turns them into what a bird
// actually moves: wing sweep, wing twist, tail angle. The air does the rest (see aero.js).
import { clamp, sstep, qrot, qinvrot, qmul, qexp, qnorm, qFromYPR, yprOf } from './vmath.js';
import { RHO, G, MASS, WING_AREA, spanFactor, halfSpan, newAcc, clearAcc, wingAero, bodyAero } from './aero.js';
import { groundAt, normalAt, windAt } from './world.js';

export const P_MAX = 5200; // W the root gives his arms and chest
const P_PEAK = 2.4 * P_MAX; // instantaneous, at the bottom of the downstroke
const ENERGY = 240e3; // J in a full stamina bar
const PHI_GLIDE = 0.1;
const A_TRIM = 0.065, A_MIN = -0.11, A_MAX = 0.185;
const LIFTOFF = 12.5; // m/s of air over the wings to leave the ground
const N_MAX = 5.5; // g the instinct lets the wings take
export const STAND = 1.0; // height of the body centre when standing

export class Bird {
  constructor() {
    this.p = [0, 0, 0]; this.v = [0, 0, 0]; this.q = [1, 0, 0, 0]; this.om = [0, 0, 0];
    this.L = { phi: PHI_GLIDE, phiDot: 0, fold: 0, sweep: 0.14, twist: 0, eta: 1, unload: 0 };
    this.R = { phi: PHI_GLIDE, phiDot: 0, fold: 0, sweep: 0.14, twist: 0, eta: 1, unload: 0 };
    this.tail = { inc: 0, rud: 0 };
    this.acc = newAcc();
    this.tune = { hz: 1.1, amp: 0.56, etaDown: 0.85, unload: 0.2, kff: 0.05, kd: 2 };
    this.windOn = true; this.wc = [0, 0, 0]; this.wl = [0, 0, 0]; this.wr = [0, 0, 0];
    this.events = [];
    this.placeGround(0, 0, 0);
  }
  _zero() {
    this.om = [0, 0, 0]; this.vi = 0; this.fold = 0; this.latch = false; this.phase = 0; this.amp = 0; this.pscale = 1;
    this.stamina = 1; this.power = 0; this.pAvg = 0; this.iP = 0; this.legs = 0; this.poseUp = 0; this.gload = 1; this.stall = 0;
    this.airspeed = 0; this.alpha = 0; this.beta = 0; this.climb = 0; this.agl = 0; this.bank = 0; this.pitch = 0; this.lift = 0;
    this.gspeed = 0; this.stride = 0; this.airTime = 0; this.crashT = 0; this.maxSpeed = 0; this.landQ = 0;
    this.L.phi = this.R.phi = PHI_GLIDE; this.L.phiDot = this.R.phiDot = 0; this.L.fold = this.R.fold = 0; this.L.twist = this.R.twist = 0;
    this.tail.inc = this.tail.rud = 0; this.uP = 0; this.uR = 0; this.gamLP = 0; this.alphaLP = A_TRIM;
  }
  placeGround(x, z, heading) {
    this._zero();
    this.mode = 'ground'; this.heading = heading; this.poseUp = 1;
    this.p = [x, groundAt(x, z) + STAND, z]; this.v = [0, 0, 0]; this.q = qFromYPR(heading, 0, 0);
  }
  placeAir(p, heading, speed, pitch = 0) {
    this._zero();
    this.mode = 'air'; this.heading = heading; this.fold = 1; this.latch = true; this.L.fold = this.R.fold = 1;
    this.p = [...p]; this.q = qFromYPR(heading, pitch + A_TRIM, 0);
    this.v = qrot(qFromYPR(heading, pitch, 0), [speed, 0, 0]);
  }

  // dt: frame time. inp: { pitch (-1..1, + = nose up), roll (-1..1, + = right), fwd (-1..1), flap, tuck, jump (edge) }
  step(dt, inp, time = 0) {
    if (this.mode === 'ground') return this._ground(dt, inp);
    if (this.mode === 'crash' || this.mode === 'water') { this.crashT += dt; this._settle(dt); return; }
    // sample the air once per frame: at the body and out on each wing, so a thermal lifts one wing first
    const sf = spanFactor(this.fold), p = this.p;
    if (this.windOn) {
      windAt(p[0], p[1], p[2], time, this.wc);
      const r = qrot(this.q, [0, 0, 2.6 * sf]);
      windAt(p[0] + r[0], p[1] + r[1], p[2] + r[2], time, this.wr);
      windAt(p[0] - r[0], p[1] - r[1], p[2] - r[2], time, this.wl);
    }
    // the air stiffens the roll axis with speed: take smaller steps when it does
    const V = Math.hypot(this.v[0] - this.wc[0], this.v[1] - this.wc[1], this.v[2] - this.wc[2]);
    const rate = (260 * V * sf * sf * sf) / (26 + 95 * sf * sf);
    const hmax = clamp(0.45 / (rate + 1), 1 / 2000, 1 / 240);
    const n = Math.ceil(dt / hmax), h = dt / n;
    for (let i = 0; i < n && this.mode === 'air'; i++) this._air(h, inp);
    this.airTime += dt;
  }

  _air(h, inp) {
    const q = this.q, v = this.v, om = this.om, wc = this.wc;
    const vb = qinvrot(q, [v[0] - wc[0], v[1] - wc[1], v[2] - wc[2]]);
    const dwr = qinvrot(q, [this.wr[0] - wc[0], this.wr[1] - wc[1], this.wr[2] - wc[2]]);
    const dwl = qinvrot(q, [this.wl[0] - wc[0], this.wl[1] - wc[1], this.wl[2] - wc[2]]);
    const V = Math.hypot(vb[0], vb[1], vb[2]);
    const alpha = Math.atan2(-vb[1], vb[0]), beta = Math.asin(clamp(vb[2] / Math.max(V, 1), -1, 1));
    const [, pitch, bank] = yprOf(q);
    const g = Math.max(groundAt(this.p[0], this.p[2]), 0), agl = this.p[1] - g;

    // ---- wings: spread, tuck ----
    if (inp.flap && !this.latch) { this.latch = true; }
    const foldT = this.latch && !inp.tuck ? 1 : 0.04;
    const f0 = this.fold;
    this.fold += clamp(foldT - this.fold, -3.6 * h, 5.5 * h);
    if (f0 < 0.5 && this.fold >= 0.5) this.events.push({ type: 'spread', speed: V });
    if (f0 > 0.5 && this.fold <= 0.5) this.events.push({ type: 'tuck', speed: V });
    const sf = spanFactor(this.fold);

    // ---- instinct: pitch by sweep and tail, roll by twist, yaw by the legs ----
    const qb = 0.5 * RHO * V * V, sched = clamp(88 / Math.max(qb, 4), 0.1, 1.5);
    const aMax = A_MAX + 0.16 * this.legs;
    let aCmd = inp.pitch >= 0 ? A_TRIM + inp.pitch * (aMax - A_TRIM) : A_TRIM + inp.pitch * (A_TRIM - A_MIN);
    aCmd += clamp(1 / Math.max(Math.cos(bank), 0.45) - 1, 0, 1.2) * 0.06 * (1 - Math.max(0, -inp.pitch));
    // phugoid damper: a bird does not porpoise. Lean against changes of the flight-path angle (washed out, so a steady climb is free)
    const gam = Math.asin(clamp(v[1] / Math.max(Math.hypot(v[0], v[1], v[2]), 1), -1, 1));
    this.gamLP += (gam - this.gamLP) * Math.min(1, h / 5);
    aCmd -= 1.6 * (A_TRIM + 0.11) * clamp(gam - this.gamLP, -0.3, 0.3) * this.fold;
    // load limit: never ask the wings for more than N_MAX g
    const Seff = WING_AREA * sf * (0.72 + 0.28 * this.fold);
    aCmd = Math.min(aCmd, (N_MAX * MASS * G) / (Math.max(qb, 1) * Seff * 4.68) - 0.088);
    if (this.fold < 0.5) aCmd = inp.pitch * 0.12; // tucked: a dart, steer it gently
    this.alphaLP += (alpha - this.alphaLP) * Math.min(1, h / (0.03 + 0.2 * this.amp));
    const e = aCmd - this.alphaLP, kf = 1 - 0.55 * this.amp;
    this.iP = clamp(this.iP + 2.0 * e * h, -0.35, 0.35);
    const uP = clamp((8 * e * kf - (0.7 + this.tune.kd * this.amp) * om[2]) * sched + this.iP, -1, 1);
    const upright = Math.abs(pitch) < 1.25;
    const pCmd = upright ? clamp(3.2 * (inp.roll * 1.25 - bank), -2.4, 2.4) : inp.roll * 2;
    const uR = clamp(0.55 * (pCmd - om[0]) * sched, -1, 1);
    this.uP = uP; this.uR = uR;
    // keep the stroke's force through the centre of mass: wings reach forward on the way down
    const sweep = 0.14 - 0.25 * uP + this.tune.kff * this.L.phiDot;
    this.tail.inc = -0.42 * uP;
    this.tail.rud = clamp(1.6 * beta + 0.5 * om[1] + 0.35 * om[0], -0.3, 0.3);

    // ---- flapping ----
    const canFlap = inp.flap && this.fold > 0.6;
    const pAvail = this.stamina > 0.01 ? P_MAX : 450;
    this.amp += clamp((canFlap ? 1 : 0) - this.amp, -4 * h, 5 * h);
    // take-off gait: below flying speed he beats faster and deeper, like a swan leaving the water
    const ls = sstep(15.5, 9.5, V), fAmp = this.tune.amp * (1 + 0.32 * ls) * this.amp;
    let phiDot = 0, up = 0;
    if (this.amp > 0.004) {
      this.pscale = clamp(this.pscale + (this.pAvg < pAvail ? 0.6 : -2.5) * h, 0.35, 1);
      const w = 2 * Math.PI * this.tune.hz * (1 + 0.3 * ls) * this.pscale * (this.power > P_PEAK ? P_PEAK / this.power : 1);
      this.phase += w * h;
      phiDot = fAmp * w * Math.cos(this.phase);
      up = Math.max(0, Math.cos(this.phase)) * this.amp;
    } else { this.phase = Math.PI / 2; this.pscale = 1; } // the first stroke is a downstroke
    const phi = PHI_GLIDE + fAmp * Math.sin(this.phase);
    const T = this.tune, eta = phiDot < 0 ? T.etaDown - 0.12 * ls : 1.0;
    const L = this.L, R = this.R;
    L.phi = R.phi = phi; L.phiDot = R.phiDot = phiDot; L.eta = R.eta = eta; L.unload = R.unload = T.unload * up * (1 - 0.85 * ls);
    // roll like a bird: pull one wing in. Less span there means less lift and less roll damping
    const fe = this.fold * (1 - (0.4 - 0.3 * ls) * up);
    R.fold = fe * (1 - 0.5 * Math.max(0, uR)); L.fold = fe * (1 - 0.5 * Math.max(0, -uR));
    L.sweep = R.sweep = sweep;
    const tw = clamp(0.25 - (alpha + 0.11), 0.03, 0.12);
    R.twist = -tw * uR; L.twist = tw * uR;

    // ---- forces ----
    const acc = this.acc; clearAcc(acc);
    wingAero(acc, 1, R, vb, om, dwr, this.vi);
    wingAero(acc, -1, L, vb, om, dwl, this.vi);
    // downwash from momentum theory, weaker near the ground
    const b2 = halfSpan(0.5 * (L.fold + R.fold)), Ad = Math.PI * b2 * b2, hb = agl / (2 * b2);
    const ge = (16 * hb) * (16 * hb) / (1 + (16 * hb) * (16 * hb));
    const viT = (acc.fn / (2 * RHO * Ad * Math.max(Math.hypot(V, this.vi), 1.5))) * ge;
    this.vi += (viT - this.vi) * Math.min(1, h / 0.1);
    this.power = acc.power; this.stall = acc.stall / (2 * 5.2 * sf);
    bodyAero(acc, this.tail, vb, om, this.vi, this.legs);
    this.pAvg += (Math.max(acc.power, 0) - this.pAvg) * Math.min(1, h / 0.5);
    if (acc.power > 0) this.stamina = Math.max(0, this.stamina - (acc.power * h) / ENERGY);
    if (this.amp < 0.1) this.stamina = Math.min(1, this.stamina + h / 40);

    // ---- integrate ----
    const Ixx = 26 + 95 * sf * sf, Iyy = 40 + 95 * sf * sf, Izz = 24;
    const fw = qrot(q, [acc.fx, acc.fy, acc.fz]);
    v[0] += (fw[0] / MASS) * h; v[1] += (fw[1] / MASS - G) * h; v[2] += (fw[2] / MASS) * h;
    om[0] += ((acc.tx - (Izz - Iyy) * om[1] * om[2]) / Ixx) * h;
    om[1] += ((acc.ty - (Ixx - Izz) * om[2] * om[0]) / Iyy) * h;
    om[2] += ((acc.tz - (Iyy - Ixx) * om[0] * om[1]) / Izz) * h;
    this.p[0] += v[0] * h; this.p[1] += v[1] * h; this.p[2] += v[2] * h;
    this.q = qnorm(qmul(q, qexp([om[0] * h, om[1] * h, om[2] * h])));

    // ---- telemetry ----
    this.airspeed = V; this.alpha = alpha; this.beta = beta; this.bank = bank; this.pitch = pitch; this.climb = v[1]; this.agl = agl;
    this.gload += (acc.fy / (MASS * G) - this.gload) * Math.min(1, h / 0.05);
    this.lift = acc.fn;
    if (V > this.maxSpeed) this.maxSpeed = V;
    const legsT = agl < 6 && V < 17 && this.fold > 0.5 ? 1 : 0;
    this.legs += clamp(legsT - this.legs, -2.5 * h, 2.5 * h);
    this.poseUp = Math.max(0, this.poseUp - 2.4 * h);

    // ---- touching the world ----
    const gr = groundAt(this.p[0], this.p[2]), c = 0.3 + 0.7 * this.legs;
    if (gr < -0.5) {
      if (this.p[1] - 0.3 <= 0) { this.mode = 'water'; this.crashT = 0; this.events.push({ type: 'splash', speed: Math.hypot(v[0], v[1], v[2]) }); }
    } else if (this.p[1] - c <= gr) {
      const nrm = normalAt(this.p[0], this.p[2], 9);
      const vn = v[0] * nrm[0] + v[1] * nrm[1] + v[2] * nrm[2];
      const vh = Math.hypot(v[0] - nrm[0] * vn, v[1] - nrm[1] * vn, v[2] - nrm[2] * vn);
      if (vn > -6.5 && vh < 18 && nrm[1] > 0.74) {
        // feet down: run it off
        this.landQ = clamp(1 - Math.max(-vn / 5.5, vh / 13), 0, 1);
        this.mode = 'ground'; this.heading = Math.atan2(-v[2], v[0]); this.gspeed = Math.min(vh, this.amp > 0.5 ? 13 : 10);
        this.p[1] = gr + STAND; this.om = [0, 0, 0]; this.vi = 0;
        this.q = qFromYPR(this.heading, 0, 0);
        this.events.push({ type: 'land', q: this.landQ, vn, vh });
      } else {
        this.mode = 'crash'; this.crashT = 0; this.p[1] = gr + 0.3;
        this.events.push({ type: 'crash', speed: Math.hypot(v[0], v[1], v[2]) });
      }
    }
  }

  _ground(dt, inp) {
    this.heading -= inp.roll * 2.3 * dt * (1 - 0.5 * this.fold);
    const dx = Math.cos(this.heading), dz = -Math.sin(this.heading), p = this.p;
    // wings out on the ground: a flapping run into the wind, like a swan. He leaves the ground when the air over the wings is fast enough to carry him
    const flapRun = inp.flap && this.stamina > 0.02;
    this.latch = flapRun;
    this.fold += clamp((flapRun ? 1 : 0) - this.fold, -3 * dt, 5 * dt);
    this.amp += clamp((flapRun && this.fold > 0.6 ? 1 : 0) - this.amp, -4 * dt, 5 * dt);
    if (this.windOn) windAt(p[0], p[1] + 1, p[2], 0, this.wc);
    const head = -(this.wc[0] * dx + this.wc[2] * dz), air = this.gspeed + head;
    let phiDot = 0, phi = PHI_GLIDE;
    if (this.amp > 0.004) {
      const w = 2 * Math.PI * this.tune.hz * 1.3, A = this.tune.amp * 1.2 * this.amp;
      this.phase += w * dt; phiDot = A * w * Math.cos(this.phase); phi = PHI_GLIDE + 0.25 * this.amp + A * Math.sin(this.phase);
      this.power = P_MAX * 0.9 * this.amp; this.stamina = Math.max(0, this.stamina - (this.power * dt) / ENERGY);
    } else { this.phase = Math.PI / 2; this.power = 0; this.stamina = Math.min(1, this.stamina + dt / 25); }
    this.L.phi = this.R.phi = phi; this.L.phiDot = this.R.phiDot = phiDot; this.L.fold = this.R.fold = this.fold;
    const top = 8.5 + 4.5 * this.amp;
    const target = inp.fwd > 0.1 || flapRun ? top * Math.max(inp.fwd, flapRun ? 1 : 0) : inp.fwd < -0.1 ? -1.5 : 0;
    this.gspeed += clamp(target - this.gspeed, -14 * dt, (9 - 4.5 * this.amp) * dt);
    const nx = p[0] + dx * this.gspeed * dt, nz = p[2] + dz * this.gspeed * dt;
    const h0 = p[1] - STAND, h1 = groundAt(nx, nz), step = Math.abs(this.gspeed * dt) + 1e-6;
    this.poseUp = Math.min(1, this.poseUp + 3 * dt);
    this.legs = 1;
    this.stride += this.gspeed * dt;
    this.airspeed = Math.abs(this.gspeed); this.climb = 0; this.agl = 0; this.gload = 1; this.stall = 0; this.lift = 0;
    this.q = qFromYPR(this.heading, 0, 0);
    const launch = (vy, pitch) => {
      this.mode = 'air'; this.airTime = 0; this.maxSpeed = 0; this.iP = 0; this.gamLP = 0; this.alphaLP = A_TRIM;
      this.v = [dx * this.gspeed, vy, dz * this.gspeed]; this.om = [0, 0, 0];
      this.q = qFromYPR(this.heading, pitch, 0);
    };
    if (this.amp > 0.9 && air > LIFTOFF) { p[1] += 0.1; launch(1.6, 0.16); this.events.push({ type: 'liftoff' }); return; }
    if (inp.jump) { p[1] += 0.05; launch(5.2, 0.12); this.events.push({ type: 'jump' }); return; }
    if ((h1 - h0) / step > 1.4) { this.gspeed = 0; return; } // a wall
    if (h1 < h0 - Math.max(0.45, step * 1.3) || (h0 - h1) / step > 1.1) { p[0] = nx; p[2] = nz; launch(0, 0.12); this.events.push({ type: 'fall' }); return; } // ran off an edge
    p[0] = nx; p[2] = nz; p[1] = h1 + STAND;
    this.v = [dx * this.gspeed, 0, dz * this.gspeed];
    if (h1 < -1.1) { this.mode = 'water'; this.crashT = 0; this.events.push({ type: 'splash', speed: 2 }); }
  }

  // after a crash: slide to a stop
  _settle(dt) {
    const v = this.v, k = Math.exp(-3 * dt);
    v[0] *= k; v[2] *= k; v[1] = 0;
    this.p[0] += v[0] * dt; this.p[2] += v[2] * dt;
    const g = groundAt(this.p[0], this.p[2]);
    this.p[1] = this.mode === 'water' ? Math.max(g + 0.3, -0.25) : g + 0.3;
    this.airspeed = Math.hypot(v[0], v[2]); this.climb = 0; this.power = 0; this.amp = 0;
  }
}
