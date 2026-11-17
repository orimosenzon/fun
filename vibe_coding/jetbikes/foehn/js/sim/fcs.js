// Flight control system of the F-5: a "unified" control law in the spirit of modern STOVL
// fighters. The rider never commands an actuator. The stick asks for a velocity vector
// (turn rate, climb rate, forward speed, side-step) and the computer works out
//   1. the acceleration that takes the bike there,
//   2. the attitude (bank for the turn, a little pitch) and the thrust vector that produce it.
//      There is no wing: every newton of lift is thrust, at any speed,
//   3. how much air the one engine must move and how the diverter shares it between the rear
//      nozzle and the lift nozzles (height first: the rear nozzle gets what the lift does not need),
//   4. a control allocation: 10 effectors (the shares of the 4 lift nozzles, their 4 vanes, two
//      axes of rear-nozzle vectoring) solved every tick by bounded weighted least squares for
//      the wanted moment. The lift valves can only re-distribute the air there is, so their sum
//      is a constraint of the problem.
// Every force still passes through the simulated engine, valves, servos and aerodynamics.
import { clamp, lerp, sstep, dot, cross, len, qmul, qconj, qnorm, qinvrot, qrot, qexp, qFromYPR, qFromBasis, yprOf, m3v, wrapPi, solve } from './vmath.js';
const dot4 = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
import { G, ENG, TILT_MAX, VANE_MAX, VEC_MAX, SHARE_MIN, SHARE_MAX, REAR, SIDE_MAX, SIDE_BLEED, SIDE_AB, aeroWrench, propWrench, ramWrench, liftFrac, mkW, newCmd, thrustOfN } from './bike.js';
import { surfaceAt } from './terrain.js';

const NU = 12, NR = 5, ZERO3 = [0, 0, 0];
// How the bike feels in the hand. These are the numbers to play with when looking for the ride
// that is precise, fun and still has to be learnt (the page takes ?tune and shows them as sliders).
export const FEEL = {
  stick: 0.07,       // s: how fast the computer follows the stick
  rollRate: 3.2,     // rad/s: how fast it may roll into a bank
  rollAcc: 14,       // rad/s^2
  yawRate: 2.0,      // rad/s: the fastest the nose may swing (hover and low speed)
  yawAcc: 6,         // rad/s^2
  bank: 1.27,        // rad: steepest bank (1.27 is 3.4 g)
  kick: 13,          // m/s^2: what the side kick asks of the side nozzles
  strafe: 9,         // m/s: side-step speed in the hover
  gust: 0.5,         // share of the air's side force that the side nozzles cancel on their own
  hairpin: 1,        // 1: a held full turn sheds speed by itself. 0: the rider brakes
  recoil: 1.0,       // m/s: how hard a shot shoves the bike back (and throws its nose up)
  muzzle: 300,       // m/s: speed of the round. Slower means more lead and more drop
  dive: 1,           // 0: the down stick only cuts the lift (the old law). 1: it is an elevator, the nose leads the dive
  // aerobatic mode (the stick turns the bike itself, see acroLaw below)
  acroRoll: 4.0,     // rad/s: roll rate at full stick (a full roll in about a second and a half)
  acroPitch: 1.7,    // rad/s: the fastest the nose may be pulled round (at speed the thrust sets the limit, not this)
  acroYaw: 0.9,      // rad/s: the rudder
  detent: 7,         // degrees: how near to upright, on its side, on its back, level or vertical the bike is drawn into exactly that (0: never)
  net: 1,            // 1: the computer takes the bike back when the ground comes too close to recover later. 0: the rider is alone
};
export const FEEL_DEFAULT = { ...FEEL };
export const FCS_HZ = 120;
export const V_MAX = 95, V_MIN = -8;

export const V_CORNER = 24, HAIR_T0 = 0.5, HAIR_T1 = 1.3;   // corner speed (m/s) and how long full stick must be held
export class FCS {
  constructor(bike) {
    this.bike = bike;
    this.cmd = newCmd();
    this._w0 = mkW(); this._w1 = mkW(); this._a0 = mkW(); this._a1 = mkW(); this._r0 = mkW();
    this._n = { tilt: 0, L: [0, 0, 0, 0], va: [0, 0, 0, 0], R: 0, rp: 0, ry: 0, sd: [0, 0] };
    this.J = new Float64Array(NR * NU);
    this.reset();
  }
  reset() {
    this.turn = 0; this.pitch = 0; this.strafe = 0; this.hold = 0; this.rollNeed = 0;
    this.iV = 0; this.iZ = 0; this.yawRef = null; this.tiltCmd = this.bike.tilt;
    this.idle = true; this.downT = 1; this.low = false; this.obstacle = this.obstacle || null;
    this.acro = false; this.rec = false; this.recT = 0; this.bail = null; this.qRef = null; this.kickA = 0; this.wStick = [0, 0, 0];
    this.tele = { phiDes: 0, thetaDes: 0, aCmd: [0, 0, 0], wb: 0, sat: 0, vzDes: 0, floor: false };
    const c = this.cmd; c.G = 0; c.s.fill(1); c.va.fill(0); c.sd.fill(0); c.kick = 0; c.rp = c.ry = 0; c.da = c.de = c.dr = c.brake = c.ab = 0; c.tilt = this.bike.tilt;
  }

  // ---------------- aerobatic mode ----------------
  // In the ordinary law the stick asks for a velocity and the computer chooses the attitude, so the
  // bike can never be on its back. Here it is the other way round, as in an aerobatic aeroplane:
  // the stick turns the bike itself (roll rate, pitch rate, rudder) and released it holds whatever
  // attitude it was left in. What a wing does for an aeroplane the computer does with thrust: it
  // makes the flight path follow the nose. The lift nozzles push along the bike's own "up" to bend
  // the path with the pitch rate, the side nozzles do the same sideways, and both carry the weight
  // wherever they point so that a straight line stays straight at any angle of roll.
  // What cannot be had stays impossible: the lift nozzles only push one way, so there is nothing
  // to hold the bike up on its back (it falls, and the nose is let down with the path), the pull is
  // limited by the engine (about 4 g), and flying on its side the weight hangs on the side nozzles
  // and their reheat, which heats the core.
  //
  // Leaving it (the rider's key, or the safety net) from an attitude the ordinary law was never
  // made for, the computer first flies the recovery itself: the short way round to upright and a
  // full pull until the bike no longer sinks (this.rec). Only then does the ordinary law take over.
  setAcro(on, why = null) {
    const b = this.bike;
    on = !!on && b.contactN === 0 && !(this.idle && len(b.v) < 5);
    if (on) {
      if (this.acro && !this.rec) return true;
      if (this.rec) return false;                       // not while the computer is still bringing the bike back
      this.acro = true; this.bail = null; this.qRef = b.q.slice(); this.kickA = 0; this.wStick = [0, 0, 0];
      return true;
    }
    if (!this.acro) return false;
    if (!this.rec && why !== 'ground' && why !== 'done') {
      const bx = qrot(b.q, [1, 0, 0]), by = qrot(b.q, [0, 1, 0]);
      this.bail = why;
      if (by[1] < 0.6 || Math.abs(bx[1]) > 0.55 || b.v[1] < -12) { this.rec = true; this.recT = 0; return false; }
    }
    this.acro = this.rec = false;
    this.iV = this.iZ = 0; this.yawRef = levelYaw(b.q); this.low = false;
    return false;
  }
  get aerobatic() { return this.acro && !this.rec; }       // the rider has the bike (not the computer on its way out)
  // the least height over the ground, the trees and the roofs: here, and where the bike is going
  // (as far ahead as the recovery would take, T seconds)
  clearance(T) {
    const b = this.bike, v = b.v, obst = this.obstacle, coarse = len(v) > 12, hBike = b.p[1] - b.mp.cg[1];
    const top = (x, z) => (obst ? Math.max(surfaceAt(x, z), obst(x, z, coarse) + 1.5) : surfaceAt(x, z));
    const slack = Math.max(0, 0.25 * Math.hypot(v[0], v[2]));
    let h = Math.min(b.agl, hBike - top(b.p[0], b.p[2]), hBike - top(b.p[0] + v[0] * 0.6, b.p[2] + v[2] * 0.6), hBike - top(b.p[0] + v[0] * 1.3, b.p[2] + v[2] * 1.3) + slack);
    for (let t = 2; t < T + 0.7; t += 0.7) h = Math.min(h, hBike - top(b.p[0] + v[0] * t, b.p[2] + v[2] * t) + slack);
    return h;
  }
  acroLaw(dt, inp, boost) {
    const b = this.bike, m = b.mp.m, v = b.v, V = len(v);
    const bx = qrot(b.q, [1, 0, 0]), by = qrot(b.q, [0, 1, 0]), bz = qrot(b.q, [0, 0, 1]);
    // The safety net: could the ordinary law still bring the bike upright and stop its fall in the
    // height that is left? When that is about to stop being true it takes the bike back.
    const gy = 0.9 * thrustOfN(1, b.sigma) * ENG.ductEta / m;      // the most the lift nozzles give, as an acceleration
    const [, pt, rl] = yprOf(b.q), steep = Math.abs(pt) > 1.25;
    const vI = qinvrot(b.q, v), u = Math.max(vI[0], 0), uq = Math.max(u, 6), fast = sstep(8, 22, V);
    if (FEEL.net && !inp.noFloor && !this.rec) {
      // the recovery it would fly: roll upright (falling meanwhile), then pull out of the dive
      // (by the sink it will have in a moment: a pull begun on its back turns level flight into a dive within a second)
      const sink = -v[1] + NET.lead * Math.max(0, -b.acc[1]), tRoll = NET.t0 + (steep ? 0 : Math.abs(rl)) / NET.rate, Vh = Math.hypot(v[0], v[2]);
      const sink1 = sink + NET.fall * G * tRoll, V1 = Math.hypot(Vh, sink1);
      const need = sink1 <= 0 ? 0 : Math.max(0, sink * tRoll + 0.5 * NET.fall * G * tRoll * tRoll) + V1 * V1 / (NET.pull * gy - G) * (1 - Vh / Math.max(V1, 1));
      if (this.clearance(tRoll + (sink1 > 0 ? V1 * Math.acos(Math.min(1, Vh / Math.max(V1, 1))) / (NET.pull * gy - G) : 0) + 0.6) < need + NET.h0 + 0.07 * V) this.setAcro(false, 'net');
    }
    // --- the recovery: the computer has the stick ---
    let sTurn = inp.turn, sPitch = inp.pitch, sYaw = inp.strafe || 0, vt = inp.vt;
    if (this.rec) {
      this.recT += dt;
      // done: upright, the nose no higher than the ordinary law ever holds it, and no longer sinking (or no speed left: it knows the hover)
      if ((by[1] > 0.6 && v[1] > -4 && bx[1] > -0.5 && bx[1] < 0.57) || vI[0] < 6 || this.recT > 9) { this.setAcro(false, 'done'); return null; }
      sYaw = 0; boost = false;
      if (bx[1] > 0.87 || (by[1] < 0 && bx[1] > 0.26)) { sTurn = 0; sPitch = 1; }                   // nose high: over the top, and roll out up there
      else if (by[1] >= 0.6 && bx[1] >= 0.57) { sTurn = clamp(-2.2 * rl, -1, 1); sPitch = -1; vt = Math.min(vt, 25); }   // a steep climb, upright: let it fall level
      else { sTurn = steep ? 0 : clamp(-2.2 * rl, -1, 1); sPitch = (steep || Math.abs(rl) < 1.0) && (v[1] < -2 || bx[1] < -0.05) ? 1 : 0; }      // roll upright, then pull out
    } else if (!this.acro) return null;
    // --- the stick: body rates ---
    // The pull is limited to what the path can follow: the most the lift nozzles give (less the
    // weight they carry, plus the weight when the bike is on its back), and the nose may lead the
    // path by A_LIM and no more. Pushing, only gravity bends the path.
    const aPos = gy - G * by[1], aNeg = Math.max(0, G * by[1] - 0.3 * G);
    const alpha = -vI[1] / Math.max(V, 10), beta = vI[2] / Math.max(V, 10);
    const qMax = aPos / uq + lerp(9, 2.5 * (A_LIM - alpha), fast), qMin = -aNeg / uq - lerp(9, 2.5 * (A_LIM + alpha), fast);
    // (The rates asked for change no faster than the bike can follow, so the attitude that is held
    // never runs ahead of it: a roll starts and stops in the same, known, quarter of a second.)
    const slew = (x, to, a) => x + clamp(to - x, -a * dt, a * dt), ws = this.wStick, w0 = ws[0], w1 = ws[1], w2 = ws[2];
    ws[0] = slew(ws[0], clamp(sTurn, -1, 1) * FEEL.acroRoll, ACRO.rollSlew);
    ws[1] = slew(ws[1], -clamp(sYaw, -1, 1) * FEEL.acroYaw, ACRO.yawSlew);
    ws[2] = slew(ws[2], clamp(clamp(sPitch, -1, 1) * FEEL.acroPitch, Math.min(qMin, qMax), qMax), ACRO.pitchSlew);
    let pc = ws[0], qc = ws[2];
    // rudder, and a weathercock: where the side nozzles cannot hold the path, the nose gives way to it
    let rc = ws[1] - clamp(1.2 * beta, -0.5, 0.5) * fast;
    // Notches. A key is all or nothing, and no one lets go of it at exactly the right moment: with the
    // stick released, an attitude that is almost upright, on its side or on its back is drawn into
    // exactly that, and a nose that is almost level or almost vertical likewise.
    const det = FEEL.detent * Math.PI / 180;
    if (det > 0 && !this.rec) {
      const [yr, pr, rr] = yprOf(this.qRef), cp = Math.cos(pr);
      if (Math.abs(sTurn) < 0.05 && Math.abs(ws[0]) < 0.6 && cp > 0.3) {
        const e = wrapPi(Math.round(rr / (Math.PI / 2)) * (Math.PI / 2) - rr);
        if (Math.abs(e) < 1.7 * det) pc += clamp(4 * e, -0.6, 0.6);      // (wider in roll: at 230 degrees a second nobody lets go within a few degrees)
      }
      if (Math.abs(sPitch) < 0.05 && Math.abs(ws[2]) < 0.3 && Math.abs(sYaw) < 0.05 && cp > 0.02) {
        const tgt = Math.abs(pr) < det ? 0 : Math.PI / 2 - Math.abs(pr) < det ? Math.sign(pr) * Math.PI / 2 : null;
        if (tgt !== null) { const k = clamp(3 * (tgt - pr), -0.35, 0.35), w = qinvrot(this.qRef, [Math.sin(yr) * k, 0, Math.cos(yr) * k]); rc += w[1]; qc += w[2]; }
      }
    }
    qc = clamp(qc, Math.min(qMin, qMax), qMax);
    // the attitude that is held: the rates, added up (and never far from where the bike really is)
    this.qRef = qnorm(qmul(this.qRef, qexp([pc * dt, rc * dt, qc * dt])));
    let qe = qmul(qconj(b.q), this.qRef);
    if (qe[0] < 0) qe = [-qe[0], -qe[1], -qe[2], -qe[3]];
    const ang = 2 * Math.acos(Math.min(1, qe[0]));
    if (ang > 0.4) { const k = Math.sin(0.2) / Math.sin(ang / 2); this.qRef = qnorm(qmul(b.q, [Math.cos(0.2), qe[1] * k, qe[2] * k, qe[3] * k])); }
    // --- the force: speed along the nose, and no speed across it ---
    const eV = vt - vI[0];
    this.iV = clamp(this.iV + 0.25 * eV * dt, -1.5, 1.5);
    const aF = boost ? clamp(1.2 * (V_MAX + 30 - vI[0]), -4, 20) : clamp(0.85 * eV + this.iV, -14, 14);
    const fx = aF + G * bx[1];
    const fy = clamp(clamp(-vI[1] / 0.3, -20, 20) + qc * u + G * by[1], 0, gy);
    const fz = clamp(clamp(-vI[2] / 0.35, -16, 16) - rc * u + G * bz[1], -16, 16);
    // on its side the weight hangs on the side nozzles: that takes their reheat
    this.kickA += ((b.abLock ? 0 : clamp((m * Math.abs(fz) - 700) / 1200, 0, 1)) - this.kickA) * Math.min(1, dt * 8);
    this.tele.aCmd = [aF, fy - G * by[1], fz - G * bz[1]]; this.tele.vzDes = v[1]; this.tele.floor = false; this.tele.phiDes = yprOf(this.qRef)[2];
    return { Fb: [m * fx, m * fy, m * fz], qDes: this.qRef, wff: [pc, rc, qc], aff: [(ws[0] - w0) / dt, (ws[1] - w1) / dt, (ws[2] - w2) / dt], eV, kickCmd: this.kickA };
  }

  // inp: { turn (+right), pitch (+climb), strafe (+right), vt (m/s forward), boost, brake }
  update(dt, inp) {
    const b = this.bike, mp = b.mp, m = mp.m, cg = mp.cg, cmd = this.cmd, W = m * G;
    const k1 = 1 - Math.exp(-dt / FEEL.stick);
    this.turn += (clamp(inp.turn, -1, 1) - this.turn) * k1;
    this.pitch += (clamp(inp.pitch, -1, 1) - this.pitch) * k1;
    this.strafe += (clamp(inp.strafe || 0, -1, 1) - this.strafe) * k1;
    // hairpin: a turn held at full stick is a request to come round, not to keep the speed. The
    // computer lets the speed fall to the corner speed, where the lift pipes turn the bike tightest,
    // and gives it back when the stick is released.
    this.hold = Math.abs(inp.turn) > 0.9 ? Math.min(this.hold + dt, 2) : Math.max(0, this.hold - 3 * dt);
    const hair = FEEL.hairpin ? sstep(HAIR_T0, HAIR_T1, this.hold) : 0;
    const vtIn = inp.vt > V_CORNER ? lerp(inp.vt, V_CORNER, hair) : inp.vt;

    let [yaw, pitch, roll] = yprOf(b.q);
    // Nose almost straight up or down (only ever on the way out of an aerobatic figure): the heading
    // of the nose means nothing there. Take the way the bike will face once it has pitched level.
    if (Math.abs(pitch) > 1.2) yaw = levelYaw(b.q);
    const fh = [Math.cos(yaw), 0, -Math.sin(yaw)], rh = [Math.sin(yaw), 0, Math.cos(yaw)];
    const v = b.v, Vf = dot(v, fh), Vr = dot(v, rh), Vu = v[1];
    const Vh = Math.hypot(v[0], v[2]), V = len(v), Va = b.airspeed;
    const qbar = 0.5 * b.rho * Va * Va;
    const onGround = b.contactN > 0.25 * W;
    if (this.acro && b.contactN > 0) this.setAcro(false, 'ground');
    if (!this.acro) this.rec = false;

    // ----- ground idle -----
    const wantsUp = inp.pitch > 0.05 || Math.abs(inp.vt) > 0.5 || Math.abs(inp.strafe || 0) > 0.05 || inp.boost;
    // The engine is cut only once the bike has sat on the ground for a moment. Cutting it at the
    // first touch of one skid left nothing to hold the bike level, and on uneven ground a firm
    // touch-down on one corner rolled it onto its side.
    const down = (onGround || (b.contactN > 0.06 * W && this.pitch < 0.02)) && !wantsUp && V < 4;
    this.downT = down ? this.downT + dt : 0;
    if (down && (this.idle || this.downT > 0.35)) {
      this.idle = true;
      cmd.G = 0; cmd.s.fill(1); cmd.va.fill(0); cmd.sd.fill(0); cmd.kick = 0; cmd.rp = cmd.ry = 0; cmd.da = cmd.de = cmd.dr = 0; cmd.ab = 0; cmd.brake = 0; cmd.tilt = Math.PI / 2;
      this.iV = this.iZ = 0; this.yawRef = yaw; this.tiltCmd = Math.PI / 2;
      return cmd;
    }
    this.idle = false;
    if (this.yawRef === null) this.yawRef = yaw;

    const boost = !!inp.boost && !b.abLock && !inp.brake;
    const ac = this.acro ? this.acroLaw(dt, inp, boost) : null, acro = !!ac;   // (null: the safety net has just taken the bike back)
    let Fb, qDes, wff, eV, kickCmd, aff = ZERO3;
    if (acro) ({ Fb, qDes, wff, aff, eV, kickCmd } = ac);
    else {
      // ----- 1. wanted acceleration -----
      eV = vtIn - Vf;
      this.iV = clamp(this.iV + 0.25 * eV * dt, -1.5, 1.5);
      let aF = boost ? clamp(1.2 * (V_MAX + 30 - Vf), -4, 20) : clamp(0.85 * eV + this.iV, -14, 14);   // with nothing but drag in the way, the overdrive needs a top speed too
      // how hard can it turn? As hard as the thrust allows: the bank at which the lift nozzles at
      // full power still carry the weight (and no more than the rider can take, about 3.4 g).
      const Tavail = thrustOfN(1, b.sigma) * ENG.ductEta;
      const liftCap = 0.9 * Tavail;
      const phiLim = clamp(Math.acos(Math.min(1, W / (0.96 * liftCap))), 0.62, FEEL.bank);
      // (by the speed over the ground, not the speed along the nose: with the side nozzles the nose can
      // swing fast enough to leave the flight path behind, and then it must not be allowed to swing faster still)
      const rateMax = Math.min(FEEL.yawRate, G * Math.tan(phiLim) / Math.max(Vh, 6));
      const yawRateCmd = -this.turn * rateMax;
      const low = 1 - sstep(16, 30, Math.abs(Vf));
      let aR = -Vf * yawRateCmd + low * clamp(1.6 * (this.strafe * FEEL.strafe - Vr), -7, 7);
      // The side kick. In the hover the side key is a side-step; at speed it fires the side nozzles
      // with their reheat: a flat shove to that side, at once, with no bank and no turn of the nose.
      const kickIn = b.abLock ? 0 : (1 - low) * this.strafe, aK = kickIn * FEEL.kick;

      // Diving: the stick is an elevator. The nose goes down first and the rear nozzle drives the bike
      // down along it, faster than a fall (suggested by Reshef Meir: cutting the lift alone left the nose
      // level and took four seconds to lose 50 m). Climbing keeps the old law.
      const dive = Math.max(0, -this.pitch) * sstep(8, 20, Va) * (inp.noDive ? 0 : FEEL.dive);
      let vzDes = this.pitch * Math.max(8, lerp(0.62 * V, Math.min(1.0 * Math.max(Vh, 12), Math.sqrt(Math.max(0, V_MAX * V_MAX - Vh * Vh))), dive));   // (by the speed over the ground: by the whole speed the dive fed itself)
      // radar-altimeter floor: looks 0.6 s ahead; released when the rider deliberately descends slowly (landing)
      const landing = this.pitch < -0.2 && Math.min(Va, V) < 16;      // (by the ground speed too: in a gale the air alone is faster than that)
      const ax = b.p[0] + v[0] * 0.6, az = b.p[2] + v[2] * 0.6;
      // the radar also sees tree tops and roofs (the game supplies them through this.obstacle)
      const obst = this.obstacle;
      const coarse = V > 12, top = (x, z) => (obst ? Math.max(surfaceAt(x, z), obst(x, z, coarse) + 1.5) : surfaceAt(x, z));
      const ax2 = b.p[0] + v[0] * 1.3, az2 = b.p[2] + v[2] * 1.3;
      const hBike = b.p[1] - cg[1];
      const aglAhead = Math.min(hBike - top(ax, az), hBike - top(ax2, az2) + Math.max(0, 0.25 * Vh));
      const hHere = obst ? Math.min(b.agl, hBike - top(b.p[0], b.p[2])) : b.agl;
      const hmin = Math.min(hHere, aglAhead), floorH = 1.6 + 0.07 * Va;
      this.tele.floor = false;
      // Once the rider has brought the bike down below the floor, the floor stays out of the way
      // until they climb, speed up or fly off: letting go of the stick a metre above the pad used
      // to send the bike back up to the floor, and the landing had to be started again.
      if (landing && hmin < floorH + 0.5) this.low = true;
      else if (inp.pitch > 0.05 || inp.boost || V > 12 || hmin > floorH + 4) this.low = false;
      if (!landing && !inp.noFloor) {
        // never sink faster than can still be arrested above the floor; below it, climb
        // (and no faster than the thrust can stop in the room that is left: a held dive used to reach
        // the floor at 20 m/s and go straight through it)
        // (low: only what a slow taxi over the ground needs)
        const room = hmin - (this.low ? 0.6 : floorH), roomL = Math.max(0, room + 0.25 * Math.min(0, Vu));
        // (nose down in a dive the thrust first has to be brought round: count on less, and on the time that takes)
        const elev = !inp.noDive && FEEL.dive > 0, dn = elev ? Math.max(0, Math.sin(-pitch)) : 0, aArr = 14 * (1 - 0.6 * dn), roomD = Math.max(0, roomL + 1.2 * dn * Math.min(0, Vu));
        const f = room < 0 ? -2.2 * room : -Math.min(1.5 * room, lerp(Math.sqrt(2 * aArr * roomD), 1.5 * room, sstep(70, 100, elev ? Math.hypot(Vh, Math.min(V, 0.62 * Vh)) : V))) - 0.5;      // (at racing speed the old rule stands: there the dives are flown on purpose)
        if (f > vzDes) { vzDes = f; this.tele.floor = true; }
      }
      if (onGround && this.pitch <= 0.05) vzDes = Math.min(vzDes, 0);
      if (down) vzDes = Math.min(vzDes, -0.4);                  // touching, and not asked to leave: settle
      // automatic landing flare: the lower, the slower
      // (the height is taken where the bike will be in half a second: the thrust answers late, and
      // without the lead the skids met the ground three times faster than the flare asks)
      if (!inp.noFloor && Math.min(Va, V) < 25) vzDes = Math.max(vzDes, -(0.6 + 0.48 * Math.max(0, (landing ? b.agl : hmin) + 0.5 * Math.min(0, Vu))));
      const eZ = vzDes - Vu;
      this.iZ = clamp(this.iZ + 0.35 * eZ * dt, -2, 2);
      // jet-borne, thrust is also the attitude control power: never let it fall too low
      // (nose down, the rear nozzle can push downward too: up to its thrust times the sine of the dive)
      const noseDn = Math.max(0, Math.sin(-pitch));
      const diveNow = this.tele.floor || landing || this.low ? 0 : dive;   // (the radar or a landing has the bike: no dive)
      let aU = clamp(1.9 * eZ + this.iZ, -(1.0 + 1.3 * noseDn * Math.min(1, 3 * diveNow)) * G, 1.9 * G);

      // what can the airframe actually deliver? Protect height first, then the turn, then speed.
      const need = m * Math.hypot(aR, aU + G);
      if (need > liftCap) {
        const k = Math.sqrt(Math.max(0, liftCap * liftCap / (m * m) - (aU + G) * (aU + G)));
        if (Math.abs(aR) > k) aR = Math.sign(aR) * Math.max(k, 0.25 * Math.abs(aR));
      }
      // bank limit (also keeps roll reversals on the short way round, through wings-level)
      const aRmax = Math.tan(phiLim) * Math.max(aU + G, 0.35 * G);
      aR = clamp(aR, -aRmax, aRmax);
      const f = [aF * fh[0] + (aR + aK) * rh[0], aU + G, aF * fh[2] + (aR + aK) * rh[2]];   // specific force, world (the bank is made for aR; the kick is the side nozzles' own)
      this.tele.aCmd = [aF, aU, aR]; this.tele.vzDes = vzDes; this.tele.wb = 0;

      // ----- 2. attitude -----
      const phiDes = clamp(Math.atan2(aR, Math.max(aU + G, 0.35 * G)), -phiLim, phiLim);
      // jet-borne: lean back to brake (flare), a touch forward to accelerate; thrust tilt does the rest
      // (and stay level close to the ground so that all four skids arrive together)
      const lvl = inp.noFloor ? 1 : clamp(0.12 + 0.11 * b.agl, 0, 1);
      // (leaning back is for the last of a stop: at speed it would only fly the bike upward on its own belly,
      // and there the air brake and the drag do the work)
      const thLow = clamp(-0.5 * aF / G, -0.20, 0.42) * (aF > 0 ? 1 - 0.6 * sstep(8, 30, Vf) : 1 - 0.85 * sstep(22, 55, Vf)) * lvl;
      // the nose follows the flight path, so the bike climbs and dives like a bike on a hill
      const gamma = Math.atan2(Vu, Math.max(Vh, 6));
      // (diving, the nose leads the path: it goes to where the stick asks the path to go, at once)
      const gammaCmd = Math.atan2(Math.min(vzDes, 0), Math.max(Vh, 6));
      const thetaH = clamp(thLow + lerp(gamma, Math.min(gamma, 1.15 * gammaCmd), Math.min(1, 1.5 * diveNow)) * sstep(10, 30, Va), -1.0, 1.0);
      const beta = b.vAirB[2] / Math.max(Va, 12);
      const yawRateH = yawRateCmd - clamp(2.2 * beta, -1.1, 1.1) * sstep(8, 26, Va) * (1 - Math.abs(kickIn));   // (during a kick the nose holds its heading: the bike crabs)
      this.yawRef = yaw + clamp(wrapPi(this.yawRef + yawRateH * dt - yaw), -0.3, 0.3);
      qDes = qFromYPR(this.yawRef, thetaH, phiDes);
      Fb = qinvrot(b.q, [m * f[0], m * f[1], m * f[2]]);
      wff = qinvrot(b.q, [0, yawRateH, 0]);
      kickCmd = clamp(Math.abs(kickIn) * 1.5, 0, 1);
      this.tele.phiDes = phiDes;
    }
    // The lift nozzles are the only roll control there is (no ailerons), and their roll moment
    // is their thrust times the 0.2 m they stand off the centre line. So they never run dry, and
    // they are opened further whenever the roll moment asked for needs it (a side wind on the
    // rider's back at speed, for instance).
    // (aerobatic: the side nozzles stand above the centre of gravity, so carrying the weight on them
    // rolls the bike, and only the lift nozzles can answer that: they are kept open for it)
    const Tmin = Math.max(0.3 * W, Math.min(1.6 * W, (Math.abs(this.rollNeed) + (acro ? ACRO.sideArm * (Math.abs(b.sd[0]) + Math.abs(b.sd[1])) : 0)) / 0.10));
    const thetaDes = Math.asin(clamp(2 * (qDes[0] * qDes[3] + qDes[1] * qDes[2]), -1, 1));   // for telemetry
    let qe = qmul(qconj(b.q), qDes);
    if (qe[0] < 0) qe = [-qe[0], -qe[1], -qe[2], -qe[3]];
    this.tele.attErr = 2 * Math.acos(Math.min(1, qe[0]));
    // rate and angular-acceleration limits keep the demand inside what the effectors can give
    const wDes = [clamp(5.0 * 2 * qe[1], -FEEL.rollRate, FEEL.rollRate) + wff[0], clamp(3.2 * 2 * qe[2], -1.2, 1.2) + wff[1], clamp(4.6 * 2 * qe[3], -1.3, 1.3) + wff[2]];
    const aL = [acro ? ACRO.rollAcc : FEEL.rollAcc, FEEL.yawAcc, acro ? 9 : 4];
    const wdot = [clamp(11 * (wDes[0] - b.w[0]) + aff[0], -aL[0], aL[0]), clamp(6.5 * (wDes[1] - b.w[1]) + aff[1], -aL[1], aL[1]), clamp(10 * (wDes[2] - b.w[2]) + aff[2], -aL[2], aL[2])];
    const Iw = m3v(mp.I, b.w), gyro = cross(b.w, Iw);
    const tauDes = m3v(mp.I, wdot); tauDes[0] += gyro[0]; tauDes[1] += gyro[1]; tauDes[2] += gyro[2];
    // the spinning spool: pitching the bike makes it want to yaw and the other way round
    const Hx = ENG.rotorI * ENG.omegaMax * b.eng.N; tauDes[1] += b.w[2] * Hx; tauDes[2] -= b.w[1] * Hx;
    this.tele.thetaDes = thetaDes;

    // ----- 3. thrust vector: how much gas, and where it goes -----
    const surf = b;
    const A0 = aeroWrench(b.vAirB, b.w, b.rho, surf, cg, this._a0);
    const R0 = ramWrench(b.vAirB, b.w, b.eng.Tdry, cg, this._r0);
    let Fx = Fb[0] - A0.F[0] - R0.F[0];
    // thrust fills in whatever lift is missing right now
    const Fy = Math.max(Fb[1] - A0.F[1] - R0.F[1], Tmin);
    // one engine: height comes first, the rear nozzle gets the gas the lift pipes do not need
    const ab = 1 + ENG.abGain * b.eng.ab, eta = ENG.ductEta, Gmax = thrustOfN(1, b.sigma);
    const cant = Math.cos(REAR.cant);
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
    const Gmain = clamp((Fx * ct + Fy * st) / Math.max(per, 0.3), Tmin, Gmax);
    // the side nozzles bleed the same engine: ask it for their air too. Height comes first here
    // as well: with the engine flat out they get only a little (more when the rider kicks).
    const sab = 1 + SIDE_AB * b.kick;
    cmd.kick = kickCmd;
    cmd.G = clamp(Gmain + (Math.abs(b.sd[0]) + Math.abs(b.sd[1])) / sab, Tmin, Gmax);
    const sideGas = clamp(Gmax - Gmain, (0.015 + 0.10 * cmd.kick) * Gmax, Math.max(SIDE_BLEED * b.eng.Tdry, 0.05 * Gmax));
    const sideCap = Math.min(SIDE_MAX, 0.6 * sab * sideGas);

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
    for (let i = 0; i < 2; i++) { n.sd[i] = b.sd[i]; u0[10 + i] = b.sd[i]; lo[10 + i] = Math.min(-sideCap - b.sd[i], 0); hi[10 + i] = Math.max(sideCap - b.sd[i], 0); }
    // Jacobian by finite differences through the same force model the physics uses.
    // rows: [sum of lift thrust, side force Fz, Mx, My, Mz]
    const J = this.J;
    const row = (w, k, s, col, sum) => { J[col] = sum; J[NU + col] = (w.F[2] - k.F[2]) / s; J[2 * NU + col] = (w.Tq[0] - k.Tq[0]) / s; J[3 * NU + col] = (w.Tq[1] - k.Tq[1]) / s; J[4 * NU + col] = (w.Tq[2] - k.Tq[2]) / s; };
    const P0 = propWrench(n, cg, this._w0);
    for (let i = 0; i < 4; i++) {
      n.L[i] += 20; row(propWrench(n, cg, this._w1), P0, 20, i, 1); n.L[i] -= 20;
      n.va[i] += 0.01; row(propWrench(n, cg, this._w1), P0, 0.01, 4 + i, 0); n.va[i] -= 0.01;
    }
    n.rp += 0.01; row(propWrench(n, cg, this._w1), P0, 0.01, 8, 0); n.rp -= 0.01;
    n.ry += 0.01; row(propWrench(n, cg, this._w1), P0, 0.01, 9, 0); n.ry -= 0.01;
    for (let i = 0; i < 2; i++) { n.sd[i] += 20; row(propWrench(n, cg, this._w1), P0, 20, 10 + i, 0); n.sd[i] -= 20; }
    // preferred positions: equal shares, vanes, nozzle and surfaces centred
    const pref = this._pref || (this._pref = new Float64Array(NU));
    for (let i = 0; i < 4; i++) { pref[i] = Ltot * 0.25 - u0[i]; pref[4 + i] = -u0[4 + i]; }
    for (let i = 8; i < NU; i++) pref[i] = -u0[i];
    const tq = [tauDes[0] - A0.Tq[0] - R0.Tq[0], tauDes[1] - A0.Tq[1] - R0.Tq[1], tauDes[2] - A0.Tq[2] - R0.Tq[2]];
    this.rollNeed += (tq[0] - this.rollNeed) * 0.5;
    // side force: what the wanted acceleration needs along the bike's own right, less what the
    // nozzles give already. This is the side nozzles' work: the lead into a turn while the bank
    // is still building, the kick, and part of a gust.
    const FzDes = Fb[2] - (acro ? 1 : FEEL.gust) * (A0.F[2] + R0.F[2]);
    const e = [0, FzDes - P0.F[2], tq[0] - P0.Tq[0], tq[1] - P0.Tq[1], tq[2] - P0.Tq[2]];
    const du = wls(J, e, WV, WU, pref, lo, hi, null);
    for (let i = 0; i < 4; i++) { cmd.s[i] = Math.max(1e-3, (u0[i] + du[i]) / b.gfx[i]); cmd.va[i] = u0[4 + i] + du[4 + i]; }
    cmd.rp = u0[8] + du[8]; cmd.ry = u0[9] + du[9];
    cmd.sd[0] = u0[10] + du[10]; cmd.sd[1] = u0[11] + du[11];
    this.tele.sat = wls.sat; this.tele.tau = tauDes; this.tele.tq = tq; this.tele.F = [Fx, Fy];
    return cmd;
  }
}

// the nose may lead the flight path by this much in aerobatic mode (rad)
const A_LIM = 0.2;
export const ACRO = { rollAcc: 30, sideArm: 0.12, rollSlew: 15, pitchSlew: 7, yawSlew: 5 };
// the safety net of aerobatic mode, as a model of the recovery it will fly: time to start, roll rate (rad/s),
// share of free fall while rolling, share of the full pull that is counted on, and the height kept in hand (m)
export const NET = { t0: 0.35, rate: 2.6, fall: 0.8, pull: 0.75, h0: 4, lead: 0.5 };
// heading the bike will have once it has pitched level by the short way (the nose itself has none when it points straight up)
export function levelYaw(q) {
  const f = qrot(q, [1, 0, 0]), u = qrot(q, [0, 1, 0]);
  return Math.atan2(-(f[2] - u[2] * f[1]), f[0] - u[0] * f[1]);
}

// row weights: [sum of lift (N), side force (N), Mx, My, Mz (N m)]
const WV = [0.5, 3 / 1000, 6 / 200, 2.5 / 200, 6 / 200];
// effector weights (cost of leaving the preferred position): lift thrust per N, angles per rad
const WU = [4e-4, 4e-4, 4e-4, 4e-4, 1.2, 1.2, 1.2, 1.2, 0.3, 0.3, 5e-4, 5e-4];

const _A = new Float64Array(NU * NU), _b = new Float64Array(NU), _du = new Float64Array(NU), _fix = new Int8Array(NU), _res = new Float64Array(NR);
// Bounded weighted least squares by an active set: solve, freeze the effector that overshoots
// its limit the most at that limit, and solve again for the rest. (Freezing everything that
// overshoots at once is cheaper but wrong when effectors are coupled: the others have already
// been set to cancel a side effect that the clipped one will not produce.)
function wls(J, e, wv, wu, pref, lo, hi, mask) {
  const du = _du, fix = _fix;
  for (let i = 0; i < NU; i++) { du[i] = 0; fix[i] = mask && !mask[i] ? 1 : 0; }
  let sat = 0;
  for (let pass = 0; pass < NU; pass++) {
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
    let worst = -1, over = 1e-9;
    for (let a = 0; a < n; a++) {
      const c = idx[a], x = _b[a], span = hi[c] - lo[c] + 1e-9;
      du[c] = x;
      const o = x < lo[c] ? (lo[c] - x) / span : x > hi[c] ? (x - hi[c]) / span : 0;
      if (o > over) { over = o; worst = c; }
    }
    if (worst < 0) break;
    du[worst] = du[worst] < lo[worst] ? lo[worst] : hi[worst]; fix[worst] = 1; sat++;
  }
  for (let c = 0; c < NU; c++) du[c] = clamp(du[c], lo[c], hi[c]);
  wls.sat = sat;
  return du.slice();
}
