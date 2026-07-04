// Rigid-body physics for the jet bike.
//
// Body frame: Y = up, Z = forward (nose). NOTE: in a right-handed system with
// these choices body +X is the RIDER'S LEFT (screen-left in the chase camera,
// which sits behind the bike looking along +Z). Steering signs below account
// for this; getting it wrong mirrors the controls (that bug shipped once).
// Orientation quaternion maps body -> world. Angular velocity is kept in the
// BODY frame and integrated with Euler's equations (including the gyroscopic
// w x Iw term). Every jet is a force applied at its nozzle position, so
// torque emerges from geometry (tau = r x F) rather than being faked.
//
// Jets:
//   - Rear jet at REAR_POS pushing along +Z (throttle + Space boost).
//   - Three downward nozzles (left / right / center) pushing along body +Y.
//     Z / X / Shift fire RCS-style PULSES on the side / center nozzles: a
//     fixed-duration burst with a fixed thrust = a repeatable "minimum
//     impulse bit" per press (like spacecraft reaction thrusters), with a
//     short refractory gap so mashing can't blend into a continuous burn.
//   - Arrow left/right command a bank angle; the flight controller achieves
//     it by differential thrust on the side nozzles (clamped, so it is
//     physically honest). T toggles the controller off for raw flying.
//   - Arrow up/down command a pitch attitude (nose up / down); the controller
//     holds it, which also sets the angle of attack of the lifting body.
//
// Aerodynamics: quadratic drag per body axis + a weathervane torque that
// aligns the nose with the airflow (like a fin), which turns banked lift
// into coordinated turns. Ground effect adds lift cushion below ~5 m.
import * as THREE from 'three';
import { terrainHeight, terrainNormal, WATER_Y, SPAWN } from './world.js';
import { Wind } from './wind.js';

const clamp = (v, a, b) => Math.min(b, Math.max(a, v));

export const TUNE = {
  mass: 280,               // kg, bike + rider
  g: 9.81,
  maxRear: 4600,           // N, rear jet at full throttle
  boostRear: 3800,         // N, Space (afterburner)
  maxLift: 5200,           // N, all three nozzles at collective = 1
  pulseSide: 480,          // N, RCS pulse on a side nozzle (Z / X)
  pulseCenter: 1800,       // N, RCS pulse on the center nozzle (Shift)
  pulseDur: 0.12,          // s, pulse burn time -> impulse bit = F * dur
  pulseGap: 0.1,           // s, refractory gap between pulses per nozzle
  cdaX: 1.8, cdaY: 2.6, cdaZ: 0.55, // drag area (m^2) per body axis
  liftSlope: 2.4,          // lifting-body CL per rad of angle of attack
  liftArea: 0.62,          // 0.5 * rho * S for the body lift term
  clMax: 0.95,             // stall limit on the lift coefficient
  gust: 70,                // Dryden turbulence gain (70 = nominal, 0 = calm/tests)
  windSpeed: 4.5,          // m/s, prevailing wind (0 disables the whole wind field)
  rotDamp: 1,              // scale on aerodynamic spin damping (0 in tests)
  clearance: 0.55,         // m, skids below center of mass
  crashSpeed: 11,          // m/s into the ground = crash
  rotorH: 60,              // kg m^2/s, turbine rotor angular momentum at full spool
  spoolUp: 0.8,            // s, rear turbine spool-up time constant (JT9D~1.5, F/A-18~0.6)
  spoolDown: 0.5,          // s, spool-down is quicker
  spoolAB: 0.25,           // s, afterburner light-up (fuel into hot exhaust, no rotor inertia)
  spoolLiftTau: 0.3,       // s, small lift turbines answer faster
  turnRateMax: 0.45,       // rad/s, commanded turn rate at full steer (fly-by-wire)
  bankMax: 1.13,           // rad (~65 deg), bank angle limit
  vecMax: 0.26,            // rad (~15 deg), rear nozzle yaw vectoring range
  vecTau: 0.15,            // s, nozzle actuator first-order lag
  pitchRange: 1.05,        // rad (~60 deg), commanded pitch attitude at full stick
};

// The three lift nozzles sit so their combined thrust line passes through the
// center of mass: equal thrust produces zero pitch torque, differential
// left/right thrust produces pure roll, center boost produces pure body-up lift.
const REAR_POS = new THREE.Vector3(0, 0, -1.55);
const NOZ_L = new THREE.Vector3(-0.75, -0.25, 0);
const NOZ_R = new THREE.Vector3(0.75, -0.25, 0);
const NOZ_C = new THREE.Vector3(0, -0.3, 0);

const _v1 = new THREE.Vector3(), _v2 = new THREE.Vector3(), _v3 = new THREE.Vector3();
const _v4 = new THREE.Vector3(), _v5 = new THREE.Vector3(), _v6 = new THREE.Vector3(), _f = new THREE.Vector3();
const _F = new THREE.Vector3(), _T = new THREE.Vector3(), _fw = new THREE.Vector3();
const _q1 = new THREE.Quaternion(), _q2 = new THREE.Quaternion();

// Accumulate a body-frame force applied at a body-frame point:
// world force into F, body torque (r x f) into T.
function addForceBody(F, T, q, fBody, rBody) {
  T.x += rBody.y * fBody.z - rBody.z * fBody.y;
  T.y += rBody.z * fBody.x - rBody.x * fBody.z;
  T.z += rBody.x * fBody.y - rBody.y * fBody.x;
  F.add(_fw.copy(fBody).applyQuaternion(q));
}

export class BikePhysics {
  constructor() {
    this.pos = new THREE.Vector3();
    this.vel = new THREE.Vector3();
    this.quat = new THREE.Quaternion();
    this.angVel = new THREE.Vector3();               // body frame, rad/s (derived from angMom)
    this.angMom = new THREE.Vector3();               // body frame angular momentum L = I*w
    this.I = new THREE.Vector3(170, 155, 48);        // pitch, yaw, roll inertia
    this.time = 0;                                   // sim time, drives gust noise
    this.throttle = 0;                               // rear jet COMMAND, 0..1
    this.collective = 0.5;                           // lift nozzle COMMAND, 0..1.25
    this.spoolRear = 0;                              // rear turbine actual state (first-order lag)
    this.spoolLift = 0;                              // lift turbines actual state
    this.ab = 0;                                     // afterburner actual state, 0..1
    this.steer = 0;                                  // -1..1 (left..right on screen)
    this.pitch = 0;                                  // -1..1 (nose down..nose up)
    this.nozzle = 0;                                 // rad, rear nozzle yaw deflection (actuator state)
    this.input = { boostRear: false };
    this.pulses = { L: 0, R: 0, C: 0 };             // s of burn remaining per RCS nozzle
    this.pulseWait = { L: 0, R: 0, C: 0 };          // refractory time remaining
    this.pulseGrace = 0;                             // s, roll controller stands down after a side pulse
    this.assist = true;                              // flight controller on/off
    this.crashed = false;
    this.crashTimer = 0;
    this._crashEvent = null;
    this.grounded = false;
    this.jetRear = 0;
    this.jetLift = 0;
    this.wind = new Wind(9917, { speed: TUNE.windSpeed });
    this.windVec = new THREE.Vector3();
    this.airSpeed = 0;
    this.reset();
  }

  reset() {
    this.pos.set(SPAWN.x, SPAWN.y + TUNE.clearance + 0.05, SPAWN.z);
    this.vel.set(0, 0, 0);
    this.quat.identity();
    this.angVel.set(0, 0, 0);
    this.angMom.set(0, 0, 0);
    this.throttle = 0;
    // low enough to rest on the pad even with ground effect
    this.collective = 0.35;
    this.spoolRear = 0;
    this.spoolLift = this.collective;   // idling at the resting command
    this.ab = 0;
    this.steer = 0;
    this.pitch = 0;
    this.nozzle = 0;
    this.pulses = { L: 0, R: 0, C: 0 };
    this.pulseWait = { L: 0, R: 0, C: 0 };
    this.pulseGrace = 0;
    this.crashed = false;
  }

  // Fire one RCS pulse ('L' | 'R' | 'C'). Returns true if the nozzle was free.
  firePulse(which) {
    if (this.crashed || this.pulses[which] > 0 || this.pulseWait[which] > 0) return false;
    this.pulses[which] = TUNE.pulseDur;
    if (which !== 'C') this.pulseGrace = 0.35;
    return true;
  }

  consumeCrashEvent() {
    const e = this._crashEvent;
    this._crashEvent = null;
    return e;
  }

  // reason is a code ('water' | 'ground' | 'flip') — the HUD localizes it
  crash(reason) {
    if (this.crashed) return;
    this.crashed = true;
    this.crashTimer = 2.6;
    this._crashEvent = reason;
    this.vel.multiplyScalar(0.1);
    this.angVel.multiplyScalar(0.2);
    this.angMom.multiplyScalar(0.2);
  }

  step(dt) {
    if (this.crashed) {
      this.crashTimer -= dt;
      if (this.crashTimer <= 0) this.reset();
      return;
    }
    const q = this.quat;
    const inv = _q1.copy(q).invert();
    const fwd = _v1.set(0, 0, 1).applyQuaternion(q);
    const right = _v2.set(1, 0, 0).applyQuaternion(q);
    const eRoll = right.y;   // body +X (rider's left) up => banked screen-right
    const ePitch = fwd.y;    // >0 = nose up

    const F = _F.set(0, -TUNE.g * TUNE.mass, 0);     // gravity, world frame
    const T = _T.set(0, 0, 0);                       // torque, body frame

    const groundH = Math.max(terrainHeight(this.pos.x, this.pos.z), WATER_Y);
    const agl = this.pos.y - groundH;
    const airFactor = clamp(1 - (this.pos.y - 500) / 400, 0.25, 1); // thin air up high
    const groundEffect = 1 + 0.35 * clamp(1 - agl / 5, 0, 1);

    // --- engine spool: thrust answers the levers through first-order lags ---
    // dS/dt = (cmd - S)/tau; rotor inertia makes spool-up slower than down,
    // the afterburner lights faster (no rotor to accelerate), and the small
    // lift turbines respond quickest. The player commands, the machine obeys
    // on its own schedule.
    const tauR = this.throttle > this.spoolRear ? TUNE.spoolUp : TUNE.spoolDown;
    const spool0 = this.spoolRear;
    this.spoolRear += (this.throttle - this.spoolRear) * Math.min(1, dt / tauR);
    this.ab += ((this.input.boostRear ? 1 : 0) - this.ab) * Math.min(1, dt / TUNE.spoolAB);
    this.spoolLift += (this.collective - this.spoolLift) * Math.min(1, dt / TUNE.spoolLiftTau);

    // --- turbine rotor gyroscopics ---
    // The spinning rotor carries angular momentum h along its axis (body +Z),
    // proportional to spool. Rotating the airframe drags that axis around, and
    // the reaction torque -w x h couples pitch into yaw: haul the nose up at
    // full spool and it also swings sideways — the machine has a character.
    // Spooling up/down reacts on the frame around the spin axis (-dh/dt).
    const hRot = TUNE.rotorH * this.spoolRear;
    T.x -= this.angVel.y * hRot;
    T.y += this.angVel.x * hRot;
    T.z -= TUNE.rotorH * (this.spoolRear - spool0) / dt;

    // --- rear jet, with yaw thrust vectoring for hover steering ---
    // The nozzle swivels toward body +X with a first-order actuator lag; the
    // deflected thrust at REAR_POS yields yaw torque via r x F. steer > 0
    // (screen-right) needs yaw toward -X, i.e. deflection toward +X.
    const rearT = (this.spoolRear * TUNE.maxRear + this.ab * TUNE.boostRear) * airFactor;
    const vecCmd = this.steer * TUNE.vecMax;
    this.nozzle += (vecCmd - this.nozzle) * Math.min(1, dt / TUNE.vecTau);
    addForceBody(F, T, q, _f.set(Math.sin(this.nozzle) * rearT, 0, Math.cos(this.nozzle) * rearT), REAR_POS);
    this.jetRear = rearT;

    // --- downward nozzles + RCS pulses ---
    // Pulses deliver an exact impulse bit F * pulseDur regardless of frame
    // subdivision (the final partial step applies a prorated force), and are
    // NOT scaled by air density / ground effect: they model a separate
    // high-pressure cold-gas system, so a press is always the same nudge.
    const pu = {};
    for (const k of ['L', 'R', 'C']) {
      const on = Math.min(dt, Math.max(0, this.pulses[k]));
      pu[k] = on > 0 ? (k === 'C' ? TUNE.pulseCenter : TUNE.pulseSide) * (on / dt) : 0;
      if (this.pulses[k] > 0) {
        this.pulses[k] -= dt;
        if (this.pulses[k] <= 0) this.pulseWait[k] = TUNE.pulseGap;
      } else if (this.pulseWait[k] > 0) this.pulseWait[k] -= dt;
    }
    if (this.pulseGrace > 0) this.pulseGrace -= dt;
    const liftScale = airFactor * groundEffect;
    const per = this.spoolLift * TUNE.maxLift / 3;
    const fl = per * liftScale + pu.L;
    const fr = per * liftScale + pu.R;
    const fc = per * liftScale + pu.C;
    addForceBody(F, T, q, _f.set(0, fl, 0), NOZ_L);
    addForceBody(F, T, q, _f.set(0, fr, 0), NOZ_R);
    addForceBody(F, T, q, _f.set(0, fc, 0), NOZ_C);
    this.jetLift = fl + fr + fc;

    // --- wind field: everything aerodynamic below uses AIR-relative velocity ---
    // Dryden gust filters advance here (stochastic ODEs at the physics rate);
    // the prevailing wind + gusts + ridge lift shift the airflow the body sees,
    // so hovering drifts downwind, the vane noses into the wind, and windward
    // slopes carry a gliding bike upward.
    this.wind.gustGain = TUNE.gust / 70;
    this.windVec.copy(this.wind.update(dt, this.airSpeed, this.pos.x, this.pos.z, agl));
    const vAir = _v6.copy(this.vel).sub(this.windVec);

    // --- aerodynamic drag, quadratic per body axis ---
    const vB = _v3.copy(vAir).applyQuaternion(inv);
    const speed = vAir.length();
    this.airSpeed = speed;
    _f.set(
      -0.6 * TUNE.cdaX * Math.abs(vB.x) * vB.x,
      -0.6 * TUNE.cdaY * Math.abs(vB.y) * vB.y,
      -0.6 * TUNE.cdaZ * Math.abs(vB.z) * vB.z
    );
    F.add(_v4.copy(_f).applyQuaternion(q));

    // --- lifting-body aerodynamics: angle of attack turns speed into climb ---
    // The hull acts as a crude wing. Pitch the nose above the flight path and
    // dynamic pressure converts forward speed into body-up lift (plus induced
    // drag), so dives build speed and pull-ups swoop — real flying feel.
    if (vB.z > 5) {
      const aoa = Math.atan2(-vB.y, vB.z);
      const cl = clamp(aoa * TUNE.liftSlope, -TUNE.clMax, TUNE.clMax);
      const qS = (vB.z * vB.z + vB.y * vB.y) * TUNE.liftArea;
      _f.set(0, cl * qS, -0.35 * cl * cl * qS);
      F.add(_f.applyQuaternion(q));
    }

    // --- weathervane: nose follows the airflow ---
    if (speed > 3) {
      _v4.copy(vB).normalize();
      // axis to rotate body +Z toward velocity: z_hat x v_hat = (-vy, vx, 0)
      const q2 = speed * speed;
      // pitch channel scaled by how forward the flow is: in a vertical climb
      // the flow comes from above and an unscaled vane pitches the nose up
      // until the bike slides backwards and swaps ends (old bug).
      const fz = clamp(Math.max(0, _v4.z), 0, 1);
      // pilot pitch authority: a held pitch command relaxes the vane so the
      // airflow doesn't steal the nose back at large angles of attack
      T.x += clamp(-_v4.y * q2 * 0.55, -1200, 1200) * fz * (1 - 0.75 * Math.abs(this.pitch));
      T.y += clamp(_v4.x * q2 * 0.9, -1500, 1500);
    }

    // --- flight controller (differential nozzle thrust, clamped) ---
    // Fly-by-wire steering: steer commands a TURN RATE, and the bank needed
    // follows the coordinated-turn relation tan(phi) = V*omega/g — the same
    // stick input turns tight when slow and gently when fast. At hover the
    // bank fades to zero and the vectored rear nozzle does the turning.
    // steer > 0 must turn SCREEN-right = toward world -X at spawn, which needs
    // body +X (rider's left) up: wantBank has the same sign as steer.
    const hSpd = Math.hypot(this.vel.x, this.vel.z);
    const bankAng = Math.min(Math.atan2(hSpd * Math.abs(this.steer) * TUNE.turnRateMax, TUNE.g), TUNE.bankMax);
    const wantBank = Math.sin(bankAng) * Math.sign(this.steer);
    const rawSide = this.pulses.L > 0 || this.pulses.R > 0 || this.pulseGrace > 0;
    if (this.assist && !rawSide) {
      T.z += clamp(-((eRoll - wantBank) * 1500 + this.angVel.z * 280), -950, 950);
      T.y -= this.angVel.y * 220;   // yaw damper: kills dutch-roll wagging
    } else {
      T.z += this.steer * 750 - this.angVel.z * 50 * TUNE.rotDamp;
    }
    // +X torque pitches the nose down, so the controller drives ePitch toward
    // the commanded attitude (pitch = 1 -> nose up ~60 deg).
    if (this.assist) {
      const wantPitch = Math.sin(this.pitch * TUNE.pitchRange);
      T.x += clamp((ePitch - wantPitch) * 2200 - this.angVel.x * 650, -2200, 2200);
    } else {
      T.x += -this.pitch * 1100;
    }

    // --- rotational damping (air resistance to spin) ---
    const ad = (1 + speed * 0.012) * TUNE.rotDamp;
    T.x -= this.angVel.x * 130 * ad;
    T.y -= this.angVel.y * 160 * ad;
    T.z -= this.angVel.z * 40 * ad;

    // --- ground contact ---
    this.grounded = false;
    if (agl < TUNE.clearance) {
      this.grounded = true;
      const hReal = terrainHeight(this.pos.x, this.pos.z);
      const overWater = hReal < WATER_Y - 0.35;
      if (overWater) { this.crash('water'); return; }
      const n = terrainNormal(this.pos.x, this.pos.z, _v5);
      const vn = this.vel.dot(n);
      const upDot = _v4.set(0, 1, 0).applyQuaternion(q).y;
      if (vn < -TUNE.crashSpeed) { this.crash('ground'); return; }
      if (upDot < 0.25 && speed > 6) { this.crash('flip'); return; }
      const pen = TUNE.clearance - agl;
      const fn = Math.max(0, pen * 42000 + Math.max(0, -vn) * 5200);
      F.addScaledVector(n, fn);
      // skid friction on the tangential velocity
      _v4.copy(this.vel).addScaledVector(n, -vn);
      F.addScaledVector(_v4, -clamp(pen * 8, 0, 1) * 1100);
      // skids resist tipping and spinning
      T.x += ePitch * 2200 - this.angVel.x * 450;
      T.z += -eRoll * 2200 - this.angVel.z * 350;
      T.y -= this.angVel.y * 350;
    }

    // --- integrate: semi-implicit Euler ---
    this.time += dt;
    this.vel.addScaledVector(F, dt / TUNE.mass);
    this.pos.addScaledVector(this.vel, dt);

    // Rotational state is the body-frame angular momentum L = I*w (w = I^-1 L).
    // Applied torques add T*dt; then both the orientation update and the
    // body-frame transport of L use the exact rotation exp(w*dt) instead of a
    // first-order Euler step. In torque-free motion the world-frame angular
    // momentum R*L is conserved to machine precision at any spin rate.
    const w = this.angVel, I = this.I, L = this.angMom;
    L.x += T.x * dt;
    L.y += T.y * dt;
    L.z += T.z * dt;
    const wn = w.length();
    if (wn > 1e-9) {
      const half = 0.5 * wn * dt;
      const s = Math.sin(half) / wn;
      _q2.set(w.x * s, w.y * s, w.z * s, Math.cos(half));
      q.multiply(_q2);           // q <- q * exp(dt/2 * (w, 0))
      q.normalize();
      L.applyQuaternion(_q2.conjugate());  // L in the newly rotated body frame
    }
    w.set(L.x / I.x, L.y / I.y, L.z / I.z);
    const wl = w.length();
    if (wl > 9) {
      w.multiplyScalar(9 / wl);
      L.set(w.x * I.x, w.y * I.y, w.z * I.z);
    }

    // hard floor safety net (never tunnel through terrain)
    const g2 = Math.max(terrainHeight(this.pos.x, this.pos.z), WATER_Y);
    if (this.pos.y < g2 + 0.2) {
      this.pos.y = g2 + 0.2;
      if (this.vel.y < 0) this.vel.y *= -0.1;
    }
  }
}
