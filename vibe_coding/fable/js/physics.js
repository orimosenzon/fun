// Rigid-body physics for the jet bike.
//
// Body frame: X = right, Y = up, Z = forward (nose).
// Orientation quaternion maps body -> world. Angular velocity is kept in the
// BODY frame and integrated with Euler's equations (including the gyroscopic
// w x Iw term). Every jet is a force applied at its nozzle position, so
// torque emerges from geometry (tau = r x F) rather than being faked.
//
// Jets:
//   - Rear jet at REAR_POS pushing along +Z (throttle + Space boost).
//   - Three downward nozzles (left / right / center) pushing along body +Y.
//     Z / X boost the left / right nozzle -> raw roll torque.
//     S boosts the center nozzle -> lift along the bike's tilted up axis.
//   - Arrow left/right command a bank angle; the flight controller achieves
//     it by differential thrust on the side nozzles (clamped, so it is
//     physically honest). T toggles the controller off for raw flying.
//
// Aerodynamics: quadratic drag per body axis + a weathervane torque that
// aligns the nose with the airflow (like a fin), which turns banked lift
// into coordinated turns. Ground effect adds lift cushion below ~5 m.
import * as THREE from 'three';
import { terrainHeight, terrainNormal, WATER_Y, SPAWN } from './world.js';

const clamp = (v, a, b) => Math.min(b, Math.max(a, v));

export const TUNE = {
  mass: 280,               // kg, bike + rider
  g: 9.81,
  maxRear: 3000,           // N, rear jet at full throttle
  boostRear: 2600,         // N, Space
  maxLift: 5200,           // N, all three nozzles at collective = 1
  boostSide: 1400,         // N, Z / X on one side nozzle
  boostCenter: 2800,       // N, S on the center nozzle
  cdaX: 1.8, cdaY: 2.6, cdaZ: 0.55, // drag area (m^2) per body axis
  clearance: 0.55,         // m, skids below center of mass
  crashSpeed: 11,          // m/s into the ground = crash
};

// The three lift nozzles sit so their combined thrust line passes through the
// center of mass: equal thrust produces zero pitch torque, differential
// left/right thrust produces pure roll, center boost produces pure body-up lift.
const REAR_POS = new THREE.Vector3(0, 0, -1.55);
const NOZ_L = new THREE.Vector3(-0.75, -0.25, 0);
const NOZ_R = new THREE.Vector3(0.75, -0.25, 0);
const NOZ_C = new THREE.Vector3(0, -0.3, 0);

const _v1 = new THREE.Vector3(), _v2 = new THREE.Vector3(), _v3 = new THREE.Vector3();
const _v4 = new THREE.Vector3(), _v5 = new THREE.Vector3(), _f = new THREE.Vector3();
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
    this.angVel = new THREE.Vector3();               // body frame, rad/s
    this.I = new THREE.Vector3(170, 155, 48);        // pitch, yaw, roll inertia
    this.throttle = 0;                               // rear jet, 0..1
    this.collective = 0.5;                           // lift nozzles, 0..1.25
    this.steer = 0;                                  // -1..1 (left..right)
    this.input = { boostRear: false, boostL: false, boostR: false, boostC: false };
    this.assist = true;                              // flight controller on/off
    this.crashed = false;
    this.crashTimer = 0;
    this._crashEvent = null;
    this.grounded = false;
    this.jetRear = 0;
    this.jetLift = 0;
    this.reset();
  }

  reset() {
    this.pos.set(SPAWN.x, SPAWN.y + TUNE.clearance + 0.05, SPAWN.z);
    this.vel.set(0, 0, 0);
    this.quat.identity();
    this.angVel.set(0, 0, 0);
    this.throttle = 0;
    // low enough to rest on the pad even with ground effect
    this.collective = 0.35;
    this.steer = 0;
    this.crashed = false;
  }

  consumeCrashEvent() {
    const e = this._crashEvent;
    this._crashEvent = null;
    return e;
  }

  crash(reason) {
    if (this.crashed) return;
    this.crashed = true;
    this.crashTimer = 2.6;
    this._crashEvent = reason;
    this.vel.multiplyScalar(0.1);
    this.angVel.multiplyScalar(0.2);
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
    const eRoll = right.y;   // >0 = banked left
    const ePitch = fwd.y;    // >0 = nose up

    const F = _F.set(0, -TUNE.g * TUNE.mass, 0);     // gravity, world frame
    const T = _T.set(0, 0, 0);                       // torque, body frame

    const groundH = Math.max(terrainHeight(this.pos.x, this.pos.z), WATER_Y);
    const agl = this.pos.y - groundH;
    const airFactor = clamp(1 - (this.pos.y - 500) / 400, 0.25, 1); // thin air up high
    const groundEffect = 1 + 0.35 * clamp(1 - agl / 5, 0, 1);

    // --- rear jet ---
    const rearT = (this.throttle * TUNE.maxRear + (this.input.boostRear ? TUNE.boostRear : 0)) * airFactor;
    addForceBody(F, T, q, _f.set(0, 0, rearT), REAR_POS);
    this.jetRear = rearT;

    // --- downward nozzles ---
    const liftScale = airFactor * groundEffect;
    const per = this.collective * TUNE.maxLift / 3;
    const fl = (per + (this.input.boostL ? TUNE.boostSide : 0)) * liftScale;
    const fr = (per + (this.input.boostR ? TUNE.boostSide : 0)) * liftScale;
    const fc = (per + (this.input.boostC ? TUNE.boostCenter : 0)) * liftScale;
    addForceBody(F, T, q, _f.set(0, fl, 0), NOZ_L);
    addForceBody(F, T, q, _f.set(0, fr, 0), NOZ_R);
    addForceBody(F, T, q, _f.set(0, fc, 0), NOZ_C);
    this.jetLift = fl + fr + fc;

    // --- aerodynamic drag, quadratic per body axis ---
    const vB = _v3.copy(this.vel).applyQuaternion(inv);
    const speed = this.vel.length();
    _f.set(
      -0.6 * TUNE.cdaX * Math.abs(vB.x) * vB.x,
      -0.6 * TUNE.cdaY * Math.abs(vB.y) * vB.y,
      -0.6 * TUNE.cdaZ * Math.abs(vB.z) * vB.z
    );
    F.add(_v4.copy(_f).applyQuaternion(q));

    // --- weathervane: nose follows the airflow ---
    if (speed > 3) {
      _v4.copy(vB).normalize();
      // axis to rotate body +Z toward velocity: z_hat x v_hat = (-vy, vx, 0)
      const q2 = speed * speed;
      T.x += clamp(-_v4.y * q2 * 0.55, -1200, 1200);
      T.y += clamp(_v4.x * q2 * 0.9, -1500, 1500);
    }

    // --- flight controller (differential nozzle thrust, clamped) ---
    const wantBank = -this.steer * 0.75;
    const rawSide = this.input.boostL || this.input.boostR;
    if (this.assist && !rawSide) {
      T.z += clamp(-((eRoll - wantBank) * 1500 + this.angVel.z * 280), -950, 950);
    } else {
      T.z += -this.steer * 750 - this.angVel.z * 50;
    }
    if (this.assist) {
      // +X torque pitches the nose down, so nose-up (ePitch > 0) needs +torque
      T.x += clamp(ePitch * 950 - this.angVel.x * 320, -900, 900);
    }

    // --- rotational damping (air resistance to spin) ---
    const ad = 1 + speed * 0.012;
    T.x -= this.angVel.x * 130 * ad;
    T.y -= this.angVel.y * 160 * ad;
    T.z -= this.angVel.z * 40 * ad;

    // --- ground contact ---
    this.grounded = false;
    if (agl < TUNE.clearance) {
      this.grounded = true;
      const hReal = terrainHeight(this.pos.x, this.pos.z);
      const overWater = hReal < WATER_Y - 0.35;
      if (overWater) { this.crash('שכשוך! נחתת במים'); return; }
      const n = terrainNormal(this.pos.x, this.pos.z, _v5);
      const vn = this.vel.dot(n);
      const upDot = _v4.set(0, 1, 0).applyQuaternion(q).y;
      if (vn < -TUNE.crashSpeed) { this.crash('ריסוק! פגיעה חזקה מדי בקרקע'); return; }
      if (upDot < 0.25 && speed > 6) { this.crash('התהפכות!'); return; }
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
    this.vel.addScaledVector(F, dt / TUNE.mass);
    this.pos.addScaledVector(this.vel, dt);

    const w = this.angVel, I = this.I;
    _v4.set(
      (T.x - w.y * w.z * (I.z - I.y)) / I.x,
      (T.y - w.z * w.x * (I.x - I.z)) / I.y,
      (T.z - w.x * w.y * (I.y - I.x)) / I.z
    );
    w.addScaledVector(_v4, dt);
    const wl = w.length();
    if (wl > 9) w.multiplyScalar(9 / wl);

    // q_dot = 0.5 * q (x) (w, 0)   [body-frame angular velocity]
    _q2.set(w.x, w.y, w.z, 0);
    _q2.multiplyQuaternions(q, _q2);
    q.x += 0.5 * _q2.x * dt;
    q.y += 0.5 * _q2.y * dt;
    q.z += 0.5 * _q2.z * dt;
    q.w += 0.5 * _q2.w * dt;
    q.normalize();

    // hard floor safety net (never tunnel through terrain)
    const g2 = Math.max(terrainHeight(this.pos.x, this.pos.z), WATER_Y);
    if (this.pos.y < g2 + 0.2) {
      this.pos.y = g2 + 0.2;
      if (this.vel.y < 0) this.vel.y *= -0.1;
    }
  }
}
