/* The jet bike - the third aircraft of the flight mode (18/9/2026).
 *
 * This is the FLYING HOG / ALPINIST from the fable project, brought over
 * whole: the rigid-body flight model, the wind it flies through, the bike
 * and rider mesh, the exhaust plumes and sparks, and the crash. Ori asked
 * for the bike as it is there - "the physics, the controls, the look, all of
 * it" - and so the code below is fable's, with three changes and no more:
 *
 *   - The ground is flat and at zero. The map is the terrain here, and in
 *     the flight mode it is drawn without relief (see explore.js for why),
 *     so the physics agrees with what is on the screen: y = 0 is the ground
 *     everywhere, there is no water, and ridge lift has no ridge.
 *   - Three.js is fetched on the first take-off, not on page load. Nobody
 *     who came for a trail should pay 650 KB for a motorcycle they may never
 *     ride; the promise is kept and the second flight is instant.
 *   - The scene holds only the bike. The world is MapLibre's satellite
 *     picture underneath a transparent canvas, and the camera in here is
 *     rebuilt every frame from the map's own camera, so the two agree to the
 *     pixel. explore.js decides where the map looks; this file draws what
 *     is in front of it.
 *
 * Coordinates, in the bike's own frame of the world: x east, y up, z south,
 * in metres from the take-off point. That is the right-handed frame that
 * keeps fable's body convention (Y up, Z forward) intact, and explore.js
 * converts to and from latitude and longitude at the edges.
 *
 * Everything else - the body-frame convention in which +X is the rider's
 * LEFT, the fly-by-wire steering, the auto-lift, the spool lags, the
 * gyroscopic rotor, the RCS pulses, the Dryden turbulence - is described in
 * fable's physics.js and README and is not repeated here beyond the
 * comments that travelled with the code. */
'use strict';

const Bike = (() => {

  const THREE_URL = 'https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.min.js';
  let THREE = null;
  let loading = null;
  let lib = null;       // the classes and builders, made once THREE is here

  /** Fetch Three.js, once. Resolves to true when the bike can be built. */
  function load() {
    if (!loading) {
      loading = import(THREE_URL).then((m) => { THREE = m; lib = defs(); return true; });
    }
    return loading;
  }

  const clamp = (v, a, b) => Math.min(b, Math.max(a, v));

  /* ---------- tuning ----------
   *
   * fable's TUNE, with four additions and one change, all in the flight
   * computer and all for the same reason: fable flies over mountains at
   * hundreds of metres, this flies over rooftops at thirty, and what was a
   * dramatic loss of height there is the ground here.
   *
   * Measured before the changes: a full-stick turn at 60 m/s from 66 m
   * banked to 65 degrees, the nose fell to 45 degrees below the horizon
   * within two seconds, and the bike was on the ground in four. Three
   * things conspire. In a bank the yaw about the tilted axis takes the
   * nose down in the world (by the yaw rate times the sine of the bank),
   * the auto-lift then flies the path after the drooping nose, and the
   * weathervane pulls the nose after the falling path: a spiral. So the
   * pitch loop gets the feed-forward a coordinated turn needs
   * (turnCoord), the auto-lift follows the pitch the rider asks for and
   * not the one the coupling produced, and the bank is capped where the
   * nozzles can still carry the weight at full collective (bankMax, 50
   * degrees instead of 65). The bike still banks, still turns as hard as
   * it can hold, and no longer falls out of the turn.
   *
   * Measured after those three, still full throttle and full stick: the
   * yaw rate ran to 3.5 rad/s and the bank wandered between 20 and 40
   * degrees under a controller asking for 50. That is the rear nozzle's
   * thrust vectoring, 1,800 N m of yaw at full throttle, which is what
   * turns the bike at a hover and is a skid at speed; with the computer
   * on it now fades out between vecFadeLo and vecFadeHi, and the bank does
   * the turning above that, as it does on anything with wings.
   *
   * The last addition is a brake: S past idle sits the rider up into the
   * wind (brakeK on the frontal drag area), because a bike that coasts
   * for a minute at 0.55 m² cannot stop to look at a photograph, and
   * looking is what this mode is for. This is the first place to look to
   * change how the bike feels. */
  const TUNE = {
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
    bankMax: 0.87,           // rad (~50 deg), bank angle limit (fable: 1.13, see above)
    vecMax: 0.26,            // rad (~15 deg), rear nozzle yaw vectoring range
    vecTau: 0.15,            // s, nozzle actuator first-order lag
    pitchRange: 1.05,        // rad (~60 deg), commanded pitch attitude at full stick
    climbRate: 8,            // m/s, vertical speed commanded by E/D at full input
    vyGain: 1.6,             // 1/s, auto-lift vertical-speed loop gain
    turnCoord: 1,            // 0..1, how much of a coordinated turn's pull the computer supplies
    vecFadeLo: 8,            // m/s: below this the nozzle vectoring is all there, hover steering
    vecFadeHi: 30,           // and above this it is gone; the bank does the turning
    brakeK: 5,               // the frontal drag area, times this, with the rider sat up (S at idle)
    // The chase camera. fable's numbers, plus a floor on how far below the
    // horizon it looks, because the map beneath cannot look up.
    camBack: 8.5,            // m behind the bike, plus camBackV per m/s
    camBackV: 0.05,
    camUp: 2.6,              // m above it, plus camUpV per m/s
    camUpV: 0.012,
    camDip: 8,               // degrees below the horizon the camera must at least look
    camFollow: 5,            // 1/s, how quickly the camera closes on where it should be
    // The rider's-eye camera: where the eyes are (fable's spot, just ahead
    // of the visor and below it; further back the inside of the helmet is
    // a black wall across the top of the screen), and how far below the
    // nose they look. fable looks straight ahead with a 75-degree field;
    // this window is under half that, and the bars and the road are below
    // it, so the rider looks down the road.
    eyePos: [0, 0.95, 0.3],
    eyeDip: 16,
    grain: 0.22,             // the ground grain's opacity at a parked bike; 0 turns it off
  };

  const STEP = 1 / 120;      // physics sub-step, seconds

  /** The ground. Flat, at zero, everywhere - see the header. */
  const groundAt = () => 0;

  /* ---------- seeded noise (fable's noise.js) ---------- */
  function mulberry32(seed) {
    let a = seed >>> 0;
    return function () {
      a |= 0; a = (a + 0x6D2B79F5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  /* Everything that needs THREE is defined in here, after it has loaded. */
  function defs() {

    /* ---------- wind (fable's wind.js) ----------
     *
     * A seeded prevailing wind plus MIL-F-8785C Dryden turbulence: Gaussian
     * white noise through shaping filters, a first-order ODE for the
     * along-wind component and second-order ones for the crosswind and
     * vertical, integrated at the physics rate. Ridge lift is gone with the
     * ridges. */
    class Wind {
      constructor(seed = 77, opts = {}) {
        this.rand = mulberry32(seed);
        const dir = opts.dir ?? this.rand() * Math.PI * 2;
        this.speed = opts.speed ?? 3 + this.rand() * 5;
        this.W = new THREE.Vector3(Math.sin(dir) * this.speed, 0, Math.cos(dir) * this.speed);
        this.u = 0;
        this.v = 0; this.vd = 0;
        this.w = 0; this.wd = 0;
        this.gustGain = 1;
        this.vec = new THREE.Vector3();
        this.gustMag = 0;
      }

      gauss() {
        const a = Math.max(this.rand(), 1e-12);
        return Math.sqrt(-2 * Math.log(a)) * Math.cos(2 * Math.PI * this.rand());
      }

      update(dt, V, x, z, agl) {
        const W = this.W;
        if (this.speed < 0.01) { this.vec.set(0, 0, 0); this.gustMag = 0; return this.vec; }
        const hft = Math.min(1000, Math.max(10, agl * 3.281));
        const coef = Math.pow(0.177 + 0.000823 * hft, 1.2);
        const Lw = Math.max(3, agl);
        const Lu = Math.max(10, agl / coef);
        const sw = 0.1 * this.speed * this.gustGain;
        const su = sw / Math.pow(0.177 + 0.000823 * hft, 0.4);
        const Vc = Math.max(5, V);
        const eta = () => this.gauss() / Math.sqrt(dt);
        this.u += (-(Vc / Lu) * this.u + su * Math.sqrt(2 * Vc / (Math.PI * Lu)) * eta()) * dt;
        const step2 = (state, L, sig) => {
          const b = 2 * L / Vc, a = 2 * Math.sqrt(3) * L / Vc;
          const K = sig * Math.sqrt(L / (Math.PI * Vc));
          state.d += ((eta() - state.x - 2 * b * state.d) / (b * b)) * dt;
          state.x += state.d * dt;
          return K * (state.x + a * state.d);
        };
        const sv = { x: this.v, d: this.vd };
        const gv = step2(sv, Lu, su);
        this.v = sv.x; this.vd = sv.d;
        const swS = { x: this.w, d: this.wd };
        const gw = step2(swS, Lw, sw);
        this.w = swS.x; this.wd = swS.d;
        const inv = 1 / this.speed;
        const ux = W.x * inv, uz = W.z * inv;
        const gx = ux * this.u + uz * gv;
        const gz = uz * this.u - ux * gv;
        this.gustMag = Math.hypot(gx, gw, gz);
        this.vec.set(W.x + gx, gw, W.z + gz);
        return this.vec;
      }
    }

    /* ---------- the flight model (fable's physics.js) ----------
     *
     * Body frame: Y = up, Z = forward (nose). NOTE: in a right-handed system
     * with these choices body +X is the RIDER'S LEFT (screen-left in the
     * chase camera, which sits behind the bike looking along +Z). Steering
     * signs below account for this; getting it wrong mirrors the controls
     * (that bug shipped once). Orientation quaternion maps body -> world.
     * Angular velocity is kept in the BODY frame and integrated with Euler's
     * equations (including the gyroscopic w x Iw term). Every jet is a force
     * applied at its nozzle position, so torque emerges from geometry
     * (tau = r x F) rather than being faked. */

    const REAR_POS = new THREE.Vector3(0, 0, -1.55);
    const NOZ_L = new THREE.Vector3(-0.75, -0.25, 0);
    const NOZ_R = new THREE.Vector3(0.75, -0.25, 0);
    const NOZ_C = new THREE.Vector3(0, -0.3, 0);

    const _v1 = new THREE.Vector3(), _v2 = new THREE.Vector3(), _v3 = new THREE.Vector3();
    const _v4 = new THREE.Vector3(), _v5 = new THREE.Vector3(), _v6 = new THREE.Vector3(), _f = new THREE.Vector3();
    const _F = new THREE.Vector3(), _T = new THREE.Vector3(), _fw = new THREE.Vector3();
    const _q1 = new THREE.Quaternion(), _q2 = new THREE.Quaternion();

    function addForceBody(F, T, q, fBody, rBody) {
      T.x += rBody.y * fBody.z - rBody.z * fBody.y;
      T.y += rBody.z * fBody.x - rBody.x * fBody.z;
      T.z += rBody.x * fBody.y - rBody.y * fBody.x;
      F.add(_fw.copy(fBody).applyQuaternion(q));
    }

    class BikePhysics {
      constructor() {
        this.pos = new THREE.Vector3();
        this.vel = new THREE.Vector3();
        this.quat = new THREE.Quaternion();
        this.angVel = new THREE.Vector3();
        this.angMom = new THREE.Vector3();
        this.I = new THREE.Vector3(170, 155, 48);        // pitch, yaw, roll inertia
        this.time = 0;
        this.throttle = 0;                               // rear jet COMMAND, 0..1
        this.collective = 0.5;                           // lift nozzle COMMAND, 0..1.25
        this.vert = 0;                                   // -1..1 climb/descend command (auto-lift)
        this.autoLift = true;
        this.liftAeroY = 0;
        this.spoolRear = 0;
        this.spoolLift = 0;
        this.ab = 0;
        this.steer = 0;                                  // -1..1 (left..right on screen)
        this.pitch = 0;                                  // -1..1 (nose down..nose up)
        this.nozzle = 0;
        this.brake = false;                              // the rider sat up into the wind
        this.input = { boostRear: false };
        this.pulses = { L: 0, R: 0, C: 0 };
        this.pulseWait = { L: 0, R: 0, C: 0 };
        this.pulseGrace = 0;
        this.assist = true;
        this.crashed = false;
        this.crashTimer = 0;
        this._crashEvent = null;
        this.grounded = false;
        this.jetRear = 0;
        this.jetLift = 0;
        this.wind = new Wind(9917, { speed: TUNE.windSpeed });
        this.windVec = new THREE.Vector3();
        this.airSpeed = 0;
        this.yaw = 0;                                    // rad, the heading the last reset faced
        this.reset();
      }

      /** Back on the ground where it is, level, facing `yaw` (radians about
       *  +Y; the default is the last heading). fable respawns on its pad; here
       *  the pad is wherever you were, because the place you got to is the
       *  point of the flight. */
      reset(x, z, yaw) {
        if (x !== undefined) this.pos.x = x;
        if (z !== undefined) this.pos.z = z;
        if (yaw !== undefined) this.yaw = yaw;
        // A hair into the skids' travel, so the first step is a grounded
        // one and the parked branch of the auto-lift takes it; fable sets
        // it a hair above, and the flight computer then held it there at a
        // hover, weightless, on the boundary of being parked.
        this.pos.y = groundAt() + TUNE.clearance - 0.02;
        this.vel.set(0, 0, 0);
        this.quat.setFromAxisAngle(_v1.set(0, 1, 0), this.yaw);
        this.angVel.set(0, 0, 0);
        this.angMom.set(0, 0, 0);
        this.throttle = 0;
        this.collective = 0.35;
        this.vert = 0;
        this.liftAeroY = 0;
        this.spoolRear = 0;
        this.spoolLift = this.collective;
        this.ab = 0;
        this.steer = 0;
        this.pitch = 0;
        this.nozzle = 0;
        this.pulses = { L: 0, R: 0, C: 0 };
        this.pulseWait = { L: 0, R: 0, C: 0 };
        this.pulseGrace = 0;
        this.crashed = false;
      }

      /** Level the bike where it is, in the air: the way out of a tumble
       *  that does not cost the altitude. Keeps the heading and the
       *  horizontal speed, drops the spin. */
      level() {
        const fwd = _v1.set(0, 0, 1).applyQuaternion(this.quat);
        const yaw = Math.atan2(fwd.x, fwd.z);
        this.quat.setFromAxisAngle(_v2.set(0, 1, 0), yaw);
        this.angVel.set(0, 0, 0);
        this.angMom.set(0, 0, 0);
        this.vel.y = 0;
        this.steer = 0;
        this.pitch = 0;
        this.nozzle = 0;
      }

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

      crash(reason) {
        if (this.crashed) return;
        this.crashed = true;
        this.crashTimer = 2.6;
        this._crashEvent = reason;
        this.vel.multiplyScalar(0.1);
        this.angVel.multiplyScalar(0.2);
        this.angMom.multiplyScalar(0.2);
      }

      /** The heading, radians about +Y, for a respawn that faces the way the
       *  wreck was going. */
      heading() {
        const fwd = _v1.set(0, 0, 1).applyQuaternion(this.quat);
        return Math.atan2(fwd.x, fwd.z);
      }

      step(dt) {
        if (this.crashed) {
          this.crashTimer -= dt;
          if (this.crashTimer <= 0) this.reset(this.pos.x, this.pos.z, this.heading());
          return;
        }
        const q = this.quat;
        const inv = _q1.copy(q).invert();
        const fwd = _v1.set(0, 0, 1).applyQuaternion(q);
        const right = _v2.set(1, 0, 0).applyQuaternion(q);
        const eRoll = right.y;
        const ePitch = fwd.y;

        const F = _F.set(0, -TUNE.g * TUNE.mass, 0);
        const T = _T.set(0, 0, 0);

        const groundH = groundAt(this.pos.x, this.pos.z);
        const agl = this.pos.y - groundH;
        const airFactor = clamp(1 - (this.pos.y - 500) / 400, 0.25, 1);
        const groundEffect = 1 + 0.35 * clamp(1 - agl / 5, 0, 1);

        // --- vertical autopilot (auto-lift): the rider never flies the nozzles ---
        if (this.assist && this.autoLift) {
          const upY = 1 - 2 * (q.x * q.x + q.z * q.z);
          const hSpd0 = Math.hypot(this.vel.x, this.vel.z);
          if (this.grounded && this.vert <= 0.05 && hSpd0 < 4 && this.throttle < 0.15) {
            this.collective += (0.3 - this.collective) * Math.min(1, dt * 3);
          } else {
            // The path follows the pitch the rider asks for, which is what
            // the nose is held at when nothing else is pulling on it. (fable
            // follows the nose itself, ePitch; see the note on TUNE.)
            const askPitch = Math.sin(this.pitch * TUNE.pitchRange);
            const vyT = clamp(hSpd0 * askPitch, -28, 28) + this.vert * TUNE.climbRate;
            const aCmd = TUNE.g + clamp((vyT - this.vel.y) * TUNE.vyGain, -14, 14);
            const eff = TUNE.maxLift * airFactor * groundEffect * Math.max(upY, 0.3);
            this.collective = clamp((TUNE.mass * aCmd - this.liftAeroY) / eff, 0, 1.25);
          }
        }

        // --- engine spool ---
        const tauR = this.throttle > this.spoolRear ? TUNE.spoolUp : TUNE.spoolDown;
        const spool0 = this.spoolRear;
        this.spoolRear += (this.throttle - this.spoolRear) * Math.min(1, dt / tauR);
        this.ab += ((this.input.boostRear ? 1 : 0) - this.ab) * Math.min(1, dt / TUNE.spoolAB);
        this.spoolLift += (this.collective - this.spoolLift) * Math.min(1, dt / TUNE.spoolLiftTau);

        // --- turbine rotor gyroscopics ---
        const hRot = TUNE.rotorH * this.spoolRear;
        T.x -= this.angVel.y * hRot;
        T.y += this.angVel.x * hRot;
        T.z -= TUNE.rotorH * (this.spoolRear - spool0) / dt;

        // --- rear jet, with yaw thrust vectoring ---
        // The vectoring is hover steering: at a standstill it is the only
        // thing that turns the nose. With the flight computer on it fades
        // out with speed, because at 4,600 N a fifteen-degree deflection
        // is 1,800 N m of yaw, and at 30 m/s that on top of a banked turn
        // is a skid the weathervane then fights (fable keeps it at all
        // speeds; see the note on TUNE).
        const rearT = (this.spoolRear * TUNE.maxRear + this.ab * TUNE.boostRear) * airFactor;
        const hSpdV = Math.hypot(this.vel.x, this.vel.z);
        const vecFade = this.assist ? clamp(1 - (hSpdV - TUNE.vecFadeLo) / (TUNE.vecFadeHi - TUNE.vecFadeLo), 0, 1) : 1;
        const vecCmd = this.steer * TUNE.vecMax * vecFade;
        this.nozzle += (vecCmd - this.nozzle) * Math.min(1, dt / TUNE.vecTau);
        addForceBody(F, T, q, _f.set(Math.sin(this.nozzle) * rearT, 0, Math.cos(this.nozzle) * rearT), REAR_POS);
        this.jetRear = rearT;

        // --- downward nozzles + RCS pulses ---
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

        // --- wind field ---
        this.wind.gustGain = TUNE.gust / 70;
        this.windVec.copy(this.wind.update(dt, this.airSpeed, this.pos.x, this.pos.z, agl));
        const vAir = _v6.copy(this.vel).sub(this.windVec);

        // --- aerodynamic drag, quadratic per body axis ---
        const vB = _v3.copy(vAir).applyQuaternion(inv);
        const speed = vAir.length();
        this.airSpeed = speed;
        const cdaZ = TUNE.cdaZ * (this.brake ? TUNE.brakeK : 1);
        _f.set(
          -0.6 * TUNE.cdaX * Math.abs(vB.x) * vB.x,
          -0.6 * TUNE.cdaY * Math.abs(vB.y) * vB.y,
          -0.6 * cdaZ * Math.abs(vB.z) * vB.z
        );
        F.add(_v4.copy(_f).applyQuaternion(q));

        // --- lifting-body aerodynamics ---
        this.liftAeroY = 0;
        if (vB.z > 5) {
          const aoa = Math.atan2(-vB.y, vB.z);
          const cl = clamp(aoa * TUNE.liftSlope, -TUNE.clMax, TUNE.clMax);
          const qS = (vB.z * vB.z + vB.y * vB.y) * TUNE.liftArea;
          _f.set(0, cl * qS, -0.35 * cl * cl * qS).applyQuaternion(q);
          this.liftAeroY = _f.y;
          F.add(_f);
        }

        // --- weathervane: nose follows the airflow ---
        if (speed > 3) {
          _v4.copy(vB).normalize();
          const q2 = speed * speed;
          const fz = clamp(Math.max(0, _v4.z), 0, 1);
          T.x += clamp(-_v4.y * q2 * 0.55, -1200, 1200) * fz * (1 - 0.75 * Math.abs(this.pitch));
          T.y += clamp(_v4.x * q2 * 0.9, -1500, 1500);
        }

        // --- flight controller ---
        const hSpd = Math.hypot(this.vel.x, this.vel.z);
        const bankAng = Math.min(Math.atan2(hSpd * Math.abs(this.steer) * TUNE.turnRateMax, TUNE.g), TUNE.bankMax);
        const wantBank = Math.sin(bankAng) * Math.sign(this.steer);
        const rawSide = this.pulses.L > 0 || this.pulses.R > 0 || this.pulseGrace > 0;
        if (this.assist && !rawSide) {
          T.z += clamp(-((eRoll - wantBank) * 1500 + this.angVel.z * 280), -950, 950);
          T.y -= this.angVel.y * 220;
        } else {
          T.z += this.steer * 750 - this.angVel.z * 50 * TUNE.rotDamp;
        }
        const ad = (1 + speed * 0.012) * TUNE.rotDamp;
        if (this.assist) {
          const wantPitch = Math.sin(this.pitch * TUNE.pitchRange);
          // Turn coordination. Banked, the yaw about the tilted axis moves
          // the nose down in the world by (yaw rate x sin bank), and holding
          // it on the horizon takes a pitch rate of (yaw rate x tan bank)
          // that the damping here and below would otherwise have to be
          // fought for with a nose-down error. Supplied as feed-forward, so
          // the nose stays where the rider put it through the turn - and
          // bounded twice, at a turn's worth of yaw rate and of bank, and
          // then inside the loop's own clamp, because an unbounded term in
          // the yaw rate fed the pitch rate that fed the yaw rate (measured:
          // 8 rad/s of body rates in a full-throttle turn).
          const cosBank = Math.sqrt(Math.max(0, 1 - eRoll * eRoll));
          const tanBank = clamp(eRoll / Math.max(cosBank, 0.3), -1.2, 1.2);
          const ff = TUNE.turnCoord * (650 + 130 * ad) * clamp(this.angVel.y, -0.8, 0.8) * tanBank;
          T.x += clamp((ePitch - wantPitch) * 2200 - this.angVel.x * 650 + ff, -2200, 2200);
        } else {
          T.x += -this.pitch * 1100;
        }

        // --- rotational damping ---
        T.x -= this.angVel.x * 130 * ad;
        T.y -= this.angVel.y * 160 * ad;
        T.z -= this.angVel.z * 40 * ad;

        // --- ground contact ---
        this.grounded = false;
        if (agl < TUNE.clearance) {
          this.grounded = true;
          const n = _v5.set(0, 1, 0);
          const vn = this.vel.dot(n);
          const upDot = _v4.set(0, 1, 0).applyQuaternion(q).y;
          if (vn < -TUNE.crashSpeed) { this.crash('ground'); return; }
          if (upDot < 0.25 && speed > 6) { this.crash('flip'); return; }
          const pen = TUNE.clearance - agl;
          const fn = Math.max(0, pen * 42000 + Math.max(0, -vn) * 5200);
          F.addScaledVector(n, fn);
          _v4.copy(this.vel).addScaledVector(n, -vn);
          F.addScaledVector(_v4, -clamp(pen * 8, 0, 1) * 1100);
          T.x += ePitch * 2200 - this.angVel.x * 450;
          T.z += -eRoll * 2200 - this.angVel.z * 350;
          T.y -= this.angVel.y * 350;
        }

        // --- integrate: semi-implicit Euler, exact rotation ---
        this.time += dt;
        this.vel.addScaledVector(F, dt / TUNE.mass);
        this.pos.addScaledVector(this.vel, dt);

        const w = this.angVel, I = this.I, L = this.angMom;
        L.x += T.x * dt;
        L.y += T.y * dt;
        L.z += T.z * dt;
        const wn = w.length();
        if (wn > 1e-9) {
          const half = 0.5 * wn * dt;
          const s = Math.sin(half) / wn;
          _q2.set(w.x * s, w.y * s, w.z * s, Math.cos(half));
          q.multiply(_q2);
          q.normalize();
          L.applyQuaternion(_q2.conjugate());
        }
        w.set(L.x / I.x, L.y / I.y, L.z / I.z);
        const wl = w.length();
        if (wl > 9) {
          w.multiplyScalar(9 / wl);
          L.set(w.x * I.x, w.y * I.y, w.z * I.z);
        }

        const g2 = groundAt(this.pos.x, this.pos.z);
        if (this.pos.y < g2 + 0.2) {
          this.pos.y = g2 + 0.2;
          if (this.vel.y < 0) this.vel.y *= -0.1;
        }
      }
    }

    /* ---------- the bike mesh (fable's bike.js) ----------
     *
     * "FLYING HOG / ALPINIST": a classic black motorcycle top half (tank
     * with lettering, brown leather saddle, chrome bars, round headlight)
     * riding on a big exposed turbojet slung underneath, wrapped in pipes,
     * with two side lift pods, plus a black-clad rider with helmet and
     * backpack. */

    function tankTexture() {
      const c = document.createElement('canvas');
      c.width = 256; c.height = 128;
      const ctx = c.getContext('2d');
      ctx.clearRect(0, 0, 256, 128);
      ctx.fillStyle = '#e8e5da';
      ctx.textAlign = 'center';
      ctx.font = 'bold 38px sans-serif';
      ctx.fillText('FLYING HOG', 128, 52);
      ctx.font = '26px sans-serif';
      ctx.fillText('ALPINIST', 128, 96);
      const tex = new THREE.CanvasTexture(c);
      tex.colorSpace = THREE.SRGBColorSpace;
      return tex;
    }

    function buildBike() {
      const g = new THREE.Group();

      const steel = new THREE.MeshStandardMaterial({ color: 0x4d5158, metalness: 0.85, roughness: 0.38 });
      const dark = new THREE.MeshStandardMaterial({ color: 0x25272b, metalness: 0.8, roughness: 0.5 });
      const chrome = new THREE.MeshStandardMaterial({ color: 0xb9c1c9, metalness: 0.95, roughness: 0.18 });
      const tankM = new THREE.MeshStandardMaterial({ color: 0x1b1d20, metalness: 0.6, roughness: 0.35 });
      const leather = new THREE.MeshStandardMaterial({ color: 0x7b4a28, roughness: 0.85 });
      const rubber = new THREE.MeshStandardMaterial({ color: 0x131418, roughness: 0.95 });
      const cloth = new THREE.MeshStandardMaterial({ color: 0x191a1d, roughness: 0.9 });
      const pack = new THREE.MeshStandardMaterial({ color: 0x53462f, roughness: 0.9 });
      const copper = new THREE.MeshStandardMaterial({ color: 0x9a6132, metalness: 0.85, roughness: 0.4 });
      const visorM = new THREE.MeshStandardMaterial({ color: 0x2a3540, metalness: 0.6, roughness: 0.2 });
      const helmetM = new THREE.MeshStandardMaterial({ color: 0x131417, metalness: 0.3, roughness: 0.3 });

      function add(geo, mat, x, y, z, rx = 0, ry = 0, rz = 0) {
        const m = new THREE.Mesh(geo, mat);
        m.position.set(x, y, z);
        m.rotation.set(rx, ry, rz);
        g.add(m);
        return m;
      }
      const _a = new THREE.Vector3(), _b = new THREE.Vector3(), _Y = new THREE.Vector3(0, 1, 0);
      function tube(ax, ay, az, bx, by, bz, r, mat) {
        _a.set(ax, ay, az); _b.set(bx, by, bz);
        const len = _a.distanceTo(_b);
        const m = new THREE.Mesh(new THREE.CylinderGeometry(r, r, len, 8), mat);
        m.position.copy(_a).add(_b).multiplyScalar(0.5);
        m.quaternion.setFromUnitVectors(_Y, _b.sub(_a).normalize());
        g.add(m);
        return m;
      }
      const RX = Math.PI / 2;

      // ---- main turbojet, slung under the frame ----
      add(new THREE.CylinderGeometry(0.36, 0.36, 1.6, 18), steel, 0, -0.16, -0.15, RX);
      add(new THREE.CylinderGeometry(0.4, 0.4, 0.28, 18), dark, 0, -0.16, 0.58, RX);
      add(new THREE.TorusGeometry(0.36, 0.05, 8, 20), chrome, 0, -0.16, 0.73);
      add(new THREE.CylinderGeometry(0.34, 0.34, 0.06, 18), rubber, 0, -0.16, 0.71, RX);
      add(new THREE.ConeGeometry(0.14, 0.28, 12), dark, 0, -0.16, 0.86, RX);
      add(new THREE.CylinderGeometry(0.34, 0.27, 0.4, 18), steel, 0, -0.16, -1.15, RX);
      add(new THREE.CylinderGeometry(0.27, 0.23, 0.45, 16), dark, 0, -0.16, -1.55, RX);
      add(new THREE.TorusGeometry(0.235, 0.03, 6, 16), dark, 0, -0.16, -1.76);
      for (const [z, mat, a0] of [[0.32, copper, 0.2], [0.02, chrome, 2.1], [-0.4, copper, 4.0]]) {
        const p = add(new THREE.TorusGeometry(0.43, 0.035, 6, 14, Math.PI * 1.25), mat, 0, -0.16, z);
        p.rotation.z = a0;
      }
      for (const s of [-1, 1]) {
        add(new THREE.CylinderGeometry(0.08, 0.08, 0.95, 10), chrome, s * 0.45, 0.0, -0.7, RX);
        add(new THREE.CylinderGeometry(0.085, 0.06, 0.12, 10), dark, s * 0.45, 0.0, -1.22, RX);
        add(new THREE.BoxGeometry(0.2, 0.26, 0.42), dark, s * 0.29, -0.4, 0.32);
      }
      add(new THREE.BoxGeometry(0.26, 0.2, 2.0), dark, 0, 0.12, -0.05);

      // ---- tank with lettering ----
      const tank = add(new THREE.CapsuleGeometry(0.28, 0.5, 6, 14), tankM, 0, 0.42, 0.4, RX);
      tank.scale.set(0.95, 1, 0.78);
      add(new THREE.CylinderGeometry(0.05, 0.05, 0.04, 10), chrome, 0, 0.65, 0.5);
      const decal = new THREE.MeshBasicMaterial({ map: tankTexture(), transparent: true });
      for (const s of [-1, 1]) {
        add(new THREE.PlaneGeometry(0.5, 0.25), decal, s * 0.27, 0.44, 0.4, 0, s * RX);
      }

      // ---- saddle, fender ----
      const seat = add(new THREE.CapsuleGeometry(0.2, 0.55, 5, 12), leather, 0, 0.33, -0.45, RX);
      seat.scale.set(1.05, 1, 0.5);
      const darkLeather = new THREE.MeshStandardMaterial({ color: 0x4e2d17, roughness: 0.9 });
      const hump = add(new THREE.SphereGeometry(0.16, 10, 8), darkLeather, 0, 0.36, -0.85);
      hump.scale.set(1.2, 0.5, 0.7);
      add(new THREE.BoxGeometry(0.38, 0.07, 0.55), dark, 0, 0.27, -1.15, -0.15);
      add(new THREE.BoxGeometry(0.12, 0.06, 0.05), new THREE.MeshStandardMaterial({
        color: 0x801812, emissive: 0xb02010, emissiveIntensity: 0.8 }), 0, 0.31, -1.42);

      // ---- front end: forks, bars, headlight ----
      for (const s of [-1, 1]) tube(s * 0.13, -0.3, 1.18, s * 0.13, 0.44, 0.95, 0.034, chrome);
      add(new THREE.BoxGeometry(0.34, 0.08, 0.12), dark, 0, -0.28, 1.16);
      add(new THREE.BoxGeometry(0.3, 0.1, 0.16), dark, 0, 0.48, 0.93);
      const bar = add(new THREE.CylinderGeometry(0.028, 0.028, 0.8, 8), chrome, 0, 0.57, 0.88);
      bar.rotation.z = RX;
      for (const s of [-1, 1]) {
        const grip = add(new THREE.CylinderGeometry(0.045, 0.045, 0.16, 8), rubber, s * 0.37, 0.57, 0.88);
        grip.rotation.z = RX;
      }
      add(new THREE.CylinderGeometry(0.13, 0.13, 0.1, 14), chrome, 0, 0.32, 1.1, RX);
      add(new THREE.CircleGeometry(0.11, 14), new THREE.MeshBasicMaterial({ color: 0xfff6d8 }), 0, 0.32, 1.16);
      tube(0, -0.28, 1.15, 0, -0.52, 1.05, 0.03, dark);
      add(new THREE.BoxGeometry(0.09, 0.05, 0.4), chrome, 0, -0.545, 1.0);

      // ---- lift pods (positions match physics NOZ_L/NOZ_R/NOZ_C) ----
      for (const s of [-1, 1]) {
        add(new THREE.BoxGeometry(0.45, 0.07, 0.2), steel, s * 0.5, -0.1, 0);
        add(new THREE.CylinderGeometry(0.15, 0.17, 0.4, 12), steel, s * 0.75, -0.25, 0);
        const lip = add(new THREE.TorusGeometry(0.12, 0.03, 6, 14), dark, s * 0.75, -0.05, 0);
        lip.rotation.x = RX;
        add(new THREE.CylinderGeometry(0.11, 0.14, 0.14, 12), chrome, s * 0.75, -0.47, 0);
      }
      add(new THREE.CylinderGeometry(0.13, 0.16, 0.12, 12), chrome, 0, -0.52, 0);

      // ---- landing skids ----
      for (const s of [-1, 1]) {
        add(new THREE.BoxGeometry(0.07, 0.05, 1.5), chrome, s * 0.42, -0.55, -0.15);
        tube(s * 0.42, -0.54, 0.25, s * 0.34, -0.3, 0.25, 0.03, dark);
        tube(s * 0.42, -0.54, -0.55, s * 0.34, -0.3, -0.55, 0.03, dark);
      }

      // ---- rider: black jacket, helmet, backpack ----
      add(new THREE.BoxGeometry(0.34, 0.2, 0.3), cloth, 0, 0.52, -0.42);
      add(new THREE.BoxGeometry(0.4, 0.55, 0.24), cloth, 0, 0.78, -0.26, 0.6);
      add(new THREE.BoxGeometry(0.44, 0.12, 0.22), cloth, 0, 0.97, -0.11, 0.6);
      add(new THREE.SphereGeometry(0.17, 14, 12), helmetM, 0, 1.14, -0.03);
      add(new THREE.BoxGeometry(0.21, 0.09, 0.05), visorM, 0, 1.15, 0.13);
      add(new THREE.BoxGeometry(0.34, 0.44, 0.18), pack, 0, 0.9, -0.45, 0.6);
      add(new THREE.BoxGeometry(0.26, 0.12, 0.06), pack, 0, 1.05, -0.35, 0.6);
      for (const s of [-1, 1]) {
        tube(s * 0.24, 0.97, -0.11, s * 0.3, 0.75, 0.3, 0.05, cloth);
        tube(s * 0.3, 0.75, 0.3, s * 0.37, 0.6, 0.82, 0.045, cloth);
        add(new THREE.SphereGeometry(0.055, 8, 8), rubber, s * 0.37, 0.58, 0.86);
        tube(s * 0.15, 0.5, -0.42, s * 0.22, 0.34, 0.12, 0.07, cloth);
        tube(s * 0.22, 0.34, 0.12, s * 0.3, 0.02, -0.08, 0.055, cloth);
        add(new THREE.BoxGeometry(0.11, 0.09, 0.3), rubber, s * 0.3, 0.0, -0.03);
        add(new THREE.BoxGeometry(0.16, 0.04, 0.08), chrome, s * 0.33, -0.02, -0.08);
      }

      return { group: g };
    }

    /* ---------- exhaust (fable's jetfx.js) ----------
     *
     * Two additive shader shells per jet: a white-blue core with stationary
     * mach diamonds inside a wider sheath that cools through blue into
     * orange downstream; ember sparks dragged by air and bouncing off the
     * ground; dust kicked up by the lift jets in ground effect; two dynamic
     * lights tied to thrust. */

    const PLUME_VERT = `
uniform float uTime;
uniform float uSeed;
uniform float uFlick;
varying float vT;
varying vec3 vNormal;
varying vec3 vView;
void main() {
  vT = 1.0 - uv.y;
  float prof = (0.92 + 0.3 * vT) * pow(1.0 - vT, 0.55);
  float rip = 1.0
    + uFlick * vT * 0.16 * sin(vT * 21.0 - uTime * 26.0 + uSeed)
    + uFlick * vT * 0.09 * sin(vT * 47.0 - uTime * 41.0 + uSeed * 2.7 + uv.x * 6.283);
  vec3 p = position;
  p.xz *= prof * rip;
  p.x += uFlick * vT * vT * 0.08 * sin(uTime * 19.0 + uSeed);
  vec4 mv = modelViewMatrix * vec4(p, 1.0);
  vNormal = normalMatrix * normal;
  vView = -mv.xyz;
  gl_Position = projectionMatrix * mv;
}`;

    const PLUME_FRAG = `
uniform vec3 uColA;
uniform vec3 uColB;
uniform vec3 uColC;
uniform float uTime;
uniform float uSeed;
uniform float uDiamonds;
uniform float uGain;
varying float vT;
varying vec3 vNormal;
varying vec3 vView;
void main() {
  float body = abs(dot(normalize(vNormal), normalize(vView)));
  float t = clamp(vT, 0.0, 1.0);
  vec3 col = mix(uColA, uColB, smoothstep(0.02, 0.3, t));
  col = mix(col, uColC, smoothstep(0.24, 0.85, t));
  float a = pow(1.0 - t, 1.15) * pow(body, 1.4);
  if (uDiamonds > 0.5) {
    float d = pow(max(sin(t * uDiamonds * 6.2832 + 1.2), 0.0), 10.0);
    col += vec3(0.9, 0.95, 1.0) * d * (1.0 - t) * 1.5;
    a += d * (1.0 - t) * 0.8;
  }
  a *= 0.82 + 0.18 * sin(uTime * 31.0 + uSeed);
  gl_FragColor = vec4(col * uGain, a);
}`;

    function plumeMaterial(colA, colB, colC, diamonds, gain) {
      return new THREE.ShaderMaterial({
        vertexShader: PLUME_VERT,
        fragmentShader: PLUME_FRAG,
        uniforms: {
          uTime: { value: 0 },
          uSeed: { value: Math.random() * 20 },
          uFlick: { value: 1 },
          uColA: { value: new THREE.Color(colA) },
          uColB: { value: new THREE.Color(colB) },
          uColC: { value: new THREE.Color(colC) },
          uDiamonds: { value: diamonds },
          uGain: { value: gain },
        },
        transparent: true,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
        side: THREE.DoubleSide,
      });
    }

    function discTexture(inner = 1, falloff = 0.5) {
      const c = document.createElement('canvas');
      c.width = c.height = 64;
      const ctx = c.getContext('2d');
      const g = ctx.createRadialGradient(32, 32, 0, 32, 32, 32);
      g.addColorStop(0, `rgba(255,255,255,${inner})`);
      g.addColorStop(falloff, 'rgba(255,255,255,0.35)');
      g.addColorStop(1, 'rgba(255,255,255,0)');
      ctx.fillStyle = g;
      ctx.fillRect(0, 0, 64, 64);
      return new THREE.CanvasTexture(c);
    }

    function plumeGeo() {
      const g = new THREE.CylinderGeometry(1, 1, 1, 18, 30, true);
      g.translate(0, -0.5, 0);
      return g;
    }

    const SPARK_N = 460;
    const DUST_N = 260;

    function makeJetFX(scene, bikeGroup, camera, renderer) {
      const geo = plumeGeo();
      const glowTex = discTexture(1, 0.4);
      const softTex = discTexture(0.8, 0.55);

      function makeJet(cfg) {
        const grp = new THREE.Group();
        grp.position.copy(cfg.pos);
        if (cfg.axis === 'z') {
          grp.rotation.order = 'YXZ';
          grp.rotation.x = Math.PI / 2;
        }
        const core = new THREE.Mesh(geo, plumeMaterial(cfg.colors[0], cfg.colors[1], cfg.colors[2], cfg.diamonds, 1.35));
        const sheath = new THREE.Mesh(geo, plumeMaterial(cfg.colors[1], cfg.colors[2], cfg.colors[3], 0, 0.8));
        core.renderOrder = sheath.renderOrder = 20;
        grp.add(sheath, core);
        const glow = new THREE.Sprite(new THREE.SpriteMaterial({
          map: glowTex, color: cfg.colors[0], transparent: true, opacity: 0.9,
          blending: THREE.AdditiveBlending, depthWrite: false,
        }));
        glow.renderOrder = 21;
        grp.add(glow);
        bikeGroup.add(grp);
        return { grp, core, sheath, glow, cfg, level: 0 };
      }

      const jets = {
        rear: makeJet({
          pos: new THREE.Vector3(0, -0.16, -1.8), axis: 'z',
          dir: new THREE.Vector3(0, 0, -1),
          r: 0.23, len: 2.8, diamonds: 5,
          colors: [0xf8fbff, 0x9cc8ff, 0xff9a3c, 0xb43a10],
          sparks: 130,
        }),
        left: makeJet({
          pos: new THREE.Vector3(-0.75, -0.5, 0), axis: 'y',
          dir: new THREE.Vector3(0, -1, 0),
          r: 0.13, len: 1.55, diamonds: 3,
          colors: [0xf2f8ff, 0x9cc8ff, 0x5f8fe8, 0x27458f],
          sparks: 36,
        }),
        right: makeJet({
          pos: new THREE.Vector3(0.75, -0.5, 0), axis: 'y',
          dir: new THREE.Vector3(0, -1, 0),
          r: 0.13, len: 1.55, diamonds: 3,
          colors: [0xf2f8ff, 0x9cc8ff, 0x5f8fe8, 0x27458f],
          sparks: 36,
        }),
        center: makeJet({
          pos: new THREE.Vector3(0, -0.6, 0), axis: 'y',
          dir: new THREE.Vector3(0, -1, 0),
          r: 0.16, len: 1.85, diamonds: 3,
          colors: [0xf2f8ff, 0x9cc8ff, 0x5f8fe8, 0x27458f],
          sparks: 48,
        }),
      };

      const rearLight = new THREE.PointLight(0xff9a40, 0, 55, 2);
      rearLight.position.set(0, 0.1, -2.4);
      const liftLight = new THREE.PointLight(0x86b8ff, 0, 30, 2);
      liftLight.position.set(0, -1.3, 0);
      bikeGroup.add(rearLight, liftLight);

      function makePool(n, { blending, texture, isDust }) {
        const pos = new Float32Array(n * 3);
        const lifeAttr = new Float32Array(n);
        const seed = new Float32Array(n);
        for (let i = 0; i < n; i++) seed[i] = Math.random();
        const g = new THREE.BufferGeometry();
        g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
        g.setAttribute('aLife', new THREE.BufferAttribute(lifeAttr, 1));
        g.setAttribute('aSeed', new THREE.BufferAttribute(seed, 1));
        const mat = new THREE.ShaderMaterial({
          uniforms: { uTex: { value: texture }, uPx: { value: 800 } },
          vertexShader: `
            attribute float aLife;
            attribute float aSeed;
            uniform float uPx;
            varying float vLife;
            varying float vSeed;
            void main() {
              vLife = aLife;
              vSeed = aSeed;
              vec4 mv = modelViewMatrix * vec4(position, 1.0);
              ${isDust
                ? 'float s = 0.7 + 2.6 * (1.0 - aLife) + aSeed * 0.6;'
                : 'float s = 0.055 + 0.075 * aLife + aSeed * 0.02;'}
              gl_PointSize = uPx * s / max(1.0, -mv.z);
              gl_Position = projectionMatrix * mv;
              if (aLife <= 0.0) gl_Position = vec4(0.0, 0.0, -10.0, 1.0);
            }`,
          fragmentShader: isDust ? `
            uniform sampler2D uTex;
            varying float vLife;
            varying float vSeed;
            void main() {
              float a = texture2D(uTex, gl_PointCoord).a;
              vec3 col = mix(vec3(0.55, 0.5, 0.4), vec3(0.72, 0.68, 0.58), vSeed);
              gl_FragColor = vec4(col, a * vLife * 0.33);
            }` : `
            uniform sampler2D uTex;
            varying float vLife;
            varying float vSeed;
            void main() {
              float a = texture2D(uTex, gl_PointCoord).a;
              vec3 hot = vec3(1.0, 0.92, 0.6);
              vec3 mid = vec3(1.0, 0.45, 0.1);
              vec3 cold = vec3(0.45, 0.08, 0.02);
              vec3 col = mix(cold, mix(mid, hot, smoothstep(0.55, 1.0, vLife)), smoothstep(0.0, 0.55, vLife));
              gl_FragColor = vec4(col, a * min(1.0, vLife * 2.5));
            }`,
          transparent: true,
          blending,
          depthWrite: false,
        });
        const points = new THREE.Points(g, mat);
        points.frustumCulled = false;
        points.renderOrder = isDust ? 18 : 22;
        scene.add(points);
        return { points, mat, pos, life: lifeAttr, vel: new Float32Array(n * 3), dur: new Float32Array(n), head: 0, n };
      }

      const sparks = makePool(SPARK_N, { blending: THREE.AdditiveBlending, texture: glowTex, isDust: false });
      const dust = makePool(DUST_N, { blending: THREE.NormalBlending, texture: softTex, isDust: true });

      function spawn(pool, x, y, z, vx, vy, vz, dur) {
        const i = pool.head;
        pool.head = (i + 1) % pool.n;
        pool.pos[i * 3] = x; pool.pos[i * 3 + 1] = y; pool.pos[i * 3 + 2] = z;
        pool.vel[i * 3] = vx; pool.vel[i * 3 + 1] = vy; pool.vel[i * 3 + 2] = vz;
        pool.dur[i] = dur;
        pool.life[i] = 1;
      }

      const _p = new THREE.Vector3(), _d = new THREE.Vector3();
      let sparkAcc = 0, dustAcc = 0, time = 0;

      function update(dt, o) {
        time += dt;
        const px = renderer.domElement.height / (2 * Math.tan(camera.fov * Math.PI / 360));

        jets.rear.grp.rotation.y = o.vec || 0;
        for (const key of ['rear', 'left', 'right', 'center']) {
          const j = jets[key], v = o.jets[key];
          j.level += (v - j.level) * Math.min(1, dt * 14);
          const lv = j.level;
          const on = lv > 0.04;
          j.core.visible = j.sheath.visible = on;
          j.glow.visible = lv > 0.02;
          if (on) {
            const flick = 0.92 + 0.13 * Math.sin(time * 37 + j.core.material.uniforms.uSeed.value);
            const w = j.cfg.r * (0.75 + 0.5 * Math.min(lv, 1.6));
            const len = j.cfg.len * lv * flick;
            j.core.scale.set(w * 0.62, Math.max(0.05, len), w * 0.62);
            j.sheath.scale.set(w * 1.35, Math.max(0.05, len * 1.18), w * 1.35);
            for (const m of [j.core.material, j.sheath.material]) {
              m.uniforms.uTime.value = time;
              m.uniforms.uFlick.value = 0.7 + 0.6 * Math.min(lv, 1.5);
            }
          }
          j.glow.scale.setScalar(0.28 + 0.5 * Math.min(lv, 1.6));
          j.glow.material.opacity = 0.55 * Math.min(1, lv * 1.6);
        }
        rearLight.intensity = 26 * Math.min(o.jets.rear, 2.2);
        liftLight.intensity = 9 * Math.min(o.jets.left + o.jets.right + o.jets.center, 3.5);

        for (const key of ['rear', 'left', 'right', 'center']) {
          const j = jets[key], v = o.jets[key];
          if (v < 0.25) continue;
          sparkAcc += j.cfg.sparks * Math.min(v, 2) * dt;
          while (sparkAcc >= 1) {
            sparkAcc -= 1;
            _p.copy(j.cfg.pos).applyQuaternion(o.quat).add(o.pos);
            _d.copy(j.cfg.dir).applyQuaternion(o.quat);
            const sp = 13 + Math.random() * 12;
            spawn(sparks,
              _p.x, _p.y, _p.z,
              _d.x * sp + (Math.random() - 0.5) * 5 + o.vel.x * 0.9,
              _d.y * sp + (Math.random() - 0.5) * 5 + o.vel.y * 0.9,
              _d.z * sp + (Math.random() - 0.5) * 5 + o.vel.z * 0.9,
              0.22 + Math.random() * 0.42);
          }
        }

        const liftSum = o.jets.left + o.jets.right + o.jets.center;
        if (o.agl < 8 && liftSum > 0.8) {
          dustAcc += (1 - o.agl / 8) * Math.min(liftSum, 3) * 65 * dt;
          while (dustAcc >= 1) {
            dustAcc -= 1;
            const ang = Math.random() * Math.PI * 2;
            const rr = 0.6 + Math.random() * 1.6;
            const rs = 4.5 + Math.random() * 6;
            spawn(dust,
              o.pos.x + Math.cos(ang) * rr, o.groundY + 0.25, o.pos.z + Math.sin(ang) * rr,
              Math.cos(ang) * rs + o.vel.x * 0.4, 0.6 + Math.random() * 1.6, Math.sin(ang) * rs + o.vel.z * 0.4,
              0.8 + Math.random() * 0.9);
          }
        }

        sparks.mat.uniforms.uPx.value = px;
        dust.mat.uniforms.uPx.value = px;
        for (let i = 0; i < sparks.n; i++) {
          if (sparks.life[i] <= 0) continue;
          sparks.life[i] -= dt / sparks.dur[i];
          const k = i * 3;
          sparks.vel[k] *= 1 - 2.2 * dt;
          sparks.vel[k + 1] -= 14 * dt;
          sparks.vel[k + 2] *= 1 - 2.2 * dt;
          sparks.pos[k] += sparks.vel[k] * dt;
          sparks.pos[k + 1] += sparks.vel[k + 1] * dt;
          sparks.pos[k + 2] += sparks.vel[k + 2] * dt;
          if (sparks.pos[k + 1] < o.groundY + 0.05 && sparks.vel[k + 1] < 0) {
            sparks.pos[k + 1] = o.groundY + 0.05;
            sparks.vel[k + 1] *= -0.35;
            sparks.vel[k] *= 0.6;
            sparks.vel[k + 2] *= 0.6;
          }
        }
        for (let i = 0; i < dust.n; i++) {
          if (dust.life[i] <= 0) continue;
          dust.life[i] -= dt / dust.dur[i];
          const k = i * 3;
          dust.vel[k] *= 1 - 1.4 * dt;
          dust.vel[k + 2] *= 1 - 1.4 * dt;
          dust.pos[k] += dust.vel[k] * dt;
          dust.pos[k + 1] += dust.vel[k + 1] * dt;
          dust.pos[k + 2] += dust.vel[k + 2] * dt;
        }
        sparks.points.geometry.attributes.position.needsUpdate = true;
        sparks.points.geometry.attributes.aLife.needsUpdate = true;
        dust.points.geometry.attributes.position.needsUpdate = true;
        dust.points.geometry.attributes.aLife.needsUpdate = true;
      }

      return { update };
    }

    /* ---------- the crash (fable's explosion.js) ----------
     *
     * Expanding additive fireball shells, ballistic debris that bounces off
     * the ground, a rising smoke column, hot sparks, a decaying orange flash
     * and a camera-shake signal. The water variant went with the water. */

    const SMOKE_N = 130, XSPARK_N = 170;

    function makeExplosion(scene, camera, renderer) {
      const tex = discTexture(1, 0.45);
      const softTex = discTexture(0.75, 0.6);

      const SHELLS = [
        { color: 0xfff6e0, size: 5, grow: 30, dur: 0.5 },
        { color: 0xff9a30, size: 3.5, grow: 40, dur: 0.75 },
        { color: 0xc04a12, size: 2.5, grow: 46, dur: 1.0 },
      ];
      const shells = SHELLS.map((cfg) => {
        const s = new THREE.Sprite(new THREE.SpriteMaterial({
          map: tex, color: cfg.color, transparent: true, opacity: 0,
          blending: THREE.AdditiveBlending, depthWrite: false,
        }));
        s.visible = false;
        s.renderOrder = 25;
        scene.add(s);
        return { s, cfg };
      });
      const light = new THREE.PointLight(0xffa040, 0, 260, 2);
      scene.add(light);

      const ND = 42;
      const debris = new THREE.InstancedMesh(
        new THREE.BoxGeometry(0.24, 0.2, 0.36),
        new THREE.MeshLambertMaterial({ color: 0x303236 }), ND);
      debris.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
      debris.frustumCulled = false;
      debris.visible = false;
      scene.add(debris);
      const dP = new Float32Array(ND * 3), dV = new Float32Array(ND * 3);
      const dR = new Float32Array(ND * 3), dW = new Float32Array(ND * 3);
      const dummy = new THREE.Object3D();

      function makePool(n, { additive, isSmoke }) {
        const pos = new Float32Array(n * 3), life = new Float32Array(n), seed = new Float32Array(n);
        for (let i = 0; i < n; i++) seed[i] = Math.random();
        const g = new THREE.BufferGeometry();
        g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
        g.setAttribute('aLife', new THREE.BufferAttribute(life, 1));
        g.setAttribute('aSeed', new THREE.BufferAttribute(seed, 1));
        const mat = new THREE.ShaderMaterial({
          uniforms: {
            uTex: { value: isSmoke ? softTex : tex }, uPx: { value: 800 },
            uColA: { value: new THREE.Color(0x2b2b2e) }, uColB: { value: new THREE.Color(0x6e6a66) },
          },
          vertexShader: `
            attribute float aLife; attribute float aSeed;
            uniform float uPx;
            varying float vLife; varying float vSeed;
            void main() {
              vLife = aLife; vSeed = aSeed;
              vec4 mv = modelViewMatrix * vec4(position, 1.0);
              ${isSmoke
                ? 'float s = 1.4 + 5.5 * (1.0 - aLife) + aSeed * 1.2;'
                : 'float s = 0.09 + 0.12 * aLife + aSeed * 0.03;'}
              gl_PointSize = uPx * s / max(1.0, -mv.z);
              gl_Position = projectionMatrix * mv;
              if (aLife <= 0.0) gl_Position = vec4(0.0, 0.0, -10.0, 1.0);
            }`,
          fragmentShader: isSmoke ? `
            uniform sampler2D uTex; uniform vec3 uColA; uniform vec3 uColB;
            varying float vLife; varying float vSeed;
            void main() {
              float a = texture2D(uTex, gl_PointCoord).a;
              vec3 col = mix(uColA, uColB, vSeed);
              gl_FragColor = vec4(col, a * vLife * 0.5);
            }` : `
            uniform sampler2D uTex;
            varying float vLife; varying float vSeed;
            void main() {
              float a = texture2D(uTex, gl_PointCoord).a;
              vec3 hot = vec3(1.0, 0.93, 0.62);
              vec3 mid = vec3(1.0, 0.45, 0.1);
              vec3 cold = vec3(0.4, 0.07, 0.02);
              vec3 col = mix(cold, mix(mid, hot, smoothstep(0.6, 1.0, vLife)), smoothstep(0.0, 0.6, vLife));
              gl_FragColor = vec4(col, a * min(1.0, vLife * 2.0));
            }`,
          transparent: true,
          blending: additive ? THREE.AdditiveBlending : THREE.NormalBlending,
          depthWrite: false,
        });
        const points = new THREE.Points(g, mat);
        points.frustumCulled = false;
        points.renderOrder = isSmoke ? 19 : 24;
        scene.add(points);
        return { points, mat, pos, life, vel: new Float32Array(n * 3), dur: new Float32Array(n), n };
      }
      const smoke = makePool(SMOKE_N, { additive: false, isSmoke: true });
      const sparks = makePool(XSPARK_N, { additive: true, isSmoke: false });

      const api = {
        active: false,
        shake: 0,
        time: 0,
        groundY: 0,
        trigger(pos, groundY) {
          api.active = true;
          api.time = 0;
          api.shake = 1.4;
          api.groundY = groundY;

          for (const { s } of shells) {
            s.visible = true;
            s.position.copy(pos);
            s.material.opacity = 0;
          }
          light.intensity = 420;
          light.position.set(pos.x, pos.y + 2, pos.z);

          debris.visible = true;
          for (let i = 0; i < ND; i++) {
            const k = i * 3;
            dP[k] = pos.x; dP[k + 1] = pos.y + 0.3; dP[k + 2] = pos.z;
            const ang = Math.random() * Math.PI * 2;
            const sp = 4 + Math.random() * 14;
            dV[k] = Math.cos(ang) * sp;
            dV[k + 1] = 4 + Math.random() * 13;
            dV[k + 2] = Math.sin(ang) * sp;
            dR[k] = Math.random() * 6; dR[k + 1] = Math.random() * 6; dR[k + 2] = Math.random() * 6;
            dW[k] = (Math.random() - 0.5) * 14; dW[k + 1] = (Math.random() - 0.5) * 14; dW[k + 2] = (Math.random() - 0.5) * 14;
          }

          for (let i = 0; i < smoke.n; i++) {
            const k = i * 3;
            const ang = Math.random() * Math.PI * 2, rr = Math.random() * 1.6;
            smoke.pos[k] = pos.x + Math.cos(ang) * rr;
            smoke.pos[k + 1] = pos.y + Math.random() * 1.2;
            smoke.pos[k + 2] = pos.z + Math.sin(ang) * rr;
            smoke.vel[k] = Math.cos(ang) * (0.7 + Math.random() * 2.2);
            smoke.vel[k + 1] = 3.2 + Math.random() * 3.5;
            smoke.vel[k + 2] = Math.sin(ang) * (0.7 + Math.random() * 2.2);
            smoke.dur[i] = 2.4 + Math.random() * 1.8;
            smoke.life[i] = 1;
          }

          for (let i = 0; i < sparks.n; i++) {
            const k = i * 3;
            sparks.pos[k] = pos.x; sparks.pos[k + 1] = pos.y + 0.4; sparks.pos[k + 2] = pos.z;
            const th = Math.random() * Math.PI * 2, ph = Math.random() * Math.PI * 0.6;
            const sp = 9 + Math.random() * 22;
            sparks.vel[k] = Math.cos(th) * Math.sin(ph + 0.4) * sp;
            sparks.vel[k + 1] = Math.cos(ph) * sp * 0.8;
            sparks.vel[k + 2] = Math.sin(th) * Math.sin(ph + 0.4) * sp;
            sparks.dur[i] = 0.4 + Math.random() * 0.9;
            sparks.life[i] = 1;
          }
          smoke.points.geometry.attributes.aLife.needsUpdate = true;
          sparks.points.geometry.attributes.aLife.needsUpdate = true;
        },

        update(dt) {
          if (!api.active) return;
          api.time += dt;
          const t = api.time;
          api.shake = Math.max(0, api.shake - dt * 2.2);

          for (const { s, cfg } of shells) {
            if (!s.visible) continue;
            const u = t / cfg.dur;
            if (u >= 1) { s.visible = false; continue; }
            s.scale.setScalar(cfg.size + cfg.grow * Math.pow(u, 0.55));
            s.material.opacity = 0.95 * (1 - u) * (1 - u);
          }
          light.intensity *= Math.pow(0.02, dt);

          if (debris.visible) {
            let alive = false;
            for (let i = 0; i < ND; i++) {
              const k = i * 3;
              dV[k + 1] -= 22 * dt;
              dP[k] += dV[k] * dt; dP[k + 1] += dV[k + 1] * dt; dP[k + 2] += dV[k + 2] * dt;
              dR[k] += dW[k] * dt; dR[k + 1] += dW[k + 1] * dt; dR[k + 2] += dW[k + 2] * dt;
              const gy = groundAt(dP[k], dP[k + 2]) + 0.12;
              if (dP[k + 1] < gy) {
                dP[k + 1] = gy;
                if (dV[k + 1] < 0) dV[k + 1] *= -0.4;
                dV[k] *= 0.6; dV[k + 2] *= 0.6;
                dW[k] *= 0.5; dW[k + 1] *= 0.5; dW[k + 2] *= 0.5;
              }
              if (Math.abs(dV[k]) + Math.abs(dV[k + 1]) + Math.abs(dV[k + 2]) > 0.6) alive = true;
              dummy.position.set(dP[k], dP[k + 1], dP[k + 2]);
              dummy.rotation.set(dR[k], dR[k + 1], dR[k + 2]);
              dummy.updateMatrix();
              debris.setMatrixAt(i, dummy.matrix);
            }
            debris.instanceMatrix.needsUpdate = true;
            if (!alive && t > 2.5) debris.visible = false;
          }

          const px = renderer.domElement.height / (2 * Math.tan(camera.fov * Math.PI / 360));
          smoke.mat.uniforms.uPx.value = px;
          sparks.mat.uniforms.uPx.value = px;
          let anyAlive = false;
          for (let i = 0; i < smoke.n; i++) {
            if (smoke.life[i] <= 0) continue;
            anyAlive = true;
            smoke.life[i] -= dt / smoke.dur[i];
            const k = i * 3;
            smoke.vel[k] *= 1 - 0.8 * dt;
            smoke.vel[k + 2] *= 1 - 0.8 * dt;
            smoke.pos[k] += smoke.vel[k] * dt;
            smoke.pos[k + 1] += smoke.vel[k + 1] * dt;
            smoke.pos[k + 2] += smoke.vel[k + 2] * dt;
          }
          for (let i = 0; i < sparks.n; i++) {
            if (sparks.life[i] <= 0) continue;
            anyAlive = true;
            sparks.life[i] -= dt / sparks.dur[i];
            const k = i * 3;
            sparks.vel[k] *= 1 - 1.8 * dt;
            sparks.vel[k + 1] -= 16 * dt;
            sparks.vel[k + 2] *= 1 - 1.8 * dt;
            sparks.pos[k] += sparks.vel[k] * dt;
            sparks.pos[k + 1] += sparks.vel[k + 1] * dt;
            sparks.pos[k + 2] += sparks.vel[k + 2] * dt;
            if (sparks.pos[k + 1] < api.groundY + 0.05 && sparks.vel[k + 1] < 0) {
              sparks.pos[k + 1] = api.groundY + 0.05;
              sparks.vel[k + 1] *= -0.35;
            }
          }
          smoke.points.geometry.attributes.position.needsUpdate = true;
          smoke.points.geometry.attributes.aLife.needsUpdate = true;
          sparks.points.geometry.attributes.position.needsUpdate = true;
          sparks.points.geometry.attributes.aLife.needsUpdate = true;

          if (!anyAlive && t > 3) api.active = false;
        },
      };
      return api;
    }

    return { Wind, BikePhysics, buildBike, makeJetFX, makeExplosion };
  }

  /* ---------- the rig ----------
   *
   * One scene, one renderer, one bike, kept between flights. The canvas is
   * transparent and sits over the map; the camera is set from outside every
   * frame (see setCamera) to wherever the map's camera is, so what is drawn
   * here lands on the picture underneath at the right place and size. */

  function makeRig(canvas) {
    const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true, premultipliedAlpha: true });
    renderer.setClearColor(0x000000, 0);
    const scene = new THREE.Scene();
    // No fog: the map underneath has its own haze, and a fogged bike ten
    // metres away would be a bike behind glass.
    const camera = new THREE.PerspectiveCamera(42, 1, 0.1, 6000);

    // fable's light: a hemisphere for the sky and the ground, and a sun.
    scene.add(new THREE.HemisphereLight(0xbdd7f2, 0x62705a, 0.85));
    const sun = new THREE.DirectionalLight(0xfff2dd, 1.6);
    sun.position.set(700, 1000, -400);
    scene.add(sun);

    const bike = lib.buildBike();
    scene.add(bike.group);
    const fx = lib.makeJetFX(scene, bike.group, camera, renderer);
    const boom = lib.makeExplosion(scene, camera, renderer);
    const phys = new lib.BikePhysics();

    // soft blob shadow on the ground - crucial for judging altitude
    const shadow = new THREE.Mesh(
      new THREE.CircleGeometry(1.7, 24),
      new THREE.MeshBasicMaterial({ color: 0x000000, transparent: true, opacity: 0.35, depthWrite: false })
    );
    shadow.rotation.x = -Math.PI / 2;
    scene.add(shadow);

    /* The ground, up close. The satellite tiles stop at zoom 18, sixty
     * centimetres to the pixel, and a parked bike looks at them from three
     * metres, stretched sixteen times: a smear. A fine grain laid over the
     * ground around the bike gives the smear a surface - not a texture the
     * ground has, only the suggestion that it has one, grey either way so
     * the picture's colours stay its own - and it fades out by the height
     * at which the tiles are sharp by themselves, and towards its own edge
     * so that edge is never seen. It follows the bike on a grid of its own
     * tile so that it does not crawl. */
    const grain = (() => {
      const c = document.createElement('canvas');
      c.width = c.height = 256;
      const ctx = c.getContext('2d');
      const img = ctx.createImageData(256, 256);
      for (let i = 0; i < img.data.length; i += 4) {
        const v = 80 + Math.random() * 110;
        img.data[i] = img.data[i + 1] = img.data[i + 2] = v;
        img.data[i + 3] = 255;
      }
      ctx.putImageData(img, 0, 0);
      const tex = new THREE.CanvasTexture(c);
      tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
      const TILE = 4, SIZE = 240;
      const mat = new THREE.ShaderMaterial({
        uniforms: { uTex: { value: tex }, uOpacity: { value: 0 }, uRepeat: { value: SIZE / TILE } },
        vertexShader: `
          varying vec2 vUv; varying vec2 vPos;
          void main() { vUv = uv; vPos = position.xy; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`,
        fragmentShader: `
          uniform sampler2D uTex; uniform float uOpacity; uniform float uRepeat;
          varying vec2 vUv; varying vec2 vPos;
          void main() {
            float g = texture2D(uTex, vUv * uRepeat).r;
            float fade = 1.0 - smoothstep(50.0, 110.0, length(vPos));
            gl_FragColor = vec4(vec3(g), uOpacity * fade);
          }`,
        transparent: true,
        depthWrite: false,
      });
      const mesh = new THREE.Mesh(new THREE.PlaneGeometry(SIZE, SIZE), mat);
      mesh.rotation.x = -Math.PI / 2;
      mesh.position.y = 0.02;
      mesh.renderOrder = -1;
      scene.add(mesh);
      return {
        update() {
          mesh.position.x = Math.round(phys.pos.x / TILE) * TILE;
          mesh.position.z = Math.round(phys.pos.z / TILE) * TILE;
          mat.uniforms.uOpacity.value = TUNE.grain * clamp(1 - (phys.pos.y - 4) / 36, 0, 1);
          mesh.visible = mat.uniforms.uOpacity.value > 0.01;
        }
      };
    })();

    const _fh = new THREE.Vector3(), _look = new THREE.Vector3(), _tmp = new THREE.Vector3();
    const _up = new THREE.Vector3(0, 1, 0), _q = new THREE.Quaternion();
    const FLIP = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), Math.PI);

    // The chase camera's own position, smoothed across frames. What it wants
    // is recomputed every frame; where it is follows.
    const camPos = new THREE.Vector3();
    let camInit = false;
    let acc = 0;
    let camMode = 0;   // 0 chase, 1 onboard
    let time = 0;

    /** Put the bike on the ground at (x, z), facing `yaw` radians. */
    function place(x, z, yaw) {
      phys.reset(x, z, yaw);
      bike.group.visible = true;
      camInit = false;
      acc = 0;
      syncMesh();
    }

    function syncMesh() {
      bike.group.position.copy(phys.pos);
      bike.group.quaternion.copy(phys.quat);
    }

    /** Advance the flight by `dt` seconds of wall clock: the physics at its
     *  own fixed step, then the mesh, the flames, the dust, the shadow and
     *  the crash. Returns the crash reason if one happened this frame. */
    function advance(dt) {
      time += dt;
      acc += dt;
      let n = 0;
      while (acc > STEP && n < 12) { phys.step(STEP); acc -= STEP; n++; }
      if (n === 12) acc = 0;   // a stalled tab does not owe a second of physics

      const crashReason = phys.consumeCrashEvent();
      if (crashReason) {
        boom.trigger(phys.pos, groundAt(phys.pos.x, phys.pos.z));
        bike.group.visible = false;
      }
      if (!phys.crashed && !bike.group.visible) bike.group.visible = true;

      syncMesh();

      const gh = groundAt(phys.pos.x, phys.pos.z);
      const agl = phys.pos.y - gh;

      fx.update(dt, {
        jets: {
          rear: phys.spoolRear + phys.ab * 0.9,
          left: phys.spoolLift * 0.7 + (phys.pulses.L > 0 ? 1.3 : 0) + Math.max(0, -phys.steer) * 0.25,
          right: phys.spoolLift * 0.7 + (phys.pulses.R > 0 ? 1.3 : 0) + Math.max(0, phys.steer) * 0.25,
          center: phys.spoolLift * 0.7 + (phys.pulses.C > 0 ? 1.3 : 0),
        },
        pos: phys.pos, quat: phys.quat, vel: phys.vel,
        vec: phys.nozzle,
        agl, groundY: gh,
      });
      boom.update(dt);

      shadow.position.set(phys.pos.x, gh + 0.08, phys.pos.z);
      shadow.scale.setScalar(1 + agl * 0.02);
      shadow.material.opacity = 0.38 * clamp(1 - agl / 70, 0, 1);
      shadow.visible = bike.group.visible && shadow.material.opacity > 0.02;

      grain.update();
      return crashReason;
    }

    /** Where the camera wants to be this frame, in world metres, and where it
     *  looks: fable's chase camera, or the rider's eyes. Writes `out` as
     *  { x, y, z, dx, dy, dz, roll } with the look direction a unit vector
     *  and the roll in degrees, right wing down positive. The map cannot
     *  look up, so the chase camera is kept at least camDip degrees above
     *  its target: raised, not tilted, so the bike stays where it was on the
     *  screen. */
    function wantCamera(dt, out) {
      const speed = phys.vel.length();
      if (camMode === 0) {
        _fh.set(0, 0, 1).applyQuaternion(phys.quat);
        _fh.y *= 0.25;
        _fh.normalize();
        const dist = TUNE.camBack + speed * TUNE.camBackV;
        _look.copy(phys.pos).addScaledVector(_fh, -dist);
        _look.y += TUNE.camUp + speed * TUNE.camUpV;
        _tmp.copy(phys.pos).addScaledVector(phys.vel, 0.1);
        _tmp.y += 0.8;
        // the floor on the dip: the horizontal distance to the target times
        // the tangent of the least dip, above the target
        const hd = Math.hypot(_look.x - _tmp.x, _look.z - _tmp.z);
        const minY = _tmp.y + hd * Math.tan(TUNE.camDip * Math.PI / 180);
        if (_look.y < minY) _look.y = minY;
        const minG = groundAt(_look.x, _look.z) + 1.0;
        if (_look.y < minG) _look.y = minG;
        if (!camInit) { camPos.copy(_look); camInit = true; }
        else camPos.lerp(_look, 1 - Math.exp(-TUNE.camFollow * dt));
        // the crash's shake, and a whisper of the gusts
        const buffet = Math.min(0.12, phys.wind.gustMag * 0.03) * clamp(speed * 0.02, 0, 1);
        const shake = Math.max(boom.shake, buffet);
        out.x = camPos.x; out.y = camPos.y; out.z = camPos.z;
        if (shake > 0.01) {
          out.x += (Math.random() - 0.5) * shake;
          out.y += (Math.random() - 0.5) * shake;
          out.z += (Math.random() - 0.5) * shake;
        }
        _tmp.sub(camPos).normalize();
        out.dx = _tmp.x; out.dy = _tmp.y; out.dz = _tmp.z;
        // the picture leans with the bike, a little: right wing down is
        // body +X (the rider's left) up, so the sign is the bank's own
        const right = _fh.set(1, 0, 0).applyQuaternion(phys.quat);
        out.roll = Math.asin(clamp(right.y, -1, 1)) * 180 / Math.PI;
      } else {
        const e = TUNE.eyePos;
        _look.set(e[0], e[1], e[2]).applyQuaternion(phys.quat).add(phys.pos);
        out.x = _look.x; out.y = _look.y; out.z = _look.z;
        // the rider's line of sight: the nose, dipped by eyeDip about the
        // body's own sideways axis, so it dips with the bike however it lies
        const dip = TUNE.eyeDip * Math.PI / 180;
        _tmp.set(0, -Math.sin(dip), Math.cos(dip)).applyQuaternion(phys.quat);
        out.dx = _tmp.x; out.dy = _tmp.y; out.dz = _tmp.z;
        const right = _fh.set(1, 0, 0).applyQuaternion(phys.quat);
        out.roll = Math.asin(clamp(right.y, -1, 1)) * 180 / Math.PI;
        camInit = false;
      }
      return out;
    }

    /** Put the scene's camera where the map's is: position in world metres,
     *  a unit look direction, a roll about it in degrees (right wing down
     *  positive), and the vertical field of view of the visible window. */
    function setCamera(x, y, z, dx, dy, dz, roll, vfov) {
      camera.position.set(x, y, z);
      _tmp.set(x + dx, y + dy, z + dz);
      camera.up.copy(_up);
      camera.lookAt(_tmp);
      if (roll) {
        _tmp.set(dx, dy, dz).normalize();
        // right wing down turns the picture counter-clockwise, which about
        // the look axis (pointing away from the viewer) is a positive turn
        _q.setFromAxisAngle(_tmp, roll * Math.PI / 180);
        camera.quaternion.premultiply(_q);
      }
      if (Math.abs(camera.fov - vfov) > 0.01) {
        camera.fov = vfov;
        camera.updateProjectionMatrix();
      }
    }

    function resize(w, h, dpr) {
      renderer.setPixelRatio(dpr);
      renderer.setSize(w, h, false);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
    }

    function render() { renderer.render(scene, camera); }

    function setCamMode(m) { camMode = m; camInit = false; }

    return {
      phys, bike, fx, boom, shadow, camera, renderer, scene,
      place, advance, wantCamera, setCamera, resize, render, setCamMode,
      getCamMode: () => camMode,
      /** The heading of the bike, degrees clockwise from north, with north
       *  being -z. */
      heading() {
        _tmp.set(0, 0, 1).applyQuaternion(phys.quat);
        return (Math.atan2(_tmp.x, -_tmp.z) * 180 / Math.PI + 360) % 360;
      },
      /** Bank, degrees, right wing down positive. */
      bank() {
        _tmp.set(1, 0, 0).applyQuaternion(phys.quat);
        return Math.asin(clamp(_tmp.y, -1, 1)) * 180 / Math.PI;
      },
      /** Nose above the horizon, degrees. */
      nose() {
        _tmp.set(0, 0, 1).applyQuaternion(phys.quat);
        return Math.asin(clamp(_tmp.y, -1, 1)) * 180 / Math.PI;
      },
    };
  }

  /** The yaw, radians about +Y, that points the bike along a compass
   *  bearing: forward is +z, north is -z, so a bearing b (clockwise from
   *  north) is a rotation of pi - b. */
  const yawFor = (bearingDeg) => Math.PI - bearingDeg * Math.PI / 180;

  return { load, ready: () => !!lib, makeRig, TUNE, STEP, yawFor };
})();
