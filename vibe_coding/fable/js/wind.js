// Atmospheric wind model (node-safe, no DOM):
//   - A seeded prevailing wind vector.
//   - MIL-F-8785C Dryden turbulence: Gaussian white noise driven through
//     shaping filters — a first-order ODE for the along-wind component u and
//     second-order ODEs for the crosswind v and vertical w — integrated in
//     real time at the physics rate. Filter time constants follow L/V and the
//     intensities follow the low-altitude spec (sigma_w = 0.1*W20 etc.), so
//     turbulence gets rougher near the ground and in stronger wind.
//   - Ridge lift: wind flowing up a terrain slope acquires a vertical
//     component that decays with height — hug a windward mountain face and
//     the mountain carries you.
import * as THREE from 'three';
import { mulberry32 } from './noise.js';
import { terrainHeight } from './world.js';

export class Wind {
  constructor(seed = 77, opts = {}) {
    this.rand = mulberry32(seed);
    const dir = opts.dir ?? this.rand() * Math.PI * 2;
    this.speed = opts.speed ?? 3 + this.rand() * 5;
    this.W = new THREE.Vector3(Math.sin(dir) * this.speed, 0, Math.cos(dir) * this.speed);
    this.u = 0;                   // along-wind gust state (1st order)
    this.v = 0; this.vd = 0;      // crosswind gust states (2nd order)
    this.w = 0; this.wd = 0;      // vertical gust states (2nd order)
    this.gustGain = 1;
    this.vec = new THREE.Vector3();   // latest total wind at the queried point
    this.gustMag = 0;                 // latest |gust| (for camera buffet etc.)
  }

  gauss() {
    const a = Math.max(this.rand(), 1e-12);
    return Math.sqrt(-2 * Math.log(a)) * Math.cos(2 * Math.PI * this.rand());
  }

  // advance the gust ODEs by dt (airspeed V m/s, height agl m) and compose the
  // total wind vector at world (x, z) into this.vec.
  update(dt, V, x, z, agl) {
    const W = this.W;
    if (this.speed < 0.01) { this.vec.set(0, 0, 0); this.gustMag = 0; return this.vec; }

    // MIL-F-8785C low-altitude scale lengths & intensities (spec is in feet)
    const hft = Math.min(1000, Math.max(10, agl * 3.281));
    const coef = Math.pow(0.177 + 0.000823 * hft, 1.2);
    const Lw = Math.max(3, agl);              // m
    const Lu = Math.max(10, agl / coef);      // m (= Lv)
    const sw = 0.1 * this.speed * this.gustGain;
    const su = sw / Math.pow(0.177 + 0.000823 * hft, 0.4);

    const Vc = Math.max(5, V);
    // u: first-order shaping filter  u' = -(V/Lu) u + su*sqrt(2V/(pi*Lu)) * eta
    const eta = () => this.gauss() / Math.sqrt(dt);
    this.u += (-(Vc / Lu) * this.u + su * Math.sqrt(2 * Vc / (Math.PI * Lu)) * eta()) * dt;
    // v, w: second-order filters  (1 + b s)^2 x = eta,  y = K (x + a x')
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

    // ridge lift: the vertical velocity the terrain forces on the moving air
    const e = 12;
    const dhx = (terrainHeight(x + e, z) - terrainHeight(x - e, z)) / (2 * e);
    const dhz = (terrainHeight(x, z + e) - terrainHeight(x, z - e)) / (2 * e);
    const ridge = Math.min(6, Math.max(0, W.x * dhx + W.z * dhz)) * Math.exp(-Math.max(0, agl) / 45);

    // compose in wind axes: u along the mean wind, v horizontal-perpendicular
    const inv = 1 / this.speed;
    const ux = W.x * inv, uz = W.z * inv;
    const gx = ux * this.u + uz * gv;
    const gz = uz * this.u - ux * gv;
    this.gustMag = Math.hypot(gx, gw, gz);
    this.vec.set(W.x + gx, gw + ridge, W.z + gz);
    return this.vec;
  }
}
