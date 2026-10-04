// Crash effects: a kerosene fireball (HDR additive, blooms), a rising smoke column, burning
// debris that bounces off the terrain, a flash light; over water a white column of spray.

import * as THREE from 'three';
import { rng } from '../vmath.js';

const N = 900;

export class Explosion {
  constructor(scene, hAt) {
    this.hAt = hAt;
    this.R = rng(99);
    this.p = new Float32Array(N * 3); this.v = new Float32Array(N * 3);
    this.age = new Float32Array(N); this.life = new Float32Array(N); this.kind = new Uint8Array(N); this.size = new Float32Array(N);
    this.n = 0;
    const g = new THREE.BufferGeometry();
    this.aP = new THREE.BufferAttribute(new Float32Array(N * 3), 3); this.aS = new THREE.BufferAttribute(new Float32Array(N), 1);
    this.aC = new THREE.BufferAttribute(new Float32Array(N * 4), 4);
    g.setAttribute('position', this.aP); g.setAttribute('size', this.aS); g.setAttribute('col', this.aC);
    const mk = (additive) => new THREE.ShaderMaterial({
      uniforms: { uScale: { value: 800 }, uAdd: { value: additive ? 1 : 0 } },
      vertexShader: /* glsl */`attribute float size; attribute vec4 col; uniform float uScale, uAdd; varying vec4 vC;
        void main(){ vec4 mv = modelViewMatrix*vec4(position,1.0); gl_Position = projectionMatrix*mv;
          // additive pass draws only hot particles (alpha>1 marks fire), normal pass only smoke
          float fire = step(1.0, col.a);
          gl_PointSize = (uAdd > 0.5 ? fire : 1.0 - fire) * clamp(size*uScale/-mv.z, 0.0, 700.0); vC = col; }`,
      fragmentShader: /* glsl */`uniform float uAdd; varying vec4 vC;
        void main(){ vec2 d = gl_PointCoord-0.5; float r = length(d); if (r > 0.5) discard;
          float a = smoothstep(0.5, 0.0, r);
          if (uAdd > 0.5) gl_FragColor = vec4(vC.rgb * a * a, 1.0);
          else gl_FragColor = vec4(vC.rgb, a * vC.a); }`,
      transparent: true, depthWrite: false,
      blending: additive ? THREE.AdditiveBlending : THREE.NormalBlending,
    });
    this.smoke = new THREE.Points(g, mk(false));
    this.fire = new THREE.Points(g, mk(true));
    this.smoke.frustumCulled = this.fire.frustumCulled = false;
    this.fire.renderOrder = 7; this.smoke.renderOrder = 6;
    scene.add(this.smoke, this.fire);
    this.light = new THREE.PointLight(0xff8a3a, 0, 120, 1.6);
    scene.add(this.light);
    this.t = 0;
  }
  add(kind, x, y, z, vx, vy, vz, life, size) {
    if (this.n >= N) return;
    const i = this.n++;
    this.p.set([x, y, z], i * 3); this.v.set([vx, vy, vz], i * 3);
    this.age[i] = 0; this.life[i] = life; this.kind[i] = kind; this.size[i] = size;
  }
  // kind: 0 fireball, 1 smoke, 2 burning debris, 3 spray, 4 steam
  trigger(pos, vel, water) {
    const R = this.R;
    this.t = 0; this.origin = pos.slice(); this.water = water;
    const [x, y, z] = pos;
    const bv = vel.map((c) => c * 0.35);
    if (water) {
      for (let i = 0; i < 380; i++) {
        const a = R() * 6.28, s = R() * 7;
        this.add(3, x, y, z, Math.cos(a) * s + bv[0], 6 + R() * 18, Math.sin(a) * s + bv[2], 1.5 + R() * 1.5, 0.3 + R() * 0.5);
      }
      for (let i = 0; i < 90; i++) this.add(4, x + (R() - 0.5) * 4, y + R() * 3, z + (R() - 0.5) * 4, bv[0] * 0.5, 2 + R() * 3, bv[2] * 0.5, 3 + R() * 3, 2 + R() * 2);
    } else {
      for (let i = 0; i < 260; i++) {
        const a = R() * 6.28, b = Math.acos(R() * 2 - 1), s = 3 + R() * 11;
        this.add(0, x, y + 0.5, z, Math.sin(b) * Math.cos(a) * s + bv[0], Math.abs(Math.cos(b)) * s * 0.8 + 2 + bv[1] * 0.2, Math.sin(b) * Math.sin(a) * s + bv[2], 0.7 + R() * 0.9, 1.2 + R() * 1.8);
      }
      for (let i = 0; i < 160; i++) this.add(1, x + (R() - 0.5) * 3, y + R() * 2, z + (R() - 0.5) * 3, bv[0] * 0.3 + (R() - 0.5) * 3, 3 + R() * 4, bv[2] * 0.3 + (R() - 0.5) * 3, 4 + R() * 5, 2 + R() * 2);
      for (let i = 0; i < 70; i++) {
        const a = R() * 6.28, s = 5 + R() * 16;
        this.add(2, x, y + 0.5, z, Math.cos(a) * s + bv[0], 5 + R() * 14, Math.sin(a) * s + bv[2], 2.5 + R() * 2.5, 0.25 + R() * 0.2);
      }
      this.light.position.set(x, y + 3, z);
    }
  }
  step(dt, wind) {
    this.t += dt;
    this.light.intensity = this.water ? 0 : Math.max(0, 3000 * Math.exp(-this.t * 2.2) * (0.85 + 0.15 * Math.sin(this.t * 40)));
    let w = 0;
    const P = this.p, V = this.v;
    for (let i = 0; i < this.n; i++) {
      const age = this.age[i] + dt;
      if (age > this.life[i]) continue;
      const k = this.kind[i];
      let vx = V[i * 3], vy = V[i * 3 + 1], vz = V[i * 3 + 2];
      if (k === 0) { const d = Math.exp(-dt * 3.5); vx *= d; vz *= d; vy = vy * d + 3 * dt; }        // hot gas: drag + buoyancy
      else if (k === 1 || k === 4) { vx += (wind[0] - vx) * dt * 0.6; vz += (wind[2] - vz) * dt * 0.6; vy += (2.5 - vy) * dt * 0.5; }
      else { vy -= 9.81 * dt; if (k === 3) { vx *= 1 - dt * 0.4; vz *= 1 - dt * 0.4; } }
      let x = P[i * 3] + vx * dt, y = P[i * 3 + 1] + vy * dt, z = P[i * 3 + 2] + vz * dt;
      const gy = this.hAt(x, z);
      if (y < gy) {
        if (k === 2) { y = gy; vy = -vy * 0.35; vx *= 0.6; vz *= 0.6; }
        else if (k === 3) continue;
        else { y = gy; vy = Math.abs(vy) * 0.2; }
      }
      P[w * 3] = x; P[w * 3 + 1] = y; P[w * 3 + 2] = z; V[w * 3] = vx; V[w * 3 + 1] = vy; V[w * 3 + 2] = vz;
      this.age[w] = age; this.life[w] = this.life[i]; this.kind[w] = k; this.size[w] = this.size[i];
      w++;
    }
    this.n = w;
    const aP = this.aP.array, aS = this.aS.array, aC = this.aC.array;
    for (let i = 0; i < this.n; i++) {
      aP[i * 3] = P[i * 3]; aP[i * 3 + 1] = P[i * 3 + 1]; aP[i * 3 + 2] = P[i * 3 + 2];
      const u = this.age[i] / this.life[i], k = this.kind[i];
      let c;
      if (k === 0) { const h = 1 - u; c = [9 * h * h + 0.5, 3.2 * h * h * h + 0.15, 0.6 * h ** 4, 1.5]; aS[i] = this.size[i] * (1 + 3 * u); }
      else if (k === 2) { const h = Math.max(0, 1 - u * 1.4); c = [6 * h + 0.05, 2 * h * h + 0.03, 0.3 * h, h > 0.02 ? 1.2 : 0.9]; aS[i] = this.size[i]; }
      else if (k === 1) { c = [0.05, 0.045, 0.04, 0.55 * Math.min(1, this.age[i] * 2) * (1 - u)]; aS[i] = this.size[i] * (1 + 4 * u); }
      else if (k === 3) { c = [0.95, 0.97, 1.0, 0.8 * (1 - u)]; aS[i] = this.size[i]; }
      else { c = [0.9, 0.92, 0.95, 0.3 * (1 - u)]; aS[i] = this.size[i] * (1 + 3 * u); }
      aC.set(c, i * 4);
    }
    this.smoke.geometry.setDrawRange(0, this.n);
    this.aP.needsUpdate = this.aS.needsUpdate = this.aC.needsUpdate = true;
  }
  clear() { this.n = 0; this.light.intensity = 0; this.smoke.geometry.setDrawRange(0, 0); }
}
