// Effects: a plume on each of the five nozzles, sized by the thrust that really leaves it (shock
// diamonds, after-burner flame on the rear one), dust / spray / snow kicked up where a jet really
// hits the surface, wing-tip vapour in hard turns, explosion and splash.
import * as THREE from '../../lib/three.module.min.js';
import { U } from './env.js';
import { groundAt, waterAt } from '../sim/terrain.js';
import { ENG, LIFT, REAR, G } from '../sim/bike.js';
import { qrot, add, scale, len } from '../sim/vmath.js';

const PLUME_FRAG = /* glsl */`
precision highp float;
uniform float uPow, uAb, uTime, uSeed;
varying vec2 vUv; varying vec3 vN; varying vec3 vV;
void main() {
  float x = vUv.y;                                   // 0 at the nozzle .. 1 at the tip
  float core = pow(abs(dot(normalize(vN), normalize(vV))), 1.6);
  float flick = 0.9 + 0.1 * sin(uTime * 61.0 + uSeed * 9.0 + x * 20.0);
  // shock diamonds: standing pressure pattern, spacing grows with thrust
  float dia = pow(0.5 + 0.5 * cos(x * (30.0 - 12.0 * uPow) - 0.6), 5.0) * exp(-x * 3.2);
  vec3 dry = vec3(0.35, 0.55, 1.0) * (0.10 * exp(-x * 5.0) + 0.25 * dia * uPow) * uPow;
  vec3 ab = (mix(vec3(1.0, 0.55, 0.22), vec3(0.55, 0.45, 1.0), smoothstep(0.0, 0.5, x)) * exp(-x * 2.3) * 2.4 + vec3(1.4, 1.1, 0.9) * dia * 4.0) * uAb;
  float fade = smoothstep(1.0, 0.75, x);
  gl_FragColor = vec4((dry + ab) * core * flick * fade * 3.0, 1.0);
}`;

const PART_VERT = /* glsl */`
attribute float aSize; attribute vec4 aColor;
varying vec4 vColor;
uniform float uScale;
void main() {
  vColor = aColor;
  vec4 mv = viewMatrix * vec4(position, 1.0);
  gl_PointSize = min(uScale * aSize / max(-mv.z, 0.5), 900.0);
  gl_Position = projectionMatrix * mv;
}`;
const PART_FRAG = /* glsl */`
precision highp float;
varying vec4 vColor;
void main() {
  vec2 p = gl_PointCoord * 2.0 - 1.0; float r = dot(p, p);
  if (r > 1.0) discard;
  float a = (1.0 - r); a *= a;
  gl_FragColor = vec4(vColor.rgb, vColor.a * a);
}`;

class Particles {
  constructor(scene, n, additive) {
    this.n = n; this.i = 0;
    this.pos = new Float32Array(n * 3); this.vel = new Float32Array(n * 3); this.col = new Float32Array(n * 4); this.size = new Float32Array(n);
    this.life = new Float32Array(n); this.max = new Float32Array(n); this.grow = new Float32Array(n); this.drag = new Float32Array(n); this.grav = new Float32Array(n); this.a0 = new Float32Array(n);
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(this.pos, 3)); g.setAttribute('aColor', new THREE.BufferAttribute(this.col, 4)); g.setAttribute('aSize', new THREE.BufferAttribute(this.size, 1));
    this.mat = new THREE.ShaderMaterial({ uniforms: { uScale: { value: 800 } }, vertexShader: PART_VERT, fragmentShader: PART_FRAG, transparent: true, depthWrite: false, blending: additive ? THREE.AdditiveBlending : THREE.NormalBlending });
    this.pts = new THREE.Points(g, this.mat); this.pts.frustumCulled = false; this.pts.renderOrder = additive ? 6 : 5;
    scene.add(this.pts); this.geo = g;
  }
  spawn(p, v, c, a, size, life, grow = 0, drag = 1, grav = 0) {
    const i = this.i; this.i = (i + 1) % this.n;
    this.pos.set(p, i * 3); this.vel.set(v, i * 3); this.col.set([c[0], c[1], c[2], a], i * 4);
    this.size[i] = size; this.life[i] = life; this.max[i] = life; this.grow[i] = grow; this.drag[i] = drag; this.grav[i] = grav; this.a0[i] = a;
  }
  update(dt, wind) {
    for (let i = 0; i < this.n; i++) {
      if (this.life[i] <= 0) { if (this.col[i * 4 + 3] !== 0) this.col[i * 4 + 3] = 0; continue; }
      this.life[i] -= dt;
      const k = Math.exp(-this.drag[i] * dt), j = i * 3;
      this.vel[j] = wind[0] + (this.vel[j] - wind[0]) * k; this.vel[j + 1] = this.vel[j + 1] * k - this.grav[i] * dt; this.vel[j + 2] = wind[2] + (this.vel[j + 2] - wind[2]) * k;
      this.pos[j] += this.vel[j] * dt; this.pos[j + 1] += this.vel[j + 1] * dt; this.pos[j + 2] += this.vel[j + 2] * dt;
      this.size[i] += this.grow[i] * dt;
      const t = Math.max(0, this.life[i] / this.max[i]);
      this.col[i * 4 + 3] = this.a0[i] * Math.min(1, t * 1.6) * Math.min(1, (1 - t) * 12);
    }
    this.geo.attributes.position.needsUpdate = true; this.geo.attributes.aColor.needsUpdate = true; this.geo.attributes.aSize.needsUpdate = true;
  }
}

export class FX {
  constructor(scene, model) {
    this.scene = scene;
    this.soft = new Particles(scene, 2600, false);
    this.fire = new Particles(scene, 900, true);
    // plumes
    this.plumes = model.nozzles.map((p, i) => {
      const k = p.r / 0.078;
      const g = new THREE.CylinderGeometry(0.035 * k, 0.08 * k, 1, 16, 12, true); g.translate(0, 0.5, 0); g.rotateZ(Math.PI / 2);   // along -x, uv.y = 0 at the nozzle
      const m = new THREE.ShaderMaterial({
        uniforms: { uPow: { value: 0 }, uAb: { value: 0 }, uTime: U.uTime, uSeed: { value: i } },
        vertexShader: 'varying vec2 vUv; varying vec3 vN; varying vec3 vV; void main(){ vUv = uv; vec4 mv = modelViewMatrix * vec4(position, 1.0); vN = normalMatrix * normal; vV = -mv.xyz; gl_Position = projectionMatrix * mv; }',
        fragmentShader: PLUME_FRAG, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, side: THREE.DoubleSide,
      });
      const mesh = new THREE.Mesh(g, m); mesh.renderOrder = 7; p.noz.add(mesh);
      return { mesh, m };
    });
    this.light = new THREE.PointLight(0xffa050, 0, 60, 1.6); scene.add(this.light);
    this.flash = new THREE.PointLight(0xffc080, 0, 400, 1.2); scene.add(this.flash);
    this.flashT = 0; this.acc = 0; this.shake = 0;
    this.washStrength = 0; this.washPos = [0, 0];
    this.debris = [];
    const dg = new THREE.TetrahedronGeometry(0.25), dm = new THREE.MeshStandardMaterial({ color: 0x222222, roughness: 0.6 });
    for (let i = 0; i < 14; i++) { const m = new THREE.Mesh(dg, dm); m.visible = false; scene.add(m); this.debris.push({ m, v: [0, 0, 0], w: [0, 0, 0], t: 0 }); }
  }

  update(dt, b, wind, alive) {
    const r = Math.random;
    let wash = 0, wx = 0, wz = 0;
    const ab = alive ? b.abOut : 0, sw = Math.max(b.tilt, Math.PI / 2);
    for (let i = 0; i < 5; i++) {
      // thrust of this nozzle, its exit and the direction its gas leaves in (body frame)
      const rear = i === 4, T = rear ? b.R : b.L[i];
      const pw = alive ? Math.min(1, rear ? b.R / (1 + ENG.abGain * b.eng.ab) / (ENG.tmax * 0.75) : T / 1500) : 0;
      const pl = this.plumes[i];
      pl.m.uniforms.uPow.value = pw; pl.m.uniforms.uAb.value = rear ? ab * 0.55 : 0;
      const L = rear ? 0.6 + 1.5 * pw + 2.2 * ab : 0.35 + 1.5 * pw;
      pl.mesh.scale.set(L, 1 + (rear ? 0.2 * ab : 0), 1 + (rear ? 0.2 * ab : 0));
      pl.mesh.visible = alive && pw > 0.02;
      if (!alive) continue;
      // where does this jet hit the surface?
      let dirB, exit;
      if (rear) { const a = REAR.cant + b.rp, cy = Math.cos(b.ry); dirB = [-Math.cos(a) * cy, -Math.sin(a) * cy, -Math.sin(b.ry)]; exit = REAR.p; }
      else { const th = sw + b.va[i]; dirB = [-Math.cos(th), -Math.sin(th), 0]; exit = LIFT[i].p; }
      const dW = qrot(b.q, dirB);
      if (dW[1] > -0.2) continue;
      const exitW = b.toWorld(exit);
      const reach = 5 + 13 * pw;
      let hit = null, isWater = false;
      for (let s = 0.5; s < reach; s += 0.8) {
        const q = add(exitW, scale(dW, s)), gy = groundAt(q[0], q[2]), wl = waterAt(q[0], q[2]), sy = Math.max(gy, wl);
        if (q[1] <= sy) { hit = [q[0], sy, q[2]]; isWater = wl > gy; hit.d = s; break; }
      }
      if (!hit) continue;
      const k = Math.max(0, T / 1450 - 0.12) * (1 - hit.d / reach);
      if (isWater) { wash += k; wx += hit[0] * k; wz += hit[2] * k; }
      this.acc += k * dt * 70;
      while (this.acc > 1) {
        this.acc -= 1;
        const a = r() * 6.283, sp = (6 + 16 * r()) * (0.4 + k);
        const v = [Math.cos(a) * sp + b.v[0] * 0.3, 1.5 + 3.5 * r() * k, Math.sin(a) * sp + b.v[2] * 0.3];
        const p = [hit[0] + Math.cos(a) * 0.4, hit[1] + 0.15, hit[2] + Math.sin(a) * 0.4];
        if (isWater) this.soft.spawn(p, [v[0] * 0.8, 3 + 7 * r() * k, v[2] * 0.8], [0.82, 0.88, 0.92], 0.42, 0.5 + r() * 0.5, 0.7 + r() * 0.5, 2.8, 1.6, 7);
        else if (hit[1] > 2380) this.soft.spawn(p, v, [0.93, 0.95, 0.98], 0.5, 0.6 + r() * 0.6, 1.0 + r() * 0.9, 3.2, 1.3, 1.2);
        else { const c = 0.34 + 0.14 * r(); this.soft.spawn(p, v, [c, c * 0.86, c * 0.64], 0.34, 0.7 + r() * 0.7, 1.2 + r() * 1.4, 3.4, 1.1, 0.4); }
      }
    }
    this.washStrength += ((wash > 0 ? Math.min(1, wash * 0.8) : 0) - this.washStrength) * Math.min(1, dt * 4);
    if (wash > 0) this.washPos = [wx / wash, wz / wash];
    // after-burner light on the surroundings
    const lp = b.toWorld([REAR.p[0] - 1.2, REAR.p[1], 0]);
    this.light.intensity = ab * 30; this.light.position.set(lp[0], lp[1], lp[2]);
    // wing-tip vapour in hard, fast turns
    if (alive) {
      const g = len([b.acc[0], b.acc[1] + G, b.acc[2]]) / G;
      if (g > 2.0 && b.airspeed > 45) for (const z of [-1.27, 1.27]) if (r() < dt * 90) this.soft.spawn(b.toWorld([-0.2, 0.86, z]), scale(b.v, 0.05), [0.95, 0.96, 1.0], 0.16 * Math.min(1, g - 1.9), 0.22, 0.55, 0.9, 0.5, 0);
    }
    if (this.flashT > 0) { this.flashT -= dt; this.flash.intensity = 60000 * Math.max(0, this.flashT / 0.5) ** 2; } else this.flash.intensity = 0;
    this.shake = Math.max(0, this.shake - dt * 1.6);
    for (const d of this.debris) {
      if (d.t <= 0) { d.m.visible = false; continue; }
      d.t -= dt; d.v[1] -= 9.8 * dt;
      d.m.position.x += d.v[0] * dt; d.m.position.y += d.v[1] * dt; d.m.position.z += d.v[2] * dt;
      d.m.rotation.x += d.w[0] * dt; d.m.rotation.y += d.w[1] * dt;
      const gy = groundAt(d.m.position.x, d.m.position.z);
      if (d.m.position.y < gy + 0.1) { d.m.position.y = gy + 0.1; d.v[0] *= 0.5; d.v[1] *= -0.35; d.v[2] *= 0.5; }
      if (r() < dt * 20) this.soft.spawn([d.m.position.x, d.m.position.y, d.m.position.z], [0, 1, 0], [0.1, 0.1, 0.1], 0.5, 0.6, 1.2, 1.6, 1, -0.5);
    }
    this.soft.update(dt, wind); this.fire.update(dt, wind);
  }

  explode(p, v, water) {
    const r = Math.random;
    this.shake = 1; this.flashT = water ? 0 : 0.5; this.flash.position.set(p[0], p[1] + 2, p[2]);
    if (water) {
      for (let i = 0; i < 260; i++) { const a = r() * 6.283, s = 2 + 9 * r(); this.soft.spawn([p[0], p[1], p[2]], [Math.cos(a) * s + v[0] * 0.2, 8 + 22 * r(), Math.sin(a) * s + v[2] * 0.2], [0.85, 0.9, 0.95], 0.6, 0.6 + r(), 1.2 + 1.2 * r(), 2.2, 0.8, 11); }
      return;
    }
    for (let i = 0; i < 130; i++) {
      const a = r() * 6.283, e = r() * 1.4, s = 4 + 20 * r();
      const d = [Math.cos(a) * Math.cos(e) * s + v[0] * 0.25, Math.sin(e) * s + 4, Math.sin(a) * Math.cos(e) * s + v[2] * 0.25];
      this.fire.spawn([p[0], p[1] + 0.5, p[2]], d, [3.2, 1.2 + r(), 0.25], 0.9, 1.2 + 2 * r(), 0.5 + 0.9 * r(), 5, 2.4, -3);
    }
    for (let i = 0; i < 90; i++) {
      const a = r() * 6.283, s = 2 + 8 * r(), c = 0.05 + 0.08 * r();
      this.soft.spawn([p[0], p[1] + 1, p[2]], [Math.cos(a) * s + v[0] * 0.15, 3 + 9 * r(), Math.sin(a) * s + v[2] * 0.15], [c, c, c], 0.75, 1.5 + 2 * r(), 2.5 + 3 * r(), 3.5, 0.9, -2.2);
    }
    for (let i = 0; i < 70; i++) { const a = r() * 6.283, e = r() * 1.5, s = 15 + 40 * r(); this.fire.spawn([p[0], p[1] + 0.5, p[2]], [Math.cos(a) * Math.cos(e) * s, Math.sin(e) * s, Math.sin(a) * Math.cos(e) * s], [4, 2.4, 0.6], 1, 0.16, 0.8 + r(), 0, 0.6, 12); }
    for (const d of this.debris) { d.t = 4 + 2 * r(); d.m.visible = true; d.m.position.set(p[0], p[1] + 0.5, p[2]); d.m.scale.setScalar(0.5 + 1.5 * r()); const a = r() * 6.283, s = 5 + 16 * r(); d.v = [Math.cos(a) * s + v[0] * 0.4, 5 + 12 * r(), Math.sin(a) * s + v[2] * 0.4]; d.w = [r() * 12, r() * 12, 0]; }
  }
}
