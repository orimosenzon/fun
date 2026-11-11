// Things that make the air visible: the scarf, vortex trails off the wingtips, speed streaks,
// motes rising in the thermals, the shadow on the ground, and bursts (spray, dust, feathers).
import * as THREE from '../../lib/three.module.min.js';
import { groundAt, normalAt, thermals, thermalCentre, CLOUDBASE, windAt } from '../sim/world.js';
import { mulberry32 } from '../sim/noise.js';
import { clamp, sstep } from '../sim/vmath.js';

const V = new THREE.Vector3(), W = new THREE.Vector3(), T = new THREE.Vector3();

// a camera-facing strip through a list of points; alpha fades head to tail
class Ribbon {
  constructor(scene, n, color, width, additive = true, opacity = 1) {
    this.n = n; this.width = width; this.pts = Array.from({ length: n }, () => new THREE.Vector3()); this.alpha = new Float32Array(n);
    this.pos = new Float32Array(n * 6); this.a = new Float32Array(n * 2);
    const idx = []; for (let i = 0; i < n - 1; i++) idx.push(i * 2, i * 2 + 1, i * 2 + 2, i * 2 + 1, i * 2 + 3, i * 2 + 2);
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(this.pos, 3)); g.setAttribute('aA', new THREE.BufferAttribute(this.a, 1)); g.setIndex(idx);
    this.mat = new THREE.ShaderMaterial({
      uniforms: { uCol: { value: new THREE.Color(color) }, uOp: { value: opacity } }, transparent: true, depthWrite: false, side: THREE.DoubleSide, blending: additive ? THREE.AdditiveBlending : THREE.NormalBlending,
      vertexShader: `attribute float aA; varying float vA; void main() { vA = aA; gl_Position = projectionMatrix * viewMatrix * vec4(position, 1.0); }`,
      fragmentShader: `uniform vec3 uCol; uniform float uOp; varying float vA; void main() { gl_FragColor = vec4(uCol, vA * uOp); }`,
    });
    this.mesh = new THREE.Mesh(g, this.mat); this.mesh.frustumCulled = false; scene.add(this.mesh);
  }
  reset(p) { for (const q of this.pts) q.copy(p); this.alpha.fill(0); }
  build(cam, taper = true) {
    const n = this.n, P = this.pts;
    for (let i = 0; i < n; i++) {
      const a = P[Math.max(0, i - 1)], b = P[Math.min(n - 1, i + 1)];
      T.subVectors(b, a); V.subVectors(cam.position, P[i]); W.crossVectors(T, V); const l = W.length() || 1;
      const w = (this.width * (taper ? 1 - (0.7 * i) / n : 1) * 0.5) / l;
      this.pos[i * 6] = P[i].x + W.x * w; this.pos[i * 6 + 1] = P[i].y + W.y * w; this.pos[i * 6 + 2] = P[i].z + W.z * w;
      this.pos[i * 6 + 3] = P[i].x - W.x * w; this.pos[i * 6 + 4] = P[i].y - W.y * w; this.pos[i * 6 + 5] = P[i].z - W.z * w;
      this.a[i * 2] = this.a[i * 2 + 1] = this.alpha[i];
    }
    this.mesh.geometry.attributes.position.needsUpdate = true; this.mesh.geometry.attributes.aA.needsUpdate = true;
  }
}

export function buildFx(scene) {
  // ---- scarf: a little rope that the relative wind drags ----
  const scarf = new Ribbon(scene, 12, 0xd3261c, 0.2, false, 1); scarf.alpha.fill(1);
  const sv = scarf.pts.map(() => new THREE.Vector3());
  // ---- wingtip trails ----
  const trails = [new Ribbon(scene, 40, 0xffffff, 0.07, true, 0.4), new Ribbon(scene, 40, 0xffffff, 0.07, true, 0.4)];
  let trailT = 0;
  // ---- speed streaks: motes hanging in the air that smear as he passes ----
  const NS = 110, sp = new Float32Array(NS * 6), sa = new Float32Array(NS * 2), seeds = [];
  const rng = mulberry32(3);
  for (let i = 0; i < NS; i++) seeds.push(new THREE.Vector3((rng() - 0.5) * 60, (rng() - 0.5) * 40, (rng() - 0.5) * 60));
  const sg = new THREE.BufferGeometry(); sg.setAttribute('position', new THREE.BufferAttribute(sp, 3)); sg.setAttribute('aA', new THREE.BufferAttribute(sa, 1));
  const streaks = new THREE.LineSegments(sg, new THREE.ShaderMaterial({ transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
    vertexShader: `attribute float aA; varying float vA; void main() { vA = aA; gl_Position = projectionMatrix * viewMatrix * vec4(position, 1.0); }`,
    fragmentShader: `varying float vA; void main() { gl_FragColor = vec4(1.0, 1.0, 1.0, vA); }` }));
  streaks.frustumCulled = false; scene.add(streaks);
  // ---- thermal motes ----
  const NM = 70 * thermals.length, mp = new Float32Array(NM * 3), ms = [];
  for (let i = 0; i < NM; i++) { const t = thermals[i % thermals.length]; ms.push({ t, r: Math.sqrt(rng()) * t.R * 0.9, a: rng() * 6.28, y: t.y0 + rng() * (CLOUDBASE - t.y0), sp: 0.6 + rng() * 0.6 }); }
  const mg = new THREE.BufferGeometry(); mg.setAttribute('position', new THREE.BufferAttribute(mp, 3));
  const motes = new THREE.Points(mg, new THREE.ShaderMaterial({ transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
    vertexShader: `varying float vA; void main() { vec4 mv = viewMatrix * vec4(position, 1.0); float d = -mv.z; vA = smoothstep(900.0, 150.0, d) * smoothstep(3.0, 20.0, d); gl_PointSize = clamp(260.0 / d, 1.0, 5.0); gl_Position = projectionMatrix * mv; }`,
    fragmentShader: `varying float vA; void main() { vec2 p = gl_PointCoord - 0.5; float a = smoothstep(0.5, 0.1, length(p)); gl_FragColor = vec4(1.0, 0.95, 0.8, a * vA * 0.55); }` }));
  motes.frustumCulled = false; scene.add(motes);
  // ---- shadow blob ----
  const shadow = new THREE.Mesh(new THREE.PlaneGeometry(1, 1).rotateX(-Math.PI / 2), new THREE.ShaderMaterial({ transparent: true, depthWrite: false, uniforms: { uA: { value: 0.5 } },
    vertexShader: `varying vec2 vUv; void main() { vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`,
    fragmentShader: `uniform float uA; varying vec2 vUv; void main() { vec2 p = (vUv - 0.5) * 2.0; float body = smoothstep(1.0, 0.2, length(p * vec2(1.0, 4.5))); float wing = smoothstep(1.0, 0.3, length(p * vec2(3.2, 1.0))); gl_FragColor = vec4(0.0, 0.0, 0.02, max(body * 0.6, wing) * uA); }` }));
  shadow.renderOrder = 6; scene.add(shadow);
  // ---- bursts ----
  const NB = 260, bp = new Float32Array(NB * 3), bc = new Float32Array(NB * 4), bv = Array.from({ length: NB }, () => ({ v: new THREE.Vector3(), life: 0, max: 1, g: 9.8 }));
  const bg = new THREE.BufferGeometry(); bg.setAttribute('position', new THREE.BufferAttribute(bp, 3)); bg.setAttribute('aC', new THREE.BufferAttribute(bc, 4));
  const bursts = new THREE.Points(bg, new THREE.ShaderMaterial({ transparent: true, depthWrite: false,
    vertexShader: `attribute vec4 aC; varying vec4 vC; void main() { vC = aC; vec4 mv = viewMatrix * vec4(position, 1.0); gl_PointSize = clamp(420.0 / -mv.z, 1.5, 16.0); gl_Position = projectionMatrix * mv; }`,
    fragmentShader: `varying vec4 vC; void main() { float a = smoothstep(0.5, 0.2, length(gl_PointCoord - 0.5)); gl_FragColor = vec4(vC.rgb, vC.a * a); }` }));
  bursts.frustumCulled = false; scene.add(bursts);
  let bi = 0;
  const wind = [0, 0, 0];

  return {
    reset(model) { model.root.updateMatrixWorld(true); const p = model.root.localToWorld(V.copy(model.neck)).clone(); scarf.reset(p); sv.forEach((v) => v.set(0, 0, 0)); trails.forEach((t) => t.reset(p)); },
    burst(p, n, color, speed, up, life = 1.4, grav = 9.8) {
      const c = new THREE.Color(color);
      for (let i = 0; i < n; i++) { const k = bi++ % NB, b = bv[k]; b.v.set((Math.random() - 0.5) * speed, Math.random() * up, (Math.random() - 0.5) * speed); b.life = b.max = life * (0.5 + Math.random() * 0.7); b.g = grav;
        bp[k * 3] = p[0] + (Math.random() - 0.5); bp[k * 3 + 1] = p[1] + Math.random() * 0.5; bp[k * 3 + 2] = p[2] + (Math.random() - 0.5); bc[k * 4] = c.r; bc[k * 4 + 1] = c.g; bc[k * 4 + 2] = c.b; }
    },
    update(dt, time, bird, model, cam) {
      model.root.updateMatrixWorld(true);
      windAt(bird.p[0], bird.p[1], bird.p[2], time, wind);
      // scarf
      const P = scarf.pts; model.root.children[0].localToWorld(P[0].copy(model.neck));
      for (let i = 1; i < P.length; i++) {
        const v = sv[i];
        v.x += (wind[0] - v.x) * Math.min(1, dt * 7); v.y += (wind[1] - 2.5 - v.y) * Math.min(1, dt * 7); v.z += (wind[2] - v.z) * Math.min(1, dt * 7);
        const fl = Math.sin(time * 17 + i * 1.3) * 0.5 * Math.min(1, bird.airspeed / 15);
        P[i].x += v.x * dt; P[i].y += (v.y + fl) * dt; P[i].z += v.z * dt;
        T.subVectors(P[i], P[i - 1]); const l = T.length() || 1e-6; P[i].copy(P[i - 1]).addScaledVector(T, 0.17 / l);
      }
      scarf.build(cam, false);
      // trails: brightest when the wings are loaded and fast
      const loadK = bird.mode === 'air' ? sstep(1.6, 3.2, Math.abs(bird.gload)) * sstep(14, 26, bird.airspeed) + sstep(50, 80, bird.airspeed) * bird.fold * 0.35 : 0;
      trailT += dt;
      const shift = trailT > 0.014; if (shift) trailT = 0;
      trails.forEach((tr, s) => {
        const tip = model.root.children[0].localToWorld(V.copy(model.tips[s]));
        if (shift) { for (let i = tr.n - 1; i > 0; i--) { tr.pts[i].copy(tr.pts[i - 1]); tr.alpha[i] = tr.alpha[i - 1] * 0.9; } }
        tr.pts[0].copy(tip); tr.alpha[0] = clamp(loadK, 0, 1);
        tr.build(cam);
      });
      // streaks
      const sk = bird.mode === 'air' ? sstep(24, 70, bird.airspeed) : 0;
      streaks.visible = sk > 0.01;
      if (streaks.visible) {
        const vx = bird.v[0] - wind[0], vy = bird.v[1] - wind[1], vz = bird.v[2] - wind[2], L = 0.035 + 0.02 * sk;
        for (let i = 0; i < NS; i++) {
          const s = seeds[i];
          // wrap each mote into a box that travels with the camera
          let x = s.x - cam.position.x, y = s.y - cam.position.y, z = s.z - cam.position.z;
          x = ((x % 60) + 90) % 60 - 30; y = ((y % 40) + 60) % 40 - 20; z = ((z % 60) + 90) % 60 - 30;
          x += cam.position.x; y += cam.position.y; z += cam.position.z;
          sp[i * 6] = x; sp[i * 6 + 1] = y; sp[i * 6 + 2] = z; sp[i * 6 + 3] = x - vx * L; sp[i * 6 + 4] = y - vy * L; sp[i * 6 + 5] = z - vz * L;
          sa[i * 2] = 0; sa[i * 2 + 1] = 0.5 * sk;
        }
        sg.attributes.position.needsUpdate = true; sg.attributes.aA.needsUpdate = true;
      }
      // motes: rise and swirl
      for (let i = 0; i < NM; i++) {
        const m = ms[i], t = m.t;
        m.y += t.w * m.sp * Math.exp(-(m.r * m.r) / (t.R * t.R)) * dt * 1.6 + 0.4 * dt; m.a += (dt * 3.2) / (20 + m.r);
        if (m.y > CLOUDBASE) { m.y = t.y0 + 2; m.r = Math.sqrt(Math.random()) * t.R * 0.9; }
        const [cx, cz] = thermalCentre(t, m.y);
        mp[i * 3] = cx + Math.cos(m.a) * m.r; mp[i * 3 + 1] = m.y; mp[i * 3 + 2] = cz + Math.sin(m.a) * m.r;
      }
      mg.attributes.position.needsUpdate = true;
      // shadow: where the sun would put it, fading with height
      const sdx = -0.74 / 0.44, sdz = 0.5 / 0.44;
      let sx = bird.p[0], sz = bird.p[2], gy = groundAt(sx, sz);
      for (let i = 0; i < 4; i++) { const h = Math.max(0, bird.p[1] - Math.max(gy, 0)); sx = bird.p[0] - sdx * h; sz = bird.p[2] - sdz * h; gy = groundAt(sx, sz); }
      const agl = bird.p[1] - Math.max(gy, 0), fold = 0.3 + 0.7 * bird.fold;
      shadow.visible = agl < 260 && gy > -0.5;
      if (shadow.visible) {
        const n = normalAt(sx, sz, 6);
        shadow.position.set(sx, gy + 0.12 + agl * 0.004, sz);
        const yaw = Math.atan2(-(bird.q ? 2 * (bird.q[1] * bird.q[3] - bird.q[0] * bird.q[2]) : 0), 1 - 2 * (bird.q[2] * bird.q[2] + bird.q[3] * bird.q[3]));
        shadow.quaternion.setFromUnitVectors(V.set(0, 1, 0), W.set(n[0], n[1], n[2])).multiply(new THREE.Quaternion().setFromAxisAngle(V.set(0, 1, 0), yaw + Math.PI / 2));
        const s = 1 + agl * 0.012; shadow.scale.set(9 * fold * s, 1, 2.6 * s);
        shadow.material.uniforms.uA.value = 0.55 * (1 - sstep(20, 260, agl)) * (bird.poseUp > 0.5 ? 0.6 : 1);
      }
      // bursts
      for (let k = 0; k < NB; k++) { const b = bv[k]; if (b.life <= 0) { bc[k * 4 + 3] = 0; continue; } b.life -= dt; b.v.y -= b.g * dt; b.v.multiplyScalar(1 - 0.8 * dt); bp[k * 3] += b.v.x * dt; bp[k * 3 + 1] += b.v.y * dt; bp[k * 3 + 2] += b.v.z * dt; bc[k * 4 + 3] = Math.max(0, b.life / b.max); }
      bg.attributes.position.needsUpdate = true; bg.attributes.aC.needsUpdate = true;
    },
  };
}
