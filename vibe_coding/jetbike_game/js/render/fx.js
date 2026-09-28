// Jet effects: faint exhaust plumes, heat-haze distortion volumes, and a particle system for
// dust blown off the ground and spray/mist lifted off the lake where the exhaust jets hit.
// Particle emission is physical in spirit: a free turbojet exhaust decays roughly as
// U(h) ~ Ve * 6D / h beyond its potential core, and the wall jet it forms on impact carries
// dust outward with that velocity scale; particles then relax to the local wind.

import * as THREE from 'three';
import { heightAt, surfaceAt, WATER_Y, PAD } from '../terrain.js';
import { rng } from '../vmath.js';

const MAX_P = 36000;

function puffTexture() {
  const N = 128, c = document.createElement('canvas');
  c.width = c.height = N * 2;
  const g = c.getContext('2d');
  const R = rng(11);
  for (let t = 0; t < 4; t++) {
    const ox = (t % 2) * N, oy = Math.floor(t / 2) * N;
    for (let k = 0; k < 26; k++) {
      const a = R() * 6.28, r = R() * N * 0.22;
      const x = ox + N / 2 + Math.cos(a) * r, y = oy + N / 2 + Math.sin(a) * r, rad = N * (0.12 + R() * 0.2);
      const gr = g.createRadialGradient(x, y, 0, x, y, rad);
      gr.addColorStop(0, 'rgba(255,255,255,0.23)'); gr.addColorStop(1, 'rgba(255,255,255,0)');
      g.fillStyle = gr; g.fillRect(ox, oy, N, N);
    }
  }
  const tex = new THREE.CanvasTexture(c);
  return tex;
}

export class Particles {
  constructor(scene, sunDir) {
    this.n = 0;
    this.pos = new Float32Array(MAX_P * 3);
    this.vel = new Float32Array(MAX_P * 3);
    this.age = new Float32Array(MAX_P);
    this.life = new Float32Array(MAX_P);
    this.size0 = new Float32Array(MAX_P);
    this.grow = new Float32Array(MAX_P);
    this.kind = new Uint8Array(MAX_P);   // 0 dust, 1 spray droplets, 2 mist, 3 grass debris
    this.seed = new Float32Array(MAX_P);
    const g = new THREE.BufferGeometry();
    this.aPos = new THREE.BufferAttribute(new Float32Array(MAX_P * 3), 3).setUsage(THREE.DynamicDrawUsage);
    this.aSize = new THREE.BufferAttribute(new Float32Array(MAX_P), 1).setUsage(THREE.DynamicDrawUsage);
    this.aAlpha = new THREE.BufferAttribute(new Float32Array(MAX_P), 1).setUsage(THREE.DynamicDrawUsage);
    this.aKind = new THREE.BufferAttribute(new Float32Array(MAX_P * 2), 2).setUsage(THREE.DynamicDrawUsage);
    g.setAttribute('position', this.aPos); g.setAttribute('size', this.aSize);
    g.setAttribute('alpha', this.aAlpha); g.setAttribute('kind', this.aKind);
    this.mat = new THREE.ShaderMaterial({
      uniforms: {
        tPuff: { value: puffTexture() }, uScale: { value: 800 }, uSun: { value: sunDir.clone() },
        fogColor: { value: new THREE.Color(0x9fb3c8) }, fogDensity: { value: 0.000085 },
      },
      vertexShader: /* glsl */`
        attribute float size; attribute float alpha; attribute vec2 kind;
        uniform float uScale;
        varying float vAlpha; varying float vKind; varying float vSeed; varying float vFog;
        void main(){
          vec4 mv = modelViewMatrix * vec4(position,1.0);
          gl_Position = projectionMatrix * mv;
          gl_PointSize = clamp(size * uScale / -mv.z, 0.0, 900.0);
          vAlpha = alpha; vKind = kind.x; vSeed = kind.y;
          vFog = -mv.z;
        }`,
      fragmentShader: /* glsl */`
        uniform sampler2D tPuff; uniform vec3 fogColor; uniform float fogDensity;
        varying float vAlpha; varying float vKind; varying float vSeed; varying float vFog;
        void main(){
          vec2 uv = gl_PointCoord;
          float r = length(uv-0.5);
          if (vKind > 0.5 && vKind < 1.5) {           // droplets: small bright dots
            float a = smoothstep(0.5,0.2,r) * vAlpha;
            gl_FragColor = vec4(vec3(1.6,1.65,1.7)*0.8, a); return;
          }
          if (vKind > 2.5) {                           // grass debris flecks
            float a = smoothstep(0.5,0.35,r) * vAlpha;
            gl_FragColor = vec4(vec3(0.12,0.16,0.05), a); return;
          }
          // rotate the puff by the particle seed, pick one of 4 atlas tiles
          float ang = vSeed * 6.2831;
          vec2 p = uv - 0.5; p = mat2(cos(ang),-sin(ang),sin(ang),cos(ang)) * p + 0.5;
          float tile = floor(vSeed*3.999);
          vec2 tuv = (p + vec2(mod(tile,2.0), floor(tile/2.0))) * 0.5;
          float a = texture2D(tPuff, tuv).a * smoothstep(0.5,0.3,r);
          // soft self-shadowing: lit from above-left
          float lit = 0.75 + 0.35*(0.5 - uv.y);
          vec3 base = vKind < 0.5 ? vec3(0.50,0.44,0.36) : vec3(1.0,1.02,1.05);
          vec3 col = base * lit * 1.1;
          float f = 1.0 - exp(-fogDensity*fogDensity*vFog*vFog);
          col = mix(col, fogColor, f);
          gl_FragColor = vec4(col, a * vAlpha);
        }`,
      transparent: true, depthWrite: false,
    });
    this.points = new THREE.Points(g, this.mat);
    this.points.frustumCulled = false;
    this.points.renderOrder = 5;
    scene.add(this.points);
    this.R = rng(77);
    this.hAt = heightAt;   // replaced by a fast grid lookup from the world
    this.acc = [0, 0, 0, 0, 0];
  }

  reset() { this.n = 0; this.acc.fill(0); }

  spawn(kind, x, y, z, vx, vy, vz, life, s0, grow) {
    if (this.n >= MAX_P) return;
    const i = this.n++;
    this.pos[i * 3] = x; this.pos[i * 3 + 1] = y; this.pos[i * 3 + 2] = z;
    this.vel[i * 3] = vx; this.vel[i * 3 + 1] = vy; this.vel[i * 3 + 2] = vz;
    this.age[i] = 0; this.life[i] = life; this.size0[i] = s0; this.grow[i] = grow;
    this.kind[i] = kind; this.seed[i] = this.R();
  }

  // exhausts: [{p (Vector3), d (unit dir of flow), T (N), r}], wind [x,y,z]
  step(dt, exhausts, wind) {
    const R = this.R;
    exhausts.forEach((e, k) => {
      if (e.T < 30) return;
      // where does the jet axis meet the ground / water?
      if (e.d.y > -0.12) return;
      let tHit = (e.p.y - surfaceAt(e.p.x, e.p.z)) / -e.d.y;
      for (let it = 0; it < 2; it++) {
        const hx = e.p.x + e.d.x * tHit, hz = e.p.z + e.d.z * tHit;
        tHit = (e.p.y - surfaceAt(hx, hz)) / -e.d.y;
      }
      if (tHit > 22 || tHit < 0) return;
      const hx = e.p.x + e.d.x * tHit, hz = e.p.z + e.d.z * tHit;
      const ground = heightAt(hx, hz);
      const onWater = ground < WATER_Y;
      const hy = onWater ? WATER_Y : ground;
      // centreline velocity at impact
      const D = 2 * e.r;
      const Ve = 560 * Math.sqrt(Math.min(1, e.T / 1250));   // T ~ mdot*V with mdot ~ V  =>  V ~ sqrt(T)
      const U = Ve * Math.min(1, 6 * D / Math.max(tHit, 0.1));
      const strength = Math.max(0, (U - 8) / 60);
      if (strength <= 0) return;
      const dPad = Math.hypot(hx - PAD[0], hz - PAD[2]);
      const surf = onWater ? 1 : dPad < 9 ? 0.25 : dPad < 17 ? 1.0 : 0.1; // concrete / gravel / meadow (grass holds the soil)
      const rate = (onWater ? 900 : 700) * strength * surf;
      this.acc[k] += rate * dt;
      while (this.acc[k] >= 1) {
        this.acc[k] -= 1;
        const a = R() * Math.PI * 2;
        const cx = Math.cos(a), cz = Math.sin(a);
        const r0 = 0.2 + R() * 0.6;
        // wall jet speed, biased by the jet's own horizontal slant
        const vw = U * (0.25 + 0.35 * R());
        let vx = cx * vw + e.d.x * U * 0.3, vz = cz * vw + e.d.z * U * 0.3;
        if (onWater) {
          if (R() < 0.55) {
            this.spawn(1, hx + cx * r0, hy + 0.05, hz + cz * r0, vx * 0.35, 3 + R() * 9 * Math.min(1.5, strength), vz * 0.35, 0.8 + R() * 1.4, 0.04 + R() * 0.06, 0);
          } else {
            this.spawn(2, hx + cx * r0, hy + 0.3 + R() * 1.5, hz + cz * r0, vx * 0.5, 0.8 + R() * 2.5, vz * 0.5, 1.5 + R() * 2.0, 0.4 + R() * 0.8, 1.0 + 1.5 * R());
          }
        } else if (dPad > 17 && R() < 0.6) {
          this.spawn(3, hx + cx * r0 * 3, hy + 0.1, hz + cz * r0 * 3, vx * 0.6, 1 + R() * 3, vz * 0.6, 1.5 + R(), 0.03 + R() * 0.03, 0);
        } else {
          this.spawn(0, hx + cx * r0, hy + 0.15, hz + cz * r0, vx, 0.4 + R() * 1.6, vz, 3 + R() * 4, 0.35 + R() * 0.5, 0.8 + R() * 0.9);
        }
      }
    });

    // integrate
    const pos = this.pos, vel = this.vel;
    let w = 0;
    for (let i = 0; i < this.n; i++) {
      const age = this.age[i] + dt;
      if (age >= this.life[i]) continue;
      const kind = this.kind[i];
      let x = pos[i * 3], y = pos[i * 3 + 1], z = pos[i * 3 + 2];
      let vx = vel[i * 3], vy = vel[i * 3 + 1], vz = vel[i * 3 + 2];
      const tau = kind === 1 ? 1.2 : kind === 3 ? 0.6 : 0.35;     // drag relaxation time
      const k = Math.min(1, dt / tau);
      vx += (wind[0] - vx) * k; vz += (wind[2] - vz) * k;
      if (kind === 1 || kind === 3) vy -= 9.81 * dt; else vy += (0.6 - vy) * k * 0.5; // droplets fall, dust drifts up
      x += vx * dt; y += vy * dt; z += vz * dt;
      const gy = kind === 1 ? WATER_Y : Math.max(WATER_Y, this.hAt(x, z)) + 0.05;
      if (y < gy) { if (kind === 1) continue; y = gy; vy = Math.abs(vy) * 0.3; }
      // compact alive particles in place
      pos[w * 3] = x; pos[w * 3 + 1] = y; pos[w * 3 + 2] = z;
      vel[w * 3] = vx; vel[w * 3 + 1] = vy; vel[w * 3 + 2] = vz;
      this.age[w] = age; this.life[w] = this.life[i]; this.size0[w] = this.size0[i]; this.grow[w] = this.grow[i];
      this.kind[w] = kind; this.seed[w] = this.seed[i];
      w++;
    }
    this.n = w;
  }

  upload() {
    const P = this.aPos.array, S = this.aSize.array, A = this.aAlpha.array, K = this.aKind.array;
    for (let i = 0; i < this.n; i++) {
      P[i * 3] = this.pos[i * 3]; P[i * 3 + 1] = this.pos[i * 3 + 1]; P[i * 3 + 2] = this.pos[i * 3 + 2];
      const u = this.age[i] / this.life[i];
      S[i] = this.size0[i] + this.grow[i] * this.age[i];
      const kind = this.kind[i];
      const fadeIn = Math.min(1, this.age[i] / 0.15);
      A[i] = kind === 0 ? 0.42 * fadeIn * (1 - u) * (1 - u) : kind === 2 ? 0.13 * fadeIn * (1 - u) * (1 - u) : kind === 1 ? 0.5 * (1 - u) : 0.9 * (1 - u);
      K[i * 2] = kind; K[i * 2 + 1] = this.seed[i];
    }
    this.points.geometry.setDrawRange(0, this.n);
    this.aPos.needsUpdate = this.aSize.needsUpdate = this.aAlpha.needsUpdate = this.aKind.needsUpdate = true;
  }
}

// Visible exhaust: a turbojet's plume is almost transparent in daylight. We draw a faint
// hot core near the nozzle (additive) plus a long heat-haze volume in a separate scene.
export class Plumes {
  constructor(scene, hazeScene, n = 5) {
    this.items = [];
    const geo = new THREE.CylinderGeometry(0.3, 1, 1, 24, 12, true).translate(0, -0.5, 0); // cone hanging along -y
    for (let i = 0; i < n; i++) {
      const core = new THREE.Mesh(geo, new THREE.ShaderMaterial({
        uniforms: { uTime: { value: 0 }, uHeat: { value: 0 }, uLen: { value: 1 } },
        vertexShader: /* glsl */`
          varying float vAx; varying float vRim; varying vec3 vW;
          void main(){
            vAx = -position.y;
            vec4 w = modelMatrix*vec4(position,1.0); vW = w.xyz;
            vec3 n = normalize(mat3(modelMatrix)*normal);
            vec3 v = normalize(cameraPosition - w.xyz);
            vRim = abs(dot(n, v));
            gl_Position = projectionMatrix*viewMatrix*w;
          }`,
        fragmentShader: /* glsl */`
          uniform float uTime, uHeat;
          varying float vAx; varying float vRim; varying vec3 vW;
          float h(vec3 p){ return fract(sin(dot(p, vec3(12.9898,78.233,37.719)))*43758.5453); }
          void main(){
            float core = pow(vRim, 2.5);
            float ax = exp(-vAx*3.2);
            float flick = 0.8 + 0.2*h(floor(vW*18.0) + floor(uTime*60.0));
            vec3 c = mix(vec3(0.35,0.55,1.2), vec3(1.4,0.55,0.18), smoothstep(0.1,0.6,vAx)) * uHeat * core * ax * flick;
            gl_FragColor = vec4(c, 1.0);
          }`,
        blending: THREE.AdditiveBlending, transparent: true, depthWrite: false, side: THREE.DoubleSide,
      }));
      core.frustumCulled = false;
      scene.add(core);
      const haze = new THREE.Mesh(geo, new THREE.ShaderMaterial({
        uniforms: {
          uTime: { value: 0 }, uStr: { value: 0 }, tDepth: { value: null }, uRes: { value: new THREE.Vector2(1, 1) },
          uNear: { value: 0.1 }, uFar: { value: 1000 }, uSpeed: { value: 10 },
        },
        vertexShader: /* glsl */`
          varying float vAx; varying float vRim; varying vec3 vL;
          void main(){
            vAx = -position.y; vL = position;
            vec4 w = modelMatrix*vec4(position,1.0);
            vec3 n = normalize(mat3(modelMatrix)*normal);
            vRim = abs(dot(n, normalize(cameraPosition - w.xyz)));
            gl_Position = projectionMatrix*viewMatrix*w;
          }`,
        fragmentShader: /* glsl */`
          uniform float uTime, uStr, uNear, uFar, uSpeed; uniform sampler2D tDepth; uniform vec2 uRes;
          varying float vAx; varying float vRim; varying vec3 vL;
          float lin(float d){ return uNear*uFar/(uFar - d*(uFar-uNear)); }
          float h21(vec2 p){ p=fract(p*vec2(123.34,456.21)); p+=dot(p,p+45.32); return fract(p.x*p.y); }
          float vn(vec2 p){ vec2 i=floor(p), f=fract(p); vec2 u=f*f*(3.0-2.0*f);
            return mix(mix(h21(i),h21(i+vec2(1,0)),u.x), mix(h21(i+vec2(0,1)),h21(i+vec2(1,1)),u.x), u.y); }
          void main(){
            vec2 suv = gl_FragCoord.xy / uRes;
            if (lin(texture2D(tDepth, suv).x) < lin(gl_FragCoord.z) - 0.02) discard;
            float ang = atan(vL.x, vL.z);
            vec2 q = vec2(ang*2.0, vAx*5.0 - uTime*uSpeed);
            vec2 n = vec2(vn(q), vn(q + 17.3)) - 0.5;
            float fall = exp(-vAx*0.55) * smoothstep(0.0, 0.25, vAx);
            float s = uStr * fall * pow(vRim, 1.5);
            gl_FragColor = vec4(n * s * 0.018, 0.0, 1.0);
          }`,
        blending: THREE.AdditiveBlending, transparent: true, depthWrite: false, depthTest: false, side: THREE.DoubleSide,
      }));
      haze.frustumCulled = false;
      hazeScene.add(haze);
      this.items.push({ core, haze });
    }
  }
  update(exhausts, time, post, camera) {
    const up = new THREE.Vector3(0, -1, 0);
    exhausts.forEach((e, i) => {
      const it = this.items[i];
      const frac = Math.min(1, e.T / 1250);
      const heat = Math.max(0, (e.egt - 700) / 400);
      const q = new THREE.Quaternion().setFromUnitVectors(up, e.d);
      // hot core: short, narrow
      const coreLen = 0.25 + 0.9 * frac;
      it.core.position.copy(e.p); it.core.quaternion.copy(q);
      it.core.scale.set(e.r * 0.9, coreLen, e.r * 0.9);
      it.core.material.uniforms.uHeat.value = heat * 0.55;
      it.core.material.uniforms.uTime.value = time;
      it.core.visible = heat > 0.02;
      // haze volume: spreads ~8 deg half-angle
      const L = 1.5 + 7 * Math.sqrt(frac);
      it.haze.position.copy(e.p); it.haze.quaternion.copy(q);
      it.haze.scale.set(e.r + L * 0.14, L, e.r + L * 0.14);
      const m = it.haze.material.uniforms;
      m.uStr.value = heat * (0.5 + frac);
      m.uTime.value = time; m.uSpeed.value = 6 + 18 * frac;
      m.tDepth.value = post.depthTexture; m.uRes.value.set(post.rtHaze.width, post.rtHaze.height);  // haze target is half size
      m.uNear.value = camera.near; m.uFar.value = camera.far;
      it.haze.visible = heat > 0.02;
    });
  }
}
