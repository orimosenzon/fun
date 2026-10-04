// Crash explosion FX: expanding additive fireball shells, ballistic debris
// chunks that bounce off the terrain, a rising smoke column, hot sparks, a
// decaying orange flash light and a camera-shake signal. A 'water' variant
// swaps the fire for a pale spray. Same visual language as jetfx.js.
import * as THREE from 'three';
import { terrainHeight, WATER_Y } from './world.js';

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

const SMOKE_N = 130, SPARK_N = 170;

export function makeExplosion(scene, camera, renderer) {
  const tex = discTexture(1, 0.45);
  const softTex = discTexture(0.75, 0.6);

  // --- fireball shells (billboard sprites, additive) ---
  const SHELLS = [
    { color: 0xfff6e0, size: 5, grow: 30, dur: 0.5 },
    { color: 0xff9a30, size: 3.5, grow: 40, dur: 0.75 },
    { color: 0xc04a12, size: 2.5, grow: 46, dur: 1.0 },
  ];
  const shells = SHELLS.map(cfg => {
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

  // --- debris chunks ---
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

  // --- particle pools: hot sparks (additive) + smoke/spray (normal) ---
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
  const sparks = makePool(SPARK_N, { additive: true, isSmoke: false });

  const api = {
    active: false,
    shake: 0,
    time: 0,
    groundY: 0,
    trigger(pos, groundY, type) {
      api.active = true;
      api.time = 0;
      api.shake = type === 'water' ? 0.6 : 1.4;
      api.groundY = groundY;
      const water = type === 'water';

      for (const { s, cfg } of shells) {
        s.visible = !water;
        s.position.copy(pos);
        s.material.opacity = 0;
      }
      light.color.setHex(water ? 0x9ecbe8 : 0xffa040);
      light.intensity = water ? 60 : 420;
      light.position.set(pos.x, pos.y + 2, pos.z);

      debris.visible = !water;
      if (!water) {
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
      }

      // smoke column (dark) or water spray (pale)
      smoke.mat.uniforms.uColA.value.setHex(water ? 0xcfe4ee : 0x232326);
      smoke.mat.uniforms.uColB.value.setHex(water ? 0xf2fbff : 0x6e6a66);
      for (let i = 0; i < smoke.n; i++) {
        const k = i * 3;
        const ang = Math.random() * Math.PI * 2, rr = Math.random() * 1.6;
        smoke.pos[k] = pos.x + Math.cos(ang) * rr;
        smoke.pos[k + 1] = pos.y + Math.random() * 1.2;
        smoke.pos[k + 2] = pos.z + Math.sin(ang) * rr;
        smoke.vel[k] = Math.cos(ang) * (0.7 + Math.random() * 2.2);
        smoke.vel[k + 1] = (water ? 7 : 3.2) + Math.random() * (water ? 6 : 3.5);
        smoke.vel[k + 2] = Math.sin(ang) * (0.7 + Math.random() * 2.2);
        smoke.dur[i] = (water ? 1.1 : 2.4) + Math.random() * (water ? 0.8 : 1.8);
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
        sparks.life[i] = water ? 0 : 1;   // no fire sparks in water
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
      light.intensity *= Math.pow(0.02, dt);   // fast flash decay

      if (debris.visible) {
        let alive = false;
        for (let i = 0; i < ND; i++) {
          const k = i * 3;
          dV[k + 1] -= 22 * dt;
          dP[k] += dV[k] * dt; dP[k + 1] += dV[k + 1] * dt; dP[k + 2] += dV[k + 2] * dt;
          dR[k] += dW[k] * dt; dR[k + 1] += dW[k + 1] * dt; dR[k + 2] += dW[k + 2] * dt;
          const gy = Math.max(terrainHeight(dP[k], dP[k + 2]), WATER_Y) + 0.12;
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
