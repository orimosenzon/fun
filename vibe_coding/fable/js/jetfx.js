// Realistic jet exhaust FX, modeled on real afterburner plumes:
//   - Two additive shader shells per jet: a white-blue CORE with stationary
//     mach diamonds (shock cells), inside a wider sheath that cools through
//     blue into orange/red downstream. Alpha follows the view angle so the
//     shells read as a glowing gas volume, and the vertex shader ripples the
//     tail with traveling turbulence.
//   - Ember sparks: a pooled point cloud ejected along the jet, dragged by
//     air, bouncing off the ground, cooling white-yellow -> orange -> dark red.
//   - Ground dust kicked up by the lift jets in ground effect.
//   - Two dynamic lights (orange rear, blue underside) tied to thrust.
import * as THREE from 'three';

const PLUME_VERT = `
uniform float uTime;
uniform float uSeed;
uniform float uFlick;
varying float vT;      // 0 at nozzle -> 1 at tip
varying vec3 vNormal;
varying vec3 vView;
void main() {
  vT = 1.0 - uv.y;
  // radius profile: slight expansion off the nozzle, closing at the tip
  float prof = (0.92 + 0.3 * vT) * pow(1.0 - vT, 0.55);
  // traveling turbulence, growing downstream
  float rip = 1.0
    + uFlick * vT * 0.16 * sin(vT * 21.0 - uTime * 26.0 + uSeed)
    + uFlick * vT * 0.09 * sin(vT * 47.0 - uTime * 41.0 + uSeed * 2.7 + uv.x * 6.283);
  vec3 p = position;
  p.xz *= prof * rip;
  // small sideways waggle of the tail
  p.x += uFlick * vT * vT * 0.08 * sin(uTime * 19.0 + uSeed);
  vec4 mv = modelViewMatrix * vec4(p, 1.0);
  vNormal = normalMatrix * normal;
  vView = -mv.xyz;
  gl_Position = projectionMatrix * mv;
}`;

const PLUME_FRAG = `
uniform vec3 uColA;    // hottest, at the nozzle
uniform vec3 uColB;    // mid plume
uniform vec3 uColC;    // cool tail
uniform float uTime;
uniform float uSeed;
uniform float uDiamonds;  // number of shock cells (0 = none)
uniform float uGain;
varying float vT;
varying vec3 vNormal;
varying vec3 vView;
void main() {
  // longest optical path through the gas column is at the silhouette center,
  // where the surface faces the camera -> volumetric look from a thin shell
  float body = abs(dot(normalize(vNormal), normalize(vView)));
  float t = clamp(vT, 0.0, 1.0);
  // hot region is short: white -> mid within the first third, cool tail after
  vec3 col = mix(uColA, uColB, smoothstep(0.02, 0.3, t));
  col = mix(col, uColC, smoothstep(0.24, 0.85, t));
  float a = pow(1.0 - t, 1.15) * pow(body, 1.4);
  // stationary shock diamonds, fading downstream as the cells weaken
  if (uDiamonds > 0.5) {
    float d = pow(max(sin(t * uDiamonds * 6.2832 + 1.2), 0.0), 10.0);
    col += vec3(0.9, 0.95, 1.0) * d * (1.0 - t) * 1.5;
    a += d * (1.0 - t) * 0.8;
  }
  // slow overall breathing of brightness
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

// soft radial dot, used by the glow sprites, sparks and dust
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

// unit plume shell: nozzle ring at y=0, tip at y=-1, so scale.y = length
function plumeGeo() {
  const g = new THREE.CylinderGeometry(1, 1, 1, 18, 30, true);
  g.translate(0, -0.5, 0);
  return g;
}

const SPARK_N = 460;
const DUST_N = 260;

export function makeJetFX(scene, bikeGroup, camera, renderer) {
  const geo = plumeGeo();
  const glowTex = discTexture(1, 0.4);
  const softTex = discTexture(0.8, 0.55);

  // --- flame shells + nozzle glow, per jet ---
  // rear: full afterburner — white core, blue mid, orange tail, shock diamonds
  // lift jets: short high-pressure blue torches
  function makeJet(cfg) {
    const grp = new THREE.Group();
    grp.position.copy(cfg.pos);
    if (cfg.axis === 'z') grp.rotation.x = Math.PI / 2; // local -Y -> -Z
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
      pos: new THREE.Vector3(0, 0, -1.72), axis: 'z',
      dir: new THREE.Vector3(0, 0, -1),
      r: 0.21, len: 2.8, diamonds: 5,
      colors: [0xf8fbff, 0x9cc8ff, 0xff9a3c, 0xb43a10],
      sparks: 130,
    }),
    left: makeJet({
      pos: new THREE.Vector3(-0.75, -0.42, 0), axis: 'y',
      dir: new THREE.Vector3(0, -1, 0),
      r: 0.13, len: 1.55, diamonds: 3,
      colors: [0xf2f8ff, 0x9cc8ff, 0x5f8fe8, 0x27458f],
      sparks: 36,
    }),
    right: makeJet({
      pos: new THREE.Vector3(0.75, -0.42, 0), axis: 'y',
      dir: new THREE.Vector3(0, -1, 0),
      r: 0.13, len: 1.55, diamonds: 3,
      colors: [0xf2f8ff, 0x9cc8ff, 0x5f8fe8, 0x27458f],
      sparks: 36,
    }),
    center: makeJet({
      pos: new THREE.Vector3(0, -0.52, 0), axis: 'y',
      dir: new THREE.Vector3(0, -1, 0),
      r: 0.16, len: 1.85, diamonds: 3,
      colors: [0xf2f8ff, 0x9cc8ff, 0x5f8fe8, 0x27458f],
      sparks: 48,
    }),
  };

  // --- dynamic lights ---
  const rearLight = new THREE.PointLight(0xff9a40, 0, 55, 2);
  rearLight.position.set(0, 0.1, -2.4);
  const liftLight = new THREE.PointLight(0x86b8ff, 0, 30, 2);
  liftLight.position.set(0, -1.3, 0);
  bikeGroup.add(rearLight, liftLight);

  // --- pooled particles: sparks (additive embers) and dust (soft billows) ---
  function makePool(n, { blending, texture, isDust }) {
    const pos = new Float32Array(n * 3);
    const lifeAttr = new Float32Array(n); // remaining fraction 0..1, 0 = dead
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
            ? 'float s = 0.7 + 2.6 * (1.0 - aLife) + aSeed * 0.6;'   // dust grows
            : 'float s = 0.055 + 0.075 * aLife + aSeed * 0.02;'}     // sparks shrink
          gl_PointSize = uPx * s / max(1.0, -mv.z);
          gl_Position = projectionMatrix * mv;
          if (aLife <= 0.0) gl_Position = vec4(0.0, 0.0, -10.0, 1.0); // cull dead
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
    return {
      points, mat,
      pos, life: lifeAttr,
      vel: new Float32Array(n * 3),
      dur: new Float32Array(n),
      head: 0, n,
    };
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

    // --- plumes, glow, lights ---
    for (const key of ['rear', 'left', 'right', 'center']) {
      const j = jets[key], v = o.jets[key];
      // smooth the visual response a touch (turbine-ish lag)
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

    // --- spark emission ---
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

    // --- dust emission in ground effect ---
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

    // --- particle integration ---
    sparks.mat.uniforms.uPx.value = px;
    dust.mat.uniforms.uPx.value = px;
    for (let i = 0; i < sparks.n; i++) {
      if (sparks.life[i] <= 0) continue;
      sparks.life[i] -= dt / sparks.dur[i];
      const k = i * 3;
      sparks.vel[k] *= 1 - 2.2 * dt;
      sparks.vel[k + 1] -= 14 * dt;          // embers sink fast (drag-limited)
      sparks.vel[k + 2] *= 1 - 2.2 * dt;
      sparks.pos[k] += sparks.vel[k] * dt;
      sparks.pos[k + 1] += sparks.vel[k + 1] * dt;
      sparks.pos[k + 2] += sparks.vel[k + 2] * dt;
      // bounce off the ground near the bike (groundY is local but sparks
      // live and die within a few meters of it)
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
