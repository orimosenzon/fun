// World generation: terrain heightfield (shared with physics), water, landing
// pad, forests, villages, ring course, clouds, sky and lighting.
import * as THREE from 'three';
import { makeNoise2D, mulberry32 } from './noise.js';

export const WATER_Y = 2.0;
export const WORLD_HALF = 2100;

const nHills = makeNoise2D(1337);
const nMask = makeNoise2D(777);
const nRidge = makeNoise2D(4242);
const nMisc = makeNoise2D(91);

const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
function smoothstep(a, b, x) {
  const t = clamp((x - a) / (b - a), 0, 1);
  return t * t * (3 - 2 * t);
}
function fbm(noise, x, y, oct) {
  let a = 1, f = 1, s = 0, norm = 0;
  for (let o = 0; o < oct; o++) { s += a * noise(x * f, y * f); norm += a; a *= 0.5; f *= 2.03; }
  return s / norm;
}

// Raw terrain: rolling hills + ridge-noise mountain ranges + a rim that keeps
// the player inside the world.
function rawHeight(x, z) {
  let h = fbm(nHills, x * 0.0016, z * 0.0016, 4) * 40 + 12;
  const m = smoothstep(0.04, 0.5, fbm(nMask, x * 0.00042, z * 0.00042, 3));
  const ridge = 1 - Math.abs(fbm(nRidge, x * 0.0011, z * 0.0011, 4));
  h += m * Math.pow(ridge, 1.6) * 240;
  const r = Math.hypot(x, z);
  h += smoothstep(1500, 2050, r) * 380;
  return h;
}

// Pick a spawn point in an open valley with a clear takeoff corridor toward
// +Z (where the ring course starts): no rising wall ahead, flat-ish around.
function findSpawn() {
  let best = null, bestScore = Infinity;
  for (let z = -1200; z <= 1200; z += 60) {
    for (let x = -1200; x <= 1200; x += 60) {
      const h0 = rawHeight(x, z);
      if (h0 < WATER_Y + 4 || h0 > 60) continue;
      let score = 0, ok = true;
      for (let d = 50; d <= 500 && ok; d += 50) {
        const h = rawHeight(x, z + d);
        if (h > h0 + 10 + d * 0.06) ok = false; // wall ahead
        else score += Math.max(0, h - h0);
      }
      if (!ok) continue;
      for (const [dx, dz] of [[150, 0], [-150, 0], [0, -150], [106, 106], [-106, 106]]) {
        const h = rawHeight(x + dx, z + dz);
        if (Math.abs(h - h0) > 25) { ok = false; break; }
        score += Math.abs(h - h0);
      }
      if (!ok) continue;
      if (score < bestScore) { bestScore = score; best = [x, z]; }
    }
  }
  return best || [0, 0];
}

export const SPAWN = new THREE.Vector3(0, 0, 0);
{
  const [sx, sz] = findSpawn();
  SPAWN.x = sx;
  SPAWN.z = sz;
}
const PAD_H = Math.max(rawHeight(SPAWN.x, SPAWN.z), WATER_Y + 4);
SPAWN.y = PAD_H;

// Physics + rendering both use this. Terrain is flattened around the pad.
export function terrainHeight(x, z) {
  const d = Math.hypot(x - SPAWN.x, z - SPAWN.z);
  if (d < 28) return PAD_H;
  const f = smoothstep(28, 85, d);
  return PAD_H * (1 - f) + rawHeight(x, z) * f;
}

export function terrainNormal(x, z, target) {
  const e = 2;
  const hx = terrainHeight(x + e, z) - terrainHeight(x - e, z);
  const hz = terrainHeight(x, z + e) - terrainHeight(x, z - e);
  return target.set(-hx / (2 * e), 1, -hz / (2 * e)).normalize();
}

function buildTerrain(scene) {
  const size = WORLD_HALF * 2, seg = 240;
  const geo = new THREE.PlaneGeometry(size, size, seg, seg);
  geo.rotateX(-Math.PI / 2);
  const pos = geo.attributes.position;
  for (let i = 0; i < pos.count; i++) {
    pos.setY(i, terrainHeight(pos.getX(i), pos.getZ(i)));
  }
  geo.computeVertexNormals();

  const colors = new Float32Array(pos.count * 3);
  const nrm = geo.attributes.normal;
  const grassA = new THREE.Color(0x5e8c4a), grassB = new THREE.Color(0x7aa85e);
  const rock = new THREE.Color(0x8a8074), snow = new THREE.Color(0xeef1f6);
  const sand = new THREE.Color(0xcbb98a);
  const c = new THREE.Color();
  for (let i = 0; i < pos.count; i++) {
    const x = pos.getX(i), z = pos.getZ(i), h = pos.getY(i);
    const ny = nrm.getY(i);
    c.copy(grassA).lerp(grassB, 0.5 + 0.5 * nMisc(x * 0.01, z * 0.01));
    c.lerp(rock, smoothstep(0.88, 0.62, ny) * 0.9 + smoothstep(120, 200, h) * 0.4);
    c.lerp(snow, smoothstep(190, 250, h + nMisc(x * 0.02, z * 0.02) * 18));
    if (h < WATER_Y + 2.2) c.lerp(sand, smoothstep(WATER_Y + 2.2, WATER_Y + 0.3, h));
    colors[i * 3] = c.r; colors[i * 3 + 1] = c.g; colors[i * 3 + 2] = c.b;
  }
  geo.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  const mesh = new THREE.Mesh(geo, new THREE.MeshLambertMaterial({ vertexColors: true }));
  scene.add(mesh);
}

function buildWater(scene) {
  const mesh = new THREE.Mesh(
    new THREE.CircleGeometry(3200, 48),
    new THREE.MeshPhongMaterial({ color: 0x2e6f9e, shininess: 90, transparent: true, opacity: 0.88 })
  );
  mesh.rotation.x = -Math.PI / 2;
  mesh.position.y = WATER_Y;
  scene.add(mesh);
}

function buildPad(scene) {
  const pad = new THREE.Mesh(
    new THREE.CylinderGeometry(14, 15, 0.4, 32),
    new THREE.MeshStandardMaterial({ color: 0x3a4048, roughness: 0.9 })
  );
  pad.position.set(SPAWN.x, PAD_H - 0.06, SPAWN.z);
  scene.add(pad);
  const ring = new THREE.Mesh(
    new THREE.TorusGeometry(10, 0.25, 8, 48),
    new THREE.MeshBasicMaterial({ color: 0xf5c542 })
  );
  ring.rotation.x = -Math.PI / 2;
  ring.position.set(SPAWN.x, PAD_H + 0.16, SPAWN.z);
  scene.add(ring);
}

function buildTrees(scene) {
  const rand = mulberry32(2026);
  const pts = [];
  const n = new THREE.Vector3();
  for (let t = 0; t < 12000 && pts.length < 900; t++) {
    const x = (rand() * 2 - 1) * 1900, z = (rand() * 2 - 1) * 1900;
    const h = terrainHeight(x, z);
    if (h < WATER_Y + 2 || h > 150) continue;
    if (Math.hypot(x - SPAWN.x, z - SPAWN.z) < 45) continue;
    terrainNormal(x, z, n);
    if (n.y < 0.86) continue;
    pts.push([x, h, z, 0.7 + rand() * 0.9, rand()]);
  }
  const trunkGeo = new THREE.CylinderGeometry(0.22, 0.4, 3);
  trunkGeo.translate(0, 1.5, 0);
  const canopyGeo = new THREE.ConeGeometry(2.1, 6, 7);
  canopyGeo.translate(0, 5.5, 0);
  const trunks = new THREE.InstancedMesh(trunkGeo, new THREE.MeshLambertMaterial({ color: 0x6b4a2f }), pts.length);
  const canopies = new THREE.InstancedMesh(canopyGeo, new THREE.MeshLambertMaterial({ color: 0xffffff }), pts.length);
  const dummy = new THREE.Object3D();
  const c = new THREE.Color();
  for (let i = 0; i < pts.length; i++) {
    const [x, h, z, s, r] = pts[i];
    dummy.position.set(x, h - 0.2, z);
    dummy.scale.setScalar(s);
    dummy.rotation.y = r * Math.PI * 2;
    dummy.updateMatrix();
    trunks.setMatrixAt(i, dummy.matrix);
    canopies.setMatrixAt(i, dummy.matrix);
    c.setHSL(0.28 + r * 0.07, 0.5, 0.28 + r * 0.12);
    canopies.setColorAt(i, c);
  }
  scene.add(trunks, canopies);
}

function buildVillages(scene) {
  const rand = mulberry32(555);
  const n = new THREE.Vector3();
  const houses = [];
  let villages = 0;
  for (let t = 0; t < 4000 && villages < 6; t++) {
    const cx = (rand() * 2 - 1) * 1500, cz = (rand() * 2 - 1) * 1500;
    const h = terrainHeight(cx, cz);
    if (h < WATER_Y + 2.5 || h > 90) continue;
    terrainNormal(cx, cz, n);
    if (n.y < 0.95) continue;
    if (Math.hypot(cx - SPAWN.x, cz - SPAWN.z) < 120) continue;
    villages++;
    const count = 8 + (rand() * 10) | 0;
    for (let k = 0; k < count; k++) {
      const x = cx + (rand() * 2 - 1) * 70, z = cz + (rand() * 2 - 1) * 70;
      const hh = terrainHeight(x, z);
      terrainNormal(x, z, n);
      if (n.y < 0.92 || hh < WATER_Y + 1.5) continue;
      houses.push([x, hh, z, rand() * Math.PI * 2, rand()]);
    }
  }
  const wallGeo = new THREE.BoxGeometry(5.5, 3.2, 4.5);
  wallGeo.translate(0, 1.4, 0);
  const roofGeo = new THREE.ConeGeometry(3.9, 2.1, 4);
  roofGeo.rotateY(Math.PI / 4);
  roofGeo.translate(0, 4.0, 0);
  const walls = new THREE.InstancedMesh(wallGeo, new THREE.MeshLambertMaterial({ color: 0xffffff }), houses.length);
  const roofs = new THREE.InstancedMesh(roofGeo, new THREE.MeshLambertMaterial({ color: 0xa8473a }), houses.length);
  const dummy = new THREE.Object3D();
  const palette = [0xf2ede2, 0xe8dcc8, 0xd9c7a8, 0xd3dade].map(h => new THREE.Color(h));
  for (let i = 0; i < houses.length; i++) {
    const [x, h, z, ry, r] = houses[i];
    dummy.position.set(x, h - 0.3, z);
    dummy.rotation.y = ry;
    dummy.scale.setScalar(0.8 + r * 0.5);
    dummy.updateMatrix();
    walls.setMatrixAt(i, dummy.matrix);
    roofs.setMatrixAt(i, dummy.matrix);
    walls.setColorAt(i, palette[(r * palette.length) | 0]);
  }
  scene.add(walls, roofs);
}

function buildRings(scene) {
  const rings = [];
  let p = new THREE.Vector3(SPAWN.x, 0, SPAWN.z + 60);
  let heading = 0;
  const dir = new THREE.Vector3();
  for (let i = 0; i < 16; i++) {
    heading += fbm(nMisc, i * 0.37 + 5.1, 8.8, 2) * 1.1;
    if (Math.hypot(p.x, p.z) > 1450) {
      const back = Math.atan2(-p.x, -p.z);
      heading = heading * 0.35 + back * 0.65;
    }
    dir.set(Math.sin(heading), 0, Math.cos(heading));
    p = p.clone().addScaledVector(dir, 150 + (i % 3) * 30);
    const h = Math.max(terrainHeight(p.x, p.z), WATER_Y);
    const y = h + 22 + (fbm(nMisc, i * 0.51, 2.2, 2) + 1) * 22;
    const mesh = new THREE.Mesh(
      new THREE.TorusGeometry(7, 0.6, 10, 40),
      new THREE.MeshStandardMaterial({ color: 0x66330a, emissive: 0xff8c1a, emissiveIntensity: 1.3, roughness: 0.4 })
    );
    mesh.position.set(p.x, y, p.z);
    mesh.lookAt(p.x + dir.x * 10, y, p.z + dir.z * 10);
    scene.add(mesh);
    rings.push({ mesh, pos: mesh.position.clone(), normal: dir.clone(), passed: false });
  }
  return rings;
}

function buildClouds(scene) {
  const rand = mulberry32(31);
  const N = 42;
  const geo = new THREE.SphereGeometry(1, 10, 7);
  // unlit material so clouds stay bright white instead of shadowed gray
  const mat = new THREE.MeshBasicMaterial({ color: 0xf4f7fb, transparent: true, opacity: 0.82 });
  const clouds = new THREE.InstancedMesh(geo, mat, N);
  const dummy = new THREE.Object3D();
  for (let i = 0; i < N; i++) {
    dummy.position.set((rand() * 2 - 1) * 1900, 380 + rand() * 180, (rand() * 2 - 1) * 1900);
    dummy.scale.set(28 + rand() * 45, 7 + rand() * 6, 24 + rand() * 40);
    dummy.updateMatrix();
    clouds.setMatrixAt(i, dummy.matrix);
  }
  scene.add(clouds);
}

function buildSky(scene) {
  const mat = new THREE.ShaderMaterial({
    side: THREE.BackSide, depthWrite: false, fog: false,
    uniforms: {
      top: { value: new THREE.Color(0x3f76c8) },
      horizon: { value: new THREE.Color(0xc6d8e8) },
    },
    vertexShader: `
      varying vec3 vDir;
      void main() {
        vDir = normalize(position);
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
      }`,
    fragmentShader: `
      uniform vec3 top; uniform vec3 horizon;
      varying vec3 vDir;
      void main() {
        float t = pow(clamp(vDir.y, 0.0, 1.0), 0.55);
        gl_FragColor = vec4(mix(horizon, top, t), 1.0);
      }`,
  });
  const sky = new THREE.Mesh(new THREE.SphereGeometry(4300, 24, 16), mat);
  scene.add(sky);
  const sun = new THREE.Mesh(
    new THREE.CircleGeometry(130, 24),
    new THREE.MeshBasicMaterial({ color: 0xfff4cc, fog: false })
  );
  sun.position.set(1750, 2500, -1000).normalize().multiplyScalar(4100);
  sun.lookAt(0, 0, 0);
  scene.add(sun);
}

export function buildWorld(scene) {
  scene.add(new THREE.HemisphereLight(0xbdd7f2, 0x62705a, 0.85));
  const sun = new THREE.DirectionalLight(0xfff2dd, 1.6);
  sun.position.set(700, 1000, -400);
  scene.add(sun);
  buildSky(scene);
  buildTerrain(scene);
  buildWater(scene);
  buildPad(scene);
  buildTrees(scene);
  buildVillages(scene);
  buildClouds(scene);
  const rings = buildRings(scene);
  return { rings };
}
