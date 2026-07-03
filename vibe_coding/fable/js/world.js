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

// High-frequency near-white detail map, multiplied with the terrain's vertex
// colors: random speckle + short dark strokes that read as grass/soil texture
// when flying low. Tiled ~14 m, so tiling is invisible in the speckle.
function detailTexture() {
  const S = 256;
  const c = document.createElement('canvas');
  c.width = c.height = S;
  const ctx = c.getContext('2d');
  const rnd = mulberry32(99);
  const img = ctx.createImageData(S, S);
  for (let i = 0; i < S * S; i++) {
    const v = 225 + (rnd() * 2 - 1) * 24;
    img.data[i * 4] = v - 4;
    img.data[i * 4 + 1] = v;
    img.data[i * 4 + 2] = v - 10;
    img.data[i * 4 + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
  for (let i = 0; i < 2400; i++) {
    const x = rnd() * S, y = rnd() * S, l = 2 + rnd() * 4;
    ctx.strokeStyle = rnd() < 0.6 ? 'rgba(30,50,20,0.16)' : 'rgba(255,255,235,0.13)';
    ctx.beginPath();
    ctx.moveTo(x, y);
    ctx.lineTo(x + (rnd() - 0.5) * 2, y - l);
    ctx.stroke();
  }
  const tex = new THREE.CanvasTexture(c);
  tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
  tex.repeat.set(300, 300);
  tex.anisotropy = 8;
  tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
}

function buildTerrain(scene) {
  const size = WORLD_HALF * 2, seg = 300;
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
  const mesh = new THREE.Mesh(geo, new THREE.MeshLambertMaterial({ vertexColors: true, map: detailTexture() }));
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
  const conifers = [], broadleafs = [];
  const n = new THREE.Vector3();
  for (let t = 0; t < 20000 && conifers.length + broadleafs.length < 1500; t++) {
    const x = (rand() * 2 - 1) * 1900, z = (rand() * 2 - 1) * 1900;
    const h = terrainHeight(x, z);
    if (h < WATER_Y + 2 || h > 150) continue;
    if (Math.hypot(x - SPAWN.x, z - SPAWN.z) < 45) continue;
    terrainNormal(x, z, n);
    if (n.y < 0.86) continue;
    // conifers dominate up high, broadleafs in the valleys
    const pConifer = 0.35 + smoothstep(40, 110, h) * 0.55;
    (rand() < pConifer ? conifers : broadleafs).push([x, h, z, 0.7 + rand() * 0.9, rand()]);
  }

  function forest(list, trunkGeo, canopyGeo, hue0, hueSpan) {
    if (!list.length) return;
    const trunks = new THREE.InstancedMesh(trunkGeo, new THREE.MeshLambertMaterial({ color: 0x6b4a2f }), list.length);
    const canopies = new THREE.InstancedMesh(canopyGeo, new THREE.MeshLambertMaterial({ color: 0xffffff }), list.length);
    const dummy = new THREE.Object3D();
    const c = new THREE.Color();
    for (let i = 0; i < list.length; i++) {
      const [x, h, z, s, r] = list[i];
      dummy.position.set(x, h - 0.2, z);
      dummy.scale.setScalar(s);
      dummy.rotation.y = r * Math.PI * 2;
      dummy.updateMatrix();
      trunks.setMatrixAt(i, dummy.matrix);
      canopies.setMatrixAt(i, dummy.matrix);
      c.setHSL(hue0 + r * hueSpan, 0.5, 0.26 + r * 0.13);
      canopies.setColorAt(i, c);
    }
    scene.add(trunks, canopies);
  }

  const pineTrunk = new THREE.CylinderGeometry(0.22, 0.4, 3);
  pineTrunk.translate(0, 1.5, 0);
  const pineCanopy = new THREE.ConeGeometry(2.1, 6, 7);
  pineCanopy.translate(0, 5.5, 0);
  forest(conifers, pineTrunk, pineCanopy, 0.28, 0.07);

  const oakTrunk = new THREE.CylinderGeometry(0.28, 0.5, 3.6);
  oakTrunk.translate(0, 1.8, 0);
  const oakCanopy = new THREE.IcosahedronGeometry(2.6, 1);
  oakCanopy.scale(1, 0.82, 1);
  oakCanopy.translate(0, 4.9, 0);
  forest(broadleafs, oakTrunk, oakCanopy, 0.23, 0.09);
}

// Unit gable-roof prism: 1x1 base centered at origin, ridge along Z at y=1.
// Non-indexed so computeVertexNormals gives flat-shaded slopes.
function gableGeo() {
  const A = [-0.5, 0, -0.5], B = [0.5, 0, -0.5], C = [0.5, 0, 0.5], D = [-0.5, 0, 0.5];
  const R1 = [0, 1, -0.5], R2 = [0, 1, 0.5];
  const tris = [
    A, D, R2, A, R2, R1,   // left slope
    B, R1, R2, B, R2, C,   // right slope
    A, R1, B,              // front gable
    D, C, R2,              // back gable
  ];
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(tris.flat(), 3));
  g.computeVertexNormals();
  return g;
}

// Villages laid out along streets, plus lone farms. Every building is
// assembled from shared instanced parts: unit boxes (walls, flat roofs,
// chimneys, towers), gable prisms, cylinders and cones (silos, steeples).
function buildSettlements(scene) {
  const rand = mulberry32(555);
  const n = new THREE.Vector3();
  const boxes = [], gables = [], cyls = [], cones = [];
  const wallPal = [0xf2ede2, 0xe8dcc8, 0xd9c7a8, 0xd3dade, 0xead9d0, 0xcfd8c6, 0xf4e6c0];
  const roofPal = [0xa8473a, 0x7d3324, 0x9c5a3c, 0x4a4e57, 0x6d7178, 0x8a6b42];
  const pick = (pal) => pal[(rand() * pal.length) | 0];

  function groundAt(x, z) {
    const h = terrainHeight(x, z);
    if (h < WATER_Y + 1.8 || h > 110) return null;
    terrainNormal(x, z, n);
    return n.y > 0.93 ? h : null;
  }

  // part pushers: [x, y, z, ry, sx, sy, sz, color]
  const box = (x, y, z, ry, sx, sy, sz, c) => boxes.push([x, y, z, ry, sx, sy, sz, c]);
  const gable = (x, y, z, ry, sx, sy, sz, c) => gables.push([x, y, z, ry, sx, sy, sz, c]);

  function chimney(x, y, z, ry, w, d, hWall, hRoof) {
    // sits astride the ridge, offset along the house length
    const off = (rand() - 0.5) * d * 0.5;
    const cx = x + Math.sin(ry) * off, cz = z + Math.cos(ry) * off;
    box(cx, y + hWall + hRoof * 0.55, cz, ry, 0.45, 1.15, 0.45, 0x9a8f85);
  }

  function addHouse(x, y, z, ry, kind) {
    const r = rand();
    const wall = pick(wallPal), roof = pick(roofPal);
    if (kind === 'cottage') {
      const w = 4.2 + r * 1.8, d = 5 + rand() * 2.2, h = 2.7;
      box(x, y, z, ry, w, h, d, wall);
      gable(x, y + h, z, ry, w + 0.55, 1.35 + rand() * 0.8, d + 0.7, roof);
      if (rand() < 0.6) chimney(x, y, z, ry, w, d, h, 1.5);
    } else if (kind === 'two-story') {
      const w = 4.8 + r * 1.4, d = 5.6 + rand() * 1.6, h = 5.3;
      box(x, y, z, ry, w, h, d, wall);
      gable(x, y + h, z, ry, w + 0.55, 1.5 + rand() * 0.7, d + 0.7, roof);
      chimney(x, y, z, ry, w, d, h, 1.7);
    } else if (kind === 'flat') {
      const w = 4.4 + r * 2.4, d = 4.2 + rand() * 2.2, h = 2.9 + rand() * 0.8;
      box(x, y, z, ry, w, h, d, wall);
      box(x, y + h, z, ry, w + 0.5, 0.25, d + 0.5, 0xb9b1a4);   // roof slab
      if (rand() < 0.45) box(x + Math.cos(ry) * 1.2, y + h + 0.25, z - Math.sin(ry) * 1.2, ry, 1.5, 1.0, 1.8, wall);
    } else if (kind === 'barn') {
      const w = 5.4 + r, d = 9 + rand() * 3.5, h = 3.6;
      box(x, y, z, ry, w, h, d, 0x8a3b2c);
      gable(x, y + h, z, ry, w + 0.5, 2.1 + rand() * 0.5, d + 0.6, 0x55595f);
    } else if (kind === 'church') {
      const w = 5.6, d = 8.5, h = 4.2;
      box(x, y, z, ry, w, h, d, 0xf3efe6);
      gable(x, y + h, z, ry, w + 0.5, 2.3, d + 0.6, 0x8a4438);
      // bell tower at the front, topped with a steeple
      const fx = x + Math.sin(ry) * (d / 2 + 1.2), fz = z + Math.cos(ry) * (d / 2 + 1.2);
      box(fx, y, fz, ry, 2.3, 7.2, 2.3, 0xf3efe6);
      cones.push([fx, y + 7.2, fz, ry, 1.7, 2.9, 1.7, 0x6d4a35]);
    }
  }

  function addFarm(x, y, z, ry) {
    addHouse(x, y, z, ry, 'barn');
    const cx = x + Math.cos(ry) * 11, cz = z - Math.sin(ry) * 11;
    const ch = groundAt(cx, cz);
    if (ch !== null) addHouse(cx, ch - 0.35, cz, ry + rand() - 0.5, 'cottage');
    if (rand() < 0.65) {
      const sx = x - Math.cos(ry) * 6.5, sz = z + Math.sin(ry) * 6.5;
      const sh = groundAt(sx, sz);
      if (sh !== null) {
        cyls.push([sx, sh - 0.3, sz, 0, 1.5, 5.6, 1.5, 0xb6bcc4]);
        cones.push([sx, sh + 5.3, sz, 0, 1.62, 1.5, 1.62, 0x88553a]);
      }
    }
  }

  function houseKind(r) {
    if (r < 0.42) return 'cottage';
    if (r < 0.62) return 'two-story';
    if (r < 0.85) return 'flat';
    return 'barn';
  }

  // one street (or two) per village, houses facing the street
  const centers = [];
  let villages = 0;
  for (let t = 0; t < 8000 && villages < 12; t++) {
    const cx = (rand() * 2 - 1) * 1600, cz = (rand() * 2 - 1) * 1600;
    if (groundAt(cx, cz) === null) continue;
    if (Math.hypot(cx - SPAWN.x, cz - SPAWN.z) < 130) continue;
    if (centers.some(([ox, oz]) => Math.hypot(cx - ox, cz - oz) < 320)) continue;
    centers.push([cx, cz]);
    villages++;
    const big = rand() < 0.45;
    const streets = big ? 2 : 1;
    let churchPlaced = !big;
    for (let st = 0; st < streets; st++) {
      const ang = rand() * Math.PI + st * Math.PI / 2;
      const ux = Math.sin(ang), uz = Math.cos(ang);
      const len = 50 + rand() * 65;
      for (let s = -len; s <= len; s += 13 + rand() * 6) {
        for (const side of [-1, 1]) {
          if (rand() < 0.22) continue;   // leave gaps
          const off = side * (8 + rand() * 4);
          const x = cx + ux * s - uz * off + (rand() - 0.5) * 3;
          const z = cz + uz * s + ux * off + (rand() - 0.5) * 3;
          const h = groundAt(x, z);
          if (h === null) continue;
          const face = ang + (side > 0 ? -Math.PI / 2 : Math.PI / 2) + (rand() - 0.5) * 0.12;
          if (!churchPlaced && Math.abs(s) < 15) {
            addHouse(x, h - 0.35, z, face, 'church');
            churchPlaced = true;
          } else {
            addHouse(x, h - 0.35, z, face, houseKind(rand()));
          }
        }
      }
    }
  }

  // scattered lone farms
  let farms = 0;
  for (let t = 0; t < 4000 && farms < 16; t++) {
    const x = (rand() * 2 - 1) * 1750, z = (rand() * 2 - 1) * 1750;
    if (groundAt(x, z) === null) continue;
    if (Math.hypot(x - SPAWN.x, z - SPAWN.z) < 100) continue;
    if (centers.some(([ox, oz]) => Math.hypot(x - ox, z - oz) < 200)) continue;
    farms++;
    addFarm(x, terrainHeight(x, z) - 0.35, z, rand() * Math.PI * 2);
  }

  function inst(geo, list, opts = {}) {
    if (!list.length) return;
    const mesh = new THREE.InstancedMesh(
      geo, new THREE.MeshLambertMaterial({ color: 0xffffff, ...opts }), list.length);
    const dummy = new THREE.Object3D();
    const c = new THREE.Color();
    list.forEach((e, i) => {
      dummy.position.set(e[0], e[1], e[2]);
      dummy.rotation.y = e[3];
      dummy.scale.set(e[4], e[5], e[6]);
      dummy.updateMatrix();
      mesh.setMatrixAt(i, dummy.matrix);
      mesh.setColorAt(i, c.setHex(e[7]));
    });
    scene.add(mesh);
  }

  const unitBox = new THREE.BoxGeometry(1, 1, 1);
  unitBox.translate(0, 0.5, 0);
  const unitCyl = new THREE.CylinderGeometry(1, 1, 1, 12);
  unitCyl.translate(0, 0.5, 0);
  const unitCone = new THREE.ConeGeometry(1, 1, 10);
  unitCone.translate(0, 0.5, 0);
  inst(unitBox, boxes);
  inst(gableGeo(), gables, { side: THREE.DoubleSide });
  inst(unitCyl, cyls);
  inst(unitCone, cones);
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
  buildSettlements(scene);
  buildClouds(scene);
  const rings = buildRings(scene);
  return { rings };
}
