// Grass tufts around the player. The world is 4 km wide, so grass everywhere
// is impossible — instead a pool of instanced tufts covers a ~120 m disc that
// follows the bike. Placement is a deterministic hash on a fixed world grid,
// so the same tuft always reappears in the same spot (no crawling), and a
// small cache keeps rebuilds cheap while flying fast. A shader patch bends
// the blades in the wind.
import * as THREE from 'three';
import { terrainHeight, terrainNormal, WATER_Y, SPAWN } from './world.js';

const CELL = 2.6;         // m, world grid pitch
const RADIUS = 118;       // m, grass disc around the player
const POOL = 4600;        // instance budget
const REBUILD_DIST = 9;   // m of travel before re-scattering

// deterministic 0..1 hash per grid cell
function hash2(ix, iz) {
  let h = (ix * 374761393 + iz * 668265263) | 0;
  h = Math.imul(h ^ (h >>> 13), 1274126177);
  h ^= h >>> 16;
  return (h >>> 0) / 4294967296;
}

// one tuft: 4 thin crossed triangles leaning outward, darker at the root
function tuftGeo() {
  const pos = [], col = [];
  for (let k = 0; k < 4; k++) {
    const a = (k / 4) * Math.PI + 0.3;
    const dx = Math.cos(a), dz = Math.sin(a);
    const w = 0.09, lean = 0.16;
    // base left, base right, tip
    pos.push(-dz * w + dx * 0, 0, dx * w + dz * 0);
    pos.push(dz * w, 0, -dx * w);
    pos.push(dx * lean, 0.8, dz * lean);
    col.push(0.5, 0.55, 0.45, 0.5, 0.55, 0.45, 1.05, 1.1, 0.95);
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  g.setAttribute('color', new THREE.Float32BufferAttribute(col, 3));
  g.computeVertexNormals();
  return g;
}

export function makeGrass(scene) {
  const timeU = { value: 0 };
  const mat = new THREE.MeshLambertMaterial({ vertexColors: true, side: THREE.DoubleSide });
  mat.onBeforeCompile = (sh) => {
    sh.uniforms.uTime = timeU;
    sh.vertexShader = ('uniform float uTime;\n' + sh.vertexShader).replace(
      '#include <begin_vertex>',
      `#include <begin_vertex>
      float gph = instanceMatrix[3].x * 0.71 + instanceMatrix[3].z * 0.93;
      float gb = transformed.y * transformed.y;
      transformed.x += gb * (0.13 * sin(uTime * 1.9 + gph) + 0.05 * sin(uTime * 4.3 + gph * 2.1));
      transformed.z += gb * 0.09 * sin(uTime * 1.6 + gph * 1.3);`
    );
  };
  const mesh = new THREE.InstancedMesh(tuftGeo(), mat, POOL);
  mesh.frustumCulled = false;
  mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
  scene.add(mesh);

  // cell cache: key -> null (no grass) or [x, y, z, scale, colorHue]
  const cache = new Map();
  const _n = new THREE.Vector3();
  function cellData(ix, iz) {
    const key = ix * 100000 + iz;
    let d = cache.get(key);
    if (d !== undefined) return d;
    d = null;
    const r = hash2(ix, iz);
    if (r < 0.62) {
      const x = (ix + 0.2 + 0.6 * hash2(ix + 7331, iz)) * CELL;
      const z = (iz + 0.2 + 0.6 * hash2(ix, iz + 911)) * CELL;
      const h = terrainHeight(x, z);
      if (h > WATER_Y + 1.6 && h < 130 && Math.hypot(x - SPAWN.x, z - SPAWN.z) > 17) {
        terrainNormal(x, z, _n);
        if (_n.y > 0.84) d = [x, h, z, 0.6 + r * 1.1, 0.26 + hash2(ix + 3, iz + 5) * 0.09];
      }
    }
    if (cache.size > 60000) cache.clear();
    cache.set(key, d);
    return d;
  }

  const dummy = new THREE.Object3D();
  const c = new THREE.Color();
  let lastX = Infinity, lastZ = Infinity;

  function rebuild(px, pz) {
    lastX = px; lastZ = pz;
    const cells = Math.ceil(RADIUS / CELL);
    const cx = Math.round(px / CELL), cz = Math.round(pz / CELL);
    let i = 0;
    for (let ix = cx - cells; ix <= cx + cells && i < POOL; ix++) {
      for (let iz = cz - cells; iz <= cz + cells && i < POOL; iz++) {
        const d = cellData(ix, iz);
        if (d === null) continue;
        const dist = Math.hypot(d[0] - px, d[2] - pz);
        if (dist > RADIUS) continue;
        dummy.position.set(d[0], d[1] - 0.03, d[2]);
        // shrink toward the disc edge so pop-in is gentle
        const edge = Math.min(1, (RADIUS - dist) / 25);
        dummy.scale.setScalar(d[3] * edge);
        dummy.rotation.y = d[4] * 60;
        dummy.updateMatrix();
        mesh.setMatrixAt(i, dummy.matrix);
        mesh.setColorAt(i, c.setHSL(d[4], 0.45, 0.32));
        i++;
      }
    }
    mesh.count = i;
    mesh.instanceMatrix.needsUpdate = true;
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
  }

  function update(playerPos, dt) {
    timeU.value += dt;
    if (Math.hypot(playerPos.x - lastX, playerPos.z - lastZ) > REBUILD_DIST) {
      rebuild(playerPos.x, playerPos.z);
    }
  }

  return { update };
}
