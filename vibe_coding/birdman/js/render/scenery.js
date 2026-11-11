// What stands on the island: trees, the village, the lighthouse, the windmill, boats, clouds over the thermals,
// and the other birds, who show where the air goes up.
import * as THREE from '../../lib/three.module.min.js';
import { U, GLSL, litMaterial, Kit } from './env.js';
import { groundAt, normalAt, POI, thermals, thermalCentre, CLOUDBASE } from '../sim/world.js';
import { mulberry32, fbm } from '../sim/noise.js';

const M4 = new THREE.Matrix4(), Q = new THREE.Quaternion(), V = new THREE.Vector3(), S = new THREE.Vector3(), E = new THREE.Euler();

export function buildTrees(scene) {
  const rng = mulberry32(7);
  const pine = new Kit()
    .put(new THREE.CylinderGeometry(0.22, 0.3, 2.4, 5), 0x4a3524, 0, 1.2, 0)
    .put(new THREE.ConeGeometry(2.3, 4.6, 6), 0x1f4a22, 0, 4.0, 0)
    .put(new THREE.ConeGeometry(1.8, 4.0, 6), 0x25562a, 0, 6.2, 0)
    .put(new THREE.ConeGeometry(1.2, 3.4, 5), 0x2c6230, 0, 8.2, 0).geometry();
  const oak = new Kit()
    .put(new THREE.CylinderGeometry(0.25, 0.4, 2.6, 5), 0x4d3a28, 0, 1.3, 0)
    .put(new THREE.IcosahedronGeometry(2.6, 1), 0x3d6a25, 0, 4.4, 0, 0, 0, 0, 1, 0.82, 1)
    .put(new THREE.IcosahedronGeometry(1.7, 0), 0x4a7a2b, 1.2, 5.3, 0.6)
    .put(new THREE.IcosahedronGeometry(1.5, 0), 0x35602a, -1.3, 4.9, -0.7).geometry();
  const cyp = new Kit()
    .put(new THREE.CylinderGeometry(0.15, 0.2, 1.2, 5), 0x4a3524, 0, 0.6, 0)
    .put(new THREE.ConeGeometry(1.0, 9.5, 6), 0x1c3d24, 0, 5.6, 0).geometry();
  const lists = [[], [], []];
  const place = (x, z, type, s) => { const y = groundAt(x, z); lists[type].push([x, y - 0.2, z, s, rng() * 6.28, 0.8 + rng() * 0.4]); };
  for (let i = 0; i < 160000 && lists[0].length + lists[1].length < 24000; i++) {
    const x = (rng() * 2 - 1) * 3600, z = (rng() * 2 - 1) * 3300, h = groundAt(x, z);
    if (h < 6 || h > 600) continue;
    const n = normalAt(x, z, 9); if (n[1] < 0.8) continue;
    const dF = Math.hypot(x - POI.fields[0], z - POI.fields[2]); if (dF < 730) continue;
    if (Math.hypot(x - POI.village[0], z - POI.village[2]) < 260) continue;
    if (Math.hypot(x - POI.cliff[0], z - POI.cliff[2]) < 110) continue;
    const f = fbm(x / 420, z / 420, 3, 19) + 0.25 * fbm(x / 90, z / 90, 2, 23);
    if (f < -0.04 + (h > 420 ? (h - 420) / 300 : 0)) continue;
    const pineP = h > 150 ? 0.85 : 0.3;
    place(x, z, rng() < pineP ? 0 : 1, 1.0 + rng() * 1.1);
  }
  // cypresses around the village and along the farm lanes
  for (let i = 0; i < 260; i++) {
    const a = rng() * 6.28, r = 120 + rng() * 240, x = POI.village[0] + Math.cos(a) * r, z = POI.village[2] + Math.sin(a) * r;
    if (groundAt(x, z) > 3) place(x, z, 2, 0.8 + rng() * 0.6);
  }
  for (let i = 0; i < 160; i++) { const t = i / 160, x = POI.fields[0] - 600 + t * 1200, z = POI.fields[2] + Math.sin(t * 9) * 60 + (i % 2 ? 320 : -330); if (groundAt(x, z) > 3) place(x, z, i % 5 ? 2 : 1, 0.8 + rng() * 0.5); }
  const out = [];
  [pine, oak, cyp].forEach((g, t) => {
    const m = new THREE.InstancedMesh(g, litMaterial({ sway: 0.0035 }), lists[t].length);
    const c = new THREE.Color();
    lists[t].forEach(([x, y, z, s, r, tint], i) => {
      M4.compose(V.set(x, y, z), Q.setFromEuler(E.set(0, r, 0)), S.set(s, s * (0.85 + tint * 0.25), s)); m.setMatrixAt(i, M4);
      m.setColorAt(i, c.setRGB(0.8 + tint * 0.3, 0.85 + tint * 0.2, 0.75 + tint * 0.2));
    });
    m.frustumCulled = false; scene.add(m); out.push(m);
  });
  return { meshes: out, count: lists[0].length + lists[1].length + lists[2].length };
}

function house(k, x, z, rot, w, d, h, wall, roof) {
  const y = groundAt(x, z), hk = new Kit();
  hk.put(new THREE.BoxGeometry(w, h + 2, d), wall, 0, h / 2 - 1, 0);
  // gable roof: a prism from a 3-sided cylinder
  hk.put(new THREE.CylinderGeometry(d * 0.74, d * 0.74, w * 1.08, 3), roof, 0, h + d * 0.18, 0, 0, 0, Math.PI / 2, 1, 1, 1);
  hk.put(new THREE.BoxGeometry(0.9, 2, 0.1), 0x5a3a22, w * 0.2, 1, d / 2 + 0.03);
  hk.put(new THREE.BoxGeometry(1.0, 1.0, 0.1), 0x3a5a78, -w * 0.22, h * 0.6, d / 2 + 0.03);
  hk.put(new THREE.BoxGeometry(0.6, 1.6, 0.6), wall, -w * 0.3, h + d * 0.45, 0);
  k.merge(hk, M4.compose(V.set(x, y, z), Q.setFromEuler(E.set(0, rot, 0)), S.set(1, 1, 1)));
}

export function buildProps(scene) {
  const k = new Kit(), rng = mulberry32(21);
  const walls = [0xf2ede2, 0xe9dcc0, 0xf5f0ea, 0xe6c9a0, 0xdfe8ee], roofs = [0xb5502f, 0xa84a2a, 0xc2603a, 0x9a4630];
  // village: houses around a square, the post office with its flag at the middle
  const [vx, , vz] = POI.village;
  for (let i = 0; i < 46; i++) {
    const a = rng() * 6.28, r = 26 + Math.sqrt(rng()) * 190, x = vx + Math.cos(a) * r * 1.25, z = vz + Math.sin(a) * r * 0.8;
    if (groundAt(x, z) < 4) continue;
    house(k, x, z, Math.round(rng() * 4) * 0.785 + rng() * 0.2, 7 + rng() * 5, 5.5 + rng() * 2.5, 3.2 + rng() * 2.6, walls[(rng() * walls.length) | 0], roofs[(rng() * roofs.length) | 0]);
  }
  const gy = groundAt(vx, vz);
  k.put(new THREE.BoxGeometry(5, 14, 5), 0xefe6d4, vx + 8, gy + 7, vz - 10).put(new THREE.ConeGeometry(4.4, 5, 4), 0xa84a2a, vx + 8, gy + 16.5, vz - 10, 0, 0.785, 0);
  k.put(new THREE.CylinderGeometry(9, 9, 0.5, 20), 0xcfc4ae, vx, gy + 0.1, vz);
  // jetty and boats
  const jy = 1.2;
  for (let i = 0; i < 9; i++) k.put(new THREE.BoxGeometry(3.2, 0.3, 8.2), 0x7a5a3c, vx - 40, jy, vz + 330 + i * 8).put(new THREE.CylinderGeometry(0.25, 0.25, 5, 5), 0x5a4230, vx - 41.4, jy - 2, vz + 330 + i * 8);
  const boat = (x, z, rot, s, sail) => {
    const b = new Kit();
    b.put(new THREE.SphereGeometry(1, 10, 6, 0, Math.PI * 2, Math.PI / 2, Math.PI / 2), 0x2a4a6a, 0, 0.5, 0, 0, 0, 0, 4.2, 1.3, 1.4);
    b.put(new THREE.BoxGeometry(7.6, 0.15, 2.5), 0xb08a5a, 0, 0.5, 0);
    b.put(new THREE.CylinderGeometry(0.1, 0.12, 8, 5), 0x6a5038, 0.4, 4.4, 0);
    const tri = new THREE.BufferGeometry(); tri.setAttribute('position', new THREE.Float32BufferAttribute([0.4, 1.4, 0.05, -3.4, 1.6, 0.4, 0.4, 8.2, 0.05, 0.4, 1.4, -0.05, 0.4, 8.2, -0.05, -3.4, 1.6, 0.4, 0.6, 1.4, 0, 0.6, 7, 0, 3.3, 1.5, -0.3, 0.6, 1.4, 0, 3.3, 1.5, -0.3, 0.6, 7, 0], 3)); tri.computeVertexNormals();
    b.add(tri, sail);
    k.merge(b, M4.compose(V.set(x, 0, z), Q.setFromEuler(E.set(0, rot, 0)), S.set(s, s, s)));
  };
  boat(POI.boat[0], POI.boat[2], 0.5, 1.5, 0xf4efe2);
  boat(vx - 46, vz + 380, 1.6, 0.8, 0xe9d9b0); boat(vx - 120, vz + 520, 2.4, 1.0, 0xd9584a); boat(-3300, -900, 1.1, 1.3, 0xf4efe2); boat(-2950, 1700, -0.6, 1.1, 0xf0c060); boat(900, 3500, 0.2, 1.2, 0xf4efe2);
  // lighthouse
  const [lx, ly, lz] = POI.light;
  k.put(new THREE.CylinderGeometry(5.5, 6.5, 3, 12), 0x8a8478, lx, ly + 1, lz);
  for (let i = 0; i < 5; i++) k.put(new THREE.CylinderGeometry(2.6 - i * 0.2 - 0.2, 2.6 - i * 0.2, 5.2, 14), i % 2 ? 0xc23a2c : 0xf4f0e8, lx, ly + 5 + i * 5.2, lz);
  k.put(new THREE.CylinderGeometry(2.9, 2.6, 0.5, 14), 0x3a3a3a, lx, ly + 28.5, lz).put(new THREE.CylinderGeometry(1.5, 1.5, 2.6, 10), 0xffe9a0, lx, ly + 30, lz).put(new THREE.ConeGeometry(2.1, 2, 10), 0xc23a2c, lx, ly + 32.3, lz);
  house(k, lx + 14, lz + 6, 0.4, 8, 6, 3.4, 0xf4f0e8, 0xb5502f);
  // shepherd's hut on the mountain shoulder, and the cairn on the summit
  house(k, POI.hut[0], POI.hut[2], 1.1, 7, 5.5, 3, 0x9a8c78, 0x6a5a48);
  const [px, py, pz] = POI.peak;
  k.put(new THREE.ConeGeometry(2.2, 3.4, 6), 0x7a746c, px, py + 1.2, pz).put(new THREE.CylinderGeometry(0.08, 0.08, 5, 5), 0x5a4230, px, py + 4.5, pz).put(new THREE.BoxGeometry(1.8, 1.0, 0.06), 0xd9382a, px + 0.95, py + 6.4, pz);
  // his shed and the take-off stones on the cliff
  const [cx, cy, cz] = POI.cliff;
  house(k, cx + 62, cz + 22, 0.2, 6, 4.5, 2.8, 0x8a6a48, 0x5a4a3a);
  for (let i = 0; i < 12; i++) { const a = i * 0.5236; const x = cx + Math.cos(a) * 9, z = cz + Math.sin(a) * 9; k.put(new THREE.IcosahedronGeometry(0.7, 0), 0xb8ae9c, x, groundAt(x, z) + 0.2, z, a, a * 2, 0); }
  // windmill tower
  const [mx, my, mz] = POI.mill;
  k.put(new THREE.CylinderGeometry(3.2, 4.4, 13, 12), 0xf2ede2, mx, my + 6.5, mz).put(new THREE.ConeGeometry(3.8, 4.5, 12), 0x6a4a3a, mx, my + 15.2, mz);
  // red mailboxes where the letters go
  for (const key of ['village', 'light', 'mill', 'hut']) { const [x, , z] = POI[key]; const bx = x + 12, bz = z + 14, by = groundAt(bx, bz); k.put(new THREE.CylinderGeometry(0.08, 0.08, 1.3, 5), 0x4a4a4a, bx, by + 0.65, bz).put(new THREE.BoxGeometry(0.5, 0.42, 0.75), 0xd0281e, bx, by + 1.45, bz); }
  // stone walls between the fields
  for (let i = 0; i < 70; i++) { const x = POI.fields[0] + (rng() * 2 - 1) * 520, z = POI.fields[2] + (rng() * 2 - 1) * 480, r = rng() < 0.5 ? 0.35 : 1.92; k.put(new THREE.BoxGeometry(40 + rng() * 50, 1.1, 0.8), 0x9a917f, x, groundAt(x, z) + 0.4, z, 0, r, 0); }
  // hay bales
  for (let i = 0; i < 60; i++) { const x = POI.fields[0] + (rng() * 2 - 1) * 480, z = POI.fields[2] + (rng() * 2 - 1) * 440; k.put(new THREE.CylinderGeometry(1.1, 1.1, 1.6, 10), 0xd6b55a, x, groundAt(x, z) + 1.0, z, Math.PI / 2, rng() * 3, 0); }
  const mesh = new THREE.Mesh(k.geometry(), litMaterial({ flat: false }));
  mesh.frustumCulled = false; scene.add(mesh);
  // windmill sails turn with the wind
  const sk = new Kit();
  for (let i = 0; i < 4; i++) { const b = new Kit().put(new THREE.BoxGeometry(0.25, 9, 0.25), 0x5a4230, 0, 4.8, 0).put(new THREE.BoxGeometry(2.2, 7, 0.08), 0xf4efe2, 1.1, 5.6, 0); sk.merge(b, M4.makeRotationZ(i * Math.PI / 2)); }
  const sails = new THREE.Mesh(sk.geometry(), litMaterial({ doubleSide: true }));
  sails.position.set(mx - 4.3, my + 13, mz); sails.rotation.y = Math.PI / 2; scene.add(sails);
  // lighthouse lamp glow
  const glow = new THREE.Mesh(new THREE.SphereGeometry(2.4, 12, 8), new THREE.MeshBasicMaterial({ color: 0xfff0b0, transparent: true, opacity: 0.55, blending: THREE.AdditiveBlending, depthWrite: false }));
  glow.position.set(lx, ly + 30, lz); scene.add(glow);
  return { mesh, sails, glow, update(t) { sails.rotation.x = t * 0.9; glow.scale.setScalar(1 + 0.35 * Math.max(0, Math.sin(t * 1.3)) ** 6); } };
}

export function buildClouds(scene) {
  const rng = mulberry32(99), items = [];
  const puff = (cx, cy, cz, size) => {
    const n = 14 + ((rng() * 8) | 0);
    for (let i = 0; i < n; i++) {
      const a = rng() * 6.28, r = Math.sqrt(rng()) * size * 1.25, s = size * (0.45 + rng() * 0.6) * (1 - r / (size * 2.4));
      items.push([cx + Math.cos(a) * r, cy + s * 0.3 + rng() * rng() * size * 0.5 * (1 - r / (size * 1.4)), cz + Math.sin(a) * r * 0.8, s]);
    }
  };
  for (const t of thermals) { const [x, z] = thermalCentre(t, CLOUDBASE); puff(x, CLOUDBASE, z, t.R * 1.5); }
  for (let i = 0; i < 16; i++) { const a = rng() * 6.28, r = 4500 + rng() * 7000; puff(Math.cos(a) * r, 1250 + rng() * 500, Math.sin(a) * r, 260 + rng() * 300); }
  const mat = new THREE.ShaderMaterial({
    uniforms: { ...U },
    vertexShader: /* glsl */`
      varying vec3 vWp; varying vec3 vN; varying float vY;
      void main() { vec3 p = position; p.y = (p.y < -0.35 ? -0.35 + (p.y + 0.35) * 0.25 : p.y) * 0.8; vY = p.y; vec4 wp = instanceMatrix * vec4(p, 1.0); vWp = wp.xyz; vN = normalize(mat3(instanceMatrix) * normal); gl_Position = projectionMatrix * viewMatrix * wp; }`,
    fragmentShader: /* glsl */`
      ${GLSL}
      varying vec3 vWp; varying vec3 vN; varying float vY;
      void main() {
        vec3 n = normalize(vN), v = normalize(cameraPosition - vWp);
        float w = smoothstep(-0.7, 0.9, dot(n, uSunDir));
        vec3 c = mix(vec3(0.70, 0.75, 0.86), vec3(1.0, 0.97, 0.92) * 1.5, w);
        c *= 0.78 + 0.22 * smoothstep(-0.3, 0.4, vY);
        float rim = pow(1.0 - abs(dot(n, v)), 2.5);
        c += vec3(1.0, 0.88, 0.7) * (0.04 + 0.7 * pow(max(dot(-v, uSunDir), 0.0), 4.0)) * rim;
        gl_FragColor = vec4(finish(haze(c, vWp)), 1.0);
      }`,
  });
  const m = new THREE.InstancedMesh(new THREE.IcosahedronGeometry(1, 3), mat, items.length);
  items.forEach(([x, y, z, s], i) => { M4.compose(V.set(x, y, z), Q.identity(), S.set(s, s, s)); m.setMatrixAt(i, M4); });
  m.frustumCulled = false; scene.add(m);
  return m;
}

// Hawks circling in the thermals, gulls riding the lift along the cliff. Cheap: three triangles each, moved on the CPU.
export function buildBirds(scene) {
  const g = new THREE.BufferGeometry();
  // x forward, z span; wings as two triangles each side (inner, outer) so they can flex
  g.setAttribute('position', new THREE.Float32BufferAttribute([
    0.5, 0, 0, -0.5, 0, 0, -0.1, 0, 1.0, 0.5, 0, 0, -0.1, 0, -1.0, -0.5, 0, 0,
    0.25, 0, 0.9, -0.2, 0, 0.9, -0.25, 0, 2.2, 0.25, 0, -0.9, -0.25, 0, -2.2, -0.2, 0, -0.9,
    -0.4, 0, 0.12, -1.0, 0, 0.3, -1.0, 0, -0.3, -0.4, 0, 0.12, -1.0, 0, -0.3, -0.4, 0, -0.12], 3));
  g.setAttribute('aTip', new THREE.Float32BufferAttribute([0, 0, 0.5, 0, 0.5, 0, 0.5, 0.5, 1, 0.5, 1, 0.5, 0, 0, 0, 0, 0, 0], 1));
  const mat = new THREE.ShaderMaterial({
    uniforms: { ...U }, side: THREE.DoubleSide,
    vertexShader: /* glsl */`
      attribute float aTip; attribute vec4 aAnim; varying vec3 vWp; varying vec3 vC;
      ${GLSL}
      void main() { vec3 p = position; p.y += aTip * aTip * sin(uTime * aAnim.x + aAnim.y) * aAnim.z * abs(p.z) * 0.5 + aTip * 0.12 * abs(p.z);
        vec4 wp = instanceMatrix * vec4(p, 1.0); vWp = wp.xyz; vC = mix(vec3(0.16, 0.12, 0.10), vec3(0.92, 0.92, 0.9), aAnim.w); gl_Position = projectionMatrix * viewMatrix * wp; }`,
    fragmentShader: `${GLSL} varying vec3 vWp; varying vec3 vC; void main() { gl_FragColor = vec4(finish(haze(vC * (uSunCol * 0.4 + 0.5), vWp)), 1.0); }`,
  });
  const list = [], rng = mulberry32(5);
  thermals.forEach((t, ti) => { for (let i = 0; i < 4; i++) list.push({ kind: 0, t, r: t.R * (0.35 + rng() * 0.4), y: t.y0 + 120 + rng() * 650, ph: rng() * 6.28, sp: 11 + rng() * 3, dir: ti % 2 ? 1 : -1, s: 1.1 + rng() * 0.5 }); });
  for (let i = 0; i < 26; i++) list.push({ kind: 1, z0: -1300 + rng() * 2100, len: 150 + rng() * 300, y: 150 + rng() * 120, x: -2420 + rng() * 90, ph: rng() * 6.28, sp: 0.05 + rng() * 0.05, s: 0.55 + rng() * 0.2 });
  for (let i = 0; i < 14; i++) list.push({ kind: 2, cx: POI.village[0] - 60 + rng() * 80, cz: POI.village[2] + 380 + rng() * 200, r: 30 + rng() * 80, y: 12 + rng() * 40, ph: rng() * 6.28, sp: 0.2 + rng() * 0.15, s: 0.5 + rng() * 0.2 });
  const m = new THREE.InstancedMesh(g, mat, list.length);
  const anim = new Float32Array(list.length * 4);
  list.forEach((b, i) => { anim[i * 4] = b.kind === 0 ? 2.2 : 7; anim[i * 4 + 1] = rng() * 6.28; anim[i * 4 + 2] = b.kind === 0 ? 0.12 : 0.55; anim[i * 4 + 3] = b.kind === 0 ? 0 : 1; });
  g.setAttribute('aAnim', new THREE.InstancedBufferAttribute(anim, 4));
  m.frustumCulled = false; scene.add(m);
  return {
    mesh: m,
    update(time) {
      list.forEach((b, i) => {
        let x, y, z, yaw, bank = 0;
        if (b.kind === 0) {
          const a = b.ph + (time * b.sp / b.r) * b.dir, yy = b.y + 40 * Math.sin(time * 0.05 + b.ph), [cx, cz] = thermalCentre(b.t, yy);
          x = cx + Math.cos(a) * b.r; z = cz + Math.sin(a) * b.r; y = yy; yaw = -a - (b.dir * Math.PI) / 2; bank = b.dir * 0.5;
        } else if (b.kind === 1) {
          const u = Math.sin(time * b.sp + b.ph); z = b.z0 + u * b.len; x = b.x + 25 * Math.sin(time * b.sp * 2 + b.ph); y = b.y + 12 * Math.sin(time * 0.3 + b.ph);
          yaw = Math.cos(time * b.sp + b.ph) > 0 ? -Math.PI / 2 : Math.PI / 2; bank = 0.15 * Math.sin(time * b.sp * 2 + b.ph);
        } else {
          const a = b.ph + time * b.sp; x = b.cx + Math.cos(a) * b.r; z = b.cz + Math.sin(a) * b.r; y = b.y + 5 * Math.sin(time * 0.4 + b.ph); yaw = -a - Math.PI / 2; bank = 0.4;
        }
        M4.compose(V.set(x, y, z), Q.setFromEuler(E.set(bank, yaw, 0, 'YXZ')), S.set(b.s, b.s, b.s)); m.setMatrixAt(i, M4);
      });
      m.instanceMatrix.needsUpdate = true;
    },
  };
}

// ---- mission markers: rings to fly through, circles to land in ----
export function buildMarkers(scene) {
  const ringMat = new THREE.MeshBasicMaterial({ color: 0xffc94a, transparent: true, opacity: 0.95 });
  const glowMat = new THREE.MeshBasicMaterial({ color: 0xffb030, transparent: true, opacity: 0.16, blending: THREE.AdditiveBlending, depthWrite: false, side: THREE.DoubleSide });
  const nextMat = new THREE.MeshBasicMaterial({ color: 0xfff1c0, transparent: true, opacity: 0.4 });
  const beamMat = new THREE.ShaderMaterial({
    transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, side: THREE.DoubleSide,
    vertexShader: `varying float vY; void main() { vY = uv.y; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`,
    fragmentShader: `varying float vY; void main() { gl_FragColor = vec4(1.0, 0.82, 0.35, 0.34 * pow(1.0 - vY, 1.5)); }`,
  });
  const group = new THREE.Group(); scene.add(group);
  const torus = new THREE.TorusGeometry(1, 0.05, 8, 48), disc = new THREE.TorusGeometry(1.0, 0.16, 6, 48), beamG = new THREE.CylinderGeometry(1, 1, 1, 32, 1, true).translate(0, 0.5, 0), circ = new THREE.TorusGeometry(1, 0.02, 6, 64).rotateX(Math.PI / 2);
  let objs = [];
  return {
    group,
    // list: [{ type: 'ring'|'land', p, r, dir? }]
    set(list) {
      for (const o of objs) group.remove(o.g);
      objs = list.map((it) => {
        const g = new THREE.Group(); g.position.set(it.p[0], it.p[1], it.p[2]);
        if (it.type === 'ring') {
          const a = new THREE.Mesh(torus, ringMat.clone()), b = new THREE.Mesh(disc, glowMat);
          a.scale.setScalar(it.r); b.scale.setScalar(it.r); g.add(a, b);
          if (it.dir) g.lookAt(it.p[0] + it.dir[0], it.p[1] + it.dir[1], it.p[2] + it.dir[2]);
        } else {
          const c = new THREE.Mesh(circ, ringMat.clone()); c.scale.set(it.r, 60, it.r); c.position.y = 0.6;
          const bm = new THREE.Mesh(beamG, beamMat); bm.scale.set(it.r, 140, it.r);
          g.add(c, bm);
        }
        group.add(g);
        return { g, it };
      });
    },
    // highlight the current target, hide the ones already taken
    update(cur, time) {
      objs.forEach((o, i) => {
        o.g.visible = i >= cur;
        const m = o.g.children[0];
        if (i === cur) { m.material = ringMat; if (o.it.type === 'ring') o.g.children[1].visible = true; const s = 1 + 0.04 * Math.sin(time * 4); if (o.it.type === 'ring') m.scale.setScalar(o.it.r * s); }
        else { m.material = nextMat; if (o.it.type === 'ring') o.g.children[1].visible = false; else o.g.visible = false; }
      });
    },
  };
}
