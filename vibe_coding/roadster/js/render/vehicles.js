// מודלים פשוטים של הרכבים החונים בשורה (כמו בהדגמה): סייברקאב, סייברטראק, סמי, מודל Y, מודל 3.
import * as THREE from 'three';
import { profileBody } from './shapes.js';

const glassMat = new THREE.MeshPhysicalMaterial({ color: 0x0c1015, metalness: 0.4, roughness: 0.08, clearcoat: 1 });
const tireMat = new THREE.MeshStandardMaterial({ color: 0x141414, roughness: 0.9 });
const rimMat = new THREE.MeshStandardMaterial({ color: 0x3a3d42, roughness: 0.35, metalness: 0.8 });

function wheel(r, w) {
  const g = new THREE.Group();
  const t = new THREE.Mesh(new THREE.CylinderGeometry(r, r, w, 28), tireMat); t.rotation.z = Math.PI / 2; t.castShadow = true;
  const rim = new THREE.Mesh(new THREE.CylinderGeometry(r * 0.62, r * 0.62, w + 0.01, 20), rimMat); rim.rotation.z = Math.PI / 2;
  g.add(t, rim);
  return g;
}

const DEFS = {
  cybertruck: {
    color: 0xb9bcc0, metal: 0.9, rough: 0.33, bevel: 0.02,
    body: [[-2.84, 0.42], [2.78, 0.42], [2.84, 0.62], [2.84, 1.0], [0.25, 1.79], [-2.84, 1.12]],
    glass: [[1.15, 1.2], [0.28, 1.7], [-1.2, 1.5], [-1.2, 1.22]],
    wheels: { r: 0.44, w: 0.32, z: [1.85, -1.75] },
  },
  semi: {
    color: 0xe8e9ea, metal: 0.3, rough: 0.35, bevel: 0.08,
    body: [[-3.3, 0.55], [3.25, 0.55], [3.3, 1.6], [3.28, 2.5, 3.35, 2.1], [2.3, 3.9, 3.1, 3.7], [0.55, 3.9], [0.55, 1.3], [-3.3, 1.3]],
    glass: [[3.24, 2.55], [2.55, 3.55], [1.9, 3.6], [1.9, 2.55]],
    wheels: { r: 0.52, w: 0.3, z: [2.4, -1.75, -2.85] },
    extra: (g) => {
      const fw = new THREE.Mesh(new THREE.CylinderGeometry(0.55, 0.55, 0.12, 24), rimMat);
      fw.position.set(0, 1.38, -2.2); g.add(fw);
    },
  },
  cybercab: {
    color: 0xcdb98f, metal: 0.75, rough: 0.28, bevel: 0.12,
    body: [[-2.15, 0.28], [2.12, 0.28], [2.15, 0.62], [0.8, 1.22, 1.9, 1.02], [-0.7, 1.45, 0.0, 1.47], [-2.15, 0.82, -1.7, 1.25]],
    glass: [[1.45, 0.98], [0.75, 1.3], [-0.9, 1.38], [-1.55, 1.05]],
    wheels: { r: 0.37, w: 0.26, z: [1.35, -1.35] },
  },
  modely: {
    color: 0xa3141d, metal: 0.4, rough: 0.25, bevel: 0.12,
    body: [[-2.37, 0.3], [2.33, 0.3], [2.37, 0.74], [1.2, 0.98, 1.9, 0.94], [0.45, 1.5], [-0.9, 1.62, -0.2, 1.65], [-2.37, 0.98, -1.9, 1.35]],
    glass: [[1.0, 1.02], [0.42, 1.46], [-1.0, 1.55], [-1.85, 1.05]],
    wheels: { r: 0.37, w: 0.26, z: [1.45, -1.45] },
  },
  model3: {
    color: 0x1f3c88, metal: 0.4, rough: 0.25, bevel: 0.12,
    body: [[-2.36, 0.3], [2.32, 0.3], [2.36, 0.66], [1.25, 0.88, 1.9, 0.84], [0.45, 1.33], [-0.8, 1.44, -0.15, 1.46], [-2.36, 0.9, -1.85, 1.15]],
    glass: [[1.05, 0.9], [0.42, 1.3], [-0.95, 1.38], [-1.8, 0.95]],
    wheels: { r: 0.35, w: 0.24, z: [1.42, -1.45] },
  },
};

export function buildParkedVehicle(v) {
  const D = DEFS[v.key];
  const g = new THREE.Group();
  const mat = new THREE.MeshPhysicalMaterial({ color: D.color, metalness: D.metal, roughness: D.rough, clearcoat: 0.6 });
  const taper = (z) => 1 - 0.1 * Math.pow(Math.min(1, Math.abs(z) / (v.len / 2)), 6);
  const body = new THREE.Mesh(profileBody(D.body, v.wid - 0.04, { bevel: D.bevel, taper }), mat);
  body.castShadow = true; body.receiveShadow = true;
  g.add(body);
  const glass = new THREE.Mesh(profileBody(D.glass, v.wid - 0.02, { bevel: 0.0, taper }), glassMat);
  g.add(glass);
  for (const z of D.wheels.z) for (const sx of [1, -1]) {
    const wh = wheel(D.wheels.r, D.wheels.w);
    wh.position.set(sx * (v.wid / 2 - D.wheels.w / 2 + 0.02), D.wheels.r, z);
    g.add(wh);
  }
  D.extra?.(g);
  g.position.set(v.x, 0, v.z);
  g.rotation.y = v.yaw;
  return g;
}
