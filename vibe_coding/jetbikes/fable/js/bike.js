// The jet bike mesh — "FLYING HOG / ALPINIST", modeled on the reference photo:
// a classic black motorcycle top half (tank with lettering, brown leather
// saddle, chrome bars, round headlight) riding on a big exposed turbojet slung
// underneath, wrapped in pipes, with two side lift pods, plus a black-clad
// rider with helmet and backpack. Body frame matches physics.js: Y = up,
// Z = forward (body +X is the rider's left). Exhaust flames live in jetfx.js.
import * as THREE from 'three';

function tankTexture() {
  const c = document.createElement('canvas');
  c.width = 256; c.height = 128;
  const ctx = c.getContext('2d');
  ctx.clearRect(0, 0, 256, 128);
  ctx.fillStyle = '#e8e5da';
  ctx.textAlign = 'center';
  ctx.font = 'bold 38px sans-serif';
  ctx.fillText('FLYING HOG', 128, 52);
  ctx.font = '26px sans-serif';
  ctx.fillText('ALPINIST', 128, 96);
  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
}

export function buildBike() {
  const g = new THREE.Group();

  const steel = new THREE.MeshStandardMaterial({ color: 0x4d5158, metalness: 0.85, roughness: 0.38 });
  const dark = new THREE.MeshStandardMaterial({ color: 0x25272b, metalness: 0.8, roughness: 0.5 });
  const chrome = new THREE.MeshStandardMaterial({ color: 0xb9c1c9, metalness: 0.95, roughness: 0.18 });
  const tankM = new THREE.MeshStandardMaterial({ color: 0x1b1d20, metalness: 0.6, roughness: 0.35 });
  const leather = new THREE.MeshStandardMaterial({ color: 0x7b4a28, roughness: 0.85 });
  const rubber = new THREE.MeshStandardMaterial({ color: 0x131418, roughness: 0.95 });
  const cloth = new THREE.MeshStandardMaterial({ color: 0x191a1d, roughness: 0.9 });
  const pack = new THREE.MeshStandardMaterial({ color: 0x53462f, roughness: 0.9 });
  const copper = new THREE.MeshStandardMaterial({ color: 0x9a6132, metalness: 0.85, roughness: 0.4 });
  const visorM = new THREE.MeshStandardMaterial({ color: 0x2a3540, metalness: 0.6, roughness: 0.2 });
  const helmetM = new THREE.MeshStandardMaterial({ color: 0x131417, metalness: 0.3, roughness: 0.3 });

  function add(geo, mat, x, y, z, rx = 0, ry = 0, rz = 0) {
    const m = new THREE.Mesh(geo, mat);
    m.position.set(x, y, z);
    m.rotation.set(rx, ry, rz);
    g.add(m);
    return m;
  }
  // cylinder stretched between two points (struts, forks, limbs)
  const _a = new THREE.Vector3(), _b = new THREE.Vector3(), _Y = new THREE.Vector3(0, 1, 0);
  function tube(ax, ay, az, bx, by, bz, r, mat) {
    _a.set(ax, ay, az); _b.set(bx, by, bz);
    const len = _a.distanceTo(_b);
    const m = new THREE.Mesh(new THREE.CylinderGeometry(r, r, len, 8), mat);
    m.position.copy(_a).add(_b).multiplyScalar(0.5);
    m.quaternion.setFromUnitVectors(_Y, _b.sub(_a).normalize());
    g.add(m);
    return m;
  }
  const RX = Math.PI / 2;

  // ---- main turbojet, slung under the frame ----
  add(new THREE.CylinderGeometry(0.36, 0.36, 1.6, 18), steel, 0, -0.16, -0.15, RX);       // casing
  add(new THREE.CylinderGeometry(0.4, 0.4, 0.28, 18), dark, 0, -0.16, 0.58, RX);          // compressor ring
  add(new THREE.TorusGeometry(0.36, 0.05, 8, 20), chrome, 0, -0.16, 0.73);                // intake lip
  add(new THREE.CylinderGeometry(0.34, 0.34, 0.06, 18), rubber, 0, -0.16, 0.71, RX);      // fan disk
  add(new THREE.ConeGeometry(0.14, 0.28, 12), dark, 0, -0.16, 0.86, RX);                  // spinner
  add(new THREE.CylinderGeometry(0.34, 0.27, 0.4, 18), steel, 0, -0.16, -1.15, RX);       // taper
  add(new THREE.CylinderGeometry(0.27, 0.23, 0.45, 16), dark, 0, -0.16, -1.55, RX);       // afterburner nozzle
  add(new THREE.TorusGeometry(0.235, 0.03, 6, 16), dark, 0, -0.16, -1.76);                // nozzle ring
  // pipes hugging the casing
  for (const [z, mat, a0] of [[0.32, copper, 0.2], [0.02, chrome, 2.1], [-0.4, copper, 4.0]]) {
    const p = add(new THREE.TorusGeometry(0.43, 0.035, 6, 14, Math.PI * 1.25), mat, 0, -0.16, z);
    p.rotation.z = a0;
  }
  // exhaust cans along the flanks
  for (const s of [-1, 1]) {
    add(new THREE.CylinderGeometry(0.08, 0.08, 0.95, 10), chrome, s * 0.45, 0.0, -0.7, RX);
    add(new THREE.CylinderGeometry(0.085, 0.06, 0.12, 10), dark, s * 0.45, 0.0, -1.22, RX);
    add(new THREE.BoxGeometry(0.2, 0.26, 0.42), dark, s * 0.29, -0.4, 0.32);              // accessories
  }
  // frame spine connecting engine to tank and seat
  add(new THREE.BoxGeometry(0.26, 0.2, 2.0), dark, 0, 0.12, -0.05);

  // ---- tank with lettering ----
  const tank = add(new THREE.CapsuleGeometry(0.28, 0.5, 6, 14), tankM, 0, 0.42, 0.4, RX);
  tank.scale.set(0.95, 1, 0.78); // capsule axis is +Y; after RX it lies along Z
  add(new THREE.CylinderGeometry(0.05, 0.05, 0.04, 10), chrome, 0, 0.65, 0.5);            // filler cap
  const decal = new THREE.MeshBasicMaterial({ map: tankTexture(), transparent: true });
  for (const s of [-1, 1]) {
    add(new THREE.PlaneGeometry(0.5, 0.25), decal, s * 0.27, 0.44, 0.4, 0, s * RX);
  }

  // ---- saddle, fender ----
  const seat = add(new THREE.CapsuleGeometry(0.2, 0.55, 5, 12), leather, 0, 0.33, -0.45, RX);
  seat.scale.set(1.05, 1, 0.5);
  const darkLeather = new THREE.MeshStandardMaterial({ color: 0x4e2d17, roughness: 0.9 });
  const hump = add(new THREE.SphereGeometry(0.16, 10, 8), darkLeather, 0, 0.36, -0.85);
  hump.scale.set(1.2, 0.5, 0.7);
  add(new THREE.BoxGeometry(0.38, 0.07, 0.55), dark, 0, 0.27, -1.15, -0.15);
  add(new THREE.BoxGeometry(0.12, 0.06, 0.05), new THREE.MeshStandardMaterial({
    color: 0x801812, emissive: 0xb02010, emissiveIntensity: 0.8 }), 0, 0.31, -1.42);

  // ---- front end: forks, bars, headlight ----
  for (const s of [-1, 1]) tube(s * 0.13, -0.3, 1.18, s * 0.13, 0.44, 0.95, 0.034, chrome);
  add(new THREE.BoxGeometry(0.34, 0.08, 0.12), dark, 0, -0.28, 1.16);                     // lower brace
  add(new THREE.BoxGeometry(0.3, 0.1, 0.16), dark, 0, 0.48, 0.93);                        // triple clamp
  const bar = add(new THREE.CylinderGeometry(0.028, 0.028, 0.8, 8), chrome, 0, 0.57, 0.88);
  bar.rotation.z = RX;
  for (const s of [-1, 1]) {
    const grip = add(new THREE.CylinderGeometry(0.045, 0.045, 0.16, 8), rubber, s * 0.37, 0.57, 0.88);
    grip.rotation.z = RX;
  }
  add(new THREE.CylinderGeometry(0.13, 0.13, 0.1, 14), chrome, 0, 0.32, 1.1, RX);         // headlight bowl
  add(new THREE.CircleGeometry(0.11, 14), new THREE.MeshBasicMaterial({ color: 0xfff6d8 }), 0, 0.32, 1.16);
  tube(0, -0.28, 1.15, 0, -0.52, 1.05, 0.03, dark);                                       // front skid leg
  add(new THREE.BoxGeometry(0.09, 0.05, 0.4), chrome, 0, -0.545, 1.0);

  // ---- lift pods (positions match physics NOZ_L/NOZ_R/NOZ_C) ----
  for (const s of [-1, 1]) {
    add(new THREE.BoxGeometry(0.45, 0.07, 0.2), steel, s * 0.5, -0.1, 0);                 // strut
    add(new THREE.CylinderGeometry(0.15, 0.17, 0.4, 12), steel, s * 0.75, -0.25, 0);      // pod
    const lip = add(new THREE.TorusGeometry(0.12, 0.03, 6, 14), dark, s * 0.75, -0.05, 0);
    lip.rotation.x = RX;
    add(new THREE.CylinderGeometry(0.11, 0.14, 0.14, 12), chrome, s * 0.75, -0.47, 0);    // nozzle
  }
  add(new THREE.CylinderGeometry(0.13, 0.16, 0.12, 12), chrome, 0, -0.52, 0);             // center nozzle

  // ---- landing skids ----
  for (const s of [-1, 1]) {
    add(new THREE.BoxGeometry(0.07, 0.05, 1.5), chrome, s * 0.42, -0.55, -0.15);
    tube(s * 0.42, -0.54, 0.25, s * 0.34, -0.3, 0.25, 0.03, dark);
    tube(s * 0.42, -0.54, -0.55, s * 0.34, -0.3, -0.55, 0.03, dark);
  }

  // ---- rider: black jacket, helmet, backpack ----
  add(new THREE.BoxGeometry(0.34, 0.2, 0.3), cloth, 0, 0.52, -0.42);                      // pelvis
  const torso = add(new THREE.BoxGeometry(0.4, 0.55, 0.24), cloth, 0, 0.78, -0.26, 0.6);
  add(new THREE.BoxGeometry(0.44, 0.12, 0.22), cloth, 0, 0.97, -0.11, 0.6);               // shoulders
  add(new THREE.SphereGeometry(0.17, 14, 12), helmetM, 0, 1.14, -0.03);
  add(new THREE.BoxGeometry(0.21, 0.09, 0.05), visorM, 0, 1.15, 0.13);                    // goggles
  const bp = add(new THREE.BoxGeometry(0.34, 0.44, 0.18), pack, 0, 0.9, -0.45, 0.6);      // backpack
  add(new THREE.BoxGeometry(0.26, 0.12, 0.06), pack, 0, 1.05, -0.35, 0.6);                // top pouch
  for (const s of [-1, 1]) {
    // arms: shoulder -> elbow -> grip
    tube(s * 0.24, 0.97, -0.11, s * 0.3, 0.75, 0.3, 0.05, cloth);
    tube(s * 0.3, 0.75, 0.3, s * 0.37, 0.6, 0.82, 0.045, cloth);
    add(new THREE.SphereGeometry(0.055, 8, 8), rubber, s * 0.37, 0.58, 0.86);             // glove
    // legs: hip -> knee -> peg
    tube(s * 0.15, 0.5, -0.42, s * 0.22, 0.34, 0.12, 0.07, cloth);
    tube(s * 0.22, 0.34, 0.12, s * 0.3, 0.02, -0.08, 0.055, cloth);
    add(new THREE.BoxGeometry(0.11, 0.09, 0.3), rubber, s * 0.3, 0.0, -0.03);             // boot
    add(new THREE.BoxGeometry(0.16, 0.04, 0.08), chrome, s * 0.33, -0.02, -0.08);         // peg
  }
  void torso; void bp;

  return { group: g };
}
