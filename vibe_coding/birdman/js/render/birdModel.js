// Yona and his wings. The body is a few capsules; each wing is a three-bone arm (upper arm, forearm,
// bamboo hand-spar) in the wing plane, carrying 44 separate feathers that fan open, twist with the stroke
// and bend under load. The geometry follows the same wing state the physics uses.
import * as THREE from '../../lib/three.module.min.js';
import { litMaterial, Kit } from './env.js';
import { SHOULDER, SEMI } from '../sim/aero.js';
import { clamp, lerp, sstep } from '../sim/vmath.js';

const L1 = 0.42, L2 = 0.5;
const M4 = new THREE.Matrix4(), X = new THREE.Vector3(), Y = new THREE.Vector3(), Z = new THREE.Vector3(), P = new THREE.Vector3(), A = new THREE.Vector3(), B = new THREE.Vector3();

function featherGeo() {
  const pos = [], col = [], idx = [], n = 7;
  for (let i = 0; i <= n; i++) {
    const t = i / n, w = 0.5 * Math.pow(Math.sin(Math.PI * Math.pow(t, 0.55)), 0.6) * (1 - 0.25 * t) + 0.02;
    const sh = 1 - 0.62 * sstep(0.55, 0.95, t);
    for (const s of [-1, 0, 1]) { pos.push(t, s === 0 ? 0.035 * Math.sin(Math.PI * t) + 0.01 : 0, s * w * (s < 0 ? 1.25 : 0.75)); const c = s === 0 ? sh * 0.86 : sh; col.push(c, c, c); }
  }
  for (let i = 0; i < n; i++) for (let j = 0; j < 2; j++) { const a = i * 3 + j, b = a + 1, c = a + 3, d = c + 1; idx.push(a, c, b, b, c, d); }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3)); g.setAttribute('color', new THREE.Float32BufferAttribute(col, 3)); g.setIndex(idx); g.computeVertexNormals();
  return g;
}

export function buildBirdModel(scene) {
  const root = new THREE.Group(), pose = new THREE.Group(); root.add(pose); scene.add(root);
  const mat = litMaterial(), matD = litMaterial({ doubleSide: true });
  const capX = (r, len) => new THREE.CapsuleGeometry(r, len, 5, 10).rotateZ(Math.PI / 2);
  // torso, head, bag
  const BLUE = 0x2c4d8c, NAVY = 0x222c44, SKIN = 0xe2b08c, LEATHER = 0x6e4526, RED = 0xc8281e;
  const k = new Kit()
    .put(capX(0.16, 0.36), BLUE, 0.13, 0, 0, 0, 0, 0, 1, 0.82, 1.22)
    .put(new THREE.SphereGeometry(0.19, 12, 8), BLUE, 0.36, 0.0, 0, 0, 0, 0, 0.8, 0.8, 1.35)
    .put(new THREE.CylinderGeometry(0.168, 0.168, 0.05, 12).rotateZ(Math.PI / 2), LEATHER, -0.08, 0, 0, 0, 0, 0, 1, 0.84, 1.22)
    .put(new THREE.BoxGeometry(0.3, 0.1, 0.24), 0x8a5a2b, 0.02, -0.17, 0.12)
    .put(new THREE.BoxGeometry(0.22, 0.03, 0.2), 0xf2ead8, 0.02, -0.225, 0.12)
    .put(new THREE.CylinderGeometry(0.055, 0.06, 0.1, 8).rotateZ(Math.PI / 2), SKIN, 0.53, 0.0, 0)
    .put(new THREE.CylinderGeometry(0.075, 0.09, 0.06, 10).rotateZ(Math.PI / 2), RED, 0.5, 0.0, 0);
  // straps of the harness across the back
  k.put(new THREE.BoxGeometry(0.5, 0.02, 0.05), LEATHER, 0.15, 0.135, 0.0, 0, 0.5, 0).put(new THREE.BoxGeometry(0.5, 0.02, 0.05), LEATHER, 0.15, 0.135, 0.0, 0, -0.5, 0);
  pose.add(new THREE.Mesh(k.geometry(), mat));
  // head: tilts back in flight to look where he is going
  const head = new THREE.Group(); head.position.set(0.58, 0.0, 0);
  const hk = new Kit()
    .put(new THREE.SphereGeometry(0.115, 14, 10), SKIN, 0.06, 0, 0)
    .put(new THREE.SphereGeometry(0.122, 14, 10, 0, Math.PI * 2, 0, Math.PI * 0.55), LEATHER, 0.06, 0.004, 0, 0, 0, -0.5)
    .put(new THREE.BoxGeometry(0.04, 0.07, 0.03), LEATHER, 0.03, -0.07, 0.11).put(new THREE.BoxGeometry(0.04, 0.07, 0.03), LEATHER, 0.03, -0.07, -0.11)
    .put(new THREE.CylinderGeometry(0.036, 0.036, 0.03, 10), 0x1a1a1e, 0.105, -0.1, 0.045, 0, 0, 0.75).put(new THREE.CylinderGeometry(0.036, 0.036, 0.03, 10), 0x1a1a1e, 0.105, -0.1, -0.045, 0, 0, 0.75)
    .put(new THREE.BoxGeometry(0.02, 0.022, 0.23), 0x3a2a1a, 0.09, -0.085, 0, 0, 0, 0.75)
    .put(new THREE.ConeGeometry(0.02, 0.04, 6), 0xd09a78, 0.07, -0.125, 0, 0, 0, 2.2);
  head.add(new THREE.Mesh(hk.geometry(), mat)); pose.add(head);
  // legs
  const legs = [1, -1].map((s) => {
    const g = new THREE.Group(); g.position.set(-0.15, -0.02, s * 0.09);
    const lk = new Kit().put(capX(0.078, 0.62), NAVY, -0.4, 0, 0).put(capX(0.07, 0.1), LEATHER, -0.84, 0, 0).put(new THREE.BoxGeometry(0.1, 0.2, 0.1), LEATHER, -0.9, -0.08, 0);
    g.add(new THREE.Mesh(lk.geometry(), mat)); pose.add(g); return g;
  });
  // tail: a fan of cloth feathers strapped between the calves, and the fin above it
  const tail = new THREE.Group(); tail.position.set(-0.62, 0.02, 0); pose.add(tail);
  const fg = featherGeo(), tk = new Kit();
  for (let i = 0; i < 9; i++) { const a = (i - 4) * 0.115; tk.merge(new Kit().add(fg, i % 2 ? 0xf1e8d6 : 0xe8dcc4), M4.compose(P.set(0, 0.004 * (i % 2), 0), new THREE.Quaternion().setFromEuler(new THREE.Euler(0, Math.PI - a, 0)), X.set(0.98 - Math.abs(i - 4) * 0.035, 1, 0.3))); }
  const fin = new THREE.BufferGeometry(); fin.setAttribute('position', new THREE.Float32BufferAttribute([-0.1, 0, 0, -0.85, 0, 0, -0.95, 0.55, 0, -0.1, 0, 0, -0.95, 0.55, 0, -0.55, 0.42, 0], 3)); fin.computeVertexNormals();
  const finMesh = new THREE.Mesh(new Kit().add(fin, 0xb8452c).geometry(), matD); tail.add(finMesh);
  tail.add(new THREE.Mesh(tk.geometry(), matD));
  // arms and spars: unit cylinders stretched between joints each frame
  const cyl = new THREE.CylinderGeometry(1, 1, 1, 7).rotateZ(Math.PI / 2).translate(0.5, 0, 0);
  const limb = (color, r) => { const m = new THREE.Mesh(new Kit().add(cyl, color).geometry(), mat); m.userData.r = r; m.matrixAutoUpdate = false; pose.add(m); return m; };
  const arms = [1, -1].map(() => ({ upper: limb(BLUE, 0.062), fore: limb(BLUE, 0.05), spar: limb(0xc9a25a, 0.022), hand: limb(SKIN, 0.045) }));
  const between = (m, a, b, rScale = 1) => {
    X.subVectors(b, a); const len = X.length(); X.divideScalar(len);
    Y.set(0, 1, 0); if (Math.abs(X.y) > 0.95) Y.set(0, 0, 1);
    Z.crossVectors(X, Y).normalize(); Y.crossVectors(Z, X);
    const r = m.userData.r * rScale;
    m.matrix.set(X.x * len, Y.x * r, Z.x * r, a.x, X.y * len, Y.y * r, Z.y * r, a.y, X.z * len, Y.z * r, Z.z * r, a.z, 0, 0, 0, 1);
  };

  // feathers: bone 0 = upper arm (tertials), 1 = forearm (secondaries), 2 = hand spar (primaries). 'c' = covert, the short row on top.
  const defs = [];
  for (let i = 0; i < 10; i++) { const t = 0.06 + 0.94 * (i / 9); defs.push({ bone: 2, t, len: lerp(1.3, 1.68, t) - 0.28 * Math.pow(t, 7), wid: 0.25, c: 0, tone: i }); }
  for (let i = 0; i < 11; i++) { const t = i / 10; defs.push({ bone: 1, t, len: 1.36, wid: 0.3, c: 0, tone: i }); }
  for (let i = 0; i < 5; i++) { const t = 0.1 + 0.9 * (i / 4); defs.push({ bone: 0, t, len: lerp(0.95, 1.33, t), wid: 0.32, c: 0, tone: i }); }
  for (let b = 0; b < 3; b++) for (let i = 0; i < 6; i++) defs.push({ bone: b, t: (i + 0.5) / 6, len: b === 2 ? 0.6 : 0.55, wid: b === 2 ? 0.42 : 0.3, c: 1, tone: i });
  const NF = defs.length;
  const feathers = new THREE.InstancedMesh(fg, matD, NF * 2);
  const col = new THREE.Color();
  for (let s = 0; s < 2; s++) defs.forEach((d, i) => {
    const v = 0.94 + 0.06 * Math.sin(d.tone * 2.3 + s);
    if (d.c) col.setRGB(1.0 * v, 0.97 * v, 0.9 * v); else if (d.bone === 2) col.setRGB(0.96 * v, 0.9 * v, 0.78 * v); else col.setRGB(0.95 * v, 0.9 * v, 0.8 * v);
    feathers.setColorAt(s * NF + i, col);
  });
  feathers.frustumCulled = false; pose.add(feathers);

  const d = new THREE.Vector3(), c0 = new THREE.Vector3(), n0 = new THREE.Vector3(), sh = new THREE.Vector3(), el = new THREE.Vector3(), wr = new THREE.Vector3(), tp = new THREE.Vector3(), dir = new THREE.Vector3(), nn = new THREE.Vector3();
  const tips = [new THREE.Vector3(), new THREE.Vector3()];
  const inPlane = (out, ang) => out.copy(d).multiplyScalar(Math.cos(ang)).addScaledVector(c0, Math.sin(ang));
  // rotate v about the span axis d by th
  const twistAbout = (v, th) => { const c = Math.cos(th), s = Math.sin(th), k = d.dot(v) * (1 - c); A.crossVectors(d, v); v.multiplyScalar(c).addScaledVector(A, s).addScaledVector(d, k); };

  function wing(sideIdx, W, bird) {
    const side = sideIdx === 0 ? 1 : -1, f = W.fold, fan = sstep(0.08, 0.85, f);
    const lam = W.sweep + (1 - f) * 0.25;
    const cl = Math.cos(lam), sl = Math.sin(lam), cp = Math.cos(W.phi), sp = Math.sin(W.phi);
    d.set(-sl * cp, sp, side * cl * cp);
    c0.set(1, 0, 0).addScaledVector(d, -d.x).normalize();
    n0.crossVectors(d, c0).multiplyScalar(side);
    sh.set(SHOULDER[0], SHOULDER[1], side * SHOULDER[2]);
    const a1 = lerp(1.0, 0.3, f), a2 = lerp(1.45, -0.12, f), a3 = lerp(-1.62, -0.3, f), L3 = lerp(1.1, 1.75, f), pl = lerp(0.6, 1, f);
    inPlane(el, a1).multiplyScalar(L1).add(sh); inPlane(wr, a2).multiplyScalar(L2).add(el); inPlane(tp, a3).multiplyScalar(L3).add(wr);
    const arm = arms[sideIdx];
    between(arm.upper, sh, el); between(arm.fore, el, wr); between(arm.spar, wr, tp);
    A.copy(wr).addScaledVector(X.subVectors(wr, el).normalize(), 0.09); between(arm.hand, wr, A);
    const Vf = Math.max(bird.airspeed, 6), load = clamp(bird.gload, -1.5, 5) * f, span = SEMI * (0.26 + 0.74 * f);
    for (let i = 0; i < NF; i++) {
      const fd = defs[i];
      let ang, len = fd.len;
      if (fd.bone === 2) { P.lerpVectors(wr, tp, fd.t); ang = a3 + lerp(-1.18, 0.06, Math.pow(fd.t, 0.85)) * fan - 0.06 * (1 - fan) * (1 - fd.t); len *= pl; }
      else if (fd.bone === 1) { P.lerpVectors(el, wr, fd.t); ang = -1.5 + 0.12 * fd.t * fan - 0.12 * (1 - fan); len *= lerp(0.78, 1, f); }
      else { P.lerpVectors(sh, el, fd.t); ang = -1.57 + 0.04 * (1 - fd.t); len *= lerp(0.8, 1, f); }
      if (fd.c) P.addScaledVector(n0, 0.022); else P.addScaledVector(n0, 0.004 * (i % 3));
      inPlane(dir, ang); nn.copy(n0);
      // where along the span this feather works: twist it like the strip it belongs to, bend it by the load it carries
      const u = clamp((P.dot(d) - sh.dot(d) + 0.5 * len * Math.max(dir.dot(d), 0)) / span, 0, 1), s = u * span;
      const th = W.twist * (0.3 + 0.7 * u) + W.eta * Math.atan2(s * W.phiDot, Vf) * 0.85 - (W.unload || 0) * u + 0.06 * (i % 2 ? 1 : -1) * 0.3;
      twistAbout(dir, th); twistAbout(nn, th);
      const bend = 0.035 * load * u * u + (fd.bone === 2 && !fd.c ? 0.05 * load * fd.t : 0) - 0.02 * W.phiDot * u * u;
      dir.addScaledVector(nn, bend).normalize();
      P.addScaledVector(n0, 0.05 * load * u * u);
      Z.crossVectors(dir, nn).normalize().multiplyScalar(side); Y.crossVectors(Z, dir).multiplyScalar(side);
      const w = fd.wid * lerp(0.75, 1, fan);
      M4.set(dir.x * len, Y.x, Z.x * w, P.x, dir.y * len, Y.y, Z.y * w, P.y, dir.z * len, Y.z, Z.z * w, P.z, 0, 0, 0, 1);
      feathers.setMatrixAt(sideIdx * NF + i, M4);
      if (fd.bone === 2 && !fd.c && i === 9) tips[sideIdx].copy(P).addScaledVector(dir, len);
    }
  }

  const q = new THREE.Quaternion();
  let headLook = 0;
  return {
    root, tips, neck: new THREE.Vector3(0.5, 0.06, 0),
    update(bird, dt) {
      root.position.set(bird.p[0], bird.p[1], bird.p[2]);
      root.quaternion.set(bird.q[1], bird.q[2], bird.q[3], bird.q[0]);
      const up = sstep(0, 1, bird.poseUp);
      // standing: body upright, leaning into the run
      pose.rotation.z = up * (1.5 - 0.035 * Math.abs(bird.gspeed || 0));
      if (bird.mode === 'crash') { pose.rotation.z = 0.1; pose.rotation.x = Math.min(bird.crashT * 6, 1.4); } else if (bird.mode === 'water') { pose.rotation.x = 0; pose.rotation.z = 0.5; } else pose.rotation.x = 0;
      const run = up * clamp(Math.abs(bird.gspeed) / 6, 0, 1), sw = Math.sin(bird.stride * 2.1) * 0.75 * run;
      const hang = bird.legs * (1 - up) * 1.0;
      legs[0].rotation.z = hang + sw; legs[1].rotation.z = hang * 0.92 - sw;
      tail.rotation.z = hang * 0.75 + bird.tail.inc * 0.8 * (1 - up) + up * 0.12;
      finMesh.rotation.y = bird.tail.rud * (1 - up);
      tail.scale.set(1, 1, 0.75 + 0.45 * bird.fold);
      headLook += ((1 - up) * 0.95 - headLook) * Math.min(1, dt * 6);
      head.rotation.z = headLook + up * 0.1; head.rotation.x = -bird.uR * 0.25 * (1 - up);
      // on the ground the wings rest folded; give them a little life while running
      const WL = { ...bird.L }, WR = { ...bird.R };
      if (bird.mode === 'ground') { const idle = 0.12 * (1 - bird.fold); WL.phi += idle; WR.phi += idle; WL.sweep = WR.sweep = 0.14; }
      wing(0, WR, bird); wing(1, WL, bird);
      feathers.instanceMatrix.needsUpdate = true;
    },
  };
}
