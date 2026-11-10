// The F-3 and its rider, built from primitives in the design frame of the physics
// (x forward, y up, z right; origin on the ground under the bike). Pods tilt, vanes, elevator,
// ailerons, rudder and air brake move with the simulated actuators; the rider leans.
import * as THREE from '../../lib/three.module.min.js';
import { PODS, ENG, SKIDS } from '../sim/bike.js';

export function buildBike(ghost = false) {
  const root = new THREE.Group();          // origin = centre of gravity
  const d = new THREE.Group(); root.add(d);  // design frame, shifted by -cg every frame
  const M = (o) => (ghost ? new THREE.MeshBasicMaterial({ color: 0x66ddff, transparent: true, opacity: 0.22, depthWrite: false }) : o);
  const paint = M(new THREE.MeshPhysicalMaterial({ color: 0xf2f0ea, roughness: 0.35, metalness: 0.1, clearcoat: 1, clearcoatRoughness: 0.08 }));
  const accent = M(new THREE.MeshPhysicalMaterial({ color: 0xe8500a, roughness: 0.35, metalness: 0.1, clearcoat: 1, clearcoatRoughness: 0.1 }));
  const carbon = M(new THREE.MeshStandardMaterial({ color: 0x17181b, roughness: 0.45, metalness: 0.4 }));
  const steel = M(new THREE.MeshStandardMaterial({ color: 0x8d9299, roughness: 0.3, metalness: 0.9 }));
  const titan = M(new THREE.MeshStandardMaterial({ color: 0x7a6a5c, roughness: 0.35, metalness: 0.95 }));
  const rubber = M(new THREE.MeshStandardMaterial({ color: 0x0c0c0d, roughness: 0.9 }));
  const suit = M(new THREE.MeshStandardMaterial({ color: 0x1d2c4a, roughness: 0.75 }));
  const suit2 = M(new THREE.MeshStandardMaterial({ color: 0xe8500a, roughness: 0.7 }));
  const helmet = M(new THREE.MeshPhysicalMaterial({ color: 0xf5f5f5, roughness: 0.2, clearcoat: 1 }));
  const visor = M(new THREE.MeshPhysicalMaterial({ color: 0x4a3520, roughness: 0.08, metalness: 1.0 }));
  const glass = M(new THREE.MeshPhysicalMaterial({ color: 0x88aacc, roughness: 0.05, transparent: true, opacity: 0.35 }));
  const all = [];
  const add = (geo, mat, x, y, z, parent = d) => { const m = new THREE.Mesh(geo, mat); m.position.set(x, y, z); m.castShadow = !ghost; m.receiveShadow = !ghost; parent.add(m); all.push(m); return m; };
  const box = (sx, sy, sz) => new THREE.BoxGeometry(sx, sy, sz);
  const cylX = (r0, r1, len, seg = 16) => { const g = new THREE.CylinderGeometry(r0, r1, len, seg); g.rotateZ(-Math.PI / 2); return g; };   // axis along +x, r0 at +x end
  const ell = (rx, ry, rz) => { const g = new THREE.SphereGeometry(1, 20, 14); g.scale(rx, ry, rz); return g; };
  const slab = (pts, thick) => {      // wing-like plate: planform in (x,z), thickness in y
    const s = new THREE.Shape(); s.moveTo(pts[0][0], pts[0][1]); for (let i = 1; i < pts.length; i++) s.lineTo(pts[i][0], pts[i][1]);
    const g = new THREE.ExtrudeGeometry(s, { depth: thick, bevelEnabled: true, bevelThickness: thick * 0.4, bevelSize: thick * 0.6, bevelSegments: 2 });
    g.rotateX(Math.PI / 2); g.translate(0, thick / 2, 0);     // shape (x, y) -> (x, z)
    return g;
  };

  // ----- airframe -----
  add(cylX(0.10, 0.12, 2.3, 10), carbon, 0, 0.74, 0);                                   // spine
  const nose = add(ell(0.62, 0.21, 0.19), paint, 0.86, 0.93, 0);                        // fairing
  add(ell(0.30, 0.10, 0.20), accent, 0.95, 0.86, 0);
  add(ell(0.42, 0.20, 0.22), paint, 0.33, 1.02, 0);                                     // tank
  add(ell(0.30, 0.03, 0.215), accent, 0.33, 1.10, 0);
  add(box(0.62, 0.09, 0.30), rubber, -0.36, 0.99, 0);                                   // seat
  add(ell(0.50, 0.13, 0.17), paint, -0.95, 1.0, 0);                                     // tail cone
  add(ell(0.2, 0.06, 0.175), accent, -1.0, 1.06, 0);
  const ws = add(ell(0.2, 0.2, 0.17), glass, 0.72, 1.12, 0); ws.scale.set(1, 1, 1); ws.rotation.z = -0.5;   // wind screen
  const head = new THREE.MeshBasicMaterial({ color: new THREE.Color(6, 5.4, 4.2), toneMapped: false });
  if (!ghost) add(ell(0.03, 0.05, 0.08), head, 1.44, 0.93, 0);                          // head lamp
  // wing with ailerons
  const wingPts = [[0.30, -1.25], [0.34, -0.2], [0.34, 0.2], [0.30, 1.25], [-0.22, 1.25], [-0.30, 0.2], [-0.30, -0.2], [-0.22, -1.25]];
  add(slab(wingPts, 0.045), paint, -0.08, 0.84, 0);
  add(box(0.5, 0.05, 0.04), accent, -0.02, 0.86, -1.26); add(box(0.5, 0.05, 0.04), accent, -0.02, 0.86, 1.26);     // tip rails
  const ail = [-1, 1].map((s) => { const g = new THREE.Group(); g.position.set(-0.32, 0.86, s * 0.86); d.add(g); add(box(0.16, 0.025, 0.62), accent, -0.08, 0, 0, g); return g; });
  // tailplanes (all-moving), fin and rudder
  const tails = [-1, 1].map((s) => { const g = new THREE.Group(); g.position.set(-1.05, 0.98, s * 0.36); d.add(g); add(slab([[0.2, -0.24], [0.2, 0.24], [-0.2, 0.24], [-0.2, -0.24]], 0.03), paint, 0, 0, 0, g); return g; });
  add(slab([[0.28, 0], [-0.1, 0.62], [-0.34, 0.62], [-0.3, 0]], 0.035), paint, -1.0, 1.0, 0).rotation.x = -Math.PI / 2;
  const rud = new THREE.Group(); rud.position.set(-1.3, 1.3, 0); d.add(rud); add(box(0.14, 0.5, 0.025), accent, -0.07, 0, 0, rud);
  // air brake petals on the tail
  const brakes = [-1, 1].map((s) => { const g = new THREE.Group(); g.position.set(-0.62, 1.04, s * 0.17); d.add(g); add(box(0.34, 0.2, 0.015), carbon, -0.17, 0, 0, g); return g; });
  // pylons to the pods
  for (const P of PODS) { const z0 = Math.sign(P.p[2]) * 0.1, len = Math.abs(P.p[2] - z0); add(box(P.p[0] > 0 ? 0.26 : 0.34, 0.035, len), carbon, P.p[0], P.p[1], (P.p[2] + z0) / 2); }
  add(box(0.2, 0.3, 0.16), carbon, 0.95, 0.83, 0); add(box(0.2, 0.3, 0.16), carbon, -1.0, 0.9, 0);
  // skids
  for (const s of [-1, 1]) {
    const rail = add(cylX(0.025, 0.025, 1.5, 8), steel, -0.02, 0.03, s * 0.42);
    add(cylX(0.03, 0.03, 0.12, 8), steel, 0.76, 0.07, s * 0.42).rotation.z = 0.7;
    for (const x of [0.5, -0.55]) { const st = add(new THREE.CylinderGeometry(0.02, 0.02, 0.76, 8), steel, x, 0.39, s * 0.27); st.rotation.x = s * 0.42; }
  }
  // handlebars and dash
  add(box(0.03, 0.03, 0.62), steel, 0.52, 1.22, 0);
  for (const s of [-1, 1]) add(cylX(0.022, 0.022, 0.13, 8), rubber, 0.50, 1.22, s * 0.29).rotation.y = Math.PI / 2;
  add(box(0.05, 0.22, 0.05), steel, 0.56, 1.12, 0).rotation.z = 0.3;
  add(box(0.02, 0.1, 0.2), new THREE.MeshBasicMaterial({ color: ghost ? 0x66ddff : 0x0a2a33 }), 0.6, 1.2, 0).rotation.z = -0.6;

  // ----- engine pods -----
  const pods = PODS.map((P) => {
    const g = new THREE.Group(); g.position.set(P.p[0], P.p[1], P.p[2]); d.add(g);
    add(cylX(0.105, 0.115, 0.5, 18), paint, 0.05, 0, 0, g);                                  // nacelle
    add(cylX(0.118, 0.105, 0.1, 18), accent, 0.33, 0, 0, g);                                 // intake lip
    add(cylX(0.06, 0.02, 0.1, 12), steel, 0.3, 0, 0, g);                                     // spinner
    add(cylX(0.115, 0.082, 0.18, 18), titan, -0.28, 0, 0, g);                                // nozzle
    const glowMat = new THREE.MeshBasicMaterial({ color: new THREE.Color(0, 0, 0), toneMapped: false });
    const glow = ghost ? null : add(new THREE.CircleGeometry(0.078, 16), glowMat, -0.365, 0, 0, g);
    if (glow) glow.rotation.y = -Math.PI / 2;
    const noz = new THREE.Group(); noz.position.set(-ENG.podLen, 0, 0); g.add(noz);          // plume attaches here, exhaust along -x
    return { g, noz, glowMat };
  });

  // ----- rider -----
  const rider = new THREE.Group(); rider.position.set(-0.36, 1.02, 0); d.add(rider);        // hips
  const torso = new THREE.Group(); rider.add(torso); torso.rotation.z = -0.62;              // leaning forward
  add(new THREE.CapsuleGeometry(0.155, 0.36, 6, 12), suit, 0, 0.3, 0, torso).scale.set(0.85, 1, 1.15);
  add(box(0.02, 0.3, 0.2), suit2, -0.13, 0.36, 0, torso);
  const headG = new THREE.Group(); headG.position.set(0.02, 0.66, 0); torso.add(headG); headG.rotation.z = 0.5;
  add(ell(0.135, 0.14, 0.125), helmet, 0, 0.06, 0, headG);
  add(ell(0.075, 0.055, 0.108), visor, 0.078, 0.065, 0, headG);
  add(box(0.2, 0.025, 0.02), accent, -0.02, 0.2, 0, headG);
  const limb = (a, b, r, mat, parent) => {     // capsule between two points (parent space)
    const va = new THREE.Vector3(...a), vb = new THREE.Vector3(...b), len = va.distanceTo(vb);
    const m = add(new THREE.CapsuleGeometry(r, Math.max(0.01, len - 2 * r), 4, 8), mat, 0, 0, 0, parent);
    m.position.copy(va).add(vb).multiplyScalar(0.5);
    m.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), vb.clone().sub(va).normalize());
    return m;
  };
  for (const s of [-1, 1]) {
    limb([-0.36, 1.02, s * 0.13], [0.1, 0.9, s * 0.22], 0.075, suit, d);        // thigh
    limb([0.1, 0.9, s * 0.22], [-0.12, 0.5, s * 0.24], 0.058, suit, d);          // shin
    add(box(0.26, 0.08, 0.1), rubber, -0.08, 0.46, s * 0.24);                   // boot on the peg
    add(cylX(0.015, 0.015, 0.14, 6), steel, -0.1, 0.42, s * 0.2).rotation.y = Math.PI / 2;
    limb([-0.04, 1.43, s * 0.2], [0.24, 1.25, s * 0.3], 0.052, suit, d);         // upper arm
    limb([0.24, 1.25, s * 0.3], [0.5, 1.23, s * 0.29], 0.044, suit2, d);         // forearm
  }

  // lights
  const lamp = (c, x, y, z) => { const mat = new THREE.MeshBasicMaterial({ color: c.clone(), toneMapped: false }); if (!ghost) add(new THREE.SphereGeometry(0.022, 8, 6), mat, x, y, z); return mat; };
  const navL = lamp(new THREE.Color(8, 0.2, 0.1), -0.02, 0.9, -1.27), navR = lamp(new THREE.Color(0.2, 8, 0.4), -0.02, 0.9, 1.27), strobe = lamp(new THREE.Color(9, 9, 9), -1.44, 1.02, 0);

  const model = {
    root, pods, design: d, all,
    update(b, time, lean = 0) {
      d.position.set(-b.mp.cg[0], -b.mp.cg[1], -b.mp.cg[2]);
      root.position.set(b.p[0], b.p[1], b.p[2]);
      root.quaternion.set(b.q[1], b.q[2], b.q[3], b.q[0]);
      for (let i = 0; i < 4; i++) {
        const p = pods[i];
        p.g.rotation.z = b.tilt;
        p.noz.rotation.set(0, -b.vb[i], b.va[i]);
        const e = b.eng[i], heat = Math.max(0, e.N - 0.3) / 0.7;
        p.glowMat.color.setRGB(heat * heat * 2.2 + e.ab * 9, heat * heat * 0.7 + e.ab * 3.5, heat * 0.12 + e.ab * 4);
      }
      ail[0].rotation.z = -b.da; ail[1].rotation.z = b.da;       // +da rolls right: right aileron up
      tails[0].rotation.z = -b.de; tails[1].rotation.z = -b.de;
      rud.rotation.y = b.dr;
      brakes[0].rotation.y = -b.brake * 1.0; brakes[1].rotation.y = b.brake * 1.0;
      // rider leans into the turn and tucks at speed
      torso.rotation.x += (lean * 0.32 - torso.rotation.x) * 0.1;
      const tuck = Math.min(1, b.airspeed / 70);
      torso.rotation.z += (-0.62 - 0.22 * tuck - torso.rotation.z) * 0.08;
      headG.rotation.z = 0.5 + 0.22 * tuck; headG.rotation.y = -lean * 0.4;
      const blink = (time * 1.1) % 1;
      strobe.color.setScalar(blink < 0.06 || (blink > 0.14 && blink < 0.2) ? 14 : 0.05);
      navL.color.setRGB(8, 0.2, 0.1); navR.color.setRGB(0.2, 8, 0.4);
    },
    // show the ghost of the best run: pose = [x,y,z,qw,qx,qy,qz]
    setPose(pose) { root.position.set(pose[0], pose[1], pose[2]); root.quaternion.set(pose[4], pose[5], pose[6], pose[3]).normalize(); d.position.set(0.01, -0.94, 0); },
  };
  return model;
}
export { SKIDS };
