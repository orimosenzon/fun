// The F-4 and its rider, built from primitives in the design frame of the physics
// (x forward, y up, z right; origin on the ground under the bike). One big engine with the rider
// astride it; the rear nozzle, the four lift pipes, elevator, ailerons, rudder and air brake move
// with the simulated actuators, every nozzle glows with the gas that really leaves it.
import * as THREE from '../../lib/three.module.min.js';
import { LIFT, REAR, ENG, INTAKE, SKIDS } from '../sim/bike.js';

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

  // ----- the engine -----
  const EY = ENG.axisY, ER = ENG.radius;
  const hot = M(new THREE.MeshStandardMaterial({ color: 0x5a4a40, roughness: 0.42, metalness: 0.95 }));
  const dark = M(new THREE.MeshStandardMaterial({ color: 0x0b0c0e, roughness: 0.7, metalness: 0.3 }));
  const tube = (r0, r1, len, mat) => { const g = new THREE.CylinderGeometry(r0, r1, len, 28, 1, true); g.rotateZ(-Math.PI / 2); const m = mat.clone(); m.side = THREE.DoubleSide; return [g, m]; };
  { const [g, m] = tube(ER + 0.03, ER + 0.008, 0.2, accent); add(g, m, INTAKE[0] - 0.08, EY, 0); }   // intake lip, open: you look at the fan
  { const [g, m] = tube(ER - 0.005, ER - 0.005, 0.2, dark); add(g, m, INTAKE[0] - 0.08, EY, 0).castShadow = false; }
  add(cylX(ER + 0.008, ER + 0.012, 0.72, 28), paint, 0.82, EY, 0);                          // fan cowl
  add(cylX(ER + 0.012, ER + 0.012, 0.62, 28), carbon, 0.15, EY, 0);                         // compressor / diverter casing
  add(cylX(ER + 0.012, ER - 0.02, 0.56, 28), steel, -0.44, EY, 0);                          // turbine casing
  add(cylX(ER - 0.02, ER - 0.045, 0.52, 28), hot, -0.98, EY, 0);                            // jet pipe, after-burner
  for (const x of [0.46, -0.16, -0.72, -1.22]) add(cylX(ER + 0.02, ER + 0.02, 0.035, 28), x < -0.5 ? hot : steel, x, EY, 0).scale.set(1, x < -0.5 ? 0.93 : 1.02, x < -0.5 ? 0.93 : 1.02);   // flanges
  if (!ghost) {
    const fanD = add(new THREE.CircleGeometry(ER - 0.01, 28), dark, 1.185, EY, 0); fanD.rotation.y = Math.PI / 2;
  }
  // fan: blades and spinner turn with the spool
  const fan = new THREE.Group(); fan.position.set(1.2, EY, 0); d.add(fan);
  for (let k = 0; k < 14; k++) { const bl = add(box(0.012, ER - 0.03, 0.07), steel, 0, 0, 0, fan); bl.geometry.translate(0, (ER - 0.03) / 2 + 0.02, 0); bl.rotation.x = k / 14 * Math.PI * 2; bl.rotation.y = 0.5; bl.castShadow = false; }
  { const sp = add(new THREE.ConeGeometry(0.085, 0.2, 16), accent, 0.1, 0, 0, fan); sp.rotation.z = -Math.PI / 2; }
  // accessory gearbox and plumbing under the belly, stripes
  add(box(0.5, 0.07, 0.22), steel, 0.12, EY - ER - 0.02, 0);
  for (const s of [-1, 1]) { add(cylX(0.014, 0.014, 1.5, 6), steel, -0.2, EY + 0.1, s * (ER + 0.012)); add(cylX(0.011, 0.011, 1.1, 6), accent, -0.1, EY - 0.12, s * (ER + 0.005)); }
  add(cylX(ER + 0.008, ER + 0.014, 0.07, 28), accent, 0.50, EY, 0);
  // rear nozzle: convergent petals on a two-axis swivel
  const rear = new THREE.Group(); rear.position.set(-1.24, EY, 0); d.add(rear);
  { const [g, m] = tube(ER - 0.04, ER - 0.075, 0.34, hot); add(g, m, -0.15, 0, 0, rear); }                        // nozzle shell, open at the back
  { const [g, m] = tube(ER - 0.07, ER - 0.1, 0.5, dark); add(g, m, -0.02, 0, 0, rear).castShadow = false; }        // liner you look into
  for (let k = 0; k < 16; k++) { const pt = add(box(0.3, 0.012, 0.075), titan, -0.17, 0, 0, rear); pt.geometry.translate(0, ER - 0.05, 0); pt.geometry.rotateZ(0.1); pt.rotation.x = k / 16 * Math.PI * 2; pt.castShadow = false; }
  const rearGlow = new THREE.MeshBasicMaterial({ color: new THREE.Color(0, 0, 0), toneMapped: false });
  if (!ghost) {
    const g1 = add(new THREE.CircleGeometry(ER - 0.09, 24), rearGlow, 0.18, 0, 0, rear); g1.rotation.y = -Math.PI / 2; g1.castShadow = false;   // flame holder, deep inside
    for (let k = 0; k < 3; k++) { const v = add(box(0.01, 2 * (ER - 0.1), 0.02), dark, 0.16, 0, 0, rear); v.rotation.x = k / 3 * Math.PI; v.castShadow = false; }
  }
  const rearNoz = new THREE.Group(); rearNoz.position.set(-0.3, 0, 0); rear.add(rearNoz);      // plume attaches here, exhaust along -x
  // lift pipes: an elbow out of the engine's flank, a short nozzle that swings fore-aft
  const lifts = LIFT.map((P) => {
    const sgn = Math.sign(P.p[2]);
    add(cylX(0.1, 0.1, 0.2, 14), steel, P.p[0], EY - 0.12, sgn * (ER - 0.02)).rotation.y = Math.PI / 2;      // stub out of the casing
    add(new THREE.SphereGeometry(0.105, 14, 10), steel, P.p[0], EY - 0.12, P.p[2]);                          // elbow
    const g = new THREE.Group(); g.position.set(P.p[0], EY - 0.12, P.p[2]); d.add(g);
    add(new THREE.CylinderGeometry(0.1, 0.082, 0.2, 16), hot, 0, -0.1, 0, g);
    add(new THREE.CylinderGeometry(0.088, 0.088, 0.025, 16), accent, 0, -0.03, 0, g);
    const glowMat = new THREE.MeshBasicMaterial({ color: new THREE.Color(0, 0, 0), toneMapped: false });
    if (!ghost) { const gl = add(new THREE.CircleGeometry(0.076, 16), glowMat, 0, -0.203, 0, g); gl.rotation.x = Math.PI / 2; gl.castShadow = false; }
    const noz = new THREE.Group(); noz.position.set(0, P.p[1] - (EY - 0.12), 0); noz.rotation.z = Math.PI / 2; g.add(noz);   // local -x = straight down
    return { g, noz, glowMat };
  });
  const nozzles = [...lifts.map((l) => ({ noz: l.noz, r: 0.076 })), { noz: rearNoz, r: ER - 0.09 }];

  // ----- airframe: a saddle, a tank and flying surfaces bolted to the engine -----
  add(ell(0.50, 0.15, 0.20), paint, 0.86, EY + ER + 0.03, 0);                           // nose fairing over the fan cowl
  add(ell(0.26, 0.07, 0.205), accent, 0.98, EY + ER + 0.0, 0);
  add(ell(0.44, 0.17, 0.25), paint, 0.30, EY + ER + 0.05, 0);                           // saddle tank
  add(ell(0.30, 0.03, 0.215), accent, 0.30, EY + ER + 0.16, 0);
  add(box(0.62, 0.09, 0.30), rubber, -0.36, 0.99, 0);                                   // seat
  add(ell(0.34, 0.07, 0.19), carbon, -0.42, 0.955, 0);
  add(ell(0.42, 0.08, 0.15), paint, -0.9, EY + ER - 0.02, 0);                           // fin root fairing
  const ws = add(ell(0.2, 0.2, 0.17), glass, 0.72, 1.12, 0); ws.rotation.z = -0.5;      // wind screen
  const head = new THREE.MeshBasicMaterial({ color: new THREE.Color(6, 5.4, 4.2), toneMapped: false });
  if (!ghost) add(ell(0.03, 0.04, 0.07), head, 1.33, EY + ER + 0.04, 0);                // head lamp
  // wing with ailerons
  const wingPts = [[0.10, -1.5], [0.62, -0.26], [0.62, 0.26], [0.10, 1.5], [-0.36, 1.5], [-0.46, 0.26], [-0.46, -0.26], [-0.36, -1.5]];   // tapered, swept leading edge (same area as the physics)
  add(slab(wingPts, 0.045), paint, -0.08, 0.84, 0);
  add(box(0.5, 0.05, 0.04), accent, -0.21, 0.86, -1.51); add(box(0.5, 0.05, 0.04), accent, -0.21, 0.86, 1.51);     // tip rails
  const ail = [-1, 1].map((s) => { const g = new THREE.Group(); g.position.set(-0.4, 0.86, s * 1.04); d.add(g); add(box(0.16, 0.025, 0.78), accent, -0.08, 0, 0, g); return g; });
  // intake strakes, tailplanes (all-moving), fin and rudder
  for (const s of [-1, 1]) add(slab([[0.2, 0], [0.14, s * 0.22], [-0.12, s * 0.22], [-0.2, 0]], 0.025), accent, 0.95, 0.72, s * (ER - 0.02));
  const tails = [-1, 1].map((s) => { const g = new THREE.Group(); g.position.set(-1.05, 0.98, s * 0.36); d.add(g); add(slab([[0.2, -0.24], [0.2, 0.24], [-0.2, 0.24], [-0.2, -0.24]], 0.03), paint, 0, 0, 0, g); return g; });
  add(slab([[0.28, 0], [-0.1, 0.68], [-0.34, 0.68], [-0.3, 0]], 0.035), paint, -1.0, 0.94, 0).rotation.x = -Math.PI / 2;
  const rud = new THREE.Group(); rud.position.set(-1.3, 1.3, 0); d.add(rud); add(box(0.14, 0.5, 0.025), accent, -0.07, 0, 0, rud);
  // air brake petals behind the seat
  const brakes = [-1, 1].map((s) => { const g = new THREE.Group(); g.position.set(-0.62, 1.02, s * 0.19); d.add(g); add(box(0.34, 0.18, 0.015), carbon, -0.17, 0, 0, g); return g; });
  // skids
  for (const s of [-1, 1]) {
    add(cylX(0.025, 0.025, 1.5, 8), steel, -0.02, 0.03, s * 0.42);
    add(cylX(0.03, 0.03, 0.12, 8), steel, 0.76, 0.07, s * 0.42).rotation.z = 0.7;
    for (const x of [0.5, -0.55]) { const st = add(new THREE.CylinderGeometry(0.02, 0.02, 0.56, 8), steel, x, 0.28, s * 0.33); st.rotation.x = s * 0.33; }
  }
  // handlebars and dash
  add(box(0.03, 0.03, 0.62), steel, 0.52, 1.22, 0);
  for (const s of [-1, 1]) add(cylX(0.022, 0.022, 0.13, 8), rubber, 0.50, 1.22, s * 0.29).rotation.y = Math.PI / 2;
  add(box(0.05, 0.22, 0.05), steel, 0.56, 1.12, 0).rotation.z = 0.3;
  add(box(0.02, 0.1, 0.2), new THREE.MeshBasicMaterial({ color: ghost ? 0x66ddff : 0x0a2a33 }), 0.6, 1.2, 0).rotation.z = -0.6;

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
    limb([-0.36, 1.04, s * 0.15], [0.1, 0.95, s * 0.31], 0.075, suit, d);       // thigh
    limb([0.1, 0.95, s * 0.31], [-0.12, 0.58, s * 0.345], 0.058, suit, d);      // shin
    add(box(0.26, 0.08, 0.1), rubber, -0.08, 0.53, s * 0.35);                   // boot on the peg
    add(cylX(0.015, 0.015, 0.2, 6), steel, -0.1, 0.49, s * 0.31).rotation.y = Math.PI / 2;
    limb([-0.04, 1.43, s * 0.2], [0.24, 1.25, s * 0.3], 0.052, suit, d);         // upper arm
    limb([0.24, 1.25, s * 0.3], [0.5, 1.23, s * 0.29], 0.044, suit2, d);         // forearm
  }

  // lights
  const lamp = (c, x, y, z) => { const mat = new THREE.MeshBasicMaterial({ color: c.clone(), toneMapped: false }); if (!ghost) add(new THREE.SphereGeometry(0.022, 8, 6), mat, x, y, z); return mat; };
  const navL = lamp(new THREE.Color(8, 0.2, 0.1), -0.02, 0.9, -1.27), navR = lamp(new THREE.Color(0.2, 8, 0.4), -0.02, 0.9, 1.27), strobe = lamp(new THREE.Color(9, 9, 9), -1.44, 1.02, 0);

  const model = {
    root, nozzles, design: d, all,
    update(b, time, lean = 0, dt = 1 / 60) {
      d.position.set(-b.mp.cg[0], -b.mp.cg[1], -b.mp.cg[2]);
      root.position.set(b.p[0], b.p[1], b.p[2]);
      root.quaternion.set(b.q[1], b.q[2], b.q[3], b.q[0]);
      const e = b.eng, sweep = Math.max(b.tilt, Math.PI / 2) - Math.PI / 2;
      fan.rotation.x += e.N * 55 * dt;
      // metal glows with the temperature of the gas and with how much of it passes
      const heat = Math.max(0, e.N - 0.3) / 0.7, h2 = heat * heat;
      for (let i = 0; i < 4; i++) {
        const l = lifts[i], f = Math.min(1, b.L[i] / 1100);
        l.g.rotation.z = sweep + b.va[i];
        l.glowMat.color.setRGB(h2 * 2.6 * f, h2 * 0.8 * f, heat * 0.12 * f);
      }
      rear.rotation.set(0, -b.ry, REAR.cant + b.rp);
      const fr = Math.min(1, b.R / 2500), ab = b.abOut;
      rearGlow.color.setRGB(h2 * 2.4 * fr + ab * 10, h2 * 0.75 * fr + ab * 4, heat * 0.12 * fr + ab * 4.5);
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
    setPose(pose) { root.position.set(pose[0], pose[1], pose[2]); root.quaternion.set(pose[4], pose[5], pose[6], pose[3]).normalize(); d.position.set(0.06, -0.87, 0); },
  };
  return model;
}
export { SKIDS };
