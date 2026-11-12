// The F-5 and its rider, built from primitives in the design frame of the physics
// (x forward, y up, z right; origin on the ground under the bike). It is a motorcycle with the
// wheels taken away: fork, fairing, tank, seat, swing arm. Where the wheels were there are two
// hubs with a pair of lift nozzles each, where the exhaust was there is the rear nozzle, and
// between the rider's knees the fusion core. No wing, no fin. The nozzles move with the simulated
// actuators and glow with the air that really leaves them.
import * as THREE from '../../lib/three.module.min.js';
import { LIFT, REAR, ENG, INTAKE, SKIDS, massProps, FUEL0 } from '../sim/bike.js';

export function buildBike(ghost = false) {
  const root = new THREE.Group();          // origin = centre of gravity
  const d = new THREE.Group(); root.add(d);  // design frame, shifted by -cg every frame
  const M = (o) => (ghost ? new THREE.MeshBasicMaterial({ color: 0x66ddff, transparent: true, opacity: 0.22, depthWrite: false }) : o);
  const paint = M(new THREE.MeshPhysicalMaterial({ color: 0xc6cbd3, roughness: 0.3, metalness: 0.55, clearcoat: 1, clearcoatRoughness: 0.06 }));
  const blue = M(new THREE.MeshPhysicalMaterial({ color: 0x1b3a86, roughness: 0.3, metalness: 0.3, clearcoat: 1, clearcoatRoughness: 0.08 }));
  const can = M(new THREE.MeshStandardMaterial({ color: 0xb9bdc4, roughness: 0.24, metalness: 1.0 }));
  const accent = M(new THREE.MeshPhysicalMaterial({ color: 0xe8500a, roughness: 0.35, metalness: 0.1, clearcoat: 1, clearcoatRoughness: 0.1 }));
  const carbon = M(new THREE.MeshStandardMaterial({ color: 0x17181b, roughness: 0.45, metalness: 0.4 }));
  const steel = M(new THREE.MeshStandardMaterial({ color: 0x8d9299, roughness: 0.3, metalness: 0.9 }));
  const gold = M(new THREE.MeshStandardMaterial({ color: 0xb08a3c, roughness: 0.3, metalness: 1.0 }));
  const hot = M(new THREE.MeshStandardMaterial({ color: 0x4a4650, roughness: 0.4, metalness: 0.95 }));
  const dark = M(new THREE.MeshStandardMaterial({ color: 0x0b0c0e, roughness: 0.7, metalness: 0.3 }));
  const rubber = M(new THREE.MeshStandardMaterial({ color: 0x0c0c0d, roughness: 0.9 }));
  const suit = M(new THREE.MeshStandardMaterial({ color: 0x25272b, roughness: 0.7 }));
  const suit2 = M(new THREE.MeshStandardMaterial({ color: 0xe8500a, roughness: 0.7 }));
  const helmet = M(new THREE.MeshPhysicalMaterial({ color: 0x1b1c20, roughness: 0.25, clearcoat: 1 }));
  const visor = M(new THREE.MeshPhysicalMaterial({ color: 0x0a0a0c, roughness: 0.08, metalness: 1.0 }));
  const glass = M(new THREE.MeshPhysicalMaterial({ color: 0x1c2630, roughness: 0.05, transparent: true, opacity: 0.6 }));
  const all = [];
  const add = (geo, mat, x, y, z, parent = d) => { const m = new THREE.Mesh(geo, mat); m.position.set(x, y, z); m.castShadow = !ghost; m.receiveShadow = !ghost; parent.add(m); all.push(m); return m; };
  const box = (sx, sy, sz) => new THREE.BoxGeometry(sx, sy, sz);
  const cylX = (r0, r1, len, seg = 16) => { const g = new THREE.CylinderGeometry(r0, r1, len, seg); g.rotateZ(-Math.PI / 2); return g; };   // axis along +x, r0 at +x end
  const cylZ = (r, len, seg = 20) => { const g = new THREE.CylinderGeometry(r, r, len, seg); g.rotateX(Math.PI / 2); return g; };              // axis along z
  const ell = (rx, ry, rz) => { const g = new THREE.SphereGeometry(1, 20, 14); g.scale(rx, ry, rz); return g; };
  const tube = (r0, r1, len, mat) => { const g = new THREE.CylinderGeometry(r0, r1, len, 24, 1, true); g.rotateZ(-Math.PI / 2); const m = mat.clone(); m.side = THREE.DoubleSide; return [g, m]; };
  const glowMat = () => new THREE.MeshBasicMaterial({ color: new THREE.Color(0, 0, 0), toneMapped: false });
  // a beam or a limb between two points
  const beam = (a, b, r, mat, parent = d, capsule = false) => {
    const va = new THREE.Vector3(...a), vb = new THREE.Vector3(...b), len = va.distanceTo(vb);
    const m = add(capsule ? new THREE.CapsuleGeometry(r, Math.max(0.01, len - 2 * r), 4, 8) : new THREE.CylinderGeometry(r, r, len, 10), mat, 0, 0, 0, parent);
    m.position.copy(va).add(vb).multiplyScalar(0.5);
    m.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), vb.clone().sub(va).normalize());
    return m;
  };
  // bodywork: a side profile (x, y) extruded across the bike
  const prof = (pts, width, bev = 0.02) => {
    const sh = new THREE.Shape(); sh.moveTo(pts[0][0], pts[0][1]); for (let i = 1; i < pts.length; i++) sh.lineTo(pts[i][0], pts[i][1]);
    const g = new THREE.ExtrudeGeometry(sh, { depth: Math.max(0.002, width - 2 * bev), bevelEnabled: true, bevelThickness: bev, bevelSize: bev, bevelSegments: 3 });
    g.translate(0, 0, -(width - 2 * bev) / 2); return g;
  };

  const EY = ENG.axisY, HUBY = 0.34, FX = LIFT[0].p[0], RX = LIFT[2].p[0];

  // ----- the fusion core: a drum between the rider's knees, like the engine of a motorcycle -----
  add(cylZ(0.2, 0.3, 28), carbon, 0.02, EY, 0);
  for (const s of [-1, 1]) {
    add(cylZ(0.17, 0.035, 28), steel, 0.02, EY, s * 0.165);                       // end bells
    add(cylZ(0.075, 0.03, 20), gold, 0.02, EY, s * 0.19);
    for (let k = 0; k < 10; k++) { const a = k / 10 * Math.PI * 2; add(cylZ(0.012, 0.02, 6), steel, 0.02 + Math.cos(a) * 0.135, EY + Math.sin(a) * 0.135, s * 0.185); }   // bolts
  }
  for (let k = -2; k <= 2; k++) add(cylZ(0.213, 0.012, 28), steel, 0.02, EY, k * 0.05);   // cooling ribs around the field coils
  // the window into the core: a ring that shines with the reactor's power
  const coreGlow = glowMat();
  if (!ghost) for (const s of [-1, 1]) { const g = new THREE.TorusGeometry(0.115, 0.013, 8, 28); const r = add(g, coreGlow, 0.02, EY, s * 0.186); r.castShadow = false; }
  // compressor in front of it, fed by the intake in the nose
  add(cylX(0.1, 0.14, 0.4, 20), steel, 0.4, EY + 0.08, 0).rotation.z = 0.15;
  { const [g, m] = tube(0.105, 0.1, 0.2, blue); add(g, m, INTAKE[0] - 0.2, INTAKE[1] + 0.03, 0); }
  { const [g, m] = tube(0.096, 0.09, 0.2, dark); add(g, m, INTAKE[0] - 0.2, INTAKE[1] + 0.03, 0).castShadow = false; }
  const fan = new THREE.Group(); fan.position.set(INTAKE[0] - 0.2, INTAKE[1] + 0.03, 0); fan.scale.setScalar(0.72); d.add(fan);
  for (let k = 0; k < 11; k++) { const bl = add(box(0.01, 0.105, 0.045), steel, 0, 0, 0, fan); bl.geometry.translate(0, 0.066, 0); bl.rotation.x = k / 11 * Math.PI * 2; bl.rotation.y = 0.5; bl.castShadow = false; }
  { const sp = add(new THREE.ConeGeometry(0.04, 0.1, 12), accent, 0.05, 0, 0, fan); sp.rotation.z = -Math.PI / 2; }
  // exhaust: from the core, under the seat, to the rear nozzle
  // The main engine: a jet pipe the size of a fighter's, slung under the tail. A polished can,
  // then the nozzle: a ring of petals that open when the overdrive lights.
  const RY = REAR.p[1], RXN = REAR.p[0] + 0.25;                                        // the nozzle swivels about (RXN, RY)
  beam([-0.16, EY + 0.02, 0], [-0.46, RY, 0], 0.07, hot);
  add(cylX(0.07, 0.165, 0.16, 28), hot, -0.52, RY, 0);                                 // diffuser cone
  add(cylX(0.165, 0.165, -0.6 - RXN, 32), can, (RXN - 0.6) / 2, RY, 0);   // the can
  for (const x of [-0.62, (RXN - 0.6) / 2, RXN - 0.03]) add(cylX(0.172, 0.172, 0.03, 32), x === RXN - 0.03 ? dark : steel, x, RY, 0);
  for (const s of [-1, 1]) beam([-0.62, 0.93, s * 0.07], [-0.7, RY + 0.15, s * 0.07], 0.014, steel);   // hangers from the sub-frame
  const rear = new THREE.Group(); rear.position.set(RXN, RY, 0); d.add(rear);
  { const [g, m] = tube(0.15, 0.118, 0.25, dark); add(g, m, -0.125, 0, 0, rear).castShadow = false; }     // liner you look into
  const petals = [];
  for (let k = 0; k < 18; k++) {
    const ring = new THREE.Group(); ring.rotation.x = k / 18 * Math.PI * 2; rear.add(ring);
    const pv = new THREE.Group(); pv.position.set(0, 0.166, 0); ring.add(pv);
    add(box(0.26, 0.011, 0.062), hot, -0.13, 0, 0, pv).castShadow = false;
    add(box(0.05, 0.014, 0.03), steel, -0.035, 0.006, 0, pv).castShadow = false;       // actuator link
    petals.push(pv);
  }
  const rearGlow = glowMat();
  if (!ghost) {
    const g1 = add(new THREE.CircleGeometry(0.14, 24), rearGlow, 0.0, 0, 0, rear); g1.rotation.y = -Math.PI / 2; g1.castShadow = false;
    for (let k = 0; k < 4; k++) { const v = add(box(0.012, 0.28, 0.016), dark, -0.012, 0, 0, rear); v.rotation.x = k / 4 * Math.PI; v.castShadow = false; }   // flame holder
    add(new THREE.SphereGeometry(0.04, 10, 8), dark, -0.012, 0, 0, rear).castShadow = false;
  }
  const rearNoz = new THREE.Group(); rearNoz.position.set(-0.25, 0, 0); rear.add(rearNoz);      // plume attaches here, exhaust along -x
  beam([0.2, 0.42, 0], [FX - 0.05, HUBY + 0.02, 0], 0.045, hot); beam([-0.14, 0.42, 0], [RX + 0.05, HUBY + 0.02, 0], 0.045, hot);   // ducts to the two hubs

  // ----- frame, fork and swing arm -----
  for (const s of [-1, 1]) {
    beam([0.30, 0.88, s * 0.13], [-0.2, 0.62, s * 0.15], 0.026, steel);               // frame spar
    beam([-0.2, 0.62, s * 0.15], [-0.62, 0.9, s * 0.09], 0.018, steel);               // seat sub-frame
    beam([0.52, 1.03, s * 0.11], [FX, HUBY, s * 0.11], 0.026, gold);                  // fork leg
    beam([0.61, 0.76, s * 0.11], [FX, HUBY, s * 0.11], 0.033, steel);                 // fork slider
    beam([-0.2, 0.56, s * 0.13], [RX, HUBY, s * 0.13], 0.035, steel);                 // swing arm
  }
  add(cylZ(0.035, 0.3, 12), steel, 0.52, 1.03, 0);                                    // yoke
  beam([-0.3, 0.8, 0], [-0.52, 0.5, 0], 0.022, gold);                                 // rear shock
  // where the wheels were: an axle, and a lift engine at each end of it, swinging fore-aft
  for (const x of [FX, RX]) add(cylZ(0.04, 0.4, 16), steel, x, HUBY, 0);
  const lifts = LIFT.map((P) => {
    const g = new THREE.Group(); g.position.set(P.p[0], HUBY, P.p[2]); d.add(g);
    add(new THREE.CylinderGeometry(0.094, 0.094, 0.25, 20), can, 0, 0.065, 0, g);
    add(new THREE.SphereGeometry(0.094, 20, 8, 0, Math.PI * 2, 0, Math.PI / 2), can, 0, 0.19, 0, g).scale.set(1, 0.45, 1);
    for (const y of [0.15, -0.045]) add(new THREE.CylinderGeometry(0.099, 0.099, 0.022, 20), y > 0 ? accent : steel, 0, y, 0, g);
    add(new THREE.CylinderGeometry(0.1, 0.084, 0.075, 20), dark, 0, -0.088, 0, g);
    for (let k = 0; k < 12; k++) { const a = k / 12 * Math.PI * 2; add(box(0.012, 0.07, 0.03), hot, Math.cos(a) * 0.094, -0.088, Math.sin(a) * 0.094, g).rotation.y = -a; }
    const gm = glowMat();
    if (!ghost) { const gl = add(new THREE.CircleGeometry(0.076, 18), gm, 0, -0.127, 0, g); gl.rotation.x = Math.PI / 2; gl.castShadow = false; }
    const noz = new THREE.Group(); noz.position.set(0, P.p[1] - HUBY, 0); noz.rotation.z = Math.PI / 2; g.add(noz);   // local -x = straight down
    return { g, noz, glowMat: gm };
  });
  const nozzles = [...lifts.map((l) => ({ noz: l.noz, r: 0.074 })), { noz: rearNoz, r: 0.12 }];

  // ----- bodywork -----
  add(prof([[0.99, 0.93], [0.94, 1.04], [0.74, 1.15], [0.46, 1.09], [0.42, 0.92], [0.62, 0.85], [0.88, 0.85]], 0.34, 0.04), paint, 0, 0, 0);          // nose fairing
  add(prof([[0.96, 0.9], [0.88, 0.87], [0.64, 0.87], [0.5, 0.93], [0.9, 0.94]], 0.36, 0.015), accent, 0, 0, 0);                                         // its stripe
  for (const s of [-1, 1]) {                                                                                                                          // side fairings, open around the core
    add(prof([[0.66, 0.88], [0.44, 0.96], [0.26, 0.9], [0.2, 0.74], [0.3, 0.6], [0.24, 0.47], [0.5, 0.45], [0.64, 0.56], [0.72, 0.72]], 0.03, 0.012), paint, 0, 0, s * 0.17);
    add(prof([[0.64, 0.56], [0.5, 0.45], [0.24, 0.47], [-0.12, 0.38], [-0.2, 0.33], [0.5, 0.31], [0.66, 0.42]], 0.032, 0.012), blue, 0, 0, s * 0.165);
    add(prof([[0.6, 0.6], [0.34, 0.53], [0.3, 0.49], [0.56, 0.52]], 0.034, 0.008), accent, 0, 0, s * 0.172);
  }
  add(prof([[0.54, 0.4], [0.5, 0.31], [-0.22, 0.31], [-0.3, 0.38], [-0.2, 0.41]], 0.3, 0.03), blue, 0, 0, 0);                                         // belly pan
  { const ws = add(ell(0.2, 0.13, 0.145), glass, 0.66, 1.17, 0); ws.rotation.z = -0.75; }                                                              // screen
  add(prof([[0.44, 1.05], [0.3, 1.12], [0.06, 1.1], [-0.1, 0.99], [-0.14, 0.9], [0.42, 0.9]], 0.3, 0.05), paint, 0, 0, 0);                              // tank
  add(prof([[0.36, 1.1], [0.3, 1.135], [0.08, 1.115], [-0.04, 1.04]], 0.1, 0.012), accent, 0, 0, 0);
  add(prof([[-0.1, 0.96], [-0.14, 0.89], [-0.56, 0.9], [-0.58, 0.97], [-0.34, 0.94]], 0.25, 0.03), rubber, 0, 0, 0);                                    // seat
  add(prof([[-0.5, 0.9], [-0.56, 0.99], [-1.04, 1.2], [-1.07, 1.15], [-0.76, 0.97]], 0.2, 0.035), paint, 0, 0, 0);                                      // tail, kicked up over the jet pipe
  add(prof([[-0.62, 1.02], [-1.02, 1.205], [-1.04, 1.18], [-0.7, 1.0]], 0.21, 0.01), blue, 0, 0, 0);
  const head = new THREE.MeshBasicMaterial({ color: new THREE.Color(6, 5.4, 4.2), toneMapped: false });
  if (!ghost) for (const s of [-1, 1]) add(ell(0.02, 0.03, 0.05), head, 0.975, 0.97, s * 0.075);   // twin head lamps
  // bars, mirrors, dash, pegs
  for (const s of [-1, 1]) {
    beam([0.52, 1.04, s * 0.1], [0.47, 1.04, s * 0.34], 0.014, steel);
    beam([0.47, 1.04, s * 0.24], [0.47, 1.04, s * 0.35], 0.021, rubber);
    beam([0.6, 1.08, s * 0.15], [0.68, 1.2, s * 0.28], 0.008, carbon); add(ell(0.012, 0.03, 0.05), carbon, 0.68, 1.21, s * 0.29);
    beam([-0.2, 0.6, s * 0.14], [-0.16, 0.46, s * 0.23], 0.012, steel); add(cylZ(0.014, 0.1, 8), steel, -0.16, 0.46, s * 0.26);
  }
  add(box(0.02, 0.08, 0.16), new THREE.MeshBasicMaterial({ color: ghost ? 0x66ddff : 0x0a2a33 }), 0.58, 1.1, 0).rotation.z = -0.6;
  // air brake petals on the tail
  const brakes = [-1, 1].map((s) => { const g = new THREE.Group(); g.position.set(-0.56, 0.95, s * 0.115); d.add(g); add(box(0.3, 0.1, 0.012), carbon, -0.15, 0, 0, g); return g; });
  // the stand it rests on: folds up against the belly in flight
  const stand = [];
  for (const s of [-1, 1]) for (const k of [0, 2]) {
    const x = SKIDS[k][0], g = new THREE.Group(); g.position.set(x * 0.7, 0.36, s * 0.1); d.add(g);
    beam([0, 0, 0], [x * 0.3, -0.34, s * 0.2], 0.017, steel, g);
    add(box(0.16, 0.02, 0.06), rubber, x * 0.3, -0.348, s * 0.2, g);
    stand.push({ g, s, k: 0 });
  }

  // ----- rider: knees on the tank, chest over it -----
  const rider = new THREE.Group(); rider.position.set(-0.34, 1.02, 0); d.add(rider);        // hips
  const torso = new THREE.Group(); rider.add(torso); torso.rotation.z = -0.95;
  add(new THREE.CapsuleGeometry(0.155, 0.36, 6, 12), suit, 0, 0.3, 0, torso).scale.set(0.85, 1, 1.15);
  add(box(0.02, 0.3, 0.2), suit2, -0.13, 0.36, 0, torso);
  const headG = new THREE.Group(); headG.position.set(0.02, 0.66, 0); torso.add(headG); headG.rotation.z = 0.85;
  add(ell(0.135, 0.14, 0.125), helmet, 0, 0.06, 0, headG);
  add(ell(0.075, 0.055, 0.108), visor, 0.078, 0.065, 0, headG);
  add(box(0.2, 0.025, 0.02), accent, -0.02, 0.2, 0, headG);
  for (const s of [-1, 1]) {
    beam([-0.34, 1.0, s * 0.13], [0.12, 0.87, s * 0.22], 0.075, suit, d, true);       // thigh
    beam([0.12, 0.87, s * 0.22], [-0.15, 0.54, s * 0.25], 0.058, suit, d, true);      // shin
    beam([-0.24, 0.5, s * 0.26], [-0.04, 0.48, s * 0.26], 0.045, rubber, d, true);    // boot on the peg
    beam([0.06, 1.3, s * 0.2], [0.28, 1.15, s * 0.31], 0.052, suit, d, true);         // upper arm
    beam([0.28, 1.15, s * 0.31], [0.46, 1.07, s * 0.3], 0.044, suit, d, true);        // forearm
    add(new THREE.SphereGeometry(0.058, 10, 8), suit2, 0.28, 1.15, s * 0.31); add(new THREE.SphereGeometry(0.078, 10, 8), suit2, 0.12, 0.87, s * 0.22);   // elbow and knee guards
  }

  // lights
  const lamp = (c, x, y, z, r = 0.02) => { const mat = new THREE.MeshBasicMaterial({ color: c.clone(), toneMapped: false }); if (!ghost) add(new THREE.SphereGeometry(r, 8, 6), mat, x, y, z); return mat; };
  const navL = lamp(new THREE.Color(8, 0.2, 0.1), 0.47, 1.04, -0.36, 0.014), navR = lamp(new THREE.Color(0.2, 8, 0.4), 0.47, 1.04, 0.36, 0.014);
  const tailLamp = lamp(new THREE.Color(6, 0.3, 0.2), -1.07, 1.17, 0, 0.03), strobe = lamp(new THREE.Color(9, 9, 9), -0.96, 1.2, 0, 0.014);

  const cg0 = massProps(FUEL0).cg;
  const model = {
    root, nozzles, design: d, all,
    update(b, time, lean = 0, dt = 1 / 60) {
      d.position.set(-b.mp.cg[0], -b.mp.cg[1], -b.mp.cg[2]);
      root.position.set(b.p[0], b.p[1], b.p[2]);
      root.quaternion.set(b.q[1], b.q[2], b.q[3], b.q[0]);
      const e = b.eng, sweep = Math.max(b.tilt, Math.PI / 2) - Math.PI / 2;
      fan.rotation.x += e.N * 55 * dt;
      // the core shines with its power; each nozzle with the air that passes through it
      const heat = Math.max(0, e.N - 0.25) / 0.75, h2 = heat * heat, od = b.abOut;
      const flick = 0.92 + 0.08 * Math.sin(time * 47) * Math.sin(time * 13.1);
      coreGlow.color.setRGB((0.25 + 1.6 * h2 + 5 * od) * flick, (0.9 + 4.2 * h2 + 4 * od) * flick, (1.4 + 6.5 * h2 + 6 * od) * flick);
      for (let i = 0; i < 4; i++) {
        const l = lifts[i], f = Math.min(1, b.L[i] / 1600);
        l.g.rotation.z = sweep + b.va[i];
        l.glowMat.color.setRGB(h2 * 1.2 * f, h2 * 1.9 * f, h2 * 3.2 * f);
      }
      rear.rotation.set(0, -b.ry, REAR.cant + b.rp);
      const fr = Math.min(1, b.R / 5000);
      for (const pv of petals) pv.rotation.z = 0.19 - 0.13 * od;                 // the nozzle opens for the overdrive
      rearGlow.color.setRGB(h2 * 1.2 * fr + od * 7, h2 * 1.9 * fr + od * 6, h2 * 3.2 * fr + od * 11);
      brakes[0].rotation.y = -b.brake * 1.0; brakes[1].rotation.y = b.brake * 1.0;
      const down = b.agl < 2.5 ? 1 : 0;
      for (const st of stand) { st.k += (down - st.k) * Math.min(1, dt * 4); st.g.rotation.x = st.s * (1 - st.k) * 1.25; st.g.scale.setScalar(0.35 + 0.65 * st.k); }
      tailLamp.color.setRGB(b.brake > 0.3 ? 16 : 3, 0.3, 0.2);
      // rider leans into the turn and tucks at speed
      torso.rotation.x += (lean * 0.32 - torso.rotation.x) * 0.1;
      const tuck = Math.min(1, b.airspeed / 70);
      torso.rotation.z += (-0.95 - 0.12 * tuck - torso.rotation.z) * 0.08;
      headG.rotation.z = 0.85 + 0.12 * tuck; headG.rotation.y = -lean * 0.4;
      const blink = (time * 1.1) % 1;
      strobe.color.setScalar(blink < 0.06 || (blink > 0.14 && blink < 0.2) ? 14 : 0.05);
      navL.color.setRGB(8, 0.2, 0.1); navR.color.setRGB(0.2, 8, 0.4);
    },
    // show the ghost of the best run: pose = [x,y,z,qw,qx,qy,qz]
    setPose(pose) { root.position.set(pose[0], pose[1], pose[2]); root.quaternion.set(pose[4], pose[5], pose[6], pose[3]).normalize(); d.position.set(-cg0[0], -cg0[1], -cg0[2]); },
  };
  return model;
}
export { SKIDS };
