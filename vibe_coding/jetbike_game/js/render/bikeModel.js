// Visual model of the MJ-5 jet motorcycle and its rider, built in the design frame of
// vehicle.js (x forward, y up, z right). Moving parts are driven by the recorded state:
// nozzle vanes follow the flight computer's commands, compressor faces spin at the recorded
// rpm, nozzle interiors glow with the exhaust gas temperature, the rider shifts with the
// felt acceleration and looks into turns.

import * as THREE from 'three';
import { RoundedBoxGeometry } from 'three/addons/geometries/RoundedBoxGeometry.js';
import { LIFT, MAIN, PADS, massProps, FUEL0, MAIN_CANT, MAIN_BEND, MAIN_DIR } from '../vehicle.js';

const V = (x, y, z) => new THREE.Vector3(x, y, z);

function tube(a, b, r, mat, seg = 12) {
  const d = new THREE.Vector3().subVectors(b, a);
  const m = new THREE.Mesh(new THREE.CylinderGeometry(r, r, d.length(), seg), mat);
  m.position.copy(a).addScaledVector(d, 0.5);
  m.quaternion.setFromUnitVectors(V(0, 1, 0), d.normalize());
  return m;
}
function capsuleBetween(a, b, r, mat) {
  const d = new THREE.Vector3().subVectors(b, a);
  const m = new THREE.Mesh(new THREE.CapsuleGeometry(r, Math.max(0.001, d.length()), 6, 14), mat);
  m.position.copy(a).addScaledVector(d, 0.5);
  m.quaternion.setFromUnitVectors(V(0, 1, 0), d.normalize());
  return m;
}
function setBetween(m, a, b) {
  const d = new THREE.Vector3().subVectors(b, a);
  m.position.copy(a).addScaledVector(d, 0.5);
  m.quaternion.setFromUnitVectors(V(0, 1, 0), d.clone().normalize());
  const L = d.length();
  m.scale.set(1, L / m.userData.len, 1);
}

// blackbody-ish glow colour for a metal / gas temperature (K), linear HDR
export function glowColor(Tk, out = new THREE.Color()) {
  const t = Math.max(0, (Tk - 650) / 700);
  out.setRGB(1.0, 0.28 + 0.5 * t, 0.06 + 0.45 * t * t);
  return out.multiplyScalar(Math.pow(t, 2.2) * 9);
}

function twoBoneIK(root, target, l1, l2, pole) {
  const d = new THREE.Vector3().subVectors(target, root);
  const L = Math.min(d.length(), l1 + l2 - 1e-4);
  const dir = d.clone().normalize();
  const a = (l1 * l1 - l2 * l2 + L * L) / (2 * L);
  const h = Math.sqrt(Math.max(0, l1 * l1 - a * a));
  const pl = pole.clone().sub(dir.clone().multiplyScalar(pole.dot(dir))).normalize();
  return root.clone().addScaledVector(dir, a).addScaledVector(pl, h);
}

export function buildBike(envTex) {
  const cg = massProps(FUEL0).cg;
  const root = new THREE.Group();          // placed at the CG in world
  const body = new THREE.Group();          // design frame
  root.add(body);
  body.position.set(-cg[0], -cg[1], -cg[2]);

  const M = {
    paint: new THREE.MeshPhysicalMaterial({ color: 0x5e120c, metalness: 0.55, roughness: 0.3, clearcoat: 1, clearcoatRoughness: 0.06 }),
    paint2: new THREE.MeshPhysicalMaterial({ color: 0x1b1d20, metalness: 0.4, roughness: 0.45, clearcoat: 0.6 }),
    carbon: new THREE.MeshStandardMaterial({ color: 0x15161a, metalness: 0.3, roughness: 0.5 }),
    ti: new THREE.MeshStandardMaterial({ color: 0x6c6964, metalness: 1, roughness: 0.38 }),
    tiDark: new THREE.MeshStandardMaterial({ color: 0x55504a, metalness: 1, roughness: 0.42 }),
    heat: new THREE.MeshStandardMaterial({ color: 0x6d5a48, metalness: 1, roughness: 0.38 }),
    steel: new THREE.MeshStandardMaterial({ color: 0xc8c8c8, metalness: 1, roughness: 0.2 }),
    black: new THREE.MeshStandardMaterial({ color: 0x0c0c0d, metalness: 0.2, roughness: 0.6 }),
    rubber: new THREE.MeshStandardMaterial({ color: 0x111111, roughness: 0.9 }),
    leather: new THREE.MeshStandardMaterial({ color: 0x171514, roughness: 0.7 }),
    hole: new THREE.MeshBasicMaterial({ color: 0x020202 }),
    glass: new THREE.MeshPhysicalMaterial({ color: 0x9fb4c0, metalness: 0, roughness: 0.03, transmission: 0.0, transparent: true, opacity: 0.28, clearcoat: 1 }),
    lamp: new THREE.MeshBasicMaterial({ color: new THREE.Color(6, 6, 5.5) }),
    tail: new THREE.MeshBasicMaterial({ color: new THREE.Color(4, 0.1, 0.05) }),
    screen: new THREE.MeshBasicMaterial({ color: new THREE.Color(0.25, 0.8, 1.1) }),
    orange: new THREE.MeshStandardMaterial({ color: 0xd85a10, roughness: 0.5 }),
  };
  const add = (m, cast = true) => { m.castShadow = cast; m.receiveShadow = true; body.add(m); return m; };

  // ---- frame and bodywork
  add(tube(V(0.72, 1.00, 0), V(-0.62, 0.86, 0), 0.045, M.tiDark));            // spine
  add(tube(V(0.72, 1.00, 0), V(0.62, 0.62, 0), 0.035, M.tiDark));             // down tube
  add(tube(V(-0.62, 0.86, 0), V(-0.70, 0.62, 0), 0.03, M.tiDark));
  // streamlined hull: a lathe profile laid along x, elliptical section, following a centre line
  {
    const prof = [[1.14, 0.0], [1.10, 0.045], [1.02, 0.085], [0.90, 0.12], [0.72, 0.15], [0.52, 0.168], [0.34, 0.165],
      [0.16, 0.145], [0.0, 0.12], [-0.25, 0.115], [-0.55, 0.112], [-0.80, 0.095], [-0.95, 0.07], [-1.02, 0.035], [-1.04, 0.0]];
    const pts = prof.map(([x, r]) => new THREE.Vector2(r, x));
    const g = new THREE.LatheGeometry(pts, 40);
    const P = g.attributes.position;
    const cy = (x) => x > 0.2 ? 1.0 + 0.06 * Math.exp(-(((x - 0.45) / 0.25) ** 2)) : 0.96 + 0.05 * Math.max(0, -x - 0.6) * 4;
    for (let i = 0; i < P.count; i++) {
      const rx = P.getX(i), ax = P.getY(i), rz = P.getZ(i); // lathe: radius in x/z, axis in y
      const y = rx * 0.82, z = rz;
      P.setXYZ(i, ax, cy(ax) + y, z);
    }
    g.computeVertexNormals();
    const hull = add(new THREE.Mesh(g, M.paint));
    hull.name = 'hull';
    // dark lower belly pan
    const belly = add(new THREE.Mesh(new RoundedBoxGeometry(1.5, 0.08, 0.2, 4, 0.035), M.paint2));
    belly.position.set(0.05, 0.84, 0);
    const cap = add(new THREE.Mesh(new THREE.CylinderGeometry(0.035, 0.035, 0.02, 20), M.steel)); cap.position.set(0.44, 1.21, 0);
  }
  // side fairings over the frame
  for (const s of [-1, 1]) {
    const f2 = add(new THREE.Mesh(new RoundedBoxGeometry(0.5, 0.1, 0.025, 4, 0.01), M.carbon));
    f2.position.set(0.28, 0.86, s * 0.155); f2.rotation.set(0, 0, -0.05);
  }
  // seat and tail
  const seat = add(new THREE.Mesh(new RoundedBoxGeometry(0.60, 0.07, 0.26, 5, 0.03), M.leather));
  seat.position.set(-0.32, 1.075, 0); seat.rotation.z = 0.03;
  const tl = add(new THREE.Mesh(new THREE.BoxGeometry(0.02, 0.025, 0.1), M.tail), false); tl.position.set(-1.0, 1.0, 0);
  // nose, headlight, screen
  for (const s of [-1, 1]) { const l2 = add(new THREE.Mesh(new THREE.BoxGeometry(0.012, 0.016, 0.08), M.lamp), false); l2.position.set(1.085, 1.0, s * 0.04); l2.rotation.y = s * 0.5; }
  const scr = new THREE.Mesh(new THREE.SphereGeometry(1, 24, 12, -Math.PI / 2, Math.PI, 0, Math.PI / 2), M.glass);
  scr.scale.set(0.2, 0.2, 0.15); scr.position.set(0.84, 1.10, 0); scr.rotation.z = -0.6;
  body.add(scr);
  // handlebars
  add(tube(V(0.64, 1.14, -0.34), V(0.64, 1.14, 0.34), 0.014, M.steel));
  add(tube(V(0.70, 1.03, 0), V(0.64, 1.14, 0), 0.03, M.tiDark));
  for (const s of [-1, 1]) {
    add(tube(V(0.64, 1.14, s * 0.26), V(0.64, 1.14, s * 0.36), 0.019, M.rubber));
    add(tube(V(0.64, 1.14, s * 0.22), V(0.70, 1.13, s * 0.33), 0.006, M.steel));      // lever
    add(tube(V(0.66, 1.16, s * 0.24), V(0.70, 1.34, s * 0.29), 0.006, M.black));      // mirror stalk
    const mir = add(new THREE.Mesh(new THREE.SphereGeometry(1, 16, 10), M.black)); mir.scale.set(0.012, 0.035, 0.055); mir.position.set(0.70, 1.35, s * 0.30);
  }
  const disp = add(new THREE.Mesh(new THREE.BoxGeometry(0.01, 0.06, 0.1), M.screen), false);
  disp.position.set(0.66, 1.18, 0); disp.rotation.z = 0.5;

  // ---- lift engines
  const lift = [];
  for (const L of LIFT) {
    const g = new THREE.Group();
    g.position.set(L.c[0], 0, L.c[2]);
    body.add(g);
    const casing = new THREE.Mesh(new THREE.CylinderGeometry(0.118, 0.112, 0.46, 36), M.ti);
    casing.position.y = 0.66;
    const band1 = new THREE.Mesh(new THREE.CylinderGeometry(0.124, 0.124, 0.03, 36), M.tiDark); band1.position.y = 0.80;
    const band2 = band1.clone(); band2.position.y = 0.55;
    const hot = new THREE.Mesh(new THREE.CylinderGeometry(0.112, 0.10, 0.12, 36), M.heat); hot.position.y = 0.37;
    // bellmouth intake and compressor face
    const lip = new THREE.Mesh(new THREE.TorusGeometry(0.118, 0.022, 12, 36), M.steel); lip.rotation.x = Math.PI / 2; lip.position.y = 0.905;
    const intake = new THREE.Mesh(new THREE.CircleGeometry(0.115, 32), M.hole); intake.rotation.x = -Math.PI / 2; intake.position.y = 0.87;
    const fan = new THREE.Group(); fan.position.y = 0.875;
    const spinner = new THREE.Mesh(new THREE.ConeGeometry(0.035, 0.05, 20), M.steel); spinner.position.y = 0.02; fan.add(spinner);
    for (let k = 0; k < 11; k++) {
      const bl = new THREE.Mesh(new THREE.BoxGeometry(0.08, 0.004, 0.022), M.tiDark);
      bl.position.set(Math.cos(k / 11 * 6.283) * 0.07, 0, Math.sin(k / 11 * 6.283) * 0.07);
      bl.rotation.set(0.5, -k / 11 * 6.283, 0);
      fan.add(bl);
    }
    // guard mesh over intake
    const guard = new THREE.Mesh(new THREE.TorusGeometry(0.06, 0.004, 6, 24), M.black); guard.rotation.x = Math.PI / 2; guard.position.y = 0.915;
    // nozzle (converging), glowing interior and two vane sets
    const noz = new THREE.Mesh(new THREE.CylinderGeometry(0.10, 0.072, 0.10, 32, 1, true), M.heat); noz.position.y = 0.26;
    noz.material = M.heat;
    const glowMat = new THREE.MeshBasicMaterial({ color: 0x000000 });
    const glow = new THREE.Mesh(new THREE.CircleGeometry(0.068, 24), glowMat); glow.rotation.x = Math.PI / 2; glow.position.y = 0.3;
    const vaneA = new THREE.Group(); vaneA.position.y = 0.215;
    const vaneB = new THREE.Group(); vaneB.position.y = 0.195;
    for (const zz of [-0.03, 0.03]) { const v = new THREE.Mesh(new THREE.BoxGeometry(0.004, 0.06, 0.13), M.tiDark); v.position.set(0, -0.02, zz * 0 + 0); v.position.x = zz; vaneA.add(v); }
    for (const xx of [-0.03, 0.03]) { const v = new THREE.Mesh(new THREE.BoxGeometry(0.13, 0.06, 0.004), M.tiDark); v.position.set(0, -0.02, xx); vaneB.add(v); }
    const accent = new THREE.Mesh(new THREE.CylinderGeometry(0.121, 0.121, 0.05, 36), M.paint); accent.position.y = 0.70;
    [casing, band1, band2, hot, lip, intake, fan, guard, noz, glow, vaneA, vaneB, accent].forEach((m) => { m.castShadow = true; g.add(m); });
    // strut to the frame
    const sgn = Math.sign(L.c[2]);
    add(tube(V(L.c[0], 0.78, L.c[2] - sgn * 0.11), V(L.c[0] * 0.8, 0.86, 0), 0.022, M.tiDark));
    add(tube(V(L.c[0], 0.50, L.c[2] - sgn * 0.11), V(L.c[0] * 0.8, 0.66, 0), 0.02, M.tiDark));
    // fuel line
    add(tube(V(L.c[0] - 0.05, 0.75, L.c[2] - sgn * 0.1), V(L.c[0] * 0.8, 0.82, sgn * 0.05), 0.005, M.black));
    lift.push({ g, fan, glow, glowMat, vaneA, vaneB });
  }

  // ---- cruise engine under the seat
  const main = {};
  {
    const g = new THREE.Group(); body.add(g);
    const len = MAIN.intake[0] - MAIN.exit[0];
    const cas = new THREE.Mesh(new THREE.CylinderGeometry(0.13, 0.13, 0.62, 36), M.ti);
    cas.rotation.z = Math.PI / 2; cas.position.set(0.02, MAIN.c[1], 0);
    const rear = new THREE.Mesh(new THREE.CylinderGeometry(0.13, 0.115, 0.28, 36), M.heat);
    rear.rotation.z = Math.PI / 2; rear.position.set(-0.43, MAIN.c[1], 0);
    // tail pipe bent down so the thrust line passes through the CG
    const bend = new THREE.Group(); bend.position.set(...MAIN_BEND); bend.rotation.z = MAIN_CANT; g.add(bend);
    const elbow = new THREE.Mesh(new THREE.SphereGeometry(0.118, 24, 16), M.heat); bend.add(elbow);
    const pipe = new THREE.Mesh(new THREE.CylinderGeometry(0.113, 0.085, 0.55, 36, 1, true), M.heat);
    pipe.rotation.z = Math.PI / 2; pipe.position.set(-0.275, 0, 0); pipe.castShadow = true; bend.add(pipe);
    const lipM = new THREE.Mesh(new THREE.TorusGeometry(0.128, 0.02, 12, 36), M.steel); lipM.rotation.y = Math.PI / 2; lipM.position.set(0.345, MAIN.c[1], 0);
    const hole = new THREE.Mesh(new THREE.CircleGeometry(0.125, 32), M.hole); hole.rotation.y = Math.PI / 2; hole.position.set(0.32, MAIN.c[1], 0);
    const fan = new THREE.Group(); fan.position.set(0.325, MAIN.c[1], 0);
    const sp = new THREE.Mesh(new THREE.ConeGeometry(0.04, 0.06, 20), M.steel); sp.rotation.z = -Math.PI / 2; sp.position.x = 0.02; fan.add(sp);
    for (let k = 0; k < 11; k++) {
      const bl = new THREE.Mesh(new THREE.BoxGeometry(0.004, 0.085, 0.024), M.tiDark);
      bl.position.set(0, Math.cos(k / 11 * 6.283) * 0.075, Math.sin(k / 11 * 6.283) * 0.075);
      bl.rotation.set(k / 11 * 6.283, 0, 0.5);
      fan.add(bl);
    }
    const glowMat = new THREE.MeshBasicMaterial({ color: 0 });
    const glow = new THREE.Mesh(new THREE.CircleGeometry(0.083, 24), glowMat); glow.rotation.y = Math.PI / 2; glow.position.set(-0.5, 0, 0); bend.add(glow);
    for (let b = 0; b < 3; b++) { const bd = new THREE.Mesh(new THREE.CylinderGeometry(0.136, 0.136, 0.02, 36), M.tiDark); bd.rotation.z = Math.PI / 2; bd.position.set(0.25 - b * 0.22, MAIN.c[1], 0); g.add(bd); }
    // heat shields protecting the rider's legs
    for (const s of [-1, 1]) {
      const hs = new THREE.Mesh(new THREE.CylinderGeometry(0.17, 0.17, 0.5, 24, 1, true, s > 0 ? 0.2 : Math.PI + 0.2, Math.PI - 0.4), M.carbon);
      hs.rotation.z = Math.PI / 2; hs.position.set(-0.05, MAIN.c[1] + 0.02, 0);
      hs.material = M.carbon; hs.castShadow = true; g.add(hs);
    }
    [cas, rear, lipM, hole, fan].forEach((m) => { m.castShadow = true; g.add(m); });
    Object.assign(main, { fan, glow, glowMat, len });
  }

  // ---- landing legs
  for (const p of PADS) {
    const top = V(p[0] * 0.7, 0.62, p[2] * 0.45);
    const foot = V(p[0], 0.04, p[2]);
    add(tube(top, foot, 0.022, M.tiDark));
    add(tube(V(foot.x, 0.30, foot.z), foot, 0.03, M.steel));
    const pad = add(new THREE.Mesh(new THREE.CylinderGeometry(0.07, 0.08, 0.035, 20), M.rubber));
    pad.position.set(p[0], 0.018, p[2]);
  }
  // footpegs
  for (const s of [-1, 1]) add(tube(V(-0.02, 0.70, s * 0.16), V(-0.02, 0.70, s * 0.27), 0.013, M.steel));

  // ---- rider: tapered limbs with joint caps, lathe-shaped torso, full-face helmet
  const suit = new THREE.MeshStandardMaterial({ color: 0x23252b, roughness: 0.58, metalness: 0.08 });
  const suitHi = new THREE.MeshStandardMaterial({ color: 0x3a3d45, roughness: 0.5, metalness: 0.1 });
  const suit2 = new THREE.MeshStandardMaterial({ color: 0xc24a12, roughness: 0.5 });
  const glove = new THREE.MeshStandardMaterial({ color: 0x101012, roughness: 0.65 });
  const helmetM = new THREE.MeshPhysicalMaterial({ color: 0xe9e7e2, roughness: 0.22, clearcoat: 1, clearcoatRoughness: 0.04 });
  const visorM = new THREE.MeshPhysicalMaterial({ color: 0x1a120a, metalness: 0.95, roughness: 0.04, clearcoat: 1 });
  const rider = new THREE.Group();
  body.add(rider);
  const mesh = (g, m) => { const o = new THREE.Mesh(g, m); o.castShadow = true; o.receiveShadow = true; rider.add(o); return o; };
  // a limb segment: tapered cylinder of unit length along +y, scaled to fit
  const limb = (r0, r1, m = suit) => { const o = mesh(new THREE.CylinderGeometry(r1, r0, 1, 14, 1).translate(0, 0.5, 0), m); o.userData.len = 1; return o; };
  const setLimb = (o, a, b) => {
    const d = new THREE.Vector3().subVectors(b, a);
    o.position.copy(a);
    o.quaternion.setFromUnitVectors(V(0, 1, 0), d.clone().normalize());
    o.scale.set(1, d.length(), 1);
  };
  const ball = (r, m = suit) => mesh(new THREE.SphereGeometry(r, 16, 12), m);
  // torso: lathe profile from waist (y=0) to collar (y=0.56), flattened front-to-back
  const torsoG = new THREE.LatheGeometry([[0, 0], [0.125, 0.0], [0.135, 0.08], [0.15, 0.2], [0.18, 0.33], [0.19, 0.42], [0.165, 0.5], [0.09, 0.56], [0.0, 0.57]]
    .map(([r, y]) => new THREE.Vector2(r, y)), 28);
  torsoG.scale(0.66, 1, 1);
  const parts = {
    pelvis: mesh(new RoundedBoxGeometry(0.24, 0.16, 0.32, 4, 0.07), suit),
    torso: mesh(torsoG, suit),
    back: mesh(new RoundedBoxGeometry(0.07, 0.36, 0.26, 4, 0.03), suitHi),     // back protector
    band: mesh(new THREE.CylinderGeometry(0.132, 0.14, 0.06, 28).scale(0.66, 1, 1), suit2),
    neck: limb(0.05, 0.047),
    upperArm: [limb(0.052, 0.044), limb(0.052, 0.044)],
    foreArm: [limb(0.043, 0.036), limb(0.043, 0.036)],
    shoulder: [ball(0.062, suit2), ball(0.062, suit2)],
    elbow: [ball(0.044), ball(0.044)],
    hand: [mesh(new RoundedBoxGeometry(0.09, 0.05, 0.1, 3, 0.022), glove), mesh(new RoundedBoxGeometry(0.09, 0.05, 0.1, 3, 0.022), glove)],
    thigh: [limb(0.085, 0.063), limb(0.085, 0.063)],
    shin: [limb(0.058, 0.044), limb(0.058, 0.044)],
    knee: [ball(0.066, suit2), ball(0.066, suit2)],
    boot: [mesh(new RoundedBoxGeometry(0.27, 0.1, 0.1, 3, 0.035), glove), mesh(new RoundedBoxGeometry(0.27, 0.1, 0.1, 3, 0.035), glove)],
    cuff: [mesh(new THREE.CylinderGeometry(0.058, 0.06, 0.12, 14), glove), mesh(new THREE.CylinderGeometry(0.058, 0.06, 0.12, 14), glove)],
  };
  const head = new THREE.Group();
  const helmet = new THREE.Mesh(new THREE.SphereGeometry(0.145, 36, 26), helmetM);
  helmet.scale.set(1.1, 1.0, 0.94);
  // phi = pi faces +x (forward) in three's sphere parametrisation
  const visor = new THREE.Mesh(new THREE.SphereGeometry(0.149, 36, 16, Math.PI - 0.9, 1.8, 1.0, 0.62), visorM);
  visor.scale.set(1.1, 1.0, 0.94);
  const chin = new THREE.Mesh(new THREE.SphereGeometry(0.147, 30, 10, Math.PI - 0.8, 1.6, 1.62, 0.5), helmetM);
  chin.scale.set(1.13, 1.0, 0.95);
  const hstripe = new THREE.Mesh(new THREE.SphereGeometry(0.1465, 36, 6, Math.PI * 0.5, Math.PI, 0.25, 0.2), suit2);
  hstripe.scale.set(1.1, 1.0, 0.94);
  const spoiler = new THREE.Mesh(new RoundedBoxGeometry(0.06, 0.03, 0.12, 2, 0.012), helmetM); spoiler.position.set(-0.155, 0.04, 0);
  [helmet, visor, chin, hstripe, spoiler].forEach((m) => { m.castShadow = true; head.add(m); });
  rider.add(head);

  // design-frame anchor points
  const GRIP = [V(0.62, 1.15, -0.30), V(0.62, 1.15, 0.30)];
  const PEG = [V(-0.02, 0.73, -0.215), V(-0.02, 0.73, 0.215)];

  function poseRider(lean, side, lookYaw, bob) {
    const hip = V(-0.36 + lean * 0.5, 1.10, 0);
    const torsoAng = 0.52 + lean;                    // forward lean from vertical, rad
    const up = V(Math.sin(torsoAng), Math.cos(torsoAng), 0);
    const roll = side;                                // lateral lean (small)
    const upR = V(up.x, up.y * Math.cos(roll), Math.sin(roll) * up.y).normalize();
    const q = new THREE.Quaternion().setFromUnitVectors(V(0, 1, 0), upR);
    parts.pelvis.position.copy(hip).add(V(0, 0.02, 0));
    parts.torso.position.copy(hip).addScaledVector(upR, 0.02 + bob); parts.torso.quaternion.copy(q);
    parts.band.position.copy(hip).addScaledVector(upR, 0.06 + bob); parts.band.quaternion.copy(q);
    const backN = V(-Math.cos(torsoAng), Math.sin(torsoAng), 0);
    parts.back.position.copy(hip).addScaledVector(upR, 0.33 + bob).addScaledVector(backN, 0.1); parts.back.quaternion.copy(q);
    const neck = hip.clone().addScaledVector(upR, 0.57 + bob);
    const headC = neck.clone().add(V(0.06, 0.13, 0));
    setLimb(parts.neck, neck.clone().addScaledVector(upR, -0.05), headC);
    head.position.copy(headC);
    // head stays closer to level than the body (vestibular reflex), looks into the turn
    head.rotation.set(-roll * 0.4, lookYaw, -0.22 + lean * 0.3, 'YXZ');
    for (let s = 0; s < 2; s++) {
      const z = s === 0 ? -1 : 1;
      const sh = hip.clone().addScaledVector(upR, 0.47 + bob).add(V(0, 0, z * 0.17));
      const grip = GRIP[s];
      const wrist = grip.clone().add(V(-0.06, 0.01, 0));
      const elbow = twoBoneIK(sh, wrist, 0.30, 0.27, V(-0.2, -1, z * 1.1));
      setLimb(parts.upperArm[s], sh, elbow);
      setLimb(parts.foreArm[s], elbow, wrist);
      parts.shoulder[s].position.copy(sh);
      parts.elbow[s].position.copy(elbow);
      parts.hand[s].position.copy(grip).add(V(-0.01, 0.01, 0));
      parts.cuff[s].position.copy(wrist); parts.cuff[s].quaternion.setFromUnitVectors(V(0, 1, 0), wrist.clone().sub(elbow).normalize());
      const hp = hip.clone().add(V(0.03, -0.02, z * 0.1));
      const ankle = PEG[s].clone().add(V(-0.05, 0.07, 0));
      const knee = twoBoneIK(hp, ankle, 0.46, 0.45, V(1, 0.7, z * 0.35));
      setLimb(parts.thigh[s], hp, knee);
      setLimb(parts.shin[s], knee, ankle);
      parts.knee[s].position.copy(knee);
      parts.boot[s].position.copy(ankle).add(V(0.06, -0.035, 0));
      parts.boot[s].rotation.z = -0.1;
    }
  }
  poseRider(0, 0, 0, 0);

  // exhaust anchor data in design frame
  const exits = LIFT.map((L) => V(...L.exit));
  const mainExit = V(...MAIN.exit);
  const tmpC = new THREE.Color();
  let fanAngle = [0, 0, 0, 0, 0];

  return {
    root, body, M, head, rider,
    update(s, dt) {
      root.position.set(s.p[0], s.p[1], s.p[2]);
      root.quaternion.set(s.q[1], s.q[2], s.q[3], s.q[0]);
      const cgNow = massProps(s.fuel).cg;
      body.position.set(-cgNow[0], -cgNow[1], -cgNow[2]);
      for (let i = 0; i < 4; i++) {
        const L = lift[i];
        const [a, b] = s.vanes[i];
        // vane tilt: rotating the flow toward +x pushes the bike toward +x
        // exhaust leaves along (-tan a, -1, -tan b): plates turn with it
        L.vaneA.rotation.z = -a;
        L.vaneB.rotation.x = b;
        fanAngle[i] += s.N[i] * 10500 * dt * (LIFT[i].spin);
        L.fan.rotation.y = fanAngle[i] % (Math.PI * 2);
        L.glowMat.color.copy(glowColor(s.egt[i], tmpC));
      }
      fanAngle[4] += s.mainN * 8900 * dt;
      main.fan.rotation.x = fanAngle[4] % (Math.PI * 2);
      main.glowMat.color.copy(glowColor(s.egt[4], tmpC));
      // rider: felt acceleration in the body frame shifts the torso a little
      const qb = new THREE.Quaternion(s.q[1], s.q[2], s.q[3], s.q[0]).invert();
      const aB = new THREE.Vector3(s.acc[0], s.acc[1] + 9.81, s.acc[2]).applyQuaternion(qb);
      const lean = THREE.MathUtils.clamp(-aB.x * 0.012, -0.06, 0.06) + THREE.MathUtils.clamp((s.airspeed - 25) * 0.004, 0, 0.1);
      const side = THREE.MathUtils.clamp(aB.z * 0.01, -0.05, 0.05);
      const yawRateB = new THREE.Vector3(...s.w).y;
      const look = THREE.MathUtils.clamp(yawRateB * 0.9, -0.45, 0.45);
      const bob = THREE.MathUtils.clamp((aB.y - 9.81) * -0.002, -0.02, 0.02);
      poseRider(lean, side, look, bob);
    },
    // world-space nozzle exits and exhaust directions (unit, direction the gas travels)
    exhausts(s) {
      const out = [];
      root.updateMatrixWorld(true);
      for (let i = 0; i < 4; i++) {
        const [a, b] = s.vanes[i];
        const dirB = V(-Math.tan(a), -1, -Math.tan(b)).normalize();
        const p = exits[i].clone(); body.localToWorld(p);
        const d = dirB.applyQuaternion(root.quaternion);
        out.push({ p, d, T: s.T[i], egt: s.egt[i], N: s.N[i], r: 0.07 });
      }
      const p = mainExit.clone(); body.localToWorld(p);
      out.push({ p, d: V(-MAIN_DIR[0], -MAIN_DIR[1], 0).applyQuaternion(root.quaternion), T: s.mainT, egt: s.egt[4], N: s.mainN, r: 0.085 });
      return out;
    },
  };
}
