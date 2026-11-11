// rig.js: הדמות התלת-ממדית. בונה קראטקה בגי לבן וממקם את חלקי הגוף לפי המפרקים ש-solve() מחזירה.

import * as THREE from 'three';
import { RoundedBoxGeometry } from 'three/addons/geometries/RoundedBoxGeometry.js';
import { DIM } from './body.js';

const Y = new THREE.Vector3(0, 1, 0);
const va = new THREE.Vector3(), vb = new THREE.Vector3(), vd = new THREE.Vector3();
const m4 = new THREE.Matrix4();
const X1 = new THREE.Vector3(), Y1 = new THREE.Vector3(), Z1 = new THREE.Vector3();

function setFrame(obj, pos, M) {
  obj.position.set(pos[0], pos[1], pos[2]);
  X1.set(M[0][0], M[1][0], M[2][0]); Y1.set(M[0][1], M[1][1], M[2][1]); Z1.set(M[0][2], M[1][2], M[2][2]);
  m4.makeBasis(X1, Y1, Z1);
  obj.quaternion.setFromRotationMatrix(m4);
}
// עצם בין שתי נקודות; f0..f1 הוא החלק מהמקטע שהיא מכסה
function setBone(mesh, a, b, f0 = 0, f1 = 1) {
  va.set(a[0], a[1], a[2]); vb.set(b[0], b[1], b[2]);
  vd.subVectors(vb, va);
  const L = vd.length() || 1e-6;
  mesh.position.copy(va).addScaledVector(vd, (f0 + f1) / 2);
  mesh.quaternion.setFromUnitVectors(Y, vd.multiplyScalar(1 / L));
}

export class Rig {
  constructor({ skin = 0xd9a47a, hair = 0x1a1410, belt = 0x111111, ribbon = null } = {}) {
    const g = this.group = new THREE.Group();
    const gi = new THREE.MeshStandardMaterial({ color: 0xf4f1e8, roughness: 0.92 });
    const giShade = new THREE.MeshStandardMaterial({ color: 0xdcd7c8, roughness: 0.95 });
    const sk = new THREE.MeshStandardMaterial({ color: skin, roughness: 0.75 });
    const hr = new THREE.MeshStandardMaterial({ color: hair, roughness: 0.6 });
    const bl = new THREE.MeshStandardMaterial({ color: belt, roughness: 0.8 });
    const dark = new THREE.MeshStandardMaterial({ color: 0x1c1512, roughness: 0.6 });
    const M = (geo, mat, parent = g) => { const m = new THREE.Mesh(geo, mat); m.castShadow = true; m.receiveShadow = true; parent.add(m); return m; };
    const cyl = (rTop, rBot, h, mat, parent) => M(new THREE.CylinderGeometry(rTop, rBot, h, 20, 1), mat, parent);
    const sph = (r, mat, parent) => M(new THREE.SphereGeometry(r, 20, 14), mat, parent);
    const P = this.p = {};

    // --- רגליים וזרועות ---
    for (const s of ['L', 'R']) {
      P['thigh' + s] = cyl(0.09, 0.112, DIM.thigh, gi);
      P['knee' + s] = sph(0.089, gi);
      P['shinSkin' + s] = cyl(0.04, 0.05, DIM.shin, sk);
      P['shin' + s] = cyl(0.092, 0.089, DIM.shin * 0.84, gi);          // מכנסי הגי רחבים עד מעל הקרסול
      P['hip' + s] = sph(0.11, gi);

      const foot = P['foot' + s] = new THREE.Group(); g.add(foot);
      sph(0.043, sk, foot);
      const sole = M(new RoundedBoxGeometry(0.17, 0.05, 0.09, 4, 0.024), sk, foot); sole.position.set(0.035, -0.052, 0);
      const toes = M(new RoundedBoxGeometry(0.09, 0.036, 0.094, 4, 0.017), sk, foot); toes.position.set(0.145, -0.06, 0);
      const heel = sph(0.04, sk, foot); heel.position.set(-0.045, -0.04, 0);

      P['shoulder' + s] = sph(0.083, gi);
      P['upper' + s] = cyl(0.068, 0.081, DIM.upper, gi);
      P['elbow' + s] = sph(0.068, gi);
      P['foreSkin' + s] = cyl(0.034, 0.045, DIM.fore, sk);
      P['sleeve' + s] = cyl(0.078, 0.068, DIM.fore * 0.72, gi);         // שרוול רחב עד אמצע האמה

      const hand = P['hand' + s] = new THREE.Group(); g.add(hand);
      const sg = s === 'L' ? -1 : 1;
      const fist = new THREE.Group(); hand.add(fist);
      const fb = M(new RoundedBoxGeometry(0.098, 0.078, 0.092, 4, 0.028), sk, fist); fb.position.set(0.062, 0.002, 0);
      const th = M(new RoundedBoxGeometry(0.05, 0.026, 0.03, 3, 0.012), sk, fist); th.position.set(0.075, -0.03, -sg * 0.022);
      const open = new THREE.Group(); hand.add(open);
      const palm = M(new RoundedBoxGeometry(0.10, 0.03, 0.086, 4, 0.013), sk, open); palm.position.set(0.062, 0, 0);
      const fing = M(new RoundedBoxGeometry(0.09, 0.022, 0.08, 4, 0.01), sk, open); fing.position.set(0.15, 0, 0);
      const thumb = M(new RoundedBoxGeometry(0.06, 0.02, 0.026, 3, 0.009), sk, open); thumb.position.set(0.085, -0.004, -sg * 0.05); thumb.rotation.y = sg * 0.35;
      const wrist = sph(0.031, sk, hand);
      P['fist' + s] = fist; P['open' + s] = open;
    }

    // --- גו: אגן, בטן, חזה ---
    const pelvis = P.pelvis = new THREE.Group(); g.add(pelvis);
    const hips = sph(1, gi, pelvis); hips.scale.set(0.135, 0.13, 0.198); hips.position.y = 0.01;
    // שולי המעיל מתחת לחגורה
    const skirt = M(new THREE.CylinderGeometry(1, 1.13, 0.25, 28, 1, true), gi, pelvis);
    skirt.material = gi.clone(); skirt.material.side = THREE.DoubleSide;
    skirt.scale.set(0.152, 1, 0.215); skirt.position.y = 0.005;
    const slit = M(new THREE.BoxGeometry(0.004, 0.2, 0.012), giShade, pelvis); slit.position.set(0.168, -0.02, 0.02);

    const waist = P.waist = new THREE.Group(); g.add(waist);
    const belly = M(new THREE.CylinderGeometry(1.1, 1, 0.26, 28, 1), gi, waist); belly.scale.set(0.138, 1, 0.19);
    // חגורה עם קשר ושני קצוות
    const beltG = P.belt = new THREE.Group(); g.add(beltG);
    const band = M(new THREE.CylinderGeometry(1, 1, 0.044, 28, 1), bl, beltG); band.scale.set(0.156, 1, 0.22);
    const knot = M(new RoundedBoxGeometry(0.035, 0.04, 0.06, 3, 0.012), bl, beltG); knot.position.set(0.16, 0, 0);
    for (const sg of [-1, 1]) {
      const tail = M(new THREE.BoxGeometry(0.008, 0.2, 0.04), bl, beltG);
      tail.position.set(0.168, -0.095, sg * 0.04); tail.rotation.x = sg * 0.28;
    }
    if (ribbon !== null) {
      // סרט זיהוי (אקא/שירו) קשור לחגורה מאחור
      const rb = new THREE.MeshStandardMaterial({ color: ribbon, roughness: 0.7 });
      const r1 = M(new RoundedBoxGeometry(0.03, 0.05, 0.05, 3, 0.012), rb, beltG); r1.position.set(-0.16, 0, 0.02);
      const r2 = M(new THREE.BoxGeometry(0.008, 0.16, 0.03), rb, beltG); r2.position.set(-0.163, -0.08, 0.02); r2.rotation.x = 0.15;
    }

    const chest = P.chest = new THREE.Group(); g.add(chest);
    const rib = sph(1, gi, chest); rib.scale.set(0.146, 0.2, 0.226); rib.position.y = -0.13;
    const upperC = sph(1, gi, chest); upperC.scale.set(0.132, 0.105, 0.245); upperC.position.set(-0.005, -0.045, 0);
    // דש המעיל: שמאל מעל ימין, ופתח צוואר
    const lap = (z0, z1, uMax = 1) => {
      // פס הדש עוקב אחרי פני החזה, מהצוואר אל המותן הנגדית
      const pts = [];
      for (let i = 0; i <= 14; i++) {
        const u = (i / 14) * uMax, y = 0.035 - 0.30 * u, z = z0 + (z1 - z0) * u;
        const q = 1 - ((y + 0.13) / 0.2) ** 2 - (z / 0.226) ** 2;
        let x = 0.146 * Math.sqrt(Math.max(0.02, q));
        if (y < -0.17) x = Math.max(x, 0.139 + (-(y + 0.17)) * 0.06);
        pts.push(new THREE.Vector3(x + 0.003, y, z));
      }
      M(new THREE.TubeGeometry(new THREE.CatmullRomCurve3(pts), 24, 0.0075, 6), giShade, chest);
    };
    lap(0.08, -0.05); lap(-0.08, 0.035, 0.64);   // שמאל מעל ימין
    const vneck = sph(1, sk, chest); vneck.scale.set(0.05, 0.085, 0.05); vneck.position.set(0.074, 0.0, 0);

    P.neck = cyl(0.052, 0.062, 0.13, sk);

    // --- ראש ---
    const head = P.head = new THREE.Group(); g.add(head);
    const skull = sph(0.104, sk, head); skull.scale.set(1.0, 1.14, 0.9);
    const jaw = sph(0.075, sk, head); jaw.scale.set(0.95, 0.9, 0.95); jaw.position.set(0.028, -0.062, 0);
    const hairM = M(new THREE.SphereGeometry(0.114, 24, 16, 0, Math.PI * 2, 0, Math.PI * 0.5), hr, head);
    hairM.scale.set(1.0, 1.12, 0.93); hairM.rotation.z = 0.6; hairM.position.set(-0.004, 0.006, 0);
    for (const sg of [-1, 1]) {
      const ear = sph(0.022, sk, head); ear.scale.set(0.6, 1, 0.45); ear.position.set(-0.005, -0.012, sg * 0.094);
      const eye = sph(0.0105, dark, head); eye.position.set(0.094, 0.012, sg * 0.036);
      const brow = M(new THREE.BoxGeometry(0.008, 0.007, 0.036), hr, head); brow.position.set(0.098, 0.036, sg * 0.036); brow.rotation.x = sg * 0.18;
    }
    const nose = sph(0.014, sk, head); nose.scale.set(1.2, 1.5, 0.9); nose.position.set(0.106, -0.014, 0);
    const mouth = M(new THREE.BoxGeometry(0.006, 0.005, 0.038), dark, head); mouth.position.set(0.094, -0.058, 0);
  }

  update(J) {
    const P = this.p;
    for (const s of ['L', 'R']) {
      const hip = J['hip' + s], kn = J['kn' + s], an = J['an' + s];
      setBone(P['thigh' + s], hip, kn);
      P['knee' + s].position.set(kn[0], kn[1], kn[2]);
      P['hip' + s].position.set(hip[0], hip[1], hip[2]);
      setBone(P['shinSkin' + s], kn, an);
      setBone(P['shin' + s], kn, an, 0, 0.84);
      setFrame(P['foot' + s], an, J['F' + s]);

      const sh = J['sh' + s], el = J['el' + s], wr = J['wr' + s];
      P['shoulder' + s].position.set(sh[0], sh[1], sh[2]);
      setBone(P['upper' + s], sh, el);
      P['elbow' + s].position.set(el[0], el[1], el[2]);
      setBone(P['foreSkin' + s], el, wr);
      setBone(P['sleeve' + s], el, wr, 0, 0.72);
      setFrame(P['hand' + s], wr, J['G' + s]);
      P['fist' + s].visible = !J['open' + s];
      P['open' + s].visible = J['open' + s];
    }
    setFrame(P.pelvis, J.pel, J.P);
    const up = (M, k) => [M[0][1] * k, M[1][1] * k, M[2][1] * k];
    const w = up(J.T, 0.24);
    setFrame(P.waist, [J.pel[0] + w[0], J.pel[1] + w[1], J.pel[2] + w[2]], J.B);
    const b = up(J.T, 0.125);
    setFrame(P.belt, [J.pel[0] + b[0], J.pel[1] + b[1], J.pel[2] + b[2]], J.P);
    setFrame(P.chest, J.chest, J.C);
    setBone(P.neck, J.chest, J.head, 0.25, 0.8);
    setFrame(P.head, J.head, J.H);
  }
}
