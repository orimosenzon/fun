// המודל התלת-ממדי של הרודסטר: מרכב בלופטינג (חתכים לאורך הרכב), חופה, גלגלים,
// כנף אחורית אקטיבית, פסי אור, נחירים, מאווררים, תא נהג, ורכיבים פנימיים לתצוגת רנטגן.
import * as THREE from 'three';
import { tireGeometry, tube } from './shapes.js';
import { AERO } from '../params.js';

const smooth = (a, b, x) => { const t = Math.min(1, Math.max(0, (x - a) / (b - a))); return t * t * (3 - 2 * t); };
// אינטרפולציה חלקה בטבלה [[z, v], ...] ממוינת לפי z
function table(rows) {
  return (z) => {
    if (z <= rows[0][0]) return rows[0][1];
    for (let i = 1; i < rows.length; i++) {
      if (z <= rows[i][0]) {
        const [z0, v0] = rows[i - 1], [z1, v1] = rows[i];
        const t = (z - z0) / (z1 - z0), s = t * t * (3 - 2 * t);
        return v0 + (v1 - v0) * s;
      }
    }
    return rows[rows.length - 1][1];
  };
}

// לופטינג: ring(z) מחזיר מערך [x, y] סגור. מחזיר גיאומטריה עם מכסים בקצוות.
function loft(zs, ring, { caps = true } = {}) {
  const pos = [], idx = [];
  const M = ring(zs[0]).length;
  for (const z of zs) for (const [x, y] of ring(z)) pos.push(x, y, z);
  for (let i = 0; i < zs.length - 1; i++) for (let j = 0; j < M; j++) {
    const a = i * M + j, b = i * M + (j + 1) % M, c = (i + 1) * M + j, d = (i + 1) * M + (j + 1) % M;
    idx.push(a, b, c, b, d, c);
  }
  if (caps) {
    for (const [k, flip] of [[0, true], [zs.length - 1, false]]) {
      let cx = 0, cy = 0;
      for (let j = 0; j < M; j++) { cx += pos[(k * M + j) * 3]; cy += pos[(k * M + j) * 3 + 1]; }
      const ci = pos.length / 3; pos.push(cx / M, cy / M, zs[k]);
      for (let j = 0; j < M; j++) {
        const a = k * M + j, b = k * M + (j + 1) % M;
        if (flip) idx.push(ci, b, a); else idx.push(ci, a, b);
      }
    }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  g.setIndex(idx);
  g.computeVertexNormals();
  return g;
}

export const BODY = {
  halfW: table([[-2.18, 0.86], [-2.05, 0.95], [-1.6, 1.0], [1.3, 1.0], [1.75, 0.95], [2.05, 0.8], [2.18, 0.55]]),
  top: table([[-2.18, 0.12], [-2.12, 0.30], [-1.9, 0.345], [-1.2, 0.33], [-0.4, 0.30], [0.7, 0.24], [1.3, 0.15], [1.8, 0.06], [2.1, -0.04], [2.18, -0.1]]),
  shoulder: table([[-2.18, 0.04], [-1.5, 0.10], [0, 0.05], [1.5, 0.0], [2.18, -0.12]]),
  bottom: table([[-2.18, -0.16], [-1.95, -0.29], [1.9, -0.30], [2.18, -0.22]]),
};

function wheelZs(P) {
  const lf = P.frontWeight;
  return { zf: P.wheelbase * (1 - lf), zr: -P.wheelbase * lf };
}

// ---------- חומרים ----------
export function carMaterials() {
  return {
    paint: new THREE.MeshPhysicalMaterial({ color: 0xf2f2ef, metalness: 0.05, roughness: 0.3, clearcoat: 1, clearcoatRoughness: 0.06 }),
    glass: new THREE.MeshPhysicalMaterial({ color: 0x0a0e13, metalness: 0.3, roughness: 0.05, transparent: true, opacity: 0.62, clearcoat: 1, side: THREE.FrontSide }),
    dark: new THREE.MeshStandardMaterial({ color: 0x15171a, roughness: 0.55, metalness: 0.2 }),
    carbon: new THREE.MeshStandardMaterial({ color: 0x1d2024, roughness: 0.35, metalness: 0.4 }),
    tire: new THREE.MeshStandardMaterial({ color: 0x121212, roughness: 0.85 }),
    rim: new THREE.MeshStandardMaterial({ color: 0x2b2e33, roughness: 0.3, metalness: 0.85 }),
    caliper: new THREE.MeshStandardMaterial({ color: 0xc8102e, roughness: 0.4, metalness: 0.3 }),
    lightW: new THREE.MeshStandardMaterial({ color: 0xffffff, emissive: 0xeaf4ff, emissiveIntensity: 3 }),
    lightR: new THREE.MeshStandardMaterial({ color: 0x550000, emissive: 0xff1a1a, emissiveIntensity: 2.2 }),
    nozzle: new THREE.MeshStandardMaterial({ color: 0x8a8f96, roughness: 0.25, metalness: 0.95 }),
    nozzleGlow: new THREE.MeshBasicMaterial({ color: 0x9fd8ff, transparent: true, opacity: 0, blending: THREE.AdditiveBlending, depthWrite: false }),
    seat: new THREE.MeshStandardMaterial({ color: 0xeeeeea, roughness: 0.7 }),
    interior: new THREE.MeshStandardMaterial({ color: 0x1a1c20, roughness: 0.8 }),
    suit: new THREE.MeshStandardMaterial({ color: 0xf5f5f5, roughness: 0.6 }),
    visor: new THREE.MeshPhysicalMaterial({ color: 0x111111, metalness: 0.9, roughness: 0.05, clearcoat: 1 }),
    copv: new THREE.MeshStandardMaterial({ color: 0x2a2d31, roughness: 0.4, metalness: 0.5 }),
    battery: new THREE.MeshStandardMaterial({ color: 0x2f6fd6, roughness: 0.5, metalness: 0.3, transparent: true, opacity: 0.9 }),
    motor: new THREE.MeshStandardMaterial({ color: 0xc77d2a, roughness: 0.35, metalness: 0.8 }),
    fan: new THREE.MeshStandardMaterial({ color: 0x3ec2a0, roughness: 0.4, metalness: 0.4 }),
    pipe: new THREE.MeshStandardMaterial({ color: 0x9fd8ff, roughness: 0.3, metalness: 0.6, emissive: 0x0a2a40 }),
    xray: new THREE.MeshBasicMaterial({ color: 0x88c8ff, wireframe: true, transparent: true, opacity: 0.12, depthWrite: false }),
  };
}

// ---------- בניית המודל ----------
export function buildCarModel(P, opts = {}) {
  const M = carMaterials();
  const root = new THREE.Group();
  const body = new THREE.Group(); root.add(body);
  const { zf, zr } = wheelZs(P);
  const yW = -P.cgHeight + P.wheelRadius;   // גובה מרכז הגלגל במנוחה
  const ARCH = P.wheelRadius + 0.055;

  // החופה (מוגדרת כאן כי גם המרכב צריך אותה: אמבט תא הנהג נחפר מתחתיה)
  const canW = table([[-1.15, 0.42], [-0.7, 0.68], [0.2, 0.72], [0.65, 0.6], [0.95, 0.3]]);
  const canTop = table([[-1.15, 0.33], [-0.8, 0.62], [-0.3, 0.73], [0.2, 0.68], [0.65, 0.42], [0.95, 0.25]]);

  // מרכב: חתך סופר-אליפטי, עם גומחות לגלגלים
  const ring = (z) => {
    const w = BODY.halfW(z), top = BODY.top(z), sh = BODY.shoulder(z), bot = BODY.bottom(z);
    const pts = [];
    const N = 72;
    for (let j = 0; j < N; j++) {
      const phi = (j / N) * Math.PI * 2;
      const c = Math.cos(phi), s = Math.sin(phi);
      let x, y;
      if (s >= 0) {
        const n = 2.4; x = w * Math.sign(c) * Math.pow(Math.abs(c), 2 / n); y = sh + (top - sh) * Math.pow(s, 2 / n);
        // כנפיים (fenders) מעל הגלגלים: בליטה בצדדים בלבד, כמו במכוניות לה-מאן
        const r = Math.abs(x) / w;
        const side = smooth(0.42, 0.8, r) * (1 - smooth(0.97, 1.0, r) * 0.6);
        const bump = 0.34 * Math.exp(-(((z - zf) / 0.6) ** 2)) + 0.13 * Math.exp(-(((z - zr) / 0.7) ** 2));
        y += bump * side * Math.pow(s, 0.35);
        // אמבט תא הנהג: מתחת לחופה המכסה העליון יורד לרצפת התא
        if (z > -1.05 && z < 0.72) {
          const cw = canW(z) * 0.9;
          const zf2 = smooth(-1.05, -0.9, z) * (1 - smooth(0.58, 0.72, z));
          const dep = smooth(cw, cw - 0.1, Math.abs(x)) * zf2;
          y = y + (-0.1 - y) * dep;
        }
      }
      else { const n = 5; x = w * Math.sign(c) * Math.pow(Math.abs(c), 2 / n); y = sh - (sh - bot) * Math.pow(-s, 2 / n); }
      // גומחת גלגל: דוחפים פנימה את מה שבתוך עיגול הקשת
      for (const zw of [zf, zr]) {
        const d = Math.hypot(z - zw, y - yW);
        if (d < ARCH && Math.abs(x) > 0.6) {
          const t = smooth(ARCH, ARCH - 0.05, d);
          x = Math.sign(x) * (Math.abs(x) + (0.6 - Math.abs(x)) * t);
        }
      }
      pts.push([x, y]);
    }
    return pts;
  };
  const zs = []; for (let i = 0; i <= 170; i++) zs.push(-2.18 + 4.36 * i / 170);
  const bodyMesh = new THREE.Mesh(loft(zs, ring), M.paint);
  bodyMesh.castShadow = true; bodyMesh.receiveShadow = true;
  body.add(bodyMesh);

  // חופה כהה (בסגנון SR-71): כיפה נמוכה וארוכה
  const canRing = (z) => {
    const w = canW(z), base = BODY.top(z) - 0.015, top = Math.max(base + 0.005, canTop(z));
    const pts = [];
    const N = 32;
    for (let j = 0; j <= N; j++) {
      const phi = Math.PI * j / N, c = Math.cos(phi), s = Math.sin(phi), n = 2.2;
      pts.push([w * Math.sign(c) * Math.pow(Math.abs(c), 2 / n), base + (top - base) * Math.pow(s, 2 / n)]);
    }
    return pts;
  };
  const czs = []; for (let i = 0; i <= 50; i++) czs.push(-1.15 + 2.1 * i / 50);
  const canopy = new THREE.Mesh(loft(czs, canRing), M.glass);
  canopy.renderOrder = 2;
  body.add(canopy);
  // מסגרת החופה (נראית מבפנים, נותנת תחושת "תא")
  const frame = new THREE.Group();
  for (const sx of [1, -1]) {
    const pts = [];
    for (let i = 0; i <= 14; i++) { const z = 0.9 - i * 0.14; const w = canW(z) * 0.98, b = BODY.top(z); pts.push([sx * w * 0.97, b + 0.01, z]); }
    frame.add(new THREE.Mesh(tube(pts, 0.012, 30), M.dark));
    // עמוד A
    const a = [[sx * canW(0.62) * 0.95, BODY.top(0.62), 0.62], [sx * 0.42, 0.66, 0.2], [sx * 0.22, 0.725, -0.15]];
    frame.add(new THREE.Mesh(tube(a, 0.009, 16), M.dark));
  }
  body.add(frame);

  // רצפה שטוחה וחצאיות
  const floor = new THREE.Mesh(new THREE.BoxGeometry(1.84, 0.03, 3.5), M.dark);
  floor.position.set(0, P.floorY + 0.02, 0.1);
  body.add(floor);
  const skirtMat = M.dark;
  for (const sx of [1, -1]) {
    const sk = new THREE.Mesh(new THREE.BoxGeometry(0.03, 0.08, 2.0), skirtMat);
    sk.position.set(sx * 0.92, P.floorY - 0.03, 0.18); body.add(sk);
  }
  // דיפיוזר אחורי
  for (let i = -3; i <= 3; i++) {
    const fin = new THREE.Mesh(new THREE.BoxGeometry(0.015, 0.09, 0.42), M.carbon);
    fin.position.set(i * 0.26, -0.235, -1.93); fin.rotation.x = 0.3; body.add(fin);
  }
  // בטנת גומחות הגלגל (כהה)
  const linerMat = M.dark.clone(); linerMat.side = THREE.DoubleSide;
  for (const zw of [zf, zr]) for (const sx of [1, -1]) {
    const lg = new THREE.CylinderGeometry(ARCH - 0.004, ARCH - 0.004, 0.36, 28, 1, true, 0, Math.PI);
    lg.rotateZ(Math.PI / 2); // ציר הגליל לאורך x, החצי העליון מעל הסרן
    const liner = new THREE.Mesh(lg, linerMat);
    liner.position.set(sx * 0.81, yW, zw);
    body.add(liner);
  }

  // פסי אור: מחפשים את פני המרכב בגובה הרצוי (נקודה בתוך החתך, סורקים מהקצה פנימה)
  const inside = (pts, x, y) => {
    let c = false;
    for (let i = 0, j = pts.length - 1; i < pts.length; j = i++) {
      const [xi, yi] = pts[i], [xj, yj] = pts[j];
      if ((yi > y) !== (yj > y) && x < (xj - xi) * (y - yi) / (yj - yi) + xi) c = !c;
    }
    return c;
  };
  const surfaceZ = (x, y, front) => {
    for (let k = 0; k < 400; k++) {
      const z = front ? 2.18 - k * 0.004 : -2.18 + k * 0.004;
      if (inside(ring(z), x, y)) return z;
    }
    return front ? 1.8 : -1.8;
  };
  const barPts = (y, half, front, out) => {
    const pts = [];
    for (let i = 0; i <= 24; i++) { const x = -half + 2 * half * i / 24; pts.push([x, y, surfaceZ(x, y, front) + (front ? out : -out)]); }
    return pts;
  };
  const fl = new THREE.Mesh(tube(barPts(-0.03, 0.5, true, 0.004), 0.012, 48), M.lightW); body.add(fl);
  const rl = new THREE.Mesh(tube(barPts(0.2, 0.82, false, 0.004), 0.016, 48), M.lightR); body.add(rl);

  // כנף אחורית אקטיבית
  const wing = new THREE.Group();
  const foil = new THREE.Shape();
  foil.moveTo(0.16, 0); foil.quadraticCurveTo(0.12, 0.035, -0.02, 0.03); foil.quadraticCurveTo(-0.16, 0.012, -0.17, -0.004);
  foil.quadraticCurveTo(-0.02, -0.004, 0.16, 0);
  const wg = new THREE.ExtrudeGeometry(foil, { depth: 1.66, bevelEnabled: false });
  wg.translate(0, 0, -0.83); wg.rotateY(Math.PI / 2);
  const wingMesh = new THREE.Mesh(wg, M.carbon); wingMesh.castShadow = true;
  wing.add(wingMesh);
  for (const sx of [1, -1]) {
    const ep = new THREE.Mesh(new THREE.BoxGeometry(0.01, 0.1, 0.32), M.carbon);
    ep.position.set(sx * 0.83, 0.0, 0); wing.add(ep);
  }
  wing.position.set(0, 0.42, -1.92);
  const struts = new THREE.Group();
  for (const sx of [1, -1]) {
    const st = new THREE.Mesh(new THREE.BoxGeometry(0.02, 0.12, 0.08), M.carbon);
    st.position.set(sx * 0.42, 0.37, -1.92); struts.add(st);
  }
  body.add(wing, struts);

  // ---------- גלגלים ----------
  const wheels = [];
  const wheelDefs = [
    { x: P.trackF / 2, z: zf, w: 0.285 }, { x: -P.trackF / 2, z: zf, w: 0.285 },
    { x: P.trackR / 2, z: zr, w: 0.335 }, { x: -P.trackR / 2, z: zr, w: 0.335 },
  ];
  for (const d of wheelDefs) {
    const knuckle = new THREE.Group();       // מסתובב בהיגוי
    const spin = new THREE.Group();          // מסתובב סביב הציר
    knuckle.add(spin);
    const tireMesh = new THREE.Mesh(tireGeometry(P.wheelRadius, d.w), M.tire);
    tireMesh.castShadow = true;
    spin.add(tireMesh);
    const side = Math.sign(d.x);
    // חישוק אווירודינמי עם חישורים
    const rim = new THREE.Mesh(new THREE.CylinderGeometry(P.wheelRadius * 0.66, P.wheelRadius * 0.66, d.w * 0.7, 32), M.rim);
    rim.rotation.z = Math.PI / 2; spin.add(rim);
    for (let k = 0; k < 6; k++) {
      const sp = new THREE.Mesh(new THREE.BoxGeometry(0.02, P.wheelRadius * 1.2, 0.05), M.dark);
      sp.position.x = side * d.w * 0.36; sp.rotation.x = k * Math.PI / 6; spin.add(sp);
    }
    const cap = new THREE.Mesh(new THREE.CylinderGeometry(0.05, 0.05, 0.02, 16), M.nozzle);
    cap.rotation.z = Math.PI / 2; cap.position.x = side * d.w * 0.37; spin.add(cap);
    const cal = new THREE.Mesh(new THREE.BoxGeometry(0.06, 0.16, 0.1), M.caliper);
    cal.position.set(side * d.w * 0.18, 0.1, d.z > 0 ? -0.12 : 0.12); knuckle.add(cal);
    knuckle.position.set(d.x, yW, d.z);
    root.add(knuckle);
    wheels.push({ knuckle, spin, x: d.x, z: d.z });
  }

  // ---------- נחירים ----------
  const nozzles = [];
  for (const t of P.thrusters) {
    const g = new THREE.Group();
    const size = t.group === 'main' ? 1 : t.group === 'lift' ? 0.85 : 0.5;
    const blen = t.group === 'lift' ? 0.05 : 0.12 * size;
    const bell = new THREE.Mesh(new THREE.CylinderGeometry(0.048 * size, 0.03 * size, blen, 20, 1, true), M.nozzle);
    if (t.group === 'lift') bell.position.y = -0.02;
    bell.material = M.nozzle;
    const inner = new THREE.Mesh(new THREE.CircleGeometry(0.045 * size, 20), M.nozzleGlow.clone());
    inner.rotation.x = -Math.PI / 2; inner.position.y = 0.055 * size; inner.material.side = THREE.DoubleSide;
    g.add(bell, inner);
    // היישור: הסילון יוצא בכיוון −dir. הגליל מצביע ב-+y → מסובבים את +y אל −dir
    const ex = new THREE.Vector3(-t.dir[0], -t.dir[1], -t.dir[2]);
    g.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), ex);
    g.position.set(...t.pos);
    body.add(g);
    nozzles.push({ g, glow: inner, exit: ex.clone(), pos: new THREE.Vector3(...t.pos), size });
  }

  // מוצאי המאווררים מעל הדיפיוזר
  const fans = [];
  for (const x of [0.62, 0.22, -0.22, -0.62]) {
    const fg = new THREE.Group();
    const ringM = new THREE.Mesh(new THREE.TorusGeometry(0.11, 0.015, 8, 24), M.carbon);
    fg.add(ringM);
    const blades = new THREE.Group();
    for (let k = 0; k < 7; k++) {
      const b = new THREE.Mesh(new THREE.BoxGeometry(0.1, 0.03, 0.005), M.rim);
      b.position.x = 0.05; b.rotation.y = 0.5;
      const holder = new THREE.Group(); holder.rotation.z = k * Math.PI * 2 / 7; holder.add(b); blades.add(holder);
    }
    fg.add(blades);
    const back = new THREE.Mesh(new THREE.CircleGeometry(0.105, 24), M.dark); back.position.z = 0.01; back.rotation.y = Math.PI; fg.add(back);
    fg.position.set(x, 0.0, -2.11);
    fg.rotation.y = Math.PI;
    body.add(fg);
    fans.push(blades);
  }

  // ---------- תא הנהג ----------
  const cockpit = new THREE.Group();
  const single = P.seats === 1;
  const driverX = single ? 0 : 0.36;
  const seatXs = single ? [0] : [0.36, -0.36];
  for (const sx of seatXs) {
    const seat = new THREE.Group();
    const base = new THREE.Mesh(new THREE.BoxGeometry(0.5, 0.1, 0.5), M.seat); base.position.set(0, -0.05, -0.45);
    const back = new THREE.Mesh(new THREE.BoxGeometry(0.5, 0.62, 0.08), M.seat); back.position.set(0, 0.08, -0.74); back.rotation.x = -0.35;
    seat.add(base, back); seat.position.x = sx; cockpit.add(seat);
  }
  // נהג (כמו ה"סטארמן" בהדגמה)
  const driver = new THREE.Group();
  const torso = new THREE.Mesh(new THREE.CapsuleGeometry(0.17, 0.32, 4, 12), M.suit); torso.position.set(0, 0.08, -0.62); torso.rotation.x = -0.35;
  const head = new THREE.Mesh(new THREE.SphereGeometry(0.135, 20, 16), M.suit); head.position.set(0, 0.47, -0.66);
  const visor = new THREE.Mesh(new THREE.SphereGeometry(0.137, 20, 16, -0.9, 1.8, 0.9, 1.0), M.visor); visor.position.copy(head.position); visor.rotation.y = 0;
  driver.add(torso, head, visor);
  driver.position.x = driverX;
  cockpit.add(driver);
  // לוח מחוונים
  // לוח מחוונים: מעטפת כהה לרוחב התא, שמסתירה את פנים המרכב
  const dashShape = new THREE.Shape();
  dashShape.moveTo(-0.05, -0.12); dashShape.lineTo(0.3, -0.08); dashShape.quadraticCurveTo(0.36, 0.04, 0.28, 0.1);
  dashShape.lineTo(0.05, 0.10); dashShape.quadraticCurveTo(-0.08, 0.06, -0.05, -0.12);
  const dg = new THREE.ExtrudeGeometry(dashShape, { depth: 1.16, bevelEnabled: true, bevelThickness: 0.02, bevelSize: 0.02, bevelSegments: 3 });
  dg.translate(0, 0, -0.58); dg.rotateY(-Math.PI / 2);
  const dash = new THREE.Mesh(dg, M.interior); dash.position.set(0, 0.17, 0.30);
  cockpit.add(dash);
  // גוש מחוונים מאחורי ההגה
  const binn = new THREE.Mesh(new THREE.BoxGeometry(0.34, 0.07, 0.16), M.interior); binn.position.set(driverX, 0.3, 0.28); binn.rotation.x = -0.4;
  cockpit.add(binn);
  // שכבת זכוכית פנימית עדינה, כדי שמבפנים ירגישו שיש חופה
  const inner = new THREE.Mesh(canopy.geometry, new THREE.MeshBasicMaterial({ color: 0x6a8aa8, transparent: true, opacity: 0.07, side: THREE.BackSide, depthWrite: false }));
  cockpit.add(inner);
  // הגה-עול (yoke)
  const yoke = new THREE.Group();
  const yokeShape = [[-0.17, -0.02, 0], [-0.17, 0.09, 0], [-0.12, 0.11, 0], [0.12, 0.11, 0], [0.17, 0.09, 0], [0.17, -0.02, 0]];
  yoke.add(new THREE.Mesh(tube(yokeShape, 0.016, 24), M.interior));
  const hub = new THREE.Mesh(new THREE.BoxGeometry(0.12, 0.08, 0.04), M.interior); hub.position.y = 0.02; yoke.add(hub);
  yoke.position.set(driverX, 0.25, 0.1); yoke.rotation.x = -0.3;
  cockpit.add(yoke);
  // מסך מרכזי עם טלמטריה (CanvasTexture)
  const screenCanvas = document.createElement('canvas'); screenCanvas.width = 512; screenCanvas.height = 300;
  const screenTex = new THREE.CanvasTexture(screenCanvas); screenTex.colorSpace = THREE.SRGBColorSpace;
  const screen = new THREE.Mesh(new THREE.PlaneGeometry(0.36, 0.21), new THREE.MeshBasicMaterial({ map: screenTex, toneMapped: false }));
  screen.position.set(single ? -0.3 : 0.0, 0.33, 0.27); screen.rotation.x = -0.5;
  screen.rotation.y = Math.PI; // פונה אחורה אל הנהג
  cockpit.add(screen);
  body.add(cockpit);

  // ---------- רכיבים פנימיים (רנטגן) ----------
  const xray = new THREE.Group(); xray.visible = false;
  const components = [];
  const addComp = (key, mesh, label) => { mesh.userData.comp = key; xray.add(mesh); components.push({ key, mesh, label }); return mesh; };
  // סוללה בתחתית
  const bat = addComp('battery', new THREE.Mesh(new THREE.BoxGeometry(1.5, 0.11, 2.2), M.battery), 'סוללה: תאי 4680 נמוך ברצפה');
  bat.position.set(0, -0.24, 0.05);
  // מנועים: אחד קדמי, שניים אחוריים
  const mf = addComp('motor', new THREE.Mesh(new THREE.CylinderGeometry(0.12, 0.12, 0.36, 20), M.motor), 'מנוע חשמלי קדמי');
  mf.rotation.z = Math.PI / 2; mf.position.set(0, -0.12, zf);
  for (const sx of [0.22, -0.22]) {
    const mr = addComp('motor', new THREE.Mesh(new THREE.CylinderGeometry(0.14, 0.14, 0.32, 20), M.motor), 'מנוע חשמלי אחורי');
    mr.rotation.z = Math.PI / 2; mr.position.set(sx, -0.1, zr);
  }
  // מכלי COPV
  if (P.tankVolume > 0) {
    const t1 = addComp('copv', new THREE.Mesh(new THREE.CapsuleGeometry(P.tankVolume > 0.3 ? 0.22 : 0.17, 1.05, 8, 20), M.copv), 'מכל COPV: אוויר ב-379 בר');
    t1.rotation.z = Math.PI / 2; t1.position.set(0, 0.0, -1.0);
    if (P.tankVolume > 0.3) {
      const t2 = addComp('copv', new THREE.Mesh(new THREE.CapsuleGeometry(0.2, 1.15, 8, 20), M.copv), 'מכל COPV שני, במקום המושב השני');
      t2.rotation.x = Math.PI / 2; t2.position.set(-0.47, -0.02, 0.05);
    }
    // צנרת אל הנחירים
    for (const t of P.thrusters) {
      const from = [0, -0.02, -1.0];
      const to = t.pos;
      const mid = [(from[0] + to[0]) / 2, Math.min(from[1], to[1]) - 0.05, (from[2] + to[2]) / 2];
      addComp('pipe', new THREE.Mesh(tube([from, mid, to], 0.012, 16), M.pipe), 'קו לחץ גבוה אל נחיר');
    }
  }
  // מאווררי היניקה
  for (const x of [0.62, 0.22, -0.22, -0.62]) {
    const fu = addComp('fan', new THREE.Mesh(new THREE.CylinderGeometry(0.12, 0.12, 0.22, 20), M.fan), 'מאוורר יניקה, 20 קילוואט');
    fu.rotation.x = Math.PI / 2; fu.position.set(x, -0.05, -1.95);
  }
  // תעלת היניקה מתחת לרצפה
  const duct = addComp('fan', new THREE.Mesh(new THREE.BoxGeometry(1.7, 0.02, 3.3), new THREE.MeshBasicMaterial({ color: 0x3ec2a0, transparent: true, opacity: 0.25, depthWrite: false })), 'החלל האטום מתחת לרצפה');
  duct.position.set(0, P.floorY - 0.04, 0.1);
  body.add(xray);

  const bodyMats = [bodyMesh, canopy];

  let xrayOn = false;
  function setXray(on) {
    xrayOn = on; xray.visible = on;
    bodyMesh.material = on ? M.xray : M.paint;
    canopy.visible = !on;
    cockpit.visible = !on;
    frame.visible = !on;
  }

  // ---------- עדכון לכל פריים ----------
  const tmpQ = new THREE.Quaternion();
  function update(car, dt, view) {
    root.position.set(car.pos.x, car.pos.y, car.pos.z);
    root.quaternion.set(car.q.x, car.q.y, car.q.z, car.q.w);
    const S = car.P.suspension;
    for (let i = 0; i < 4; i++) {
      const w = car.wheels[i], vw = wheels[i];
      vw.knuckle.position.y = w.mount.y - (S.rest - w.comp);
      vw.knuckle.rotation.y = w.steer;
      vw.spin.rotation.x = w.spin;
    }
    // כנף: זווית לפי מצב האווירודינמיקה
    const am = AERO.modes[car.aeroMode];
    const target = am.wing === 0 ? 0 : 0.05 + am.wing * 0.32;
    wing.rotation.x += (target - wing.rotation.x) * Math.min(1, dt * 4);
    const lift = am.wing === 0 ? -0.07 : 0;
    wing.position.y += (0.42 + lift - wing.position.y) * Math.min(1, dt * 4);
    struts.visible = am.wing !== 0 || wing.position.y > 0.37;
    // מאווררים
    for (const f of fans) f.rotation.z += car.fan.n * 160 * dt;
    // הילת הנחירים
    for (let i = 0; i < nozzles.length; i++) {
      const t = car.thr[i], nz = nozzles[i];
      nz.glow.material.opacity = Math.min(1, t.u * 1.4);
    }
    yoke.rotation.z = -car.wheels[0].steer * 4;
    // בתצוגת תא הנהג מסתירים את הנהג (אנחנו הוא)
    driver.visible = view !== 'cockpit' && !xrayOn;
  }

  return { root, body, wheels, nozzles, fans, wing, cockpit, screen, screenCanvas, screenTex, driver, xray, components, setXray, update, materials: M, driverX };
}
