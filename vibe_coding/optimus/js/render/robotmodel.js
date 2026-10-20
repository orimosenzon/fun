// המודל החזותי של אופטימוס 3. כל חלק בנוי במערכת הצירים של הגוף המתאים ב-MuJoCo
// (z למעלה לאורך האיבר), ומקבל בכל פריים את המיקום והסיבוב מהסימולציה.
import * as THREE from 'three';
import { loft, tube, capsuleBetween } from './loft.js';
import { materials } from './materials.js';
import { G, FINGERS } from '../robot/spec.js';

const V = (x, y, z) => new THREE.Vector3(x, y, z);

function mesh(geo, mat, cast = true) {
  const m = new THREE.Mesh(geo, mat);
  m.castShadow = cast; m.receiveShadow = true;
  return m;
}

// ---------- ראש: קסדה מבריקה עם מגן פנים מראתי ומסגרת כסופה דקה ----------
function buildHead(M) {
  const g = new THREE.Group();
  const hc = G.headC - G.neckZ;   // מרכז הראש מעל מפרק הצוואר
  const shell = loft([
    { z: 0.035, a: 0.05, b: 0.045, n: 2 },
    { z: 0.06, a: 0.078, ab: 0.07, b: 0.066, n: 2 },
    { z: 0.095, a: 0.098, ab: 0.09, b: 0.079, n: 2.1 },
    { z: 0.135, a: 0.104, ab: 0.098, b: 0.084, n: 2.1 },
    { z: 0.175, a: 0.1, ab: 0.098, b: 0.083, n: 2.1 },
    { z: 0.21, a: 0.084, ab: 0.085, b: 0.07, n: 2 },
    { z: 0.232, a: 0.055, ab: 0.058, b: 0.047, n: 2 },
    { z: 0.243, a: 0.018, ab: 0.02, b: 0.016, n: 2 },
  ].map(r => ({ ...r, x: 0.008 })), { seg: 56 });
  g.add(mesh(shell, M.helmet));
  // מגן הפנים: טלאי על אליפסואיד מעט גדול יותר, בגבול של "מלבן מעוגל"
  const C = V(0.012, 0, hc + 0.004), Rx = 0.103, Ry = 0.087, Rz = 0.112;
  const NU = 26, NV = 26, pts = [], uv = [];
  const az0 = 1.12, elTop = 0.62, elBot = -0.98;
  const toPos = (u, v) => {   // u,v ∈ [-1,1]
    const az = u * az0, el = v >= 0 ? v * elTop : -v * elBot;
    return V(C.x + Rx * Math.cos(el) * Math.cos(az) * 1.045, C.y + Ry * Math.cos(el) * Math.sin(az) * 1.04, C.z + Rz * Math.sin(el) * 1.02);
  };
  // רשת פולארית בתוך סופר-אליפסה, כדי שהקצוות יהיו מעוגלים
  const ring = (t, r) => { const c = Math.cos(t), s = Math.sin(t), e = 2 / 3.2; return [r * Math.sign(c) * Math.abs(c) ** e, r * Math.sign(s) * Math.abs(s) ** e]; };
  for (let i = 0; i <= NV; i++) for (let j = 0; j < NU; j++) {
    const [u, v] = ring((j / NU) * Math.PI * 2, i / NV);
    pts.push(toPos(u, v)); uv.push(u, v);
  }
  const pos = [], idx = [];
  pts.forEach(p => pos.push(p.x, p.y, p.z));
  for (let i = 0; i < NV; i++) for (let j = 0; j < NU; j++) {
    const a = i * NU + j, b = i * NU + (j + 1) % NU, c = (i + 1) * NU + j, d = (i + 1) * NU + (j + 1) % NU;
    idx.push(a, c, d, a, d, b);
  }
  const vg = new THREE.BufferGeometry();
  vg.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  vg.setIndex(idx); vg.computeVertexNormals();
  const visor = mesh(vg, M.visor, false);
  visor.material.side = THREE.DoubleSide;
  g.add(visor);
  // מסגרת כסופה סביב המגן
  const rimPts = [];
  for (let j = 0; j <= 64; j++) { const [u, v] = ring((j / 64) * Math.PI * 2, 1); rimPts.push(toPos(u, v).multiplyScalar(1)); }
  const rim = new THREE.Mesh(new THREE.TubeGeometry(new THREE.CatmullRomCurve3(rimPts, true), 160, 0.0028, 8, true), M.rim);
  g.add(rim);
  return g;
}

// ---------- גו: חזה זהב מעוגל, בטן בחליפה שחורה, "כתפיות" שחורות ----------
function buildTorso(M) {
  const g = new THREE.Group();
  const sz = G.shoulderZ - G.waistZ;   // 0.38
  const chest = loft([
    { z: 0.085, a: 0.098, ab: 0.09, b: 0.128, n: 2.5 },
    { z: 0.13, a: 0.11, ab: 0.095, b: 0.138, n: 2.6 },
    { z: 0.2, a: 0.122, ab: 0.1, b: 0.152, n: 2.7 },
    { z: 0.27, a: 0.128, ab: 0.104, b: 0.163, n: 2.8 },
    { z: 0.33, a: 0.124, ab: 0.104, b: 0.172, n: 2.9 },
    { z: sz + 0.01, a: 0.108, ab: 0.098, b: 0.168, n: 2.8 },
    { z: sz + 0.035, a: 0.085, ab: 0.085, b: 0.14, n: 2.6 },
    { z: sz + 0.052, a: 0.055, ab: 0.06, b: 0.085, n: 2.4 },
    { z: sz + 0.058, a: 0.04, ab: 0.045, b: 0.05, n: 2 },
  ], { seg: 56 });
  g.add(mesh(chest, M.gold));
  // הבטן: חליפה שחורה (חלק מה"בגד" שממשיך אל האגן)
  const belly = loft([
    { z: -0.04, a: 0.092, ab: 0.088, b: 0.122, n: 2.4 },
    { z: 0.02, a: 0.096, ab: 0.09, b: 0.126, n: 2.4 },
    { z: 0.09, a: 0.097, ab: 0.088, b: 0.127, n: 2.5 },
  ], { seg: 48 });
  g.add(mesh(belly, M.fabric));
  // כתפיות: רצועות שחורות שטוחות מעל הכתפיים, מהצוואר אל מפרק הכתף
  for (const sg of [1, -1]) {
    const strap = capsuleBetween(V(-0.008, sg * 0.05, sz + 0.047), V(-0.004, sg * 0.15, sz + 0.03), 0.03, M.fabric);
    strap.scale.set(1.5, 1, 0.55);
    g.add(strap);
  }
  // הצוואר (על הגו): שרוול שחור
  const neck = tube([[sz + 0.03, 0.05], [sz + 0.06, 0.044], [sz + 0.09, 0.04], [sz + 0.12, 0.041]], { seg: 28 });
  g.add(mesh(neck, M.fabric));
  return g;
}

// ---------- אגן: "בגד גוף" שחור ----------
function buildPelvis(M) {
  const g = new THREE.Group();
  const brief = loft([
    { z: -0.14, a: 0.04, ab: 0.05, b: 0.045, n: 2 },
    { z: -0.11, a: 0.07, ab: 0.085, b: 0.1, n: 2.2 },
    { z: -0.065, a: 0.09, ab: 0.098, b: 0.14, n: 2.5 },
    { z: -0.01, a: 0.094, ab: 0.098, b: 0.143, n: 2.6 },
    { z: 0.05, a: 0.092, ab: 0.09, b: 0.132, n: 2.5 },
    { z: 0.1, a: 0.092, ab: 0.088, b: 0.124, n: 2.4 },
  ], { seg: 48 });
  g.add(mesh(brief, M.fabric));
  return g;
}

// ---------- ירך: זהב, עם פתח הבגד בחלק העליון וברך שחורה ----------
function buildThigh(M, side) {
  const g = new THREE.Group();
  const s = side === 'l' ? 1 : -1;
  const L = G.thigh;
  const gold = loft([
    { z: -0.045, a: 0.08, ab: 0.086, b: 0.082, n: 2.2, tilt: [0.0, s * 0.055] },
    { z: -0.085, a: 0.082, ab: 0.083, b: 0.08, n: 2.3, tilt: [0.0, s * 0.03] },
    { z: -0.13, a: 0.08, ab: 0.078, b: 0.076, n: 2.3 },
    { z: -0.2, a: 0.074, ab: 0.07, b: 0.07, n: 2.3 },
    { z: -0.29, a: 0.066, ab: 0.062, b: 0.062, n: 2.3 },
    { z: -L + 0.07, a: 0.06, ab: 0.056, b: 0.057, n: 2.2 },
    { z: -L + 0.05, a: 0.056, ab: 0.052, b: 0.054, n: 2.1 },
  ].map(r => ({ ...r, y: -s * 0.006 })), { seg: 44, capBottom: false });
  g.add(mesh(gold, M.gold));
  // צד פנימי עליון של הירך: חלק מ"בגד הגוף" השחור (חיתוך גבוה)
  const inner = loft([
    { z: 0.0, a: 0.078, ab: 0.084, b: 0.08, n: 2.2 },
    { z: -0.1, a: 0.08, ab: 0.082, b: 0.078, n: 2.3 },
  ].map(r => ({ ...r, y: -s * 0.006, a: r.a - 0.002, ab: r.ab - 0.002, b: r.b - 0.002 })), { seg: 44, capTop: false, capBottom: true });
  g.add(mesh(inner, M.fabric));
  // ברך: כיסוי שחור
  const knee = tube([[-L + 0.075, 0.058], [-L + 0.04, 0.063], [-L + 0.0, 0.064], [-L - 0.03, 0.06]], { seg: 32 });
  g.add(mesh(knee, M.fabric));
  return g;
}

// ---------- שוק: "מגף" שחור עם שריר תאום מאחור (שם יושב המפעיל הקווי) ----------
function buildShin(M) {
  const g = new THREE.Group();
  const L = G.shin;
  const boot = loft([
    { z: -0.005, a: 0.058, ab: 0.064, b: 0.056, n: 2.2 },
    { z: -0.06, a: 0.054, ab: 0.078, b: 0.056, n: 2.3 },
    { z: -0.14, a: 0.05, ab: 0.074, b: 0.052, n: 2.3 },
    { z: -0.24, a: 0.045, ab: 0.056, b: 0.045, n: 2.2 },
    { z: -0.33, a: 0.04, ab: 0.043, b: 0.04, n: 2.1 },
    { z: -L + 0.02, a: 0.038, ab: 0.04, b: 0.038, n: 2 },
  ], { seg: 44 });
  g.add(mesh(boot, M.boot));
  // קפל בחלק העליון של המגף
  const top = tube([[0.0, 0.061], [-0.018, 0.063]], { seg: 32, capTop: false, capBottom: false });
  g.add(mesh(top, M.boot));
  return g;
}

// ---------- כף רגל: מגף שחור, עם קו מפרק בהונות ----------
function buildFoot(M) {
  const g = new THREE.Group();
  const az = G.ankleZ;
  const foot = loft([
    { z: -0.075, a: 0.026, b: 0.034, x: -az + 0.026, n: 3 },
    { z: -0.06, a: 0.04, b: 0.043, x: -az + 0.04, n: 3 },
    { z: -0.025, a: 0.058, b: 0.047, x: -az + 0.058, n: 3 },
    { z: 0.03, a: 0.05, b: 0.05, x: -az + 0.05, n: 3.2 },
    { z: 0.085, a: 0.036, b: 0.051, x: -az + 0.036, n: 3.4 },
    { z: 0.125, a: 0.026, b: 0.05, x: -az + 0.026, n: 3.6 },
    { z: 0.16, a: 0.019, b: 0.046, x: -az + 0.019, n: 3.4 },
    { z: 0.178, a: 0.01, b: 0.03, x: -az + 0.01, n: 2.6 },
  ], { seg: 40, axis: 'x' });
  g.add(mesh(foot, M.boot));
  // סוליה
  const sole = loft([
    { z: -0.072, a: 0.004, b: 0.032, x: -az + 0.004, n: 4 },
    { z: 0.0, a: 0.004, b: 0.047, x: -az + 0.004, n: 4 },
    { z: 0.16, a: 0.004, b: 0.045, x: -az + 0.004, n: 4 },
    { z: 0.177, a: 0.004, b: 0.03, x: -az + 0.004, n: 4 },
  ], { seg: 32, axis: 'x' });
  g.add(mesh(sole, M.pad));
  // קו מפרק הבהונות
  const seam = mesh(new THREE.BoxGeometry(0.003, 0.098, 0.03), M.joint, false);
  seam.position.set(0.118, 0, -az + 0.02); g.add(seam);
  // מפרק הקרסול
  const ank = mesh(new THREE.CylinderGeometry(0.032, 0.032, 0.07, 20), M.joint);
  ank.position.set(0, 0, 0.0); g.add(ank);
  return g;
}

// ---------- זרוע עליונה: כיפת כתף זהב, זרוע זהב, מרפק שחור ----------
function buildUpperArm(M) {
  const g = new THREE.Group();
  const L = G.upperArm;
  const arm = loft([
    { z: 0.035, a: 0.03, b: 0.03, n: 2 },
    { z: 0.02, a: 0.055, b: 0.054, n: 2 },
    { z: -0.01, a: 0.062, b: 0.06, n: 2.1 },
    { z: -0.07, a: 0.057, b: 0.055, n: 2.2 },
    { z: -0.15, a: 0.051, b: 0.05, n: 2.2 },
    { z: -L + 0.06, a: 0.047, b: 0.046, n: 2.1 },
    { z: -L + 0.045, a: 0.046, b: 0.045, n: 2 },
  ], { seg: 40 });
  g.add(mesh(arm, M.gold));
  const elbow = tube([[-L + 0.05, 0.046], [-L + 0.0, 0.047], [-L - 0.035, 0.045]], { seg: 28 });
  g.add(mesh(elbow, M.fabric));
  return g;
}

// ---------- אמה: עבה בחלק העליון (שם יושבים מנועי האצבעות ב-V3), שרוול שחור בשורש כף היד ----------
function buildForearm(M) {
  const g = new THREE.Group();
  const L = G.forearm;
  const arm = loft([
    { z: -0.03, a: 0.045, b: 0.046, n: 2.1 },
    { z: -0.065, a: 0.051, b: 0.05, n: 2.2 },
    { z: -0.12, a: 0.049, b: 0.047, n: 2.2 },
    { z: -0.18, a: 0.041, b: 0.039, n: 2.2 },
    { z: -L + 0.04, a: 0.034, b: 0.032, n: 2.1 },
    { z: -L + 0.028, a: 0.032, b: 0.03, n: 2 },
  ], { seg: 40 });
  g.add(mesh(arm, M.gold));
  const wrist = tube([[-L + 0.035, 0.032, 0.03], [-L + 0.005, 0.03, 0.028], [-L - 0.012, 0.027, 0.025]], { seg: 28 });
  g.add(mesh(wrist, M.fabric));
  return g;
}

// ---------- כף יד: כף שחורה מעוגלת ----------
function buildPalm(M, side) {
  const g = new THREE.Group();
  const s = side === 'l' ? 1 : -1, ps = -s;   // כיוון כף היד (פונה פנימה)
  const palm = loft([
    { z: -0.004, a: 0.028, b: 0.014, n: 2.2 },
    { z: -0.022, a: 0.041, b: 0.0175, n: 3 },
    { z: -0.06, a: 0.045, b: 0.018, n: 3.2 },
    { z: -0.088, a: 0.045, b: 0.016, n: 3.2 },
    { z: -G.palmLen - 0.004, a: 0.04, b: 0.012, n: 3 },
  ].map(r => ({ ...r, y: ps * 0.002 })), { seg: 36 });
  g.add(mesh(palm, M.hand));
  // כרית כף היד (חיישני מגע)
  const pad = mesh(new THREE.BoxGeometry(0.07, 0.004, 0.06), M.pad, false);
  pad.position.set(0, ps * 0.0175, -0.06); g.add(pad);
  return g;
}

function buildPhalanx(M, len, r, tip) {
  const g = new THREE.Group();
  const seg = capsuleBetween(V(0, 0, -r * 0.3), V(0, 0, -len + (tip ? r * 0.6 : r * 0.3)), r, M.hand);
  seg.castShadow = true;
  g.add(seg);
  const k = mesh(new THREE.CylinderGeometry(r * 1.02, r * 1.02, r * 1.2, 14), M.knuckle, false);
  k.rotation.z = Math.PI / 2;
  g.add(k);
  if (tip) {
    const pad = mesh(new THREE.SphereGeometry(r * 0.85, 12, 8), M.pad, false);
    pad.position.set(0, 0, -len + r * 0.7); pad.scale.set(0.9, 0.7, 1.3); g.add(pad);
  }
  return g;
}

// ---------- המודל המלא ----------
export function buildRobot(world) {
  const M = materials();
  const root = new THREE.Group();     // במערכת הצירים של MuJoCo; ההורה מסובב ל-y למעלה
  const byName = {};
  const add = (name, obj) => { byName[name] = obj; };
  add('pelvis', buildPelvis(M));
  add('torso', buildTorso(M));
  add('head', buildHead(M));
  for (const s of ['l', 'r']) {
    add(`${s}_thigh`, buildThigh(M, s));
    add(`${s}_shin`, buildShin(M));
    add(`${s}_foot`, buildFoot(M));
    add(`${s}_upper_arm`, buildUpperArm(M));
    add(`${s}_forearm`, buildForearm(M));
    add(`${s}_hand`, buildPalm(M, s));
    for (const f of FINGERS) {
      f.segs.forEach((len, i) => add(`${s}_${f.key}_${i + 1}`, buildPhalanx(M, len, f.r * (1 - 0.05 * i), i === 2)));
    }
  }
  const parts = [];
  for (const [name, obj] of Object.entries(byName)) {
    const id = world.bodyId(name);
    if (id < 0) continue;
    root.add(obj);
    parts.push({ id, obj });
  }
  const tmpQ = new THREE.Quaternion();
  return {
    root, parts,
    update(d) {
      const xp = d.xpos, xq = d.xquat;
      for (const p of parts) {
        const i = p.id;
        p.obj.position.set(xp[3 * i], xp[3 * i + 1], xp[3 * i + 2]);
        p.obj.quaternion.set(xq[4 * i + 1], xq[4 * i + 2], xq[4 * i + 3], xq[4 * i]);
      }
    },
    setVisible(v) { root.visible = v; },
    headVisible(v) { byName.head.visible = v; },
  };
}
