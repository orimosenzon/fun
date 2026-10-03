// launch_site.js — אתרי השיגור בגובה הקרקע: קרקע מפורטת, כן השיגור, מגדלים, מבנים, צמחייה ושמיים.
// מיקומי הכנים מדויקים (ויקיפדיה). מידות המבנים העיקריים לפי מקורות פתוחים; את הסביבה (עצים, בתים,
// דרכים) בנינו מהרכב הנוף האמיתי של כל אתר, אבל המיקומים הם קירוב ולא מפה.
// יחידות פנימיות: מטרים. ציר X מזרחה, Y למעלה, Z דרומה. ראשית הצירים: ציר הרקטה בגובה הקרקע.
import * as THREE from 'three';
import { mergeGeometries } from 'three/addons/utils/BufferGeometryUtils.js';
import { RE_KM, ecefDir } from './world.js';

export const GROUND_M = 4;          // גובה הקרקע המקומית מעל כדור הייחוס (מעל הקטע הלווייני שב-2 מ')
const RE_M = RE_KM * 1000;
const TERRAIN_R = 6500;             // רדיוס הקרקע המפורטת
const APRON = 900;                  // צלע הריבוע המפורט סביב הכן

// ---------- אתרים ----------
// הקואורדינטות המדויקות של כל כן נמצאות ב-rockets.js (site).
// bearing: אזימוט מהצפון עם כיוון השעון. sea: הים מתחיל במרחק dist מהכן בכיוון bearing.
export const SITES = {
  falcon9: {
    pad: 'SLC-40, קייפ קנוורל', baseH: 5,
    biome: 'florida', sea: { bearing: 72, dist: 1050 },
  },
  saturn5: {
    pad: 'LC-39A, מרכז החלל קנדי', baseH: 19.6,
    biome: 'florida', sea: { bearing: 80, dist: 650 },
  },
  soyuz: {
    pad: 'אתר 31/6, בייקונור', baseH: 2,
    biome: 'steppe',
  },
  cz5: {
    pad: 'LC-101, ון־צ\'אנג', baseH: 8,
    biome: 'tropic', sea: { bearing: 100, dist: 1250 },
  },
  starship: {
    pad: 'כן 2 (OLP-2), סטארבייס', baseH: 16,
    biome: 'flats', sea: { bearing: 90, dist: 720 },
  },
};

const BIOME = {
  florida: { base: '#5f7440', blobs: ['#4d6334', '#6f7f48', '#7d8455', '#56703a', '#8a8a5e'], sand: '#d9cfae', sea: '#2d6f8f', shallow: '#4f98a8', trees: 2600, kinds: ['palm', 'broad', 'shrub', 'shrub'] },
  steppe: { base: '#b3a27a', blobs: ['#a8976c', '#c2b38c', '#9d9a73', '#bcae86', '#a59a78'], sand: '#cdbf9a', sea: '#3d6f8f', shallow: '#5d8f9f', trees: 380, kinds: ['shrub', 'shrub', 'shrub', 'broad'] },
  tropic: { base: '#41702f', blobs: ['#356326', '#4f7d38', '#2f5a23', '#5d8a42', '#6b8f4a'], sand: '#e2d6b0', sea: '#1f6e8a', shallow: '#3f9aa6', trees: 4200, kinds: ['palm', 'broad', 'broad', 'palm', 'shrub'] },
  flats: { base: '#a99f7b', blobs: ['#9a946c', '#b8ad86', '#8c8e66', '#c3b892', '#7f8a5e'], sand: '#e0d4b0', sea: '#3a7c94', shallow: '#6aa4a8', trees: 700, kinds: ['shrub', 'shrub', 'shrub', 'palm'] },
};

// ---------- חומרים ----------
const M = {};
const mat = (k, p) => (M[k] ??= new THREE.MeshStandardMaterial({ roughness: 0.85, metalness: 0.05, ...p }));
const concrete = () => mat('concrete', { color: 0xb9b6ad, roughness: 0.95 });
const darkConcrete = () => mat('dconcrete', { color: 0x6f6d68, roughness: 0.95 });
const steelGray = () => mat('steelgray', { color: 0x6b7077, roughness: 0.6, metalness: 0.6 });
const darkSteel = () => mat('darksteel', { color: 0x3c4046, roughness: 0.6, metalness: 0.6 });
const shinySteel = () => mat('shiny', { color: 0xc9ccd1, roughness: 0.3, metalness: 0.9 });
const white = () => mat('white', { color: 0xeeeeea, roughness: 0.7 });
const apolloRed = () => mat('apollored', { color: 0xa8382c, roughness: 0.7, metalness: 0.3 });
const pit = () => mat('pit', { color: 0x1d1b19, roughness: 1 });

// ---------- גאומטריה ----------
const Y = new THREE.Vector3(0, 1, 0);
function beamGeo(a, b, t) {
  const A = new THREE.Vector3(...a), B = new THREE.Vector3(...b);
  const d = B.clone().sub(A), len = d.length();
  const g = new THREE.BoxGeometry(t, len, t);
  g.applyQuaternion(new THREE.Quaternion().setFromUnitVectors(Y, d.normalize()));
  g.translate((A.x + B.x) / 2, (A.y + B.y) / 2, (A.z + B.z) / 2);
  return g;
}
// מגדל סריג: 4 עמודים, קורות אופקיות ואלכסונים בכל פאה
function latticeGeo(w, d, h, step = 6, t = 0.6) {
  const gs = [];
  const C = [[-w / 2, -d / 2], [w / 2, -d / 2], [w / 2, d / 2], [-w / 2, d / 2]];
  for (const [x, z] of C) gs.push(beamGeo([x, 0, z], [x, h, z], t * 1.6));
  for (let y = 0; y <= h + 1e-6; y += step) {
    for (let i = 0; i < 4; i++) {
      const [x1, z1] = C[i], [x2, z2] = C[(i + 1) % 4];
      gs.push(beamGeo([x1, y, z1], [x2, y, z2], t));
      if (y + step <= h + 1e-6) gs.push(beamGeo([x1, y, z1], [x2, y + step, z2], t * 0.7));
    }
  }
  return mergeGeometries(gs);
}
const mesh = (geo, m, x = 0, y = 0, z = 0) => { const o = new THREE.Mesh(geo, m); o.position.set(x, y, z); return o; };
const boxM = (w, h, d, m, x, y0, z) => mesh(new THREE.BoxGeometry(w, h, d), m, x, y0 + h / 2, z);
const at = (o, x, z, ry = 0) => { o.position.set(x, 0, z); o.rotation.y = ry; return o; };
const cylM = (r, h, m, x, y0, z, seg = 24, rTop = r) => mesh(new THREE.CylinderGeometry(rTop, r, h, seg), m, x, y0 + h / 2, z);

// מבנה עם גג כהה וחלונות מצוירים
function building(w, h, d, color, roof = 0x55585c, plain = false) {
  const g = new THREE.Group();
  const tex = windowTex(color, plain ? 1 : Math.max(1, Math.round(h / 4)), Math.max(2, Math.round(w / 5)));
  const side = new THREE.MeshStandardMaterial({ map: tex, roughness: 0.85 });
  const top = mat('roof' + roof, { color: roof, roughness: 0.9 });
  const b = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), [side, side, top, top, side, side]);
  b.position.y = h / 2; g.add(b);
  return g;
}
const winCache = {};
function windowTex(color, rows, cols) {
  const k = color + rows + '_' + cols;
  if (winCache[k]) return winCache[k];
  const c = document.createElement('canvas'); c.width = 128; c.height = 128;
  const g = c.getContext('2d');
  g.fillStyle = color; g.fillRect(0, 0, 128, 128);
  if (rows > 1) {
    g.fillStyle = 'rgba(40,55,70,0.75)';
    const cw = 128 / cols, rh = 128 / rows;
    for (let r = 0; r < rows; r++) for (let q = 0; q < cols; q++) g.fillRect(q * cw + cw * 0.25, r * rh + rh * 0.3, cw * 0.5, rh * 0.4);
  }
  const t = new THREE.CanvasTexture(c); t.colorSpace = THREE.SRGBColorSpace;
  return (winCache[k] = t);
}

// ---------- רעש פשוט, דטרמיניסטי ----------
function rng(seed) { let s = seed % 2147483647; if (s <= 0) s += 2147483646; return () => (s = s * 16807 % 2147483647) / 2147483647; }

// ======================= בניית אתר =======================
export function buildSite(rid, quality = 'high') {
  const S = SITES[rid];
  const B = BIOME[S.biome];
  const root = new THREE.Group();
  const feat = { roads: [], slabs: [], water: [], keep: [], houses: [], trees: [] };
  const L = LAYOUT[rid](root, feat, S);
  feat.keep.push({ x: 0, z: 0, r: L?.clear ?? 140 });

  // ---------- קרקע ----------
  const seaDir = S.sea ? [Math.sin(S.sea.bearing * Math.PI / 180), -Math.cos(S.sea.bearing * Math.PI / 180)] : null;
  const seaD = (x, z) => seaDir ? x * seaDir[0] + z * seaDir[1] - S.sea.dist : -1e9;
  const inWater = (x, z) => seaD(x, z) > -55 || feat.water.some(w => ((x - w.x) / w.rx) ** 2 + ((z - w.z) / w.rz) ** 2 < 1.15);
  const distSeg = (px, pz, a, b) => {
    const dx = b[0] - a[0], dz = b[1] - a[1], l2 = dx * dx + dz * dz;
    const t = Math.max(0, Math.min(1, ((px - a[0]) * dx + (pz - a[1]) * dz) / l2));
    return Math.hypot(px - a[0] - t * dx, pz - a[1] - t * dz);
  };
  const blocked = (x, z, m = 0) =>
    inWater(x, z) ||
    feat.keep.some(k => Math.hypot(x - k.x, z - k.z) < k.r + m) ||
    feat.slabs.some(s => Math.abs(x - s.x) < s.w / 2 + 6 + m && Math.abs(z - s.z) < s.d / 2 + 6 + m) ||
    feat.roads.some(r => r.pts.some((p, i) => i > 0 && distSeg(x, z, r.pts[i - 1], p) < r.w / 2 + 5 + m));

  const big = quality === 'saver' ? 1024 : 2048;
  const terrainTex = paintGround(big, TERRAIN_R, B, feat, S, seaDir, 1, true);
  const apronTex = paintGround(big, APRON / 2, B, feat, S, seaDir, 2, false);
  // הקרקע שקופה רק בשוליים, אבל נצבעת בתור השקופים: חייבת לבוא לפני הסילון והעשן, אחרת תכסה אותם
  const terrain = curvedPlane(TERRAIN_R, 128, terrainTex, 0, true);
  terrain.renderOrder = -6;
  root.add(terrain);
  const apron = curvedPlane(APRON / 2, 64, apronTex, 0.12, false);
  apron.renderOrder = -5;
  root.add(apron);

  // ---------- צמחייה ----------
  const r = rng(rid.length * 7919 + 13);
  const kinds = { palm: [], broad: [], shrub: [] };
  let tries = 0;
  const nTrees = quality === 'saver' ? B.trees / 2 : B.trees;
  for (const t of feat.trees) kinds[t.kind].push(t);
  while (Object.values(kinds).reduce((s, a) => s + a.length, 0) < nTrees && tries++ < nTrees * 12) {
    // יותר עצים ליד הכן (שם הם נראים), פחות רחוק
    const rad = 160 + Math.pow(r(), 1.6) * 4200, ang = r() * Math.PI * 2;
    // עצים בקבוצות: דוחים נקודות בהסתברות שתלויה ב"צפיפות" מקומית
    const x = Math.cos(ang) * rad, z = Math.sin(ang) * rad;
    const dens = 0.5 + 0.5 * Math.sin(x * 0.004 + Math.sin(z * 0.003) * 2) * Math.cos(z * 0.0035 - x * 0.001);
    if (r() > dens) continue;
    if (blocked(x, z)) continue;
    const kind = B.kinds[Math.floor(r() * B.kinds.length)];
    kinds[kind].push({ x, z, s: 0.7 + r() * 0.7, c: r() });
  }
  root.add(...vegetation(kinds, S.biome));

  // ---------- בתים ----------
  const hs = feat.houses.filter(h => !blocked(h.x, h.z, -4));
  if (hs.length) root.add(...houses(hs, S.biome, rid));

  // עקמומיות כדור הארץ: כל עצם יורד לפי מרחקו מהמרכז
  for (const ch of root.children) {
    if (ch.userData.curved) continue;
    const { x, z } = ch.position;
    ch.position.y -= (x * x + z * z) / (2 * RE_M);
  }
  root.traverse(o => { if (o.isMesh && !o.userData.noShadow) { o.castShadow = true; o.receiveShadow = true; } });
  return { root, site: S, baseH: S.baseH, plumeDir: L?.plumeDir ?? [1, 0] };
}

// קרקע מעוקלת לפי כדור הארץ, עם טקסטורה ושוליים שקופים
function curvedPlane(half, seg, tex, lift, circle) {
  const g = circle ? new THREE.CircleGeometry(half, seg * 2) : new THREE.PlaneGeometry(half * 2, half * 2, seg, seg);
  g.rotateX(-Math.PI / 2);
  const p = g.attributes.position, uv = g.attributes.uv;
  for (let i = 0; i < p.count; i++) {
    const x = p.getX(i), z = p.getZ(i);
    p.setY(i, lift - (x * x + z * z) / (2 * RE_M));
    uv.setXY(i, (x + half) / (2 * half), 1 - (z + half) / (2 * half));
  }
  g.computeVertexNormals();
  const m = new THREE.Mesh(g, new THREE.MeshStandardMaterial({ map: tex, roughness: 0.95, metalness: 0, transparent: true, depthWrite: true }));
  m.userData.curved = true;
  m.userData.noShadow = true;
  m.receiveShadow = true;
  return m;
}

// ציור הקרקע לקנבס: צבע בסיס, כתמים, ים וחוף, מים פנימיים, דרכים ומשטחי בטון
function paintGround(N, half, B, feat, S, seaDir, seed, fadeEdge) {
  const c = document.createElement('canvas'); c.width = c.height = N;
  const g = c.getContext('2d');
  const k = N / (2 * half);
  g.setTransform(k, 0, 0, k, N / 2, N / 2);
  g.fillStyle = B.base; g.fillRect(-half, -half, 2 * half, 2 * half);
  const r = rng(9973 * seed + N);
  // כתמי צמחייה ואדמה בכמה קני מידה
  const sizes = fadeEdge ? [600, 220, 70] : [90, 30, 9, 3];
  for (const s of sizes) {
    const n = Math.min(9000, Math.round(((2 * half) / s) ** 2 * 2.2));
    for (let i = 0; i < n; i++) {
      const x = (r() * 2 - 1) * half, z = (r() * 2 - 1) * half, rr = s * (0.4 + r());
      g.globalAlpha = 0.18 + r() * 0.3;
      g.fillStyle = B.blobs[Math.floor(r() * B.blobs.length)];
      g.beginPath(); g.ellipse(x, z, rr, rr * (0.5 + r() * 0.5), r() * 3, 0, 7); g.fill();
    }
  }
  g.globalAlpha = 1;
  // מים פנימיים (לגונות, שטחי גאות)
  for (const w of feat.water) {
    g.fillStyle = w.color ?? B.shallow;
    g.beginPath(); g.ellipse(w.x, w.z, w.rx, w.rz, w.rot ?? 0, 0, 7); g.fill();
  }
  // ים וחוף: מסובבים כך שכיוון הים הוא +x
  if (seaDir) {
    g.save();
    g.rotate(Math.atan2(seaDir[1], seaDir[0]));
    const L = 3 * half + 20000, d = S.sea.dist;
    g.fillStyle = B.sand; g.fillRect(d - 70, -L, L, 2 * L);
    const gr = g.createLinearGradient(d, 0, d + 500, 0);
    gr.addColorStop(0, B.shallow); gr.addColorStop(0.05, B.shallow); gr.addColorStop(1, B.sea);
    g.fillStyle = gr; g.fillRect(d, -L, L, 2 * L);
    // קצף גלים
    g.strokeStyle = 'rgba(255,255,255,0.55)'; g.lineWidth = Math.max(1.5 / k, 2);
    for (const off of [6, 22, 45]) { g.beginPath(); g.moveTo(d + off, -L); g.lineTo(d + off, L); g.stroke(); }
    g.restore();
  }
  // דרכים
  for (const rd of feat.roads) {
    g.strokeStyle = rd.color ?? '#5d5c58'; g.lineWidth = rd.w; g.lineCap = 'round'; g.lineJoin = 'round';
    g.beginPath(); rd.pts.forEach(([x, z], i) => i ? g.lineTo(x, z) : g.moveTo(x, z)); g.stroke();
    if (rd.rail) {
      g.strokeStyle = '#3b3530'; g.lineWidth = 0.4;
      for (const s of [-0.75, 0.75]) { g.beginPath(); rd.pts.forEach(([x, z], i) => i ? g.lineTo(x + s, z) : g.moveTo(x + s, z)); g.stroke(); }
    }
    if (rd.stripe && 2 * half < 2000) {
      g.strokeStyle = '#d8c48a'; g.lineWidth = 0.25; g.setLineDash([3, 6]);
      g.beginPath(); rd.pts.forEach(([x, z], i) => i ? g.lineTo(x, z) : g.moveTo(x, z)); g.stroke(); g.setLineDash([]);
    }
  }
  // משטחי בטון ואספלט
  for (const s of feat.slabs) {
    g.save(); g.translate(s.x, s.z); g.rotate(s.rot ?? 0);
    g.fillStyle = s.color ?? '#a9a69d';
    if (s.round) { g.beginPath(); g.arc(0, 0, s.w / 2, 0, 7); g.fill(); } else g.fillRect(-s.w / 2, -s.d / 2, s.w, s.d);
    if (2 * half < 2000) { // תפרי יציקה וכתמי פיח בתקריב
      g.strokeStyle = 'rgba(0,0,0,0.12)'; g.lineWidth = 0.15;
      for (let x = -s.w / 2; x < s.w / 2; x += 6) { g.beginPath(); g.moveTo(x, -s.d / 2); g.lineTo(x, s.d / 2); g.stroke(); }
      for (let z = -s.d / 2; z < s.d / 2; z += 6) { g.beginPath(); g.moveTo(-s.w / 2, z); g.lineTo(s.w / 2, z); g.stroke(); }
    }
    g.restore();
  }
  // פיח ממנועים סביב הכן
  const soot = g.createRadialGradient(0, 0, 0, 0, 0, 45);
  soot.addColorStop(0, 'rgba(25,22,20,0.55)'); soot.addColorStop(1, 'rgba(25,22,20,0)');
  g.fillStyle = soot; g.fillRect(-45, -45, 90, 90);
  // שוליים שקופים, כדי שהקרקע תתמזג בתצלום הלוויין שמתחתיה
  g.setTransform(1, 0, 0, 1, 0, 0);
  g.globalCompositeOperation = 'destination-in';
  const f = g.createRadialGradient(N / 2, N / 2, 0, N / 2, N / 2, N / 2);
  f.addColorStop(0, 'rgba(0,0,0,1)'); f.addColorStop(fadeEdge ? 0.72 : 0.8, 'rgba(0,0,0,1)'); f.addColorStop(1, 'rgba(0,0,0,0)');
  g.fillStyle = f; g.fillRect(0, 0, N, N);
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace; t.anisotropy = 8;
  return t;
}

// ---------- עצים: InstancedMesh לכל סוג ----------
function vegetation(kinds, biome) {
  const out = [];
  const dummy = new THREE.Object3D();
  const greens = { florida: [0x3f5a2a, 0x4a6b30, 0x5c7438], steppe: [0x7d8358, 0x8c8a5c, 0x6f7a4c], tropic: [0x2f5c22, 0x3d6e2a, 0x4c7d30], flats: [0x6f7a4a, 0x7f8656, 0x5f6e40] }[biome];
  const col = new THREE.Color();
  const make = (list, parts) => {
    for (const [geo, m, colored] of parts) {
      const im = new THREE.InstancedMesh(geo, m, list.length);
      list.forEach((t, i) => {
        dummy.position.set(t.x, -(t.x * t.x + t.z * t.z) / (2 * RE_M), t.z);
        dummy.rotation.set(0, t.c * 6.28, 0);
        dummy.scale.setScalar(t.s);
        dummy.updateMatrix();
        im.setMatrixAt(i, dummy.matrix);
        if (colored) im.setColorAt(i, col.setHex(greens[Math.floor(t.c * greens.length) % greens.length]).offsetHSL(0, 0, (t.s - 1) * 0.08));
      });
      im.castShadow = true; im.receiveShadow = true;
      im.userData.curved = true;
      out.push(im);
    }
  };
  const trunk = mat('trunk', { color: 0x6b5338, roughness: 1 });
  const leaf = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.95, flatShading: true });
  if (kinds.broad.length) make(kinds.broad, [
    [new THREE.CylinderGeometry(0.25, 0.4, 5, 5).translate(0, 2.5, 0), trunk],
    [new THREE.IcosahedronGeometry(3.6, 1).scale(1, 0.85, 1).translate(0, 7, 0), leaf, true],
  ]);
  if (kinds.palm.length) {
    // דקל: גזע דק וגבוה וכתר של עלים כפופים
    const fronds = [];
    for (let i = 0; i < 7; i++) {
      const f = new THREE.ConeGeometry(0.7, 4.2, 3).rotateZ(Math.PI / 2 + 0.45).translate(2, 0, 0).rotateY(i * Math.PI * 2 / 7);
      fronds.push(f.translate(0, 0, 0));
    }
    make(kinds.palm, [
      [new THREE.CylinderGeometry(0.18, 0.3, 9, 5).translate(0, 4.5, 0), trunk],
      [mergeGeometries(fronds).translate(0, 9, 0), leaf, true],
    ]);
  }
  if (kinds.shrub.length) make(kinds.shrub, [
    [new THREE.IcosahedronGeometry(1.6, 0).scale(1.3, 0.7, 1.1).translate(0, 0.8, 0), leaf, true],
  ]);
  return out;
}

// ---------- בתים: קופסה + גג משופע, InstancedMesh ----------
function houses(list, biome, rid) {
  const dummy = new THREE.Object3D(), col = new THREE.Color();
  const walls = { florida: [0xe9e4d8, 0xd9d2c0], steppe: [0xd8d3c6, 0xc9c2b0, 0xe2ddd0], tropic: [0xf0ece2, 0xe8dcc0, 0xd6e0d8], flats: [0xe6e1d4, 0xc9d3d8, 0xe8dcc4] }[biome];
  const roofs = { florida: [0x6a6e72, 0x8a5a44], steppe: [0x7a7d80, 0x5d6a74], tropic: [0xa65a3c, 0x8c3f2c, 0x6d7c86], flats: [0x5f6366, 0x8a8f94, 0x9c6a4c] }[biome];
  const body = new THREE.InstancedMesh(new THREE.BoxGeometry(1, 1, 1).translate(0, 0.5, 0), new THREE.MeshStandardMaterial({ roughness: 0.9 }), list.length);
  const roofG = new THREE.CylinderGeometry(0.72, 0.72, 1, 3).rotateZ(Math.PI / 2).rotateX(-Math.PI / 2).translate(0, 0.36, 0).scale(1, 1 / 1.08, 0.83);
  const roof = new THREE.InstancedMesh(roofG, new THREE.MeshStandardMaterial({ roughness: 0.85 }), list.length);
  list.forEach((h, i) => {
    const y0 = -(h.x * h.x + h.z * h.z) / (2 * RE_M);
    dummy.position.set(h.x, y0, h.z); dummy.rotation.set(0, h.rot ?? 0, 0); dummy.scale.set(h.w, h.h, h.d); dummy.updateMatrix();
    body.setMatrixAt(i, dummy.matrix);
    body.setColorAt(i, col.setHex(walls[i % walls.length]));
    dummy.position.y = y0 + h.h; dummy.scale.set(h.w * 1.02, h.flat ? 0.01 : Math.min(h.w, h.d) * 0.55, h.d * 1.04); dummy.updateMatrix();
    roof.setMatrixAt(i, dummy.matrix);
    roof.setColorAt(i, col.setHex(roofs[(i * 7) % roofs.length]));
  });
  for (const m of [body, roof]) { m.castShadow = m.receiveShadow = true; m.userData.curved = true; }
  return [body, roof];
}
// שכונה: רחובות ברשת ובתים לאורכם
function village(feat, cx, cz, nx, nz, opts = {}) {
  const sp = opts.spacing ?? 28, rot = opts.rot ?? 0, r = rng(Math.round(cx * 13 + cz * 7));
  const cs = Math.cos(rot), sn = Math.sin(rot);
  const tr = (u, v) => [cx + u * cs - v * sn, cz + u * sn + v * cs];
  const W = nx * sp, D = nz * sp;
  for (let j = 0; j <= nz; j += 2) feat.roads.push({ pts: [tr(-W / 2 - 10, -D / 2 + j * sp), tr(W / 2 + 10, -D / 2 + j * sp)], w: 7, color: '#6a6862' });
  feat.roads.push({ pts: [tr(-W / 2, -D / 2), tr(-W / 2, D / 2)], w: 7, color: '#6a6862' });
  feat.roads.push({ pts: [tr(W / 2, -D / 2), tr(W / 2, D / 2)], w: 7, color: '#6a6862' });
  for (let i = 0; i < nx; i++) for (let j = 0; j < nz; j++) {
    if (j % 2 === 0) continue; // שורת רחוב
    for (const side of [-1, 1]) {
      if (r() < (opts.gaps ?? 0.15)) continue;
      const [x, z] = tr(-W / 2 + (i + 0.5) * sp, -D / 2 + j * sp + side * sp * 0.33);
      const big = opts.blocks;
      feat.houses.push({ x, z, rot: rot + (r() - 0.5) * 0.1, w: big ? 50 : 9 + r() * 6, d: big ? 13 : 8 + r() * 5, h: big ? 15 : 3.2 + (r() < 0.3 ? 3 : 0), flat: big || opts.flat });
      if (!big && r() < 0.6) feat.trees.push({ kind: opts.tree ?? 'broad', x: x + (r() - 0.5) * 12, z: z + side * 9, s: 0.6 + r() * 0.5, c: r() });
    }
  }
}

// ======================= פריסת כל אתר =======================
const LAYOUT = {
  // SLC-40: מגדל גישת צוות קבוע (נבנה 2023), זרוע מזקפת (TE) שנסוגה לפני השיגור, 4 תרני ברקים,
  // מגדל מים להשתקת רעש, מכלי חמצן נוזלי וקרוסין, ומבנה ההרכבה האופקי (HIF) מדרום לכן.
  falcon9(root, f) {
    f.slabs.push({ x: 0, z: 0, w: 120, d: 110, color: '#b3b0a6' }, { x: 0, z: 330, w: 170, d: 120, color: '#a4a198' });
    f.slabs.push({ x: -140, z: 60, w: 70, d: 60, color: '#a19e95' }, { x: 120, z: -70, w: 60, d: 50, color: '#a19e95' });
    f.roads.push({ pts: [[0, 50], [0, 280]], w: 14, color: '#8f8b82' });                 // מסילות ה-TE
    f.roads.push({ pts: [[-60, 330], [-900, 380], [-2400, 260], [-4200, 520]], w: 9, stripe: true });
    f.roads.push({ pts: [[0, 0], [-220, -180], [-260, -900], [-120, -2200], [300, -4400]], w: 8, stripe: true });
    f.roads.push({ pts: [[-180, -40], [-180, 120], [180, 120], [180, -150], [-180, -150], [-180, -40]], w: 6 });
    // לוח הכן ותעלת הלהבות (צפונה)
    root.add(boxM(26, 5, 22, concrete(), 0, 0, 0));
    root.add(boxM(9, 0.4, 40, pit(), 0, 0.02, -30));
    root.add(boxM(13, 3, 4, darkConcrete(), 0, 0, -12));
    // זרוע המזקפת: סריג לאורך הרקטה, נטוי 3° הרחק ממנה
    const te = new THREE.Group();
    te.add(mesh(latticeGeo(3.2, 3.2, 68, 4, 0.35), darkSteel()));
    te.position.set(-5.2, 5, 0); te.rotation.z = 0.05;
    root.add(te);
    // מגדל גישת הצוות (~80 מ') עם זרוע גישה מסובבת הצידה
    const cat = new THREE.Group();
    cat.add(mesh(latticeGeo(9, 9, 80, 5, 0.5), steelGray()));
    cat.add(boxM(10, 6, 10, steelGray(), 0, 80, 0));
    const arm = boxM(3, 3, 16, white(), 0, 62, 9);
    cat.add(arm);
    cat.position.set(-17, 5, -15);
    root.add(cat);
    // ארבעה תרני ברקים וחוטים ביניהם
    const mastP = [[-75, -75], [75, -75], [75, 75], [-75, 75]];
    for (const [x, z] of mastP) root.add(cylM(1.1, 90, steelGray(), x, 0, z, 8, 0.45));
    root.add(wires(mastP.map(([x, z]) => [x, 90, z]), 8));
    // מגדל מים
    root.add(cylM(1.6, 52, steelGray(), 150, 0, 110, 10), mesh(new THREE.SphereGeometry(9, 20, 14), white(), 150, 58, 110));
    // מכלים
    root.add(mesh(new THREE.SphereGeometry(9, 24, 16), white(), -140, 10, 60));
    for (const z of [42, 78]) { const t = cylM(3, 24, white(), -110, 0, z, 16); t.rotation.z = Math.PI / 2; t.position.y = 3.5; root.add(t); }
    // מבנה ההרכבה (HIF)
    const hif = building(90, 30, 60, '#e8e8e4', 0x8c9095); hif.position.set(0, 0, 330); root.add(hif);
    root.add(at(building(36, 12, 22, '#d9d6cc'), 120, -70));
    // אזור התעשייה של הכף (מבנים טכניים) ובתי שמירה לאורך הדרכים
    const r = rng(40);
    for (let i = 0; i < 18; i++) f.houses.push({ x: -1900 + r() * 700, z: 600 + r() * 600, w: 20 + r() * 40, d: 15 + r() * 25, h: 6 + r() * 10, flat: true, rot: 0.1 });
    for (let i = 0; i < 6; i++) f.houses.push({ x: -800 - i * 250, z: 360 + r() * 40, w: 12, d: 10, h: 4, flat: true });
    f.water.push({ x: -2600, z: -1400, rx: 900, rz: 400, rot: 0.6, color: '#3f7f8c' });   // בריכות ואגמונים בכף
    f.keep.push({ x: 0, z: 330, r: 120 }, { x: -140, z: 60, r: 40 }, { x: 150, z: 110, r: 20 });
    return { clear: 140, plumeDir: [0, -1] };
  },

  // LC-39A בעידן אפולו: גבעת הכן (כ-12 מ'), המשגר הנייד (ML) ועליו מגדל הטבור (LUT) האדום בגובה 116 מ'
  // עם עגורן ראש־פטיש; דרך הזחלנים מערבה אל בניין ההרכבה האנכי (VAB) שבמרחק 5.5 ק"מ.
  saturn5(root, f) {
    f.slabs.push({ x: 0, z: 0, w: 210, d: 210, round: true, color: '#9fa18f' });
    // דרך הזחלן: שני נתיבים צמודים, לכיוון 250°
    const a = (250 - 90) * Math.PI / 180, dir = [Math.cos(a), Math.sin(a)];
    const crawler = (s) => ({ pts: [[dir[0] * 120 + s * dir[1], dir[1] * 120 - s * dir[0]], [dir[0] * 5400 + s * dir[1], dir[1] * 5400 - s * dir[0]]], w: 12, color: '#c9c3b2' });
    f.roads.push(crawler(-10), crawler(10));
    f.roads.push({ pts: [[0, 110], [300, 900], [600, 3000], [700, 4800]], w: 8, stripe: true });
    // הגבעה: חרוט קטום בעל 8 צלעות
    root.add(mesh(new THREE.CylinderGeometry(58, 95, 12, 8).translate(0, 6, 0), mat('mound', { color: 0x9a9a8a, roughness: 1 })));
    root.add(boxM(16, 0.5, 120, pit(), 0, 11.8, 0)); // תעלת הלהבות חוצה את הגבעה
    // המשגר הנייד: משטח 49×41 מ' על 6 רגליים
    root.add(boxM(49, 7.6, 41, mat('mlgray', { color: 0x7d8086, roughness: 0.8, metalness: 0.3 }), -12, 12, 0));
    for (const [x, z] of [[-30, -15], [-30, 15], [6, -15], [6, 15]]) root.add(boxM(3, 4, 3, darkSteel(), x, 11.5, z));
    // מגדל הטבור האדום
    const lut = new THREE.Group();
    lut.add(mesh(latticeGeo(12, 12, 116, 6, 0.7), apolloRed()));
    const crane = boxM(32, 3, 3, apolloRed(), 4, 116, 0); lut.add(crane);
    lut.add(boxM(3, 8, 3, apolloRed(), 0, 116, 0));
    // זרועות הנדנדה, פתוחות בזמן השיגור
    for (const [y, len] of [[18, 14], [32, 14], [50, 14], [68, 14], [83, 14], [97, 12]]) {
      const arm = new THREE.Group(); arm.add(boxM(len, 2.2, 2.6, apolloRed(), len / 2, 0, 0));
      arm.position.set(6, y, -6); arm.rotation.y = 1.25; lut.add(arm);
    }
    lut.position.set(-20, 19.6, 0);
    root.add(lut);
    // מגדל מים וכדורי מימן/חמצן בשולי הגבעה
    root.add(mesh(new THREE.SphereGeometry(10.5, 24, 16), white(), -150, 11, 120), mesh(new THREE.SphereGeometry(10.5, 24, 16), white(), 140, 11, -130));
    root.add(cylM(1.6, 60, steelGray(), 170, 0, 140, 10), mesh(new THREE.SphereGeometry(9, 20, 14), white(), 170, 66, 140));
    // בניין ההרכבה האנכי: 218×158 מ', גובה 160 מ'
    const vab = building(158, 160, 218, '#d8d8d2', 0x8f9396, true);
    vab.position.set(dir[0] * 5500, 0, dir[1] * 5500); vab.rotation.y = -a;
    root.add(vab);
    root.add(at(building(150, 25, 90, '#cfcfc6'), dir[0] * 5500 - 60, dir[1] * 5500 + 210));
    f.water.push({ x: -1500, z: 900, rx: 1300, rz: 450, rot: 1.2, color: '#4a7d84' }, { x: -2300, z: -1700, rx: 700, rz: 1500, rot: 0.3, color: '#4a7d84' });
    f.keep.push({ x: dir[0] * 5500, z: dir[1] * 5500, r: 260 });
    return { clear: 150, plumeDir: [0, -1] };
  },

  // אתר 31/6 בבייקונור: הרקטה תלויה מעל בור להבות על ארבע זרועות תמיכה ("צבעוני") שנפתחות בהמראה,
  // שני תרני כבלים, מסילת רכבת שמביאה את הרקטה שוכבת, ובערבה מבני הרכבה ושיכון במרחק כמה ק"מ.
  soyuz(root, f) {
    f.slabs.push({ x: 0, z: 0, w: 90, d: 70, color: '#a7a399' }, { x: 0, z: -95, w: 60, d: 100, color: '#4a4540' });
    f.roads.push({ pts: [[0, 35], [0, 1600], [-200, 2600]], w: 6, rail: true, color: '#8a8070' });
    f.roads.push({ pts: [[40, 30], [300, 400], [1200, 900], [2600, 1800]], w: 8 });
    f.roads.push({ pts: [[-40, 30], [-700, 500], [-2600, 1700], [-3600, 2300]], w: 8, stripe: true });
    // משטח הכן עם פתח לבור, ובור הלהבות פתוח צפונה
    root.add(boxM(40, 2, 14, concrete(), 0, 0, 13), boxM(40, 2, 14, concrete(), 0, 0, -13), boxM(13, 2, 12, concrete(), 13.5, 0, 0), boxM(13, 2, 12, concrete(), -13.5, 0, 0));
    root.add(boxM(14, 0.3, 12, pit(), 0, -6, 0));
    root.add(boxM(30, 0.3, 70, pit(), 0, -0.2, -55));
    // ארבע זרועות התמיכה, פתוחות מעט ("פרח שנפתח")
    for (let i = 0; i < 4; i++) {
      const ang = i * Math.PI / 2 + Math.PI / 4;
      const arm = new THREE.Group();
      arm.add(mesh(latticeGeo(1.6, 1.6, 16, 2, 0.25), steelGray()));
      arm.position.set(Math.cos(ang) * 6.5, 2, Math.sin(ang) * 6.5);
      arm.rotation.set(Math.sin(ang) * 0.32, 0, -Math.cos(ang) * 0.32);
      root.add(arm);
    }
    // תרני כבלים (טבור) ומגדל שירות מקופל הצידה
    for (const s of [-1, 1]) { const m = mesh(latticeGeo(2.4, 2.4, 34, 3, 0.3), steelGray()); m.position.set(s * 15, 2, 4); m.rotation.z = s * 0.12; root.add(m); }
    const gantry = mesh(latticeGeo(8, 8, 50, 5, 0.45), mat('ygantry', { color: 0x8f8a7a, roughness: 0.7, metalness: 0.4 }));
    gantry.rotation.z = Math.PI / 2; gantry.position.set(-190, 4, 30); root.add(gantry);
    // תרני תאורה וברקים
    for (const [x, z] of [[-60, -60], [60, -60], [60, 60], [-60, 60]]) root.add(cylM(0.9, 70, steelGray(), x, 0, z, 8, 0.4));
    // מבנה ההרכבה (MIK) ומבני תמיכה
    const mik = building(60, 30, 200, '#cfc8b8', 0x7a7d80); mik.position.set(-150, 0, 1700); root.add(mik);
    for (const [x, z, w, d, h] of [[220, 300, 40, 25, 10], [-260, 260, 30, 30, 8], [350, -400, 50, 20, 7], [-420, -350, 25, 25, 12]]) root.add(at(building(w, h, d, '#d4cdbd'), x, z));
    // שיכון: בלוקים סובייטיים בני חמש קומות
    village(f, -2900, 2100, 6, 9, { spacing: 70, rot: 0.55, blocks: true, gaps: 0.25 });
    village(f, 2300, 1500, 8, 6, { spacing: 30, rot: -0.3, gaps: 0.35, tree: 'broad' });
    f.water.push({ x: 3200, z: -2600, rx: 500, rz: 200, rot: 0.2, color: '#7f8f84' });
    f.keep.push({ x: -150, z: 1700, r: 130 }, { x: 0, z: 70, r: 30 });
    return { clear: 120, plumeDir: [0, -1] };
  },

  // LC-101 בוון־צ'אנג: משגר נייד שמגיע במסילה מבניין ההרכבה (2.8 ק"מ), מגדל טבור קבוע,
  // ארבעה מגדלי ברקים גבוהים; ג'ונגל טרופי, דקלי קוקוס, כפרים וחוף מזרחה.
  cz5(root, f) {
    f.slabs.push({ x: 0, z: 0, w: 150, d: 130, color: '#b0ada4' });
    const a = (315 - 90) * Math.PI / 180, dir = [Math.cos(a), Math.sin(a)];
    f.roads.push({ pts: [[0, 0], [dir[0] * 2800, dir[1] * 2800]], w: 16, color: '#9a968c', rail: true });
    f.roads.push({ pts: [[60, 60], [400, 900], [300, 2500], [-200, 4500]], w: 9, stripe: true });
    f.roads.push({ pts: [[-2400, 1600], [-1000, 1800], [400, 2000], [1100, 2600]], w: 7 });
    root.add(boxM(18, 0.4, 70, pit(), 0, 0.02, 45));
    // המשגר הנייד
    root.add(boxM(30, 8, 26, darkSteel(), -4, 0, 0));
    // מגדל הטבור על המשגר
    const tower = mesh(latticeGeo(9, 9, 82, 5, 0.5), mat('cztower', { color: 0xd8d8d4, roughness: 0.7, metalness: 0.3 }));
    tower.position.set(-17, 8, 0); root.add(tower);
    root.add(boxM(10, 6, 10, mat('czred', { color: 0xb33a2e, roughness: 0.7 }), -17, 90, 0));
    for (const y of [30, 46, 60]) { const arm = boxM(10, 1.6, 2, white(), -9, 8 + y, 4); arm.rotation.y = 0.9; root.add(arm); }
    // מבנה השירות הקבוע (סגור, לבן) נסוג מהרקטה
    root.add(at(building(22, 96, 22, '#ecebe6', 0x9aa0a6, true), 38, -10));
    // ארבעה מגדלי ברקים, כ-130 מ'
    const mastP = [[-70, -65], [70, -65], [70, 65], [-70, 65]];
    for (const [x, z] of mastP) root.add(mesh(latticeGeo(3, 3, 128, 6, 0.3), steelGray(), x, 0, z));
    root.add(wires(mastP.map(([x, z]) => [x, 128, z]), 10));
    // בניין ההרכבה האנכי בקצה המסילה (99 מ')
    const vab = building(70, 99, 100, '#e9e8e2', 0x8c9095); vab.position.set(dir[0] * 2800, 0, dir[1] * 2800); vab.rotation.y = -a; root.add(vab);
    for (const [x, z] of [[-300, 300], [350, -260], [-420, -200]]) root.add(at(building(40, 12, 25, '#e2e0d8', 0x6b7a86), x, z));
    // כפרים
    village(f, -2600, 1900, 9, 7, { spacing: 26, rot: 0.2, tree: 'palm' });
    village(f, 600, 3300, 12, 5, { spacing: 24, rot: -0.15, tree: 'palm' });
    village(f, -3400, -900, 6, 7, { spacing: 26, rot: 0.9, tree: 'palm' });
    f.water.push({ x: -1600, z: -2300, rx: 450, rz: 260, rot: 0.4, color: '#4f7f6f' }, { x: 1700, z: 1500, rx: 200, rz: 120, color: '#4f7f6f' });
    f.keep.push({ x: dir[0] * 2800, z: dir[1] * 2800, r: 150 });
    return { clear: 150, plumeDir: [0, 1] };
  },

  // סטארבייס, כן 2: שולחן השיגור (OLM) עם מגן מים ותעלת להבות, מגדל השיגור (146 מ')
  // עם "המקלות" (chopsticks), חוות המכלים, החוף ממזרח, שטחי גאות, וכפר בוקה צ'יקה כ-3 ק"מ מערבה.
  starship(root, f) {
    f.slabs.push({ x: 0, z: 0, w: 140, d: 120, color: '#a8a59c' }, { x: -210, z: 40, w: 150, d: 110, color: '#9f9c92' });
    f.roads.push({ pts: [[-4800, 120], [-2500, 80], [-300, 60], [300, 40], [700, 30]], w: 9, stripe: true }); // כביש 4 עד החוף
    f.roads.push({ pts: [[-150, -40], [-150, -400], [200, -700]], w: 7 });
    root.add(boxM(16, 0.4, 60, pit(), 0, 0.02, -38));
    // שולחן השיגור: טבעת עבה על שש רגליים, ומעליו מלחציים
    const base = 16;
    for (let i = 0; i < 6; i++) { const t = i * Math.PI / 3 + Math.PI / 6; root.add(cylM(1.6, base - 4, steelGray(), Math.cos(t) * 9.5, 0, Math.sin(t) * 9.5, 12)); }
    const ring = new THREE.Mesh(new THREE.RingGeometry(5.2, 11.5, 48).rotateX(-Math.PI / 2), darkSteel());
    root.add(cylM(11.5, 4, darkSteel(), 0, base - 4, 0, 48));
    ring.position.y = base + 0.01; root.add(ring);
    root.add(mesh(new THREE.CylinderGeometry(13, 13, 0.5, 48), mat('deluge', { color: 0x4a4d52, metalness: 0.7, roughness: 0.5 }), 0, 0.3, 0));
    // מגדל השיגור: מודולי פלדה כהים, 146 מ'
    const tw = new THREE.Group();
    tw.add(mesh(latticeGeo(11, 11, 140, 9, 0.9), darkSteel()));
    tw.add(boxM(13, 6, 13, darkSteel(), 0, 140, 0));
    tw.add(cylM(0.5, 12, steelGray(), 0, 146, 0, 6));
    // "המקלות": שתי זרועות ארוכות, פתוחות ומורמות
    for (const s of [-1, 1]) {
      const arm = new THREE.Group();
      arm.add(boxM(30, 3.2, 2.2, shinySteel(), 15, 0, 0));
      arm.position.set(5.5, 128, s * 6.5); arm.rotation.y = -s * 0.55;
      tw.add(arm);
    }
    tw.add(boxM(13, 6, 13, darkSteel(), 0, 125, 0));
    tw.position.set(-24, 0, 0);
    root.add(tw);
    // חוות המכלים: מכלים אנכיים ואופקיים
    for (let i = 0; i < 8; i++) root.add(cylM(4.5, 38, shinySteel(), -250 + (i % 4) * 14, 0, 10 + Math.floor(i / 4) * 16, 24));
    for (let i = 0; i < 6; i++) { const t = cylM(2.6, 30, shinySteel(), -190, 0, 50 + i * 8, 16); t.rotation.z = Math.PI / 2; t.position.y = 3; root.add(t); }
    root.add(at(building(30, 14, 20, '#d0d2d4', 0x55585c), -150, -110));
    // כפר בוקה צ'יקה (סטארבייס): בתים, ומפעלי הייצור הגבוהים
    village(f, -3050, 220, 7, 5, { spacing: 30, tree: 'palm', gaps: 0.25 });
    for (const [x, z, w, d, h] of [[-2350, -150, 70, 60, 85], [-2450, -150, 70, 60, 85], [-2600, -320, 260, 110, 30], [-2200, -350, 60, 60, 45]]) root.add(at(building(w, h, d, '#dfe2e4', 0x6d7276), x, z));
    // שטחי גאות ובוץ מצפון וממערב, ונהר הריו גרנדה בדרום
    f.water.push({ x: -1200, z: -1700, rx: 1700, rz: 700, rot: 0.1, color: '#8da7a0' }, { x: 300, z: -2600, rx: 900, rz: 500, rot: -0.3, color: '#7d9c98' }, { x: -1600, z: 2900, rx: 2600, rz: 260, rot: -0.05, color: '#5d8a8a' });
    f.keep.push({ x: -210, z: 40, r: 80 }, { x: -2450, z: -250, r: 220 });
    return { clear: 130, plumeDir: [0, -1] };
  },
};

// חוטי ברקים: קשתות שמשתלשלות בין ראשי התרנים
function wires(tops, sag) {
  const pts = [];
  for (let i = 0; i < tops.length; i++) {
    const a = tops[i], b = tops[(i + 1) % tops.length];
    for (let k = 0; k <= 16; k++) {
      const t = k / 16;
      pts.push(new THREE.Vector3(a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t - sag * 4 * t * (1 - t), a[2] + (b[2] - a[2]) * t));
    }
  }
  const l = new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts), new THREE.LineBasicMaterial({ color: 0x30343a }));
  return l;
}

// מיקום ההר במערכת כדור הארץ (ECEF ב-three): X מזרח, Y למעלה, Z דרום
export function placeSite(root, lat, lon) {
  const d = ecefDir(lat, lon);
  const up = new THREE.Vector3(d[0], d[2], -d[1]);
  const lo = lon * Math.PI / 180, la = lat * Math.PI / 180;
  const eE = [-Math.sin(lo), Math.cos(lo), 0];
  const nE = [-Math.sin(la) * Math.cos(lo), -Math.sin(la) * Math.sin(lo), Math.cos(la)];
  const east = new THREE.Vector3(eE[0], eE[2], -eE[1]);
  const south = new THREE.Vector3(-nE[0], -nE[2], nE[1]);
  root.quaternion.setFromRotationMatrix(new THREE.Matrix4().makeBasis(east, up, south));
  root.position.copy(up).multiplyScalar(RE_KM + GROUND_M / 1000);
  root.scale.setScalar(0.001);
  return { up, east, south };
}

// ======================= שמיים =======================
// כיפה סביב המצלמה: כחול בזנית, אובך בהיר באופק, הילת שמש. נמוגה עם הגובה (גובה סקאלה ~7.5 ק"מ).
// הצבעים כבר במרחב התצוגה (sRGB), כדי שהאופק יתמזג בדיוק בערפל של הקרקע.
export const HAZE_SRGB = new THREE.Color().setRGB(0.70, 0.79, 0.88, THREE.SRGBColorSpace);
export function makeSky() {
  const m = new THREE.ShaderMaterial({
    uniforms: { sunDir: { value: new THREE.Vector3(1, 0, 0) }, up: { value: new THREE.Vector3(0, 1, 0) }, alt: { value: 0 }, haze: { value: new THREE.Vector3(0.70, 0.79, 0.88) } },
    vertexShader: `
      #include <common>
      #include <logdepthbuf_pars_vertex>
      varying vec3 vP;
      void main(){ vec4 wp = modelMatrix * vec4(position,1.0); vP = wp.xyz; gl_Position = projectionMatrix * viewMatrix * wp;
        #include <logdepthbuf_vertex>
      }`,
    fragmentShader: `
      #include <common>
      #include <logdepthbuf_pars_fragment>
      uniform vec3 sunDir; uniform vec3 up; uniform float alt; uniform vec3 haze; varying vec3 vP;
      void main(){
        #include <logdepthbuf_fragment>
        vec3 V = normalize(vP - cameraPosition);
        float dip = sqrt(max(2.0 * alt / 6378.0, 0.0));
        float h = dot(V, up) + dip;
        float dens = exp(-alt / 7.5);
        float vis = exp(-alt / 22.0);  // השמיים נשארים כחולים עמוקים עד עשרות ק"מ
        float t = pow(clamp(h, 0.0, 1.0), 0.42);
        vec3 zen = mix(vec3(0.03, 0.07, 0.22), vec3(0.20, 0.42, 0.80), exp(-alt / 11.0));
        vec3 col = mix(haze, zen, t);
        float mu = max(dot(V, sunDir), 0.0);
        col += vec3(1.0, 0.92, 0.75) * (0.22 * pow(mu, 6.0) + 0.5 * pow(mu, 200.0)) * dens;
        float sunUp = smoothstep(-0.12, 0.08, dot(up, sunDir));
        col *= mix(0.06, 1.0, sunUp);
        float band = exp(-abs(h) * 14.0) * exp(-alt / 45.0);
        float a = clamp(max(vis * 1.35 * (0.8 + 0.2 * (1.0 - t)), band), 0.0, 1.0);
        gl_FragColor = vec4(col, a);
      }`,
    transparent: true, depthWrite: false, side: THREE.BackSide, fog: false,
  });
  const sky = new THREE.Mesh(new THREE.SphereGeometry(1, 64, 32), m);
  sky.scale.setScalar(4e5);
  sky.renderOrder = -10;
  sky.frustumCulled = false;
  return sky;
}

// ======================= עשן ההמראה =======================
let _puffTex;
function puffTex() {
  if (_puffTex) return _puffTex;
  const c = document.createElement('canvas'); c.width = c.height = 128;
  const g = c.getContext('2d');
  const r = rng(5);
  for (let i = 0; i < 14; i++) {
    const x = 34 + r() * 60, y = 34 + r() * 60, rr = 18 + r() * 26;
    const gr = g.createRadialGradient(x, y, 0, x, y, rr);
    gr.addColorStop(0, 'rgba(255,255,255,0.55)'); gr.addColorStop(1, 'rgba(255,255,255,0)');
    g.fillStyle = gr; g.fillRect(0, 0, 128, 128);
  }
  _puffTex = new THREE.CanvasTexture(c); _puffTex.colorSpace = THREE.SRGBColorSpace;
  return _puffTex;
}
// מערכת חלקיקי עשן במערכת המקומית של האתר (מטרים)
export class Smoke {
  constructor(root, dir, baseH) {
    this.root = root; this.dir = dir; this.baseH = baseH; this.list = []; this.acc = 0; this.r = rng(77);
  }
  update(dt, t, rocketAlt, thrusting) {
    const r = this.r;
    // פולטים עשן כל עוד הסילון נוגע בקרקע (עד ~150 מ')
    if (thrusting && rocketAlt < 160 && dt > 0) {
      this.acc += dt * 26;
      while (this.acc > 1 && this.list.length < 260) {
        this.acc--;
        const m = new THREE.SpriteMaterial({ map: puffTex(), color: 0xf2efe8, transparent: true, depthWrite: false, opacity: 0.0, fog: true });
        const s = new THREE.Sprite(m);
        const side = r() < 0.75;
        const sp = 25 + r() * 45;
        const vx = side ? this.dir[0] * sp + (r() - 0.5) * 18 : (r() - 0.5) * 30;
        const vz = side ? this.dir[1] * sp + (r() - 0.5) * 18 : (r() - 0.5) * 30;
        s.position.set(this.dir[0] * 10, Math.max(3, rocketAlt * 0.3 + this.baseH * 0.4), this.dir[1] * 10);
        s.userData = { v: new THREE.Vector3(vx, 3 + r() * 9, vz), age: 0, life: 30 + r() * 25, size: 12 + r() * 10, rot: r() * 6 };
        m.rotation = s.userData.rot;
        this.root.add(s); this.list.push(s);
      }
    }
    for (const s of this.list) {
      const u = s.userData;
      u.age += dt;
      s.position.addScaledVector(u.v, dt);
      u.v.multiplyScalar(Math.exp(-dt * 0.25));
      u.v.y += dt * 0.6;
      const k = u.age / u.life;
      s.scale.setScalar(u.size * (1 + u.age * 0.9));
      s.material.opacity = Math.min(1, u.age * 2) * 0.8 * (1 - k);
      s.visible = k < 1;
    }
  }
  clear() { for (const s of this.list) { this.root.remove(s); s.material.dispose(); } this.list = []; }
}
