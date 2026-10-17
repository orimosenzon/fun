// עולם ההתנגשויות: קרקע אנליטית, משטחים כרשתות משולשים (עם רשת מרחבית), וקופסאות מסובבות.
// שתי שאילתות: raycast (לגלגלים ולרצפה) ו-penetrate (לנקודות בגוף הרכב).

import { V3 } from './math.js';

// ---------- קרקע ----------
// בתוך רדיוס 700 מ' הכול שטוח (אתר הניסויים). מחוץ לזה גבעות נמוכות.
export function groundHeight(x, z) {
  const r = Math.hypot(x, z);
  if (r < 700) return 0;
  const t = Math.min(1, (r - 700) / 900);
  const s = t * t * (3 - 2 * t);
  const n = Math.sin(x * 0.0041 + 1.3) * Math.cos(z * 0.0037 - 0.7) * 0.6
    + Math.sin(x * 0.011 + z * 0.008) * 0.25 + 0.5;
  return s * (25 + 70 * n);
}
function groundNormal(x, z, out) {
  const e = 0.5;
  const hx = groundHeight(x + e, z) - groundHeight(x - e, z);
  const hz = groundHeight(x, z + e) - groundHeight(x, z - e);
  return out.set(-hx / (2 * e), 1, -hz / (2 * e)).norm();
}

// ---------- רשת משולשים ----------
const CELL = 2.5;
export class TriMesh {
  // positions: Float32Array (x,y,z לכל קודקוד), normals: כנ"ל (נורמלים חלקים), indices: Uint32Array
  constructor(positions, normals, indices, { mu = 1, name = '', thickness = 0.9 } = {}) {
    this.p = positions; this.n = normals; this.idx = indices; this.mu = mu; this.name = name;
    // נקודה נחשבת חודרת רק עד חצי עובי הלוח, אחרת היא שייכת לפאה שמהצד השני
    this.thick = thickness * 0.5;
    this.grid = new Map();
    const nt = indices.length / 3;
    this.stamp = new Int32Array(nt);
    this.stampId = 0;
    // נורמל פאה לכל משולש
    this.fn = new Float32Array(nt * 3);
    let minX = Infinity, minY = Infinity, minZ = Infinity, maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
    for (let t = 0; t < nt; t++) {
      const a = indices[t * 3] * 3, b = indices[t * 3 + 1] * 3, c = indices[t * 3 + 2] * 3;
      const ux = positions[b] - positions[a], uy = positions[b + 1] - positions[a + 1], uz = positions[b + 2] - positions[a + 2];
      const vx = positions[c] - positions[a], vy = positions[c + 1] - positions[a + 1], vz = positions[c + 2] - positions[a + 2];
      let nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
      const l = Math.hypot(nx, ny, nz) || 1;
      this.fn[t * 3] = nx / l; this.fn[t * 3 + 1] = ny / l; this.fn[t * 3 + 2] = nz / l;
      const xs = [positions[a], positions[b], positions[c]], ys = [positions[a + 1], positions[b + 1], positions[c + 1]], zs = [positions[a + 2], positions[b + 2], positions[c + 2]];
      const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys), z0 = Math.min(...zs), z1 = Math.max(...zs);
      minX = Math.min(minX, x0); minY = Math.min(minY, y0); minZ = Math.min(minZ, z0);
      maxX = Math.max(maxX, x1); maxY = Math.max(maxY, y1); maxZ = Math.max(maxZ, z1);
      for (let i = Math.floor(x0 / CELL); i <= Math.floor(x1 / CELL); i++)
        for (let j = Math.floor(y0 / CELL); j <= Math.floor(y1 / CELL); j++)
          for (let k = Math.floor(z0 / CELL); k <= Math.floor(z1 / CELL); k++) {
            const key = cellKey(i, j, k);
            let arr = this.grid.get(key);
            if (!arr) this.grid.set(key, arr = []);
            arr.push(t);
          }
    }
    const pad = 1;
    this.box = { minX: minX - pad, minY: minY - pad, minZ: minZ - pad, maxX: maxX + pad, maxY: maxY + pad, maxZ: maxZ + pad };
  }
  inBox(x0, y0, z0, x1, y1, z1) {
    const b = this.box;
    return !(x1 < b.minX || x0 > b.maxX || y1 < b.minY || y0 > b.maxY || z1 < b.minZ || z0 > b.maxZ);
  }
  // אוסף משולשים מועמדים בתיבה
  forTris(x0, y0, z0, x1, y1, z1, fn) {
    const s = ++this.stampId;
    for (let i = Math.floor(x0 / CELL); i <= Math.floor(x1 / CELL); i++)
      for (let j = Math.floor(y0 / CELL); j <= Math.floor(y1 / CELL); j++)
        for (let k = Math.floor(z0 / CELL); k <= Math.floor(z1 / CELL); k++) {
          const arr = this.grid.get(cellKey(i, j, k));
          if (!arr) continue;
          for (let q = 0; q < arr.length; q++) {
            const t = arr[q];
            if (this.stamp[t] === s) continue;
            this.stamp[t] = s;
            fn(t);
          }
        }
  }
  raycast(o, d, maxT, hit, twoSided = false) {
    const ex = o.x + d.x * maxT, ey = o.y + d.y * maxT, ez = o.z + d.z * maxT;
    const x0 = Math.min(o.x, ex), x1 = Math.max(o.x, ex), y0 = Math.min(o.y, ey), y1 = Math.max(o.y, ey), z0 = Math.min(o.z, ez), z1 = Math.max(o.z, ez);
    if (!this.inBox(x0, y0, z0, x1, y1, z1)) return false;
    const P = this.p, I = this.idx, FN = this.fn;
    let found = false;
    this.forTris(x0, y0, z0, x1, y1, z1, (t) => {
      // רק פאות שפונות אל הקרן
      if (!twoSided && FN[t * 3] * d.x + FN[t * 3 + 1] * d.y + FN[t * 3 + 2] * d.z >= 0) return;
      const a = I[t * 3] * 3, b = I[t * 3 + 1] * 3, c = I[t * 3 + 2] * 3;
      // מולר-טרומבור
      const e1x = P[b] - P[a], e1y = P[b + 1] - P[a + 1], e1z = P[b + 2] - P[a + 2];
      const e2x = P[c] - P[a], e2y = P[c + 1] - P[a + 1], e2z = P[c + 2] - P[a + 2];
      const px = d.y * e2z - d.z * e2y, py = d.z * e2x - d.x * e2z, pz = d.x * e2y - d.y * e2x;
      const det = e1x * px + e1y * py + e1z * pz;
      if (Math.abs(det) < 1e-12) return;
      const inv = 1 / det;
      const tx = o.x - P[a], ty = o.y - P[a + 1], tz = o.z - P[a + 2];
      const u = (tx * px + ty * py + tz * pz) * inv;
      if (u < -1e-6 || u > 1 + 1e-6) return;
      const qx = ty * e1z - tz * e1y, qy = tz * e1x - tx * e1z, qz = tx * e1y - ty * e1x;
      const v = (d.x * qx + d.y * qy + d.z * qz) * inv;
      if (v < -1e-6 || u + v > 1 + 1e-6) return;
      const tt = (e2x * qx + e2y * qy + e2z * qz) * inv;
      if (tt < 0 || tt > hit.t) return;
      hit.t = tt;
      // נורמל חלק: אינטרפולציה בריצנטרית של נורמלי הקודקודים
      const N = this.n, w = 1 - u - v;
      hit.n.set(N[a] * w + N[b] * u + N[c] * v, N[a + 1] * w + N[b + 1] * u + N[c + 1] * v, N[a + 2] * w + N[b + 2] * u + N[c + 2] * v).norm();
      hit.mu = this.mu; hit.kind = this.name;
      found = true;
    });
    return found;
  }
  // חדירה של נקודה: נקודה מאחורי פאה, בעובי עד thick
  penetrate(p, res, thick = this.thick) {
    if (!this.inBox(p.x - thick, p.y - thick, p.z - thick, p.x + thick, p.y + thick, p.z + thick)) return false;
    const P = this.p, I = this.idx, FN = this.fn;
    let found = false;
    this.forTris(p.x - thick, p.y - thick, p.z - thick, p.x + thick, p.y + thick, p.z + thick, (t) => {
      const nx = FN[t * 3], ny = FN[t * 3 + 1], nz = FN[t * 3 + 2];
      const a = I[t * 3] * 3;
      const dd = (p.x - P[a]) * nx + (p.y - P[a + 1]) * ny + (p.z - P[a + 2]) * nz;
      if (dd >= 0 || dd < -thick) return;
      // האם ההטלה של הנקודה נמצאת בתוך המשולש
      const qx = p.x - dd * nx, qy = p.y - dd * ny, qz = p.z - dd * nz;
      if (!pointInTri(P, I, t, qx, qy, qz, nx, ny, nz)) return;
      const depth = -dd;
      if (depth > res.depth) { res.depth = depth; res.n.set(nx, ny, nz); res.mu = this.mu; found = true; }
    });
    return found;
  }
}
function cellKey(i, j, k) { return ((i + 2048) * 4096 + (j + 2048)) * 4096 + (k + 2048); }
function pointInTri(P, I, t, x, y, z, nx, ny, nz) {
  for (let e = 0; e < 3; e++) {
    const a = I[t * 3 + e] * 3, b = I[t * 3 + (e + 1) % 3] * 3;
    const ex = P[b] - P[a], ey = P[b + 1] - P[a + 1], ez = P[b + 2] - P[a + 2];
    const px = x - P[a], py = y - P[a + 1], pz = z - P[a + 2];
    const cx = ey * pz - ez * py, cy = ez * px - ex * pz, cz = ex * py - ey * px;
    if (cx * nx + cy * ny + cz * nz < -1e-6) return false;
  }
  return true;
}

// ---------- קופסה מסובבת (רכבים חונים, מכשולים) ----------
export class OBB {
  constructor({ center, half, yaw = 0, mu = 0.6, name = 'box' }) {
    this.c = new V3(...center); this.h = new V3(...half); this.yaw = yaw; this.mu = mu; this.name = name;
    this.cs = Math.cos(yaw); this.sn = Math.sin(yaw);
    this.r = Math.hypot(half[0], half[1], half[2]);
  }
  toLocal(p, out) {
    const dx = p.x - this.c.x, dy = p.y - this.c.y, dz = p.z - this.c.z;
    return out.set(this.cs * dx - this.sn * dz, dy, this.sn * dx + this.cs * dz);
  }
  dirToLocal(d, out) { return out.set(this.cs * d.x - this.sn * d.z, d.y, this.sn * d.x + this.cs * d.z); }
  dirToWorld(d, out) { return out.set(this.cs * d.x + this.sn * d.z, d.y, -this.sn * d.x + this.cs * d.z); }
  raycast(o, d, maxT, hit) {
    const lo = this.toLocal(o, _a), ld = this.dirToLocal(d, _b);
    let tmin = 0, tmax = Math.min(maxT, hit.t), axis = -1, sign = 0;
    const H = [this.h.x, this.h.y, this.h.z], O = [lo.x, lo.y, lo.z], D = [ld.x, ld.y, ld.z];
    for (let i = 0; i < 3; i++) {
      if (Math.abs(D[i]) < 1e-9) { if (O[i] < -H[i] || O[i] > H[i]) return false; continue; }
      let t1 = (-H[i] - O[i]) / D[i], t2 = (H[i] - O[i]) / D[i], s = -1;
      if (t1 > t2) { const tmp = t1; t1 = t2; t2 = tmp; s = 1; }
      if (t1 > tmin) { tmin = t1; axis = i; sign = s; }
      if (t2 < tmax) tmax = t2;
      if (tmin > tmax) return false;
    }
    if (axis < 0) return false; // המקור בתוך הקופסה
    hit.t = tmin;
    const n = _c.set(0, 0, 0); if (axis === 0) n.x = sign; else if (axis === 1) n.y = sign; else n.z = sign;
    this.dirToWorld(n, hit.n); hit.mu = this.mu; hit.kind = this.name;
    return true;
  }
  penetrate(p, res) {
    const l = this.toLocal(p, _a);
    const dx = this.h.x - Math.abs(l.x), dy = this.h.y - Math.abs(l.y), dz = this.h.z - Math.abs(l.z);
    if (dx <= 0 || dy <= 0 || dz <= 0) return false;
    const n = _c.set(0, 0, 0); let depth;
    if (dx < dy && dx < dz) { depth = dx; n.x = Math.sign(l.x); }
    else if (dy < dz) { depth = dy; n.y = Math.sign(l.y) || 1; }
    else { depth = dz; n.z = Math.sign(l.z); }
    if (depth > res.depth) { res.depth = depth; this.dirToWorld(n, res.n); res.mu = this.mu; return true; }
    return false;
  }
}
const _a = new V3(), _b = new V3(), _c = new V3();

// ---------- העולם ----------
export class World {
  constructor() { this.meshes = []; this.boxes = []; }
  addMesh(m) { this.meshes.push(m); return m; }
  addBox(b) { this.boxes.push(b); return b; }
  // מחזיר את הפגיעה הקרובה ביותר. hit = {t, n, mu, kind}
  raycast(o, d, maxT, hit) {
    hit.t = maxT; hit.kind = null; let found = false;
    // קרקע: חיתוך עם מישור הגובה המקומי ושני תיקונים
    if (d.y < -1e-4) {
      let gy = groundHeight(o.x, o.z), t = (gy - o.y) / d.y;
      for (let i = 0; i < 2 && t > 0 && t < maxT * 2; i++) {
        gy = groundHeight(o.x + d.x * t, o.z + d.z * t);
        t = (gy - o.y) / d.y;
      }
      if (t >= 0 && t < hit.t) {
        hit.t = t; groundNormal(o.x + d.x * t, o.z + d.z * t, hit.n); hit.mu = 1; hit.kind = 'ground'; found = true;
      }
    }
    for (const m of this.meshes) if (m.raycast(o, d, maxT, hit)) found = true;
    for (const b of this.boxes) {
      if (o.dist(b.c) > b.r + maxT) continue;
      if (b.raycast(o, d, maxT, hit)) found = true;
    }
    return found;
  }
  // קרן דו-צדדית (למצלמה): פוגעת גם בגב של משטחים
  raycastAny(o, d, maxT, hit) {
    hit.t = maxT; let found = false;
    for (const m of this.meshes) if (m.raycast(o, d, maxT, hit, true)) found = true;
    for (const b of this.boxes) { if (o.dist(b.c) > b.r + maxT) continue; if (b.raycast(o, d, maxT, hit)) found = true; }
    return found;
  }
  penetrate(p, res) {
    res.depth = 0; let found = false;
    const gy = groundHeight(p.x, p.z);
    if (p.y < gy) { res.depth = gy - p.y; groundNormal(p.x, p.z, res.n); res.mu = 0.7; found = true; }
    for (const m of this.meshes) if (m.penetrate(p, res)) found = true;
    for (const b of this.boxes) {
      if (p.dist(b.c) > b.r) continue;
      if (b.penetrate(p, res)) found = true;
    }
    return found;
  }
}
export function makeHit() { return { t: 0, n: new V3(), mu: 1, kind: null }; }
export function makePen() { return { depth: 0, n: new V3(), mu: 1 }; }
