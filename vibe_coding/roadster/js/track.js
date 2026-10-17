// הגיאומטריה של אתר הניסויים: מבנים כמשטחים פרמטריים. אותם מערכים משמשים
// גם את מנוע ההתנגשויות וגם את הציור, כך שמה שרואים הוא בדיוק מה שהרכב מרגיש.

import { TriMesh, OBB } from './collide.js';

// ---------- מסלול דו-ממדי מאינטגרציה של עקמומיות ----------
// segments: [{len, k0, k1}] כשהעקמומיות משתנה לינארית לאורך הקטע (קלותואידה, Euler spiral).
// מחזיר פונקציה s → {h (אופקי), v (אנכי), th (זווית)} ואת האורך הכולל.
export function curvePath(segments, step = 0.05) {
  const pts = [{ s: 0, h: 0, v: 0, th: 0, k: segments[0].k0 }];
  let s = 0, h = 0, v = 0, th = 0;
  for (const seg of segments) {
    const n = Math.max(1, Math.ceil(seg.len / step));
    const ds = seg.len / n;
    for (let i = 0; i < n; i++) {
      const k = seg.k0 + (seg.k1 - seg.k0) * ((i + 0.5) / n);
      const thMid = th + k * ds / 2;
      h += Math.cos(thMid) * ds; v += Math.sin(thMid) * ds; th += k * ds; s += ds;
      pts.push({ s, h, v, th, k: seg.k0 + (seg.k1 - seg.k0) * ((i + 1) / n) });
    }
  }
  const total = s;
  const at = (q) => {
    q = Math.max(0, Math.min(total, q));
    let lo = 0, hi = pts.length - 1;
    while (hi - lo > 1) { const m = (lo + hi) >> 1; if (pts[m].s <= q) lo = m; else hi = m; }
    const a = pts[lo], b = pts[hi], t = (q - a.s) / Math.max(b.s - a.s, 1e-9);
    return { h: a.h + (b.h - a.h) * t, v: a.v + (b.v - a.v) * t, th: a.th + (b.th - a.th) * t, k: a.k + (b.k - a.k) * t };
  };
  return { total, at, pts };
}

// ---------- משטח פרמטרי עם עובי ----------
// fn(u, v) → [x, y, z]. הנורמל של הצד העליון = ∂P/∂u × ∂P/∂v.
// thickness > 0 מוסיף גב (נורמלים הפוכים) כך שאפשר להתנגש גם מהצד השני.
export function paramSurface(fn, u0, u1, nu, v0, v1, nv, { thickness = 0.35, flip = false } = {}) {
  const P = [], N = [], UV = [], I = [];
  const eps = 1e-3;
  const sides = thickness > 0 ? 2 : 1;
  for (let side = 0; side < sides; side++) {
    const base = P.length / 3;
    for (let j = 0; j <= nv; j++) {
      const v = v0 + (v1 - v0) * j / nv;
      for (let i = 0; i <= nu; i++) {
        const u = u0 + (u1 - u0) * i / nu;
        const p = fn(u, v);
        const pu = fn(u + eps, v), pv = fn(u, v + eps);
        const du = [pu[0] - p[0], pu[1] - p[1], pu[2] - p[2]];
        const dv = [pv[0] - p[0], pv[1] - p[1], pv[2] - p[2]];
        let n = [du[1] * dv[2] - du[2] * dv[1], du[2] * dv[0] - du[0] * dv[2], du[0] * dv[1] - du[1] * dv[0]];
        const l = Math.hypot(...n) || 1; n = n.map(c => c / l);
        if (flip) n = n.map(c => -c);
        if (side === 1) {
          P.push(p[0] - n[0] * thickness, p[1] - n[1] * thickness, p[2] - n[2] * thickness);
          N.push(-n[0], -n[1], -n[2]);
        } else {
          P.push(...p); N.push(...n);
        }
        UV.push(u, v);
      }
    }
    const row = nu + 1;
    for (let j = 0; j < nv; j++) for (let i = 0; i < nu; i++) {
      const a = base + j * row + i, b = a + 1, c = a + row, d = c + 1;
      const front = (side === 0) !== flip;
      if (front) I.push(a, b, d, a, d, c); else I.push(a, d, b, a, c, d);
    }
  }
  return { positions: new Float32Array(P), normals: new Float32Array(N), uvs: new Float32Array(UV), indices: new Uint32Array(I), nu, nv, thickness };
}

// ---------- הפריסה ----------
export const LAYOUT = {
  wall: { x: 0, z: 180, width: 10, kMax: 1 / 10.2, clothoid: 15, angleDeg: 170, approach: 0 },
  lineup: { x: -90, z: 110 },
  drag: { x: 120, z0: -300, z1: 350 },
  tunnel: { x: -260, z0: -80, z1: 80, r: 8 },
  loop: { x: 260, z: -60, r: 8.5, clothoid: 9, width: 6, shift: 8 },
  wallRide: { x: -440, z0: -220, z1: 220, r: 7, height: 13 },
  kickers: [{ x: 60, z: 130 }, { x: 60, z: 260 }],
  station: { x: 28, z: -30, r: 6 },
  hover: { x: 0, z: -160 },
};

// הקיר מההדגמה: גישה ישרה, קלותואידה, וקשת עד 170°
export function wallProfile(L = LAYOUT.wall) {
  const th = L.angleDeg * Math.PI / 180;
  const thClo = L.kMax * L.clothoid / 2;
  const arc = (th - thClo) / L.kMax;
  return curvePath([
    { len: 8, k0: 0, k1: 0 },
    { len: L.clothoid, k0: 0, k1: L.kMax },
    { len: arc, k0: L.kMax, k1: L.kMax },
  ]);
}

export function loopPath(L = LAYOUT.loop) {
  const k = 1 / L.r;
  const thClo = k * L.clothoid / 2;            // זווית בכל קלותואידה
  const arc = (2 * Math.PI - 2 * thClo) / k;
  return curvePath([
    { len: 6, k0: 0, k1: 0 },
    { len: L.clothoid, k0: 0, k1: k },
    { len: arc, k0: k, k1: k },
    { len: L.clothoid, k0: k, k1: 0 },
    { len: 6, k0: 0, k1: 0 },
  ]);
}

export function wallRideProfile(L = LAYOUT.wallRide) {
  const k = 1 / L.r, clo = 6, thClo = k * clo / 2;
  return curvePath([
    { len: clo, k0: 0, k1: k },
    { len: (Math.PI / 2 - thClo) / k, k0: k, k1: k },
    { len: L.height, k0: 0, k1: 0 },
  ]);
}

export function kickerProfile() {
  // רמפת קפיצה: עלייה מעוגלת עד 22°
  const k = 1 / 14;
  return curvePath([{ len: (22 * Math.PI / 180) / k, k0: k, k1: k }]);
}

const smooth = (t) => t * t * (3 - 2 * t);

// ההזזה הצידית של הלולאה לאורך המסלול
export function loopShift(s, pr, L = LAYOUT.loop) {
  return L.shift * smooth(Math.min(1, Math.max(0, (s - 6) / (pr.total - 12))));
}

// סיוע היגוי בלולאה: הסטייה (מטרים) של הרכב מקו האמצע, בכיוון "שמאל" של הרכב
let _loopPr = null;
export function loopGuide(pos, left) {
  const L = LAYOUT.loop;
  const pr = _loopPr || (_loopPr = loopPath(L));
  if (Math.abs(pos.x - L.x) > 20 || pos.z < L.z - 2 || pos.z > L.z + 40) return null;
  let best = 1e9, bs = 0;
  for (let i = 0; i < pr.pts.length; i += 4) {
    const q = pr.pts[i], dz = L.z + q.h - pos.z, dy = q.v - pos.y;
    const d = dz * dz + dy * dy;
    // מעדיפים את הענף הנכון לפי הצד: הכניסה ב-x גבוה, היציאה ב-x נמוך
    const sh = loopShift(q.s, pr, L), dx = (L.x - sh) - pos.x;
    const dd = d + dx * dx * 0.05;
    if (dd < best) { best = dd; bs = q.s; }
  }
  const q = pr.at(bs);
  const cx = L.x - loopShift(bs, pr, L), cy = q.v, cz = L.z + q.h;
  const err = (cx - pos.x) * left.x + (cy - pos.y) * left.y + (cz - pos.z) * left.z;
  return { err, s: bs, total: pr.total };
}

// בונה את כל המבנים. מחזיר רשימה לציור + רשתות התנגשות + קופסאות.
export function buildTrack(world) {
  const parts = [];
  const add = (name, geo, mat, extra = {}) => {
    const mesh = new TriMesh(geo.positions, geo.normals, geo.indices, { name, mu: extra.mu ?? 1, thickness: geo.thickness });
    world?.addMesh(mesh);
    parts.push({ name, geo, mat, ...extra });
  };

  // הקיר: כיוון +z, רוחב לאורך x. u = רוחב, v = מרחק לאורך הפרופיל
  {
    const L = LAYOUT.wall, pr = wallProfile(L);
    const fn = (u, s) => { const q = pr.at(s); return [L.x - u, q.v, L.z + q.h]; };
    const n = Math.ceil(pr.total / 0.3);
    add('wall', paramSurface(fn, -L.width / 2, L.width / 2, 8, 0, pr.total, n, { thickness: 0.6 }), 'ramp', { profile: pr, layout: L });
  }

  // הלולאה, עם הזזה צידית כדי שהיציאה לא תעלה על הכניסה
  {
    const L = LAYOUT.loop, pr = loopPath(L);
    const fn = (u, s) => {
      const q = pr.at(s), sh = L.shift * smooth(Math.min(1, Math.max(0, (s - 6) / (pr.total - 12))));
      return [L.x - u - sh, q.v, L.z + q.h];
    };
    add('loop', paramSurface(fn, -L.width / 2, L.width / 2, 6, 0, pr.total, Math.ceil(pr.total / 0.3), { thickness: 0.6 }), 'loop', { profile: pr, layout: L });
  }

  // מנהרה: צינור שלם. u = זווית, v = z. הנורמל פונה פנימה.
  {
    const L = LAYOUT.tunnel;
    const fn = (a, z) => [L.x + Math.sin(a) * L.r, L.r - Math.cos(a) * L.r + 0.01, z];
    add('tunnel', paramSurface(fn, 0, Math.PI * 2, 96, L.z0, L.z1, 80, { thickness: 0.4, flip: true }), 'tunnel', { layout: L });
  }

  // קיר הרכיבה: רבע צינור שהופך לקיר אנכי. u = z, v = פרופיל, עולה לכיוון −x.
  {
    const L = LAYOUT.wallRide, pr = wallRideProfile(L);
    const fn = (z, s) => { const q = pr.at(s); return [L.x - q.h, q.v, z]; };
    add('wallride', paramSurface(fn, L.z0, L.z1, 90, 0, pr.total, Math.ceil(pr.total / 0.35), { thickness: 0.6, flip: true }), 'wallride', { profile: pr, layout: L });
  }

  // רמפות קפיצה
  for (const K of LAYOUT.kickers) {
    const pr = kickerProfile();
    const fn = (u, s) => { const q = pr.at(s); return [K.x - u, q.v, K.z + q.h]; };
    add('kicker', paramSurface(fn, -3, 3, 4, 0, pr.total, 24, { thickness: 0.3 }), 'kicker', { profile: pr, layout: K });
    // הגב האנכי של הרמפה כקופסה
    const end = pr.at(pr.total);
    world?.addBox(new OBB({ center: [K.x, end.v / 2, K.z + end.h - 0.6], half: [3, end.v / 2, 0.6], name: 'kicker' }));
  }

  // שורת הרכבים החונים (כמו בהדגמה): צד אל הדרך, הקפיצה לאורך +z
  const L = LAYOUT.lineup;
  const vehicles = [
    { key: 'cybercab', nameHe: 'סייברקאב', len: 4.3, wid: 1.95, hgt: 1.45 },
    { key: 'cybertruck', nameHe: 'סייברטראק', len: 5.68, wid: 2.03, hgt: 1.79 },
    { key: 'semi', nameHe: 'סמי', len: 6.6, wid: 2.59, hgt: 3.9 },
    { key: 'modely', nameHe: 'מודל Y', len: 4.75, wid: 1.92, hgt: 1.62 },
    { key: 'model3', nameHe: 'מודל 3', len: 4.72, wid: 1.85, hgt: 1.44 },
  ];
  let z = L.z;
  const parked = [];
  for (const v of vehicles) {
    z += v.wid / 2;
    const c = { ...v, x: L.x, z, yaw: Math.PI / 2 };
    parked.push(c);
    // קופסה אחת לגוף. הסמי מקבל גם קופסת תא
    world?.addBox(new OBB({ center: [L.x, v.hgt / 2, z], half: [v.len / 2, v.hgt / 2, v.wid / 2], name: v.key, mu: 0.5 }));
    z += v.wid / 2 + 1.1;
  }
  const lineupEnd = z;

  return { parts, parked, lineupEnd };
}

// נקודות התחלה לכל תרחיש: [x, z, כיוון (רדיאנים, 0 = +z)]
export const SPAWNS = {
  free: [0, -10, 0],
  wall: [0, 40, 0],
  kickflip: [LAYOUT.lineup.x, -30, 0],
  drag: [LAYOUT.drag.x, LAYOUT.drag.z0, 0],
  tunnel: [LAYOUT.tunnel.x, -190, 0],
  loop: [LAYOUT.loop.x, -200, 0],
  wallride: [LAYOUT.wallRide.x + 30, -300, 0],
  hover: [LAYOUT.hover.x, LAYOUT.hover.z - 25, 0],
};
