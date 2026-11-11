// body.js: מודל הגוף. קוד טהור (בלי three.js) כדי שאפשר יהיה לבדוק אותו ב-node.
// מערכת צירים מקומית של לוחם: x קדימה (אל היריב), y למעלה, z ימינה.
// תנוחה (pose) היא אוסף מספרים; solve() מחזירה את מיקומי המפרקים.

export const D2R = Math.PI / 180;

// מידות גוף במטרים (גובה כ-1.77)
export const DIM = {
  thigh: 0.44, shin: 0.43, ankleH: 0.08, hipW: 0.095,
  spine: 0.50, shW: 0.20, upper: 0.30, fore: 0.27, fist: 0.08,
  neck: 0.08, head: 0.14, headR: 0.11, torsoR: 0.16,
};

// ---------- וקטורים ומטריצות ----------
export const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
export const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
export const mul = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
export const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
export const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
export const len = (a) => Math.hypot(a[0], a[1], a[2]);
export const norm = (a) => { const l = len(a) || 1; return [a[0] / l, a[1] / l, a[2] / l]; };

// סיבוב סביב y: זווית חיובית מפנה את "קדימה" ימינה (אל +z)
export const Ry = (a) => { const c = Math.cos(a), s = Math.sin(a); return [[c, 0, -s], [0, 1, 0], [s, 0, c]]; };
// הטיה קדימה: זווית חיובית מטה את "למעלה" אל +x (ואת "קדימה" מטה)
export const Rz = (a) => { const c = Math.cos(a), s = Math.sin(a); return [[c, s, 0], [-s, c, 0], [0, 0, 1]]; };
// הטיה הצידה: זווית חיובית מטה את "למעלה" ימינה (אל +z)
export const Rx = (a) => { const c = Math.cos(a), s = Math.sin(a); return [[1, 0, 0], [0, c, -s], [0, s, c]]; };
export const mmul = (A, B) => {
  const R = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) R[i][j] = A[i][0] * B[0][j] + A[i][1] * B[1][j] + A[i][2] * B[2][j];
  return R;
};
export const mv = (M, v) => [
  M[0][0] * v[0] + M[0][1] * v[1] + M[0][2] * v[2],
  M[1][0] * v[0] + M[1][1] * v[1] + M[1][2] * v[2],
  M[2][0] * v[0] + M[2][1] * v[1] + M[2][2] * v[2],
];

const wrap = (a) => { a = (a + Math.PI) % (2 * Math.PI); if (a < 0) a += 2 * Math.PI; return a - Math.PI; };
const clamp = (x, a, b) => Math.max(a, Math.min(b, x));

// ---------- תנוחות ----------
const SWAP = [['fL', 'fR'], ['fLr', 'fRr'], ['kL', 'kR'], ['hL', 'hR'], ['eL', 'eR'], ['gL', 'gR']];

export function clonePose(p) {
  const q = {};
  for (const k in p) q[k] = Array.isArray(p[k]) ? p[k].slice() : p[k];
  return q;
}

export function lerpPose(a, b, t, out = {}) {
  for (const k in a) {
    const x = a[k], y = b[k];
    if (Array.isArray(x)) {
      const o = out[k] || (out[k] = new Array(x.length));
      for (let i = 0; i < x.length; i++) o[i] = x[i] + (y[i] - x[i]) * t;
    } else out[k] = x + (y - x) * t;
  }
  return out;
}

// שיקוף ימין-שמאל: הופך עמידת שמאל-קדימה לעמידת ימין-קדימה
export function mirrorPose(p) {
  const q = clonePose(p);
  for (const [l, r] of SWAP) { const t = q[l]; q[l] = q[r]; q[r] = t; }
  for (const k of ['pel', 'fL', 'fR', 'kL', 'kR', 'hL', 'hR', 'eL', 'eR']) q[k][2] = -q[k][2];
  q.rot[0] = -q.rot[0]; q.rot[2] = -q.rot[2];
  q.tw = -q.tw; q.head[0] = -q.head[0];
  for (const k of ['fLr', 'fRr']) { q[k][0] = -q[k][0]; q[k][2] = -q[k][2]; }
  return q;
}

// הזזת התנוחה כולה לאורך ציר הקרב
export function shiftPose(p, dx) {
  const q = clonePose(p);
  q.pel[0] += dx; q.fL[0] += dx; q.fR[0] += dx;
  return q;
}

export function torsoFrame(p) {
  return mmul(mmul(Ry(p.rot[0]), Rz(p.rot[1])), Rx(p.rot[2]));
}
export function chestOf(p) {
  return add(p.pel, mv(torsoFrame(p), [0, DIM.spine, 0]));
}

// ---------- קינמטיקה הפוכה לשתי עצמות ----------
// pole: הכיוון שאליו המפרק האמצעי (ברך/מרפק) פונה
export function ik2(root, target, l1, l2, pole) {
  let d = sub(target, root);
  let dist = len(d);
  const max = (l1 + l2) * 0.999, min = Math.abs(l1 - l2) + 0.03;
  const stretch = dist / (l1 + l2);
  let dir = dist > 1e-6 ? mul(d, 1 / dist) : [0, -1, 0];
  dist = clamp(dist, min, max);
  const a = (l1 * l1 - l2 * l2 + dist * dist) / (2 * dist);
  const h = Math.sqrt(Math.max(0, l1 * l1 - a * a));
  let pp = sub(pole, mul(dir, dot(pole, dir)));
  if (len(pp) < 1e-4) {
    pp = cross(dir, [0, 0, 1]);
    if (len(pp) < 1e-4) pp = cross(dir, [1, 0, 0]);
  }
  pp = norm(pp);
  return {
    mid: add(add(root, mul(dir, a)), mul(pp, h)),
    end: add(root, mul(dir, dist)),
    stretch,
  };
}

function handFrame(el, wr, roll, side) {
  const d = norm(sub(wr, el));
  let u = sub([0, 1, 0], mul(d, d[1]));
  if (len(u) < 0.05) u = sub([-1, 0, 0], mul(d, -d[0]));
  u = norm(u);
  const w = cross(d, u);
  // גב כף היד: roll=0 גב היד למעלה (אגרוף גמור), roll=180 כף היד למעלה (היקיטה)
  const b = norm(add(mul(u, Math.cos(roll)), mul(w, Math.sin(roll) * side)));
  const z = cross(d, b);
  // עמודות: x לאורך האמה, y גב כף היד, z
  return [[d[0], b[0], z[0]], [d[1], b[1], z[1]], [d[2], b[2], z[2]]];
}

// ---------- פתרון תנוחה: מספרים -> מפרקים ----------
export function solve(p) {
  const yaw = p.rot[0], pitch = p.rot[1], roll = p.rot[2];
  const T = torsoFrame(p);
  const P = mmul(mmul(Ry(yaw), Rz(pitch * 0.45)), Rx(roll * 0.45)); // האגן נוטה פחות מהגב
  const C = mmul(T, Ry(p.tw));
  const B = mmul(T, Ry(p.tw * 0.5));
  const pel = p.pel;
  const chest = add(pel, mv(T, [0, DIM.spine, 0]));
  const belly = add(pel, mv(T, [0, DIM.spine * 0.5, 0]));
  const J = { pel, chest, belly, P, T, C, B };

  J.hipL = add(pel, mv(P, [0, 0, -DIM.hipW]));
  J.hipR = add(pel, mv(P, [0, 0, DIM.hipW]));
  J.shL = add(chest, mv(C, [0, -0.03, -DIM.shW]));
  J.shR = add(chest, mv(C, [0, -0.03, DIM.shW]));
  J.neck = add(chest, mv(C, [0, DIM.neck, 0]));

  // הראש מביט אל היריב ככל שהצוואר מאפשר
  const chestYaw = yaw + p.tw;
  const rel = clamp(wrap(p.head[0] - chestYaw), -85 * D2R, 85 * D2R);
  const hy = chestYaw + rel;
  const hp = p.head[1] + 0.55 * pitch * Math.cos(rel);
  const H = mmul(Ry(hy), Rz(hp));
  J.H = H;
  J.head = add(J.neck, mv(H, [0.01, DIM.head, 0]));

  let stretch = 0;
  for (const s of ['L', 'R']) {
    const f = p['f' + s];
    const leg = ik2(J['hip' + s], [f[0], DIM.ankleH + f[1], f[2]], DIM.thigh, DIM.shin, p['k' + s]);
    J['kn' + s] = leg.mid; J['an' + s] = leg.end;
    const shinDir = norm(sub(leg.end, leg.mid));
    J['foot' + s] = add(leg.end, mul(shinDir, 0.09)); // נקודת הפגיעה של הרגל
    const fr = p['f' + s + 'r'];
    J['F' + s] = mmul(mmul(Ry(fr[0]), Rz(fr[1])), Rx(fr[2]));
    stretch = Math.max(stretch, leg.stretch);

    const arm = ik2(J['sh' + s], add(chest, p['h' + s]), DIM.upper, DIM.fore, p['e' + s]);
    J['el' + s] = arm.mid; J['wr' + s] = arm.end;
    const foreDir = norm(sub(arm.end, arm.mid));
    J['hand' + s] = add(arm.end, mul(foreDir, DIM.fist)); // נקודת הפגיעה של היד
    J['G' + s] = handFrame(arm.mid, arm.end, p['g' + s][0], s === 'L' ? -1 : 1);
    J['open' + s] = p['g' + s][1] > 0.5;
    J['armStretch' + s] = arm.stretch;
  }
  J.legStretch = stretch;
  return J;
}

// מרחק מנקודה לקטע
export function distToSeg(p, a, b) {
  const ab = sub(b, a), ap = sub(p, a);
  const t = clamp(dot(ap, ab) / (dot(ab, ab) || 1), 0, 1);
  return len(sub(ap, mul(ab, t)));
}
