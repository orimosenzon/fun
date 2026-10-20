// אלגברה לינארית צפופה וקטנה, בלי תלויות. מטריצות שטוחות בסדר שורות (row-major).

// פתרון מערכת סימטרית חיובית H x = g בפירוק צ'ולסקי. H נדרסת.
export function cholSolve(H, g, n, x = new Float64Array(n)) {
  for (let j = 0; j < n; j++) {
    let s = H[j * n + j];
    for (let k = 0; k < j; k++) s -= H[j * n + k] * H[j * n + k];
    if (s <= 1e-12) s = 1e-12;
    const d = Math.sqrt(s);
    H[j * n + j] = d;
    for (let i = j + 1; i < n; i++) {
      let t = H[i * n + j];
      for (let k = 0; k < j; k++) t -= H[i * n + k] * H[j * n + k];
      H[i * n + j] = t / d;
    }
  }
  for (let i = 0; i < n; i++) {
    let t = g[i];
    for (let k = 0; k < i; k++) t -= H[i * n + k] * x[k];
    x[i] = t / H[i * n + i];
  }
  for (let i = n - 1; i >= 0; i--) {
    let t = x[i];
    for (let k = i + 1; k < n; k++) t -= H[k * n + i] * x[k];
    x[i] = t / H[i * n + i];
  }
  return x;
}

// פירוק צ'ולסקי במקום (המשולש התחתון), ופתרון עם הפירוק לכמה אגפים ימניים
export function cholFactor(H, n) {
  for (let j = 0; j < n; j++) {
    let s = H[j * n + j];
    for (let k = 0; k < j; k++) s -= H[j * n + k] * H[j * n + k];
    const d = Math.sqrt(Math.max(s, 1e-12));
    H[j * n + j] = d;
    for (let i = j + 1; i < n; i++) {
      let t = H[i * n + j];
      for (let k = 0; k < j; k++) t -= H[i * n + k] * H[j * n + k];
      H[i * n + j] = t / d;
    }
  }
}
export function cholSolveF(L, g, n, x = new Float64Array(n)) {
  for (let i = 0; i < n; i++) { let t = g[i]; for (let k = 0; k < i; k++) t -= L[i * n + k] * x[k]; x[i] = t / L[i * n + i]; }
  for (let i = n - 1; i >= 0; i--) { let t = x[i]; for (let k = i + 1; k < n; k++) t -= L[k * n + i] * x[k]; x[i] = t / L[i * n + i]; }
  return x;
}

// פתרון כללי עם פירוק LU וציר חלקי (למערכות קטנות לא סימטריות)
export function luSolve(A, b, n) {
  const a = Float64Array.from(A), x = Float64Array.from(b);
  for (let k = 0; k < n; k++) {
    let p = k, mx = Math.abs(a[k * n + k]);
    for (let i = k + 1; i < n; i++) { const v = Math.abs(a[i * n + k]); if (v > mx) { mx = v; p = i; } }
    if (mx < 1e-14) continue;
    if (p !== k) {
      for (let j = 0; j < n; j++) { const t = a[k * n + j]; a[k * n + j] = a[p * n + j]; a[p * n + j] = t; }
      const t = x[k]; x[k] = x[p]; x[p] = t;
    }
    for (let i = k + 1; i < n; i++) {
      const f = a[i * n + k] / a[k * n + k];
      if (!f) continue;
      for (let j = k; j < n; j++) a[i * n + j] -= f * a[k * n + j];
      x[i] -= f * x[k];
    }
  }
  for (let i = n - 1; i >= 0; i--) {
    let t = x[i];
    for (let j = i + 1; j < n; j++) t -= a[i * n + j] * x[j];
    x[i] = Math.abs(a[i * n + i]) > 1e-14 ? t / a[i * n + i] : 0;
  }
  return x;
}

// ---- וקטורים תלת-ממדיים וקווטרניונים (w, x, y, z כמו ב-MuJoCo) ----
export const v = {
  add: (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]],
  sub: (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]],
  scale: (a, s) => [a[0] * s, a[1] * s, a[2] * s],
  dot: (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2],
  cross: (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]],
  len: (a) => Math.hypot(a[0], a[1], a[2]),
  lerp: (a, b, t) => [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t],
};

// מטריצת סיבוב 3x3 (row-major, כמו xmat של MuJoCo) מקווטרניון
export function quatToMat(q) {
  const [w, x, y, z] = q;
  return [
    1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
    2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
    2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
  ];
}
export function matMul3(A, B) {
  const C = new Array(9);
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) C[i * 3 + j] = A[i * 3] * B[j] + A[i * 3 + 1] * B[3 + j] + A[i * 3 + 2] * B[6 + j];
  return C;
}
export function matT3(A) { return [A[0], A[3], A[6], A[1], A[4], A[7], A[2], A[5], A[8]]; }
export function matVec3(A, x) { return [A[0] * x[0] + A[1] * x[1] + A[2] * x[2], A[3] * x[0] + A[4] * x[1] + A[5] * x[2], A[6] * x[0] + A[7] * x[1] + A[8] * x[2]]; }

// שגיאת סיבוב: הווקטור (ציר × זווית) שמסובב את R אל Rd, בקואורדינטות עולם
export function rotError(Rd, R) {
  const E = matMul3(Rd, matT3(R));
  const tr = E[0] + E[4] + E[8];
  const c = Math.max(-1, Math.min(1, (tr - 1) / 2));
  const ang = Math.acos(c);
  const w = [E[7] - E[5], E[2] - E[6], E[3] - E[1]];
  const s = Math.sin(ang);
  if (s < 1e-6) return [w[0] / 2, w[1] / 2, w[2] / 2];
  const k = ang / (2 * s);
  return [w[0] * k, w[1] * k, w[2] * k];
}
// סיבוב סביב ציר z, ואחריו הטיה (pitch סביב y) ו-roll סביב x
export function rotZYX(yaw, pitch = 0, roll = 0) {
  const cz = Math.cos(yaw), sz = Math.sin(yaw), cy = Math.cos(pitch), sy = Math.sin(pitch), cx = Math.cos(roll), sx = Math.sin(roll);
  return [
    cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx,
    sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx,
    -sy, cy * sx, cy * cx,
  ];
}
export function yawOf(R) { return Math.atan2(R[3], R[0]); }
export const wrapAngle = (a) => Math.atan2(Math.sin(a), Math.cos(a));
export const clamp = (x, a, b) => (x < a ? a : x > b ? b : x);
export const smooth5 = (s) => { s = clamp(s, 0, 1); return s * s * s * (10 + s * (-15 + 6 * s)); };
export const smooth5d = (s) => { s = clamp(s, 0, 1); return 30 * s * s * (1 - s) * (1 - s); };
