// מתמטיקה וקטורית מינימלית לליבת הפיזיקה (בלי three, כדי שתרוץ גם ב-node).
// מערכת הצירים זהה ל-three: y למעלה. בגוף הרכב: z קדימה, y למעלה, x שמאלה.

export class V3 {
  constructor(x = 0, y = 0, z = 0) { this.x = x; this.y = y; this.z = z; }
  set(x, y, z) { this.x = x; this.y = y; this.z = z; return this; }
  copy(v) { this.x = v.x; this.y = v.y; this.z = v.z; return this; }
  clone() { return new V3(this.x, this.y, this.z); }
  add(v) { this.x += v.x; this.y += v.y; this.z += v.z; return this; }
  sub(v) { this.x -= v.x; this.y -= v.y; this.z -= v.z; return this; }
  addScaled(v, s) { this.x += v.x * s; this.y += v.y * s; this.z += v.z * s; return this; }
  scale(s) { this.x *= s; this.y *= s; this.z *= s; return this; }
  dot(v) { return this.x * v.x + this.y * v.y + this.z * v.z; }
  len() { return Math.sqrt(this.x * this.x + this.y * this.y + this.z * this.z); }
  len2() { return this.x * this.x + this.y * this.y + this.z * this.z; }
  norm() { const l = this.len(); if (l > 1e-12) this.scale(1 / l); return this; }
  crossVectors(a, b) {
    const x = a.y * b.z - a.z * b.y, y = a.z * b.x - a.x * b.z, z = a.x * b.y - a.y * b.x;
    this.x = x; this.y = y; this.z = z; return this;
  }
  subVectors(a, b) { this.x = a.x - b.x; this.y = a.y - b.y; this.z = a.z - b.z; return this; }
  addVectors(a, b) { this.x = a.x + b.x; this.y = a.y + b.y; this.z = a.z + b.z; return this; }
  lerp(v, t) { this.x += (v.x - this.x) * t; this.y += (v.y - this.y) * t; this.z += (v.z - this.z) * t; return this; }
  dist(v) { const dx = this.x - v.x, dy = this.y - v.y, dz = this.z - v.z; return Math.sqrt(dx * dx + dy * dy + dz * dz); }
  // הטלה על המישור שהנורמל שלו n (n יחידה)
  projectOnPlane(n) { const d = this.dot(n); this.x -= n.x * d; this.y -= n.y * d; this.z -= n.z * d; return this; }
}

export function cross(a, b) { return new V3().crossVectors(a, b); }

export class Quat {
  constructor(x = 0, y = 0, z = 0, w = 1) { this.x = x; this.y = y; this.z = z; this.w = w; }
  copy(q) { this.x = q.x; this.y = q.y; this.z = q.z; this.w = q.w; return this; }
  clone() { return new Quat(this.x, this.y, this.z, this.w); }
  setAxisAngle(ax, a) {
    const s = Math.sin(a / 2);
    this.x = ax.x * s; this.y = ax.y * s; this.z = ax.z * s; this.w = Math.cos(a / 2); return this;
  }
  multiply(b) { return this.multiplyQuats(this, b); }
  premultiply(a) { return this.multiplyQuats(a, this); }
  multiplyQuats(a, b) {
    const x = a.x * b.w + a.w * b.x + a.y * b.z - a.z * b.y;
    const y = a.y * b.w + a.w * b.y + a.z * b.x - a.x * b.z;
    const z = a.z * b.w + a.w * b.z + a.x * b.y - a.y * b.x;
    const w = a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z;
    this.x = x; this.y = y; this.z = z; this.w = w; return this;
  }
  norm() {
    const l = Math.hypot(this.x, this.y, this.z, this.w) || 1;
    this.x /= l; this.y /= l; this.z /= l; this.w /= l; return this;
  }
  conj() { this.x = -this.x; this.y = -this.y; this.z = -this.z; return this; }
  // סיבוב וקטור v (במקום) מהגוף לעולם
  rotate(v) {
    const qx = this.x, qy = this.y, qz = this.z, qw = this.w;
    const tx = 2 * (qy * v.z - qz * v.y), ty = 2 * (qz * v.x - qx * v.z), tz = 2 * (qx * v.y - qy * v.x);
    v.x += qw * tx + qy * tz - qz * ty;
    v.y += qw * ty + qz * tx - qx * tz;
    v.z += qw * tz + qx * ty - qy * tx;
    return v;
  }
  // סיבוב הפוך: מהעולם לגוף
  unrotate(v) {
    const qx = -this.x, qy = -this.y, qz = -this.z, qw = this.w;
    const tx = 2 * (qy * v.z - qz * v.y), ty = 2 * (qz * v.x - qx * v.z), tz = 2 * (qx * v.y - qy * v.x);
    v.x += qw * tx + qy * tz - qz * ty;
    v.y += qw * ty + qz * tx - qx * tz;
    v.z += qw * tz + qx * ty - qy * tx;
    return v;
  }
  // אינטגרציה: q += 0.5 * (ω,0) ⊗ q * dt, ω בעולם
  integrate(w, dt) {
    const hx = w.x * dt * 0.5, hy = w.y * dt * 0.5, hz = w.z * dt * 0.5;
    const x = this.x, y = this.y, z = this.z, qw = this.w;
    this.x += hx * qw + hy * z - hz * y;
    this.y += hy * qw + hz * x - hx * z;
    this.z += hz * qw + hx * y - hy * x;
    this.w += -hx * x - hy * y - hz * z;
    return this.norm();
  }
  slerp(qb, t) {
    let cos = this.x * qb.x + this.y * qb.y + this.z * qb.z + this.w * qb.w;
    let bx = qb.x, by = qb.y, bz = qb.z, bw = qb.w;
    if (cos < 0) { cos = -cos; bx = -bx; by = -by; bz = -bz; bw = -bw; }
    let k0, k1;
    if (cos > 0.9995) { k0 = 1 - t; k1 = t; }
    else {
      const a = Math.acos(cos), s = Math.sin(a);
      k0 = Math.sin((1 - t) * a) / s; k1 = Math.sin(t * a) / s;
    }
    this.x = this.x * k0 + bx * k1; this.y = this.y * k0 + by * k1;
    this.z = this.z * k0 + bz * k1; this.w = this.w * k0 + bw * k1;
    return this.norm();
  }
  // בסיס אורתונורמלי (x=שמאל, y=למעלה, z=קדימה) לקווטרניון
  setFromBasis(xa, ya, za) {
    const m00 = xa.x, m01 = ya.x, m02 = za.x, m10 = xa.y, m11 = ya.y, m12 = za.y, m20 = xa.z, m21 = ya.z, m22 = za.z;
    const tr = m00 + m11 + m22;
    if (tr > 0) {
      const s = 0.5 / Math.sqrt(tr + 1);
      this.w = 0.25 / s; this.x = (m21 - m12) * s; this.y = (m02 - m20) * s; this.z = (m10 - m01) * s;
    } else if (m00 > m11 && m00 > m22) {
      const s = 2 * Math.sqrt(1 + m00 - m11 - m22);
      this.w = (m21 - m12) / s; this.x = 0.25 * s; this.y = (m01 + m10) / s; this.z = (m02 + m20) / s;
    } else if (m11 > m22) {
      const s = 2 * Math.sqrt(1 + m11 - m00 - m22);
      this.w = (m02 - m20) / s; this.x = (m01 + m10) / s; this.y = 0.25 * s; this.z = (m12 + m21) / s;
    } else {
      const s = 2 * Math.sqrt(1 + m22 - m00 - m11);
      this.w = (m10 - m01) / s; this.x = (m02 + m20) / s; this.y = (m12 + m21) / s; this.z = 0.25 * s;
    }
    return this.norm();
  }
}

export const clamp = (x, a, b) => (x < a ? a : x > b ? b : x);
export const lerp = (a, b, t) => a + (b - a) * t;
export const smoothstep = (a, b, x) => { const t = clamp((x - a) / (b - a), 0, 1); return t * t * (3 - 2 * t); };
// התקרבות אקספוננציאלית שלא תלויה בקצב הפריימים
export const damp = (cur, target, rate, dt) => cur + (target - cur) * (1 - Math.exp(-rate * dt));
export const G0 = 9.80665;
