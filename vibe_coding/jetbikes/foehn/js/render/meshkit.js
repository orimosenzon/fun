// A small mesh builder for everything that is modelled in code: quads and triangles with a colour,
// optional texture coordinates, a transform stack, and the handful of shapes the fortress and its
// furnishings are made of (boxes, towers, roofs, arches, turned shapes, limbs, extruded outlines).
import * as THREE from '../../lib/three.module.min.js';

export class Builder {
  constructor(uv = false) { this.p = []; this.n = []; this.c = []; this.w = []; this.uv = uv ? [] : null; this.M = null; this.stack = []; this.wv = 1; }
  // transform: rows of a 3x3 matrix, then the translation
  push(pos, yaw = 0, s = 1, roll = 0) {
    const cy = Math.cos(yaw), sy = Math.sin(yaw), cr = Math.cos(roll), sr = Math.sin(roll);
    // Ry(yaw) * Rz(roll) * s : local +z looks along (sin yaw, 0, cos yaw)
    const L = [cy * cr * s, -cy * sr * s, sy * s, sr * s, cr * s, 0, -sy * cr * s, sy * sr * s, cy * s, pos[0], pos[1], pos[2]];
    const P = this.M;
    this.stack.push(P);
    if (!P) { this.M = L; return this; }
    const m = new Array(12);
    for (let r = 0; r < 3; r++) {
      for (let c = 0; c < 3; c++) m[r * 3 + c] = P[r * 3] * L[c] + P[r * 3 + 1] * L[3 + c] + P[r * 3 + 2] * L[6 + c];
      m[9 + r] = P[r * 3] * L[9] + P[r * 3 + 1] * L[10] + P[r * 3 + 2] * L[11] + P[9 + r];
    }
    this.M = m; return this;
  }
  pop() { this.M = this.stack.pop(); return this; }
  X(v) { const M = this.M; return M ? [M[0] * v[0] + M[1] * v[1] + M[2] * v[2] + M[9], M[3] * v[0] + M[4] * v[1] + M[5] * v[2] + M[10], M[6] * v[0] + M[7] * v[1] + M[8] * v[2] + M[11]] : v; }
  D(v) { const M = this.M; if (!M) return v; const x = M[0] * v[0] + M[1] * v[1] + M[2] * v[2], y = M[3] * v[0] + M[4] * v[1] + M[5] * v[2], z = M[6] * v[0] + M[7] * v[1] + M[8] * v[2], l = Math.hypot(x, y, z) || 1; return [x / l, y / l, z / l]; }
  tri(a, b, c, col, hint, nrm, uvs) {
    if (this.M) { a = this.X(a); b = this.X(b); c = this.X(c); if (hint) hint = this.D(hint); if (nrm) nrm = nrm.map((q) => this.D(q)); }
    let n = [(b[1] - a[1]) * (c[2] - a[2]) - (b[2] - a[2]) * (c[1] - a[1]), (b[2] - a[2]) * (c[0] - a[0]) - (b[0] - a[0]) * (c[2] - a[2]), (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])];
    const l = Math.hypot(n[0], n[1], n[2]);
    if (l < 1e-10) return;
    n = [n[0] / l, n[1] / l, n[2] / l];
    let v = [a, b, c], ns = nrm, us = uvs;
    if (hint && n[0] * hint[0] + n[1] * hint[1] + n[2] * hint[2] < 0) { v = [a, c, b]; n = [-n[0], -n[1], -n[2]]; if (ns) ns = [ns[0], ns[2], ns[1]]; if (us) us = [us[0], us[2], us[1]]; }
    for (let k = 0; k < 3; k++) {
      this.p.push(v[k][0], v[k][1], v[k][2]); const q = ns ? ns[k] : n; this.n.push(q[0], q[1], q[2]); this.c.push(col[0], col[1], col[2]); this.w.push(this.wv);
      if (this.uv) { const u = us ? us[k] : [0, 0]; this.uv.push(u[0], u[1]); }
    }
  }
  quad(a, b, c, d, col, hint, nrm, uvs) { this.tri(a, b, c, col, hint, nrm && [nrm[0], nrm[1], nrm[2]], uvs && [uvs[0], uvs[1], uvs[2]]); this.tri(a, c, d, col, hint, nrm && [nrm[0], nrm[2], nrm[3]], uvs && [uvs[0], uvs[2], uvs[3]]); }
  // a, b, c, d = bottom left, bottom right, top right, top left of the rectangle r = [u0, v0, u1, v1]
  quadUV(a, b, c, d, r, hint, col = [1, 1, 1]) { this.quad(a, b, c, d, col, hint, null, [[r[0], r[1]], [r[2], r[1]], [r[2], r[3]], [r[0], r[3]]]); }
  get count() { return this.p.length / 9; }
  geometry(ox = 0, oy = 0, oz = 0) {
    const g = new THREE.BufferGeometry(), p = new Float32Array(this.p);
    for (let i = 0; i < p.length; i += 3) { p[i] += ox; p[i + 1] += oy; p[i + 2] += oz; }
    g.setAttribute('position', new THREE.BufferAttribute(p, 3));
    g.setAttribute('normal', new THREE.BufferAttribute(new Float32Array(this.n), 3));
    g.setAttribute('color', new THREE.BufferAttribute(new Float32Array(this.c), 3));
    g.setAttribute('aW', new THREE.BufferAttribute(new Float32Array(this.w), 1));
    if (this.uv) g.setAttribute('uv', new THREE.BufferAttribute(new Float32Array(this.uv), 2));
    g.computeBoundingSphere(); g.computeBoundingBox();
    return g;
  }
}

export function addBox(B, s, col) {
  const { x0, x1, y0, y1, z0, z1 } = s;
  B.quad([x1, y0, z1], [x1, y0, z0], [x1, y1, z0], [x1, y1, z1], col, [1, 0, 0]);
  B.quad([x0, y0, z0], [x0, y0, z1], [x0, y1, z1], [x0, y1, z0], col, [-1, 0, 0]);
  B.quad([x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1], col, [0, 0, 1]);
  B.quad([x1, y0, z0], [x0, y0, z0], [x0, y1, z0], [x1, y1, z0], col, [0, 0, -1]);
  B.quad([x0, y1, z1], [x1, y1, z1], [x1, y1, z0], [x0, y1, z0], col, [0, 1, 0]);
  B.quad([x0, y0, z0], [x1, y0, z0], [x1, y0, z1], [x0, y0, z1], col, [0, -1, 0]);
}
export const pbox = (B, x0, x1, y0, y1, z0, z1, col) => addBox(B, { x0, x1, y0, y1, z0, z1 }, col);
export function addCyl(B, s, col) {
  const n = s.n || Math.max(10, Math.min(32, Math.round(Math.max(s.r, s.r1) * 2.2))), dy = s.y1 - s.y0, sl = s.r - s.r1, nl = Math.hypot(dy, sl);
  for (let k = 0; k < n; k++) {
    const a0 = k / n * 6.28319, a1 = (k + 1) / n * 6.28319, c0 = Math.cos(a0), s0 = Math.sin(a0), c1 = Math.cos(a1), s1 = Math.sin(a1);
    const n0 = [c0 * dy / nl, sl / nl, s0 * dy / nl], n1 = [c1 * dy / nl, sl / nl, s1 * dy / nl], mid = [(c0 + c1), 0.2 * Math.sign(sl), (s0 + s1)];
    const A = [s.x + c0 * s.r, s.y0, s.z + s0 * s.r], Bq = [s.x + c1 * s.r, s.y0, s.z + s1 * s.r], Cq = [s.x + c1 * s.r1, s.y1, s.z + s1 * s.r1], Dq = [s.x + c0 * s.r1, s.y1, s.z + s0 * s.r1];
    if (s.r1 > 0.01) B.quad(A, Bq, Cq, Dq, col, mid, [n0, n1, n1, n0]); else B.tri(A, Bq, [s.x, s.y1, s.z], col, mid, [n0, n1, [(n0[0] + n1[0]) / 2, n0[1], (n0[2] + n1[2]) / 2]]);
    if (s.r1 > 0.01) B.tri([s.x, s.y1, s.z], Dq, Cq, col, [0, 1, 0]);
  }
}
export function addGable(B, s, col) {
  const { x0, x1, y0, y1, z0, z1 } = s;
  if (s.axis === 'x') {
    const c = (z0 + z1) / 2;
    B.quad([x0, y0, z0], [x1, y0, z0], [x1, y1, c], [x0, y1, c], col, [0, 1, -1]); B.quad([x0, y0, z1], [x1, y0, z1], [x1, y1, c], [x0, y1, c], col, [0, 1, 1]);
    B.tri([x0, y0, z0], [x0, y0, z1], [x0, y1, c], col, [-1, 0, 0]); B.tri([x1, y0, z0], [x1, y0, z1], [x1, y1, c], col, [1, 0, 0]);
  } else {
    const c = (x0 + x1) / 2;
    B.quad([x0, y0, z0], [x0, y0, z1], [c, y1, z1], [c, y1, z0], col, [-1, 1, 0]); B.quad([x1, y0, z0], [x1, y0, z1], [c, y1, z1], [c, y1, z0], col, [1, 1, 0]);
    B.tri([x0, y0, z0], [x1, y0, z0], [c, y1, z0], col, [0, 0, -1]); B.tri([x0, y0, z1], [x1, y0, z1], [c, y1, z1], col, [0, 0, 1]);
  }
  B.quad([x0, y0, z0], [x1, y0, z0], [x1, y0, z1], [x0, y0, z1], col, [0, -1, 0]);
}
export function addPyr(B, s, col) {
  const { x0, x1, y0, y1, z0, z1 } = s, t = [(x0 + x1) / 2, y1, (z0 + z1) / 2];
  B.tri([x0, y0, z0], [x1, y0, z0], t, col, [0, 1, -1]); B.tri([x0, y0, z1], [x1, y0, z1], t, col, [0, 1, 1]);
  B.tri([x0, y0, z0], [x0, y0, z1], t, col, [-1, 1, 0]); B.tri([x1, y0, z0], [x1, y0, z1], t, col, [1, 1, 0]);
  B.quad([x0, y0, z0], [x1, y0, z0], [x1, y0, z1], [x0, y0, z1], col, [0, -1, 0]);
}
// block with an arched tunnel: (u, y, w) = across, up, along the tunnel
export function addArch(B, s, col, dark) {
  const ax = s.axis === 'x';
  const M = (u, y, w) => (ax ? [w, y, u] : [u, y, w]), Vn = (u, y, w) => (ax ? [w, y, u] : [u, y, w]);
  const u0 = ax ? s.z0 : s.x0, u1 = ax ? s.z1 : s.x1, w0 = ax ? s.x0 : s.z0, w1 = ax ? s.x1 : s.z1, { y0, y1, c, hw, ys } = s;
  const sill = s.yb !== undefined, yb = sill ? s.yb : y0, N = 14, wv = B.wv;
  const pts = []; for (let k = 0; k <= N; k++) { const a = Math.PI * k / N; pts.push([c + hw * Math.cos(a), ys + hw * Math.sin(a), Math.cos(a), Math.sin(a)]); }
  for (const [w, sg] of [[w0, -1], [w1, 1]]) {
    const h = Vn(0, 0, sg);
    B.quad(M(u0, y0, w), M(c - hw, y0, w), M(c - hw, y1, w), M(u0, y1, w), col, h);
    B.quad(M(c + hw, y0, w), M(u1, y0, w), M(u1, y1, w), M(c + hw, y1, w), col, h);
    if (sill) B.quad(M(c - hw, y0, w), M(c + hw, y0, w), M(c + hw, yb, w), M(c - hw, yb, w), col, h);
    for (let k = 0; k < N; k++) B.quad(M(pts[k][0], pts[k][1], w), M(pts[k + 1][0], pts[k + 1][1], w), M(pts[k + 1][0], y1, w), M(pts[k][0], y1, w), col, h);
  }
  B.quad(M(u0, y0, w0), M(u0, y0, w1), M(u0, y1, w1), M(u0, y1, w0), col, Vn(-1, 0, 0));
  B.quad(M(u1, y0, w0), M(u1, y0, w1), M(u1, y1, w1), M(u1, y1, w0), col, Vn(1, 0, 0));
  B.quad(M(u0, y1, w0), M(u1, y1, w0), M(u1, y1, w1), M(u0, y1, w1), col, [0, 1, 0]);
  B.quad(M(u0, y0, w0), M(c - hw, y0, w0), M(c - hw, y0, w1), M(u0, y0, w1), col, [0, -1, 0]);
  B.quad(M(c + hw, y0, w0), M(u1, y0, w0), M(u1, y0, w1), M(c + hw, y0, w1), col, [0, -1, 0]);
  // inside the tunnel: jambs, sill, soffit (a little darker: less sky reaches them; no ivy in there)
  B.wv = 0;
  if (ys > yb) {
    B.quad(M(c - hw, yb, w0), M(c - hw, yb, w1), M(c - hw, ys, w1), M(c - hw, ys, w0), dark, Vn(1, 0, 0));
    B.quad(M(c + hw, yb, w0), M(c + hw, yb, w1), M(c + hw, ys, w1), M(c + hw, ys, w0), dark, Vn(-1, 0, 0));
  }
  if (sill) B.quad(M(c - hw, yb, w0), M(c + hw, yb, w0), M(c + hw, yb, w1), M(c - hw, yb, w1), dark, [0, 1, 0]);
  for (let k = 0; k < N; k++) {
    const a = pts[k], b = pts[k + 1], na = Vn(-a[2], -a[3], 0), nb = Vn(-b[2], -b[3], 0);
    B.quad(M(a[0], a[1], w0), M(a[0], a[1], w1), M(b[0], b[1], w1), M(b[0], b[1], w0), dark, [(na[0] + nb[0]) / 2, (na[1] + nb[1]) / 2, (na[2] + nb[2]) / 2], [na, na, nb, nb]);
  }
  B.wv = wv;
}

// a turned shape: the profile [[radius, y], ...] swept round the y axis through (cx, cz), optionally squashed
export function lathe(B, prof, col, o = {}) {
  const n = o.n || 10, sx = o.sx || 1, sz = o.sz || 1, cx = o.cx || 0, cz = o.cz || 0, cy = o.cy || 0, m = prof.length;
  const sn = []; for (let i = 0; i < m - 1; i++) { const dr = prof[i + 1][0] - prof[i][0], dy = prof[i + 1][1] - prof[i][1], l = Math.hypot(dr, dy) || 1; sn.push([dy / l, -dr / l]); }
  const pn = prof.map((_, i) => { const a = sn[Math.max(0, i - 1)], b = sn[Math.min(m - 2, i)], x = a[0] + b[0], y = a[1] + b[1], l = Math.hypot(x, y) || 1; return [x / l, y / l]; });
  for (let k = 0; k < n; k++) {
    const a0 = k / n * 6.28319, a1 = (k + 1) / n * 6.28319, c0 = Math.cos(a0), s0 = Math.sin(a0), c1 = Math.cos(a1), s1 = Math.sin(a1);
    for (let i = 0; i < m - 1; i++) {
      const [r0, y0] = prof[i], [r1, y1] = prof[i + 1], p = pn[i], q = pn[i + 1];
      const A = [cx + r0 * c0 * sx, cy + y0, cz + r0 * s0 * sz], Bq = [cx + r0 * c1 * sx, cy + y0, cz + r0 * s1 * sz], Cq = [cx + r1 * c1 * sx, cy + y1, cz + r1 * s1 * sz], Dq = [cx + r1 * c0 * sx, cy + y1, cz + r1 * s0 * sz];
      const nA = [p[0] * c0, p[1], p[0] * s0], nB = [p[0] * c1, p[1], p[0] * s1], nC = [q[0] * c1, q[1], q[0] * s1], nD = [q[0] * c0, q[1], q[0] * s0];
      const hint = [sn[i][0] * (c0 + c1) / 2, sn[i][1], sn[i][0] * (s0 + s1) / 2];
      if (r0 < 1e-4) B.tri(A, Cq, Dq, col, hint, [nA, nC, nD]); else if (r1 < 1e-4) B.tri(A, Bq, Cq, col, hint, [nA, nB, nC]); else B.quad(A, Bq, Cq, Dq, col, hint, [nA, nB, nC, nD]);
    }
  }
}
export function ball(B, c, r, col, o = {}) {
  const k = o.k || 5, prof = []; for (let i = 0; i <= k; i++) { const a = -Math.PI / 2 + Math.PI * i / k; prof.push([Math.max(0, r * Math.cos(a)), r * (o.sy || 1) * Math.sin(a)]); }
  prof[0][0] = 0; prof[k][0] = 0;
  lathe(B, prof, col, { n: o.n || 8, cx: c[0], cy: c[1], cz: c[2], sx: o.sx || 1, sz: o.sz || 1 });
}
// a round bar from p0 to p1, radius r0 to r1
export function limb(B, p0, p1, r0, r1, col, n = 6, cap = true) {
  const d = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]], L = Math.hypot(d[0], d[1], d[2]); if (L < 1e-6) return;
  const w = [d[0] / L, d[1] / L, d[2] / L], t = Math.abs(w[1]) < 0.9 ? [0, 1, 0] : [1, 0, 0];
  let u = [w[1] * t[2] - w[2] * t[1], w[2] * t[0] - w[0] * t[2], w[0] * t[1] - w[1] * t[0]]; const ul = Math.hypot(u[0], u[1], u[2]); u = [u[0] / ul, u[1] / ul, u[2] / ul];
  const v = [w[1] * u[2] - w[2] * u[1], w[2] * u[0] - w[0] * u[2], w[0] * u[1] - w[1] * u[0]];
  const R = (a) => [u[0] * Math.cos(a) + v[0] * Math.sin(a), u[1] * Math.cos(a) + v[1] * Math.sin(a), u[2] * Math.cos(a) + v[2] * Math.sin(a)];
  for (let k = 0; k < n; k++) {
    const e0 = R(k / n * 6.28319), e1 = R((k + 1) / n * 6.28319), mid = [e0[0] + e1[0], e0[1] + e1[1], e0[2] + e1[2]];
    const A = [p0[0] + e0[0] * r0, p0[1] + e0[1] * r0, p0[2] + e0[2] * r0], Bq = [p0[0] + e1[0] * r0, p0[1] + e1[1] * r0, p0[2] + e1[2] * r0];
    const Cq = [p1[0] + e1[0] * r1, p1[1] + e1[1] * r1, p1[2] + e1[2] * r1], Dq = [p1[0] + e0[0] * r1, p1[1] + e0[1] * r1, p1[2] + e0[2] * r1];
    if (r1 > 1e-4) B.quad(A, Bq, Cq, Dq, col, mid, [e0, e1, e1, e0]); else B.tri(A, Bq, p1, col, mid, [e0, e1, mid.map((q) => q / 2)]);
    if (cap && r1 > 1e-4) B.tri(p1, Dq, Cq, col, w);
    if (cap && r0 > 1e-4) B.tri(p0, A, Bq, col, [-w[0], -w[1], -w[2]]);
  }
}
// a convex outline in the local xy plane, extruded from z0 to z1
export function slab(B, pts, z0, z1, col, colSide = col) {
  let cx = 0, cy = 0; for (const p of pts) { cx += p[0] / pts.length; cy += p[1] / pts.length; }
  for (let i = 0; i < pts.length; i++) {
    const a = pts[i], b = pts[(i + 1) % pts.length];
    B.tri([cx, cy, z1], [a[0], a[1], z1], [b[0], b[1], z1], col, [0, 0, 1]);
    B.tri([cx, cy, z0], [a[0], a[1], z0], [b[0], b[1], z0], col, [0, 0, -1]);
    B.quad([a[0], a[1], z0], [b[0], b[1], z0], [b[0], b[1], z1], [a[0], a[1], z1], colSide, [(a[0] + b[0]) / 2 - cx, (a[1] + b[1]) / 2 - cy, 0]);
  }
}

export function canvasTex(w, h, draw, repeat = true) {
  const c = document.createElement('canvas'); c.width = w; c.height = h; draw(c.getContext('2d'), w, h);
  const t = new THREE.CanvasTexture(c); t.colorSpace = THREE.SRGBColorSpace; t.anisotropy = 8;
  if (repeat) t.wrapS = t.wrapT = THREE.RepeatWrapping;
  return t;
}
export function rng(seed) { let s = seed >>> 0; return () => ((s = (s * 1664525 + 1013904223) >>> 0) / 4294967296); }
