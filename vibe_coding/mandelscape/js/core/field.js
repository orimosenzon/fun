// The field: for a point c of the plane, how fast it escapes and how far it is from the set.
//
//   sl  smooth iteration count (the classic colouring value; continuous across the bands)
//   d   distance estimate to the Mandelbrot set, d = |z| ln|z| / |dz/dc|, in units of the plane.
//       It is zero on the set and grows about linearly away from it, at every scale alike.
//       The map uses it to draw the filaments as crisp lines, the landscape uses it as HEIGHT.
//
// Deep zooms cannot be computed in doubles directly (the pixels differ in the 30th digit). So one
// point, the "reference", is iterated exactly (BigInt fixed point) and every other point is
// iterated as a small difference from that orbit ("perturbation"), which ordinary floats carry
// well. When the difference grows larger than the value itself the point is re-based onto the
// beginning of the reference orbit (Zhuoran's rebasing), which removes the classic glitches.
//
// This file is the CPU side: it builds the reference and evaluates single points in doubles (for
// the physics of the bike, for the colour statistics, for tests). gpu/fieldpass.js does the same
// arithmetic in a shader for whole images. Both must stay in step.
import { bpAlign, bitsFor, bpToD } from './fixed.js';

export const R2 = 4096;          // bail-out radius squared (64^2): large, so that the estimates are smooth
export const SLC = 3;            // constant in the smooth count, the same in the shader
const REF_BAIL = 1e6;            // the reference runs on a little past the bail-out, so that points near it can escape along it
// Skipping ("bilinear approximation"): while a point is still very close to the reference, its
// difference d obeys d' = 2 Z d + dc almost exactly (the d^2 term is below rounding). Such steps
// are linear, so 2, 4, 8 ... of them fold into one: d' = A d + B dc. The table below holds A, B
// for every aligned block of the reference orbit, and the two radii inside which the block may
// be used (r0 for |d|, rc for |dc|). A deep point then takes a few hundred steps, not 40,000.
const BLA_EPS = 1e-6, BLA_BIG = 1e30, BLA_ZMAX2 = 64;
export const BLA_W = 2048, BLA_LEVELS = 17;

export function iterCount(scale, quality = 1, cap = 12000) {
  const zoom = Math.max(1, 1.35 / scale);
  const n = Math.round((130 + 95 * Math.pow(Math.log10(zoom) + 1, 1.35)) * quality);
  return Math.min(cap, Math.max(80, n));
}

export class Field {
  constructor() { this.ref = null; this.version = 0; }

  // exact orbit of centre (a fixed-point point), at most maxLen terms
  build(center, scale, iter) {
    const bits = Math.max(center.bits, bitsFor(scale) + 32);
    const c = bpAlign(center, bits);
    const maxLen = iter + 2;
    const B = BigInt(bits), bail = BigInt(REF_BAIL) << B;
    const cx = c.x, cy = c.y, sh = B - 53n, inv = 2 ** -53;
    const toF = (v) => Number(v >> sh) * inv;
    const arr = new Float64Array(maxLen * 2);
    let zx = 0n, zy = 0n, zx2 = 0n, zy2 = 0n, len = 1, escaped = false;
    for (let n = 1; n < maxLen; n++) {
      const zxy = (zx * zy) >> B;
      zx = zx2 - zy2 + cx; zy = (zxy << 1n) + cy;
      arr[2 * n] = toF(zx); arr[2 * n + 1] = toF(zy);
      len = n + 1;
      zx2 = (zx * zx) >> B; zy2 = (zy * zy) >> B;
      if (zx2 + zy2 > bail) { escaped = true; break; }
    }
    this.ref = { center: c, arr, len, maxLen, escaped, scale };
    this.arr = arr; this.len = len;
    this.c0 = bpToD(c);
    this.buildBla();
    this.version++;
    return this.ref;
  }

  // Block (l, j), l >= 1, folds the 2^l steps that start at orbit index 1 + j 2^l.
  // (Index 0 is left out: Z = 0 there, nothing is linear about that step.)
  buildBla() {
    const A = this.arr, L = this.len, steps = Math.max(0, L - 2);
    const off = new Int32Array(BLA_LEVELS + 1), cnt = new Int32Array(BLA_LEVELS + 1);
    let total = 0;
    for (let l = 1; l <= BLA_LEVELS; l++) { cnt[l] = Math.floor(steps / 2 ** l); off[l] = total; total += cnt[l]; }
    const rows = Math.max(1, Math.ceil(total / BLA_W));
    const ab = new Float32Array(rows * BLA_W * 4), rr = new Float32Array(rows * BLA_W * 4);
    // level 0 lives only in these scratch arrays; every level is made from the one below
    let n0 = steps, ax = new Float64Array(n0), ay = new Float64Array(n0), bx = new Float64Array(n0), by = new Float64Array(n0), r0 = new Float64Array(n0), rc = new Float64Array(n0);
    for (let k = 0; k < n0; k++) {
      const m = k + 1, zx = A[2 * m], zy = A[2 * m + 1], nx = A[2 * m + 2], ny = A[2 * m + 3];
      ax[k] = 2 * zx; ay[k] = 2 * zy; bx[k] = 1; by[k] = 0; rc[k] = Infinity;
      r0[k] = nx * nx + ny * ny > BLA_ZMAX2 ? 0 : 2 * BLA_EPS * Math.hypot(zx, zy);   // where the reference itself is leaving, every step is taken singly: the escape must be seen on time
    }
    let maxL = 0;
    for (let l = 1; l <= BLA_LEVELS && cnt[l] > 0; l++) {
      const n = cnt[l], Ax = new Float64Array(n), Ay = new Float64Array(n), Bx = new Float64Array(n), By = new Float64Array(n), R0 = new Float64Array(n), Rc = new Float64Array(n);
      let any = false;
      for (let j = 0; j < n; j++) {
        const x = 2 * j, y = x + 1;
        // y after x:  A = Ay Ax,  B = Ay Bx + By
        const a = ax[y] * ax[x] - ay[y] * ay[x], b = ax[y] * ay[x] + ay[y] * ax[x];
        const c = ax[y] * bx[x] - ay[y] * by[x] + bx[y], d = ax[y] * by[x] + ay[y] * bx[x] + by[y];
        const aX = Math.hypot(ax[x], ay[x]), bX = Math.hypot(bx[x], by[x]);
        let q0 = Math.min(r0[x], r0[y] / (2 * aX)), qc = Math.min(rc[x], rc[y], r0[y] / (2 * bX));
        const big = Math.max(Math.abs(a), Math.abs(b), Math.abs(c), Math.abs(d));
        if (!(big < BLA_BIG) || !(q0 > 1e-37) || !(qc > 1e-37)) { q0 = 0; qc = 0; }
        Ax[j] = a; Ay[j] = b; Bx[j] = c; By[j] = d; R0[j] = q0; Rc[j] = qc;
        const o = (off[l] + j) * 4;
        if (q0 > 0) { ab[o] = a; ab[o + 1] = b; ab[o + 2] = c; ab[o + 3] = d; rr[o] = q0; rr[o + 1] = Math.min(qc, 1e37); any = true; }
      }
      if (any) maxL = l;
      ax = Ax; ay = Ay; bx = Bx; by = By; r0 = R0; rc = Rc;
    }
    // the CPU reads the same single-precision numbers the shader does, so both skip alike
    this.bla = { ab, rr, off, cnt, rows, maxL };
  }

  // does the reference still serve a view whose centre is (offX, offY) away from it?
  stale(offX, offY, scale, iter, maxOff = 40) {
    const r = this.ref;
    if (!r) return true;
    if (Math.max(Math.abs(offX), Math.abs(offY)) > maxOff * scale) return true;
    if (r.center.bits < bitsFor(scale) + 32) return true;
    if (!r.escaped && r.len < iter + 1) return true;
    return false;
  }

  // One point, (dcx, dcy) away from the reference centre. Fills out.sl, out.d, out.ux, out.uy
  // (ux, uy: unit vector of the gradient, for relief shading). Inside the set: sl = -1, d = 0.
  eval(dcx, dcy, maxIter, out) {
    const A = this.arr, L = this.len;
    // the two big components of the set have closed formulas: no need to iterate inside them
    const cx = this.c0[0] + dcx, cy = this.c0[1] + dcy;
    if (this.ref.scale > 1e-9 || Math.abs(dcx) + Math.abs(dcy) > 1e-9) {
      const xq = cx - 0.25, q = xq * xq + cy * cy;
      if (q * (q + xq) < 0.25 * cy * cy * 0.999999 || (cx + 1) * (cx + 1) + cy * cy < 0.0625 * 0.999999) { out.sl = -1; out.d = 0; out.ux = 0; out.uy = 0; return out; }
    }
    let dzx = 0, dzy = 0, zx = 0, zy = 0, drx = 0, dry = 0, m = 0;
    const bla = this.bla, maxL = this.useBla === false ? 0 : bla.maxL, ab = bla.ab, rr = bla.rr, off = bla.off, cnt = bla.cnt;
    const dcA = Math.max(Math.abs(dcx), Math.abs(dcy)) * 1.4143;
    let steps = 0;
    for (let i = 1; i <= maxIter; i++) {
      steps++;
      if (m > 0 && maxL > 0) {
        const q = m - 1, dA = Math.max(Math.abs(dzx), Math.abs(dzy)) * 1.4143;
        let hit = false;
        for (let l = maxL; l >= 1; l--) {
          if (q & ((1 << l) - 1)) continue;
          const j = q >> l;
          if (j >= cnt[l]) continue;
          const o = (off[l] + j) * 4;
          if (!(dA < rr[o] && dcA < rr[o + 1])) continue;
          const a = ab[o], b = ab[o + 1], c = ab[o + 2], d = ab[o + 3];
          const nx = a * dzx - b * dzy + c * dcx - d * dcy; dzy = a * dzy + b * dzx + c * dcy + d * dcx; dzx = nx;
          const tx = a * drx - b * dry + c; dry = a * dry + b * drx + d; drx = tx;
          m += 1 << l; i += (1 << l) - 1;
          zx = A[2 * m] + dzx; zy = A[2 * m + 1] + dzy;
          if (m >= L - 1) { dzx = zx; dzy = zy; m = 0; }
          hit = true; break;
        }
        if (hit) continue;
      }
      const t = 2 * (zx * drx - zy * dry) + 1; dry = 2 * (zx * dry + zy * drx); drx = t;
      const Zx = A[2 * m], Zy = A[2 * m + 1];
      const nx = 2 * (Zx * dzx - Zy * dzy) + dzx * dzx - dzy * dzy + dcx;
      dzy = 2 * (Zx * dzy + Zy * dzx) + 2 * dzx * dzy + dcy; dzx = nx;
      m++;
      zx = A[2 * m] + dzx; zy = A[2 * m + 1] + dzy;
      const zz = zx * zx + zy * zy;
      if (zz > R2) {
        const lz = 0.5 * Math.log(zz), dr = Math.hypot(drx, dry);
        out.sl = i - Math.log2(lz) + SLC;
        out.d = Math.sqrt(zz) * lz / dr;
        // z / z' : points away from the set
        const ux = zx * drx + zy * dry, uy = zy * drx - zx * dry, ul = Math.hypot(ux, uy) || 1;
        out.ux = ux / ul; out.uy = uy / ul; out.steps = steps;
        return out;
      }
      if (zz < dzx * dzx + dzy * dzy || m >= L - 1) { dzx = zx; dzy = zy; m = 0; }
    }
    out.sl = -1; out.d = 0; out.ux = 0; out.uy = 0; out.steps = steps;
    return out;
  }
}

// plain doubles, no reference: the yardstick for the tests (good down to scales of about 1e-12)
export function evalDirect(cx, cy, maxIter, out = {}) {
  let zx = 0, zy = 0, drx = 0, dry = 0;
  for (let i = 1; i <= maxIter; i++) {
    const t = 2 * (zx * drx - zy * dry) + 1; dry = 2 * (zx * dry + zy * drx); drx = t;
    const nx = zx * zx - zy * zy + cx; zy = 2 * zx * zy + cy; zx = nx;
    const zz = zx * zx + zy * zy;
    if (zz > R2) { const lz = 0.5 * Math.log(zz); out.sl = i - Math.log2(lz) + SLC; out.d = Math.sqrt(zz) * lz / Math.hypot(drx, dry); return out; }
  }
  out.sl = -1; out.d = 0; return out;
}

// Statistics of a neighbourhood, from a few hundred points sampled on the CPU:
//   lo, w   the level the colours start from and their spread. The palette is indexed by
//           log(1 + (sl - lo) / w), which gives the same richness of colour at every depth.
//   hi      how long the slow points take to escape: the iteration budget follows it, so a
//           deep place that needs 40,000 iterations gets them and a shallow one is not charged.
export class Tone {
  constructor(n = 192) { this.buf = new Float64Array(n); this.n = 0; this.i = 0; this.lo = 2; this.w = 6; this.tLo = 2; this.tW = 6; this.hi = 0; this._s = new Float64Array(n); }
  reset() { this.n = 0; this.i = 0; }
  push(sl) { if (sl < 0) return; this.buf[this.i] = sl; this.i = (this.i + 1) % this.buf.length; if (this.n < this.buf.length) this.n++; }
  // dt: seconds since the last call; the colours follow their targets smoothly
  settle(dt, tau = 0.45) {
    if (this.n >= 12) {
      const s = this._s.subarray(0, this.n); s.set(this.buf.subarray(0, this.n)); s.sort();
      const lo = s[Math.floor(this.n * 0.04)], mid = s[Math.floor(this.n * 0.55)];
      this.tLo = lo; this.tW = Math.max(1.5, mid - lo); this.hi = s[Math.floor(this.n * 0.9)];
    }
    const k = dt <= 0 ? 1 : 1 - Math.exp(-dt / tau);
    this.lo += (this.tLo - this.lo) * k; this.w += (this.tW - this.w) * k;
    return Math.abs(this.tLo - this.lo) > 1e-3 * this.w || Math.abs(this.tW - this.w) > 1e-3 * this.w;
  }
  // iterations a picture of this neighbourhood needs
  need(scale, quality, cap) { return Math.min(cap, Math.round(Math.max(iterCount(scale), 1.6 * this.hi + 250) * quality)); }
}

// The iteration budget of a view, kept from the points the CPU samples in it (map and flight alike).
// The samples are followed well past what the picture uses ("probe"), to notice when it needs more.
export class Budget {
  constructor(tone, opt) { this.tone = tone; this.opt = opt; this.iter = 300; this.boost = 0; this.n = 0; this.miss = 0; this.found = 0; this.from = 0; this.fruitless = 0; }
  get probe() { return Math.min(this.opt.cap, Math.max(2 * this.iter + 1000, this.boost)); }
  // one sampled point
  note(sl) {
    this.tone.push(sl);
    this.n++; if (sl < 0) this.miss++; else if (sl > this.from) this.found++;
    // Many points that never escaped: either the view is inside the set, or the place is slower
    // than the probe. Double the probe and see whether that finds anything; give up after two
    // doublings in a row that found nothing (then it really is the inside).
    if (this.n < 60) return;
    const probe = this.probe;
    if (this.from > 0) this.fruitless = this.found < 0.03 * this.n ? this.fruitless + 1 : 0;
    if (this.miss > 0.25 * this.n && probe < this.opt.cap && this.fruitless < 2) { this.from = probe; this.boost = Math.min(this.opt.cap, probe * 2); } else this.from = 0;
    this.n = this.miss = this.found = 0;
  }
  // a new place: whatever was learnt about the old one no longer holds
  forget() { this.fruitless = 0; this.from = 0; this.n = this.miss = this.found = 0; this.boost = 0; }
  // true when the iteration count changed. It grows at once and shrinks only when clearly too
  // large, because every change means computing the picture again.
  settle(scale, floor = 0) {
    const need = Math.max(this.tone.need(scale, this.opt.quality, this.opt.cap), floor);
    if (need > this.iter || need < 0.55 * this.iter) { this.iter = Math.min(this.opt.cap, Math.round(need * 1.15)); return true; }
    return false;
  }
}
