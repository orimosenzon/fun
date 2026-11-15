// The landscape and the bike's place in it. No rendering here: it also runs in node (tests/flight.mjs).
//
// THE LAND. The Mandelbrot set is a lake at height zero. The ground rises away from it:
//     height(c) = slope * d(c)          d = distance estimate to the set (core/field.js)
// A distance looks the same at every magnification, so the land does too: wherever the bike is
// and however small it has become, there are valleys of its own size around it, with the next
// smaller ones along the shore below.
//
// THE SCALE. The bike flies in metres; S says how much of the plane a metre is. The physics, the
// camera and the mesh all live in a local frame whose origin is the point of the lake surface
// directly under the bike (x east = real axis, z south = minus the imaginary axis, y up).
//
// THE WAY DOWN. The details are at the shore, so going to the details is going down, and the
// altimeter is logarithmic: between K_LO and K_HI metres above the water the bike flies in an
// ordinary world. Below K_LO it does not get any closer to the water: it shrinks instead (S gets
// smaller), so the world grows around it and the water stays K_LO below. Keep the nose down and
// the zoom goes on for as long as you like, 10^30 and more. Above K_HI it grows back.
import { bpAlign, bpAddD, bitsFor } from './fixed.js';
import { iterCount, Budget, Tone } from './field.js';

export const K_LO = 40, K_HI = 120;
export const S_MAX = 1 / 600;        // the largest the bike gets: the whole set is then a lake two kilometres long
export const S_MIN = 2e-34;
export const D0 = 1.5;               // far from the set the land levels off into a plateau (in units of the plane)
export const Y_TOP = 2600;           // ceiling at full size, metres
export const VIEW_M = 250;           // a map of half-height "scale" corresponds to a bike for which scale = VIEW_M metres
// Terraces. Every band of the colouring (one more iteration before escape) is also a step of the
// land: a gently sloping tread, and a short drop towards the lake where the next band begins. So
// the picture one knows from the flat map is the relief itself, band for band.
export const TERR_K = 0.2, TERR_R = 0.22;
export function terrace(sl, dTex) {            // dTex: distance to the set in texels of the height map (bands too thin for the mesh to carry are left smooth)
  const s = sl - Math.floor(sl), e = Math.min(1, Math.max(0, (s - (1 - TERR_R)) / TERR_R)), a = Math.min(1, Math.max(0, (0.6931472 * dTex - 12) / 28));
  return 1 + TERR_K * a * a * (3 - 2 * a) * (Math.min(s / (1 - TERR_R), 1) * (1 - e * e * (3 - 2 * e)) - 0.5);
}
const LMIN = -3;                     // coarsest level: half-extent 8, the whole set and its surroundings
const mod = (a, n) => a - Math.floor(a / n) * n;

export class World {
  // opt: { quality, cap }   tone: shared colour statistics
  constructor(field, tone, opt) {
    this.f = field; this.bud = new Budget(tone, opt); this.opt = opt;
    this.S = S_MAX; this.ox = 0; this.oy = 0;     // origin of the local frame, as an offset from the reference point of the field
    this.slope = 0.5;
    this.N = 512; this.NL = 9; this.H0 = 80;      // texels across a level, number of levels, least half-extent of the finest one (m)
    this.levels = [];                             // fine -> coarse; see plan()
    this.l0 = null; this.free = [];
    this.cache = new Map(); this.lat = null;
    this.onShift = null;
    this._o = {}; this.evals = 0;
    this.setRes(512, 9);
  }
  setRes(N, NL) { this.N = N; this.NL = NL; this.levels = []; this.free = []; for (let i = 0; i < NL + 2; i++) this.free.push(i); this.l0 = null; this.cache.clear(); this.lat = null; }

  // ---------- where ----------
  // put the origin exactly on a point of the plane
  place(center, S) {
    this.S = Math.min(S_MAX, Math.max(S_MIN, S));
    const h0 = this.h0();
    this.f.build(bpAlign(center, bitsFor(h0) + 32), h0, this.bud.probe);
    this.ox = this.oy = 0;
    this.setRes(this.N, this.NL); this.bud.forget(); this.bud.tone.reset();
  }
  origin() { return bpAddD(this.f.ref.center, this.ox, this.oy); }
  // half-extent of the finest level, in units of the plane: a power of two between H0 and 2 H0 metres
  h0() {
    const want = this.H0 * this.S;
    if (this.l0 === null || 2 ** -this.l0 < want * 0.92 || 2 ** -this.l0 > want * 2.2) this.l0 = Math.max(LMIN, Math.floor(-Math.log2(want)));
    return 2 ** -this.l0;
  }
  // keep the reference orbit near the bike, deep enough and long enough. Returns true when it was rebuilt.
  ensureRef() {
    const f = this.f, h0 = this.h0(), probe = this.bud.probe;
    if (f.ref && Math.max(Math.abs(this.ox), Math.abs(this.oy)) < 24 * h0 && f.ref.center.bits >= bitsFor(h0) + 32 && (f.ref.escaped || f.len >= probe + 2)) return false;
    const sx = this.ox, sy = this.oy;
    f.build(bpAddD(bpAlign(f.ref.center, bitsFor(h0) + 32), sx, sy), h0, probe);
    this.ox = this.oy = 0;
    // everything that was measured from the old reference point moves with it
    for (const L of this.levels) { const g = 8 * L.tau; L.cx -= sx; L.cy -= sy; L.phx = mod(L.phx - sx, g); L.phy = mod(L.phy - sy, g); if (L.job) { L.job.cx -= sx; L.job.cy -= sy; } }
    this.cache.clear(); this.lat = null;
    if (this.onShift) this.onShift(sx, sy);
    return true;
  }
  // a few points near the bike: colour levels and iteration budget, as on the map
  sample(n) {
    const h = this.h0() * 1.5, o = this._o, probe = this.bud.probe;
    for (let k = 0; k < n; k++) {
      const a = Math.random() * 6.2832, r = Math.sqrt(Math.random()) * h;
      this.f.eval(this.ox + Math.cos(a) * r, this.oy + Math.sin(a) * r, probe, o);
      this.bud.note(o.sl);
    }
    return this.bud.settle(this.h0());
  }

  // Colours. On a flat map one pair of numbers (where the colours start, how far they spread)
  // serves the whole picture. From the bike one sees many scales at once: the ground below, and
  // mountains a thousand times larger on the horizon, whose points escape far sooner. So every
  // level keeps its own pair, from points sampled all over it, and the picture blends from one
  // level's colours to the next as it blends their heights.
  sampleLevels(n, dt) {
    const o = this._o;
    for (const L of this.levels) {
      const probe = Math.min(this.bud.probe, 4 * iterCount(L.h) + 2000);
      for (let k = 0; k < n; k++) { this.f.eval(this.ox + (Math.random() * 2 - 1) * L.h, this.oy + (Math.random() * 2 - 1) * L.h, probe, o); L.tone.push(o.sl); }
      L.tone.settle(dt, 0.9);
    }
  }

  // ---------- the ground under the bike ----------
  // Heights come from the same lattice of points that the finest level of the picture is made
  // of (its texel centres), joined bilinearly as the GPU joins them: physics = picture.
  lattice() {
    if (this.lat) return this.lat;
    const L = this.levels[0];
    this.lat = L && L.l === this.l0 ? { tau: L.tau, phx: L.phx, phy: L.phy } : { tau: 2 * 2 ** -this.l0 / this.N, phx: 0, phy: 0 };
    this.cache.clear();
    return this.lat;
  }
  // field at a lattice point: [distance, smooth count], remembered
  latAt(ix, iy) {
    const key = (ix + 1048576) * 2097152 + (iy + 1048576);
    let v = this.cache.get(key);
    if (v === undefined) {
      const t = this.lat, o = this._o;
      this.f.eval(t.phx + (ix + 0.5) * t.tau, t.phy + (iy + 0.5) * t.tau, this.bud.iter, o);
      v = [o.d, o.sl < 0 ? this.bud.iter + 3 : o.sl]; this.evals++;
      if (this.cache.size > 60000) this.cache.clear();
      this.cache.set(key, v);
    }
    return v;
  }
  // height above the lake (m) at local metres (x, z); 0 on the water
  heightAt(x, z) {
    const t = this.lattice();
    const fx = (this.ox + this.S * x - t.phx) / t.tau - 0.5, fy = (this.oy - this.S * z - t.phy) / t.tau - 0.5;
    const ix = Math.floor(fx), iy = Math.floor(fy), u = fx - ix, v = fy - iy;
    const a = this.latAt(ix, iy), b = this.latAt(ix + 1, iy), c = this.latAt(ix, iy + 1), d = this.latAt(ix + 1, iy + 1);
    const dd = (a[0] + (b[0] - a[0]) * u) * (1 - v) + (c[0] + (d[0] - c[0]) * u) * v;
    if (dd <= 0) return 0;
    return this.hOf(dd, (a[1] + (b[1] - a[1]) * u) * (1 - v) + (c[1] + (d[1] - c[1]) * u) * v, t.tau);
  }
  hOf(d, sl, tau = this.lattice().tau) { return this.slope * d / (1 + d / D0) / this.S * terrace(sl, d / tau); }
  groundAt(x, z) { const h = this.heightAt(x, z); return h > 0 ? h : -3; }       // (the lake is three metres deep, as far as anything that falls in can tell)
  waterAt() { return 0; }
  normalAt(x, z) {
    const e = this.lattice().tau / this.S;
    const nx = this.heightAt(x - e, z) - this.heightAt(x + e, z), nz = this.heightAt(x, z - e) - this.heightAt(x, z + e);
    const l = Math.hypot(nx, 2 * e, nz);
    return [nx / l, 2 * e / l, nz / l];
  }
  // which way is downhill (towards the set), as a yaw for the bike; null over the water
  downhill() {
    const o = this._o; this.f.eval(this.ox, this.oy, this.bud.probe, o);
    return o.sl < 0 ? null : Math.atan2(-o.uy, -o.ux);          // forward = (cos yaw, 0, -sin yaw), east = +re, north = +im = -z
  }

  // ---------- following the bike ----------
  // After the physics has moved the bike: bring the origin back under it and apply the
  // logarithmic altimeter. p is the bike's position (changed in place). Returns what the
  // caller must do to everything else that lives in local metres: subtract (sx, sz), divide by f.
  follow(p, landed, dt) {
    const sx = p[0], sz = p[2];
    this.ox += this.S * sx; this.oy -= this.S * sz; p[0] = 0; p[2] = 0;
    let f = 1;
    const y = Math.max(p[1], 1);
    if (!landed) { if (y < K_LO) f = y / K_LO; else if (y > K_HI) f = y / K_HI; }
    f = Math.min(1 + 3 * dt, Math.max(1 - 3 * dt, f));
    const S2 = Math.min(S_MAX, Math.max(S_MIN, this.S * f));
    f = S2 / this.S; this.S = S2;
    p[1] /= f;
    this.zooming = f < 0.9999 ? -1 : f > 1.0001 ? 1 : 0;
    return { sx, sz, f };
  }

  // ---------- the levels of the picture ----------
  // The ground is drawn from a pyramid of square height maps around the bike, each twice the size
  // of the one before (level l has half-extent 2^-l). Their texels are fixed in the plane, so the
  // ground does not swim as the bike moves; a level is recomputed (around a new centre) when the
  // bike has left its middle. plan() says what to compute next:
  //    { L, cx, cy, iter, layer }   or null when everything is in place.
  plan(fx = 0, fy = 0) {
    const h0 = this.h0(), l0 = this.l0, lv = this.levels;
    const lEnd = Math.max(LMIN, l0 - this.NL + 1);
    // levels that are no longer wanted give their layers back
    for (let i = lv.length - 1; i >= 0; i--) if (lv[i].l > l0 || lv[i].l < lEnd) { const L = lv[i]; if (L.layer >= 0) this.free.push(L.layer); if (L.job) this.free.push(L.job.layer); lv.splice(i, 1); }
    for (let l = l0; l >= lEnd; l--) if (!lv.some((L) => L.l === l)) {
      const h = 2 ** -l, tone = new Tone(72), near = lv.length ? lv.reduce((a, b) => (Math.abs(b.l - l) < Math.abs(a.l - l) ? b : a)) : null;
      if (near) { tone.lo = tone.tLo = near.tone.lo; tone.w = tone.tW = near.tone.w; }       // until its own points come in, the colours of the level next to it
      lv.push({ l, h, tau: 2 * h / this.N, cx: 0, cy: 0, phx: 0, phy: 0, valid: false, layer: -1, fade: 0, iter: 0, job: null, tone });
    }
    lv.sort((a, b) => b.l - a.l);
    if (lv[0].tau !== (this.lat && this.lat.tau)) this.lat = null;
    if (lv.some((L) => L.job)) return null;                       // one at a time
    let best = null, score = 0;
    for (let i = 0; i < lv.length; i++) {
      const L = lv[i], g = 8 * L.tau;
      // centred a little ahead of the bike: that is where one looks
      const tx = this.ox + fx * 0.3 * L.h, ty = this.oy + fy * 0.3 * L.h;
      const cx = L.phx + Math.round((tx - L.phx) / g) * g, cy = L.phy + Math.round((ty - L.phy) / g) * g;
      const iter = Math.min(this.bud.iter, 4 * iterCount(L.h) + 2000);
      let s = 0;
      if (!L.valid) s = 100 + (lv.length - i);                    // nothing there yet: the coarse ones first, they cover the most
      else {
        const r = Math.max(Math.abs(tx - L.cx), Math.abs(ty - L.cy)) / L.h;
        if (r > 0.22) s = 10 * r + (lv.length - i) * 0.01;
        else if (iter > 1.3 * L.iter) s = 1;                      // the place turned out to need more iterations than this level got
      }
      if (s > score) { score = s; best = { L, cx, cy, iter }; }
    }
    if (!best) return null;
    best.layer = this.free.pop();
    best.L.job = best;
    return best;
  }
  finish(job) {
    const L = job.L;
    if (L.job !== job) return;                                    // the level was dropped meanwhile
    if (L.layer >= 0) this.free.push(L.layer); else L.fade = this.levels.some((x) => x.valid) ? 0 : 1;
    L.layer = job.layer; L.cx = job.cx; L.cy = job.cy; L.iter = job.iter; L.valid = true; L.job = null;
  }
}
