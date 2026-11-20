// Bakes the alpine world: an authored valley network + ridged mountains, then hydraulic
// (droplet) erosion, lakes, rivers and landing pads.
//   node tools/bake.mjs          -> assets/terrain.bin, assets/backdrop.bin, assets/meta.json
// The game and the headless tests read the same files, so the ground the physics touches is
// exactly the ground that is drawn.
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { fbm, ridged, gnoise, mulberry32 } from '../js/sim/noise.js';
import { SITE, FORECOURT, siteHeight } from '../js/sim/castleSite.js';
import { LAKE2, ISLE, HOLE, ringSiteHeight, lakeNorm } from '../js/sim/ringSite.js';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const N = 1025, CELL = 12, HALF = (N - 1) * CELL / 2;       // 12.288 km square
const BN = 385, BCELL = 96, BHALF = (BN - 1) * BCELL / 2;   // 36.9 km backdrop
const HSCALE = 4000 / 65535;
const clamp = (x, a, b) => Math.min(b, Math.max(a, x));
const sstep = (a, b, x) => { const t = clamp((x - a) / (b - a), 0, 1); return t * t * (3 - 2 * t); };
const smin = (a, b, k) => { const h = clamp(0.5 + 0.5 * (b - a) / k, 0, 1); return b + (a - b) * h - k * h * (1 - h); };

// ---------- authored valleys: [x km, z km, floor m, half-width m] ----------
const VALLEYS = {
  main: { kind: 'U', slope: 0.60, nodes: [
    [-20, 3.0, 1130, 380], [-6.5, 2.2, 1052, 380], [-4.5, 1.6, 1026, 420], [-2.5, 2.4, 996, 480],
    [-0.5, 1.4, 966, 450], [1.2, 1.9, 930, 520], [2.6, 1.3, 926, 560], [4.0, 0.9, 925, 440],
    [5.2, 1.3, 886, 380], [6.5, 1.1, 862, 360], [20, 0.5, 780, 400]] },
  gorge: { kind: 'V', slope: 0.55, nodes: [
    [-0.5, 1.4, 966, 16], [-0.65, 0.55, 1035, 15], [-0.15, -0.35, 1150, 13], [-0.85, -1.25, 1330, 13],
    [-0.45, -2.15, 1560, 18], [-0.35, -2.9, 1790, 70], [-0.3, -3.5, 1850, 380]] },
  pass: { kind: 'U', slope: 0.72, nodes: [
    [-0.3, -3.5, 1850, 300], [-1.3, -3.75, 2150, 90], [-2.2, -3.6, 2420, 70], [-3.2, -2.8, 2080, 120],
    [-4.1, -1.4, 1650, 200], [-4.5, 0.2, 1250, 180], [-4.5, 1.6, 1026, 300]] },
  east: { kind: 'U', slope: 0.66, nodes: [
    [3.3, 1.0, 925, 200], [4.0, -0.6, 1180, 160], [3.6, -1.9, 1480, 220], [3.0, -2.9, 1640, 420]] },
};
const PEAKS = [ // x km, z km, extra height, radius m
  [1.3, -2.2, 950, 1100], [-2.4, -1.2, 500, 1300], [-2.3, 4.6, 700, 2200], [1.5, 4.8, 800, 2000],
  [4.9, -3.8, 600, 1800], [-5.2, -4.4, 500, 1800], [5.0, 3.6, 450, 1500], [-0.2, -5.2, 550, 1500],
];
const LAKE = { level: 922, x0: 1150, x1: 4060 };
const TARN = { level: 1856, c: [-300, -3520], r: 270 };

function catmull(nodes, step) {
  const out = [];
  const P = (i) => nodes[clamp(i, 0, nodes.length - 1)];
  for (let i = 0; i < nodes.length - 1; i++) {
    const p0 = P(i - 1), p1 = P(i), p2 = P(i + 1), p3 = P(i + 2);
    const L = Math.hypot(p2[0] - p1[0], p2[1] - p1[1]) * 1000;
    const n = Math.max(1, Math.round(L / step));
    for (let j = 0; j < n; j++) {
      const t = j / n, t2 = t * t, t3 = t2 * t;
      const q = [0, 1].map((k) => 500 * ((2 * p1[k]) + (-p0[k] + p2[k]) * t + (2 * p0[k] - 5 * p1[k] + 4 * p2[k] - p3[k]) * t2 + (-p0[k] + 3 * p1[k] - 3 * p2[k] + p3[k]) * t3));
      const s = t * t * (3 - 2 * t);
      out.push([q[0], q[1], p1[2] + (p2[2] - p1[2]) * s, p1[3] + (p2[3] - p1[3]) * s]);
    }
  }
  const e = nodes[nodes.length - 1];
  out.push([e[0] * 1000, e[1] * 1000, e[2], e[3]]);
  return out;
}
for (const v of Object.values(VALLEYS)) v.pts = catmull(v.nodes, 150);

// nearest point on a dense polyline: returns [dist, floor, halfWidth, index+t]
function nearest(pts, x, z) {
  let best = 1e18, bf = 0, bw = 0, bs = 0;
  for (let i = 0; i < pts.length - 1; i++) {
    const a = pts[i], b = pts[i + 1];
    const ex = b[0] - a[0], ez = b[1] - a[1];
    const t = clamp(((x - a[0]) * ex + (z - a[1]) * ez) / (ex * ex + ez * ez), 0, 1);
    const dx = x - a[0] - ex * t, dz = z - a[1] - ez * t;
    const d = dx * dx + dz * dz;
    if (d < best) { best = d; bf = a[2] + (b[2] - a[2]) * t; bw = a[3] + (b[3] - a[3]) * t; bs = i + t; }
  }
  return [Math.sqrt(best), bf, bw, bs];
}

function analytic(x, z) {
  // warp so that valley walls are not perfect offsets of the centre lines
  const wx = x + 130 * fbm(x / 900, z / 900, 3, 71), wz = z + 130 * fbm(x / 900 + 40, z / 900, 3, 72);
  let hv = 1e9, floor = 0;
  for (const v of Object.values(VALLEYS)) {
    // the slot canyon keeps to its centre line (only a few metres of wobble); glacial valleys wander
    const [d, f, w] = v.kind === 'V' ? nearest(v.pts, x + (wx - x) * 0.1, z + (wz - z) * 0.1) : nearest(v.pts, wx, wz);
    const e = Math.max(0, d - w);
    let g;
    if (v.kind === 'V') g = Math.min(2.6 * e, 250 + v.slope * (e - 96));       // slot canyon, then open slopes
    else g = v.slope * e * (0.75 + 0.5 * sstep(0, 500, e)) + 26 * sstep(0, 1, e / 120) * 0;    // glacial U
    const h = f + g;
    if (h < hv) floor = f;
    hv = hv > 1e8 ? h : smin(hv, h, 70);
  }
  let B = 2480 + 520 * fbm(x / 5200, z / 5200, 3, 11) + 260 * sstep(1500, 6000, z) + 200 * sstep(6000, 12000, Math.hypot(x, z));
  for (const [px, pz, a, r] of PEAKS) {
    const q = Math.max(0, 1 - Math.hypot(x - px * 1000, z - pz * 1000) / r);
    B += a * Math.pow(q, 1.6);
  }
  const h0 = smin(hv, B, 240);
  const m = sstep(15, 520, h0 - floor);
  const rx = x + 380 * fbm(x / 2600, z / 2600, 3, 5), rz = z + 380 * fbm(x / 2600 + 9, z / 2600 + 3, 3, 6);
  let h = h0 + m * (ridged(rx / 1750, rz / 1750, 6, 21) - 0.42) * 640;
  h += m * fbm(x / 260, z / 260, 4, 33) * 34;
  h += (1 - m) * (fbm(x / 420, z / 420, 3, 44) * 5 + fbm(x / 90, z / 90, 2, 45) * 0.8);
  return h;
}

console.time('analytic');
const H = new Float32Array(N * N);
for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) H[j * N + i] = analytic(-HALF + i * CELL, -HALF + j * CELL);
const A0 = H.slice();
console.timeEnd('analytic');

// ---------- hydraulic erosion (droplets, after H. Beyer / S. Lague), heights in cells ----------
console.time('erosion');
{
  const M = H.map((v) => v / CELL);
  const rnd = mulberry32(12345);
  const R = 3, brush = [];
  let wsum = 0;
  for (let dz = -R; dz <= R; dz++) for (let dx = -R; dx <= R; dx++) {
    const d = Math.hypot(dx, dz);
    if (d < R) { brush.push([dx, dz, 1 - d / R]); wsum += 1 - d / R; }
  }
  brush.forEach((b) => (b[2] /= wsum));
  const inertia = 0.06, capK = 5, minCap = 0.01, depK = 0.3, eroK = 0.32, evap = 0.012, grav = 4;
  const DROPS = 700000;
  for (let n = 0; n < DROPS; n++) {
    let px = 2 + rnd() * (N - 5), pz = 2 + rnd() * (N - 5), dx = 0, dz = 0, vel = 1, water = 1, sed = 0;
    for (let life = 0; life < 70; life++) {
      const ix = px | 0, iz = pz | 0, fx = px - ix, fz = pz - iz, k = iz * N + ix;
      const h00 = M[k], h10 = M[k + 1], h01 = M[k + N], h11 = M[k + N + 1];
      const gx = (h10 - h00) * (1 - fz) + (h11 - h01) * fz, gz = (h01 - h00) * (1 - fx) + (h11 - h10) * fx;
      const hOld = h00 * (1 - fx) * (1 - fz) + h10 * fx * (1 - fz) + h01 * (1 - fx) * fz + h11 * fx * fz;
      dx = dx * inertia - gx * (1 - inertia); dz = dz * inertia - gz * (1 - inertia);
      const l = Math.hypot(dx, dz);
      if (l < 1e-9) break;
      dx /= l; dz /= l; px += dx; pz += dz;
      if (px < 2 || px > N - 3 || pz < 2 || pz > N - 3) break;
      const jx = px | 0, jz = pz | 0, ux = px - jx, uz = pz - jz, kk = jz * N + jx;
      const hNew = M[kk] * (1 - ux) * (1 - uz) + M[kk + 1] * ux * (1 - uz) + M[kk + N] * (1 - ux) * uz + M[kk + N + 1] * ux * uz;
      const dh = hNew - hOld;
      const cap = Math.max(-dh * vel * water * capK, minCap);
      if (sed > cap || dh > 0) {
        const dep = dh > 0 ? Math.min(dh, sed) : (sed - cap) * depK;
        sed -= dep;
        M[k] += dep * (1 - fx) * (1 - fz); M[k + 1] += dep * fx * (1 - fz); M[k + N] += dep * (1 - fx) * fz; M[k + N + 1] += dep * fx * fz;
      } else {
        const ero = Math.min((cap - sed) * eroK, -dh);
        for (const [bx, bz, w] of brush) {
          const q = (iz + bz) * N + ix + bx;
          if (q < 0 || q >= N * N) continue;
          const de = Math.min(M[q], ero * w);
          M[q] -= de; sed += de;
        }
      }
      vel = Math.sqrt(Math.max(0, vel * vel - dh * grav));
      water *= 1 - evap;
    }
  }
  for (let i = 0; i < N * N; i++) H[i] = M[i] * CELL;
}
console.timeEnd('erosion');

// keep the border identical to the analytic function so the backdrop continues it seamlessly
for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) {
  const e = Math.min(i, j, N - 1 - i, N - 1 - j) * CELL;
  const t = sstep(0, 420, e);
  H[j * N + i] = A0[j * N + i] * (1 - t) + H[j * N + i] * t;
}

const idx = (x, z) => [clamp(Math.round((x + HALF) / CELL), 0, N - 1), clamp(Math.round((z + HALF) / CELL), 0, N - 1)];
const hAt = (x, z) => { const [i, j] = idx(x, z); return H[j * N + i]; };
function forDisc(x, z, r, fn) {
  const [ci, cj] = idx(x, z), n = Math.ceil(r / CELL) + 1;
  for (let j = Math.max(0, cj - n); j <= Math.min(N - 1, cj + n); j++) for (let i = Math.max(0, ci - n); i <= Math.min(N - 1, ci + n); i++) {
    const d = Math.hypot(-HALF + i * CELL - x, -HALF + j * CELL - z);
    if (d <= r) fn(j * N + i, d);
  }
}

// ---------- re-cut the slot canyon (erosion silts narrow gorges up) ----------
const GORGE_CUT = VALLEYS.gorge.pts.filter((p) => p[3] < 60);
for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) {
  const x = -HALF + i * CELL, z = -HALF + j * CELL;
  if (x < -1700 || x > 700 || z < -3100 || z > 1500) continue;
  const [d, f, w] = nearest(GORGE_CUT, x, z);
  if (d > 420) continue;
  const e = Math.max(0, d - w * (1 + 0.5 * fbm(x / 70, z / 70, 2, 96)));
  const wall = 2.5 * e * (0.85 + 0.45 * fbm(x / 160, z / 160, 2, 97));
  const prof = f + Math.min(wall, 230 + 0.5 * (e - 92)) + 3 * fbm(x / 40, z / 40, 2, 98);
  const k = j * N + i;
  const cut = Math.min(H[k], prof);
  H[k] = cut + (H[k] - cut) * sstep(120, 300, e);
}

// ---------- lake basin ----------
const mainPts = VALLEYS.main.pts;
for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) {
  const x = -HALF + i * CELL, z = -HALF + j * CELL;
  if (x < LAKE.x0 - 200 || x > LAKE.x1 + 400 || z < -400 || z > 3200) continue;
  const [d, , w] = nearest(mainPts, x, z);
  const taper = sstep(LAKE.x0, LAKE.x0 + 500, x) * (1 - sstep(LAKE.x1 - 420, LAKE.x1 - 40, x));
  const W = (w - 90 + 60 * fbm(x / 500, z / 500, 2, 91)) * (0.25 + 0.75 * taper);
  const k = j * N + i;
  if (d < W) {
    const q = 1 - (d / W) * (d / W);
    H[k] = Math.min(H[k], LAKE.level + 0.8 - 30 * taper * Math.pow(q, 0.8));
  }
  // natural sill at the outflow: nothing east of the lake may be lower than the lake
  if (x > LAKE.x1 - 60 && x < LAKE.x1 + 260) H[k] = Math.max(H[k], LAKE.level + 1.6 - (x - LAKE.x1 - 20) * 0.02);
}
// rock island
forDisc(2750, 1330, 150, (k, d) => { H[k] += 62 * Math.pow(1 - d / 150, 2) * (1 + 0.5 * fbm(k % N / 6, (k / N | 0) / 6, 2, 93)); });
// tarn in the cirque (irregular shore)
forDisc(TARN.c[0], TARN.c[1], TARN.r + 260, (k, d) => {
  const x = -HALF + (k % N) * CELL, z = -HALF + ((k / N) | 0) * CELL;
  const r = TARN.r * (0.78 + 0.34 * fbm(x / 260, z / 260, 3, 95) + 0.25 * Math.cos(Math.atan2(z - TARN.c[1], x - TARN.c[0]) * 2 + 1));
  if (d < r) H[k] = Math.min(H[k], TARN.level + 0.5 - 13 * (1 - (d / r) ** 2));
  else H[k] = Math.max(H[k], TARN.level + 0.5 + Math.min(d - r, 60) * 0.035);
});

// ---------- landing pads / building plots ----------
const PADS = {
  base: [-1800, 2330, 75], village: [520, 2040, 170, 0.55], hutTarn: [-640, -3300, 30], hutAlp: [3000, -2880, 30],
  hutLedge: [2250, 3120, 32], hutCol: [-2200, -3600, 24], pier: [1500, 2260, 26],
};
const pads = {};
for (const [name, [x, z, r, amt = 1]] of Object.entries(PADS)) {
  let s = 0, n = 0;
  forDisc(x, z, r * 0.6, (k) => { s += H[k]; n++; });
  let y = s / n;
  if (name === 'pier') y = LAKE.level + 1.6;
  if (name === 'hutLedge') y += 6;
  forDisc(x, z, r * 2.6, (k, d) => { const t = (1 - sstep(r, r * 2.6, d)) * amt; H[k] = H[k] * (1 - t) + y * t; });
  pads[name] = [x, +y.toFixed(2), z, r];
}

// ---------- the fortress rock: a plateau cut into the ridge end, cliffs below it, a dry moat ----------
{
  const HS = 4000 / 65535, y = Math.round(SITE.y / HS) * HS;
  for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) {
    const lx = -HALF + i * CELL - SITE.x, lz = -HALF + j * CELL - SITE.z;
    if (lx < -420 || lx > 320 || Math.abs(lz) > 280) continue;
    const k = j * N + i;
    H[k] = siteHeight(lx, lz, H[k], 2 * fbm((lx + 900) / 180, (lz + 900) / 180, 2, 131));
  }
  pads.castle = [SITE.x + FORECOURT.x, +y.toFixed(2), SITE.z + FORECOURT.z, FORECOURT.r];
}

// ---------- the second lake, its moraine and its island (below it: the ring, js/sim/tunnel.js) ----------
let hole = null;
{
  const wobAt = (x, z) => 2 * fbm(x / 240, z / 240, 3, 141);
  for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) {
    const x = -HALF + i * CELL, z = -HALF + j * CELL;
    if (Math.abs(x - LAKE2.cx) > LAKE2.a + 520 || Math.abs(z - LAKE2.cz) > LAKE2.b + 520) continue;
    const k = j * N + i;
    H[k] = ringSiteHeight(x, z, H[k], wobAt(x, z), 2 * fbm(x / 30, z / 30, 2, 142));
  }
  const HS = 4000 / 65535;
  hole = { x: HOLE.x, z: HOLE.z, r: HOLE.r, y: HOLE.floor, rim: +(Math.round((LAKE2.level + ISLE.top) / HS) * HS).toFixed(3) };
}

// ---------- rivers ----------
// Every river begins somewhere that makes sense (the edge of the map, a lake that spills over, a
// spring) and ends somewhere that does (a lake, a bigger river, the edge of the map). They are
// cut last, into the finished ground, so nothing built later can bury them.
const RSTEP = 6;                                                // metres between the points of a river
const gAt = (x, z) => {
  const fx = clamp((x + HALF) / CELL, 0, N - 1.001), fz = clamp((z + HALF) / CELL, 0, N - 1.001), i = fx | 0, j = fz | 0, u = fx - i, v = fz - j, k = j * N + i;
  return (H[k] * (1 - u) + H[k + 1] * u) * (1 - v) + (H[k + N] * (1 - u) + H[k + N + 1] * u) * v;
};
function valleyLine(pts, i0, i1, meander, ease = 0) {                // ease: it starts on the middle line, and begins to wander over so many points                     // a line that wanders about the middle of a valley
  const out = [];
  for (let i = i0; i < i1; i++) {
    for (let t = 0; t < 1; t += 0.1) {
      const a = pts[i], b = pts[i + 1], s = i + t;
      const tx = b[0] - a[0], tz = b[1] - a[1], l = Math.hypot(tx, tz);
      const off = meander * (Math.sin(s * 0.9) * 0.6 + gnoise(s * 0.31, 3.3, 5) * 0.8) * (ease ? sstep(i0 + 1, i0 + 1 + ease, s) : 1);
      out.push([a[0] + tx * t - tz / l * off, a[1] + tz * t + tx / l * off]);
    }
  }
  return out;
}
function resample(path, step = RSTEP) {
  const out = [path[0].slice()];
  let need = step;
  for (let i = 1; i < path.length; i++) {
    let a = path[i - 1], b = path[i], l = Math.hypot(b[0] - a[0], b[1] - a[1]), at = 0;
    while (l - at >= need) { at += need; out.push([a[0] + (b[0] - a[0]) * at / l, a[1] + (b[1] - a[1]) * at / l]); need = step; }
    need -= l - at;
  }
  return out;
}
const inMap = (p) => Math.abs(p[0]) <= HALF && Math.abs(p[1]) <= HALF;
const LAKES = { lake: { level: LAKE.level, x0: LAKE.x0 - 150, x1: LAKE.x1, z0: 200, z1: 3000 },
  tarn: { level: TARN.level, x0: TARN.c[0] - TARN.r - 30, x1: TARN.c[0] + TARN.r + 30, z0: TARN.c[1] - TARN.r - 30, z1: TARN.c[1] + TARN.r + 30 },
  lake2: { level: LAKE2.level, x0: LAKE2.x0, x1: LAKE2.x1, z0: LAKE2.z0, z1: LAKE2.z1 } };
const inRect = (L, p, m = 0) => p[0] > L.x0 - m && p[0] < L.x1 + m && p[1] > L.z0 - m && p[1] < L.z1 + m;
const inLake = (L, p) => inRect(L, p) && gAt(p[0], p[1]) < L.level - 0.3;
const rivers = [], springs = [], falls = [];
// path: [x, z] points downstream. o.from / o.to: { lake } | { river } | 'spring' | 'edge'. o.fall: { after, drop, run }
function river(name, width, path, o = {}) {
  let P = resample(path);
  const tangent = (i) => { const a = P[Math.max(0, i - 1)], b = P[Math.min(P.length - 1, i + 1)], l = Math.hypot(b[0] - a[0], b[1] - a[1]) || 1; return [(b[0] - a[0]) / l, (b[1] - a[1]) / l]; };
  let hold = 0, y0, y1;
  if (o.from && o.from.lake) {
    // it leaves a lake: it starts a little way out in the water, and keeps the level of the lake until it is clear of it
    const L = o.from.lake; let last = 0;
    P.forEach((p, i) => { if (inLake(L, p)) last = i; });
    P = P.slice(Math.max(0, last - 5));
    const e = Math.min(P.length - 2, 10), l = Math.hypot(P[e + 1][0] - P[e - 1][0], P[e + 1][1] - P[e - 1][1]);
    L.out = [+P[e][0].toFixed(1), +P[e][1].toFixed(1), +((P[e + 1][0] - P[e - 1][0]) / l).toFixed(3), +((P[e + 1][1] - P[e - 1][1]) / l).toFixed(3)];   // the lake ends here: beyond this line the water is the river's
    hold = 7 + (o.fall ? Math.round(o.fall.after / RSTEP) : 0); y0 = L.level - 0.05;
  }
  if (o.to && o.to.lake) {
    const L = o.to.lake, i = P.findIndex((p) => inLake(L, p));
    if (i < 0) throw new Error(name + ': never reaches its lake');
    P = P.slice(0, Math.min(P.length, i + 6)); y1 = L.level + 0.03;
  }
  if (o.to && o.to.river) {
    // it joins a bigger river: straight to the nearest point of it
    const R = o.to.river, e = P[P.length - 1];
    let m = 0, best = 1e18; R.pts.forEach((q, i) => { const d = Math.hypot(q[0] - e[0], q[2] - e[1]); if (d < best) { best = d; m = i; } });
    const q = R.pts[Math.min(R.pts.length - 1, m + 4)];          // (a little downstream: it runs in at an angle)
    P = resample([...P, [q[0], q[2]]]); y1 = q[1] + 0.02;
  }
  const n = P.length, surf = new Float64Array(n), fl = 0.0004 * RSTEP;
  const low = (i) => { const [tx, tz] = tangent(i), [x, z] = P[i]; let m = hAt(x, z); for (const d of width > 8 ? [width / 2 + 10, width / 2 + 22] : [width / 2 + 9]) m = Math.min(m, hAt(x - tz * d, z + tx * d), hAt(x + tz * d, z - tx * d)); return m; };
  for (let i = 0; i < n; i++) {
    if (i <= hold && y0 !== undefined) surf[i] = y0 - i * 0.0005;
    else if (i === 0) surf[i] = low(0) - 0.9;
    else if (o.fall && i > hold && i <= hold + Math.round(o.fall.run / RSTEP)) surf[i] = surf[i - 1] - o.fall.drop / Math.round(o.fall.run / RSTEP);
    else surf[i] = Math.min(low(i) - 0.9, surf[i - 1] - fl);
  }
  if (y1 !== undefined) for (let i = 0; i < n; i++) surf[i] = Math.max(surf[i], y1 + (n - 1 - i) * 0.0012 * RSTEP);
  // a big river does not drop over a step: it cuts back into it, and runs down as rapids
  if (o.grade) for (let i = n - 2; i > hold + (o.fall ? Math.round(o.fall.run / RSTEP) : 0); i--) surf[i] = Math.min(surf[i], surf[i + 1] + o.grade * RSTEP);
  let deep = 0; P.forEach(([x, z], i) => { deep = Math.max(deep, gAt(x, z) - surf[i]); }); console.log(' ', name, 'cuts at most', deep.toFixed(1), 'm into the ground');
  // the bed, a strip of gravel on both sides that is just under the water, then the banks
  const near = new Map(), d1 = width / 2 + 1 + CELL * 1.45, bank = 30;
  P.forEach(([x, z], i) => forDisc(x, z, d1 + bank, (k, d) => { const c = near.get(k); if (!c || d < c[0]) near.set(k, [d, i]); }));
  for (const [k, [d, i]] of near) {
    const open = y0 === undefined ? 1 : sstep(10, 20, i);                          // (0: still at the lake)
    const s = Math.min(surf[i], surf[Math.min(n - 1, i + 1)], surf[Math.min(n - 1, i + 2)]);
    const mouth = y1 !== undefined && o.to.lake && s < y1 + 0.6 ? 0.5 : 0;            // (where it meets a lake, the lake floods the gravel)
    if (d <= width / 2 + 1) H[k] = Math.min(H[k], s - 1.0 - mouth);
    else if (d <= d1) H[k] = Math.min(H[k], s - 0.2 - mouth) * open + H[k] * (1 - open);
    else { const t = 1 - (1 - sstep(d1, d1 + bank, d)) * open; H[k] = Math.min(H[k], s - 0.2) * (1 - t) + H[k] * t; }
  }
  const pts = P.map(([x, z], i) => [Math.round(x * 10) / 10, +surf[i].toFixed(2), Math.round(z * 10) / 10]);
  const r = { name, width, from: o.from === 'spring' ? 'spring' : o.from === 'edge' ? 'edge' : 'lake', to: o.to === 'edge' ? 'edge' : o.to.lake ? 'lake' : 'river', pts };
  rivers.push(r);
  if (o.from === 'spring') { const [tx, tz] = tangent(2); springs.push({ name, x: pts[0][0], y: pts[0][1], z: pts[0][2], dx: +tx.toFixed(3), dz: +tz.toFixed(3) }); }
  return r;
}
const segOf = (x) => mainPts.findIndex((p) => p[0] >= x);
const upper = river('upper', 17, valleyLine(mainPts, segOf(-HALF - 150) - 2, segOf(LAKE.x0 + 600), 95).filter(inMap), { from: 'edge', to: { lake: LAKES.lake }, grade: 0.1 });
river('outlet', 19, valleyLine(mainPts, segOf(LAKE.x1 - 260), segOf(LAKE2.cx + 100), 50, 4), { from: { lake: LAKES.lake }, to: { lake: LAKES.lake2 }, grade: 0.1 });
river('lower', 19, valleyLine(mainPts, segOf(LAKE2.cx), segOf(HALF + 150) + 2, 50, 5).filter(inMap), { from: { lake: LAKES.lake2 }, to: 'edge', fall: { after: 96, drop: 9, run: 12 }, grade: 0.1 });
{
  // the tarn spills over into the slot canyon, and the canyon brings it down to the river
  const g = VALLEYS.gorge.pts;
  river('gorge', 4.5, valleyLine(g, 0, g.length - 1, 2).reverse(), { from: { lake: LAKES.tarn }, to: { river: upper } });
  // a spring high in the eastern valley, and its stream down to the lake
  const e = VALLEYS.east.pts, ie = e.findIndex((p) => p[2] > 1545);
  river('east', 4.5, valleyLine(e, 0, ie, 12).reverse(), { from: 'spring', to: { lake: LAKES.lake } });
  // meltwater comes out of the scree below the pass, and runs down the western valley to the river
  const w = VALLEYS.pass.pts, col = w.reduce((m, p, i) => (p[2] > w[m][2] ? i : m), 0), iw = w.findIndex((p, i) => i > col && p[2] < 1900);
  river('pass', 4.5, valleyLine(w, iw, w.length - 1, 10), { from: 'spring', to: { river: upper } });
}
// nothing may stand above the water where the water is drawn: cut what does, until nothing is left
for (let pass = 0, bad = 1; pass < 8 && bad; pass++) {
  bad = 0;
  for (const r of rivers) r.pts.forEach((p, i) => {
    const a = r.pts[Math.max(0, i - 1)], b = r.pts[Math.min(r.pts.length - 1, i + 1)], l = Math.hypot(b[0] - a[0], b[2] - a[2]) || 1, tx = (b[0] - a[0]) / l, tz = (b[2] - a[2]) / l;
    for (const d of [0, -(r.width / 2 + 1), r.width / 2 + 1]) {
      const x = p[0] - tz * d, z = p[2] + tx * d;
      if (gAt(x, z) <= p[1] - 0.12) continue;
      bad++;
      const i0 = clamp(Math.floor((x + HALF) / CELL), 0, N - 2), j0 = clamp(Math.floor((z + HALF) / CELL), 0, N - 2);
      for (const k of [j0 * N + i0, j0 * N + i0 + 1, (j0 + 1) * N + i0, (j0 + 1) * N + i0 + 1]) H[k] = Math.min(H[k], p[1] - 0.3);
    }
  });
  console.log('rivers: cut', bad, 'places that stood in the water');
}
// the falls: where the water drops at least 8 m, steeper than 35 degrees (short level steps between two drops count as one fall)
for (const r of rivers) {
  const P = r.pts, steep = (i) => P[i - 1][1] - P[i][1] >= 0.4 * RSTEP;
  for (let i = 1; i < P.length; i++) {
    if (!steep(i)) continue;
    let j = i;
    for (let q = i + 1; q < P.length && q <= j + 4; q++) if (steep(q)) j = q;
    const drop = P[i - 1][1] - P[j][1], l = Math.hypot(P[j][0] - P[i - 1][0], P[j][2] - P[i - 1][2]) || 1;
    if (drop >= 8 && drop / l >= 0.7) falls.push({ river: r.name, width: r.width, x: P[j][0], z: P[j][2], top: P[i - 1][1], foot: P[j][1], run: +l.toFixed(1), dx: +((P[j][0] - P[i - 1][0]) / l).toFixed(3), dz: +((P[j][2] - P[i - 1][2]) / l).toFixed(3) });
    i = j;
  }
}
console.log('rivers', rivers.map((r) => `${r.name} ${r.from}>${r.to} ${(r.pts.length * RSTEP / 1000).toFixed(1)}km ${r.pts[0][1]}>${r.pts.at(-1)[1]}`).join(' | '));
console.log('springs', JSON.stringify(springs)); console.log('falls', falls.length, falls.map((f) => `${f.river}@${Math.round(f.x)},${Math.round(f.z)} ${(f.top - f.foot).toFixed(0)}m/${f.run}m`).join(' '));

// ---------- write ----------
const toU16 = (F) => { const u = new Uint16Array(F.length); for (let i = 0; i < F.length; i++) u[i] = clamp(Math.round(F[i] / HSCALE), 0, 65535); return u; };
fs.mkdirSync(path.join(ROOT, 'assets'), { recursive: true });
fs.writeFileSync(path.join(ROOT, 'assets/terrain.bin'), Buffer.from(toU16(H).buffer));

console.time('backdrop');
const Bk = new Float32Array(BN * BN);
for (let j = 0; j < BN; j++) for (let i = 0; i < BN; i++) Bk[j * BN + i] = analytic(-BHALF + i * BCELL, -BHALF + j * BCELL);
fs.writeFileSync(path.join(ROOT, 'assets/backdrop.bin'), Buffer.from(toU16(Bk).buffer));
console.timeEnd('backdrop');

// beyond the edge of the map the main river goes on up and down the valley, drawn on the backdrop
const bAt = (x, z) => {
  const fx = clamp((x + BHALF) / BCELL, 0, BN - 1.001), fz = clamp((z + BHALF) / BCELL, 0, BN - 1.001), i = fx | 0, j = fz | 0, u = fx - i, v = fz - j, k = j * BN + i;
  return (Bk[k] * (1 - u) + Bk[k + 1] * u) * (1 - v) + (Bk[k + BN] * (1 - u) + Bk[k + BN + 1] * u) * v;
};
const farRiver = (path, width, [jx, joinY, jz], atEnd) => {
  const joinP = [jx, jz];
  const out = path.filter((p) => !inMap(p)), P = resample(atEnd ? [...out, joinP] : [joinP, ...out], 24), n = P.length;
  return { width, pts: P.map(([x, z], i) => {
    const a = P[Math.max(0, i - 1)], b = P[Math.min(n - 1, i + 1)], l = Math.hypot(b[0] - a[0], b[1] - a[1]) || 1, tx = (b[0] - a[0]) / l, tz = (b[1] - a[1]) / l;
    let y = -1e9; for (const d of [-width, 0, width]) for (const e of [-48, 0, 48]) y = Math.max(y, bAt(x - tz * d + tx * e, z + tx * d + tz * e));
    const k = sstep(0, 200, (atEnd ? n - 1 - i : i) * 24);                      // it meets the river of the map at the map's own height
    return [Math.round(x), +Math.max(y - 4.5, (y - 4.5) * k + joinY * (1 - k)).toFixed(1), Math.round(z)];     // (the backdrop is drawn 6 m low)
  }) };
};
const far = [
  farRiver(valleyLine(mainPts, segOf(-13500), segOf(-HALF - 150) + 2, 95), 20, rivers[0].pts[0], true),
  farRiver(valleyLine(mainPts, segOf(HALF + 150) - 2, segOf(13500), 50), 22, rivers[2].pts.at(-1), false),
];

const meta = {
  n: N, cell: CELL, half: HALF, hscale: HSCALE, bn: BN, bcell: BCELL, bhalf: BHALF,
  lakes: ['lake', 'tarn', 'lake2'].map((name) => ({ name, ...LAKES[name] })),
  rivers, far, springs, falls, pads, hole,
  valleys: Object.fromEntries(Object.entries(VALLEYS).map(([k, v]) => [k, v.pts.filter((p) => Math.abs(p[0]) < HALF && Math.abs(p[1]) < HALF).map((p) => p.map((q) => Math.round(q)))])),
};
fs.writeFileSync(path.join(ROOT, 'assets/meta.json'), JSON.stringify(meta));

// ---------- preview (hill-shaded map, PPM) ----------
{
  const out = Buffer.alloc(N * N * 3);
  let hmin = 1e9, hmax = -1e9;
  for (const v of H) { hmin = Math.min(hmin, v); hmax = Math.max(hmax, v); }
  for (let j = 1; j < N - 1; j++) for (let i = 1; i < N - 1; i++) {
    const k = j * N + i, h = H[k];
    const nx = (H[k - 1] - H[k + 1]) / (2 * CELL), nz = (H[k - N] - H[k + N]) / (2 * CELL);
    const l = 1 / Math.hypot(nx, 1, nz);
    const sh = clamp((nx * -0.5 + 0.65 + nz * -0.55) * l, 0, 1);
    const slope = Math.acos(l) * 57.3;
    let c = h < 1500 ? [70, 120, 50] : h < 2100 ? [110, 130, 70] : h < 2400 ? [130, 125, 115] : [235, 240, 248];
    if (slope > 38 && h < 2700) c = [125, 120, 112];
    const x = -HALF + i * CELL, z = -HALF + j * CELL;
    for (const L of meta.lakes) if (h < L.level && x > L.x0 && x < L.x1 && z > L.z0 && z < L.z1) c = [40, 110, 170];
    for (let q = 0; q < 3; q++) out[k * 3 + q] = Math.min(255, c[q] * (0.35 + 0.85 * sh));
  }
  const mark = (x, z, col, s = 3) => { const [i, j] = idx(x, z); for (let b = -s; b <= s; b++) for (let a = -s; a <= s; a++) { const k = ((j + b) * N + i + a) * 3; if (k >= 0 && k < out.length - 3) { out[k] = col[0]; out[k + 1] = col[1]; out[k + 2] = col[2]; } } };
  for (const r of rivers) for (const p of r.pts) mark(p[0], p[2], [60, 150, 230], 0);
  for (const p of Object.values(pads)) mark(p[0], p[2], [255, 60, 60]);
  fs.writeFileSync(process.argv[2] || '/tmp/foehn_preview.ppm', Buffer.concat([Buffer.from(`P6\n${N} ${N}\n255\n`), out]));
  console.log('height range', hmin.toFixed(0), hmax.toFixed(0));
  console.log('pads', JSON.stringify(pads));
}
