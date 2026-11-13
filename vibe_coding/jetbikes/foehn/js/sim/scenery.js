// Deterministic scenery shared by the renderer and the collision code: forest, buildings.
import { T, groundAt, normalAt, waterAt } from './terrain.js';
import { fbm, hash2, mulberry32 } from './noise.js';
import { clamp, sstep } from './vmath.js';
import { EVENTS, buildEvent } from './courses.js';
import { buildCastle, hitCastle } from './castle.js';

export const S = { trees: null, count: 0, cellOf: null, cellStart: null, grid: 48, gn: 0, props: [], mask: null, ready: false };
const TREELINE = 1980;

function makeProps() {
  const P = T.meta.pads, out = [];
  const add = (type, x, z, yaw, w, d, h, extra = {}) => out.push({ type, x, z, y: groundAt(x, z), yaw, w, d, h, ...extra });
  // home base: hangar and tower beside the pad
  add('hangar', P.base[0] - 12, P.base[2] + 52, 0, 30, 18, 8);
  add('tower', P.base[0] + 34, P.base[2] + 50, 0, 5, 5, 15);
  // village: chalets on rings around the square, a church
  const r = mulberry32(4711), vx = P.village[0], vz = P.village[2];
  add('church', vx + 44, vz - 30, 0.3, 10, 20, 11, { steeple: 27 });
  let n = 0;
  for (let ring = 0; ring < 3; ring++) {
    const R = 46 + ring * 34, cnt = 8 + ring * 5;
    for (let k = 0; k < cnt; k++) {
      const a = (k + 0.5 * (ring % 2) + 0.25 * (r() - 0.5)) / cnt * 6.283;
      const x = vx + Math.cos(a) * R * (0.92 + 0.16 * r()), z = vz + Math.sin(a) * R * (0.92 + 0.16 * r());
      if (Math.hypot(x - (vx + 44), z - (vz - 30)) < 24) continue;
      add('chalet', x, z, a + 1.57 + (r() - 0.5) * 0.5, 9 + 3 * r(), 7 + 2 * r(), 5.2 + 1.5 * r(), { seed: n++ });
    }
  }
  // mountain huts beside their pads
  for (const name of ['hutTarn', 'hutAlp', 'hutLedge', 'hutCol']) {
    const p = P[name], a = hash2(p[0], p[2], 3) * 6.283;
    add('hut', p[0] + Math.cos(a) * (p[3] * 0.5 + 9), p[2] + Math.sin(a) * (p[3] * 0.5 + 9), a + 1.57, 9, 7, 4.6, { seed: 100 + out.length });
  }
  add('boathouse', P.pier[0] - 26, P.pier[2] + 20, -0.6, 9, 6, 4.2, { seed: 200 });
  // a few lonely hay barns on the meadows
  const barns = [[-3200, 2050], [-2900, 2620], [-1000, 2450], [-150, 2250], [900, 1500], [4600, 1150], [3350, -2650], [-3900, -900]];
  barns.forEach(([x, z], i) => { if (normalAt(x, z)[1] > 0.97 && waterAt(x, z) < groundAt(x, z)) add('barn', x, z, hash2(x, z, 9) * 3.14, 7, 5, 3.6, { seed: 300 + i }); });
  return out;
}

// cells of the terrain grid where nothing may grow
function makeMask() {
  const n = T.n, m = new Uint8Array(n * n), c = T.cell, half = T.half;
  const disc = (x, z, r) => {
    const i0 = Math.max(0, Math.floor((x - r + half) / c)), i1 = Math.min(n - 1, Math.ceil((x + r + half) / c));
    const j0 = Math.max(0, Math.floor((z - r + half) / c)), j1 = Math.min(n - 1, Math.ceil((z + r + half) / c));
    for (let j = j0; j <= j1; j++) for (let i = i0; i <= i1; i++) if (Math.hypot(-half + i * c - x, -half + j * c - z) <= r + c * 0.5) m[j * n + i] = 1;
  };
  for (const rv of T.meta.rivers) for (const p of rv.pts) disc(p[0], p[2], rv.width / 2 + 10);
  for (const [name, p] of Object.entries(T.meta.pads)) disc(p[0], p[2], name === 'village' ? 150 : p[3] * 1.5 + 14);
  for (const p of S.props) disc(p.x, p.z, Math.max(p.w, p.d) * 0.9 + 6);
  // the racing lines of all events (only where they run low)
  for (const ev of EVENTS) {
    const b = buildEvent(ev);
    for (const q of b.path) if (q[1] - groundAt(q[0], q[2]) < 34) disc(q[0], q[2], 34);
  }
  return m;
}

export function buildScenery() {
  buildCastle();
  S.props = makeProps();
  S.mask = makeMask();
  const n = T.n, c = T.cell, half = T.half;
  const STEP = 9.5, cnt = Math.floor(2 * (half - 30) / STEP);
  const buf = [];
  for (let j = 0; j < cnt; j++) {
    for (let i = 0; i < cnt; i++) {
      const h1 = hash2(i, j, 11), h2 = hash2(i, j, 12);
      const x = -half + 30 + (i + h1) * STEP, z = -half + 30 + (j + h2) * STEP;
      const gi = Math.round((x + half) / c), gj = Math.round((z + half) / c);
      if (S.mask[gj * n + gi]) continue;
      const y = groundAt(x, z);
      const line = TREELINE + 170 * fbm(x / 700, z / 700, 2, 61);
      if (y > line) continue;
      if (waterAt(x, z) > y - 1.2) continue;
      const nrm = normalAt(x, z);
      if (nrm[1] < 0.76) continue;                           // too steep
      // forest cover: dense on the lower slopes, groves on the valley floor, thinning to the tree line
      const slope = 1 - nrm[1];
      const cover = fbm(x / 420, z / 420, 3, 62) + 0.9 * sstep(0.012, 0.07, slope) - 0.35 - 1.3 * sstep(line - 260, line, y) + 0.5 * fbm(x / 90, z / 90, 2, 63);
      const h3 = hash2(i, j, 13);
      if (cover < h3 * 0.55 - 0.05) continue;
      const hgt = (9 + 15 * hash2(i, j, 14) * hash2(i, j, 15) + 7 * clamp(cover, 0, 1)) * (1 - 0.45 * sstep(line - 320, line, y));
      buf.push(x, y - 0.25, z, hgt, hash2(i, j, 16));
    }
  }
  S.trees = new Float32Array(buf); S.count = buf.length / 5;
  // spatial hash for collisions
  const gs = S.grid, gn = S.gn = Math.ceil(2 * half / gs);
  const cnts = new Int32Array(gn * gn + 1);
  const cellOf = new Int32Array(S.count);
  for (let k = 0; k < S.count; k++) { const ci = Math.floor((S.trees[k * 5] + half) / gs), cj = Math.floor((S.trees[k * 5 + 2] + half) / gs); cellOf[k] = cj * gn + ci; cnts[cellOf[k] + 1]++; }
  for (let k = 0; k < gn * gn; k++) cnts[k + 1] += cnts[k];
  const order = new Int32Array(S.count), fill = cnts.slice();
  for (let k = 0; k < S.count; k++) order[fill[cellOf[k]]++] = k;
  S.cellStart = cnts; S.order = order; S.ready = true;
  // canopy map (24 m cells): height of the highest tree top, what a terrain radar would see
  const cg = S.cg = 24, cn = S.cn = Math.ceil(2 * half / cg), can = S.canopy = new Float32Array(cn * cn).fill(-1e9);
  for (let k = 0; k < S.count; k++) {
    const x = S.trees[k * 5], z = S.trees[k * 5 + 2], top = S.trees[k * 5 + 1] + S.trees[k * 5 + 3], r = 0.2 * S.trees[k * 5 + 3] + 2;
    const i0 = Math.max(0, Math.floor((x - r + half) / cg)), i1 = Math.min(cn - 1, Math.floor((x + r + half) / cg));
    const j0 = Math.max(0, Math.floor((z - r + half) / cg)), j1 = Math.min(cn - 1, Math.floor((z + r + half) / cg));
    for (let j = j0; j <= j1; j++) for (let i = i0; i <= i1; i++) if (top > can[j * cn + i]) can[j * cn + i] = top;
  }
  for (const b of S.props) {
    const r = Math.max(b.w, b.d) / 2 + 3, top = b.y + (b.steeple || b.h * 1.5);
    for (let j = Math.floor((b.z - r + half) / cg); j <= Math.floor((b.z + r + half) / cg); j++) for (let i = Math.floor((b.x - r + half) / cg); i <= Math.floor((b.x + r + half) / cg); i++) { const q = j * cn + i; if (q >= 0 && q < cn * cn) can[q] = Math.max(can[q], top); }
  }
}

// Does the point (world) touch a tree (trunk or the cone of its crown) or a building?
export function hitScenery(p, pad = 0.35) {
  if (hitCastle(p, pad)) return 'wall';
  const half = T.half, gs = S.grid, gn = S.gn;
  const ci = Math.floor((p[0] + half) / gs), cj = Math.floor((p[2] + half) / gs);
  for (let dj = -1; dj <= 1; dj++) for (let di = -1; di <= 1; di++) {
    const a = ci + di, b = cj + dj;
    if (a < 0 || b < 0 || a >= gn || b >= gn) continue;
    const cell = b * gn + a;
    for (let q = S.cellStart[cell]; q < S.cellStart[cell + 1]; q++) {
      const k = S.order[q] * 5, tx = S.trees[k], ty = S.trees[k + 1], tz = S.trees[k + 2], h = S.trees[k + 3];
      const y = p[1] - ty;
      if (y < 0 || y > h + pad) continue;
      const d = Math.hypot(p[0] - tx, p[2] - tz);
      const crown = y > 0.16 * h ? 0.19 * h * (1 - (y - 0.16 * h) / (0.84 * h)) : 0;
      if (d < Math.max(0.3, crown) + pad) return 'tree';
    }
  }
  for (const b of S.props) {
    const dx = p[0] - b.x, dz = p[2] - b.z;
    if (Math.abs(dx) > 40 || Math.abs(dz) > 40) continue;
    const c = Math.cos(b.yaw), s = Math.sin(b.yaw);
    const lx = dx * c - dz * s, lz = dx * s + dz * c;      // into the building frame (yaw about +y)
    const top = b.y + (b.type === 'church' && Math.abs(lx) < 3 && lz < -b.d / 2 + 6 ? b.steeple : b.h * 1.45);
    if (Math.abs(lx) < b.w / 2 + pad && Math.abs(lz) < b.d / 2 + pad && p[1] < top + pad && p[1] > b.y - 2) return 'building';
  }
  return null;
}

// top of whatever stands at (x,z): tree canopy or roof, or -inf on open ground
export function canopyAt(x, z) {
  if (!S.canopy) return -1e9;
  const i = Math.floor((x + T.half) / S.cg), j = Math.floor((z + T.half) / S.cg);
  return i < 0 || j < 0 || i >= S.cn || j >= S.cn ? -1e9 : S.canopy[j * S.cn + i];
}
