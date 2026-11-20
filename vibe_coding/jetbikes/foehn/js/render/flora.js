// The small things on the ground: boulders on the slopes, bushes and berries at the edge of the
// forest, flowers in patches in the meadows, stumps and fallen logs among the trees, hay beside
// the cattle. Too many for a list (the valley is 144 square kilometres), so they are worked out
// a 64 m square at a time, from a hash of the place, when the camera first comes near; the same
// place always grows the same things. Nothing here can be hit: it is all below the radar's floor.
import { T, groundAt, normalAt, isWater } from '../sim/terrain.js';
import { canopyAt } from '../sim/scenery.js';
import { hitCastle } from '../sim/castle.js';
import { loadModels, poseGeometry, instanced, instMaterial } from './fauna.js';

const CELL = 64, SUB = 10;            // 10 x 10 places to try in every square
// size: height in metres. far: how far it is drawn. sink: how deep it sits in the ground (of its height)
// tint: the models are painted darker than this valley (its limestone is pale, its meadows sunlit):
// a gain on red, green, blue, and how far the colour is washed to grey
const ROCK = [3.3, 3.0, 3.1, 0.75], BOULDER = [2.5, 2.5, 2.55, 0.97];
const KINDS = {
  rockA: { size: [0.5, 1.5], far: 420, sink: 0.12, tint: ROCK }, rockB: { size: [0.5, 1.6], far: 420, sink: 0.14, tint: ROCK }, rockC: { size: [0.5, 1.5], far: 420, sink: 0.12, tint: ROCK },
  shards: { size: [0.4, 1.1], far: 380, sink: 0.05, tint: [1.5, 1.45, 1.4, 0.3] },
  bush: { size: [0.8, 1.8], far: 360, sink: 0.04, tint: [1.5, 1.7, 1.2, 0] }, berries: { size: [0.8, 1.4], far: 320, sink: 0.04, tint: [1.6, 1.6, 1.3, 0] },
  flowers: { size: [0.55, 0.95], far: 170, sink: 0, tint: [2.6, 2.2, 2.6, 0] },
  daisies: { model: 'flowers', size: [0.5, 0.85], far: 170, sink: 0, tint: [3.6, 3.0, 1.3, 0.85] },      // the same flowers, washed pale yellow
  stump: { size: [0.4, 0.8], far: 260, sink: 0.02, tint: [1.5, 1.5, 1.4, 0] }, log: { size: [0.3, 0.5], far: 280, sink: 0.08, tint: [1.5, 1.5, 1.4, 0] },
  hay: { size: [1.3, 1.7], far: 420, sink: 0, tint: [1, 1, 1, 0] },
  // beside the water (js/render/flora.js places them along the rivers of the map, at the springs and at the falls)
  boulderA: { size: [1.4, 3.4], far: 1100, sink: 0.22, tint: BOULDER }, boulderB: { size: [1.4, 3.4], far: 1100, sink: 0.22, tint: BOULDER }, boulderC: { size: [1.4, 3.4], far: 1100, sink: 0.22, tint: BOULDER },
  mossRock: { size: [0.6, 1.8], far: 520, sink: 0.15, tint: [2.4, 2.5, 2.1, 0.35] },
  fern: { size: [0.5, 1.0], far: 190, sink: 0.02, tint: [1.5, 1.7, 1.2, 0] }, reed: { size: [1.5, 2.3], far: 260, sink: 0.03, tint: [1.5, 1.6, 1.2, 0] },
};
const hash = (a, b, c = 0) => { let h = Math.imul(a | 0, 374761393) ^ Math.imul(b | 0, 668265263) ^ Math.imul(c | 0, 2147483647); h = Math.imul(h ^ (h >>> 13), 1274126177); return ((h ^ (h >>> 16)) >>> 0) / 4294967296; };
// smooth noise 0..1 with features of about s metres: where the flowers and the bushes gather
function patch(x, z, s, seed) {
  const fx = x / s, fz = z / s, i = Math.floor(fx), j = Math.floor(fz), u = fx - i, v = fz - j, a = u * u * (3 - 2 * u), b = v * v * (3 - 2 * v);
  return (hash(i, j, seed) * (1 - a) + hash(i + 1, j, seed) * a) * (1 - b) + (hash(i, j + 1, seed) * (1 - a) + hash(i + 1, j + 1, seed) * a) * b;
}
const wooded = (x, z) => canopyAt(x, z, true) > -1e8;

// what grows at one place: a kind, or null
function grows(x, z, r) {
  if (Math.abs(x) > T.half - 30 || Math.abs(z) > T.half - 30 || isWater(x, z)) return null;
  const y = groundAt(x, z), steep = 1 - normalAt(x, z)[1];
  if (steep > 0.16) return null;                                         // a wall: the rock is painted on it already
  for (const p of Object.values(T.meta.pads)) if (Math.hypot(x - p[0], z - p[2]) < Math.min(p[3], 40) + 8) return null;
  if (hitCastle([x, y + 1, z], 8)) return null;
  const high = y > 1750, snow = y > 2250;
  if (wooded(x, z)) return r < 0.035 ? 'stump' : r < 0.05 ? 'log' : r < 0.062 ? 'rockC' : null;
  if (steep > 0.045) return r < (high ? 0.16 : 0.07) ? ['rockA', 'rockB', 'rockC'][Math.floor(r * 1e4) % 3] : high && r < 0.24 ? 'shards' : null;
  if (snow) return r < 0.02 ? 'shards' : null;
  const edge = wooded(x + 30, z) || wooded(x - 30, z) || wooded(x, z + 30) || wooded(x, z - 30);
  if (r < 0.012) return ['rockA', 'rockB', 'rockC'][Math.floor(r * 1e5) % 3];
  if (r < 0.012 + (edge ? 0.11 : 0.012) * (high ? 0.4 : 1) * (0.4 + 1.2 * patch(x, z, 140, 3))) return Math.floor(r * 1e5) % 3 ? 'bush' : 'berries';
  if (!high && patch(x, z, 55, 7) > 0.66 && r > 0.55) return patch(x, z, 300, 11) > 0.5 ? 'flowers' : 'daisies';
  return null;
}

export async function buildFlora(scene, fauna) {
  const out = { ready: false, shown: 0, update() {} };
  const file = await loadModels(), names = Object.keys(KINDS).filter((k) => file.head.models[KINDS[k].model || k]);
  const sets = {};
  for (const k of names) sets[k] = instanced(scene, poseGeometry(file, file.head.models[KINDS[k].model || k], 'still'), instMaterial(KINDS[k].far * 0.8, KINDS[k].far, KINDS[k].tint), KINDS[k].far < 200 ? 4000 : 2500);
  // hay stands where the cattle and the horses are
  const fixed = new Map();
  if (fauna) for (const a of fauna.start) if (a.lead && /cow|holstein|horse|bull/.test(a.kind)) for (let k = 0; k < 3; k++) {
    const an = hash(a.x, a.z, k) * 6.283, d = 14 + 16 * hash(a.z, a.x, k), x = a.x + Math.cos(an) * d, z = a.z + Math.sin(an) * d;
    if (isWater(x, z) || wooded(x, z)) continue;
    const key = Math.floor((x + T.half) / CELL) * 4096 + Math.floor((z + T.half) / CELL);
    if (!fixed.has(key)) fixed.set(key, []);
    fixed.get(key).push(['hay', x, groundAt(x, z), z, 1.3 + 0.4 * hash(x, z, 9), an]);
  }
  // ---- beside the water ----
  // stones in the beds of the streams and on the gravel of the rivers, cattails where the water is slow,
  // big boulders and ferns round every spring and at the top and the foot of every fall
  const put = (kind, x, z, sz, yaw, y) => {
    if (!sets[kind] || Math.abs(x) > T.half - 20 || Math.abs(z) > T.half - 20) return;
    const key = Math.floor((x + T.half) / CELL) * 4096 + Math.floor((z + T.half) / CELL);
    if (!fixed.has(key)) fixed.set(key, []);
    fixed.get(key).push([kind, x, (y ?? groundAt(x, z)) - KINDS[kind].sink * sz, z, sz, yaw]);
  };
  const BOULDERS = ['boulderA', 'boulderB', 'boulderC'], STONES = ['rockA', 'rockB', 'rockC', 'mossRock'];
  for (const [ri, r] of (T.meta.rivers || []).entries()) {
    const P = r.pts, big = r.width > 8, step = big ? 2 : 1;
    for (let i = 2; i < P.length - 2; i += step) {
      const p = P[i], a = P[i - 1], b = P[i + 1], l = Math.hypot(b[0] - a[0], b[2] - a[2]) || 1, tx = (b[0] - a[0]) / l, tz = (b[2] - a[2]) / l, slope = (a[1] - b[1]) / l;
      const h = (k) => hash(ri * 7919 + i, k, 77), side = h(1) < 0.5 ? -1 : 1, an = h(4) * 6.283;
      const at = (d) => [p[0] - tz * d * side, p[2] + tx * d * side];
      if (big) {
        if (h(2) < 0.22) { const [x, z] = at(r.width / 2 + 1 + 13 * h(3)); if (!isWater(x, z)) put(STONES[Math.floor(h(5) * 4)], x, z, 0.5 + 1.1 * h(6) ** 2, an); }
        if (slope < 0.012 && patch(p[0], p[2], 90, 21) > 0.55 && h(7) < 0.5) for (let k = 0; k < 5; k++) { const [x, z] = at(r.width / 2 + 0.5 + 3.5 * h(10 + k)); put('reed', x + (h(20 + k) - 0.5) * 5, z + (h(30 + k) - 0.5) * 5, 1.5 + 0.8 * h(40 + k), h(50 + k) * 6.283); }
      } else {
        if (h(2) < 0.5) { const [x, z] = at(h(3) < 0.3 ? r.width * 0.3 * h(8) : r.width / 2 + 0.3 + 5 * h(3)); put(STONES[Math.floor(h(5) * 4)], x, z, 0.45 + 1.0 * h(6) ** 2, an); }
        if (h(7) < 0.05 && slope > 0.12) { const [x, z] = at(r.width / 2 + 2 + 4 * h(9)); put(BOULDERS[Math.floor(h(11) * 3)], x, z, 1.3 + 1.2 * h(12), an); }
        if (p[1] < 1750 && h(13) < 0.22) { const [x, z] = at(r.width / 2 + 1 + 3 * h(14)); put('fern', x, z, 0.5 + 0.5 * h(15), an); }
      }
    }
  }
  // a spring: the water comes out from under a heap of boulders
  for (const [si, sp] of (T.meta.springs || []).entries()) {
    const h = (k) => hash(si + 31, k, 91);
    for (let k = 0; k < 9; k++) {
      const an = Math.atan2(-sp.dz, -sp.dx) + (k / 8 - 0.5) * 3.6, d = 3.5 + 3.5 * h(k) + (k % 2) * 2.5;      // an arc on the uphill side
      put(BOULDERS[k % 3], sp.x + Math.cos(an) * d, sp.z + Math.sin(an) * d, 1.8 + 1.8 * h(20 + k), h(40 + k) * 6.283);
    }
    for (let k = 0; k < 8; k++) { const an = h(60 + k) * 6.283, d = 2 + 8 * h(70 + k); put(k < 4 ? 'mossRock' : 'fern', sp.x + Math.cos(an) * d, sp.z + Math.sin(an) * d, k < 4 ? 0.8 + 0.9 * h(80 + k) : 0.6 + 0.4 * h(80 + k), an); }
  }
  // a fall: boulders on both sides of the lip, and round the pool at its foot
  for (const [fi, f] of (T.meta.falls || []).entries()) {
    const h = (k) => hash(fi + 57, k, 93), w = f.width / 2 + 1, tall = Math.min(1.6, 1 + (f.top - f.foot) / 80);
    for (const side of [-1, 1]) {
      for (let k = 0; k < 3; k++) {
        const d = w + 1.5 + 3.2 * k, back = f.run + 1 + 3 * h(k + side), x = f.x - f.dx * back - f.dz * d * side, z = f.z - f.dz * back + f.dx * d * side;
        put(BOULDERS[(k + (side > 0 ? 1 : 0)) % 3], x, z, (2.2 + 1.6 * h(10 + k + side)) * tall, h(20 + k + side) * 6.283);
      }
      for (let k = 0; k < 4; k++) {
        const d = w + 0.5 + 2.6 * k * h(30 + k), fwd = -2 + 5 * k * h(40 + k + side), x = f.x + f.dx * fwd - f.dz * d * side, z = f.z + f.dz * fwd + f.dx * d * side;
        put(k % 2 ? 'mossRock' : BOULDERS[k % 3], x, z, k % 2 ? 1.0 + 0.8 * h(50 + k) : (1.6 + 1.4 * h(50 + k + side)) * tall, h(60 + k) * 6.283);
        put('fern', x - f.dz * 2.5 * side, z + f.dx * 2.5 * side, 0.6 + 0.4 * h(70 + k), h(80 + k) * 6.283);
      }
    }
  }
  const cells = new Map();
  const cell = (ci, cj) => {
    const key = ci * 4096 + cj; let c = cells.get(key); if (c) return c;
    c = (fixed.get(key) || []).slice();
    for (let j = 0; j < SUB; j++) for (let i = 0; i < SUB; i++) {
      const gi = ci * SUB + i, gj = cj * SUB + j, x = -T.half + (gi + hash(gi, gj, 1)) * CELL / SUB, z = -T.half + (gj + hash(gi, gj, 2)) * CELL / SUB;
      const kind = grows(x, z, hash(gi, gj, 5)); if (!kind || !sets[kind]) continue;
      const K = KINDS[kind], sz = K.size[0] + (K.size[1] - K.size[0]) * hash(gi, gj, 6) ** 1.6;
      if (K.far < 200) for (let f = 0; f < 5; f++) { const fx = x + (hash(gi, gj, 20 + f) - 0.5) * 5, fz = z + (hash(gi, gj, 30 + f) - 0.5) * 5; c.push([kind, fx, groundAt(fx, fz), fz, sz * (0.8 + 0.4 * hash(gi, gj, 40 + f)), hash(gi, gj, 50 + f) * 6.283]); }
      else c.push([kind, x, groundAt(x, z) - K.sink * sz, z, sz, hash(gi, gj, 7) * 6.283]);
    }
    if (cells.size > 6000) cells.clear();
    cells.set(key, c); return c;
  };
  let lx = 1e9, lz = 1e9, lq = 0;
  // q: the quality setting's reach (0.6 .. 1)
  out.update = (cam, q = 1) => {
    const p = cam.position;
    if (Math.hypot(p.x - lx, p.z - lz) < 20 && q === lq) return;                  // the list is good for another 20 m
    lx = p.x; lz = p.z; lq = q;
    for (const k of names) { sets[k].n = 0; sets[k].m.material.uniforms.uFade.value.set(KINDS[k].far * 0.8 * q, KINDS[k].far * q); }
    const R = 440 * q + 24, c0 = Math.floor((p.x - R + T.half) / CELL), c1 = Math.floor((p.x + R + T.half) / CELL), d0 = Math.floor((p.z - R + T.half) / CELL), d1 = Math.floor((p.z + R + T.half) / CELL);
    let shown = 0;
    for (let cj = d0; cj <= d1; cj++) for (let ci = c0; ci <= c1; ci++) for (const it of cell(ci, cj)) {
      const s = sets[it[0]], far = KINDS[it[0]].far * q + 24, dx = it[1] - p.x, dz = it[3] - p.z;
      if (dx * dx + dz * dz > far * far || s.n >= s.cap) continue;
      const i = s.n++; s.ip[i * 4] = it[1]; s.ip[i * 4 + 1] = it[2]; s.ip[i * 4 + 2] = it[3]; s.ip[i * 4 + 3] = it[4]; s.iy[i] = it[5]; shown++;
    }
    for (const k of names) sets[k].show(sets[k].n);
    out.shown = shown; out.counts = Object.fromEntries(names.map((k) => [k, sets[k].n]));
  };
  out.ready = true;
  return out;
}
