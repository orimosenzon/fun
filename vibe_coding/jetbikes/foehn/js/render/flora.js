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
const ROCK = [3.3, 3.0, 3.1, 0.75];
const KINDS = {
  rockA: { size: [0.5, 1.5], far: 420, sink: 0.12, tint: ROCK }, rockB: { size: [0.5, 1.6], far: 420, sink: 0.14, tint: ROCK }, rockC: { size: [0.5, 1.5], far: 420, sink: 0.12, tint: ROCK },
  shards: { size: [0.4, 1.1], far: 380, sink: 0.05, tint: [1.5, 1.45, 1.4, 0.3] },
  bush: { size: [0.8, 1.8], far: 360, sink: 0.04, tint: [1.5, 1.7, 1.2, 0] }, berries: { size: [0.8, 1.4], far: 320, sink: 0.04, tint: [1.6, 1.6, 1.3, 0] },
  flowers: { size: [0.55, 0.95], far: 170, sink: 0, tint: [2.6, 2.2, 2.6, 0] },
  daisies: { model: 'flowers', size: [0.5, 0.85], far: 170, sink: 0, tint: [3.6, 3.0, 1.3, 0.85] },      // the same flowers, washed pale yellow
  stump: { size: [0.4, 0.8], far: 260, sink: 0.02, tint: [1.5, 1.5, 1.4, 0] }, log: { size: [0.3, 0.5], far: 280, sink: 0.08, tint: [1.5, 1.5, 1.4, 0] },
  hay: { size: [1.3, 1.7], far: 420, sink: 0, tint: [1, 1, 1, 0] },
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
