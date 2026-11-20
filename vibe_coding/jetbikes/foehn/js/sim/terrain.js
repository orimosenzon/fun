// The baked alpine terrain: one height field shared by physics, scenery and rendering.
// Heights are interpolated on the same two triangles per cell that the renderer draws.
import { clamp } from './vmath.js';

// T.ext: another world may stand in for the valley. When it is set ({ groundAt, normalAt, waterAt }),
// the bike, its flight computer and the effects fly over that world instead (the Mandelbrot
// landscape in ../../mandelscape does this). Nothing in this game sets it.
// T.hole: one place where the ground is open, straight down: { x, z, r, y (the bottom), rim } (the shaft
// on the island of the second lake). T.under: while the bike is below the earth (js/sim/tunnel.js
// decides, from where it is) the ground is the floor of the tunnels, at this height, and there is
// no water; null in the open.
export const T = { n: 0, cell: 12, half: 0, H: null, meta: null, lakes: [], ready: false, ext: null, hole: null, under: null };

export function initTerrain(arrayBuffer, meta) {
  const u = new Uint16Array(arrayBuffer);
  const H = new Float32Array(u.length);
  for (let i = 0; i < u.length; i++) H[i] = u[i] * meta.hscale;
  T.n = meta.n; T.cell = meta.cell; T.half = meta.half; T.H = H; T.meta = meta; T.lakes = meta.lakes; T.hole = meta.hole || null; T.under = null; T.ready = true;
  return T;
}

export function groundAt(x, z) {
  if (T.ext) return T.ext.groundAt(x, z);
  if (T.under !== null) return T.under;
  const o = T.hole;
  if (o && (x - o.x) * (x - o.x) + (z - o.z) * (z - o.z) < o.r * o.r) return o.y;
  return gridAt(x, z);
}
// the height field itself (the shaft is not in it: the island is whole)
function gridAt(x, z) {
  const n = T.n, H = T.H;
  let gx = (x + T.half) / T.cell, gz = (z + T.half) / T.cell;
  gx = clamp(gx, 0, n - 1.0001); gz = clamp(gz, 0, n - 1.0001);
  const i = gx | 0, j = gz | 0, fx = gx - i, fz = gz - j, k = j * n + i;
  const h00 = H[k], h10 = H[k + 1], h01 = H[k + n], h11 = H[k + n + 1];
  if (fx + fz <= 1) return h00 + (h10 - h00) * fx + (h01 - h00) * fz;
  return h11 + (h01 - h11) * (1 - fx) + (h10 - h11) * (1 - fz);
}

// smooth normal (central differences over one cell)
export function normalAt(x, z) {
  if (T.ext) return T.ext.normalAt(x, z);
  if (T.under !== null) return [0, 1, 0];
  const e = T.cell * 0.5;
  const nx = gridAt(x - e, z) - gridAt(x + e, z), nz = gridAt(x, z - e) - gridAt(x, z + e);
  const l = Math.hypot(nx, 2 * e, nz);
  return [nx / l, 2 * e / l, nz / l];
}

// water level at (x,z), or -Infinity when there is no lake there
export function waterAt(x, z) {
  if (T.ext) return T.ext.waterAt(x, z);
  if (T.under !== null) return -Infinity;
  const o = T.hole;
  if (o && (x - o.x) * (x - o.x) + (z - o.z) * (z - o.z) < o.r * o.r) return -Infinity;
  for (const L of T.lakes) if (x > L.x0 && x < L.x1 && z > L.z0 && z < L.z1) return L.level;
  return -Infinity;
}
export function surfaceAt(x, z) { return Math.max(groundAt(x, z), waterAt(x, z)); }
export function isWater(x, z) { return waterAt(x, z) > groundAt(x, z); }

// ISA troposphere
export function airDensity(h) { return 1.225 * Math.pow(Math.max(0.1, 1 - 2.25577e-5 * h), 4.2559); }
export function airTemp(h) { return 288.15 - 0.0065 * h; }
