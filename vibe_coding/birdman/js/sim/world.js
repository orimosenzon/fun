// The island: an analytic height function sampled once to a grid, plus the air above it
// (sea breeze, ridge lift where the wind meets a slope, thermals over sun-warmed ground).
// Physics and rendering read the same grid, so what you see is what you hit.
import { fbm, ridged, gnoise } from './noise.js';
import { clamp, lerp, sstep } from './vmath.js';

export const N = 449, CELL = 18, HALF = ((N - 1) * CELL) / 2; // 8064 m square
export const SEABED = -45;
export const WIND_DIR = [1, 0, 0.12]; // sea breeze from the west
export const WIND_U = 6.5; // m/s at height

const gauss = (x, z, cx, cz, r) => Math.exp(-((x - cx) * (x - cx) + (z - cz) * (z - cz)) / (2 * r * r));

export const STACKS = [[-2760, -620, 95, 75], [-2900, -260, 130, 85], [-2700, 150, 70, 60], [-3000, 380, 55, 55]];

export function heightRaw(x, z) {
  const n1 = fbm(x / 1700, z / 1700, 4, 5), n2 = fbm(x / 1700 + 31, z / 1700 - 17, 4, 9);
  const wx = x + 380 * n1, wz = z + 380 * n2;
  const r = Math.hypot(wx / 3050, wz / 2750);
  const land = sstep(1.02, 0.84, r);
  const inl = sstep(0.98, 0.5, r);
  let h = lerp(SEABED, 5, sstep(1.4, 0.99, r));
  const hills = 0.5 + 0.5 * fbm(x / 640, z / 640, 5, 3);
  h += land * 5 + inl * (10 + 110 * hills * hills);
  // the mountain
  const g = gauss(x, z, 1450, -950, 950);
  const rid = ridged(x / 1300 + 3.1, z / 1300 + 8.2, 6, 41);
  h += land * g * (330 + 620 * rid);
  // farmland plain
  const flat = gauss(x, z, 250, 950, 620);
  h = lerp(h, 26 + 5 * fbm(x / 500, z / 500, 2, 61), Math.min(1, flat * 1.25) * land);
  // south bay and the village shelf above it
  const bay = gauss(x, z, -620, 2950, 430);
  h -= 75 * bay;
  const vil = gauss(x, z, -560, 2080, 300);
  h = lerp(h, 9 + 2 * fbm(x / 200, z / 200, 2, 71), Math.min(1, vil * 1.3) * sstep(-2, 3, h));
  // the west cliff: a plateau that ends in a wall over the sea
  const dzc = Math.abs(z + 250);
  const cz = Math.exp(-Math.pow(dzc / 1150, 4));
  const edgeX = x + 80 * fbm(z / 330, x / 900, 3, 77) + 900 * sstep(800, 1900, dzc);
  const wall = sstep(-2390, -2300, edgeX);
  const plateau = 185 * (0.4 + 0.6 * sstep(-500, -1900, x)) * (0.9 + 0.1 * fbm(x / 260, z / 260, 3, 88));
  h = lerp(h, Math.max(h, plateau), cz * sstep(-300, -900, x));
  h = lerp(h, SEABED * 0.7, sstep(1900, 1500, dzc) * (1 - wall));
  // a sand cove south of the wall: the first landing
  const cove = gauss(x, z, -2640, 930, 200);
  h = lerp(h, 2.4 + 1.2 * sstep(-2760, -2500, x), Math.min(1, cove * 1.7));
  // talus and surf rocks at the foot
  h += cz * 14 * sstep(-2480, -2390, edgeX) * (1 - wall) * (0.5 + 0.5 * gnoise(x / 60, z / 60, 5));
  // sea stacks
  for (const [sx, sz, sh, sr] of STACKS) {
    const d = Math.hypot(x - sx, z - sz) * (1 + 0.25 * gnoise(x / 70, z / 70, 9));
    h = Math.max(h, lerp(SEABED, sh * (0.85 + 0.15 * gnoise(x / 40, z / 40, 3)), sstep(sr, sr * 0.45, d)));
  }
  // lighthouse head
  const lh = gauss(x + 60 * gnoise(x / 150, z / 150, 15), z + 60 * gnoise(x / 150 + 9, z / 150, 16), -1900, 2250, 170);
  h = Math.max(h, lerp(SEABED, 44, Math.min(1, lh * 1.7)));
  return h;
}

export let H = null; // Float32Array N*N
export function initWorld() {
  if (H) return;
  H = new Float32Array(N * N);
  for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) H[j * N + i] = heightRaw(i * CELL - HALF, j * CELL - HALF);
  // the summit is wherever the noise put it; the beach is where the sand is
  let best = 0;
  for (let k = 0; k < N * N; k++) if (H[k] > H[best]) best = k;
  POI.peak[0] = (best % N) * CELL - HALF; POI.peak[2] = ((best / N) | 0) * CELL - HALF;
  for (const k in POI) POI[k][1] = groundAt(POI[k][0], POI[k][2]);
  buildThermals();
}

// same two triangles per cell as the rendered mesh
export function groundAt(x, z) {
  const fx = (x + HALF) / CELL, fz = (z + HALF) / CELL;
  if (fx < 0 || fz < 0 || fx >= N - 1 || fz >= N - 1) return SEABED;
  const i = fx | 0, j = fz | 0, u = fx - i, v = fz - j, k = j * N + i;
  const a = H[k], b = H[k + 1], c = H[k + N], d = H[k + N + 1];
  return u + v <= 1 ? a + (b - a) * u + (c - a) * v : d + (c - d) * (1 - u) + (b - d) * (1 - v);
}
export function normalAt(x, z, e = CELL) {
  const dx = groundAt(x + e, z) - groundAt(x - e, z), dz = groundAt(x, z + e) - groundAt(x, z - e);
  const l = Math.hypot(dx, 2 * e, dz);
  return [-dx / l, (2 * e) / l, -dz / l];
}

// named places. y is filled in by initWorld
export const POI = {
  cliff: [-2130, 0, -330],
  beach: [-2640, 0, 930],
  village: [-560, 0, 2080],
  light: [-1900, 0, 2250],
  fields: [250, 0, 950],
  peak: [1450, 0, -950],
  hut: [640, 0, -420],
  mill: [-150, 0, 1250],
  boat: [-900, 0, 3300],
};

// ---- the air ----
export const thermals = [];
export const CLOUDBASE = 1150;
const LEAN = 0.42; // metres downwind per metre of height: the columns tilt with the breeze
function buildThermals() {
  thermals.length = 0;
  const spots = [[250, 950, 150, 4.6], [620, 1350, 120, 3.6], [-250, 620, 125, 3.8], [900, 300, 135, 4.4], [1150, -350, 150, 5.2], [-1100, 300, 115, 3.4], [-700, 1500, 110, 3.2], [1900, -300, 130, 4.2], [300, -500, 120, 3.6], [-1500, 1300, 100, 3.0]];
  for (const [x, z, R, w] of spots) thermals.push({ x, z, R, w: w * 1.3, y0: groundAt(x, z), ph: (x * 0.013 + z * 0.007) % 6.28 });
}
// centre of a thermal at height y: the column leans downwind
export function thermalCentre(t, y) {
  const lean = Math.max(0, y - t.y0) * LEAN;
  return [t.x + WIND_DIR[0] * lean, t.z + WIND_DIR[2] * lean];
}

const out3 = [0, 0, 0];
// wind vector at a point (world, m/s). time gives the thermals a slow pulse and adds gusts.
export function windAt(x, y, z, time = 0, out = out3) {
  const g = groundAt(x, z), base = Math.max(g, 0), agl = Math.max(0, y - base);
  // a sea breeze: strongest at the coast and in the lowest few hundred metres, lighter inland and aloft
  const U = WIND_U * (0.5 + 0.5 * sstep(0, 150, agl)) * (1 - 0.55 * sstep(-1800, 300, x)) * (1 - 0.55 * sstep(250, 700, y));
  // ridge lift: the smoothed slope along the wind turns horizontal flow up (or down in the lee)
  const e = 70;
  const slope = (Math.max(groundAt(x + WIND_DIR[0] * e, z + WIND_DIR[2] * e), 0) - Math.max(groundAt(x - WIND_DIR[0] * e, z - WIND_DIR[2] * e), 0)) / (2 * e);
  // upwind of a wall the air already rises: look ahead
  const e2 = 160;
  const ahead = (Math.max(groundAt(x + WIND_DIR[0] * e2, z + WIND_DIR[2] * e2), 0) - base) / e2;
  let w = WIND_U * clamp(0.6 * slope + 0.36 * Math.max(ahead, 0), -0.5, 0.8) * Math.exp(-agl / 260) * sstep(0, 12, agl);
  let hw = 0, Ucore = 0;
  for (const t of thermals) {
    if (y > CLOUDBASE + 150) break;
    const lean = Math.max(0, y - t.y0) * LEAN;
    const dx = x - t.x - WIND_DIR[0] * lean, dz = z - t.z - WIND_DIR[2] * lean, d2 = dx * dx + dz * dz, R = t.R;
    if (d2 > R * R * 9) continue;
    const d = Math.sqrt(d2);
    const shape = sstep(0, 70, y - t.y0) * sstep(CLOUDBASE + 150, CLOUDBASE - 100, y);
    const pulse = 0.85 + 0.15 * Math.sin(time * 0.05 + t.ph);
    const core = Math.exp(-d2 / (R * R));
    w += t.w * pulse * shape * (core - 0.22 * Math.exp(-Math.pow((d - 1.9 * R) / (0.55 * R), 2)));
    // the rising air keeps the slow sideways speed it had near the ground: inside the core the wind is just the column's own tilt
    if (core * shape > hw) { hw = core * shape; Ucore = LEAN * Math.max(t.w * pulse * core - 1.2, 0); }
  }
  // gusts: stronger near rough ground
  const gk = 0.35 + 0.65 * Math.exp(-agl / 120);
  const gx = gnoise(x / 90 + time * 0.11, z / 90 + y / 70, 101), gy = gnoise(x / 70 - time * 0.13, z / 70 + y / 60, 202), gz = gnoise(x / 90 + y / 80, z / 90 + time * 0.1, 303);
  const Ue = U + (Ucore - U) * hw;
  out[0] = WIND_DIR[0] * Ue + gx * gk * 0.9;
  out[1] = w + gy * gk * 0.5;
  out[2] = WIND_DIR[2] * Ue + gz * gk * 0.9;
  return out;
}
