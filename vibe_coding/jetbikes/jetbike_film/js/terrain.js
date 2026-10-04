// Procedural alpine valley. One height function is shared by the physics (ground contact,
// jet ground effect) and by the renderer (mesh, trees, dust impingement).
// World frame: x east, y up, z south. Metres. y=0 is the launch pad (1100 m above sea level).

import { smoothstep, clamp } from './vmath.js';

export const BASE_ALTITUDE = 1100; // m above sea level at y=0
export const WATER_Y = -3.0;
export const PAD = [0, 0, 0];
export const LAKE = { x: 640, dz: 18, rx: 250, rz: 125 };
export const HILL = { x: 1080, dz: 0, r: 80, h: 105 };
export const LAND = { x: 425, z: -14 };   // lakeside meadow where the flight ends

function hash(ix, iz) {
  let h = Math.imul(ix, 374761393) ^ Math.imul(iz, 668265263);
  h = Math.imul(h ^ (h >>> 13), 1274126177);
  h ^= h >>> 16;
  return (h >>> 0) / 4294967296;
}

// 2D gradient noise, output ~[0,1]
function vnoise(x, z) {
  const ix = Math.floor(x), iz = Math.floor(z);
  const fx = x - ix, fz = z - iz;
  const ux = fx * fx * fx * (fx * (fx * 6 - 15) + 10);
  const uz = fz * fz * fz * (fz * (fz * 6 - 15) + 10);
  const g = (i, j, dx, dz) => {
    const a = hash(i, j) * 6.283185307;
    return Math.cos(a) * dx + Math.sin(a) * dz;
  };
  const a = g(ix, iz, fx, fz), b = g(ix + 1, iz, fx - 1, fz);
  const c = g(ix, iz + 1, fx, fz - 1), d = g(ix + 1, iz + 1, fx - 1, fz - 1);
  const v = a + (b - a) * ux + (c - a) * uz + (a - b - c + d) * ux * uz;
  return 0.5 + v * 0.75;
}

// each octave is rotated to hide the lattice
const ROT = [0.8, 0.6];
function rot(x, z, k) {
  let c = 1, s = 0;
  for (let i = 0; i < k; i++) { const c2 = c * ROT[0] - s * ROT[1]; s = s * ROT[0] + c * ROT[1]; c = c2; }
  return [x * c - z * s, x * s + z * c];
}

export function fbm(x, z, oct = 4) {
  let s = 0, a = 0.5, f = 1, n = 0;
  for (let i = 0; i < oct; i++) {
    const [rx, rz] = rot(x * f, z * f, i);
    s += a * vnoise(rx + i * 17.3, rz - i * 9.1);
    n += a;
    a *= 0.5;
    f *= 2.03;
  }
  return s / n;
}

export function ridged(x, z, oct = 5) {
  let s = 0, a = 0.5, f = 1, n = 0, w = 1;
  for (let i = 0; i < oct; i++) {
    const [rx, rz] = rot(x * f, z * f, i);
    let v = 1 - Math.abs(vnoise(rx + i * 31.7, rz + i * 11.9) * 2 - 1);
    v *= v;
    s += a * v * w;
    w = clamp(v * 1.6, 0, 1);
    n += a;
    a *= 0.5;
    f *= 2.1;
  }
  return s / n;
}

const Z0 = 230 * Math.sin(0) + 110 * Math.sin(0.9);
export function valleyZ(x) {
  return 230 * Math.sin(x / 760) + 110 * Math.sin(x / 310 + 0.9) - Z0;
}

function rawHeight(x, z) {
  const zc = valleyZ(x);
  const dz = z - zc;
  const d = Math.abs(dz);

  // valley floor: gentle meadows
  let floor = 7 * (fbm(x * 0.004, z * 0.004, 3) - 0.5) + 6 * (fbm(x * 0.0011 + 3.3, z * 0.0011, 2) - 0.5);
  floor += 2.5;

  // lake basin
  const lx = (x - LAKE.x) / LAKE.rx;
  const lz = (z - valleyZ(LAKE.x) - LAKE.dz) / LAKE.rz;
  const le = Math.sqrt(lx * lx + lz * lz) + 0.12 * (fbm(x * 0.01, z * 0.01, 2) - 0.5);
  floor -= 13 * smoothstep(1.15, 0.35, le);

  // rocky island hill that the bike turns around
  const hz = valleyZ(HILL.x) + HILL.dz;
  const hr = Math.hypot(x - HILL.x, z - hz);
  const rock = 0.75 + 0.5 * ridged(x * 0.02, z * 0.02, 3);
  floor += HILL.h * Math.exp(-Math.pow(hr / HILL.r, 2)) * rock;

  // valley walls rising into ridged mountains
  const wobble = 90 * (fbm(x * 0.0025 + 7, z * 0.0025, 3) - 0.5);
  const wt = smoothstep(120, 700, d + wobble);
  const mtn = 40 + 900 * Math.pow(ridged(x * 0.0008 + 11.2, z * 0.0008 - 7.4, 6), 1.25)
    + 380 * fbm(x * 0.00045 - 2, z * 0.00045 + 5, 4) + 0.35 * Math.max(0, d - 300);
  let h = floor + mtn * wt * (0.6 + 0.4 * wt);
  // gullies, spurs and ledges on the walls
  h += wt * (95 * (ridged(x * 0.0055 + 3.1, z * 0.0055 - 1.7, 4) - 0.45) + 28 * (fbm(x * 0.021, z * 0.021, 3) - 0.5));

  return h;
}

let landY = null;
export function heightAt(x, z) {
  let h = rawHeight(x, z);
  // launch pad: flattened gravel circle
  const rp = Math.hypot(x - PAD[0], z - PAD[2]);
  h = h + (PAD[1] - h) * (1 - smoothstep(16, 55, rp));
  // landing meadow: levelled at its own natural height
  landY ??= rawHeight(LAND.x, LAND.z);
  const rl = Math.hypot(x - LAND.x, z - LAND.z);
  h = h + (landY - h) * (1 - smoothstep(12, 45, rl));
  return h;
}

// surface under a point, including the lake
export function surfaceAt(x, z) {
  return Math.max(heightAt(x, z), WATER_Y);
}

export function normalAt(x, z, e = 0.5) {
  const hx = heightAt(x + e, z) - heightAt(x - e, z);
  const hz = heightAt(x, z + e) - heightAt(x, z - e);
  const n = [-hx, 2 * e, -hz];
  const l = Math.hypot(n[0], n[1], n[2]);
  return [n[0] / l, n[1] / l, n[2] / l];
}

// air density [kg/m^3] at world height y (ISA troposphere)
export function airDensity(y) {
  const alt = BASE_ALTITUDE + y;
  return 1.225 * Math.pow(1 - 2.25577e-5 * alt, 4.2559);
}
