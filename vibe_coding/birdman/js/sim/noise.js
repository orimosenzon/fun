// Deterministic noise helpers (shared by the terrain bake, the wind field and the scenery).

export function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// integer lattice hash -> [0,1)
export function hash2(ix, iz, seed = 0) {
  let h = Math.imul(ix | 0, 374761393) ^ Math.imul(iz | 0, 668265263) ^ Math.imul(seed | 0, 1274126177);
  h = Math.imul(h ^ (h >>> 13), 1274126177);
  h ^= h >>> 16;
  return (h >>> 0) / 4294967296;
}

const fade = (t) => t * t * t * (t * (t * 6 - 15) + 10);

// gradient noise, range about [-1,1]
export function gnoise(x, z, seed = 0) {
  const ix = Math.floor(x), iz = Math.floor(z);
  const fx = x - ix, fz = z - iz;
  const g = (i, j, dx, dz) => {
    const a = hash2(i, j, seed) * 6.283185307;
    return Math.cos(a) * dx + Math.sin(a) * dz;
  };
  const u = fade(fx), v = fade(fz);
  const a = g(ix, iz, fx, fz), b = g(ix + 1, iz, fx - 1, fz);
  const c = g(ix, iz + 1, fx, fz - 1), d = g(ix + 1, iz + 1, fx - 1, fz - 1);
  return (a + (b - a) * u + (c - a + (a - b + d - c) * u) * v) * 1.41;
}

export function fbm(x, z, oct = 5, seed = 0, lac = 2.03, gain = 0.5) {
  let s = 0, a = 1, n = 0;
  for (let i = 0; i < oct; i++) {
    s += a * gnoise(x, z, seed + i * 17);
    n += a; a *= gain; x = x * lac + 13.7; z = z * lac - 7.1;
  }
  return s / n;
}

// ridged multifractal, range about [0,1]
export function ridged(x, z, oct = 6, seed = 0, lac = 2.07, gain = 0.52) {
  let s = 0, a = 1, n = 0, w = 1;
  for (let i = 0; i < oct; i++) {
    let v = 1 - Math.abs(gnoise(x, z, seed + i * 31));
    v *= v;
    s += a * v * w;
    n += a;
    w = Math.min(1, Math.max(0, v * 1.6));
    a *= gain; x = x * lac + 5.3; z = z * lac + 9.1;
  }
  return s / n;
}
