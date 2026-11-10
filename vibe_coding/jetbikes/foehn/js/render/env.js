// Sun, sky and air. A single-scattering atmosphere (Rayleigh + Mie + ozone) is ray-marched on
// the CPU into a small sky table whenever the time of day changes; every shader in the game
// reads its sky colour, its haze colour and its ambient light from that one table, so distant
// mountains fade into exactly the sky that is behind them.
import * as THREE from '../../lib/three.module.min.js';

export const TIMES = {
  morning: { name: 'בוקר', az: 0.35, el: 0.33, cloud: 0.40, haze: 1.25 },
  noon: { name: 'צהריים', az: 1.45, el: 1.0, cloud: 0.5, haze: 0.8 },
  afternoon: { name: 'אחר הצהריים', az: 2.75, el: 0.52, cloud: 0.46, haze: 1.0 },
  golden: { name: 'שעת הזהב', az: 2.95, el: 0.2, cloud: 0.46, haze: 1.2 },
  dusk: { name: 'דמדומים', az: 3.05, el: 0.06, cloud: 0.55, haze: 1.5 },
};

// uniforms shared (by reference) between all custom materials
export const U = {
  uSunDir: { value: new THREE.Vector3(0, 1, 0) },
  uSunCol: { value: new THREE.Color(1, 1, 1) },
  uAmbSky: { value: new THREE.Color(0.3, 0.4, 0.6) },
  uAmbGnd: { value: new THREE.Color(0.1, 0.1, 0.08) },
  uSkyLUT: { value: null },
  uFog: { value: new THREE.Vector4(2.2e-5, 1 / 1500, 900, 1) },   // density at base height, falloff 1/m, base height, haze brightness
  uTime: { value: 0 },
  uNoise: { value: null },
  uHeight: { value: null }, uNormal: { value: null }, uAux: { value: null },
  uWorld: { value: new THREE.Vector4(6144, 12, 1025, 0) },         // half size, cell, grid n
  uShadowMap: { value: null }, uShadowMat: { value: new THREE.Matrix4() }, uShadowOn: { value: 0 },
  uCloud: { value: new THREE.Vector4(0.46, 4300, 0, 0) },          // coverage, altitude
  uClipY: { value: -1e9 },
  uSunAz: { value: 0 },
};

const RE = 6360e3, RA = 6460e3, HR = 8000, HM = 1200;
const BR = [5.8e-6, 13.5e-6, 33.1e-6], BM = 21e-6, BO = [0.65e-6, 1.88e-6, 0.085e-6];
function rayOut(o, d, R) {
  const b = o[0] * d[0] + o[1] * d[1] + o[2] * d[2], c = o[0] * o[0] + o[1] * o[1] + o[2] * o[2] - R * R, disc = b * b - c;
  return disc < 0 ? -1 : -b + Math.sqrt(disc);
}
function optDepth(o, d, n = 6) {
  const L = rayOut(o, d, RA), ds = L / n;
  let r = 0, m = 0, oz = 0;
  for (let i = 0; i < n; i++) {
    const t = (i + 0.5) * ds, h = Math.hypot(o[0] + d[0] * t, o[1] + d[1] * t, o[2] + d[2] * t) - RE;
    if (h < -100) return [1e9, 1e9, 1e9];
    r += Math.exp(-h / HR) * ds; m += Math.exp(-h / HM) * ds; oz += Math.max(0, 1 - Math.abs(h - 25000) / 15000) * ds;
  }
  return [r, m, oz];
}
// radiance arriving at height h0 from direction d, sun direction s
function scatter(d, s, h0, mieScale) {
  const o = [0, RE + h0, 0];
  let L = rayOut(o, d, RA);
  // stop at the ground
  const b = o[1] * d[1], c = o[1] * o[1] - RE * RE, disc = b * b - c;
  if (disc > 0 && -b - Math.sqrt(disc) > 0) L = -b - Math.sqrt(disc);
  const N = 14, ds = L / N, mu = d[0] * s[0] + d[1] * s[1] + d[2] * s[2];
  const pr = 3 / (16 * Math.PI) * (1 + mu * mu), g = 0.78;
  const pm = 3 / (8 * Math.PI) * ((1 - g * g) * (1 + mu * mu)) / ((2 + g * g) * Math.pow(1 + g * g - 2 * g * mu, 1.5));
  const out = [0, 0, 0];
  let dr = 0, dm = 0, doz = 0;
  for (let i = 0; i < N; i++) {
    const t = (i + 0.5) * ds, p = [o[0] + d[0] * t, o[1] + d[1] * t, o[2] + d[2] * t];
    const h = Math.hypot(p[0], p[1], p[2]) - RE;
    const er = Math.exp(-h / HR) * ds, em = Math.exp(-h / HM) * ds * mieScale;
    dr += er; dm += em; doz += Math.max(0, 1 - Math.abs(h - 25000) / 15000) * ds;
    const [sr, sm, so] = optDepth(p, s);
    if (sr > 1e8) continue;
    for (let k = 0; k < 3; k++) {
      const tau = BR[k] * (dr + sr) + 1.1 * BM * (dm + sm * mieScale) + BO[k] * (doz + so);
      out[k] += Math.exp(-tau) * (er * BR[k] * pr + em * BM * pm);
    }
  }
  return out;
}
export function sunTransmittance(s, h0, mieScale) {
  const [r, m, o] = optDepth([0, RE + h0, 0], s, 12);
  return [0, 1, 2].map((k) => Math.exp(-(BR[k] * r + 1.1 * BM * m * mieScale + BO[k] * o)));
}

const LW = 96, LH = 48, SUN_I = 22;
export const lutV = (el) => Math.sqrt(Math.min(1, Math.max(0, (el + 0.12) / (Math.PI / 2 + 0.12))));
let lutData = null;

export function setTime(key, renderer) {
  const t = TIMES[key];
  const s = [Math.cos(t.el) * Math.cos(t.az), Math.sin(t.el), Math.cos(t.el) * Math.sin(t.az)];
  U.uSunDir.value.set(s[0], s[1], s[2]); U.uSunAz.value = t.az;
  const f32 = new Float32Array(LW * LH * 4);
  // table: u = azimuth from the sun (0..pi), v = elevation (denser near the horizon)
  for (let j = 0; j < LH; j++) {
    const v = (j + 0.5) / LH, el = v * v * (Math.PI / 2 + 0.12) - 0.12;
    for (let i = 0; i < LW; i++) {
      const az = (i + 0.5) / LW * Math.PI;
      const d = [Math.cos(el) * Math.cos(az), Math.sin(el), Math.cos(el) * Math.sin(az)];
      const c = scatter(d, [Math.cos(t.el), Math.sin(t.el), 0], 1400, t.haze);
      const k = (j * LW + i) * 4;
      f32[k] = c[0] * SUN_I; f32[k + 1] = c[1] * SUN_I; f32[k + 2] = c[2] * SUN_I; f32[k + 3] = 1;
    }
  }
  lutData = f32;
  const half = new Uint16Array(f32.length);
  for (let i = 0; i < f32.length; i++) half[i] = THREE.DataUtils.toHalfFloat(Math.min(f32[i], 60000));
  if (U.uSkyLUT.value) U.uSkyLUT.value.dispose();
  const tex = new THREE.DataTexture(half, LW, LH, THREE.RGBAFormat, THREE.HalfFloatType);
  tex.minFilter = tex.magFilter = THREE.LinearFilter; tex.wrapS = THREE.MirroredRepeatWrapping; tex.wrapT = THREE.ClampToEdgeWrapping; tex.needsUpdate = true;
  U.uSkyLUT.value = tex;
  // direct sun
  const tr = sunTransmittance(s, 1400, t.haze);
  const k = SUN_I * 0.115;
  U.uSunCol.value.setRGB(tr[0] * k, tr[1] * k, tr[2] * k);
  // ambient: mean of the upper sky, a little bounce from below
  const amb = [0, 0, 0]; let n = 0;
  for (let j = Math.floor(LH * 0.35); j < LH; j++) for (let i = 0; i < LW; i += 4) { const q = (j * LW + i) * 4; amb[0] += f32[q]; amb[1] += f32[q + 1]; amb[2] += f32[q + 2]; n++; }
  const ak = 2.8 / n;
  // shade is lit by blue sky but also by clouds and bright slopes: pull the ambient towards neutral
  const am = (amb[0] + amb[1] + amb[2]) / 3;
  for (let k = 0; k < 3; k++) amb[k] = amb[k] * 0.62 + am * 0.38;
  U.uAmbSky.value.setRGB(amb[0] * ak, amb[1] * ak, amb[2] * ak);
  const sc = U.uSunCol.value, up = Math.max(0.05, s[1]);
  U.uAmbGnd.value.setRGB(0.10 * sc.r * up + amb[0] * ak * 0.25, 0.10 * sc.g * up + amb[1] * ak * 0.25, 0.07 * sc.b * up + amb[2] * ak * 0.2);
  U.uCloud.value.x = t.cloud;
  U.uFog.value.set(2.0e-5 * t.haze, 1 / 1700, 900, 1);
  return { sun: s, key };
}
// sky radiance looked up on the CPU (for lights, fog of built-in materials)
export function skyColor(az, el) {
  const u = Math.abs(((az - U.uSunAz.value + Math.PI * 3) % (2 * Math.PI)) - Math.PI) / Math.PI;
  const i = Math.min(LW - 1, Math.floor(u * LW)), j = Math.min(LH - 1, Math.floor(lutV(el) * LH)), k = (j * LW + i) * 4;
  return [lutData[k], lutData[k + 1], lutData[k + 2]];
}

export function makeNoiseTexture() {
  const N = 256, d = new Uint8Array(N * N * 4);
  let s = 1234567;
  const r = () => ((s = (s * 1664525 + 1013904223) >>> 0) / 4294967296);
  for (let i = 0; i < d.length; i++) d[i] = Math.floor(r() * 256);
  const tex = new THREE.DataTexture(d, N, N, THREE.RGBAFormat);
  tex.wrapS = tex.wrapT = THREE.RepeatWrapping; tex.minFilter = THREE.LinearMipmapLinearFilter; tex.magFilter = THREE.LinearFilter;
  tex.generateMipmaps = true; tex.needsUpdate = true;
  U.uNoise.value = tex;
  return tex;
}

// ---------------- shared GLSL ----------------
export const GLSL_COMMON = /* glsl */`
uniform vec3 uSunDir; uniform vec3 uSunCol; uniform vec3 uAmbSky; uniform vec3 uAmbGnd;
uniform sampler2D uSkyLUT; uniform sampler2D uNoise; uniform vec4 uFog; uniform float uTime; uniform vec4 uCloud;
uniform float uSunAz; uniform float uClipY;
const float PI = 3.14159265;
float vnoise(vec2 p) { return texture2D(uNoise, p / 256.0).r; }          // smooth value noise, period 256
vec4 vnoise4(vec2 p) { return texture2D(uNoise, p / 256.0); }
float fbm3(vec2 p) { return 0.57 * vnoise(p) + 0.29 * vnoise(p * 2.13 + 31.7) + 0.14 * vnoise(p * 4.37 + 71.3); }
vec3 skyLUT(vec3 d) {
  float az = atan(d.z, d.x) - uSunAz;
  float u = abs(mod(az + PI, 2.0 * PI) - PI) / PI;
  float el = asin(clamp(d.y, -1.0, 1.0));
  float v = sqrt(clamp((el + 0.12) / (PI * 0.5 + 0.12), 0.0, 1.0));
  return texture2D(uSkyLUT, vec2(u, v)).rgb;
}
// cloud cover seen from the ground at world xz (also drives cloud shadows)
float cloudDensity(vec2 xz) {
  vec2 p = xz / 9000.0 + vec2(uTime * 0.0012, uTime * 0.0005);
  float n = 0.5 * vnoise(p * 16.0) + 0.27 * vnoise(p * 37.0 + 11.0) + 0.15 * vnoise(p * 83.0 + 5.0) + 0.08 * vnoise(p * 190.0);
  return smoothstep(1.0 - uCloud.x - 0.1, 1.0 - uCloud.x + 0.16, n);
}
float cloudShadow(vec3 wp) {
  vec2 xz = wp.xz + uSunDir.xz / max(uSunDir.y, 0.08) * (uCloud.y - wp.y);
  return 1.0 - 0.62 * cloudDensity(xz);
}
// aerial perspective: exponential height fog (closed form), blue-shifted extinction,
// in-scatter colour taken from the sky table near the horizon in the viewing direction
vec3 applyFog(vec3 col, vec3 wp) {
  vec3 d = wp - cameraPosition; float dist = length(d); vec3 dir = d / max(dist, 1e-3);
  float k = dir.y * uFog.y * dist;
  float od = uFog.x * exp(-(cameraPosition.y - uFog.z) * uFog.y) * dist * (abs(k) > 1e-3 ? (1.0 - exp(-k)) / k : 1.0);
  vec3 T = exp(-od * vec3(0.72, 1.0, 1.42));
  vec3 hz = skyLUT(normalize(vec3(dir.x, 0.035 + 0.25 * max(dir.y, 0.0), dir.z)));
  return mix(hz * uFog.w, col, T);
}
`;
