// Palettes: a few colour stops around a circle, blended in OKLab (so the way from one stop to
// the next stays vivid instead of passing through grey) into a 2048-wide strip with mip-maps.
// The mip-maps matter: where the colour bands get thinner than a pixel, the shader reads a
// coarser level and gets the true average of the bands instead of confetti.
import { THREE } from '../foehn.js';

export const LUT_W = 2048;
export const PALETTES = [
  { id: 'classic', name: 'קלאסי', stops: [[0, '#00076a'], [0.16, '#1c6fd6'], [0.42, '#f1ffff'], [0.6425, '#ffaa00'], [0.8575, '#1a0500']] },
  { id: 'flame', name: 'להבה', stops: [[0, '#12001f'], [0.18, '#8c0a12'], [0.36, '#ff4a0a'], [0.54, '#ffc21a'], [0.68, '#fffbe0'], [0.84, '#ff7a1a']] },
  { id: 'neon', name: 'נאון', stops: [[0, '#0b0226'], [0.17, '#ff1f6e'], [0.36, '#ffd23f'], [0.52, '#06e6e0'], [0.7, '#e9fbff'], [0.86, '#7a16ff']] },
  { id: 'aurora', name: 'זוהר צפוני', stops: [[0, '#02142b'], [0.2, '#00d9a3'], [0.38, '#c8ff3d'], [0.55, '#f7fff2'], [0.72, '#ff3fa4'], [0.88, '#4a1fd6']] },
  { id: 'gold', name: 'זהב עתיק', stops: [[0, '#140a02'], [0.22, '#9a5a08'], [0.42, '#ffcf4d'], [0.56, '#fff8dc'], [0.74, '#2f8f8a'], [0.9, '#102a5c']] },
  { id: 'ocean', name: 'אוקיינוס', stops: [[0, '#010b24'], [0.22, '#0a4fb8'], [0.42, '#19c8d8'], [0.58, '#f4fffb'], [0.74, '#ffb347'], [0.9, '#b8336a']] },
  { id: 'candy', name: 'קשת', stops: [[0, '#ff2a2a'], [0.14, '#ff9a00'], [0.28, '#ffe600'], [0.44, '#27d94a'], [0.6, '#12c8ff'], [0.76, '#3a3cff'], [0.9, '#d326ff']] },
  { id: 'ink', name: 'דיו', stops: [[0, '#05060a'], [0.3, '#f6f3ea'], [0.5, '#0a0c12'], [0.62, '#d7263d'], [0.8, '#f6f3ea']] },
];

const hex = (h) => [parseInt(h.slice(1, 3), 16) / 255, parseInt(h.slice(3, 5), 16) / 255, parseInt(h.slice(5, 7), 16) / 255];
const toLin = (c) => (c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4));
const toSrgb = (c) => (c <= 0.0031308 ? 12.92 * c : 1.055 * Math.pow(c, 1 / 2.4) - 0.055);
function oklab([r, g, b]) {
  r = toLin(r); g = toLin(g); b = toLin(b);
  const l = Math.cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b), m = Math.cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b), s = Math.cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b);
  return [0.2104542553 * l + 0.793617785 * m - 0.0040720468 * s, 1.9779984951 * l - 2.428592205 * m + 0.4505937099 * s, 0.0259040371 * l + 0.7827717662 * m - 0.808675766 * s];
}
function unOklab([L, a, b]) {
  const l = (L + 0.3963377774 * a + 0.2158037573 * b) ** 3, m = (L - 0.1055613458 * a - 0.0638541728 * b) ** 3, s = (L - 0.0894841775 * a - 1.291485548 * b) ** 3;
  return [4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s, -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s, -0.0041960863 * l - 0.7034186147 * m + 1.707614701 * s].map((v) => Math.min(1, Math.max(0, toSrgb(Math.min(1, Math.max(0, v))))));
}
// colour of palette p at t (any real; the palette is a circle)
export function paletteAt(p, t) {
  const st = p.lab || (p.lab = p.stops.map(([x, h]) => [x, oklab(hex(h))])), n = st.length;
  t -= Math.floor(t);
  let i = n - 1;                                   // the arc that closes the circle, unless t lies between two stops
  for (let k = 0; k < n - 1; k++) if (t >= st[k][0] && t < st[k + 1][0]) { i = k; break; }
  const a = st[i], b = st[(i + 1) % n], x0 = a[0], x1 = i === n - 1 ? b[0] + 1 : b[0];
  let u = ((t < x0 ? t + 1 : t) - x0) / (x1 - x0);
  u = u * u * (3 - 2 * u) * 0.35 + u * 0.65;       // a little ease: the stops themselves show slightly longer
  return unOklab(a[1].map((v, k) => v + (b[1][k] - v) * u));
}
export function paletteCss(p) {
  const parts = [];
  for (let i = 0; i <= 16; i++) { const c = paletteAt(p, i / 16).map((v) => Math.round(v * 255)); parts.push(`rgb(${c[0]},${c[1]},${c[2]}) ${(i / 16 * 100).toFixed(0)}%`); }
  return `linear-gradient(90deg,${parts.join(',')})`;
}

export class PaletteTex {
  constructor() {
    this.data = new Uint8Array(LUT_W * 4);
    this.tex = new THREE.DataTexture(this.data, LUT_W, 1, THREE.RGBAFormat, THREE.UnsignedByteType);
    this.tex.colorSpace = THREE.SRGBColorSpace;               // the shader gets linear light, and the mip-maps are averaged in linear light
    this.tex.wrapS = THREE.RepeatWrapping; this.tex.wrapT = THREE.ClampToEdgeWrapping;
    this.tex.minFilter = THREE.LinearMipmapLinearFilter; this.tex.magFilter = THREE.LinearFilter; this.tex.generateMipmaps = true;
    this.mean = [0.5, 0.5, 0.5];
    this.set(0);
  }
  set(i) {
    const p = PALETTES[i]; this.index = i;
    const m = [0, 0, 0];
    for (let k = 0; k < LUT_W; k++) {
      const c = paletteAt(p, (k + 0.5) / LUT_W);
      this.data[k * 4] = Math.round(c[0] * 255); this.data[k * 4 + 1] = Math.round(c[1] * 255); this.data[k * 4 + 2] = Math.round(c[2] * 255); this.data[k * 4 + 3] = 255;
      m[0] += toLin(c[0]); m[1] += toLin(c[1]); m[2] += toLin(c[2]);
    }
    this.mean = m.map((v) => v / LUT_W);
    this.tex.needsUpdate = true;
  }
}

// the colouring that the map, the mini-map and the landscape share (GLSL)
export const TONE_GLSL = /* glsl */`
uniform sampler2D uPal;
uniform vec4 uTone;        // level the colours start from, their spread, bands per doubling, offset (colour cycling)
const float LN2 = 0.6931472;
// lo, w: the level the colours start from and their spread (uTone.xy for a flat picture; the land has a pair per level)
// dsl: how much the smooth count changes across one pixel. Bands thinner than a pixel are averaged.
vec3 paletteLW(float sl, float dsl, float lo, float w) {
  float x = (sl - lo) / w, t = uTone.z * sign(x) * log2(1.0 + abs(x)) + uTone.w;
  float dt = uTone.z / (LN2 * (w + abs(sl - lo))) * dsl;
  return textureLod(uPal, vec2(t, 0.5), log2(max(dt * ${LUT_W}.0, 1.0))).rgb;
}
vec3 paletteAA(float sl, float dsl) { return paletteLW(sl, dsl, uTone.x, uTone.y); }
vec3 toSrgb(vec3 c) { c = clamp(c, 0.0, 1.0); return mix(12.92 * c, 1.055 * pow(c, vec3(1.0 / 2.4)) - 0.055, step(vec3(0.0031308), c)); }
`;
