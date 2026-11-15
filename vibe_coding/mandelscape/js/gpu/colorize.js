// From field data to a picture. Recolouring never recomputes the fractal: a new palette, colour
// cycling or a different band density only run this cheap pass again. The refined picture is the
// average of up to eight differently jittered samples per pixel, each coloured on its own.
import { THREE } from '../foehn.js';
import { rawMat } from './fieldpass.js';
import { TONE_GLSL } from './palette.js';

const FRAG = /* glsl */`
precision highp float; precision highp int; precision highp sampler2D; precision highp sampler2DArray;
out vec4 outColor;
uniform sampler2DArray uData;     // refined samples, one layer each
uniform sampler2D uMotion;        // the single quick sample shown while the view moves
uniform int uCount;               // refined layers that are ready; 0 = show uMotion
uniform vec4 uRect;               // where on the canvas (x, y, w, h in pixels)
uniform vec2 uDataRes;
uniform float uPx;                // field distance units -> canvas pixels
uniform float uRelief, uEdge;
uniform vec2 uLight;
${TONE_GLSL}
vec3 shade(vec4 d) {
  if (d.x < 0.0) return vec3(0.0);
  float sl = d.x * 32.0 + d.y;
  float de = d.z * uPx;                                   // distance to the set, in pixels
  vec3 c = paletteAA(sl, 1.0 / (LN2 * max(de, 1e-4)));
  // relief: light from the upper left on the slope of the distance field (not where it is finer than the pixels)
  float l = dot(vec2(cos(d.w), sin(d.w)), uLight);
  c *= mix(1.0, clamp(0.60 + 0.62 * l, 0.12, 1.35), uRelief * smoothstep(0.6, 3.0, de));
  // the set and its filaments as a crisp dark line
  c *= smoothstep(0.0, 1.0, de * uEdge);
  return c;
}
void main() {
  vec2 p = gl_FragCoord.xy - uRect.xy;
  vec3 col = vec3(0.0);
  if (uCount == 0) {
    vec2 q = p / uRect.zw * uDataRes - 0.5; ivec2 i = ivec2(floor(q)); vec2 f = q - vec2(i);
    ivec2 mx = ivec2(uDataRes) - 1;
    vec3 a = shade(texelFetch(uMotion, clamp(i, ivec2(0), mx), 0)), b = shade(texelFetch(uMotion, clamp(i + ivec2(1, 0), ivec2(0), mx), 0));
    vec3 c = shade(texelFetch(uMotion, clamp(i + ivec2(0, 1), ivec2(0), mx), 0)), d = shade(texelFetch(uMotion, clamp(i + ivec2(1, 1), ivec2(0), mx), 0));
    col = mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
  } else {
    ivec2 i = ivec2(p);
    for (int k = 0; k < uCount; k++) col += shade(texelFetch(uData, ivec3(i, k), 0));
    col /= float(uCount);
  }
  outColor = vec4(toSrgb(col), 1.0);
}`;

export function makeColorize(pal, tone) {
  const u = {
    uData: { value: null }, uMotion: { value: null }, uCount: { value: 0 }, uRect: { value: new THREE.Vector4() }, uDataRes: { value: new THREE.Vector2(1, 1) },
    uPx: { value: 1 }, uRelief: { value: 0 }, uEdge: { value: 0.9 }, uLight: { value: new THREE.Vector2(-0.6, 0.8) },
    uPal: { value: pal.tex }, uTone: tone,
  };
  return { u, mat: rawMat(FRAG, u) };
}

// a float target with several layers
export function layersRT(w, h, n, format, type, linear = false) {
  const rt = new THREE.WebGLArrayRenderTarget(w, h, n, { depthBuffer: false });
  const t = rt.texture; t.format = format; t.type = type; t.generateMipmaps = false;
  t.minFilter = t.magFilter = linear ? THREE.LinearFilter : THREE.NearestFilter;
  return rt;
}
export function floatRT(w, h, format = THREE.RGBAFormat, type = THREE.HalfFloatType) {
  const rt = new THREE.WebGLRenderTarget(w, h, { format, type, depthBuffer: false, minFilter: THREE.NearestFilter, magFilter: THREE.NearestFilter });
  rt.texture.generateMipmaps = false;
  return rt;
}
