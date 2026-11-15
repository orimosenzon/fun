// The landscape on the GPU.
//
//   Clip      keeps the pyramid of height maps that core/world.js plans: one float texture with a
//             layer per level, each layer filled by the field shader a strip per frame.
//   Terrain   draws it: one square grid per level, its vertices sitting on fixed texels of that
//             level (so nothing swims), lifted by the distance stored there. Every pixel then
//             reads the field again for its colour and its normal, so the detail on the ground is
//             as fine as the height maps, not as coarse as the mesh.
//
// A level is drawn only outside the finer one inside it, and towards its own rim it blends into
// the coarser one around it, heights and colours alike: no seams, no popping.
import { THREE } from '../foehn.js';
import { layersRT } from './colorize.js';
import { TONE_GLSL } from './palette.js';
import { D0, TERR_K, TERR_R } from '../core/world.js';

export const MAXL = 10, GRID = 128;

export const SKY_GLSL = /* glsl */`
uniform vec3 uSun;
// an evening sky over a strange planet: deep blue overhead, a pale band at the horizon, a low white sun
vec3 skyCol(vec3 d) {
  float h = clamp(d.y, 0.0, 1.0);
  vec3 c = mix(vec3(0.34, 0.47, 0.72), vec3(0.015, 0.035, 0.13), 1.0 - exp(-3.2 * h));
  float sd = max(dot(d, uSun), 0.0), flat_ = 1.0 - h;
  c += vec3(1.0, 0.52, 0.22) * pow(sd, 6.0) * 0.38 * flat_ * flat_;
  c += vec3(1.0, 0.86, 0.66) * pow(sd, 90.0) * 0.9;
  return c;
}`;

const COMMON = /* glsl */`
precision highp float; precision highp int; precision highp sampler2DArray; precision highp sampler2D;
#define MAXL ${MAXL}
uniform sampler2DArray uLev;
uniform int uNL, uI;
uniform float uLay[MAXL], uHalf[MAXL], uMet[MAXL], uSat[MAXL], uFade[MAXL], uTLo[MAXL], uTW[MAXL];
uniform vec2 uCen[MAXL];
uniform float uSlope, uTexN;
vec2 toUv(int i, vec2 xz) { vec2 r = (xz - uCen[i]) / uHalf[i]; return vec2(r.x, -r.y) * 0.5 + 0.5; }
// (smooth count, distance) of level i at a place: bilinear between texel centres
vec2 fetchL(int i, vec2 uv) {
#ifdef MANUAL_BILINEAR
  vec2 q = uv * uTexN - 0.5; ivec2 p = ivec2(floor(q)); vec2 f = q - vec2(p); int z = int(uLay[i] + 0.5); ivec2 mx = ivec2(int(uTexN) - 1);
  vec2 a = texelFetch(uLev, ivec3(clamp(p, ivec2(0), mx), z), 0).xy, b = texelFetch(uLev, ivec3(clamp(p + ivec2(1, 0), ivec2(0), mx), z), 0).xy;
  vec2 c = texelFetch(uLev, ivec3(clamp(p + ivec2(0, 1), ivec2(0), mx), z), 0).xy, d = texelFetch(uLev, ivec3(clamp(p + ivec2(1, 1), ivec2(0), mx), z), 0).xy;
  return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
#else
  return textureLod(uLev, vec3(uv, uLay[i]), 0.0).xy;
#endif
}
// how much of the coarser level around it level i shows at this place: all of it at the rim, and while it is new
float blendW(int i, vec2 xz) {
  if (i >= uNL - 1) return 0.0;
  vec2 q = abs(xz - uCen[i]) / uHalf[i];
  return max(smoothstep(0.70, 0.96, max(q.x, q.y)), 1.0 - uFade[i]);
}
// x: smooth count, y: distance to the set in metres, z: height in metres
// terraces: one step of the land per colour band (the same formula as terrace() in core/world.js)
float terrace(float sl, float dTex) {
  float s = fract(sl), e = clamp((s - ${(1 - TERR_R).toFixed(3)}) / ${TERR_R.toFixed(3)}, 0.0, 1.0);
  return 1.0 + ${TERR_K.toFixed(3)} * smoothstep(12.0, 40.0, 0.6931472 * dTex) * (min(s / ${(1 - TERR_R).toFixed(3)}, 1.0) * (1.0 - e * e * (3.0 - 2.0 * e)) - 0.5);
}
vec3 levelAt(int i, vec2 xz) { vec2 s = fetchL(i, toUv(i, xz)); return vec3(s.x, s.y * uMet[i], uSlope * s.y * uMet[i] / (1.0 + s.y * uSat[i]) * terrace(s.x, s.y * uTexN * 0.5)); }
vec3 fieldAt(int i, vec2 xz) {
  float w = blendW(i, xz);
  vec3 a = levelAt(i, xz);
  if (w > 0.0) a = mix(a, levelAt(i + 1, xz), w);
  return a;
}`;

const VERT = /* glsl */`
${COMMON}
in vec3 position;
uniform mat4 viewMatrix, projectionMatrix;
out vec2 vXZ;
void main() {
  vec2 xz = uCen[uI] + (position.xy * 2.0 - 1.0) * vec2(1.0, -1.0) * uHalf[uI];
  vXZ = xz;
  gl_Position = projectionMatrix * viewMatrix * vec4(xz.x, fieldAt(uI, xz).z, xz.y, 1.0);
  // well inside the finer level this one is not seen at all: throw the vertex away (whatever is left of its triangles is discarded per pixel)
  if (uI > 0) { vec2 q = abs(xz - uCen[uI - 1]); if (max(q.x, q.y) < uHalf[uI - 1] - 6.0 * uHalf[uI] / ${GRID}.0) gl_Position = vec4(0.0, 0.0, -5.0, 1.0); }
}`;

const SHADE = /* glsl */`
uniform vec3 cameraPosition;
uniform vec4 uShadow;          // the bike's shadow on the ground: x, z, radius, strength
uniform vec3 uFog;             // density (1/m), time, palette saturation boost
uniform vec2 uWater;           // offset of the ripple pattern (m)
${SKY_GLSL}
${TONE_GLSL}
vec3 finish(vec3 col, vec3 P) {
  vec3 v = P - cameraPosition; float dist = length(v);
  float f = 1.0 - exp(-dist * uFog.x);
  vec3 d = v / dist;
  return mix(col, skyCol(normalize(vec3(d.x, max(d.y, 0.0) * 0.5 + 0.015, d.z))) * 0.8, f);
}
vec3 landLight(vec3 alb, vec3 N, vec3 P) {
  float ndl = max(dot(N, uSun), 0.0);
  vec3 amb = mix(vec3(0.07, 0.07, 0.10), vec3(0.24, 0.29, 0.42), N.y * 0.5 + 0.5);
  float sh = 1.0 - uShadow.w * (1.0 - smoothstep(0.35, 1.0, length(P.xz - uShadow.xy) / uShadow.z));
  vec3 V = normalize(cameraPosition - P);
  float spec = pow(max(dot(N, normalize(uSun + V)), 0.0), 40.0) * 0.35 * sh;          // a little polish, as on glazed tile
  return alb * (amb + vec3(2.6, 2.42, 2.15) * ndl * sh) + vec3(1.0, 0.95, 0.85) * spec * ndl;
}
vec3 waterCol(vec3 P, float foot) {
  vec3 V = normalize(cameraPosition - P);
  vec2 p = P.xz + uWater; float t = uFog.y, k = 1.0 / (1.0 + foot * foot * 3.0);
  vec2 g = vec2(0.8, 0.6) * cos(dot(p, vec2(0.8, 0.6)) * 0.55 + t * 1.3) * 0.030 + vec2(-0.5, 0.87) * cos(dot(p, vec2(-0.5, 0.87)) * 1.3 - t * 1.9) * 0.022
         + vec2(0.3, -0.95) * cos(dot(p, vec2(0.3, -0.95)) * 3.1 + t * 2.7) * 0.014 + vec2(-0.9, -0.4) * cos(dot(p, vec2(-0.9, -0.4)) * 0.13 + t * 0.5) * 0.02;
  vec3 N = normalize(vec3(-g.x * k, 1.0, -g.y * k));
  vec3 R = reflect(-V, N); R.y = abs(R.y);
  float F = 0.025 + 0.975 * pow(1.0 - max(dot(N, V), 0.0), 5.0);
  vec3 c = mix(vec3(0.003, 0.006, 0.016), skyCol(R), F);
  c += vec3(1.0, 0.9, 0.75) * pow(max(dot(R, uSun), 0.0), 900.0) * 60.0 * F * k;
  float sh = 1.0 - 0.5 * uShadow.w * (1.0 - smoothstep(0.35, 1.0, length(P.xz - uShadow.xy) / uShadow.z));
  return c * sh;
}`;

const FRAG = /* glsl */`
${COMMON}
${SHADE}
in vec2 vXZ;
out vec4 outColor;
void main() {
  if (uI > 0) { vec2 q = abs(vXZ - uCen[uI - 1]); if (max(q.x, q.y) < uHalf[uI - 1] * (1.0 - 1.0 / ${GRID}.0)) discard; }
  float foot = max(length(dFdx(vXZ)), length(dFdy(vXZ)));               // metres of ground under one pixel
  // field and colour: this level's, blended towards the coarser one's at the rim
  float w = blendW(uI, vXZ);
  vec3 F = levelAt(uI, vXZ);
  vec3 alb = paletteLW(F.x, foot / (LN2 * max(F.y, 1e-3)), uTLo[uI], uTW[uI]);
  if (w > 0.0) { vec3 G = levelAt(uI + 1, vXZ); alb = mix(alb, paletteLW(G.x, foot / (LN2 * max(G.y, 1e-3)), uTLo[uI + 1], uTW[uI + 1]), w); F = mix(F, G, w); }
  float e = max(2.0 * uHalf[uI] / uTexN, 0.5 * foot);
  float hx = fieldAt(uI, vXZ + vec2(e, 0.0)).z - fieldAt(uI, vXZ - vec2(e, 0.0)).z, hz = fieldAt(uI, vXZ + vec2(0.0, e)).z - fieldAt(uI, vXZ - vec2(0.0, e)).z;
  vec3 N = normalize(vec3(-hx, 2.0 * e, -hz)), P = vec3(vXZ.x, F.z, vXZ.y);
  float l = dot(alb, vec3(0.2126, 0.7152, 0.0722)); alb = max(mix(vec3(l), alb, uFog.z), 0.0);
  alb *= mix(0.22, 1.0, smoothstep(0.0, max(1.2, 2.5 * foot), F.y));    // dark, wet ground at the water's edge: the filaments show as lines
  vec3 col = landLight(alb, N, P);
  float wat = 1.0 - smoothstep(0.0, max(0.25, foot), F.y);
  if (wat > 0.0) col = mix(col, waterCol(P, foot), wat);
  outColor = vec4(finish(col, P), 1.0);
}`;

// beyond the coarsest level, at full size only: the land there is known in closed form (d = |c| ln|c|)
const SKIRT_VERT = /* glsl */`
precision highp float;
in vec3 position;              // a ring in the plane: (re, im, 0)
uniform mat4 viewMatrix, projectionMatrix;
uniform vec2 uOrg; uniform float uInvS, uSlope;
out vec2 vXZ; out vec2 vC;
void main() {
  float r = length(position.xy), d = r * log(r);
  vXZ = (position.xy - uOrg) * vec2(uInvS, -uInvS); vC = position.xy;
  gl_Position = projectionMatrix * viewMatrix * vec4(vXZ.x, uSlope * d / (1.0 + d / ${D0.toFixed(2)}) * uInvS, vXZ.y, 1.0);
}`;
const SKIRT_FRAG = /* glsl */`
precision highp float; precision highp int; precision highp sampler2D;
${SHADE}
in vec2 vXZ; in vec2 vC;
uniform vec3 uIn;              // centre and half-extent of the coarsest level (m)
uniform vec2 uSkTone;          // colour levels of that level
uniform float uInvS, uSlope;
out vec4 outColor;
void main() {
  vec2 q = abs(vXZ - uIn.xy); if (max(q.x, q.y) < uIn.z * (1.0 - 1.0 / ${GRID}.0)) discard;
  float vR = length(vC), d = vR * log(vR), k = 1.0 + d / ${D0.toFixed(2)}, sl = 4.0 - log2(log(vR));
  float foot = max(length(dFdx(vXZ)), length(dFdy(vXZ)));
  vec3 alb = paletteLW(sl, foot / (LN2 * d * uInvS), uSkTone.x, uSkTone.y);
  float l = dot(alb, vec3(0.2126, 0.7152, 0.0722)); alb = max(mix(vec3(l), alb, uFog.z), 0.0);
  float sl_ = uSlope * (1.0 + log(vR)) / (k * k);                       // slope of the land, radially outward
  vec2 rd = vec2(vC.x, -vC.y) / vR;
  vec3 N = normalize(vec3(-sl_ * rd.x, 1.0, -sl_ * rd.y)), P = vec3(vXZ.x, uSlope * d / k * uInvS, vXZ.y);
  outColor = vec4(finish(landLight(alb, N, P), P), 1.0);
}`;

const SKY_VERT = /* glsl */`
precision highp float;
in vec3 position;
uniform mat4 viewMatrix, projectionMatrix; uniform vec3 cameraPosition;
out vec3 vDir;
void main() { vDir = position; gl_Position = projectionMatrix * viewMatrix * vec4(position * 30000.0 + cameraPosition, 1.0); gl_Position.z = gl_Position.w * 0.99999; }`;
const SKY_FRAG = /* glsl */`
precision highp float;
${SKY_GLSL}
in vec3 vDir; out vec4 outColor;
float hash(vec3 p) { p = fract(p * 0.3183099 + 0.1); p *= 17.0; return fract(p.x * p.y * p.z * (p.x + p.y + p.z)); }
void main() {
  vec3 d = normalize(vDir);
  vec3 c = skyCol(vec3(d.x, abs(d.y), d.z));
  c += vec3(1.0, 0.95, 0.85) * (smoothstep(0.99985, 0.99995, dot(d, uSun)) * 60.0);
  // stars, more of them towards the zenith where the sky is dark
  vec3 g = floor(d * 220.0); float s = hash(g);
  c += vec3(0.8, 0.9, 1.0) * step(0.9972, s) * smoothstep(0.08, 0.5, d.y) * (0.4 + 2.0 * fract(s * 91.7)) * (1.0 - smoothstep(0.0, 0.45, length(fract(d * 220.0) - 0.5) * 2.0 - 0.2));
  outColor = vec4(c, 1.0);
}`;

export class Clip {
  constructor(renderer, fp, world, floatLinear) {
    this.r = renderer; this.fp = fp; this.world = world; this.linear = floatLinear;
    this.rt = null; this.job = null; this.rows = 128; this.N = 0;
    this.alloc();
  }
  alloc() {
    const w = this.world;
    if (this.rt) this.rt.dispose();
    this.N = w.N; this.job = null;
    this.rt = layersRT(w.N, w.N, w.NL + 2, THREE.RGFormat, THREE.FloatType, this.linear);
    this.rt.texture.wrapS = this.rt.texture.wrapT = THREE.ClampToEdgeWrapping;
  }
  // a strip of the height map that is due; with all = true, everything that is due (behind a fade)
  work(dt, fx, fy, all = false) {
    const w = this.world;
    if (!all) { if (dt > 0.03 && this.rows > 16) this.rows >>= 1; else if (dt < 0.019 && this.rows < this.N) this.rows <<= 1; }
    for (let guard = 0; guard < 64; guard++) {
      const planned = w.plan(fx, fy);
      if (this.job && this.job.L.job !== this.job) this.job = null;         // its level was dropped
      if (!this.job) { this.job = planned; if (!this.job) return; this.job.y = 0; }
      const j = this.job, y1 = all ? this.N : Math.min(this.N, j.y + this.rows);
      this.fp.render(this.rt, j.layer, { scale: j.L.h, offX: j.cx, offY: j.cy, halfX: 1, halfY: 1, iter: j.iter, pack: false, strip: [j.y, y1] });
      j.y = y1;
      if (j.y >= this.N) { w.finish(j); this.job = null; }
      if (!all) return;
    }
  }
}

export class Terrain {
  constructor(scene, clip, world, pal, toneU, floatLinear) {
    this.clip = clip; this.world = world;
    const arr = (n, f) => Array.from({ length: n }, f);
    const U = this.U = {
      uLev: { value: clip.rt.texture }, uNL: { value: 0 },
      uLay: { value: new Float32Array(MAXL) }, uHalf: { value: new Float32Array(MAXL).fill(1) }, uMet: { value: new Float32Array(MAXL) }, uSat: { value: new Float32Array(MAXL) }, uFade: { value: new Float32Array(MAXL).fill(1) }, uTLo: { value: new Float32Array(MAXL) }, uTW: { value: new Float32Array(MAXL).fill(1) },
      uCen: { value: arr(MAXL, () => new THREE.Vector2()) },
      uSlope: { value: 0.5 }, uTexN: { value: world.N },
      uShadow: { value: new THREE.Vector4(0, 0, 1, 0) }, uFog: { value: new THREE.Vector3(1e-4, 0, 1.15) }, uWater: { value: new THREE.Vector2() },
      uSun: { value: new THREE.Vector3(0.55, 0.42, 0.72).normalize() },
      uPal: { value: pal.tex }, uTone: toneU,
    };
    const g = new THREE.BufferGeometry(), n = GRID + 1, pos = new Float32Array(n * n * 3), idx = new Uint32Array(GRID * GRID * 6);
    for (let j = 0, k = 0; j < n; j++) for (let i = 0; i < n; i++) { pos[k++] = i / GRID; pos[k++] = j / GRID; pos[k++] = 0; }
    for (let j = 0, k = 0; j < GRID; j++) for (let i = 0; i < GRID; i++) { const a = j * n + i; idx[k++] = a; idx[k++] = a + 1; idx[k++] = a + n; idx[k++] = a + 1; idx[k++] = a + n + 1; idx[k++] = a + n; }
    g.setAttribute('position', new THREE.BufferAttribute(pos, 3)); g.setIndex(new THREE.BufferAttribute(idx, 1));
    const defines = floatLinear ? {} : { MANUAL_BILINEAR: 1 };
    this.rings = arr(MAXL, (_, i) => {
      const m = new THREE.Mesh(g, new THREE.RawShaderMaterial({ glslVersion: THREE.GLSL3, vertexShader: VERT, fragmentShader: FRAG, uniforms: { ...U, uI: { value: i } }, defines, side: THREE.DoubleSide }));
      m.frustumCulled = false; m.visible = false; scene.add(m); return m;
    });
    // the land beyond the last level
    const sg = new THREE.BufferGeometry(), R = [7, 9, 12, 16, 22, 30, 42, 60, 90, 140], SEG = 96, sp = [], si = [];
    for (const r of R) for (let a = 0; a < SEG; a++) sp.push(r * Math.cos(a / SEG * 6.283185), r * Math.sin(a / SEG * 6.283185), 0);
    for (let k = 0; k < R.length - 1; k++) for (let a = 0; a < SEG; a++) { const p = k * SEG + a, q = k * SEG + (a + 1) % SEG; si.push(p, q, p + SEG, q, q + SEG, p + SEG); }
    sg.setAttribute('position', new THREE.Float32BufferAttribute(sp, 3)); sg.setIndex(si);
    this.SU = { uOrg: { value: new THREE.Vector2() }, uInvS: { value: 1 }, uIn: { value: new THREE.Vector3() }, uSkTone: { value: new THREE.Vector2(2, 6) } };
    this.skirt = new THREE.Mesh(sg, new THREE.RawShaderMaterial({ glslVersion: THREE.GLSL3, vertexShader: SKIRT_VERT, fragmentShader: SKIRT_FRAG, uniforms: { ...U, ...this.SU }, side: THREE.DoubleSide }));
    this.skirt.frustumCulled = false; scene.add(this.skirt);
    this.sky = new THREE.Mesh(new THREE.SphereGeometry(1, 48, 24), new THREE.RawShaderMaterial({ glslVersion: THREE.GLSL3, vertexShader: SKY_VERT, fragmentShader: SKY_FRAG, uniforms: { uSun: U.uSun }, side: THREE.BackSide, depthWrite: false }));
    this.sky.frustumCulled = false; this.sky.renderOrder = -10; scene.add(this.sky);
  }
  // per frame: where the levels are, in the bike's metres
  update(dt, time) {
    const w = this.world, U = this.U, S = w.S;
    let n = 0, last = null;
    for (const L of w.levels) {
      if (!L.valid || n >= MAXL) continue;
      L.fade = Math.min(1, L.fade + dt / 0.5);
      U.uLay.value[n] = L.layer; U.uHalf.value[n] = L.h / S; U.uMet.value[n] = L.h / S; U.uSat.value[n] = L.h / D0; U.uFade.value[n] = L.fade; U.uTLo.value[n] = L.tone.lo; U.uTW.value[n] = L.tone.w;
      U.uCen.value[n].set((L.cx - w.ox) / S, -(L.cy - w.oy) / S);
      n++; last = L;
    }
    U.uNL.value = n; U.uSlope.value = w.slope; U.uTexN.value = w.N; U.uLev.value = this.clip.rt.texture; U.uFog.value.y = time;
    this.rings.forEach((m, i) => { m.visible = i < n; });
    // the closed-form land is right only far from the set, so it is used only around the level that holds all of it
    const far = last && last.h >= 7.9;
    this.skirt.visible = !!far;
    if (far) { const c = w.f.c0; this.SU.uOrg.value.set(c[0] + w.ox, c[1] + w.oy); this.SU.uInvS.value = 1 / S; this.SU.uIn.value.set((last.cx - w.ox) / S, -(last.cy - w.oy) / S, last.h / S); this.SU.uSkTone.value.set(last.tone.lo, last.tone.w); }
    this.reach = last ? last.h / S : 1000;
  }
}
