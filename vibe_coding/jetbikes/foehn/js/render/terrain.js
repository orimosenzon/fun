// Terrain rendering: 16x16 chunks with four levels of detail built from the same height grid
// the physics uses, shaded entirely in the fragment shader (rock strata, scree, meadows, forest
// floor, snow, shores) with sun shadows of the mountains baked into a texture, a real-time
// shadow map near the camera, cloud shadows and aerial perspective.
import * as THREE from '../../lib/three.module.min.js';
import { U, GLSL_COMMON } from './env.js';
import { T } from '../sim/terrain.js';
import { S } from '../sim/scenery.js';

const CH = 64;                 // cells per chunk
const LODS = [1, 2, 4, 8];     // grid step per level
export const terrain = { group: null, chunks: [], aux: null, auxData: null, mat: null };

function buildTextures() {
  const n = T.n, H = T.H, c = T.cell;
  // heights (for water depth, grass, etc.)
  const hh = new Uint16Array(n * n);
  for (let i = 0; i < n * n; i++) hh[i] = THREE.DataUtils.toHalfFloat(H[i]);
  const ht = new THREE.DataTexture(hh, n, n, THREE.RedFormat, THREE.HalfFloatType);
  ht.minFilter = ht.magFilter = THREE.LinearFilter; ht.needsUpdate = true;
  U.uHeight.value = ht;
  // normals
  const nd = new Uint8Array(n * n * 4);
  for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
    const k = j * n + i;
    const hx = (H[j * n + Math.min(n - 1, i + 1)] - H[j * n + Math.max(0, i - 1)]) / (2 * c), hz = (H[Math.min(n - 1, j + 1) * n + i] - H[Math.max(0, j - 1) * n + i]) / (2 * c);
    const l = 1 / Math.hypot(hx, 1, hz);
    nd[k * 4] = (-hx * l * 0.5 + 0.5) * 255; nd[k * 4 + 1] = l * 255; nd[k * 4 + 2] = (-hz * l * 0.5 + 0.5) * 255; nd[k * 4 + 3] = 255;
  }
  const nt = new THREE.DataTexture(nd, n, n, THREE.RGBAFormat);
  nt.minFilter = THREE.LinearMipmapLinearFilter; nt.magFilter = THREE.LinearFilter; nt.generateMipmaps = true; nt.needsUpdate = true;
  U.uNormal.value = nt;
  // aux: R sun visibility (baked later), G ambient occlusion, B forest cover, A unused
  const ad = new Uint8Array(n * n * 4).fill(255);
  // AO from the difference to a blurred copy of the terrain (two box blurs via running sums)
  const blur = (src, r) => {
    const tmp = new Float32Array(n * n), out = new Float32Array(n * n);
    for (let j = 0; j < n; j++) { let s = 0; for (let i = -r; i < n; i++) { if (i + r < n) s += src[j * n + i + r]; if (i - r - 1 >= 0) s -= src[j * n + i - r - 1]; if (i >= 0) tmp[j * n + i] = s / (Math.min(n - 1, i + r) - Math.max(0, i - r) + 1); } }
    for (let i = 0; i < n; i++) { let s = 0; for (let j = -r; j < n; j++) { if (j + r < n) s += tmp[(j + r) * n + i]; if (j - r - 1 >= 0) s -= tmp[(j - r - 1) * n + i]; if (j >= 0) out[j * n + i] = s / (Math.min(n - 1, j + r) - Math.max(0, j - r) + 1); } }
    return out;
  };
  const b1 = blur(H, 3), b2 = blur(H, 14);
  for (let k = 0; k < n * n; k++) {
    const ao = 1 - Math.min(0.55, Math.max(0, (b1[k] - H[k]) * 0.05 + (b2[k] - H[k]) * 0.008));
    ad[k * 4 + 1] = ao * 255;
    ad[k * 4 + 2] = 0;
  }
  // forest cover: splat the trees, blur
  const cover = new Float32Array(n * n);
  for (let t = 0; t < S.count; t++) {
    const i = Math.round((S.trees[t * 5] + T.half) / c), j = Math.round((S.trees[t * 5 + 2] + T.half) / c);
    cover[j * n + i] += 0.55;
  }
  const cb = blur(cover, 1);
  for (let k = 0; k < n * n; k++) ad[k * 4 + 2] = Math.min(1, cb[k]) * 255;
  const at = new THREE.DataTexture(ad, n, n, THREE.RGBAFormat);
  at.minFilter = at.magFilter = THREE.LinearFilter; at.needsUpdate = true;
  U.uAux.value = at; terrain.aux = at; terrain.auxData = ad;
  U.uWorld.value.set(T.half, c, n, 0);
}

// Shadows of the mountains: march from every texel towards the sun over the height field.
export function bakeSunShadow(sun) {
  const n = T.n, H = T.H, c = T.cell, ad = terrain.auxData;
  const hl = Math.hypot(sun[0], sun[2]) || 1e-6, dx = sun[0] / hl, dz = sun[2] / hl, rise = sun[1] / hl;   // per metre of ground distance
  const STEPS = 44;
  for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
    const h0 = H[j * n + i] + 1.5;
    let vis = 1, d = c * 1.2;
    for (let s = 0; s < STEPS; s++) {
      const x = i + dx * d / c, z = j + dz * d / c;
      if (x < 0 || z < 0 || x > n - 1 || z > n - 1) break;
      const hr = h0 + rise * d, ht = H[(z | 0) * n + (x | 0)];
      const m = (hr - ht) / (d * 0.045);          // soft edge: 0 at the ray .. 1 when well clear
      if (m < vis) { vis = m; if (vis <= 0) { vis = 0; break; } }
      if (hr > 3600) break;
      d *= 1.16;
    }
    ad[(j * n + i) * 4] = Math.max(0, Math.min(1, vis)) * 255;
  }
  // 3x3 blur of the visibility to hide the grid
  const tmp = new Uint8Array(n * n);
  for (let j = 1; j < n - 1; j++) for (let i = 1; i < n - 1; i++) { let s = 0; for (let b = -1; b <= 1; b++) for (let a = -1; a <= 1; a++) s += ad[((j + b) * n + i + a) * 4]; tmp[j * n + i] = s / 9; }
  for (let j = 1; j < n - 1; j++) for (let i = 1; i < n - 1; i++) ad[(j * n + i) * 4] = tmp[j * n + i];
  terrain.aux.needsUpdate = true;
}

const VERT = /* glsl */`
varying vec3 vW;
void main() { vec4 w = modelMatrix * vec4(position, 1.0); vW = w.xyz; gl_Position = projectionMatrix * viewMatrix * w; }
`;
const FRAG = /* glsl */`
precision highp float;
#include <packing>
${GLSL_COMMON}
uniform sampler2D uHeight, uNormal, uAux, uShadowMap; uniform mat4 uShadowMat; uniform float uShadowOn; uniform vec4 uWorld;
uniform vec4 uLake; uniform vec4 uLakeBox; uniform float uFar;
varying vec3 vW;
float shadowTap(vec2 uv, float z) { return step(z, unpackRGBAToDepth(texture2D(uShadowMap, uv))); }
float realShadow(vec3 wp) {
  if (uShadowOn < 0.5) return 1.0;
  vec4 sc = uShadowMat * vec4(wp, 1.0); sc.xyz /= sc.w;
  if (sc.x < 0.0 || sc.x > 1.0 || sc.y < 0.0 || sc.y > 1.0 || sc.z > 1.0) return 1.0;
  float z = sc.z - 0.0012; vec2 px = vec2(1.0 / 2048.0);
  float s = shadowTap(sc.xy, z) * 2.0 + shadowTap(sc.xy + px * vec2(1.3, 0.4), z) + shadowTap(sc.xy + px * vec2(-0.4, 1.3), z) + shadowTap(sc.xy + px * vec2(-1.3, -0.4), z) + shadowTap(sc.xy + px * vec2(0.4, -1.3), z);
  vec2 e = min(sc.xy, 1.0 - sc.xy);
  return mix(1.0, s / 6.0, smoothstep(0.0, 0.08, min(e.x, e.y)));
}
void main() {
  if (vW.y < uClipY) discard;
  vec2 uv = (vW.xz + uWorld.x) / (2.0 * uWorld.x) * (uWorld.z - 1.0) / uWorld.z + 0.5 / uWorld.z;
  vec3 n = texture2D(uNormal, uv, -1.0).xyz; n.xz = n.xz * 2.0 - 1.0; n = normalize(n);
  vec4 aux = texture2D(uAux, uv);
  float h = vW.y, dist = length(vW - cameraPosition);
  float nearK = 1.0 - smoothstep(60.0, 420.0, dist), midK = 1.0 - smoothstep(500.0, 3500.0, dist);
  // noise at several scales
  float nA = fbm3(vW.xz / 310.0), nB = fbm3(vW.xz / 41.0), nC = vnoise(vW.xz / 5.3), nD = vnoise(vW.xz / 0.9), nE = fbm3(vW.xz / 900.0 + 40.0);
  // large-scale slope decides the material (the fine normal only shades it)
  vec3 nL = texture2D(uNormal, uv, 1.5).xyz; nL.xz = nL.xz * 2.0 - 1.0; nL = normalize(nL);
  float slopeL = 1.0 - nL.y;
  float rockW = smoothstep(0.17, 0.26, slopeL * 0.8 + (1.0 - n.y) * 0.2 + (nB - 0.5) * 0.07 + (nC - 0.5) * 0.04 * nearK);
  // bedding planes: strata dip gently across the range and catch the light as ledges
  float strataH = h + vW.x * 0.11 + vW.z * 0.05 + (nA - 0.5) * 120.0;
  float st1 = vnoise(vec2(strataH * 0.55, nB * 5.0 + nA * 3.0)), st2 = vnoise(vec2(strataH * 0.09, 7.0 + nA * 2.0));
  float ledge = (vnoise(vec2((strataH + 0.8) * 0.55, nB * 5.0 + nA * 3.0)) - st1);
  // on steep faces project the detail from the side, or it smears down the wall
  float steep = smoothstep(0.22, 0.45, slopeL);
  float wX = abs(nL.x) + 1e-3, wZ = abs(nL.z) + 1e-3;
  float sC = (vnoise(vW.zy / 4.1) * wX + vnoise(vW.xy / 4.1) * wZ) / (wX + wZ);
  float sD = (fbm3(vW.zy / 17.0 + 3.0) * wX + fbm3(vW.xy / 17.0 + 3.0) * wZ) / (wX + wZ);
  float sE = (vnoise(vW.zy / 0.8) * wX + vnoise(vW.xy / 0.8) * wZ) / (wX + wZ);
  nC = mix(nC, sC, steep);
  // detail normals at three scales
  vec2 dn = vec2(vnoise(vW.xz / 2.3 + 7.0) - vnoise(vW.xz / 2.3 - 7.0), vnoise(vW.zx / 2.3 + 3.0) - vnoise(vW.zx / 2.3 - 3.0));
  vec2 dm = vec2(fbm3(vW.xz / 19.0 + 5.0) - fbm3(vW.xz / 19.0 - 5.0), fbm3(vW.zx / 19.0 + 9.0) - fbm3(vW.zx / 19.0 - 9.0));
  vec2 df = vec2(vnoise(vW.xz / 0.37 + 3.0) - vnoise(vW.xz / 0.37 - 3.0), vnoise(vW.zx / 0.37 + 5.0) - vnoise(vW.zx / 0.37 - 5.0));
  vec3 tang = normalize(vec3(n.x, 0.0, n.z) + 1e-4);
  n = normalize(n + vec3(dm.x, 0.0, dm.y) * (1.5 * rockW + 0.10) * midK + vec3(dn.x, 0.0, dn.y) * (1.0 * rockW + 0.22) * nearK
      + vec3(df.x, 0.0, df.y) * (0.5 * rockW + 0.25) * (1.0 - smoothstep(20.0, 110.0, dist))
      + (tang * 0.9 - vec3(0.0, 0.5, 0.0)) * ledge * 1.3 * rockW * midK);
  float slope = 1.0 - n.y;
  // --- materials ---
  float alp = smoothstep(1500.0, 2150.0, h + (nA - 0.5) * 300.0);
  vec3 grass = mix(vec3(0.075, 0.135, 0.030), vec3(0.150, 0.185, 0.045), nA);
  grass = mix(grass, vec3(0.205, 0.195, 0.060), smoothstep(0.45, 0.75, nE) * 0.65);         // drier pasture
  grass = mix(grass, vec3(0.055, 0.105, 0.030), smoothstep(0.55, 0.8, 1.0 - nE) * 0.5);     // lush hollows
  grass = mix(grass, vec3(0.235, 0.225, 0.085), alp * 0.85);                                // alpine turf
  grass *= 0.78 + 0.44 * nB; grass *= 0.86 + 0.28 * nC * nearK; grass *= 0.9 + 0.2 * nD * (1.0 - smoothstep(15.0, 80.0, dist));
  // flowers in the meadows, seen close
  float fl = smoothstep(0.80, 0.95, vnoise(vW.xz / 0.22)) * smoothstep(0.5, 0.8, nD) * (1.0 - smoothstep(25.0, 140.0, dist)) * smoothstep(0.52, 0.66, nB) * (1.0 - alp);
  grass = mix(grass, mix(vec3(0.75, 0.62, 0.10), vec3(0.72, 0.68, 0.78), step(0.5, nC)), fl * 0.55);
  vec3 forest = vec3(0.028, 0.058, 0.024) * (0.8 + 0.5 * nB);
  vec3 col = mix(grass, forest, smoothstep(0.1, 0.7, aux.b) * 0.92);
  // rock: banded limestone, weathered streaks, lichen
  vec3 rock = mix(vec3(0.27, 0.25, 0.225), vec3(0.47, 0.44, 0.40), st1 * 0.35 + st2 * 0.65);
  rock = mix(rock, vec3(0.42, 0.33, 0.24), smoothstep(0.5, 0.85, nA) * 0.45);               // iron-stained bands
  float streak = (vnoise(vec2(vW.z, h * 0.2) / 3.0) * wX + vnoise(vec2(vW.x, h * 0.2) / 3.0) * wZ) / (wX + wZ);
  rock *= mix(1.0, 0.62 + 0.7 * streak, steep);                                             // water streaks down the walls
  rock *= 0.72 + 0.56 * mix(fbm3(vW.xz / 6.0), sD, steep);
  rock *= 0.88 + 0.24 * mix(nD, sE, steep) * nearK;
  float cav = mix(0.5, sD * 0.6 + sC * 0.4, rockW);                                         // crevices take less sky light
  vec3 scree = vec3(0.37, 0.355, 0.33) * (0.72 + 0.4 * nC) * (0.85 + 0.3 * nD);
  col = mix(col, scree, smoothstep(0.09, 0.19, slopeL + (nB - 0.5) * 0.1) * (0.3 + 0.7 * alp) * (1.0 - 0.6 * smoothstep(0.1, 0.6, aux.b)));
  col = mix(col, rock, rockW);
  // shore and lake bed
  float inLake = step(uLakeBox.x, vW.x) * step(vW.x, uLakeBox.z) * step(uLakeBox.y, vW.z) * step(vW.z, uLakeBox.w);
  float shore = inLake * (1.0 - smoothstep(0.3, 2.6, h - uLake.x)) * (1.0 - rockW);
  col = mix(col, vec3(0.42, 0.40, 0.34) * (0.7 + 0.5 * nC), shore);
  // snow: above the snow line, less on steep and sun-facing rock, drifts in gullies
  float snowLine = 2330.0 + (nA - 0.5) * 420.0 - (1.0 - aux.g) * 500.0 - n.z * 60.0;
  float snow = smoothstep(snowLine - 60.0, snowLine + 90.0, h) * (1.0 - smoothstep(0.38, 0.62, slope + (nB - 0.5) * 0.2));
  snow = max(snow, smoothstep(snowLine - 300.0, snowLine - 80.0, h) * smoothstep(0.5, 0.75, nA * 0.6 + nB * 0.4) * (1.0 - rockW) * 0.85);
  col = mix(col, vec3(0.86, 0.89, 0.93), snow);
  // --- light ---
  float sunVis = aux.r * realShadow(vW) * cloudShadow(vW);
  float ndl = max(dot(n, uSunDir), 0.0);
  float ao = aux.g * (1.0 - 0.35 * smoothstep(0.2, 0.8, aux.b) * (1.0 - rockW));
  // sky light with direction: slopes facing the bright part of the sky get more of it
  vec3 amb = (mix(uAmbGnd, uAmbSky, n.y * 0.5 + 0.5) * 0.45 + skyLUT(normalize(n + vec3(0.0, 0.55, 0.0))) * 0.75) * ao * (0.55 + 0.9 * cav) * 1.25;
  amb += uSunCol * 0.085 * (1.0 - aux.r) * ao * (0.6 + 0.4 * n.y);                                             // light bounced off the sunlit side
  vec3 light = uSunCol * ndl * sunVis + amb;
  // snow glows a little in shade and sparkles in the sun
  light += snow * uAmbSky * 0.25;
  vec3 V = normalize(cameraPosition - vW), Hh = normalize(V + uSunDir);
  float spec = pow(max(dot(n, Hh), 0.0), 60.0) * snow * 0.5 * sunVis;
  col = col * light + uSunCol * spec;
  gl_FragColor = vec4(applyFog(col, vW), 1.0);
}
`;

function chunkGeometry(ci, cj, step) {
  const n = T.n, H = T.H, c = T.cell, half = T.half;
  const m = CH / step, w = m + 1;
  // grid + skirt ring
  const pos = new Float32Array((w * w + 4 * w) * 3);
  let ymin = 1e9, ymax = -1e9;
  for (let j = 0; j < w; j++) for (let i = 0; i < w; i++) {
    const gi = ci * CH + i * step, gj = cj * CH + j * step, y = H[gj * n + gi], k = (j * w + i) * 3;
    pos[k] = -half + gi * c; pos[k + 1] = y; pos[k + 2] = -half + gj * c;
    ymin = Math.min(ymin, y); ymax = Math.max(ymax, y);
  }
  const drop = step * 5 + 2;
  const edge = [];
  for (let i = 0; i < w; i++) edge.push(i);                       // top
  for (let i = 0; i < w; i++) edge.push((w - 1) * w + i);         // bottom
  for (let j = 0; j < w; j++) edge.push(j * w);                   // left
  for (let j = 0; j < w; j++) edge.push(j * w + w - 1);           // right
  edge.forEach((src, e) => { const k = (w * w + e) * 3; pos[k] = pos[src * 3]; pos[k + 1] = pos[src * 3 + 1] - drop; pos[k + 2] = pos[src * 3 + 2]; });
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  g.setIndex(indexFor(m));
  g.boundingBox = new THREE.Box3(new THREE.Vector3(pos[0], ymin - drop, pos[2]), new THREE.Vector3(pos[0] + CH * c, ymax, pos[2] + CH * c));
  g.boundingSphere = new THREE.Sphere(); g.boundingBox.getBoundingSphere(g.boundingSphere);
  return g;
}
const idxCache = {};
function indexFor(m) {
  if (idxCache[m]) return idxCache[m];
  const w = m + 1, idx = [];
  // same diagonal as the physics: (0,0)-(0,1)-(1,0) and (1,0)-(0,1)-(1,1)
  for (let j = 0; j < m; j++) for (let i = 0; i < m; i++) { const a = j * w + i, b = a + 1, c = a + w, d = c + 1; idx.push(a, c, b, b, c, d); }
  const s0 = w * w;
  const strip = (base, grid, flip) => { for (let i = 0; i < m; i++) { const a = grid(i), b = grid(i + 1), c = base + i, d = base + i + 1; if (flip) idx.push(a, b, c, b, d, c); else idx.push(a, c, b, b, c, d); } };
  strip(s0, (i) => i, true); strip(s0 + w, (i) => (w - 1) * w + i, false);
  strip(s0 + 2 * w, (j) => j * w, false); strip(s0 + 3 * w, (j) => j * w + w - 1, true);
  return (idxCache[m] = new THREE.BufferAttribute(w * w + 4 * w > 65535 ? new Uint32Array(idx) : new Uint16Array(idx), 1));
}

export async function buildTerrain(scene, yieldFn) {
  buildTextures();
  const lake = T.meta.lakes[0];
  const mat = new THREE.ShaderMaterial({
    vertexShader: VERT, fragmentShader: FRAG, side: THREE.DoubleSide,
    uniforms: { ...U, uLake: { value: new THREE.Vector4(lake.level, 0, 0, 0) }, uLakeBox: { value: new THREE.Vector4(lake.x0, lake.z0, lake.x1 + 300, lake.z1) }, uFar: { value: 0 } },
  });
  terrain.mat = mat;
  const group = new THREE.Group();
  const nc = (T.n - 1) / CH;
  for (let cj = 0; cj < nc; cj++) {
    for (let ci = 0; ci < nc; ci++) {
      const lods = LODS.map((s) => { const m = new THREE.Mesh(chunkGeometry(ci, cj, s), mat); m.visible = false; m.matrixAutoUpdate = false; group.add(m); return m; });
      const bb = lods[0].geometry.boundingBox;
      terrain.chunks.push({ lods, cx: (bb.min.x + bb.max.x) / 2, cz: (bb.min.z + bb.max.z) / 2, y0: bb.min.y, y1: bb.max.y, cur: -1 });
    }
    if (yieldFn) await yieldFn(cj / nc);
  }
  scene.add(group); terrain.group = group;
  buildBackdrop(scene, mat);
  return terrain;
}

// pick a level of detail per chunk
export function updateTerrain(cam, bias = 1) {
  const p = cam.position;
  for (const ch of terrain.chunks) {
    const dx = Math.max(0, Math.abs(p.x - ch.cx) - CH * T.cell / 2), dz = Math.max(0, Math.abs(p.z - ch.cz) - CH * T.cell / 2);
    const dy = Math.max(0, p.y - ch.y1, ch.y0 - p.y);
    const d = Math.hypot(dx, dz, dy) * bias;
    const lod = d < 520 ? 0 : d < 1500 ? 1 : d < 3600 ? 2 : 3;
    if (lod !== ch.cur) { if (ch.cur >= 0) ch.lods[ch.cur].visible = false; ch.lods[lod].visible = true; ch.cur = lod; }
  }
}

// far mountains beyond the playable square (same analytic terrain, coarse grid, hole in the middle)
async function buildBackdrop(scene, mat) {
  const meta = T.meta;
  const buf = await (await fetch('assets/backdrop.bin')).arrayBuffer();
  const u = new Uint16Array(buf), n = meta.bn, c = meta.bcell, half = meta.bhalf;
  const pos = new Float32Array(n * n * 3), idx = [];
  for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) { const k = (j * n + i) * 3; pos[k] = -half + i * c; pos[k + 1] = u[j * n + i] * meta.hscale - 6; pos[k + 2] = -half + j * c; }
  const inner = T.half - c;      // overlap one cell under the main terrain
  for (let j = 0; j < n - 1; j++) for (let i = 0; i < n - 1; i++) {
    const x0 = -half + i * c, z0 = -half + j * c;
    if (x0 > -inner && x0 + c < inner && z0 > -inner && z0 + c < inner) continue;
    const a = j * n + i, b = a + 1, d = a + n, e = d + 1; idx.push(a, d, b, b, d, e);
  }
  const nor = new Float32Array(n * n * 3);
  for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
    const hx = (u[j * n + Math.min(n - 1, i + 1)] - u[j * n + Math.max(0, i - 1)]) * meta.hscale / (2 * c), hz = (u[Math.min(n - 1, j + 1) * n + i] - u[Math.max(0, j - 1) * n + i]) * meta.hscale / (2 * c);
    const l = 1 / Math.hypot(hx, 1, hz), k = (j * n + i) * 3; nor[k] = -hx * l; nor[k + 1] = l; nor[k + 2] = -hz * l;
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(pos, 3)); g.setAttribute('normal', new THREE.BufferAttribute(nor, 3)); g.setIndex(idx);
  g.computeBoundingSphere();
  // the backdrop has no normal/aux texels of its own: a simpler material derives the normal from derivatives
  const bm = new THREE.ShaderMaterial({
    uniforms: { ...U },
    vertexShader: 'varying vec3 vW; varying vec3 vN; void main(){ vW = position; vN = normal; gl_Position = projectionMatrix * viewMatrix * vec4(position, 1.0); }',
    fragmentShader: /* glsl */`
      precision highp float;
      ${GLSL_COMMON}
      varying vec3 vW; varying vec3 vN;
      void main() {
        if (vW.y < uClipY) discard;
        float nA = fbm3(vW.xz / 310.0), nB = fbm3(vW.xz / 41.0);
        vec2 dm = vec2(fbm3(vW.xz / 60.0 + 5.0) - fbm3(vW.xz / 60.0 - 5.0), fbm3(vW.zx / 60.0 + 9.0) - fbm3(vW.zx / 60.0 - 9.0));
        vec3 n = normalize(vN + vec3(dm.x, 0.0, dm.y) * 1.2);
        float slope = 1.0 - normalize(vN).y + (nB - 0.5) * 0.1;
        vec3 col = mix(vec3(0.07, 0.13, 0.04), vec3(0.30, 0.29, 0.10), smoothstep(1500.0, 2150.0, vW.y));
        col = mix(col, vec3(0.37, 0.355, 0.33) * (0.8 + 0.4 * nA), smoothstep(0.15, 0.3, slope));
        float sl = 2330.0 + (nA - 0.5) * 420.0;
        col = mix(col, vec3(0.86, 0.89, 0.93), smoothstep(sl - 60.0, sl + 90.0, vW.y) * (1.0 - smoothstep(0.4, 0.62, slope + (nB - 0.5) * 0.2)));
        vec3 light = uSunCol * max(dot(n, uSunDir), 0.0) * cloudShadow(vW) + mix(uAmbGnd, uAmbSky, n.y * 0.5 + 0.5) * 0.45 + skyLUT(normalize(n + vec3(0.0, 0.55, 0.0))) * 0.75;
        gl_FragColor = vec4(applyFog(col * light, vW), 1.0);
      }`,
  });
  const mesh = new THREE.Mesh(g, bm); mesh.matrixAutoUpdate = false;
  scene.add(mesh); terrain.backdrop = mesh;
}
