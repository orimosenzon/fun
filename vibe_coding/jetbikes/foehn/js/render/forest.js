// Spruce forest: ~150 000 instanced trees in the same 768 m chunks as the terrain.
// Close chunks draw a layered tree, far ones a single cone; beyond that the terrain's own
// forest shading takes over. Trees take the baked mountain shadow at their foot and sway.
import * as THREE from '../../lib/three.module.min.js';
import { U, GLSL_COMMON } from './env.js';
import { T } from '../sim/terrain.js';
import { S } from '../sim/scenery.js';

export const forest = { chunks: [] };

function treeGeometry(detail) {
  const pos = [], nor = [], col = [], idx = [];
  const cone = (y0, y1, r0, sides, shade) => {
    const base = pos.length / 3;
    for (let i = 0; i < sides; i++) {
      const a = i / sides * 6.283, c = Math.cos(a), s = Math.sin(a);
      pos.push(c * r0, y0, s * r0); nor.push(c * 0.82, 0.57, s * 0.82); col.push(shade * 0.72);
    }
    pos.push(0, y1, 0); nor.push(0, 1, 0); col.push(shade * 1.15);
    for (let i = 0; i < sides; i++) idx.push(base + i, base + sides, base + (i + 1) % sides);
  };
  if (detail) {
    cone(0.14, 0.52, 0.20, 7, 0.72); cone(0.34, 0.74, 0.155, 7, 0.86); cone(0.56, 0.90, 0.11, 6, 0.96); cone(0.74, 1.0, 0.065, 5, 1.05);
    // trunk
    const base = pos.length / 3;
    for (let i = 0; i < 5; i++) { const a = i / 5 * 6.283, c = Math.cos(a), s = Math.sin(a); pos.push(c * 0.022, 0, s * 0.022, c * 0.018, 0.2, s * 0.018); nor.push(c, 0, s, c, 0, s); col.push(-1, -1); }
    for (let i = 0; i < 5; i++) { const a = base + i * 2, b = base + ((i + 1) % 5) * 2; idx.push(a, a + 1, b, b, a + 1, b + 1); }
  } else { cone(0.12, 1.0, 0.17, 5, 0.86); }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  g.setAttribute('normal', new THREE.Float32BufferAttribute(nor, 3));
  g.setAttribute('aShade', new THREE.Float32BufferAttribute(col, 1));
  g.setIndex(idx);
  return g;
}

const VERT = /* glsl */`
attribute vec4 iPos; attribute float iRnd; attribute float aShade;
uniform float uTime; uniform vec3 uSunDir;
varying vec3 vW; varying vec3 vN; varying float vShade; varying float vRnd; varying vec3 vBase;
void main() {
  float a = iRnd * 6.283, c = cos(a), s = sin(a);
  float hgt = iPos.w, wid = hgt * (0.85 + 0.5 * fract(iRnd * 7.3));
  vec3 p = vec3(position.x * c - position.z * s, position.y, position.x * s + position.z * c);
  p.xz *= wid; p.y *= hgt;
  // wind: the top sways
  float sway = position.y * position.y * hgt * 0.012;
  p.x += sin(uTime * 1.3 + iPos.x * 0.05 + iRnd * 6.0) * sway; p.z += cos(uTime * 1.1 + iPos.z * 0.05) * sway * 0.7;
  vec3 w = iPos.xyz + p;
  vW = w; vBase = iPos.xyz; vShade = aShade; vRnd = iRnd;
  vN = normalize(vec3(normal.x * c - normal.z * s, normal.y, normal.x * s + normal.z * c));
  gl_Position = projectionMatrix * viewMatrix * vec4(w, 1.0);
}`;
const FRAG = /* glsl */`
precision highp float;
${GLSL_COMMON}
uniform sampler2D uAux; uniform vec4 uWorld;
varying vec3 vW; varying vec3 vN; varying float vShade; varying float vRnd; varying vec3 vBase;
void main() {
  if (vW.y < uClipY) discard;
  vec2 uv = (vBase.xz + uWorld.x) / (2.0 * uWorld.x);
  vec4 aux = texture2D(uAux, uv);
  vec3 col;
  if (vShade < 0.0) col = vec3(0.16, 0.11, 0.07);
  else {
    col = mix(vec3(0.040, 0.105, 0.045), vec3(0.10, 0.17, 0.05), fract(vRnd * 3.7)) * vShade;
    col = mix(col, vec3(0.16, 0.15, 0.04), step(0.93, fract(vRnd * 11.3)) * 0.7);        // the odd larch
  }
  float sunVis = aux.r * cloudShadow(vW);
  float ndl = dot(vN, uSunDir);
  // needles let light through: wrap the diffuse term, add a rim towards the sun
  float diff = clamp(ndl * 0.6 + 0.4, 0.0, 1.0);
  vec3 V = normalize(cameraPosition - vW);
  float back = pow(max(dot(-V, uSunDir), 0.0), 4.0) * 0.6;
  vec3 light = uSunCol * (diff + back) * sunVis + (mix(uAmbGnd, uAmbSky, vN.y * 0.5 + 0.5) * 0.5 + skyLUT(normalize(vN + vec3(0.0, 0.5, 0.0))) * 0.8) * (0.35 + 0.65 * abs(vShade)) * aux.g;
  gl_FragColor = vec4(applyFog(col * light, vW), 1.0);
}`;

export function buildForest(scene) {
  const CHM = 64 * T.cell, nc = Math.round(2 * T.half / CHM);
  const lists = Array.from({ length: nc * nc }, () => []);
  for (let k = 0; k < S.count; k++) {
    const ci = Math.min(nc - 1, Math.floor((S.trees[k * 5] + T.half) / CHM)), cj = Math.min(nc - 1, Math.floor((S.trees[k * 5 + 2] + T.half) / CHM));
    lists[cj * nc + ci].push(k);
  }
  const gNear = treeGeometry(true), gFar = treeGeometry(false);
  const mat = new THREE.ShaderMaterial({ uniforms: { ...U }, vertexShader: VERT, fragmentShader: FRAG, side: THREE.DoubleSide });
  const depthMat = new THREE.ShaderMaterial({
    uniforms: { uTime: U.uTime, uSunDir: U.uSunDir },
    vertexShader: VERT.replace('void main() {', 'varying float vDepthZ; void main() {'),
    fragmentShader: '#include <packing>\nvoid main(){ gl_FragColor = packDepthToRGBA(gl_FragCoord.z); }', side: THREE.DoubleSide,
  });
  lists.forEach((list, c) => {
    if (!list.length) return;
    const ip = new Float32Array(list.length * 4), ir = new Float32Array(list.length);
    let y0 = 1e9, y1 = -1e9;
    list.forEach((k, i) => { ip[i * 4] = S.trees[k * 5]; ip[i * 4 + 1] = S.trees[k * 5 + 1]; ip[i * 4 + 2] = S.trees[k * 5 + 2]; ip[i * 4 + 3] = S.trees[k * 5 + 3]; ir[i] = S.trees[k * 5 + 4]; y0 = Math.min(y0, ip[i * 4 + 1]); y1 = Math.max(y1, ip[i * 4 + 1] + 30); });
    const cx = -T.half + ((c % nc) + 0.5) * CHM, cz = -T.half + (Math.floor(c / nc) + 0.5) * CHM;
    const mk = (base) => {
      const g = new THREE.InstancedBufferGeometry();
      g.index = base.index; g.attributes.position = base.attributes.position; g.attributes.normal = base.attributes.normal; g.attributes.aShade = base.attributes.aShade;
      g.setAttribute('iPos', new THREE.InstancedBufferAttribute(ip, 4)); g.setAttribute('iRnd', new THREE.InstancedBufferAttribute(ir, 1));
      g.instanceCount = list.length;
      g.boundingSphere = new THREE.Sphere(new THREE.Vector3(cx, (y0 + y1) / 2, cz), Math.hypot(CHM * 0.71, (y1 - y0) / 2));
      const m = new THREE.Mesh(g, mat); m.matrixAutoUpdate = false; m.visible = false;
      scene.add(m); return m;
    };
    const near = mk(gNear), far = mk(gFar);
    near.castShadow = true; near.customDepthMaterial = depthMat;
    forest.chunks.push({ near, far, cx, cz, cy: (y0 + y1) / 2, r: CHM * 0.5 });
  });
  return forest;
}

export function updateForest(cam, q = 1) {
  const p = cam.position;
  for (const ch of forest.chunks) {
    const d = Math.hypot(Math.max(0, Math.abs(p.x - ch.cx) - ch.r), Math.max(0, Math.abs(p.z - ch.cz) - ch.r), (p.y - ch.cy) * 0.6);
    ch.near.visible = d < 420 * q;
    ch.far.visible = !ch.near.visible && d < 3400 * q;
  }
}
