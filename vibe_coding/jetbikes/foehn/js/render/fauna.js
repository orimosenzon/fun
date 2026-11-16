// Draws the animals that sim/fauna.js keeps. Every kind comes from assets/models/models.bin
// (tools/bake_models.mjs) frozen in six poses; each pose is one instanced mesh, and every frame
// the animals near the camera are sorted into the pose they are in. No skeletons: a galloping
// herd is four still shapes shown in turn.
import * as THREE from '../../lib/three.module.min.js';
import { U, GLSL_COMMON } from './env.js';
import { POSES } from '../sim/fauna.js';

const VERT = /* glsl */`
attribute vec4 iPos; attribute float iYaw; attribute vec4 color;
uniform vec2 uFade;      // things shrink into the ground between these two distances, so nothing pops up
varying vec3 vW; varying vec3 vN; varying vec3 vCol; varying vec3 vBase;
void main() {
  float c = cos(iYaw), s = sin(iYaw);
  vec3 p = position * iPos.w * (1.0 - smoothstep(uFade.x, uFade.y, distance(iPos.xyz, cameraPosition)));
  vec3 w = iPos.xyz + vec3(p.x * c + p.z * s, p.y, -p.x * s + p.z * c);
  vW = w; vBase = iPos.xyz; vCol = color.rgb * color.rgb;
  vN = vec3(normal.x * c + normal.z * s, normal.y, -normal.x * s + normal.z * c);
  gl_Position = projectionMatrix * viewMatrix * vec4(w, 1.0);
}`;
export const INST_VERT = VERT;
const FRAG = /* glsl */`
precision highp float;
${GLSL_COMMON}
uniform sampler2D uAux; uniform vec4 uWorld; uniform vec4 uTint;      // rgb: a gain on the model's colours. a: how much of the colour is washed out to grey
varying vec3 vW; varying vec3 vN; varying vec3 vCol; varying vec3 vBase;
void main() {
  if (vW.y < uClipY) discard;
  vec4 aux = texture2D(uAux, (vBase.xz + uWorld.x) / (2.0 * uWorld.x));
  vec3 n = normalize(vN);
  float sunVis = aux.r * cloudShadow(vW);
  vec3 light = uSunCol * (max(dot(n, uSunDir), 0.0) * 0.85 + 0.08) * sunVis
    + (mix(uAmbGnd, uAmbSky, n.y * 0.5 + 0.5) * 0.5 + skyLUT(normalize(n + vec3(0.0, 0.5, 0.0))) * 0.8) * aux.g * (0.62 + 0.38 * smoothstep(0.0, 0.9, vW.y - vBase.y));   // darker under the belly
  vec3 base = mix(vCol, vec3(dot(vCol, vec3(0.3, 0.55, 0.15))), uTint.a) * uTint.rgb;
  vec3 col = mix(base, vec3(0.62, 0.66, 0.72), uWx.z * 0.25 * max(n.y, 0.0)) * (1.0 - 0.15 * uWx.y);       // a dusting of snow on the back, a wet coat
  gl_FragColor = vec4(applyFog(col * light, vW), 1.0);
}`;

// the one file of models, read once (the animals and the plants both come out of it)
let modelsFile = null;
export function loadModels() {
  return modelsFile || (modelsFile = (async () => {
    const buf = await (await fetch('assets/models/models.bin')).arrayBuffer();
    const hl = new DataView(buf).getUint32(0, true), head = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 4, hl)));
    return { buf, head, base: 4 + hl };
  })());
}
// one pose of one model as plain geometry
export function poseGeometry(file, M, pose) {
  const { buf, base } = file, q = new Int16Array(buf, base + M.poses[pose], M.n * 3), p = new Float32Array(M.n * 3);
  for (let i = 0; i < p.length; i++) p[i] = q[i] / M.q;
  const idx = new THREE.BufferAttribute(M.wide ? new Uint32Array(buf, base + M.i, M.ni) : new Uint16Array(buf, base + M.i, M.ni), 1);
  const g = new THREE.BufferGeometry(); g.setAttribute('position', new THREE.BufferAttribute(p, 3)); g.setIndex(idx); g.computeVertexNormals();
  g.setAttribute('color', new THREE.BufferAttribute(new Uint8Array(buf, base + M.c, M.n * 4), 4, true));
  return g;
}
// an instanced mesh of that geometry with room for cap things; fill ip (x, y, z, size) and iy (yaw), then call show(n)
export function instanced(scene, g0, mat, cap, depthMat) {
  const g = new THREE.InstancedBufferGeometry();
  g.index = g0.index; g.attributes.position = g0.attributes.position; g.attributes.normal = g0.attributes.normal; g.attributes.color = g0.attributes.color;
  const ip = new Float32Array(cap * 4), iy = new Float32Array(cap);
  g.setAttribute('iPos', new THREE.InstancedBufferAttribute(ip, 4).setUsage(THREE.DynamicDrawUsage)); g.setAttribute('iYaw', new THREE.InstancedBufferAttribute(iy, 1).setUsage(THREE.DynamicDrawUsage));
  g.instanceCount = 0;
  const m = new THREE.Mesh(g, mat); m.frustumCulled = false; m.matrixAutoUpdate = false; m.visible = false;
  if (depthMat) { m.castShadow = true; m.customDepthMaterial = depthMat; }
  scene.add(m);
  const show = (n) => { m.visible = n > 0; g.instanceCount = n; if (n) { const a = g.attributes.iPos, b = g.attributes.iYaw; a.clearUpdateRanges(); a.addUpdateRange(0, n * 4); a.needsUpdate = true; b.clearUpdateRanges(); b.addUpdateRange(0, n); b.needsUpdate = true; } };
  return { m, g, ip, iy, n: 0, cap, show };
}
export const instMaterial = (near, far, tint = [1, 1, 1, 0]) => new THREE.ShaderMaterial({ uniforms: { ...U, uFade: { value: new THREE.Vector2(near, far) }, uTint: { value: new THREE.Vector4(...tint) } }, vertexShader: VERT, fragmentShader: FRAG });
export const instDepth = () => new THREE.ShaderMaterial({ uniforms: { uFade: { value: new THREE.Vector2(1e8, 1e9) } }, vertexShader: VERT, fragmentShader: '#include <packing>\nvoid main(){ gl_FragColor = packDepthToRGBA(gl_FragCoord.z); }' });

export async function buildFauna(scene, fauna) {
  const out = { meshes: [], ready: false, shown: 0, update() {} };
  const file = await loadModels(), head = file.head;
  const mat = instMaterial(1e8, 1e9), depthMat = instDepth();
  const count = {}; for (const a of fauna.list) count[a.kind] = (count[a.kind] || 0) + 1;
  const sets = {};
  for (const [kind, M] of Object.entries(head.models)) {
    if (!count[kind]) continue;
    sets[kind] = POSES.map((pose) => { const s = instanced(scene, poseGeometry(file, M, pose), mat, count[kind], depthMat); out.meshes.push(s.m); return s; });
  }
  // the animals within sight of the camera, each into the mesh of its pose
  out.update = (cam, far = 1500) => {
    const p = cam.position; let shown = 0;
    for (const k in sets) for (const s of sets[k]) s.n = 0;
    for (const a of fauna.list) {
      const set = sets[a.kind]; if (!set) continue;
      const dx = a.x - p.x, dz = a.z - p.z; if (dx * dx + dz * dz > far * far) continue;
      const s = set[a.pose], i = s.n++;
      s.ip[i * 4] = a.x; s.ip[i * 4 + 1] = a.y; s.ip[i * 4 + 2] = a.z; s.ip[i * 4 + 3] = a.size; s.iy[i] = a.yaw; shown++;
    }
    for (const k in sets) for (const s of sets[k]) s.show(s.n);
    out.shown = shown;
  };
  out.ready = true;
  return out;
}
