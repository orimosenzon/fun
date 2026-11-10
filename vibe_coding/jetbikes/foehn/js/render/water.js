// Lakes (planar reflection, depth-dependent glacial colour, shore foam, sun glitter, jet wash)
// and rivers (flowing ribbons with white water on the steep reaches).
import * as THREE from '../../lib/three.module.min.js';
import { U, GLSL_COMMON } from './env.js';
import { T } from '../sim/terrain.js';

export const water = { lakes: [], rt: null, cam: null, reflMat: new THREE.Matrix4(), wash: new THREE.Vector4(0, 0, 0, 0), level: 0 };

const LAKE_FRAG = /* glsl */`
precision highp float;
${GLSL_COMMON}
uniform sampler2D uHeight, uAux, uRefl; uniform mat4 uReflMat; uniform vec4 uWorld; uniform float uLevel, uHasRefl; uniform vec4 uWash;
varying vec3 vW;
vec2 grad(vec2 p, float e) { return vec2(vnoise(p + vec2(e, 0.0)) - vnoise(p - vec2(e, 0.0)), vnoise(p + vec2(0.0, e)) - vnoise(p - vec2(0.0, e))); }
void main() {
  vec2 uv = (vW.xz + uWorld.x) / (2.0 * uWorld.x) * (uWorld.z - 1.0) / uWorld.z + 0.5 / uWorld.z;
  float depth = uLevel - texture2D(uHeight, uv).r;
  if (depth < -0.02) discard;
  vec4 aux = texture2D(uAux, uv);
  vec3 Vv = cameraPosition - vW; float dist = length(Vv); vec3 V = Vv / dist;
  // wind ripples at three scales, calmer far away and near the shore
  float calm = smoothstep(0.0, 2.5, depth);
  vec2 w1 = grad(vW.xz / 1.7 + uTime * vec2(0.55, 0.2), 0.6), w2 = grad(vW.xz / 6.5 - uTime * vec2(0.22, 0.35), 0.6), w3 = grad(vW.xz / 31.0 + uTime * vec2(0.1, -0.05), 0.6);
  float gust = 0.55 + 0.9 * vnoise(vW.xz / 240.0 + uTime * 0.02);
  vec2 g = (w1 * 0.55 / (1.0 + dist / 90.0) + w2 * 0.8 / (1.0 + dist / 500.0) + w3 * 0.6) * gust * (0.35 + 0.65 * calm);
  // jet wash: rings running out from under the bike
  vec2 wd = vW.xz - uWash.xy; float wr = length(wd);
  float ring = uWash.w * exp(-wr / (uWash.z + 1.0)) * sin(wr * 1.6 - uTime * 14.0);
  g += normalize(wd + 1e-4) * ring * 1.6;
  vec3 n = normalize(vec3(-g.x * 0.5, 1.0, -g.y * 0.5));
  float fres = 0.02 + 0.98 * pow(1.0 - max(dot(n, V), 0.0), 5.0);
  // reflection
  vec3 R = reflect(-V, n); R.y = abs(R.y);
  vec3 refl = skyLUT(R);
  if (uHasRefl > 0.5) {
    vec4 rc = uReflMat * vec4(vW + vec3(n.x, 0.0, n.z) * min(dist * 0.12, 26.0), 1.0);
    vec2 ruv = rc.xy / rc.w;
    if (rc.w > 0.0) refl = texture2D(uRefl, clamp(ruv, 0.002, 0.998)).rgb;
  }
  // what is below: lit lake bed fading into milky glacial turquoise
  float sunVis = aux.r * cloudShadow(vW);
  vec3 light = uSunCol * max(uSunDir.y, 0.0) * sunVis + uAmbSky;
  float caust = 0.75 + 0.5 * vnoise(vW.xz / 0.9 + g * 6.0 + uTime * 0.4);
  vec3 bed = vec3(0.34, 0.33, 0.27) * (0.6 + 0.6 * vnoise(vW.xz / 2.3)) * caust;
  float path = depth / max(V.y, 0.12);
  vec3 body = bed * exp(-path * vec3(0.42, 0.10, 0.085)) + vec3(0.015, 0.20, 0.21) * (1.0 - exp(-path * 0.20));
  vec3 col = mix(body * light, refl, fres);
  // sun glitter
  float sp = pow(max(dot(reflect(-uSunDir, n), V), 0.0), 520.0);
  col += uSunCol * sp * 28.0 * sunVis;
  // foam at the shore and in the jet wash
  float foam = (1.0 - smoothstep(0.0, 0.45, depth + 0.25 * (vnoise(vW.xz / 0.7 + uTime * 0.3) - 0.5))) * 0.75;
  foam = max(foam, clamp(uWash.w * exp(-wr / (uWash.z * 0.6 + 0.5)) * (0.4 + vnoise(vW.xz / 0.4 + uTime)) * 1.6, 0.0, 1.0));
  col = mix(col, vec3(0.85) * light, foam);
  gl_FragColor = vec4(applyFog(col, vW), 1.0);
}`;

const RIVER_FRAG = /* glsl */`
precision highp float;
${GLSL_COMMON}
uniform sampler2D uAux; uniform vec4 uWorld;
varying vec3 vW; varying vec2 vUv; varying float vSlope;
void main() {
  if (vW.y < uClipY) discard;
  vec2 uv = (vW.xz + uWorld.x) / (2.0 * uWorld.x);
  vec4 aux = texture2D(uAux, uv);
  float speed = 1.2 + 16.0 * vSlope;
  vec2 f = vec2(vUv.x * 4.0, vUv.y * 0.6 - uTime * speed * 0.12);
  float n1 = vnoise(f * 3.0), n2 = vnoise(f * 9.0 + 4.0);
  vec3 V = normalize(cameraPosition - vW);
  vec3 n = normalize(vec3((n1 - 0.5) * 0.5, 1.0, (n2 - 0.5) * 0.5));
  float fres = 0.03 + 0.97 * pow(1.0 - max(dot(n, V), 0.0), 5.0);
  float sunVis = aux.r * cloudShadow(vW);
  vec3 light = uSunCol * max(uSunDir.y, 0.0) * sunVis + uAmbSky;
  vec3 R = reflect(-V, n); R.y = abs(R.y);
  vec3 col = mix(vec3(0.05, 0.25, 0.24) * light, skyLUT(R), fres);
  float white = smoothstep(0.035, 0.16, vSlope + (n1 * n2 - 0.25) * 0.12) + smoothstep(0.62, 0.9, n1 * 0.6 + n2 * 0.5) * 0.35;
  col = mix(col, vec3(0.62) * light, clamp(white, 0.0, 1.0) * 0.8);
  col += uSunCol * pow(max(dot(reflect(-uSunDir, n), V), 0.0), 200.0) * 6.0 * sunVis;
  float a = smoothstep(0.0, 0.16, vUv.x) * smoothstep(1.0, 0.84, vUv.x);
  gl_FragColor = vec4(applyFog(col, vW), a);
}`;

export function buildWater(scene, renderer) {
  const rt = new THREE.WebGLRenderTarget(1024, 576, { type: THREE.HalfFloatType, depthBuffer: true });
  water.rt = rt; water.cam = new THREE.PerspectiveCamera();
  T.meta.lakes.forEach((L, i) => {
    const g = new THREE.PlaneGeometry(L.x1 - L.x0, L.z1 - L.z0, 1, 1); g.rotateX(-Math.PI / 2);
    g.translate((L.x0 + L.x1) / 2, L.level, (L.z0 + L.z1) / 2);
    const mat = new THREE.ShaderMaterial({
      uniforms: { ...U, uRefl: { value: rt.texture }, uReflMat: { value: water.reflMat }, uLevel: { value: L.level }, uHasRefl: { value: i === 0 ? 1 : 0 }, uWash: { value: water.wash } },
      vertexShader: 'varying vec3 vW; void main(){ vW = position; gl_Position = projectionMatrix * viewMatrix * vec4(position, 1.0); }',
      fragmentShader: LAKE_FRAG,
    });
    const m = new THREE.Mesh(g, mat); m.matrixAutoUpdate = false; m.userData.isWater = true;
    scene.add(m); water.lakes.push(m);
  });
  water.level = T.meta.lakes[0].level;
  // rivers
  const rmat = new THREE.ShaderMaterial({
    uniforms: { ...U }, transparent: true, depthWrite: false,
    vertexShader: 'attribute float aSlope; varying vec3 vW; varying vec2 vUv; varying float vSlope; void main(){ vW = position; vUv = uv; vSlope = aSlope; gl_Position = projectionMatrix * viewMatrix * vec4(position, 1.0); }',
    fragmentShader: RIVER_FRAG,
  });
  for (const r of T.meta.rivers) {
    const P = r.pts, n = P.length, pos = new Float32Array(n * 6), uv = new Float32Array(n * 4), sl = new Float32Array(n * 2), idx = [];
    let s = 0;
    for (let i = 0; i < n; i++) {
      const a = P[Math.max(0, i - 1)], b = P[Math.min(n - 1, i + 1)], p = P[i];
      const tx = b[0] - a[0], tz = b[2] - a[2], l = Math.hypot(tx, tz) || 1, w = r.width / 2 + 1.0;
      if (i > 0) s += Math.hypot(p[0] - P[i - 1][0], p[2] - P[i - 1][2]);
      pos.set([p[0] - tz / l * w, p[1] + 0.12, p[2] + tx / l * w, p[0] + tz / l * w, p[1] + 0.12, p[2] - tx / l * w], i * 6);
      uv.set([0, s / r.width, 1, s / r.width], i * 4);
      const slope = Math.max(0, (a[1] - b[1]) / l); sl[i * 2] = sl[i * 2 + 1] = slope;
      if (i < n - 1) idx.push(i * 2, i * 2 + 2, i * 2 + 1, i * 2 + 1, i * 2 + 2, i * 2 + 3);
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(pos, 3)); g.setAttribute('uv', new THREE.BufferAttribute(uv, 2)); g.setAttribute('aSlope', new THREE.BufferAttribute(sl, 1));
    g.setIndex(idx);
    const m = new THREE.Mesh(g, rmat); m.matrixAutoUpdate = false; m.renderOrder = 2; m.material.side = THREE.DoubleSide; m.userData.isWater = true;
    scene.add(m);
  }
  return water;
}

const _n = new THREE.Vector3(0, 1, 0), _p = new THREE.Vector3(), _t = new THREE.Vector3(), _up = new THREE.Vector3(), _q = new THREE.Vector3();
const BIAS = new THREE.Matrix4().set(0.5, 0, 0, 0.5, 0, 0.5, 0, 0.5, 0, 0, 0.5, 0.5, 0, 0, 0, 1);
const clipPlane = new THREE.Plane();
// Render the mirrored scene for the main lake. `hide` objects are skipped (the water itself, fx).
export function renderReflection(renderer, scene, cam, hide, before) {
  const L = water.level;
  const lake = water.lakes[0];
  // skip when the lake cannot be seen or the camera is far above
  if (cam.position.y < L + 0.3 || cam.position.y > L + 1500) { lake.material.uniforms.uHasRefl.value = 0; return false; }
  lake.material.uniforms.uHasRefl.value = 1;
  const v = water.cam;
  _p.copy(cam.position); _p.y = 2 * L - _p.y;
  cam.getWorldDirection(_t); _q.copy(cam.position).add(_t); _q.y = 2 * L - _q.y;
  _up.set(0, 1, 0).applyQuaternion(cam.quaternion); _up.y = -_up.y;
  v.position.copy(_p); v.up.copy(_up); v.lookAt(_q);
  v.fov = cam.fov; v.aspect = cam.aspect; v.near = cam.near; v.far = cam.far; v.updateProjectionMatrix(); v.updateMatrixWorld();
  water.reflMat.copy(BIAS).multiply(v.projectionMatrix).multiply(v.matrixWorldInverse);
  for (const o of hide) { o.userData._v = o.visible; o.visible = false; }
  for (const m of water.lakes) m.visible = false;
  U.uClipY.value = L - 0.4;
  clipPlane.set(_n, -(L - 0.4));
  renderer.clippingPlanes = [clipPlane];
  if (before) before(v);
  const old = renderer.getRenderTarget(), sh = renderer.shadowMap.autoUpdate;
  renderer.shadowMap.autoUpdate = false;
  renderer.setRenderTarget(water.rt); renderer.clear(); renderer.render(scene, v);
  renderer.setRenderTarget(old);
  renderer.shadowMap.autoUpdate = sh;
  renderer.clippingPlanes = [];
  U.uClipY.value = -1e9;
  for (const o of hide) o.visible = o.userData._v;
  for (const m of water.lakes) m.visible = true;
  return true;
}
