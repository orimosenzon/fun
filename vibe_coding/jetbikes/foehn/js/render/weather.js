// Rain and snow: a box of drops that travels with the camera. Each drop keeps its place in the
// world (the box wraps around), falls, and is carried by the wind that the physics feels, so a
// gust is something you can see coming across the valley. Rain is drawn as streaks whose length
// and direction are the drop's motion relative to the camera during the exposure: standing still
// it falls, at 300 km/h it comes at you almost level.
import * as THREE from '../../lib/three.module.min.js';
import { U } from './env.js';

const WRAP = /* glsl */`
uniform vec3 uOff, uBox;
vec3 wrapped(vec3 seed) { return mod(seed * uBox + uOff - cameraPosition, uBox) - 0.5 * uBox; }   // relative to the camera
float edgeFade(vec3 p) { float d = length(p); return (1.0 - smoothstep(0.36, 0.5, d / uBox.x)) * smoothstep(0.6, 3.0, d); }`;

export function buildWeather(scene) {
  const rnd = (() => { let s = 99991; return () => ((s = (s * 1664525 + 1013904223) >>> 0) / 4294967296); })();
  const mkUni = (box) => ({ uOff: { value: new THREE.Vector3() }, uBox: { value: new THREE.Vector3(...box) }, uStreak: { value: new THREE.Vector3() }, uAmt: { value: 0 }, uLight: { value: new THREE.Color(1, 1, 1) }, uTime: U.uTime, uPx: { value: 900 } });

  // ----- rain: line segments -----
  const NR = 9000, rp = new Float32Array(NR * 6), re = new Float32Array(NR * 2);
  for (let i = 0; i < NR; i++) { const x = rnd(), y = rnd(), z = rnd(); rp.set([x, y, z, x, y, z], i * 6); re[i * 2 + 1] = 1; }
  const rg = new THREE.BufferGeometry();
  rg.setAttribute('position', new THREE.BufferAttribute(rp, 3)); rg.setAttribute('aEnd', new THREE.BufferAttribute(re, 1));
  const rainU = mkUni([64, 40, 64]);
  const rain = new THREE.LineSegments(rg, new THREE.ShaderMaterial({
    uniforms: rainU, transparent: true, depthWrite: false,
    vertexShader: /* glsl */`
      attribute float aEnd; uniform vec3 uStreak; uniform float uAmt; varying float vA;
      ${WRAP}
      void main() {
        vec3 p = wrapped(position);
        vA = edgeFade(p) * (1.0 - 0.75 * aEnd) * step(fract(position.x * 37.0 + position.z * 91.0), uAmt);
        gl_Position = projectionMatrix * viewMatrix * vec4(p + cameraPosition - uStreak * aEnd, 1.0);
      }`,
    fragmentShader: /* glsl */`
      precision highp float; uniform vec3 uLight; varying float vA;
      void main() { gl_FragColor = vec4(uLight, vA * 0.32); }`,
  }));

  // ----- snow: soft points that sway as they fall -----
  const NS = 20000, sp = new Float32Array(NS * 3);
  for (let i = 0; i < NS * 3; i++) sp[i] = rnd();
  const sg = new THREE.BufferGeometry(); sg.setAttribute('position', new THREE.BufferAttribute(sp, 3));
  const snowU = mkUni([56, 40, 56]);
  const snow = new THREE.Points(sg, new THREE.ShaderMaterial({
    uniforms: snowU, transparent: true, depthWrite: false,
    vertexShader: /* glsl */`
      uniform float uAmt, uTime, uPx; varying float vA;
      ${WRAP}
      void main() {
        vec3 p = wrapped(position);
        float ph = position.x * 40.0 + position.y * 17.0;
        p.xz += 0.35 * vec2(sin(uTime * 1.3 + ph), cos(uTime * 1.1 + ph * 1.7));
        vA = edgeFade(p) * step(fract(position.y * 53.0 + position.z * 29.0), uAmt);
        vec4 mv = viewMatrix * vec4(p + cameraPosition, 1.0);
        gl_PointSize = clamp(uPx * (0.035 + 0.03 * fract(ph)) / max(-mv.z, 0.3), 1.5, 14.0);
        gl_Position = projectionMatrix * mv;
      }`,
    fragmentShader: /* glsl */`
      precision highp float; uniform vec3 uLight; varying float vA;
      void main() { vec2 q = gl_PointCoord * 2.0 - 1.0; float r = dot(q, q); if (r > 1.0) discard; gl_FragColor = vec4(uLight, vA * (1.0 - r) * 0.85); }`,
  }));
  for (const o of [rain, snow]) { o.frustumCulled = false; o.renderOrder = 8; o.visible = false; scene.add(o); }

  const camPrev = new THREE.Vector3(), camVel = new THREE.Vector3(), d = new THREE.Vector3();
  let type = 'clear', first = true;
  return {
    objects: [rain, snow],
    set(t) { type = t; rain.visible = t === 'rain'; snow.visible = t === 'snow'; },
    // wind: what the physics feels at the bike (m/s, world). shelter: 0 in the open .. 1 under a roof
    update(dt, camera, wind, shelter, viewH) {
      if (type === 'clear' || dt <= 0) { first = true; return; }
      d.copy(camera.position).sub(camPrev); camPrev.copy(camera.position);
      if (first || d.length() > 250 * dt + 5) d.set(0, 0, 0);                    // the camera was cut to another place
      first = false;
      camVel.lerp(d.divideScalar(dt), Math.min(1, dt * 12));
      const isRain = type === 'rain', u = isRain ? rainU : snowU, fall = isRain ? 9.5 : 1.3, carry = isRain ? 0.75 : 1;
      const vx = wind[0] * carry, vy = -fall + wind[1] * (isRain ? 0.3 : 0.8), vz = wind[2] * carry;
      u.uOff.value.x += vx * dt; u.uOff.value.y += vy * dt; u.uOff.value.z += vz * dt;
      const tau = 0.02;
      u.uStreak.value.set((vx - camVel.x) * tau, (vy - camVel.y) * tau, (vz - camVel.z) * tau);
      u.uAmt.value = 1 - shelter;
      u.uPx.value = viewH / (2 * Math.tan(camera.fov * Math.PI / 360));
      const a = U.uAmbSky.value, k = isRain ? 2.4 : 3.4;
      u.uLight.value.setRGB(Math.min(1.6, a.r * k + 0.05), Math.min(1.6, a.g * k + 0.05), Math.min(1.6, a.b * k + 0.06));
    },
  };
}
