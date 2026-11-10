// Sky dome: scattering table + sun disc and glow + two procedural cloud layers lit by the sun.
import * as THREE from '../../lib/three.module.min.js';
import { U, GLSL_COMMON } from './env.js';

export function buildSky(scene) {
  const mat = new THREE.ShaderMaterial({
    uniforms: { ...U },
    side: THREE.BackSide, depthWrite: false, depthTest: true,
    vertexShader: /* glsl */`
      varying vec3 vDir;
      void main() { vDir = position; vec4 p = projectionMatrix * viewMatrix * vec4(position * 1000.0 + cameraPosition, 1.0); gl_Position = p.xyww; }`,
    fragmentShader: /* glsl */`
      precision highp float;
      ${GLSL_COMMON}
      varying vec3 vDir;
      float cloudN(vec2 p) { return 0.5 * vnoise(p * 16.0) + 0.27 * vnoise(p * 37.0 + 11.0) + 0.15 * vnoise(p * 83.0 + 5.0) + 0.08 * vnoise(p * 190.0); }
      void main() {
        vec3 d = normalize(vDir);
        if (uClipY > -1e8) d.y = abs(d.y);
        vec3 col = skyLUT(d);
        float mu = dot(d, uSunDir);
        // sun disc with limb darkening, and a tight glow
        float disc = smoothstep(0.99985, 0.99996, mu);
        col += uSunCol * (disc * 60.0 + pow(max(mu, 0.0), 900.0) * 3.0 + pow(max(mu, 0.0), 60.0) * 0.12);
        if (d.y > 0.004) {
          // cumulus deck
          float t = (uCloud.y - cameraPosition.y) / d.y;
          vec2 xz = cameraPosition.xz + d.xz * t;
          vec2 p = xz / 9000.0 + vec2(uTime * 0.0012, uTime * 0.0005);
          float n = cloudN(p);
          float lo = 1.0 - uCloud.x - 0.1;
          float dens = smoothstep(lo, lo + 0.26, n);
          // self-shadowing: density towards the sun
          vec2 sdir = normalize(uSunDir.xz + 1e-4) * (0.012 + 0.03 * (1.0 - uSunDir.y));
          float n2 = cloudN(p + sdir), n3 = cloudN(p + sdir * 2.4);
          float shade = clamp(1.0 - 2.6 * max(n2 - lo, 0.0) - 1.6 * max(n3 - lo, 0.0), 0.0, 1.0);
          float edge = 1.0 - smoothstep(0.0, 0.5, dens);
          float fwd = pow(max(mu, 0.0), 6.0);
          vec3 lit = uSunCol * (0.5 + 0.9 * shade + 2.2 * fwd * edge) + uAmbSky * 0.9;
          vec3 dark = uAmbSky * 0.55 + uSunCol * 0.06;
          vec3 cc = mix(dark, lit, 0.25 + 0.75 * shade) * 0.52;
          float far = exp(-t / 52000.0);
          cc = mix(skyLUT(normalize(vec3(d.x, 0.03, d.z))), cc, far);
          col = mix(col, cc, dens * smoothstep(0.004, 0.05, d.y) * (0.35 + 0.65 * far) * 0.96);
          // thin cirrus far above
          float t2 = (11000.0 - cameraPosition.y) / d.y;
          vec2 q = (cameraPosition.xz + d.xz * t2) / 30000.0 + uTime * 0.0006;
          float ci = smoothstep(0.52, 0.8, 0.6 * vnoise(q * vec2(40.0, 12.0)) + 0.4 * vnoise(q * vec2(90.0, 30.0) + 3.0));
          col += ci * (uSunCol * (0.10 + 0.5 * pow(max(mu, 0.0), 8.0)) + uAmbSky * 0.08) * exp(-t2 / 90000.0) * (1.0 - dens);
        }
        gl_FragColor = vec4(col, 1.0);
      }`,
  });
  const mesh = new THREE.Mesh(new THREE.IcosahedronGeometry(1, 3), mat);
  mesh.frustumCulled = false; mesh.renderOrder = -10; mesh.matrixAutoUpdate = false;
  scene.add(mesh);
  return mesh;
}
