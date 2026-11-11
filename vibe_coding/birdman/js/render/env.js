// Shared look: one sun, one sky, one haze. Every material in the scene uses the same GLSL so things sit in the same light.
import * as THREE from '../../lib/three.module.min.js';

export const SUN_DIR = new THREE.Vector3(-0.74, 0.44, 0.5).normalize();
export const U = {
  uSunDir: { value: SUN_DIR },
  uSunCol: { value: new THREE.Color(1.0, 0.86, 0.68).multiplyScalar(2.2) },
  uTime: { value: 0 },
  uWind: { value: new THREE.Vector2(1, 0.12) },
};

export const GLSL = /* glsl */`
  uniform vec3 uSunDir; uniform vec3 uSunCol; uniform float uTime; uniform vec2 uWind;
  float hash21(vec2 p) { uint h = uint(int(p.x)) * 1597334677u ^ uint(int(p.y)) * 3812015801u; h ^= h >> 16; h *= 0x7feb352du; h ^= h >> 15; h *= 0x846ca68bu; h ^= h >> 16; return float(h >> 8) / 16777216.0; }
  float vnoise(vec2 p) { vec2 i = floor(p), f = fract(p); f = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash21(i), hash21(i + vec2(1, 0)), f.x), mix(hash21(i + vec2(0, 1)), hash21(i + vec2(1, 1)), f.x), f.y); }
  float fbm(vec2 p) { float s = 0.0, a = 0.5; for (int i = 0; i < 4; i++) { s += a * vnoise(p); p = p * 2.03 + 7.1; a *= 0.5; } return s; }
  vec3 skyCol(vec3 d) {
    float y = max(d.y, 0.0);
    vec3 zen = vec3(0.09, 0.27, 0.70), hor = vec3(0.50, 0.70, 0.92);
    vec3 c = mix(hor, zen, pow(y, 0.45));
    float mu = max(dot(normalize(vec3(d.x, max(d.y, 0.0), d.z)), uSunDir), 0.0);
    c += vec3(1.0, 0.62, 0.30) * (0.55 * pow(mu, 6.0) * (1.0 - y * 0.6) + 0.25 * pow(mu, 2.0) * exp(-y * 5.0));
    c = mix(c, vec3(0.80, 0.82, 0.84) + vec3(0.30, 0.14, 0.0) * pow(mu, 3.0), exp(-y * 13.0) * 0.5);
    return c;
  }
  vec3 ambient(vec3 n) { return mix(vec3(0.30, 0.27, 0.22), vec3(0.34, 0.47, 0.70), n.y * 0.5 + 0.5) * 0.95; }
  // drifting cloud shadows on the ground
  float cloudShade(vec3 wp) { vec2 p = (wp.xz + uSunDir.xz / uSunDir.y * (1400.0 - wp.y)) / 2600.0 - uWind * uTime * 0.0035; return 1.0 - 0.5 * smoothstep(0.52, 0.72, fbm(p * 1.3)); }
  vec3 lit(vec3 albedo, vec3 n, float sunVis, float ao) {
    float ndl = max(dot(n, uSunDir), 0.0);
    return albedo * (uSunCol * ndl * sunVis + ambient(n) * ao);
  }
  vec3 haze(vec3 col, vec3 wp) {
    vec3 v = wp - cameraPosition; float d = length(v); v /= d;
    // denser near the sea, thinning with height
    float h = max(0.5 * (wp.y + cameraPosition.y), 0.0);
    float f = 1.0 - exp(-d * 0.000062 * exp(-h / 1100.0));
    return mix(col, skyCol(vec3(v.x, max(v.y, 0.0) * 0.4 + 0.03, v.z)) * 0.95, f);
  }
  vec3 finish(vec3 c) {
    c *= 0.82;
    c = (c * (2.51 * c + 0.03)) / (c * (2.43 * c + 0.59) + 0.14);
    return pow(clamp(c, 0.0, 1.0), vec3(1.0 / 2.2));
  }
`;

// Lit material for props, trees and the man: vertex colours, optional instancing, optional wind sway.
export function litMaterial({ sway = 0, doubleSide = false, emissive = 0, flat = false } = {}) {
  return new THREE.ShaderMaterial({
    uniforms: { ...U, uSway: { value: sway }, uEmissive: { value: emissive }, uTint: { value: new THREE.Color(1, 1, 1) } },
    vertexColors: true, side: doubleSide ? THREE.DoubleSide : THREE.FrontSide,
    vertexShader: /* glsl */`
      uniform float uSway; ${GLSL}
      varying vec3 vWp; varying vec3 vN; varying vec3 vCol;
      void main() {
        vec4 p = vec4(position, 1.0); vec3 n = normal;
        #ifdef USE_INSTANCING
          p = instanceMatrix * p; n = mat3(instanceMatrix) * n;
        #endif
        vec4 wp = modelMatrix * p;
        if (uSway > 0.0) { float k = position.y * position.y * uSway; wp.xz += uWind * k * (0.6 + 0.4 * sin(uTime * 1.7 + wp.x * 0.11 + wp.z * 0.07)); }
        vWp = wp.xyz; vN = normalize(mat3(modelMatrix) * n);
        vCol = color;
        #ifdef USE_INSTANCING_COLOR
          vCol *= instanceColor;
        #endif
        gl_Position = projectionMatrix * viewMatrix * wp;
      }`,
    fragmentShader: /* glsl */`
      uniform float uEmissive; uniform vec3 uTint; ${GLSL}
      varying vec3 vWp; varying vec3 vN; varying vec3 vCol;
      void main() {
        vec3 n = normalize(vN);
        ${flat ? 'n = normalize(cross(dFdx(vWp), dFdy(vWp)));' : ''}
        if (!gl_FrontFacing) n = -n;
        vec3 v = normalize(cameraPosition - vWp);
        vec3 alb = vCol * uTint;
        vec3 c = lit(alb, n, cloudShade(vWp), 1.0);
        ${doubleSide ? 'c += alb * uSunCol * 0.35 * max(dot(-n, uSunDir), 0.0);' : ''}
        c += alb * 0.25 * pow(1.0 - max(dot(n, v), 0.0), 3.0) * skyCol(n);
        c += alb * uEmissive;
        gl_FragColor = vec4(finish(haze(c, vWp)), 1.0);
      }`,
  });
}

// ---- tiny geometry kit: merge coloured pieces into one mesh ----
export class Kit {
  constructor() { this.pos = []; this.nor = []; this.col = []; }
  add(geo, color, matrix) {
    const g = geo.index ? geo.toNonIndexed() : geo.clone();
    if (matrix) g.applyMatrix4(matrix);
    const p = g.attributes.position.array, n = g.attributes.normal.array, c = new THREE.Color(color);
    for (let i = 0; i < p.length; i += 3) { this.pos.push(p[i], p[i + 1], p[i + 2]); this.nor.push(n[i], n[i + 1], n[i + 2]); this.col.push(c.r, c.g, c.b); }
    g.dispose();
    return this;
  }
  // shorthand: position, rotation (euler), scale
  put(geo, color, x = 0, y = 0, z = 0, rx = 0, ry = 0, rz = 0, sx = 1, sy = 1, sz = 1) {
    const m = new THREE.Matrix4().compose(new THREE.Vector3(x, y, z), new THREE.Quaternion().setFromEuler(new THREE.Euler(rx, ry, rz)), new THREE.Vector3(sx, sy, sz));
    return this.add(geo, color, m);
  }
  merge(other, matrix) {
    const v = new THREE.Vector3(), nm = new THREE.Matrix3().getNormalMatrix(matrix);
    for (let i = 0; i < other.pos.length; i += 3) {
      v.set(other.pos[i], other.pos[i + 1], other.pos[i + 2]).applyMatrix4(matrix); this.pos.push(v.x, v.y, v.z);
      v.set(other.nor[i], other.nor[i + 1], other.nor[i + 2]).applyMatrix3(nm).normalize(); this.nor.push(v.x, v.y, v.z);
      this.col.push(other.col[i], other.col[i + 1], other.col[i + 2]);
    }
    return this;
  }
  geometry() {
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(this.pos, 3));
    g.setAttribute('normal', new THREE.Float32BufferAttribute(this.nor, 3));
    g.setAttribute('color', new THREE.Float32BufferAttribute(this.col, 3));
    return g;
  }
}
