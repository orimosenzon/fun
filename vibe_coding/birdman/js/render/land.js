// Sky dome, the island mesh (with sun shadows baked from the height grid) and the sea.
import * as THREE from '../../lib/three.module.min.js';
import { U, GLSL, SUN_DIR } from './env.js';
import { H, N, CELL, HALF, POI } from '../sim/world.js';

export function buildSky(scene) {
  const mat = new THREE.ShaderMaterial({
    uniforms: { ...U }, side: THREE.BackSide, depthWrite: false,
    vertexShader: `varying vec3 vDir; void main() { vDir = position; vec4 p = projectionMatrix * viewMatrix * vec4(position * 100.0 + cameraPosition, 1.0); gl_Position = p.xyww; }`,
    fragmentShader: /* glsl */`
      ${GLSL}
      varying vec3 vDir;
      void main() {
        vec3 d = normalize(vDir);
        vec3 col = skyCol(d);
        float mu = dot(d, uSunDir);
        col += uSunCol * (smoothstep(0.99975, 0.9999, mu) * 30.0 + pow(max(mu, 0.0), 500.0) * 1.2 + pow(max(mu, 0.0), 40.0) * 0.10);
        if (d.y > 0.01) {
          // high thin cloud, streaked by the upper wind
          vec2 p = d.xz / (d.y + 0.08) * 1.6 + uWind * uTime * 0.003;
          float n = fbm(vec2(p.x * 0.7 + p.y * 0.4, p.y * 2.3 - p.x * 0.3));
          float c = smoothstep(0.50, 0.85, n) * smoothstep(0.01, 0.2, d.y) * 0.55;
          vec3 cc = mix(vec3(0.95, 0.93, 0.92), vec3(1.0, 0.82, 0.62), pow(max(mu, 0.0), 3.0)) * 1.25;
          col = mix(col, cc, c);
        }
        gl_FragColor = vec4(finish(col), 1.0);
      }`,
  });
  const m = new THREE.Mesh(new THREE.SphereGeometry(1, 32, 16), mat);
  m.frustumCulled = false; m.renderOrder = -10;
  scene.add(m);
  return m;
}

// how much sun reaches each grid vertex (soft), and a cheap sky-occlusion term
function bakeShade() {
  const sun = new Float32Array(N * N), ao = new Float32Array(N * N);
  const sh = Math.hypot(SUN_DIR.x, SUN_DIR.z), dx = SUN_DIR.x / sh, dz = SUN_DIR.z / sh, rise = (SUN_DIR.y / sh) * CELL;
  const at = (fi, fj) => {
    if (fi < 0 || fj < 0 || fi > N - 1.001 || fj > N - 1.001) return -50;
    const i = fi | 0, j = fj | 0, u = fi - i, v = fj - j, k = j * N + i;
    return (H[k] * (1 - u) + H[k + 1] * u) * (1 - v) + (H[k + N] * (1 - u) + H[k + N + 1] * u) * v;
  };
  for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) {
    const k = j * N + i, h0 = Math.max(H[k], 0) + 1.5;
    let vis = 1;
    for (let s = 1; s < 110; s++) {
      const hr = h0 + rise * s; if (hr > 840) break;
      const d = (hr - at(i + dx * s, j + dz * s)) / (s * CELL);
      if (d < 0.11) { vis = Math.min(vis, Math.max(0, d / 0.11)); if (vis <= 0) break; }
    }
    sun[k] = vis;
    let acc = 0, cnt = 0;
    for (let a = 0; a < 8; a++) { const ang = a * 0.785, r = 4; acc += Math.max(0, at(i + Math.cos(ang) * r, j + Math.sin(ang) * r) - H[k]); cnt++; }
    ao[k] = Math.max(0.45, 1 - (acc / cnt) / 110);
  }
  return { sun, ao };
}

export function buildTerrain(scene) {
  const pos = new Float32Array(N * N * 3), nor = new Float32Array(N * N * 3), shade = new Float32Array(N * N * 2);
  const { sun, ao } = bakeShade();
  for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) {
    const k = j * N + i;
    pos[k * 3] = i * CELL - HALF; pos[k * 3 + 1] = H[k]; pos[k * 3 + 2] = j * CELL - HALF;
    const hx = H[j * N + Math.min(N - 1, i + 1)] - H[j * N + Math.max(0, i - 1)], hz = H[Math.min(N - 1, j + 1) * N + i] - H[Math.max(0, j - 1) * N + i];
    const l = Math.hypot(hx, 2 * CELL, hz);
    nor[k * 3] = -hx / l; nor[k * 3 + 1] = (2 * CELL) / l; nor[k * 3 + 2] = -hz / l;
    shade[k * 2] = sun[k]; shade[k * 2 + 1] = ao[k];
  }
  const idx = new Uint32Array((N - 1) * (N - 1) * 6);
  let o = 0;
  for (let j = 0; j < N - 1; j++) for (let i = 0; i < N - 1; i++) {
    const a = j * N + i, b = a + 1, c = a + N, d = c + 1;
    idx[o++] = a; idx[o++] = c; idx[o++] = b; idx[o++] = b; idx[o++] = c; idx[o++] = d;
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  geo.setAttribute('normal', new THREE.BufferAttribute(nor, 3));
  geo.setAttribute('aShade', new THREE.BufferAttribute(shade, 2));
  geo.setIndex(new THREE.BufferAttribute(idx, 1));
  const mat = new THREE.ShaderMaterial({
    uniforms: { ...U, uFields: { value: new THREE.Vector3(POI.fields[0], POI.fields[2], 620) } },
    vertexShader: `attribute vec2 aShade; varying vec3 vWp; varying vec3 vN; varying vec2 vShade;
      void main() { vWp = position; vN = normal; vShade = aShade; gl_Position = projectionMatrix * viewMatrix * vec4(position, 1.0); }`,
    fragmentShader: /* glsl */`
      ${GLSL}
      uniform vec3 uFields;
      varying vec3 vWp; varying vec3 vN; varying vec2 vShade;
      void main() {
        vec3 n = normalize(vN);
        float h = vWp.y, dist = length(vWp - cameraPosition);
        float n1 = fbm(vWp.xz / 190.0), n2 = fbm(vWp.xz / 23.0 + 31.0), n3 = vnoise(vWp.xz / 3.1);
        float near = 1.0 - smoothstep(60.0, 420.0, dist);
        // detail bumps fade with distance
        vec2 e = vec2(1.6, 0.0);
        vec3 nb = normalize(n + (vec3(vnoise((vWp.xz + e.xy) / 5.0) - vnoise((vWp.xz - e.xy) / 5.0), 0.0, vnoise((vWp.xz + e.yx) / 5.0) - vnoise((vWp.xz - e.yx) / 5.0)) * 0.55 * near
                               + vec3(n2 - 0.5, 0.0, fbm(vWp.zx / 23.0 + 9.0) - 0.5) * 0.35));
        float slope = 1.0 - n.y;
        // grass: two greens and a dry tint that grows with height and on sun-facing slopes
        vec3 grass = mix(vec3(0.10, 0.24, 0.05), vec3(0.27, 0.36, 0.08), n1);
        grass = mix(grass, vec3(0.07, 0.17, 0.05), smoothstep(0.45, 0.7, fbm(vWp.xz / 420.0 + 3.0)) * 0.7);
        grass = mix(grass, vec3(0.50, 0.45, 0.20), clamp(smoothstep(0.35, 0.8, n2) * 0.7 + smoothstep(250.0, 600.0, h) * 0.5, 0.0, 0.85));
        float near2 = 1.0 - smoothstep(8.0, 70.0, dist);
        grass *= 0.72 + 0.4 * n3 * near + (0.3 * vnoise(vWp.xz / 0.6) + 0.25 * vnoise(vWp.xz / 0.13)) * near2;
        // anemones and daisies in the grass, close up
        vec2 fc = floor(vWp.xz * 2.2); float fh = hash21(fc);
        float dot_ = smoothstep(0.16, 0.08, length(fract(vWp.xz * 2.2) - 0.5 - 0.3 * (vec2(hash21(fc + 7.0), hash21(fc + 13.0)) - 0.5)));
        vec3 flower = fh > 0.985 ? vec3(0.85, 0.08, 0.06) : fh > 0.97 ? vec3(0.95, 0.93, 0.85) : vec3(0.95, 0.8, 0.15);
        grass = mix(grass, flower, dot_ * step(0.955, fh) * near2 * smoothstep(0.45, 0.6, fbm(vWp.xz / 60.0 + 5.0)));
        // farmland: a patchwork of strips
        float fm = exp(-dot(vWp.xz - uFields.xy, vWp.xz - uFields.xy) / (2.0 * uFields.z * uFields.z));
        vec2 fp = mat2(0.94, -0.34, 0.34, 0.94) * vWp.xz;
        vec2 cell = floor(fp / vec2(85.0, 150.0) + vec2(0.0, floor(fp.x / 85.0) * 0.37));
        float id = hash21(cell);
        vec3 field = id < 0.3 ? vec3(0.70, 0.56, 0.20) : id < 0.55 ? vec3(0.26, 0.40, 0.10) : id < 0.75 ? vec3(0.42, 0.30, 0.18) : id < 0.9 ? vec3(0.50, 0.55, 0.16) : vec3(0.72, 0.62, 0.22);
        field *= 0.88 + 0.16 * sin(fp.x * (1.2 + id) ) * near + 0.1 * n2;
        vec2 fe = abs(fract(fp / vec2(85.0, 150.0) + vec2(0.0, floor(fp.x / 85.0) * 0.37)) - 0.5);
        field = mix(field, vec3(0.20, 0.26, 0.08), smoothstep(0.47, 0.49, max(fe.x, fe.y)));
        vec3 col = mix(grass, field, smoothstep(0.42, 0.5, fm + (n2 - 0.5) * 0.08) * (1.0 - smoothstep(0.02, 0.06, slope)));
        // rock on steep ground, with strata; scree up high
        float strata = smoothstep(0.25, 0.75, fbm(vec2(h * 0.055 + n1 * 2.0, dot(vWp.xz, vec2(0.004, 0.003)))));
        vec3 rock = mix(vec3(0.25, 0.21, 0.18), vec3(0.46, 0.39, 0.30), strata) * (0.7 + 0.6 * n2) * (0.85 + 0.3 * n3 * near);
        rock = mix(rock, mix(vec3(0.27, 0.26, 0.26), vec3(0.42, 0.40, 0.38), strata) * (0.7 + 0.6 * n2), smoothstep(450.0, 760.0, h));
        float rk = smoothstep(0.16, 0.34, slope + (n2 - 0.5) * 0.12) + smoothstep(560.0, 780.0, h + n1 * 120.0) * 0.8;
        col = mix(col, rock, clamp(rk, 0.0, 1.0));
        // sand
        vec3 sand = vec3(0.80, 0.70, 0.50) * (0.9 + 0.2 * n2);
        sand = mix(sand * vec3(0.62, 0.60, 0.56), sand, smoothstep(0.0, 1.1, h));
        col = mix(sand, col, smoothstep(4.2, 7.5, h + (n2 - 0.5) * 3.0));
        col = mix(col, vec3(0.30, 0.34, 0.30), smoothstep(-0.5, -6.0, h) * 0.6);
        float vis = vShade.x * cloudShade(vWp);
        vec3 c = lit(col, nb, vis, vShade.y);
        gl_FragColor = vec4(finish(haze(c, vWp)), 1.0);
      }`,
  });
  const mesh = new THREE.Mesh(geo, mat);
  mesh.frustumCulled = false;
  scene.add(mesh);
  return mesh;
}

export function buildWater(scene) {
  // the height grid as a texture: the sea knows how deep it is and where the shore runs
  const data = new Uint16Array(N * N);
  for (let k = 0; k < N * N; k++) data[k] = THREE.DataUtils.toHalfFloat(Math.max(-60, Math.min(60, H[k])));
  const tex = new THREE.DataTexture(data, N, N, THREE.RedFormat, THREE.HalfFloatType);
  tex.minFilter = tex.magFilter = THREE.LinearFilter; tex.wrapS = tex.wrapT = THREE.ClampToEdgeWrapping; tex.needsUpdate = true;
  const mat = new THREE.ShaderMaterial({
    uniforms: { ...U, uH: { value: tex }, uHalf: { value: HALF } },
    transparent: true,
    vertexShader: `varying vec3 vWp; void main() { vec4 wp = modelMatrix * vec4(position, 1.0); wp.xz += floor(cameraPosition.xz / 64.0) * 64.0; vWp = wp.xyz; gl_Position = projectionMatrix * viewMatrix * wp; }`,
    fragmentShader: /* glsl */`
      ${GLSL}
      uniform sampler2D uH; uniform float uHalf;
      varying vec3 vWp;
      float wv(vec2 p, float t) { return vnoise(p + uWind * t) + 0.5 * vnoise(p * 2.3 - vec2(0.3, 1.0) * t * 1.4); }
      void main() {
        vec3 v = cameraPosition - vWp; float dist = length(v); v /= dist;
        vec2 uv = (vWp.xz + uHalf) / (2.0 * uHalf);
        float inside = step(0.0, uv.x) * step(uv.x, 1.0) * step(0.0, uv.y) * step(uv.y, 1.0);
        float depth = mix(45.0, -texture2D(uH, uv).r, inside);
        // swell + chop, as slopes
        float t = uTime;
        vec2 p1 = vWp.xz / 38.0, p2 = vWp.xz / 7.0;
        float e = 0.35;
        float a1 = 1.0 / (1.0 + dist / 2500.0), a2 = 1.0 / (1.0 + dist / 260.0);
        vec2 g = vec2(wv(p1 + vec2(e, 0), t * 0.11) - wv(p1 - vec2(e, 0), t * 0.11), wv(p1 + vec2(0, e), t * 0.11) - wv(p1 - vec2(0, e), t * 0.11)) * 0.38 * a1
               + vec2(wv(p2 + vec2(e, 0), t * 0.35) - wv(p2 - vec2(e, 0), t * 0.35), wv(p2 + vec2(0, e), t * 0.35) - wv(p2 - vec2(0, e), t * 0.35)) * 0.30 * a2;
        vec3 n = normalize(vec3(-g.x, 1.0, -g.y));
        float fres = 0.02 + 0.98 * pow(1.0 - max(dot(n, v), 0.0), 5.0);
        vec3 r = reflect(-v, n); r.y = abs(r.y);
        vec3 refl = skyCol(r);
        float sp = max(dot(r, uSunDir), 0.0);
        refl += uSunCol * (pow(sp, 600.0) * 14.0 + pow(sp, 60.0) * 0.35);
        // water body: turquoise over sand, deep blue further out
        vec3 shallow = vec3(0.10, 0.50, 0.50), deep = vec3(0.015, 0.10, 0.22);
        vec3 body = mix(shallow, deep, smoothstep(0.5, 16.0, depth)) * (0.55 + 0.45 * cloudShade(vWp));
        vec3 col = mix(body, refl, fres);
        // surf line that breathes
        float fo = smoothstep(1.6, 0.0, depth + 0.5 * sin(t * 0.7 + vWp.x * 0.02 + vWp.z * 0.03)) * smoothstep(0.35, 0.75, vnoise(vWp.xz / 3.0 + t * 0.2) + 0.25);
        fo += smoothstep(0.80, 0.95, wv(p2 * 0.6, t * 0.3) * 0.62) * 0.25 * a2;
        col = mix(col, vec3(0.95) * (uSunCol * 0.32 + 0.35), clamp(fo, 0.0, 1.0));
        float alpha = clamp(smoothstep(-0.1, 2.2, depth) * 0.75 + fres + fo + smoothstep(2.0, 9.0, depth) * 0.25, 0.0, 1.0);
        gl_FragColor = vec4(finish(haze(col, vWp)), alpha);
      }`,
  });
  const geo = new THREE.PlaneGeometry(60000, 60000, 1, 1).rotateX(-Math.PI / 2);
  const mesh = new THREE.Mesh(geo, mat);
  mesh.frustumCulled = false; mesh.renderOrder = -5;
  scene.add(mesh);
  return mesh;
}
