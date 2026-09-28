// The valley: terrain (near detailed mesh + far mountains), baked terrain self-shadowing,
// lake, forest, rocks, launch pad, meadow grass that bends in the jet blast, sky and sun.

import * as THREE from 'three';
import { Sky } from 'three/addons/objects/Sky.js';
import { Water } from 'three/addons/objects/Water.js';
import { heightAt, WATER_Y, LAKE, HILL, LAND, PAD, valleyZ, fbm } from '../terrain.js';
import { rng } from '../vmath.js';

export const SUN_DIR = new THREE.Vector3(0.42, 0.52, 0.74).normalize();

const NOISE_GLSL = /* glsl */`
float h21(vec2 p){ p = fract(p*vec2(123.34,456.21)); p += dot(p,p+45.32); return fract(p.x*p.y); }
float vn(vec2 p){ vec2 i=floor(p), f=fract(p); vec2 u=f*f*(3.0-2.0*f);
  return mix(mix(h21(i),h21(i+vec2(1,0)),u.x), mix(h21(i+vec2(0,1)),h21(i+vec2(1,1)),u.x), u.y); }
float fbm2(vec2 p){ float s=0.0,a=0.5; mat2 r=mat2(0.8,-0.6,0.6,0.8); for(int i=0;i<5;i++){ s+=a*vn(p); p=r*p*2.03; a*=0.5;} return s; }
`;

function terrainMaterial(uniforms) {
  const mat = new THREE.MeshStandardMaterial({ roughness: 0.95, metalness: 0 });
  mat.onBeforeCompile = (sh) => {
    Object.assign(sh.uniforms, uniforms);
    sh.vertexShader = sh.vertexShader
      .replace('#include <common>', '#include <common>\nattribute float sunVis;\nvarying float vSunVis;\nvarying vec3 vWPos;\nvarying vec3 vWNrm;')
      .replace('#include <worldpos_vertex>', '#include <worldpos_vertex>\nvSunVis = sunVis;\nvWPos = (modelMatrix*vec4(transformed,1.0)).xyz;\nvWNrm = normalize(mat3(modelMatrix)*objectNormal);');
    sh.fragmentShader = sh.fragmentShader
      .replace('#include <common>', `#include <common>
varying float vSunVis; varying vec3 vWPos; varying vec3 vWNrm;
uniform float uWaterY; uniform vec2 uPad; uniform vec2 uLand;
${NOISE_GLSL}`)
      .replace('#include <color_fragment>', `#include <color_fragment>
{
  vec2 xz = vWPos.xz;
  float slope = 1.0 - vWNrm.y;
  float n1 = fbm2(xz*0.02), n2 = fbm2(xz*0.35), n3 = vn(xz*3.1);
  vec3 grassA = vec3(0.085,0.16,0.035), grassB = vec3(0.17,0.21,0.06), dry = vec3(0.30,0.26,0.10);
  vec3 col = mix(grassA, grassB, smoothstep(0.3,0.7,n1));
  col = mix(col, dry, smoothstep(0.55,0.8,fbm2(xz*0.008+7.0))*0.6);
  col *= 0.8 + 0.4*n2;
  col *= 0.9 + 0.2*n3;
  // alpine meadow -> scree -> rock -> snow with altitude and slope
  float strata = fbm2(vec2(xz.x*0.03 + xz.y*0.02, vWPos.y*0.55));
  vec3 rock = mix(vec3(0.115,0.11,0.10), vec3(0.24,0.225,0.20), strata) * (0.75 + 0.5*n2);
  rock = mix(rock, vec3(0.10,0.13,0.06), smoothstep(0.55,0.75,fbm2(xz*0.05+3.0))*0.6); // lichen and moss
  rock *= 0.8 + 0.35*smoothstep(0.2,0.8,vn(vec2(xz.x*0.2, vWPos.y*2.0)));             // ledges
  vec3 scree = vec3(0.30,0.28,0.25) * (0.8+0.4*n2);
  float alt = vWPos.y + 60.0*(n1-0.5);
  col = mix(col, mix(col, scree, 0.6), smoothstep(250.0, 480.0, alt));
  col = mix(col, rock, smoothstep(0.30, 0.50, slope + 0.15*(n2-0.5)));
  float snow = smoothstep(900.0, 1000.0, alt + 140.0*n2) * (1.0 - smoothstep(0.5, 0.7, slope));
  col = mix(col, vec3(0.80,0.83,0.88), snow);
  // lake shore mud and lake bed
  float shore = 1.0 - smoothstep(uWaterY+0.2, uWaterY+1.6, vWPos.y);
  col = mix(col, vec3(0.13,0.11,0.08)*(0.8+0.4*n2), shore);
  // launch pad gravel apron
  float dp = length(xz - uPad);
  col = mix(col, vec3(0.26,0.24,0.21)*(0.75+0.5*n3)*(0.85+0.3*n2), 1.0 - smoothstep(13.0, 19.0, dp + 2.0*n2));
  // trodden path at the landing meadow
  float dl = length(xz - uLand);
  col = mix(col, col*0.8 + vec3(0.03,0.025,0.0), (1.0 - smoothstep(4.0, 9.0, dl))*0.5);
  diffuseColor.rgb = col;
  // baked terrain self-shadow: shadowed ground only gets skylight
  diffuseColor.rgb *= mix(0.42, 1.0, vSunVis);
}`)
      .replace('#include <normal_fragment_maps>', `#include <normal_fragment_maps>
{
  // procedural micro relief for close shots
  vec2 xz = vWPos.xz; float e = 0.15;
  float hC = fbm2(xz*0.9), hX = fbm2((xz+vec2(e,0.0))*0.9), hZ = fbm2((xz+vec2(0.0,e))*0.9);
  vec3 bumpW = vec3(-(hX-hC)/e, 0.0, -(hZ-hC)/e) * 0.35;
  normal = normalize(normal + (viewMatrix*vec4(bumpW,0.0)).xyz);
}`);
  };
  return mat;
}

// Building the world takes seconds of pure computation. The heavy loops pause every ~50 ms so
// the browser can repaint the loading bar (a single long task would freeze the page).
// stage(a, b, label) maps the next loop's 0..1 onto a..b of the whole build; label may be a
// function of the sub-phase ('height' / 'shade' in gridMesh).
function makeSlicer(onProgress) {
  let last = performance.now(), a = 0, b = 1, label = '';
  const yieldTask = globalThis.scheduler?.yield ? () => scheduler.yield() : () => new Promise((r) => setTimeout(r, 0));
  const say = (f, sub) => onProgress(a + (b - a) * f, typeof label === 'function' ? label(sub) : label);
  return {
    due: () => performance.now() - last > 50,
    async pause(f, sub) { say(f, sub); await yieldTask(); last = performance.now(); },
    async stage(a1, b1, lab) { a = a1; b = b1; label = lab; await this.pause(0); },
  };
}

async function gridMesh(x0, z0, w, d, nx, nz, mat, { sink = null, fallback = () => -1e9, slicer, hFrac = 0.8 } = {}) {
  // sink(x, z) -> metres to lower the rendered surface (hidden areas / under the pad)
  // hFrac: share of the progress for the height pass (heightAt is the expensive part)
  const g = new THREE.BufferGeometry();
  const pos = new Float32Array(nx * nz * 3);
  const H = new Float32Array(nx * nz);
  for (let j = 0; j < nz; j++) {
    for (let i = 0; i < nx; i++) {
      const x = x0 + w * i / (nx - 1), z = z0 + d * j / (nz - 1);
      const h = heightAt(x, z);
      const k = j * nx + i;
      H[k] = h;
      pos[k * 3] = x; pos[k * 3 + 1] = sink ? h - sink(x, z) : h; pos[k * 3 + 2] = z;
    }
    if (slicer.due()) await slicer.pause(hFrac * j / nz, 'height');
  }
  const nrm = new Float32Array(nx * nz * 3);
  const dx = w / (nx - 1), dz = d / (nz - 1);
  for (let j = 0; j < nz; j++) for (let i = 0; i < nx; i++) {
    const hl = H[j * nx + Math.max(0, i - 1)], hr = H[j * nx + Math.min(nx - 1, i + 1)];
    const hu = H[Math.max(0, j - 1) * nx + i], hd = H[Math.min(nz - 1, j + 1) * nx + i];
    const n = new THREE.Vector3(-(hr - hl) / (2 * dx), 1, -(hd - hu) / (2 * dz)).normalize();
    const k = (j * nx + i) * 3;
    nrm[k] = n.x; nrm[k + 1] = n.y; nrm[k + 2] = n.z;
  }
  // sun visibility: march the height grid toward the sun (soft horizon)
  const vis = new Float32Array(nx * nz);
  const sx = SUN_DIR.x, sz = SUN_DIR.z, sh = SUN_DIR.y / Math.hypot(sx, sz);
  const stepM = Math.max(dx, 6);
  const sample = (x, z) => {
    const fi = (x - x0) / dx, fj = (z - z0) / dz;
    if (fi < 0 || fj < 0 || fi > nx - 1 || fj > nz - 1) return null;
    const i = Math.floor(fi), j = Math.floor(fj), u = fi - i, v = fj - j;
    const i1 = Math.min(i + 1, nx - 1), j1 = Math.min(j + 1, nz - 1);
    return (H[j * nx + i] * (1 - u) + H[j * nx + i1] * u) * (1 - v) + (H[j1 * nx + i] * (1 - u) + H[j1 * nx + i1] * u) * v;
  };
  const hz = Math.hypot(sx, sz);
  for (let k = 0; k < nx * nz; k++) {
    if ((k & 1023) === 0 && slicer.due()) await slicer.pause(hFrac + (1 - hFrac) * k / (nx * nz), 'shade');
    const x = pos[k * 3], y = H[k], z = pos[k * 3 + 2];
    let minMargin = 1e9;
    for (let s = stepM; s < 2600; s *= 1.12) {
      const px = x + sx / hz * s, pz = z + sz / hz * s;
      const h = sample(px, pz) ?? fallback(px, pz);
      const ray = y + 1.5 + s * sh;
      minMargin = Math.min(minMargin, (ray - h) / (s * 0.08 + 2));
      if (ray > 1900) break;
    }
    vis[k] = THREE.MathUtils.clamp(minMargin * 0.5 + 0.5, 0, 1);
  }
  const idx = [];
  for (let j = 0; j < nz - 1; j++) for (let i = 0; i < nx - 1; i++) {
    const a = j * nx + i, b = a + 1, c = a + nx, d2 = c + 1;
    idx.push(a, c, b, b, c, d2);
  }
  g.setIndex(idx);
  g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  g.setAttribute('normal', new THREE.BufferAttribute(nrm, 3));
  g.setAttribute('sunVis', new THREE.BufferAttribute(vis, 1));
  g.computeBoundingSphere();
  const m = new THREE.Mesh(g, mat);
  m.receiveShadow = true;
  return { mesh: m, sampleH: (x, z) => sample(x, z) ?? heightAt(x, z), sunVisAt: (x, z) => {
    const fi = (x - x0) / dx, fj = (z - z0) / dz;
    if (fi < 0 || fj < 0 || fi > nx - 1 || fj > nz - 1) return 1;
    return vis[Math.round(fj) * nx + Math.round(fi)];
  } };
}

// tileable normal map for the lake surface
function waterNormals() {
  const N = 256, c = document.createElement('canvas');
  c.width = c.height = N;
  const ctx = c.getContext('2d');
  const img = ctx.createImageData(N, N);
  const R = rng(5);
  const waves = Array.from({ length: 40 }, () => {
    const a = R() * Math.PI * 2, k = 1 + Math.floor(R() * 9);
    return { kx: Math.round(Math.cos(a) * k), kz: Math.round(Math.sin(a) * k), amp: 1 / (k * 1.3), ph: R() * 6.28 };
  });
  for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) {
    let gx = 0, gz = 0;
    for (const w of waves) {
      const arg = 2 * Math.PI * (w.kx * i + w.kz * j) / N + w.ph;
      const c0 = Math.cos(arg) * w.amp;
      gx += c0 * w.kx; gz += c0 * w.kz;
    }
    const n = new THREE.Vector3(-gx * 0.08, 1, -gz * 0.08).normalize();
    const k = (j * N + i) * 4;
    img.data[k] = (n.x * 0.5 + 0.5) * 255; img.data[k + 1] = (n.z * 0.5 + 0.5) * 255;
    img.data[k + 2] = (n.y * 0.5 + 0.5) * 255; img.data[k + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
  const t = new THREE.CanvasTexture(c);
  t.wrapS = t.wrapT = THREE.RepeatWrapping;
  return t;
}

function coniferGeometry(seed, tiers = 8, seg = 9) {
  // spruce: many drooping tiers with ragged edges, "fuzzy" normals (radial + up) so the
  // canopy lights like foliage instead of like a faceted cone
  const R = rng(seed);
  const parts = [];
  const trunk = new THREE.CylinderGeometry(0.012, 0.035, 0.35, 5).translate(0, 0.175, 0);
  paint(trunk, [0.10, 0.07, 0.05]);
  parts.push(trunk);
  for (let i = 0; i < tiers; i++) {
    const u = i / (tiers - 1);
    const y0 = 0.10 + u * 0.80;
    const r = 0.30 * (1 - u) ** 0.9 + 0.035;
    const h = 0.20 - 0.07 * u;
    const g = new THREE.ConeGeometry(r, h, seg, 1, true).translate(0, y0 + h / 2, 0);
    const p = g.attributes.position;
    const nrm = new Float32Array(p.count * 3), col = new Float32Array(p.count * 3);
    for (let k = 0; k < p.count; k++) {
      let x = p.getX(k), y = p.getY(k), z = p.getZ(k);
      const rr = Math.hypot(x, z);
      if (rr > r * 0.6) { // ragged, drooping rim
        const f = 0.8 + 0.4 * R();
        x *= f; z *= f; y -= (0.02 + 0.03 * R()) * (1 - u);
      }
      p.setXYZ(k, x, y, z);
      const n = new THREE.Vector3(x, 0.55 * r + 0.05, z).normalize();
      nrm[k * 3] = n.x; nrm[k * 3 + 1] = n.y; nrm[k * 3 + 2] = n.z;
      const tip = Math.min(1, rr / r);
      const shade = (0.55 + 0.45 * tip) * (0.75 + 0.25 * u);
      col[k * 3] = 0.030 * shade; col[k * 3 + 1] = 0.068 * shade; col[k * 3 + 2] = 0.036 * shade;
    }
    g.setAttribute('normal', new THREE.BufferAttribute(nrm, 3));
    g.setAttribute('color', new THREE.BufferAttribute(col, 3));
    parts.push(g);
  }
  return merge(parts);
}

function broadleafGeometry(seed, blobs = 6) {
  const R = rng(seed);
  const parts = [];
  const trunk = new THREE.CylinderGeometry(0.02, 0.04, 0.5, 6).translate(0, 0.25, 0);
  paint(trunk, [0.22, 0.20, 0.17]);
  parts.push(trunk);
  for (let k = 0; k < blobs; k++) {
    const b = new THREE.IcosahedronGeometry(blobs === 1 ? 0.3 : 0.16 + 0.1 * R(), blobs === 1 ? 0 : 1);
    const cx = blobs === 1 ? 0 : (R() - 0.5) * 0.32, cy = blobs === 1 ? 0.66 : 0.5 + R() * 0.38, cz = blobs === 1 ? 0 : (R() - 0.5) * 0.32;
    b.translate(cx, cy, cz);
    const p = b.attributes.position, nrm = new Float32Array(p.count * 3), col = new Float32Array(p.count * 3);
    for (let i = 0; i < p.count; i++) {
      const v = new THREE.Vector3(p.getX(i), p.getY(i) - 0.62, p.getZ(i));
      const n = v.clone().normalize();
      nrm[i * 3] = n.x; nrm[i * 3 + 1] = n.y; nrm[i * 3 + 2] = n.z;
      const sh = 0.6 + 0.4 * Math.max(0, n.y);
      col[i * 3] = 0.10 * sh; col[i * 3 + 1] = 0.165 * sh; col[i * 3 + 2] = 0.05 * sh;
    }
    b.setAttribute('normal', new THREE.BufferAttribute(nrm, 3));
    b.setAttribute('color', new THREE.BufferAttribute(col, 3));
    parts.push(b);
  }
  return merge(parts);
}
function paint(g, rgb, darkBottom = false) {
  const n = g.attributes.position.count, col = new Float32Array(n * 3);
  const box = new THREE.Box3().setFromBufferAttribute(g.attributes.position);
  for (let i = 0; i < n; i++) {
    const y = g.attributes.position.getY(i);
    const f = darkBottom ? 0.6 + 0.4 * (y - box.min.y) / (box.max.y - box.min.y + 1e-6) : 1;
    col[i * 3] = rgb[0] * f; col[i * 3 + 1] = rgb[1] * f; col[i * 3 + 2] = rgb[2] * f;
  }
  g.setAttribute('color', new THREE.BufferAttribute(col, 3));
}
export function merge(geoms) {
  const gs = geoms.map((g) => (g.index ? g.toNonIndexed() : g));
  let n = 0;
  for (const g of gs) n += g.attributes.position.count;
  const pos = new Float32Array(n * 3), nor = new Float32Array(n * 3), col = new Float32Array(n * 3);
  let o = 0;
  for (const g of gs) {
    if (!g.attributes.normal) g.computeVertexNormals();
    pos.set(g.attributes.position.array, o * 3);
    nor.set(g.attributes.normal.array, o * 3);
    if (g.attributes.color) col.set(g.attributes.color.array, o * 3); else col.fill(1, o * 3, (o + g.attributes.position.count) * 3);
    o += g.attributes.position.count;
  }
  const m = new THREE.BufferGeometry();
  m.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  m.setAttribute('normal', new THREE.BufferAttribute(nor, 3));
  m.setAttribute('color', new THREE.BufferAttribute(col, 3));
  return m;
}

function rockGeometry(seed) {
  const g = new THREE.IcosahedronGeometry(1, 2);
  const R = rng(seed);
  const p = g.attributes.position;
  const offs = Array.from({ length: 6 }, () => [R() * 2 - 1, R() * 2 - 1, R() * 2 - 1, R()]);
  for (let i = 0; i < p.count; i++) {
    const v = new THREE.Vector3().fromBufferAttribute(p, i);
    let s = 1;
    for (const [a, b, c, w] of offs) s += 0.18 * w * Math.sin(3 * (a * v.x + b * v.y + c * v.z) + w * 6);
    v.multiplyScalar(s);
    v.y = v.y > 0 ? v.y * 0.75 : v.y * 0.4;
    p.setXYZ(i, v.x, v.y, v.z);
  }
  g.computeVertexNormals();
  paint(g, [0.28, 0.27, 0.25]);
  return g;
}

// Grass blades that lean away from jet impingement points (vertex shader)
async function makeGrass(centers, uniforms, slicer) {
  const blade = new THREE.PlaneGeometry(0.05, 1, 1, 4).translate(0, 0.5, 0);
  const pos = blade.attributes.position;
  for (let i = 0; i < pos.count; i++) {
    const y = pos.getY(i);
    pos.setX(i, pos.getX(i) * (1 - y * 0.85));
  }
  blade.computeVertexNormals();
  const items = [];
  const R = rng(17);
  for (const c of centers) {
    for (let k = 0; k < c.count; k++) {
      if ((k & 1023) === 0 && slicer.due()) await slicer.pause(k / c.count);
      const r = c.r0 + Math.sqrt(R()) * (c.r1 - c.r0), a = R() * Math.PI * 2;
      // thin out toward the ring edges so the patch melts into the terrain
      const edge = Math.min((r - c.r0) / 4 + (c.r0 === 0 ? 1 : 0), (c.r1 - r) / 9, 1);
      if (R() > edge) continue;
      const x = c.x + Math.cos(a) * r, z = c.z + Math.sin(a) * r;
      const y = heightAt(x, z);
      if (y < WATER_Y + 0.4) continue;
      items.push([x, y, z, R() * Math.PI * 2, 0.12 + R() * R() * 0.4, R()]);
    }
  }
  const mat = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.8, side: THREE.DoubleSide });
  mat.onBeforeCompile = (sh) => {
    Object.assign(sh.uniforms, uniforms);
    sh.vertexShader = sh.vertexShader
      .replace('#include <common>', `#include <common>
uniform vec4 uBlast[5]; uniform float uTime; uniform vec3 uWind;
attribute float aTone;
varying float vTone; varying float vH;`)
      .replace('#include <begin_vertex>', `#include <begin_vertex>
vec4 wpos0 = instanceMatrix * vec4(0.0,0.0,0.0,1.0);
float hgt = position.y;
vec2 push = uWind.xz * 0.05 * (0.6 + 0.4*sin(uTime*2.3 + wpos0.x*0.7 + wpos0.z*0.5));
for (int i=0;i<5;i++){
  vec2 d = wpos0.xz - uBlast[i].xy;
  float r = length(d) + 0.3;
  // wall jet from the impingement point: strength falls with radius
  float s = uBlast[i].z * exp(-r / max(uBlast[i].w, 0.5)) ;
  float flick = 0.75 + 0.25*sin(uTime*23.0 + r*3.1 + wpos0.x*5.0);
  push += normalize(d) * s * flick;
}
float bend = clamp(length(push), 0.0, 1.45);
vec2 dir = length(push) > 1e-4 ? normalize(push) : vec2(0.0);
// rotate the blade toward dir, stronger at the tip; keep its length
float ang = bend * hgt * hgt;
mat3 imx = mat3(instanceMatrix);
float hs = length(imx[1]);
vec3 bw = vec3(dir.x*sin(ang), cos(ang)-1.0, dir.y*sin(ang)) * hgt * hs;
vec3 local = inverse(imx) * bw;
transformed += local;
vTone = aTone; vH = hgt;`);
    sh.fragmentShader = sh.fragmentShader
      .replace('#include <common>', '#include <common>\nvarying float vTone; varying float vH;')
      .replace('#include <color_fragment>', `#include <color_fragment>
diffuseColor.rgb = mix(vec3(0.075,0.15,0.03), vec3(0.17,0.21,0.06), vTone) * (0.55 + 0.6*vH);`);
  };
  const mesh = new THREE.InstancedMesh(blade, mat, items.length);
  const m4 = new THREE.Matrix4(), q = new THREE.Quaternion();
  const tone = new Float32Array(items.length);
  items.forEach(([x, y, z, rot, h, t], i) => {
    q.setFromAxisAngle(new THREE.Vector3(0, 1, 0), rot);
    m4.compose(new THREE.Vector3(x, y - 0.02, z), q, new THREE.Vector3(1, h, 1));
    mesh.setMatrixAt(i, m4);
    tone[i] = t;
  });
  blade.setAttribute('aTone', new THREE.InstancedBufferAttribute(tone, 1));
  mesh.computeBoundingSphere();
  mesh.receiveShadow = true;
  return mesh;
}

function makePad() {
  // concrete launch pad with painted markings (canvas texture)
  const c = document.createElement('canvas');
  c.width = c.height = 1024;
  const g = c.getContext('2d');
  g.fillStyle = '#8b8a86'; g.fillRect(0, 0, 1024, 1024);
  const R = rng(9);
  for (let i = 0; i < 26000; i++) {
    const v = 110 + R() * 60;
    g.fillStyle = `rgba(${v},${v - 2},${v - 6},${0.25 * R()})`;
    g.fillRect(R() * 1024, R() * 1024, 1 + R() * 3, 1 + R() * 3);
  }
  for (let i = 0; i < 40; i++) { // soot stains
    const x = 512 + (R() - 0.5) * 300, y = 512 + (R() - 0.5) * 300, r = 30 + R() * 110;
    const gr = g.createRadialGradient(x, y, 0, x, y, r);
    gr.addColorStop(0, 'rgba(30,28,26,0.22)'); gr.addColorStop(1, 'rgba(30,28,26,0)');
    g.fillStyle = gr; g.fillRect(x - r, y - r, 2 * r, 2 * r);
  }
  g.strokeStyle = '#d9d4c3'; g.lineWidth = 16;
  g.beginPath(); g.arc(512, 512, 400, 0, Math.PI * 2); g.stroke();
  g.strokeStyle = '#e0b21e'; g.lineWidth = 10;
  g.beginPath(); g.arc(512, 512, 372, 0, Math.PI * 2); g.stroke();
  g.fillStyle = '#d9d4c3';
  g.fillRect(372, 320, 60, 384); g.fillRect(592, 320, 60, 384); g.fillRect(372, 482, 280, 60);
  // expansion joints
  g.strokeStyle = 'rgba(40,38,35,0.6)'; g.lineWidth = 3;
  for (let k = 1; k < 4; k++) { g.beginPath(); g.moveTo(k * 256, 0); g.lineTo(k * 256, 1024); g.stroke(); g.beginPath(); g.moveTo(0, k * 256); g.lineTo(1024, k * 256); g.stroke(); }
  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.anisotropy = 8;
  const mesh = new THREE.Mesh(new THREE.CylinderGeometry(9, 9.3, 0.3, 64),
    [new THREE.MeshStandardMaterial({ color: 0x77756f, roughness: 0.9 }), new THREE.MeshStandardMaterial({ map: tex, roughness: 0.85 }), new THREE.MeshStandardMaterial({ color: 0x55534f })]);
  mesh.position.set(PAD[0], PAD[1] - 0.15 + 0.0, PAD[2]);
  mesh.receiveShadow = true;
  mesh.rotation.y = Math.PI / 2;
  return mesh;
}

export async function buildWorld(scene, renderer, { clearings = [], quality = 1, corridor = [] } = {}, onProgress = () => {}) {
  const slicer = makeSlicer(onProgress);
  const uniforms = {
    uWaterY: { value: WATER_Y }, uPad: { value: new THREE.Vector2(PAD[0], PAD[2]) },
    uLand: { value: new THREE.Vector2(LAND.x, LAND.z) },
  };
  const tmat = terrainMaterial(uniforms);
  // near: detailed flight area
  // the whole course, from the launch pad to the landing meadow beyond the gorge
  const NEAR = { x0: -520, z0: -820, w: 3720, d: 1560 };
  // far: mountains out to ~14 km, sunk under the near patch
  const inNear = (x, z) => x > NEAR.x0 + 30 && x < NEAR.x0 + NEAR.w - 30 && z > NEAR.z0 + 30 && z < NEAR.z0 + NEAR.d - 30;
  await slicer.stage(0, 0.25, (ph) => (ph === 'shade' ? 'מחשב את הצללים שההרים מטילים' : 'מרים את רכסי ההרים סביב העמק'));
  const far = await gridMesh(-13000, -14000, 28000, 28000, 561, 561, tmat, { sink: (x, z) => (inNear(x, z) ? 40 : 0), slicer, hFrac: 0.9 });
  scene.add(far.mesh);
  await slicer.stage(0.25, 0.81, (ph) => (ph === 'shade' ? 'מחשב לכל נקודה בעמק אם השמש מגיעה אליה' : 'מעצב את קרקע העמק, נקודה כל 3 מטרים'));
  const near = await gridMesh(NEAR.x0, NEAR.z0, NEAR.w, NEAR.d, quality > 0.7 ? 1241 : 931, quality > 0.7 ? 521 : 391, tmat, { fallback: far.sampleH, sink: (x, z) => (Math.hypot(x - PAD[0], z - PAD[2]) < 9.6 ? 0.25 : 0), slicer, hFrac: 0.72 });
  scene.add(near.mesh);
  const sunVisAt = (x, z) => (inNear(x, z) ? near.sunVisAt(x, z) : far.sunVisAt(x, z));
  const hAt = (x, z) => (inNear(x, z) ? near.sampleH(x, z) : far.sampleH(x, z));

  // sky
  await slicer.stage(0.81, 0.825, 'שמיים, אור שמש ואגם');
  const sky = new Sky();
  sky.scale.setScalar(45000);
  const su = sky.material.uniforms;
  su.turbidity.value = 3.2; su.rayleigh.value = 1.1; su.mieCoefficient.value = 0.004; su.mieDirectionalG.value = 0.8;
  su.sunPosition.value.copy(SUN_DIR).multiplyScalar(1000);
  scene.add(sky);

  // environment for reflections: render the sky into a PMREM
  const pmrem = new THREE.PMREMGenerator(renderer);
  const envScene = new THREE.Scene();
  const sky2 = new Sky(); sky2.scale.setScalar(1000);
  Object.keys(su).forEach((k) => { sky2.material.uniforms[k].value = su[k].value; });
  envScene.add(sky2);
  const ground = new THREE.Mesh(new THREE.CircleGeometry(900, 32).rotateX(-Math.PI / 2), new THREE.MeshBasicMaterial({ color: 0x1e2414 }));
  ground.position.y = -20;
  envScene.add(ground);
  const env = pmrem.fromScene(envScene, 0.02).texture;
  scene.environment = env;
  scene.environmentIntensity = 0.75;

  // lights
  const sun = new THREE.DirectionalLight(0xfff0dc, 3.4);
  sun.position.copy(SUN_DIR).multiplyScalar(300);
  sun.castShadow = true;
  sun.shadow.mapSize.set(quality > 0.7 ? 2048 : 1024, quality > 0.7 ? 2048 : 1024);
  const sc = sun.shadow.camera;
  sc.left = -18; sc.right = 18; sc.top = 18; sc.bottom = -18; sc.near = 1; sc.far = 900;
  sun.shadow.bias = -0.00015;
  sun.shadow.normalBias = 0.02;
  scene.add(sun, sun.target);
  const hemi = new THREE.HemisphereLight(0xa9c4e8, 0x4a4a2a, 0.9);
  scene.add(hemi);
  scene.fog = new THREE.FogExp2(0x9fb3c8, 0.000085);

  // lake
  const waterGeom = new THREE.PlaneGeometry(LAKE.rx * 2.8, LAKE.rz * 2.8, 1, 1);
  const water = new Water(waterGeom, {
    textureWidth: 512, textureHeight: 512, waterNormals: waterNormals(),
    sunDirection: SUN_DIR.clone(), sunColor: 0xfff0dc, waterColor: 0x0b2a2e, distortionScale: 2.2,
    fog: true, alpha: 1.0,
  });
  water.rotation.x = -Math.PI / 2;
  water.position.set(LAKE.x, WATER_Y, valleyZ(LAKE.x) + LAKE.dz);
  water.material.uniforms.size.value = 3.0;
  scene.add(water);
  // the mirror pass re-renders the whole scene: refresh it every other frame
  { const mirror = water.onBeforeRender; let n = 0; water.onBeforeRender = function (...a) { if ((n++ & 1) === 0) mirror.apply(this, a); }; }

  // forest
  const R = rng(3);
  await slicer.stage(0.825, 0.83, 'מכין דגמים של עצים');
  // detailed trees near the camera, simple ones beyond (swapped per 200 m chunk every frame)
  const species = [coniferGeometry(1), coniferGeometry(2), broadleafGeometry(3)];
  const speciesLo = [coniferGeometry(1, 3, 6), coniferGeometry(2, 3, 6), broadleafGeometry(3, 1)];
  // horizontal distance from a point to the course polyline, and the course height there
  const corridorAt = (x, z) => {
    let best = 1e9, y = 0;
    for (const [a, b] of corridor) {
      const dx = b[0] - a[0], dz = b[2] - a[2], L2 = dx * dx + dz * dz || 1;
      const u = Math.max(0, Math.min(1, ((x - a[0]) * dx + (z - a[2]) * dz) / L2));
      const d = Math.hypot(x - a[0] - dx * u, z - a[2] - dz * u);
      if (d < best) { best = d; y = a[1] + (b[1] - a[1]) * u; }
    }
    return [best, y];
  };
  const treeMat = new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.92 });
  const nearT = [], farT = [];
  const place = (x, z, list, minS, maxS) => {
    if (Math.hypot(x - PAD[0], z - PAD[2]) < 70 || Math.hypot(x - LAND.x, z - LAND.z) < 60) return;
    for (const [cx, cz] of clearings) if (Math.hypot(x - cx, z - cz) < 55) return;
    const dens = fbm(x * 0.006 + 3, z * 0.006, 3);
    if (dens < 0.3) return;
    const y = hAt(x, z);
    if (y < WATER_Y + 1.2 || y > 620 + 120 * fbm(x * 0.01, z * 0.01, 2)) return;
    const e = 4, sl = Math.hypot(hAt(x + e, z) - hAt(x - e, z), hAt(x, z + e) - hAt(x, z - e)) / (2 * e);
    if (sl > 0.85) return;
    const hz = valleyZ(HILL.x) + HILL.dz;
    const valleyD = Math.abs(z - valleyZ(x));
    const want = valleyD < 150 ? dens - 0.62 : dens - 0.38 + Math.min(0.2, (y - 20) / 600);
    if (want < 0 && Math.hypot(x - HILL.x, z - hz) > 95) return;
    if (R() > Math.min(1, want * 3 + (Math.hypot(x - HILL.x, z - hz) < 95 ? 0.35 : 0))) return;
    const s = minS + (maxS - minS) * Math.pow(R(), 1.5);
    // keep the line between rings flyable
    if (corridor.length) { const [cd, cy] = corridorAt(x, z); if (cd < 26 && y + s > cy - 8) return; }
    // broadleaf trees on the low valley floor, spruce higher up
    const sp = y < 25 && R() < 0.45 ? 2 : R() < 0.5 ? 0 : 1;
    list.push([x, y - 0.3, z, sp === 2 ? s * 0.75 : s, R(), sp]);
  };
  const nNear = Math.round(38000 * quality), nFar = Math.round(30000 * quality);
  await slicer.stage(0.83, 0.915, () => `שותל יער: ${(nearT.length + farT.length).toLocaleString('he-IL')} עצים`);
  for (let i = 0; i < 300000 && nearT.length < nNear; i++) {
    if ((i & 255) === 0 && slicer.due()) await slicer.pause(0.6 * i / 300000);
    place(NEAR.x0 + R() * NEAR.w, NEAR.z0 + R() * NEAR.d, nearT, 9, 24);
  }
  for (let i = 0; i < 260000 && farT.length < nFar; i++) {
    if ((i & 255) === 0 && slicer.due()) await slicer.pause(0.6 + 0.4 * i / 260000);
    const x = -5000 + R() * 10000, z = -5000 + R() * 10000;
    if (inNear(x, z)) continue;
    place(x, z, farT, 12, 26);
  }
  const chunks = [];
  const mkInst = async (all, cast, cell, f0, f1) => {
    const groups = new Map();
    for (const e of all) {
      const k = `${Math.floor(e[0] / cell)},${Math.floor(e[2] / cell)},${e[5]}`;
      if (!groups.has(k)) groups.set(k, []);
      groups.get(k).push(e);
    }
    const m4 = new THREE.Matrix4(), q = new THREE.Quaternion(), col = new THREE.Color();
    let gi = 0;
    for (const [k, list] of groups) {
      if (slicer.due()) await slicer.pause(f0 + (f1 - f0) * gi / groups.size);
      gi++;
      const sp = list[0][5];
      const make = (geo) => {
        const im = new THREE.InstancedMesh(geo, treeMat, list.length);
        list.forEach(([x, y, z, s, r], i) => {
          q.setFromEuler(new THREE.Euler((r - 0.5) * 0.06, r * 6.28, (r * 7 % 1 - 0.5) * 0.06));
          const wdt = sp === 2 ? 1.2 + 0.5 * r : 0.75 + 0.35 * r;
          m4.compose(new THREE.Vector3(x, y, z), q, new THREE.Vector3(s * wdt, s, s * wdt));
          im.setMatrixAt(i, m4);
          const v = sunVisAt(x, z);
          const hue = r * 13 % 1;
          col.setRGB(0.9 + 0.25 * hue, 1, 0.85 + 0.3 * (1 - hue)).multiplyScalar((0.8 + 0.45 * r) * (0.45 + 0.55 * v));
          im.setColorAt(i, col);
        });
        im.computeBoundingSphere();
        im.receiveShadow = true;
        scene.add(im);
        return im;
      };
      const lo = make(speciesLo[sp]);
      const hi = cast ? make(species[sp]) : null;
      if (hi) { hi.castShadow = true; hi.visible = false; }
      const [cx, cz] = k.split(',').map((v) => (+v + 0.5) * cell);
      chunks.push({ cx, cz, hi, lo });
    }
  };
  await slicer.stage(0.915, 0.92, 'מסדר את היער בגושים לפי מרחק');
  await mkInst(nearT, true, 400, 0, 0.7);
  await mkInst(farT, false, 2500, 0.7, 1);
  const updateLOD = (cam) => {
    for (const c of chunks) {
      if (!c.hi) continue;
      // distance from the camera to the chunk's square (400 m cells)
      const dx = Math.max(0, Math.abs(cam.x - c.cx) - 200), dz = Math.max(0, Math.abs(cam.z - c.cz) - 200);
      const near = Math.hypot(dx, dz) < 160;
      c.hi.visible = near; c.lo.visible = !near;
    }
  };

  // boulders
  await slicer.stage(0.92, 0.93, 'מפזר סלעים');
  const rocks = [rockGeometry(1), rockGeometry(2), rockGeometry(3)];
  const rockMat = new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.92 });
  rocks.forEach((rg, ri) => {
    const list = [];
    for (let i = 0; i < 3000 && list.length < 700; i++) {
      const hz = valleyZ(HILL.x) + HILL.dz;
      let x, z;
      if (i % 3 === 0) { const a = R() * 6.28, r = R() * 120; x = HILL.x + Math.cos(a) * r; z = hz + Math.sin(a) * r; }
      else { x = NEAR.x0 + R() * NEAR.w; z = NEAR.z0 + R() * NEAR.d; }
      const y = hAt(x, z);
      if (y < WATER_Y - 0.5 || Math.hypot(x - PAD[0], z - PAD[2]) < 25 || Math.hypot(x - LAND.x, z - LAND.z) < 22) continue;
      list.push([x, y, z, 0.4 + 2.8 * Math.pow(R(), 3), R()]);
    }
    const im = new THREE.InstancedMesh(rg, rockMat, list.length);
    const m4 = new THREE.Matrix4(), q = new THREE.Quaternion(), col = new THREE.Color();
    list.forEach(([x, y, z, s, r], i) => {
      q.setFromEuler(new THREE.Euler(r * 0.5, r * 6.28 * (ri + 1), r * 0.3));
      m4.compose(new THREE.Vector3(x, y - s * 0.25, z), q, new THREE.Vector3(s, s, s));
      im.setMatrixAt(i, m4);
      col.setScalar((0.8 + 0.4 * r) * (0.45 + 0.55 * sunVisAt(x, z)));
      im.setColorAt(i, col);
    });
    im.castShadow = true; im.receiveShadow = true;
    scene.add(im);
  });

  scene.add(makePad());

  // blast-reactive grass around the landing meadow and the pad apron
  const grassU = {
    uBlast: { value: Array.from({ length: 5 }, () => new THREE.Vector4(0, 0, 0, 1)) },
    uTime: { value: 0 }, uWind: { value: new THREE.Vector3() },
  };
  // two separate patches so each is frustum-culled on its own
  await slicer.stage(0.93, 0.972, 'מגדל עשב בשדה הנחיתה');
  scene.add(await makeGrass([{ x: LAND.x, z: LAND.z, r0: 0, r1: 38, count: Math.round(90000 * quality) }], grassU, slicer));
  await slicer.stage(0.972, 1, 'מגדל עשב סביב רחבת ההמראה');
  scene.add(await makeGrass([{ x: PAD[0], z: PAD[2], r0: 17, r1: 42, count: Math.round(60000 * quality) }], grassU, slicer));

  return { sun, water, sky, grassU, sunVisAt, hAt, near, envTex: env, trees: nearT, updateLOD };
}
