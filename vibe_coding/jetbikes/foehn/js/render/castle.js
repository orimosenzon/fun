// Draws the fortress from the same list of solids the bike collides with (js/sim/castle.js).
// Everything is merged into two meshes (stone, slate roofs); the masonry is a world-space
// tri-planar texture, so blocks line up across pieces without any UV work.
import * as THREE from '../../lib/three.module.min.js';
import { C } from '../sim/castle.js';
import { SITE } from '../sim/castleSite.js';
import { U } from './env.js';

// ---------- a tiny mesh builder: flat or smooth quads and triangles with a colour ----------
class Builder {
  constructor() { this.p = []; this.n = []; this.c = []; }
  tri(a, b, c, col, hint, nrm) {
    let n = [(b[1] - a[1]) * (c[2] - a[2]) - (b[2] - a[2]) * (c[1] - a[1]), (b[2] - a[2]) * (c[0] - a[0]) - (b[0] - a[0]) * (c[2] - a[2]), (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])];
    const l = Math.hypot(n[0], n[1], n[2]);
    if (l < 1e-9) return;
    n = [n[0] / l, n[1] / l, n[2] / l];
    let v = [a, b, c], ns = nrm;
    if (hint && n[0] * hint[0] + n[1] * hint[1] + n[2] * hint[2] < 0) { v = [a, c, b]; n = [-n[0], -n[1], -n[2]]; if (ns) ns = [ns[0], ns[2], ns[1]]; }
    for (let k = 0; k < 3; k++) { this.p.push(v[k][0], v[k][1], v[k][2]); const q = ns ? ns[k] : n; this.n.push(q[0], q[1], q[2]); this.c.push(col[0], col[1], col[2]); }
  }
  quad(a, b, c, d, col, hint, nrm) { this.tri(a, b, c, col, hint, nrm && [nrm[0], nrm[1], nrm[2]]); this.tri(a, c, d, col, hint, nrm && [nrm[0], nrm[2], nrm[3]]); }
  geometry(ox, oy, oz) {
    const g = new THREE.BufferGeometry(), p = new Float32Array(this.p);
    for (let i = 0; i < p.length; i += 3) { p[i] += ox; p[i + 1] += oy; p[i + 2] += oz; }
    g.setAttribute('position', new THREE.BufferAttribute(p, 3));
    g.setAttribute('normal', new THREE.BufferAttribute(new Float32Array(this.n), 3));
    g.setAttribute('color', new THREE.BufferAttribute(new Float32Array(this.c), 3));
    g.computeBoundingSphere(); g.computeBoundingBox();
    return g;
  }
}

function addBox(B, s, col) {
  const { x0, x1, y0, y1, z0, z1 } = s;
  B.quad([x1, y0, z1], [x1, y0, z0], [x1, y1, z0], [x1, y1, z1], col, [1, 0, 0]);
  B.quad([x0, y0, z0], [x0, y0, z1], [x0, y1, z1], [x0, y1, z0], col, [-1, 0, 0]);
  B.quad([x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1], col, [0, 0, 1]);
  B.quad([x1, y0, z0], [x0, y0, z0], [x0, y1, z0], [x1, y1, z0], col, [0, 0, -1]);
  B.quad([x0, y1, z1], [x1, y1, z1], [x1, y1, z0], [x0, y1, z0], col, [0, 1, 0]);
  B.quad([x0, y0, z0], [x1, y0, z0], [x1, y0, z1], [x0, y0, z1], col, [0, -1, 0]);
}
function addCyl(B, s, col) {
  const n = Math.max(10, Math.min(32, Math.round(Math.max(s.r, s.r1) * 2.2))), dy = s.y1 - s.y0, sl = s.r - s.r1, nl = Math.hypot(dy, sl);
  for (let k = 0; k < n; k++) {
    const a0 = k / n * 6.28319, a1 = (k + 1) / n * 6.28319, c0 = Math.cos(a0), s0 = Math.sin(a0), c1 = Math.cos(a1), s1 = Math.sin(a1);
    const n0 = [c0 * dy / nl, sl / nl, s0 * dy / nl], n1 = [c1 * dy / nl, sl / nl, s1 * dy / nl], mid = [(c0 + c1), 0.2 * Math.sign(sl), (s0 + s1)];
    const A = [s.x + c0 * s.r, s.y0, s.z + s0 * s.r], Bq = [s.x + c1 * s.r, s.y0, s.z + s1 * s.r], Cq = [s.x + c1 * s.r1, s.y1, s.z + s1 * s.r1], Dq = [s.x + c0 * s.r1, s.y1, s.z + s0 * s.r1];
    if (s.r1 > 0.01) B.quad(A, Bq, Cq, Dq, col, mid, [n0, n1, n1, n0]); else B.tri(A, Bq, [s.x, s.y1, s.z], col, mid, [n0, n1, [(n0[0] + n1[0]) / 2, n0[1], (n0[2] + n1[2]) / 2]]);
    if (s.r1 > 0.01) B.tri([s.x, s.y1, s.z], Dq, Cq, col, [0, 1, 0]);
  }
}
function addGable(B, s, col) {
  const { x0, x1, y0, y1, z0, z1 } = s;
  if (s.axis === 'x') {
    const c = (z0 + z1) / 2;
    B.quad([x0, y0, z0], [x1, y0, z0], [x1, y1, c], [x0, y1, c], col, [0, 1, -1]); B.quad([x0, y0, z1], [x1, y0, z1], [x1, y1, c], [x0, y1, c], col, [0, 1, 1]);
    B.tri([x0, y0, z0], [x0, y0, z1], [x0, y1, c], col, [-1, 0, 0]); B.tri([x1, y0, z0], [x1, y0, z1], [x1, y1, c], col, [1, 0, 0]);
  } else {
    const c = (x0 + x1) / 2;
    B.quad([x0, y0, z0], [x0, y0, z1], [c, y1, z1], [c, y1, z0], col, [-1, 1, 0]); B.quad([x1, y0, z0], [x1, y0, z1], [c, y1, z1], [c, y1, z0], col, [1, 1, 0]);
    B.tri([x0, y0, z0], [x1, y0, z0], [c, y1, z0], col, [0, 0, -1]); B.tri([x0, y0, z1], [x1, y0, z1], [c, y1, z1], col, [0, 0, 1]);
  }
  B.quad([x0, y0, z0], [x1, y0, z0], [x1, y0, z1], [x0, y0, z1], col, [0, -1, 0]);
}
function addPyr(B, s, col) {
  const { x0, x1, y0, y1, z0, z1 } = s, t = [(x0 + x1) / 2, y1, (z0 + z1) / 2];
  B.tri([x0, y0, z0], [x1, y0, z0], t, col, [0, 1, -1]); B.tri([x0, y0, z1], [x1, y0, z1], t, col, [0, 1, 1]);
  B.tri([x0, y0, z0], [x0, y0, z1], t, col, [-1, 1, 0]); B.tri([x1, y0, z0], [x1, y0, z1], t, col, [1, 1, 0]);
  B.quad([x0, y0, z0], [x1, y0, z0], [x1, y0, z1], [x0, y0, z1], col, [0, -1, 0]);
}
// block with an arched tunnel: (u, y, w) = across, up, along the tunnel
function addArch(B, s, col, dark) {
  const ax = s.axis === 'x';
  const M = (u, y, w) => (ax ? [w, y, u] : [u, y, w]), Vn = (u, y, w) => (ax ? [w, y, u] : [u, y, w]);
  const u0 = ax ? s.z0 : s.x0, u1 = ax ? s.z1 : s.x1, w0 = ax ? s.x0 : s.z0, w1 = ax ? s.x1 : s.z1, { y0, y1, c, hw, ys } = s;
  const sill = s.yb !== undefined, yb = sill ? s.yb : y0, N = 14;
  const pts = []; for (let k = 0; k <= N; k++) { const a = Math.PI * k / N; pts.push([c + hw * Math.cos(a), ys + hw * Math.sin(a), Math.cos(a), Math.sin(a)]); }
  for (const [w, sg] of [[w0, -1], [w1, 1]]) {
    const h = Vn(0, 0, sg);
    B.quad(M(u0, y0, w), M(c - hw, y0, w), M(c - hw, y1, w), M(u0, y1, w), col, h);
    B.quad(M(c + hw, y0, w), M(u1, y0, w), M(u1, y1, w), M(c + hw, y1, w), col, h);
    if (sill) B.quad(M(c - hw, y0, w), M(c + hw, y0, w), M(c + hw, yb, w), M(c - hw, yb, w), col, h);
    for (let k = 0; k < N; k++) B.quad(M(pts[k][0], pts[k][1], w), M(pts[k + 1][0], pts[k + 1][1], w), M(pts[k + 1][0], y1, w), M(pts[k][0], y1, w), col, h);
  }
  B.quad(M(u0, y0, w0), M(u0, y0, w1), M(u0, y1, w1), M(u0, y1, w0), col, Vn(-1, 0, 0));
  B.quad(M(u1, y0, w0), M(u1, y0, w1), M(u1, y1, w1), M(u1, y1, w0), col, Vn(1, 0, 0));
  B.quad(M(u0, y1, w0), M(u1, y1, w0), M(u1, y1, w1), M(u0, y1, w1), col, [0, 1, 0]);
  B.quad(M(u0, y0, w0), M(c - hw, y0, w0), M(c - hw, y0, w1), M(u0, y0, w1), col, [0, -1, 0]);
  B.quad(M(c + hw, y0, w0), M(u1, y0, w0), M(u1, y0, w1), M(c + hw, y0, w1), col, [0, -1, 0]);
  // inside the tunnel: jambs, sill, soffit (a little darker: less sky reaches them)
  if (ys > yb) {
    B.quad(M(c - hw, yb, w0), M(c - hw, yb, w1), M(c - hw, ys, w1), M(c - hw, ys, w0), dark, Vn(1, 0, 0));
    B.quad(M(c + hw, yb, w0), M(c + hw, yb, w1), M(c + hw, ys, w1), M(c + hw, ys, w0), dark, Vn(-1, 0, 0));
  }
  if (sill) B.quad(M(c - hw, yb, w0), M(c + hw, yb, w0), M(c + hw, yb, w1), M(c - hw, yb, w1), dark, [0, 1, 0]);
  for (let k = 0; k < N; k++) {
    const a = pts[k], b = pts[k + 1], na = Vn(-a[2], -a[3], 0), nb = Vn(-b[2], -b[3], 0);
    B.quad(M(a[0], a[1], w0), M(a[0], a[1], w1), M(b[0], b[1], w1), M(b[0], b[1], w0), dark, [(na[0] + nb[0]) / 2, (na[1] + nb[1]) / 2, (na[2] + nb[2]) / 2], [na, na, nb, nb]);
  }
}

// ---------- textures ----------
function canvasTex(w, h, draw) {
  const c = document.createElement('canvas'); c.width = w; c.height = h; draw(c.getContext('2d'), w, h);
  const t = new THREE.CanvasTexture(c); t.colorSpace = THREE.SRGBColorSpace; t.anisotropy = 8; t.wrapS = t.wrapT = THREE.RepeatWrapping;
  return t;
}
function rng(seed) { let s = seed; return () => ((s = (s * 1664525 + 1013904223) >>> 0) / 4294967296); }
// ashlar masonry: 8 courses, blocks of uneven length, 6.4 m to the tile
function stoneTex() {
  return canvasTex(512, 512, (g, w, h) => {
    const r = rng(77);
    g.fillStyle = '#5b5549'; g.fillRect(0, 0, w, h);
    for (let row = 0; row < 8; row++) {
      // block lengths that add up to the tile exactly, so the pattern wraps without a seam
      const ws = []; let sum = 0; while (sum < w - 60) { const bw = 70 + r() * 80; ws.push(bw); sum += bw; }
      let x = r() * 90;
      for (const b0 of ws) {
        const bw = b0 * w / sum, v = 0.84 + 0.2 * r(), warm = r();
        for (const ox of [0, w]) {
          g.fillStyle = `rgb(${Math.round(203 * v + 10 * warm)},${Math.round(190 * v + 3 * warm)},${Math.round(165 * v - 9 * warm)})`;
          g.fillRect(x - ox + 2, row * 64 + 2, bw - 3, 60);
          g.fillStyle = 'rgba(255,255,255,0.10)'; g.fillRect(x - ox + 2, row * 64 + 2, bw - 3, 3);
          g.fillStyle = 'rgba(0,0,0,0.13)'; g.fillRect(x - ox + 2, row * 64 + 57, bw - 3, 5);
        }
        x += bw;
      }
    }
    for (let i = 0; i < 9000; i++) { const a = r() * 0.10; g.fillStyle = r() < 0.5 ? `rgba(0,0,0,${a})` : `rgba(255,250,235,${a})`; g.fillRect(r() * w, r() * h, 1 + r() * 2.5, 1 + r() * 2.5); }
    for (let i = 0; i < 26; i++) { g.fillStyle = `rgba(60,55,40,${0.03 + r() * 0.05})`; const x = r() * w; g.fillRect(x, r() * h, 3 + r() * 8, 40 + r() * 140); }
  });
}
// slate: staggered rows, 3.2 m to the tile
function slateTex() {
  return canvasTex(256, 256, (g, w, h) => {
    const r = rng(5);
    g.fillStyle = '#20262f'; g.fillRect(0, 0, w, h);
    for (let row = 0; row < 8; row++) for (let k = -1; k < 9; k++) {
      const v = 0.8 + 0.4 * r();
      g.fillStyle = `rgb(${Math.round(72 * v)},${Math.round(88 * v)},${Math.round(112 * v)})`;
      g.fillRect(k * 32 + (row % 2) * 16 + 1, row * 32 + 1, 30, 30);
      g.fillStyle = 'rgba(0,0,0,0.28)'; g.fillRect(k * 32 + (row % 2) * 16 + 1, row * 32 + 25, 30, 6);
    }
  });
}
// MeshStandardMaterial whose colour comes from a texture projected along the three world axes
function triplanar(map, scale, opt, weather) {
  const m = new THREE.MeshStandardMaterial({ vertexColors: true, ...opt });
  m.onBeforeCompile = (sh) => {
    sh.uniforms.tMap = { value: map }; sh.uniforms.tScale = { value: scale }; sh.uniforms.uNoise = U.uNoise; sh.uniforms.uBaseY = { value: C.y0 };
    sh.vertexShader = sh.vertexShader.replace('#include <common>', '#include <common>\nvarying vec3 vWp; varying vec3 vWn;')
      .replace('#include <begin_vertex>', '#include <begin_vertex>\nvWp = (modelMatrix * vec4(transformed, 1.0)).xyz; vWn = normalize(mat3(modelMatrix) * objectNormal);');
    sh.fragmentShader = sh.fragmentShader.replace('#include <common>', '#include <common>\nvarying vec3 vWp; varying vec3 vWn; uniform sampler2D tMap; uniform sampler2D uNoise; uniform float tScale; uniform float uBaseY;')
      .replace('#include <map_fragment>', `
        vec3 bw = pow(abs(normalize(vWn)), vec3(5.0)); bw /= bw.x + bw.y + bw.z;
        vec3 tx = texture2D(tMap, vWp.zy * tScale).rgb * bw.x + texture2D(tMap, vWp.xz * tScale).rgb * bw.y + texture2D(tMap, vWp.xy * tScale).rgb * bw.z;
        ${weather ? `float mac = texture2D(uNoise, (vWp.xz + vWp.y * vec2(0.7, 0.45)) / 5200.0).r * 0.6 + texture2D(uNoise, (vWp.zy + vWp.x * 0.6) / 1500.0).r * 0.4;
        tx *= 0.80 + 0.36 * mac;
        float hgt = vWp.y - uBaseY;
        tx *= mix(vec3(0.72, 0.69, 0.63), vec3(1.0), smoothstep(-1.0, 7.0 + 6.0 * mac, hgt));` : ''}
        diffuseColor.rgb *= tx;`);
  };
  return m;
}

export function buildCastleMesh(scene) {
  const group = new THREE.Group(); scene.add(group);
  const ox = SITE.x, oy = C.y0, oz = SITE.z;
  const stone = new Builder(), roof = new Builder(), glow = new Builder(), fire = new Builder();
  const tone = (s, k = 1) => { const t = (s.tone || 1) * k; return [t, t, t * (s.m === 'roof' ? 1 : 0.985)]; };
  for (const s of C.solids) {
    const B = s.m === 'roof' ? roof : stone, col = tone(s);
    if (s.t === 'box') addBox(B, s, col);
    else if (s.t === 'cyl') addCyl(B, s, col);
    else if (s.t === 'gable') addGable(B, s, col);
    else if (s.t === 'pyr') addPyr(B, s, col);
    else if (s.t === 'arch') addArch(B, s, col, tone(s, 0.6));
  }
  const flags = [];
  for (const d of C.deco) {
    if (d.t === 'box') addBox(stone, d, tone(d));
    else if (d.t === 'pave') stone.quad([d.x0, 0.07, d.z1], [d.x1, 0.07, d.z1], [d.x1, 0.07, d.z0], [d.x0, 0.07, d.z0], [0.5, 0.5, 0.52], [0, 1, 0]);
    else if (d.t === 'glow') {
      const tx = -d.nz * d.w / 2, tz = d.nx * d.w / 2, e = 0.04, x = d.x + d.nx * e, z = d.z + d.nz * e, hint = [d.nx, 0, d.nz], n = 5, r = d.w / 2, yb = d.y - d.h / 2, yt = d.y + d.h / 2 - r;
      glow.quad([x - tx, yb, z - tz], [x + tx, yb, z + tz], [x + tx, yt, z + tz], [x - tx, yt, z - tz], [1, 1, 1], hint);
      for (let k = 0; k < n; k++) { const a0 = Math.PI * k / n, a1 = Math.PI * (k + 1) / n, f0 = Math.cos(a0), f1 = Math.cos(a1); glow.tri([x, yt, z], [x + tx * f0, yt + r * Math.sin(a0), z + tz * f0], [x + tx * f1, yt + r * Math.sin(a1), z + tz * f1], [1, 1, 1], hint); }
      // a stone sill under it
      addBox(stone, { x0: d.x - Math.abs(tx) - 0.25 - Math.abs(d.nx) * 0.0, x1: d.x + Math.abs(tx) + 0.25, y0: yb - 0.45, y1: yb, z0: d.z - Math.abs(tz) - 0.25, z1: d.z + Math.abs(tz) + 0.25 }, [1.05, 1.05, 1.03]);
    } else if (d.t === 'torch') {
      addBox(stone, { x0: d.x - 0.25, x1: d.x + 0.25, y0: d.y - 1.1, y1: d.y - 0.5, z0: d.z - 0.25, z1: d.z + 0.25 }, [0.2, 0.2, 0.2]);
      addCyl(fire, { x: d.x, z: d.z, r: 0.1, r1: 0.24, y0: d.y - 0.5, y1: d.y - 0.2 }, [1, 1, 1]); addCyl(fire, { x: d.x, z: d.z, r: 0.24, r1: 0, y0: d.y - 0.2, y1: d.y + 0.5 }, [1, 1, 1]);
    } else if (d.t === 'flag') {
      addBox(stone, { x0: d.x - 0.14, x1: d.x + 0.14, y0: d.y, y1: d.y + d.h, z0: d.z - 0.14, z1: d.z + 0.14 }, [0.25, 0.24, 0.22]);
      flags.push(d);
    }
  }
  const stoneMat = triplanar(stoneTex(), 1 / 6.4, { roughness: 0.94, metalness: 0 }, true);
  const roofMat = triplanar(slateTex(), 1 / 3.2, { roughness: 0.62, metalness: 0.08 }, false);
  for (const [B, mat] of [[stone, stoneMat], [roof, roofMat]]) { const m = new THREE.Mesh(B.geometry(ox, oy, oz), mat); m.castShadow = true; m.receiveShadow = true; group.add(m); }
  const glowMat = new THREE.MeshBasicMaterial({ color: 0x10151c, toneMapped: false });
  group.add(new THREE.Mesh(glow.geometry(ox, oy, oz), glowMat));
  const fireMat = new THREE.MeshBasicMaterial({ color: new THREE.Color(5, 1.5, 0.25), toneMapped: false });
  group.add(new THREE.Mesh(fire.geometry(ox, oy, oz), fireMat));
  // banners
  const flagMat = new THREE.MeshStandardMaterial({ color: 0xa3121a, roughness: 0.8, side: THREE.DoubleSide });
  const flagObjs = flags.map((d, i) => {
    const w = d.s, h = d.s * 0.55, g = new THREE.PlaneGeometry(w, h, 10, 2); g.translate(w / 2, 0, 0);
    const m = new THREE.Mesh(g, flagMat); m.position.set(ox + d.x, oy + d.y + d.h - h / 2 - 0.2, oz + d.z); m.castShadow = true; group.add(m);
    return { m, w, base: g.attributes.position.array.slice(), ph: i * 1.7 };
  });
  return {
    group,
    setNight(k) { glowMat.color.setRGB(0.012 + 2.6 * k, 0.016 + 1.45 * k, 0.022 + 0.42 * k); },
    update(t, wind) {
      const f = 0.85 + 0.15 * Math.sin(t * 17) * Math.sin(t * 7.3);
      fireMat.color.setRGB(5 * f, 1.5 * f, 0.25 * f);
      const yaw = wind ? Math.atan2(-wind[2], wind[0]) : 0.4;
      for (const o of flagObjs) {
        const p = o.m.geometry.attributes.position, a = p.array;
        for (let i = 0; i < a.length; i += 3) { const u = o.base[i] / o.w; a[i + 2] = Math.sin(u * 5.5 - t * 6 + o.ph) * 0.11 * o.w * u; a[i + 1] = o.base[i + 1] - 0.07 * o.w * u * u + Math.sin(u * 4 - t * 4.4 + o.ph) * 0.03 * o.w * u; }
        p.needsUpdate = true; o.m.rotation.y = yaw + 0.25 * Math.sin(t * 0.6 + o.ph);
      }
    },
  };
}
