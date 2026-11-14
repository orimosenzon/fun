// Draws the fortress from the same list of solids the bike collides with (js/sim/castle.js), and
// its furnishings from the list of ornaments. Everything is merged into one mesh per material.
// The masonry is a world-space tri-planar texture, so stones line up across pieces without any UV
// work; age is painted over it in the shader (rain streaks, damp, lichen, moss, ivy, old lime
// render flaking off). The torches really light what is near them.
import * as THREE from '../../lib/three.module.min.js';
import { C } from '../sim/castle.js';
import { SITE } from '../sim/castleSite.js';
import { U } from './env.js';
import { Builder, addBox, addCyl, addGable, addPyr, addArch } from './meshkit.js';
import { stoneTex, slateTex, fachTex, woodTex, atlasTex, haloTex } from './castleTex.js';
import { buildProp, shutters } from './castleProps.js';

const NL = 24;                                                       // lights the shader takes at once: the nearest ones
const NR = 8;                                                       // rooms: boxes where the sky does not reach
const ROOMA = { value: [] }, ROOMB = { value: [] }, ORG = { value: new THREE.Vector3() };
const TORCH = { value: Array.from({ length: NL }, () => new THREE.Vector4(0, 0, 0, 0)) }, TORCHK = { value: 1 };

// MeshStandardMaterial plus the light of the nearest fires; with `tri`, its colour comes from a
// texture projected along the three world axes and weathered according to `tri.mode`
function castleMat(opt, tri, wrap = 0.08, gain = 1) {
  const m = new THREE.MeshStandardMaterial(opt), mode = tri && tri.mode;
  // three.js shares a compiled program between materials whose settings look alike, and cannot see
  // that these differ in the code added below: tell it
  m.customProgramCacheKey = () => `castle|${mode || 'plain'}|${wrap}|${gain}|${tri ? tri.bump : 0}`;
  m.onBeforeCompile = (sh) => {
    sh.uniforms.uTorch = TORCH; sh.uniforms.uTorchK = TORCHK; sh.uniforms.uOrg = ORG; sh.uniforms.uRoomA = ROOMA; sh.uniforms.uRoomB = ROOMB;
    if (tri) { sh.uniforms.tMap = { value: tri.map }; sh.uniforms.tScale = { value: tri.scale }; sh.uniforms.uNoise = U.uNoise; }
    sh.vertexShader = sh.vertexShader.replace('#include <common>', '#include <common>\nvarying vec3 vWp; varying vec3 vWn; varying float vW; attribute float aW;')
      .replace('#include <begin_vertex>', '#include <begin_vertex>\nvWp = (modelMatrix * vec4(transformed, 1.0)).xyz; vWn = normalize(mat3(modelMatrix) * objectNormal); vW = aW;');
    let f = sh.fragmentShader.replace('#include <common>', `#include <common>
      varying vec3 vWp; varying vec3 vWn; varying float vW;
      uniform vec4 uTorch[${NL}]; uniform float uTorchK; uniform vec3 uOrg; uniform vec3 uRoomA[${NR}]; uniform vec3 uRoomB[${NR}];
      // 1 deep inside a hall, 0 in the open; it rises over two metres from every opening
      float indoor(vec3 p) { float k = 0.0; for (int i = 0; i < ${NR}; i++) { vec3 q = min(p - uRoomA[i], uRoomB[i] - p); k = max(k, clamp(min(q.x, min(q.y, q.z)) * 0.5, 0.0, 1.0)); } return k; }
      vec3 torchLight(vec3 n, float wrap) {
        float sum = 0.0;
        for (int i = 0; i < ${NL}; i++) {
          float w = uTorch[i].w; if (w <= 0.0) continue;
          vec3 L = uTorch[i].xyz - vWp; float d2 = dot(L, L); if (d2 > 1024.0) continue;
          float nl = max(dot(n, L) * inversesqrt(d2 + 1e-3), 0.0), fo = 1.0 - d2 / 1024.0;
          sum += w * mix(nl, 1.0, wrap) * fo * fo / (1.0 + d2 * 0.10);
        }
        return sum * vec3(1.0, 0.55, 0.19);
      }
      ${tri ? 'uniform sampler2D tMap; uniform sampler2D uNoise; uniform float tScale;\nfloat nz(vec2 p, float cell) { return texture2D(uNoise, p / (256.0 * cell)).r; }\nfloat n3(vec3 p, vec3 bw, float cell) { return nz(p.zy, cell) * bw.x + nz(p.xz, cell) * bw.y + nz(p.xy, cell) * bw.z; }' : ''}`)
      .replace('#include <lights_fragment_end>', `#include <lights_fragment_end>
        float ind = indoor(vWp - uOrg), occ = 1.0 - 0.66 * ind, occD = 1.0 - 0.85 * ind;      // the sun only finds its way in near the openings
        reflectedLight.directDiffuse *= occD; reflectedLight.indirectDiffuse *= occ; reflectedLight.directSpecular *= occD; reflectedLight.indirectSpecular *= occ;
        reflectedLight.directDiffuse += diffuseColor.rgb * torchLight(inverseTransformDirection(normal, viewMatrix), ${wrap.toFixed(2)}) * uTorchK * ${gain.toFixed(2)};`);
    if (tri) {
      f = f.replace('#include <map_fragment>', `
        vec3 P = vWp - uOrg; vec3 nW = normalize(vWn); vec3 an = abs(nW);
        vec3 bw = pow(an, vec3(5.0)); bw /= bw.x + bw.y + bw.z;
        vec3 tx = texture2D(tMap, P.zy * tScale).rgb * bw.x + texture2D(tMap, P.xz * tScale).rgb * bw.y + texture2D(tMap, P.xy * tScale).rgb * bw.z;
        float tH = dot(tx, vec3(0.333));
        float side = 1.0 - an.y, hgt = P.y, fc = P.x + P.z, wx = vW * (1.0 - indoor(P));
        float mac = n3(P, bw, 20.0) * 0.6 + n3(P + 31.0, bw, 6.0) * 0.4;
        float fine = n3(P + 7.0, bw, 0.45);
        ${mode === 'plaster' ? `
        // lime render: it has fallen off in patches and the stone shows through
        float pm = nz(vec2(fc, P.y * 1.3), 7.0) * 0.55 + nz(vec2(fc * 1.3, P.y), 1.7) * 0.3 + fine * 0.15;
        float pk = smoothstep(0.42, 0.48, pm) * side;
        tx = mix(tx, vec3(0.60, 0.53, 0.41) * (0.80 + 0.34 * fine), pk); tH = mix(tH, 0.6, pk);` : ''}
        tx *= 0.72 + 0.38 * mac;
        ${mode === 'roof' ? `
        float mo = n3(P + 53.0, bw, 2.6) * 0.6 + fine * 0.4;
        tx = mix(tx, vec3(0.045, 0.062, 0.024) * (0.6 + 0.8 * fine), smoothstep(0.60, 0.74, mo + 0.08 * max(-nW.z, 0.0)) * 0.6);
        tx = mix(tx, vec3(0.22, 0.20, 0.10), smoothstep(0.74, 0.82, n3(P + 91.0, bw, 0.9)) * 0.3);` : `
        // where the rain has run down for centuries
        float st = texture2D(uNoise, vec2(fc / 330.0, P.y / 3600.0)).g;
        tx *= mix(1.0, 0.58 + 0.42 * smoothstep(0.30, 0.68, st), side * (0.35 + 0.65 * wx));
        // damp at the foot of the walls
        tx *= mix(vec3(0.64, 0.62, 0.56), vec3(1.0), smoothstep(-1.0, 7.0 + 6.0 * mac, hgt));`}
        ${mode === 'stone' || mode === 'plaster' ? `
        float li = n3(P + 17.0, bw, 0.9);
        tx = mix(tx, vec3(0.24, 0.235, 0.13) * (0.7 + 0.6 * fine), smoothstep(0.70, 0.80, li) * 0.4 * wx);
        float mo = n3(P + 53.0, bw, 3.4) * 0.65 + fine * 0.35;
        float mk = smoothstep(0.52, 0.66, mo + 0.22 * max(nW.y, 0.0) + 0.10 * max(-nW.z, 0.0) - 0.006 * max(hgt, 0.0) * side) * wx;
        tx = mix(tx, vec3(0.042, 0.058, 0.022) * (0.6 + 0.9 * fine), mk * 0.7);
        // ivy: it climbs from the ground in some stretches of wall, higher in some than in others
        float leaf = nz(vec2(fc, P.y), 0.2);
        float ivF = nz(vec2(fc, P.y * 0.3), 7.0) * 0.72 + nz(vec2(fc + 40.0, P.y * 0.6), 1.6) * 0.28;
        float ivm = smoothstep(0.0, 0.05, ivF - 0.47 - max(hgt, 0.0) / 50.0 + 0.09 * (leaf - 0.5)) * side * smoothstep(0.7, 1.0, wx);
        tx = mix(tx, mix(vec3(0.007, 0.017, 0.005), vec3(0.028, 0.058, 0.013), leaf * leaf * 1.6), ivm); tH = mix(tH, leaf * 0.9, ivm);` : ''}
        diffuseColor.rgb *= tx;`)
        // the joints between the stones catch the light: a bump from the brightness of the texture
        .replace('#include <normal_fragment_maps>', `#include <normal_fragment_maps>
        {
          vec3 sX = dFdx(-vViewPosition), sY = dFdy(-vViewPosition), vN = normal;
          vec3 R1 = cross(sY, vN), R2 = cross(vN, sX); float fDet = dot(sX, R1);
          vec2 dH = vec2(dFdx(tH), dFdy(tH)) * ${(tri.bump || 0.07).toFixed(3)} * smoothstep(240.0, 40.0, length(vViewPosition));
          normal = normalize(abs(fDet) * vN - sign(fDet) * (dH.x * R1 + dH.y * R2));
        }`);
    }
    sh.fragmentShader = f;
  };
  return m;
}

const shape = (B, s, col, dark) => {
  if (s.t === 'box') addBox(B, s, col); else if (s.t === 'cyl') addCyl(B, s, col); else if (s.t === 'gable') addGable(B, s, col); else if (s.t === 'pyr') addPyr(B, s, col); else if (s.t === 'arch') addArch(B, s, col, dark);
};

export function buildCastleMesh(scene) {
  const group = new THREE.Group(); scene.add(group);
  const ox = SITE.x, oy = C.y0, oz = SITE.z;
  const K = { stone: new Builder(), plaster: new Builder(), roof: new Builder(), fach: new Builder(), wood: new Builder(), plain: new Builder(), metal: new Builder(), cloth: new Builder(true), glass: new Builder(true), fire: new Builder(), lights: [] };
  const glow = new Builder();
  const tone = (s, k = 1) => { const t = (s.tone || 1) * k; return [t, t, t * (s.m === 'roof' ? 1 : 0.985)]; };
  const METAL = { iron: [0.10, 0.10, 0.11], gold: [0.95, 0.70, 0.24] };
  const put = (s) => {
    if (s.m === 'hide') return;
    if (METAL[s.m]) { shape(K.metal, s, METAL[s.m]); return; }
    const B = K[s.m] || K.stone; B.wv = s.w !== undefined ? s.w : 1;
    shape(B, s, tone(s), tone(s, 0.82)); B.wv = 1;
  };
  for (const s of C.solids) put(s);
  const flags = [];
  for (const d of C.deco) {
    if (d.t === 'box' || d.t === 'gable' || d.t === 'cyl' || d.t === 'pyr') put(d);
    else if (d.t === 'pave') { K.stone.wv = 0.3; K.stone.quad([d.x0, 0.07, d.z1], [d.x1, 0.07, d.z1], [d.x1, 0.07, d.z0], [d.x0, 0.07, d.z0], [0.5, 0.5, 0.52], [0, 1, 0]); K.stone.wv = 1; }
    else if (d.t === 'glow') {
      const tx = -d.nz * d.w / 2, tz = d.nx * d.w / 2, e = 0.04, x = d.x + d.nx * e, z = d.z + d.nz * e, hint = [d.nx, 0, d.nz], n = 5, r = d.w / 2, yb = d.y - d.h / 2, yt = d.y + d.h / 2 - r;
      glow.quad([x - tx, yb, z - tz], [x + tx, yb, z + tz], [x + tx, yt, z + tz], [x - tx, yt, z - tz], [1, 1, 1], hint);
      for (let k = 0; k < n; k++) { const a0 = Math.PI * k / n, a1 = Math.PI * (k + 1) / n, f0 = Math.cos(a0), f1 = Math.cos(a1); glow.tri([x, yt, z], [x + tx * f0, yt + r * Math.sin(a0), z + tz * f0], [x + tx * f1, yt + r * Math.sin(a1), z + tz * f1], [1, 1, 1], hint); }
      // a stone sill under it
      addBox(K.stone, { x0: d.x - Math.abs(tx) - 0.25, x1: d.x + Math.abs(tx) + 0.25, y0: yb - 0.45, y1: yb, z0: d.z - Math.abs(tz) - 0.25, z1: d.z + Math.abs(tz) + 0.25 }, [1.05, 1.05, 1.03]);
      if (d.sh) shutters(d, K);
    } else if (d.t === 'flag') {
      addBox(K.metal, { x0: d.x - 0.14, x1: d.x + 0.14, y0: d.y, y1: d.y + d.h, z0: d.z - 0.14, z1: d.z + 0.14 }, METAL.iron);
      flags.push(d);
    } else buildProp(d, K);
  }
  ORG.value.set(ox, oy, oz);
  ROOMA.value = []; ROOMB.value = [];
  for (let i = 0; i < NR; i++) { const r = C.rooms[i] || [0, 0, 0, 0, 0, 0]; ROOMA.value.push(new THREE.Vector3(r[0], r[2], r[4])); ROOMB.value.push(new THREE.Vector3(r[1], r[3], r[5])); }
  const stoneMap = stoneTex();
  const mats = {
    stone: castleMat({ vertexColors: true, roughness: 0.95, metalness: 0 }, { map: stoneMap, scale: 1 / 6.4, mode: 'stone', bump: 0.09 }),
    plaster: castleMat({ vertexColors: true, roughness: 0.95, metalness: 0 }, { map: stoneMap, scale: 1 / 6.4, mode: 'plaster', bump: 0.09 }),
    roof: castleMat({ vertexColors: true, roughness: 0.66, metalness: 0.06 }, { map: slateTex(), scale: 1 / 3.2, mode: 'roof', bump: 0.05 }),
    fach: castleMat({ vertexColors: true, roughness: 0.9, metalness: 0 }, { map: fachTex(), scale: 1 / 6.4, mode: 'aged', bump: 0.03 }),
    wood: castleMat({ vertexColors: true, roughness: 0.88, metalness: 0 }, { map: woodTex(), scale: 1 / 3.2, mode: 'aged', bump: 0.04 }),
    plain: castleMat({ vertexColors: true, roughness: 0.86, metalness: 0 }),
    metal: castleMat({ vertexColors: true, roughness: 0.4, metalness: 0.6, envMapIntensity: 1.2 }, null, 0.3, 0.6),
  };
  const atlas = atlasTex();
  mats.cloth = castleMat({ map: atlas, vertexColors: true, roughness: 0.95, metalness: 0, side: THREE.DoubleSide, polygonOffset: true, polygonOffsetFactor: -2, polygonOffsetUnits: -2 }, null, 0.4);
  for (const k of Object.keys(mats)) {
    if (!K[k].count) continue;
    const m = new THREE.Mesh(K[k].geometry(ox, oy, oz), mats[k]); m.castShadow = k !== 'cloth'; m.receiveShadow = true; group.add(m);
  }
  const glassMat = new THREE.MeshBasicMaterial({ map: atlas, alphaTest: 0.5, toneMapped: false });
  group.add(new THREE.Mesh(K.glass.geometry(ox, oy, oz), glassMat));
  const glowMat = new THREE.MeshBasicMaterial({ color: 0x10151c, toneMapped: false });
  group.add(new THREE.Mesh(glow.geometry(ox, oy, oz), glowMat));
  const fireMat = new THREE.MeshBasicMaterial({ vertexColors: true, toneMapped: false, blending: THREE.AdditiveBlending, depthWrite: false, transparent: true });
  group.add(new THREE.Mesh(K.fire.geometry(ox, oy, oz), fireMat));
  // the fires: a soft glow round each, and their light on the stone
  const L = K.lights.map((l, i) => ({ x: ox + l.x, y: oy + l.y, z: oz + l.z, I: l.I, ph: i * 2.399, d: 0 }));
  const hp = new Float32Array(L.length * 3), hs = new Float32Array(L.length);
  L.forEach((l, i) => { hp[i * 3] = l.x; hp[i * 3 + 1] = l.y - 0.1; hp[i * 3 + 2] = l.z; hs[i] = K.lights[i].halo; });
  const hg = new THREE.BufferGeometry(); hg.setAttribute('position', new THREE.BufferAttribute(hp, 3)); hg.setAttribute('aSize', new THREE.BufferAttribute(hs, 1));
  const haloMat = new THREE.ShaderMaterial({
    uniforms: { tMap: { value: haloTex() }, uK: { value: 1 }, uScale: { value: 600 } },
    vertexShader: 'attribute float aSize; uniform float uScale; void main() { vec4 mv = modelViewMatrix * vec4(position, 1.0); gl_Position = projectionMatrix * mv; gl_PointSize = min(aSize * uScale / max(-mv.z, 0.5), 220.0); }',
    fragmentShader: 'uniform sampler2D tMap; uniform float uK; void main() { vec4 c = texture2D(tMap, gl_PointCoord); gl_FragColor = vec4(c.rgb * c.a * uK, 1.0); }',
    blending: THREE.AdditiveBlending, depthWrite: false, transparent: true,
  });
  const halos = new THREE.Points(hg, haloMat); halos.frustumCulled = false; group.add(halos);
  // banners on the roofs
  const flagMat = new THREE.MeshStandardMaterial({ color: 0xa3121a, roughness: 0.8, side: THREE.DoubleSide });
  const flagObjs = flags.map((d, i) => {
    const w = d.s, h = d.s * 0.55, g = new THREE.PlaneGeometry(w, h, 10, 2); g.translate(w / 2, 0, 0);
    const m = new THREE.Mesh(g, flagMat); m.position.set(ox + d.x, oy + d.y + d.h - h / 2 - 0.2, oz + d.z); m.castShadow = true; group.add(m);
    return { m, w, base: g.attributes.position.array.slice(), ph: i * 1.7 };
  });
  let night = 0;
  const order = L.slice();
  return {
    group, lights: L,
    setNight(k) { night = k; glowMat.color.setRGB(0.012 + 2.6 * k, 0.016 + 1.45 * k, 0.022 + 0.42 * k); TORCHK.value = 1.0 + 0.2 * k; glassMat.color.setScalar(1.15 - 0.35 * k); haloMat.uniforms.uK.value = 0.55 + 0.45 * k; },
    update(t, wind, cam, viewH = 900, fov = 60) {
      const f = 0.85 + 0.15 * Math.sin(t * 17) * Math.sin(t * 7.3);
      fireMat.color.setScalar(1.9 * f);
      haloMat.uniforms.uScale.value = viewH / (2 * Math.tan(fov * Math.PI / 360));
      // hand the shader the nearest fires; the furthest of them fade in, so none pops
      if (cam) {
        const far = Math.abs(cam.x - ox) > 520 || Math.abs(cam.z - oz) > 440;
        if (!far) { for (const l of L) l.d = Math.hypot(l.x - cam.x, l.y - cam.y, l.z - cam.z); order.sort((a, b) => a.d - b.d); }
        const lim = far ? 0 : Math.min(120, order.length > NL ? order[NL].d : 120);
        for (let i = 0; i < NL; i++) {
          const l = order[i], v = TORCH.value[i];
          if (far || !l || l.d >= lim) { v.w = 0; continue; }
          const fl = 0.84 + 0.16 * Math.sin(t * 11 + l.ph) * Math.sin(t * 6.3 + l.ph * 1.7);
          v.set(l.x, l.y, l.z, l.I * fl * 3.0 * Math.min(1, (lim - l.d) / (0.3 * lim)));
        }
      }
      const yaw = wind ? Math.atan2(-wind[2], wind[0]) : 0.4;
      for (const o of flagObjs) {
        const p = o.m.geometry.attributes.position, a = p.array;
        for (let i = 0; i < a.length; i += 3) { const u = o.base[i] / o.w; a[i + 2] = Math.sin(u * 5.5 - t * 6 + o.ph) * 0.11 * o.w * u; a[i + 1] = o.base[i + 1] - 0.07 * o.w * u * u + Math.sin(u * 4 - t * 4.4 + o.ph) * 0.03 * o.w * u; }
        p.needsUpdate = true; o.m.rotation.y = yaw + 0.25 * Math.sin(t * 0.6 + o.ph);
      }
    },
  };
}
