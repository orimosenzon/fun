// Draws what js/sim/tunnel.js describes: the shaft on the island, the hall under it, the ring
// tunnel with its beam line, the sector doors, and the hall with the steel door. No daylight
// reaches down there: the light is the old strip lamps (some dead, some flickering), the amber
// rims of the door openings, and the bike's own lamp and glowing core. The tunnel is 31 km long,
// so it is one piece of 14 m drawn many times around the camera.
// Local frame of every part: x outwards from the ring's centre line, y up from the floor, z along the ring.
import * as THREE from '../../lib/three.module.min.js';
import { U, GLSL_COMMON } from './env.js';
import { Builder, pbox } from './meshkit.js';
import { T } from '../sim/terrain.js';
import { RING, fromRing, toRing, doorAt, rimY } from '../sim/tunnel.js';
import { HOLE } from '../sim/ringSite.js';

const NL = 10, NSEG = 2244, SPAN = 46;          // lamps the shader takes at once; pieces round the ring; pieces drawn each way from the camera
const LAMP = { value: Array.from({ length: NL }, () => new THREE.Vector4()) }, LAMPC = { value: Array.from({ length: NL }, () => new THREE.Vector3()) };
const BIKE = { value: new THREE.Vector3() }, BIKEF = { value: new THREE.Vector3(1, 0, 0) }, SHAFT = { value: new THREE.Vector4() }, BIKEK = { value: 1 };

const VERT = /* glsl */`
attribute float aW;
#ifdef SEGS
attribute vec3 aLamp; varying vec3 vLamp; varying vec3 vLoc; varying vec3 vNl;
#endif
varying vec3 vWp; varying vec3 vN; varying vec3 vC; varying float vE;
void main() {
  vec4 p = vec4(position, 1.0); vec3 n = normal;
  #ifdef USE_INSTANCING
  p = instanceMatrix * p; n = mat3(instanceMatrix) * n;
  #endif
  #ifdef SEGS
  vLamp = aLamp; vLoc = position; vNl = normal;
  #endif
  vWp = p.xyz; vN = n; vC = color; vE = aW;
  gl_Position = projectionMatrix * viewMatrix * p;
}`;
const FRAG = /* glsl */`
precision highp float;
${GLSL_COMMON}
uniform vec4 uLamp[${NL}]; uniform vec3 uLampC[${NL}]; uniform vec3 uBike; uniform vec3 uBikeF; uniform vec4 uShaft; uniform float uBikeK;
varying vec3 vWp; varying vec3 vN; varying vec3 vC; varying float vE;
#ifdef SEGS
varying vec3 vLamp; varying vec3 vLoc; varying vec3 vNl;
#endif
void main() {
  vec3 n = normalize(vN), toCam = cameraPosition - vWp;
  float flip = dot(n, toCam) < 0.0 ? -1.0 : 1.0; n *= flip;
  // old concrete: mottled, with the lines of the shuttering
  float m1 = vnoise(vWp.xz * 0.9 + vWp.y * 0.37), m2 = vnoise(vec2(vWp.x + vWp.z, vWp.y) * 0.21 + 17.0), m3 = vnoise(vec2(vWp.x - vWp.z, vWp.y * 3.0) * 0.05);
  vec3 alb = vC * (0.74 + 0.36 * m1) * (0.8 + 0.34 * m2) * (0.88 + 0.2 * smoothstep(0.35, 0.65, m3));
  vec3 L = vec3(0.016, 0.018, 0.022);
  for (int i = 0; i < ${NL}; i++) {
    float w = uLamp[i].w; if (w <= 0.0) continue;
    vec3 d = uLamp[i].xyz - vWp; float d2 = dot(d, d);
    L += uLampC[i] * w * (0.3 + 0.7 * max(dot(n, d) * inversesqrt(d2 + 1e-3), 0.0)) / (1.0 + d2 * 0.014);
  }
  float glow = vE;
  #ifdef SEGS
  // the strip lamps of this piece of tunnel and of its two neighbours
  vec3 nl = normalize(vNl) * flip;
  for (int j = 0; j < 3; j++) {
    float w = j == 0 ? vLamp.x : (j == 1 ? vLamp.y : vLamp.z); if (w <= 0.0) continue;
    if (w < 0.9) w *= step(0.42, vnoise(vec2(uTime * 9.0, w * 173.0 + float(j))));                    // a tube on its last legs
    vec3 d = vec3(0.0, ${(RING.ys + RING.hw - 0.5).toFixed(2)}, ${RING.seg.toFixed(4)} * (float(j) - 0.5)) - vLoc; float d2 = dot(d, d);
    L += vec3(0.80, 1.0, 0.88) * w * 1.25 * (0.3 + 0.7 * max(dot(nl, d) * inversesqrt(d2 + 1e-3), 0.0)) / (1.0 + d2 * 0.011);
  }
  float me = vLamp.y; if (me < 0.9) me *= step(0.42, vnoise(vec2(uTime * 9.0, me * 173.0 + 1.0)));
  glow *= me;
  #endif
  // daylight: all of it at the mouth of the shaft, a pool of it on the floor of the hall below
  vec3 so = uShaft.xyz - vWp; float depth = max(so.y, 0.0), hr = length(so.xz), r2 = uShaft.w * uShaft.w;
  float open = so.y < 0.3 ? 1.0 : r2 / (r2 + 2.2 * depth * depth) * exp(-max(hr - uShaft.w, 0.0) / 9.0);
  vec3 day = (uSunCol * max(dot(n, uSunDir), 0.0) * step(so.y, 0.3) + uAmbSky * 0.7 + skyLUT(vec3(0.0, 1.0, 0.0)) * 0.7) * open;
  // the bike: its core glows on whatever is near, and its lamp throws a cone ahead
  vec3 db = uBike - vWp; float b2 = dot(db, db); vec3 bd = db * inversesqrt(b2 + 1e-3);
  vec3 bl = vec3(0.42, 0.68, 1.0) * 1.5 * (0.3 + 0.7 * max(dot(n, bd), 0.0)) / (1.0 + b2 * 0.06);
  bl += vec3(1.0, 0.95, 0.84) * 1.5 * smoothstep(0.78, 0.99, dot(-bd, uBikeF)) * max(dot(n, bd), 0.0) * smoothstep(4.0, 60.0, b2) / (1.0 + b2 * 0.0011);
  vec3 col = alb * (L + day + bl * uBikeK) + vC * glow;
  float f = exp(-length(toCam) * 0.0042);
  col = open > 0.99 ? applyFog(col, vWp) : mix(vec3(0.005, 0.007, 0.010), col, mix(f, 1.0, open));
  gl_FragColor = vec4(col, 1.0);
}`;

function ringMat(segs) {
  return new THREE.ShaderMaterial({
    uniforms: { ...U, uLamp: LAMP, uLampC: LAMPC, uBike: BIKE, uBikeF: BIKEF, uShaft: SHAFT, uBikeK: BIKEK },
    vertexShader: VERT, fragmentShader: FRAG, vertexColors: true, side: THREE.DoubleSide, defines: segs ? { SEGS: 1 } : {},
  });
}

// a turned shape about any axis: O origin, A the axis, prof = [[distance along A, radius], ...]; inward: seen from inside
function lathe(B, O, A, prof, col, n = 16, inward = false) {
  const P = Math.abs(A[1]) > 0.9 ? [1, 0, 0] : [0, 1, 0], Q = [A[1] * P[2] - A[2] * P[1], A[2] * P[0] - A[0] * P[2], A[0] * P[1] - A[1] * P[0]];
  const at = (d, r, k) => { const a = k / n * 6.28319, c = Math.cos(a) * r, s = Math.sin(a) * r; return [O[0] + A[0] * d + P[0] * c + Q[0] * s, O[1] + A[1] * d + P[1] * c + Q[1] * s, O[2] + A[2] * d + P[2] * c + Q[2] * s]; };
  for (let i = 0; i < prof.length - 1; i++) {
    const [d0, r0] = prof[i], [d1, r1] = prof[i + 1], c = typeof col[0] === 'number' ? col : col[i % col.length];
    for (let k = 0; k < n; k++) {
      const a = (k + 0.5) / n * 6.28319, sg = inward ? -1 : 1;
      let h = [(P[0] * Math.cos(a) + Q[0] * Math.sin(a)) * sg, (P[1] * Math.cos(a) + Q[1] * Math.sin(a)) * sg, (P[2] * Math.cos(a) + Q[2] * Math.sin(a)) * sg];
      if (Math.abs(d1 - d0) < 1e-6) { const s2 = (r1 < r0 ? 1 : -1) * (inward ? -1 : 1); h = [A[0] * s2, A[1] * s2, A[2] * s2]; }      // a flat face: it looks along the axis
      if (r0 < 1e-6) B.tri(at(d0, 0, 0), at(d1, r1, k), at(d1, r1, k + 1), c, h);
      else if (r1 < 1e-6) B.tri(at(d0, r0, k), at(d0, r0, k + 1), at(d1, 0, 0), c, h);
      else B.quad(at(d0, r0, k), at(d0, r0, k + 1), at(d1, r1, k + 1), at(d1, r1, k), c, h);
    }
  }
}
// the outline of the tunnel: is (u, v) inside it?
const inSection = (u, v) => Math.abs(u) < RING.hw && v > 0 && (v < RING.ys || Math.hypot(u, v - RING.ys) < RING.hw);
function edgeFrom(u0, v0, a) {            // how far from (u0, v0) in direction a the outline is
  let lo = 0, hi = 30;
  for (let i = 0; i < 24; i++) { const m = (lo + hi) / 2; if (inSection(u0 + Math.cos(a) * m, v0 + Math.sin(a) * m)) lo = m; else hi = m; }
  return lo;
}
const CONC = [0.34, 0.33, 0.31], FLOORC = [0.2, 0.2, 0.2], STEEL = [0.2, 0.23, 0.24], BLUE = [0.04, 0.15, 0.44], DARK = [0.06, 0.06, 0.07];

// ---- one piece of tunnel, 14 m ----
function segmentGeometry() {
  const B = new Builder(), S = RING.seg + 0.04, hw = RING.hw, ys = RING.ys, N = 14;
  B.wv = 0;
  B.quad([-hw, 0, 0], [hw, 0, 0], [hw, 0, S], [-hw, 0, S], FLOORC, [0, 1, 0]);
  B.quad([-0.1, 0.012, 2], [0.1, 0.012, 2], [0.1, 0.012, 8.5], [-0.1, 0.012, 8.5], [0.5, 0.4, 0.08], [0, 1, 0]);                 // the worn centre line
  B.quad([-hw, 0, 0], [-hw, 0, S], [-hw, ys, S], [-hw, ys, 0], CONC, [1, 0, 0]);
  B.quad([hw, 0, 0], [hw, 0, S], [hw, ys, S], [hw, ys, 0], CONC, [-1, 0, 0]);
  const arc = (r, k) => { const a = Math.PI * k / N; return [r * Math.cos(a), ys + r * Math.sin(a), -Math.cos(a), -Math.sin(a)]; };
  for (let k = 0; k < N; k++) {
    const a = arc(hw, k), b = arc(hw, k + 1), na = [a[2], a[3], 0], nb = [b[2], b[3], 0];
    B.quad([a[0], a[1], 0], [a[0], a[1], S], [b[0], b[1], S], [b[0], b[1], 0], CONC, na, [na, na, nb, nb]);
    // a rib where two pieces meet
    const c = arc(hw - 0.3, k), d = arc(hw - 0.3, k + 1), RIBC = [0.27, 0.265, 0.25];
    B.quad([c[0], c[1], 0], [c[0], c[1], 0.55], [d[0], d[1], 0.55], [d[0], d[1], 0], RIBC, na);
    B.quad([a[0], a[1], 0.55], [b[0], b[1], 0.55], [d[0], d[1], 0.55], [c[0], c[1], 0.55], RIBC, [0, 0, 1]);
  }
  for (const sg of [-1, 1]) { pbox(B, Math.min(sg * hw, sg * (hw - 0.3)), Math.max(sg * hw, sg * (hw - 0.3)), 0, ys, 0, 0.55, [0.27, 0.265, 0.25]); }
  // the beam line along the outer wall: blue cryostats on stands, a thin silver line above, a kerb in front
  lathe(B, [6.5, 1.45, 0], [0, 0, 1], [[0, 0.5], [6.6, 0.5], [6.6, 0.64], [7.4, 0.64], [7.4, 0.5], [S, 0.5]], BLUE, 12);
  lathe(B, [7.45, 2.3, 0], [0, 0, 1], [[0, 0.14], [S, 0.14]], [0.5, 0.52, 0.55], 6);
  for (const z of [3.5, 10.5]) pbox(B, 6.15, 6.85, 0, 1.0, z - 0.25, z + 0.25, STEEL);
  pbox(B, RING.beam.u0, RING.beam.u0 + 0.18, 0, 0.22, 0, S, [0.42, 0.36, 0.1]);
  // cable trays on the inner wall
  for (const [v, r] of [[2.9, 0.07], [3.15, 0.09], [3.4, 0.06]]) lathe(B, [-hw + 0.2, v, 0], [0, 0, 1], [[0, r], [S, r]], DARK, 5);
  pbox(B, -hw, -hw + 0.34, 2.72, 2.78, 0, S, STEEL);
  // the strip lamp under the crown
  B.wv = 5; pbox(B, -0.16, 0.16, ys + hw - 0.42, ys + hw - 0.3, S / 2 - 1.3, S / 2 + 1.3, [0.8, 1.0, 0.88]); B.wv = 0;
  pbox(B, -0.24, 0.24, ys + hw - 0.3, ys + hw - 0.2, S / 2 - 1.45, S / 2 + 1.45, DARK);
  return B.geometry();
}

// a wall across the frame with the outline `inside(x, y)` cut out of the rectangle [-hu, hu] x [0, h], at z
function endWall(B, hu, h, z, face, col) {
  const hw = RING.hw, ys = RING.ys, N = 14, hint = [0, 0, face];
  B.quad([-hu, 0, z], [-hw, 0, z], [-hw, h, z], [-hu, h, z], col, hint);
  B.quad([hw, 0, z], [hu, 0, z], [hu, h, z], [hw, h, z], col, hint);
  for (let k = 0; k < N; k++) { const a0 = Math.PI * k / N, a1 = Math.PI * (k + 1) / N, x0 = hw * Math.cos(a0), x1 = hw * Math.cos(a1); B.quad([x0, ys + hw * Math.sin(a0), z], [x1, ys + hw * Math.sin(a1), z], [x1, h, z], [x0, h, z], col, hint); }
}
// a hall: floor, walls, roof (with a round hole in it where the shaft comes down), lamps
function hall(B, lamps, s0, hu, hs, h, shaft) {
  const a = RING.th0 + s0 / RING.R, org = fromRing(s0, 0, 0), W = [0.37, 0.36, 0.34];
  B.push(org, -a); B.wv = 0;
  B.quad([-hu, 0, -hs], [hu, 0, -hs], [hu, 0, hs], [-hu, 0, hs], [0.21, 0.21, 0.21], [0, 1, 0]);
  B.quad([-hu, 0, -hs], [-hu, 0, hs], [-hu, h, hs], [-hu, h, -hs], W, [1, 0, 0]);
  B.quad([hu, 0, -hs], [hu, 0, hs], [hu, h, hs], [hu, h, -hs], W, [-1, 0, 0]);
  endWall(B, hu, h, -hs, 1, W); endWall(B, hu, h, hs, -1, W);
  if (shaft) {
    const n = 40, edge = (k) => { const q = k / n * 6.28319, c = Math.cos(q), s = Math.sin(q), t = Math.min(hu / Math.abs(c || 1e-9), hs / Math.abs(s || 1e-9)); return [[c * shaft, h, s * shaft], [c * t, h, s * t]]; };
    for (let k = 0; k < n; k++) { const [a0, b0] = edge(k), [a1, b1] = edge(k + 1); B.quad(a0, b0, b1, a1, W, [0, -1, 0]); }
  } else B.quad([-hu, h, -hs], [hu, h, -hs], [hu, h, hs], [-hu, h, hs], W, [0, -1, 0]);
  // a band of paint round the walls, and the pillars that carry the roof
  for (const sg of [-1, 1]) {
    pbox(B, Math.min(sg * hu, sg * (hu - 0.05)), Math.max(sg * hu, sg * (hu - 0.05)), 1.1, 1.5, -hs, hs, [0.5, 0.4, 0.08]);
    for (let z = -hs + 7; z < hs; z += 14) pbox(B, Math.min(sg * hu, sg * (hu - 0.9)), Math.max(sg * hu, sg * (hu - 0.9)), 0, h, z - 0.6, z + 0.6, [0.3, 0.295, 0.28]);
  }
  // lamps under the roof, near the corners (one of them has given up)
  let i = 0;
  for (const sx of [-1, 1]) for (const sz of [-1, 1]) {
    const x = sx * (hu - 5), z = sz * (hs - 8), on = i++ !== 1;
    pbox(B, x - 0.5, x + 0.5, h - 0.5, h - 0.3, z - 0.5, z + 0.5, DARK);
    if (on) { B.wv = 6; pbox(B, x - 0.36, x + 0.36, h - 0.62, h - 0.5, z - 0.36, z + 0.36, [1.0, 0.86, 0.6]); B.wv = 0; lamps.push({ p: B.X([x, h - 1.4, z]), w: 2.4, c: [1.0, 0.84, 0.58] }); }
  }
  return org;
}

// the steel door of the vault: on the outer wall of its hall
function vaultDoor(B, lamps, hu) {
  const y = 5.4, r = 4.3, ST = [0.33, 0.35, 0.37], A = [-1, 0, 0], O = [hu, y, 0];
  lathe(B, O, A, [[0, r + 0.9], [0.45, r + 0.9], [0.45, r + 0.1], [0, r + 0.1]], [0.24, 0.25, 0.26], 40);                       // the frame
  lathe(B, O, A, [[0.25, r], [0.75, r], [0.9, r - 0.4], [0.9, 1.1], [1.25, 1.0], [1.25, 0.001]], ST, 40);                            // the door, and its hub
  lathe(B, O, A, [[1.25, 1.75], [1.42, 1.75], [1.42, 1.5], [1.25, 1.5]], [0.5, 0.42, 0.1], 24);                                // the hand wheel
  for (let k = 0; k < 6; k++) { const q = k / 6 * 6.28319; pbox(B, hu - 1.4, hu - 1.3, y + Math.sin(q) * 0.8 - 0.07 + Math.sin(q) * 0.4, y + Math.sin(q) * 0.8 + 0.07 + Math.sin(q) * 0.4, Math.cos(q) * 1.2 - 0.07, Math.cos(q) * 1.2 + 0.07, [0.5, 0.42, 0.1]); }
  for (let k = 0; k < 12; k++) { const q = (k + 0.5) / 12 * 6.28319; lathe(B, [hu - 0.9, y + Math.sin(q) * (r - 0.9), Math.cos(q) * (r - 0.9)], A, [[0, 0.2], [0.14, 0.2], [0.14, 0.001]], [0.5, 0.52, 0.55], 6); }   // bolts
  pbox(B, hu - 0.9, hu, y - 1.2, y + 1.2, r + 0.2, r + 1.5, [0.24, 0.25, 0.26]);                                                // the hinge
  // one lamp still burns over it, and a small red one says: locked
  B.wv = 6; pbox(B, hu - 1.6, hu - 0.9, y + r + 2.6, y + r + 2.8, -0.7, 0.7, [1.0, 0.92, 0.75]); B.wv = 4; pbox(B, hu - 0.3, hu - 0.12, y + 0.5, y + 0.8, -r - 1.6, -r - 1.3, [1.0, 0.08, 0.04]); B.wv = 0;
  lamps.push({ p: B.X([hu - 3.2, y + r + 1.6, 0]), w: 3.4, c: [1.0, 0.92, 0.75] }, { p: B.X([hu - 1.2, y + 0.6, -r - 1.5]), w: 0.5, c: [1.0, 0.1, 0.05] });
  return [hu - 0.06, y + r + 2.0, 0];
}

function signTexture() {
  const c = document.createElement('canvas'); c.width = 1024; c.height = 160;
  const g = c.getContext('2d');
  g.fillStyle = '#1b1d1e'; g.fillRect(0, 0, 1024, 160);
  g.strokeStyle = '#c8b46a'; g.lineWidth = 5; g.strokeRect(9, 9, 1006, 142);
  g.fillStyle = '#d8cfae'; g.font = '700 84px "DIN Alternate", "Arial Narrow", Arial, sans-serif'; g.textAlign = 'center'; g.textBaseline = 'middle';
  g.fillText('ANNEAU  II   ·   CHAMBRE  6', 512, 84);
  // forty years of dust
  for (let i = 0; i < 2600; i++) { g.fillStyle = `rgba(20,20,20,${Math.random() * 0.25})`; g.fillRect(Math.random() * 1024, Math.random() * 160, 2 + Math.random() * 10, 1 + Math.random() * 3); }
  const t = new THREE.CanvasTexture(c); t.colorSpace = THREE.SRGBColorSpace; t.anisotropy = 4;
  return t;
}

export function buildTunnel(scene) {
  if (!T.hole) return { update() {}, depth: () => 0 };
  const rim = rimY(), F = RING.floor, lamps = [];
  U.uHole.value.set(HOLE.x, HOLE.z, RING.shaftR + 0.7, 0);
  SHAFT.value.set(HOLE.x, rim, HOLE.z, RING.shaftR);

  const B = new Builder(); B.wv = 0;
  // ---- the shaft: rings of concrete poured one under the other, a ladder, a curb round the mouth ----
  const top = rim + RING.curb, y0 = F + RING.hall.h, prof = [], cols = [];
  for (let y = y0; y < top - 0.01; y += 6.2) { prof.push([Math.min(y, top), RING.shaftR]); cols.push(cols.length % 2 ? [0.33, 0.32, 0.3] : [0.29, 0.285, 0.27]); }
  prof.push([top, RING.shaftR]);
  lathe(B, [HOLE.x, 0, HOLE.z], [0, 1, 0], prof, cols, 32, true);
  lathe(B, [HOLE.x, 0, HOLE.z], [0, 1, 0], [[top, RING.shaftR + RING.curbW], [top, RING.shaftR]], [0.5, 0.49, 0.46], 32);
  lathe(B, [HOLE.x, 0, HOLE.z], [0, 1, 0], [[top, RING.shaftR + RING.curbW], [rim - 1.5, RING.shaftR + RING.curbW + 0.25]], [0.44, 0.43, 0.4], 32);
  { const lx = HOLE.x - (RING.shaftR - 0.22), z = HOLE.z, RUST = [0.3, 0.16, 0.09];
    for (const dz of [-0.3, 0.3]) pbox(B, lx - 0.04, lx + 0.04, y0, top + 1.1, z + dz - 0.04, z + dz + 0.04, RUST);
    for (let y = y0 + 0.5; y < top + 1; y += 0.5) pbox(B, lx - 0.025, lx + 0.025, y - 0.025, y + 0.025, z - 0.3, z + 0.3, RUST); }
  for (const [dy, on] of [[24, 1], [50, 0], [74, 1]]) {
    const q = dy * 0.7, x = HOLE.x + Math.cos(q) * (RING.shaftR - 0.25), z = HOLE.z + Math.sin(q) * (RING.shaftR - 0.25), y = rim - dy;
    pbox(B, x - 0.3, x + 0.3, y - 0.2, y + 0.2, z - 0.3, z + 0.3, DARK);
    if (on) { B.wv = 5; pbox(B, x - 0.2 - Math.cos(q) * 0.2, x + 0.2 - Math.cos(q) * 0.2, y - 0.14, y + 0.14, z - 0.2 - Math.sin(q) * 0.2, z + 0.2 - Math.sin(q) * 0.2, [1.0, 0.8, 0.5]); B.wv = 0; lamps.push({ p: [HOLE.x + Math.cos(q) * (RING.shaftR - 1.5), y, HOLE.z + Math.sin(q) * (RING.shaftR - 1.5)], w: 1.5, c: [1.0, 0.8, 0.5] }); }
  }
  // ---- the hall under the shaft, and the hall of the steel door ----
  hall(B, lamps, 0, RING.hall.hu, RING.hall.hs, RING.hall.h, RING.shaftR); B.pop();
  const V = RING.vault;
  hall(B, lamps, V.s, V.hu, V.hs, V.h, 0);
  const signAt = vaultDoor(B, lamps, V.hu), signW = B.X(signAt), signA = RING.th0 + V.s / RING.R;
  // a yellow circle on the floor: where one may set down
  for (let k = 0; k < 48; k++) { const a0 = k / 48 * 6.28319, a1 = (k + 1) / 48 * 6.28319; B.quad([Math.cos(a0) * 11.4, 0.02, Math.sin(a0) * 11.4], [Math.cos(a0) * 12.2, 0.02, Math.sin(a0) * 12.2], [Math.cos(a1) * 12.2, 0.02, Math.sin(a1) * 12.2], [Math.cos(a1) * 11.4, 0.02, Math.sin(a1) * 11.4], [0.55, 0.45, 0.08], [0, 1, 0]); }
  B.pop();
  // ---- the sector doors: a steel wall across the tunnel, one round opening, its rim alight ----
  for (let k = 1; k < RING.nb; k++) {
    const d = doorAt(k); if (!d) continue;
    const a = RING.th0 + d.s / RING.R, th = RING.door.th / 2, r = RING.door.r, M = 40;
    B.push(fromRing(d.s, 0, 0), -a);
    const P = (q, rr, z) => [d.u + Math.cos(q) * rr, d.v + Math.sin(q) * rr, z];
    for (let i = 0; i < M; i++) {
      const q0 = i / M * 6.28319, q1 = (i + 1) / M * 6.28319, e0 = edgeFrom(d.u, d.v, q0), e1 = edgeFrom(d.u, d.v, q1), band = Math.floor(i / 2) % 2 ? [0.62, 0.5, 0.06] : [0.05, 0.05, 0.05];
      for (const [z, f] of [[-th, -1], [th, 1]]) {
        B.wv = 2.2; B.quad(P(q0, r, z), P(q1, r, z), P(q1, r + 0.16, z), P(q0, r + 0.16, z), [1.0, 0.5, 0.12], [0, 0, f]); B.wv = 0;
        B.quad(P(q0, r + 0.16, z), P(q1, r + 0.16, z), P(q1, Math.min(r + 1.0, e1), z), P(q0, Math.min(r + 1.0, e0), z), band, [0, 0, f]);
        if (e0 > r + 1 || e1 > r + 1) B.quad(P(q0, Math.min(r + 1.0, e0), z), P(q1, Math.min(r + 1.0, e1), z), P(q1, Math.max(e1, r + 1), z), P(q0, Math.max(e0, r + 1), z), STEEL, [0, 0, f]);
      }
      B.quad(P(q0, r, -th), P(q1, r, -th), P(q1, r, th), P(q0, r, th), [0.13, 0.14, 0.15], [-Math.cos(q0), -Math.sin(q0), 0]);
    }
    for (const z of [-3.5, 3.5]) lamps.push({ p: B.X([d.u, d.v, z]), w: 1.0, c: [1.0, 0.55, 0.16] });
    B.pop();
  }
  const mesh = new THREE.Mesh(B.geometry(), ringMat(false)); mesh.frustumCulled = false; mesh.matrixAutoUpdate = false; scene.add(mesh);
  const sign = new THREE.Mesh(new THREE.PlaneGeometry(9, 1.4), new THREE.MeshBasicMaterial({ map: signTexture(), color: 0x8a8a8a, toneMapped: false }));
  sign.position.set(signW[0], signW[1], signW[2]); sign.rotation.y = -signA - Math.PI / 2; scene.add(sign);

  // ---- the tunnel itself: one piece, drawn where the camera is ----
  const geo = segmentGeometry(), lampAttr = new THREE.InstancedBufferAttribute(new Float32Array((2 * SPAN + 1) * 3), 3);
  lampAttr.setUsage(THREE.DynamicDrawUsage); geo.setAttribute('aLamp', lampAttr);
  const segs = new THREE.InstancedMesh(geo, ringMat(true), 2 * SPAN + 1); segs.frustumCulled = false; segs.matrixAutoUpdate = false; segs.count = 0; scene.add(segs);
  const hs = (k, n) => { const x = Math.sin(k * 12.9898 + n * 78.233) * 43758.5453; return x - Math.floor(x); };
  const lampOf = (k) => { k = ((k % NSEG) + NSEG) % NSEG; if (k % 2) return 0; const h = hs(k, 1); return h < 0.24 ? 0 : h < 0.34 ? 0.3 + 0.5 * hs(k, 2) : 1; };   // every other piece has a lamp; some are dead, some flicker
  const inHall = (k) => { k = ((k % NSEG) + NSEG) % NSEG; return k < 2 || k >= NSEG - 2 || (k >= 211 && k < 217); };
  const m4 = new THREE.Matrix4(), near = [];
  // the light that falls on the bike itself down there (its own materials know nothing of this shader)
  const glow = new THREE.PointLight(0xdfffee, 0, 70, 1.6); scene.add(glow);
  let lastK = 1e9;

  return {
    mesh, segs, glow,
    // 0 in the open .. 1 under the earth (for the camera at p): what the daylight, the weather and the exposure go by
    depth(p) { const hr = Math.hypot(p[0] - HOLE.x, p[2] - HOLE.z); return p[1] < RING.ySwitch ? 1 : hr < RING.shaftR + 1 ? Math.min(1, Math.max(0, (rim - p[1]) / 45)) : 0; },
    update(camera, bike, fwd) {
      const c = camera.position, [s, u] = toRing(c.x, c.z), near2 = c.y < rim + 120 && Math.abs(u) < 300;
      mesh.visible = sign.visible = c.y < rim + 2500;
      BIKE.value.set(bike.p[0], bike.p[1] + 0.5, bike.p[2]); BIKEF.value.set(fwd[0], fwd[1], fwd[2]);
      BIKEK.value = bike.p[1] < rim + 3 ? 1 : 0;
      if (!near2) { segs.count = 0; glow.intensity = 0; lastK = 1e9; return; }
      const k0 = Math.floor(s / RING.seg);
      if (k0 !== lastK) {
        lastK = k0; let n = 0;
        for (let k = k0 - SPAN; k <= k0 + SPAN; k++) {
          if (inHall(k)) continue;
          const a = RING.th0 + k * RING.seg / RING.R, p = fromRing(k * RING.seg, 0, 0), ca = Math.cos(a), sa = Math.sin(a);
          m4.set(ca, 0, -sa, p[0], 0, 1, 0, p[1], sa, 0, ca, p[2], 0, 0, 0, 1); segs.setMatrixAt(n, m4);
          lampAttr.setXYZ(n, lampOf(k - 1), lampOf(k), lampOf(k + 1)); n++;
        }
        segs.count = n; segs.instanceMatrix.needsUpdate = true; lampAttr.needsUpdate = true;
      }
      // the lamps nearest to the camera are the ones that shine
      near.length = 0;
      for (const l of lamps) { const d = (l.p[0] - c.x) ** 2 + (l.p[1] - c.y) ** 2 + (l.p[2] - c.z) ** 2; if (d < 700 * 700) near.push([d, l]); }
      near.sort((a, b) => a[0] - b[0]);
      for (let i = 0; i < NL; i++) { const l = near[i] && near[i][1]; if (l) { LAMP.value[i].set(l.p[0], l.p[1], l.p[2], l.w); LAMPC.value[i].set(l.c[0], l.c[1], l.c[2]); } else LAMP.value[i].w = 0; }
      // light on the bike: from the nearest lamp of the tunnel, or straight above it
      const kd = this.depth(bike.p), [bs] = toRing(bike.p[0], bike.p[2]), kb = Math.round((bs / RING.seg - 0.5) / 2) * 2, lp = fromRing((kb + 0.5) * RING.seg, 0, RING.ys + RING.hw - 1);
      glow.position.set(lp[0], Math.max(lp[1], bike.p[1] + 4), lp[2]); glow.intensity = kd * (lampOf(kb) > 0.9 ? 260 : 90);
    },
  };
}
