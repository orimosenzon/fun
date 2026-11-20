// The second ring: an accelerator that was dug in secret and abandoned. A shaft goes straight
// down from the island of the second lake into a hall, and through the hall runs the tunnel: a
// circle ten kilometres across, level, its floor at 800 m, below every valley of the map.
// Like the fortress, one description is both what is drawn (js/render/tunnel.js) and what the
// bike can hit. Ring coordinates: s metres along the ring from the shaft (positive: southwards
// at first), u metres outwards from the centre line, v metres above the floor.
import { T } from './terrain.js';
import { HOLE } from './ringSite.js';

const TH0 = Math.asin(0.1);                      // where on the circle the shaft is
export const RING = {
  R: 5000, th0: TH0, cx: HOLE.x - 5000 * Math.cos(TH0), cz: HOLE.z - 5000 * Math.sin(TH0),
  floor: HOLE.floor, len: 2 * Math.PI * 5000,
  hw: 8, ys: 4,                                  // the tunnel: 16 m wide, walls 4 m, a half circle above (12 m at the crown)
  beam: { u0: 5.3, v1: 2.5 },                    // the beam line along the outer wall: nothing flies there
  nb: 63,                                        // sector doors, one every 499 m all the way round
  door: { r: 5.2, th: 1.6 },                     // each is a wall with one round opening
  hall: { hu: 16, h: 22 },                       // under the shaft
  vault: { k: 6, hu: 22, h: 26 },                // the hall where the sixth door would be: what was left for Yakir is behind its steel door
  shaftR: HOLE.r, curb: 0.6, curbW: 1.4,
  ySwitch: HOLE.floor + 40,                      // nothing above the ground is this low: below it, the bike is under the earth
};
RING.sp = RING.len / RING.nb;
RING.seg = RING.len / 2244;                      // the tunnel is drawn in pieces of 14 m; the halls begin and end where a piece does
RING.hall.hs = 2 * RING.seg;
RING.vault.hs = 3 * RING.seg; RING.vault.s = 214 * RING.seg;
export const rimY = () => (T.hole ? T.hole.rim : 897);

// ---- ring coordinates ----
export function toRing(x, z) {
  const dx = x - RING.cx, dz = z - RING.cz;
  let a = Math.atan2(dz, dx) - RING.th0;
  a -= 2 * Math.PI * Math.round(a / (2 * Math.PI));
  return [RING.R * a, Math.hypot(dx, dz) - RING.R];           // s in (-len/2, len/2], u
}
export function fromRing(s, u = 0, v = 0) {
  const a = RING.th0 + s / RING.R, r = RING.R + u;
  return [RING.cx + r * Math.cos(a), RING.floor + v, RING.cz + r * Math.sin(a)];
}
// heading (the game's yaw: 0 east, positive turning north) of the tunnel at s, flying towards +s
export const ringYaw = (s) => { const a = RING.th0 + s / RING.R; return Math.atan2(-Math.cos(a), -Math.sin(a)); };

// the sector door at index k (0 .. nb-1): null where a hall stands instead
function hash(k, n) { const x = Math.sin(k * 127.1 + n * 311.7) * 43758.5453; return x - Math.floor(x); }
export function doorAt(k) {
  k = ((k % RING.nb) + RING.nb) % RING.nb;
  if (k === 0 || k === RING.vault.k) return null;
  const side = k % 2 ? 1 : -1;                                 // left, right, left: the rider has to weave
  return { k, s: k * RING.sp, u: side * (1.2 + 1.1 * hash(k, 1)), v: 5.6 + 0.9 * hash(k, 2) };
}
const wrapS = (s) => s - RING.len * Math.round(s / RING.len);

// is the point in the open space below the ground (shaft, halls, tunnel)? pad shrinks the space
export function inCavity(x, y, z, pad = 0) {
  const v = y - RING.floor;
  if (v < 0) return true;                                       // (the floor itself is the ground: the physics holds the bike up)
  if (Math.hypot(x - HOLE.x, z - HOLE.z) < RING.shaftR - pad) return true;
  const [s, u] = toRing(x, z), au = Math.abs(u);
  if (au > 30) return false;
  const h = RING.hall, V = RING.vault;
  if (Math.abs(s) < h.hs - pad && au < h.hu - pad) return v < h.h - pad;
  const sv = wrapS(s - V.s);
  if (Math.abs(sv) < V.hs - pad && au < V.hu - pad) return v < V.h - pad;
  // the tunnel itself
  if (au >= RING.hw - pad) return false;
  if (v > RING.ys && Math.hypot(u, v - RING.ys) >= RING.hw - pad) return false;
  if (u > RING.beam.u0 - pad && v < RING.beam.v1 + pad) return false;
  const k = Math.round(s / RING.sp), ds = s - k * RING.sp;
  if (Math.abs(ds) < RING.door.th / 2 + pad) { const d = doorAt(k); if (d && Math.hypot(u - d.u, v - d.v) > RING.door.r - pad) return false; }
  return true;
}

// Is the bike at p under the earth? In the shaft from a metre below its rim, and anywhere below
// the level no open ground reaches. The game sets T.under from this, at every step.
export function underAt(p) {
  if (!T.hole) return false;
  if (p[1] < RING.ySwitch) return true;
  return p[1] < rimY() - 1 && Math.hypot(p[0] - HOLE.x, p[2] - HOLE.z) < RING.shaftR + 1;
}
export const syncUnder = (p) => { T.under = underAt(p) ? RING.floor : null; };

// does the world point touch rock or concrete of the ring?
export function hitTunnel(p, pad = 0.35) {
  if (!T.hole) return false;
  const y = p[1], rim = rimY();
  if (y > rim + RING.curb + pad) return false;
  if (y >= RING.ySwitch) {
    // up here only the shaft is ours: its wall (and the rock right behind it), and the curb round its mouth
    const hr = Math.hypot(p[0] - HOLE.x, p[2] - HOLE.z);
    return hr >= RING.shaftR - pad && (hr < RING.shaftR + RING.curbW + pad || (hr < RING.shaftR + 12 && y < rim - 2));
  }
  return !inCavity(p[0], y, p[2], pad);
}

// ----- the way down and along, for the event: nodes like the ones of js/sim/courses.js -----
export function tunnelCourse() {
  const rim = rimY(), F = RING.floor, out = [];
  const N = (s, u, v, o = {}) => { const p = fromRing(s, u, v); return { x: p[0], z: p[2], alt: p[1], r: 30, hint: true, under: true, slow: true, v: 50, ...o }; };
  // over the hole, down the shaft, and on to the floor of the hall
  out.push({ x: HOLE.x, z: HOLE.z, alt: rim + 24, r: 30, hint: true });
  out.push({ x: HOLE.x, z: HOLE.z, alt: rim - 16, r: 30, hint: true, under: true, slow: true, v: 10 });
  out.push({ x: HOLE.x, z: HOLE.z, alt: F + 26, r: 30, hint: true, under: true, slow: true, v: 10 });
  out.push({ land: 'shaft', at: [HOLE.x, F, HOLE.z, 16], lift: 6, top: rim, under: true, say: { t: 'ring.shaft.t', s: 'ring.shaft.s' } });
  // out of the hall and round the ring, through the open half of five sector doors
  out.push(N(16, 0, 5.5, { v: 14 }), N(34, 0, 5.5, { v: 24 }), N(90, 0, 5.5));
  for (let k = 1; k < RING.vault.k; k++) {
    const d = doorAt(k);
    out.push(N(d.s - 150, d.u * 0.5, 5.5), N(d.s - 40, d.u, d.v), N(d.s, d.u, d.v, { hint: false, r: 4.6, say: k === 1 ? { s: 'ring.door' } : undefined }), N(d.s + 40, d.u, d.v), N(d.s + 150, d.u * 0.5, 5.5));
  }
  const V = RING.vault;
  out.push(N(V.s - 150, 0, 5.5), N(V.s - 80, 0, 5.5, { v: 24 }));
  out.push({ land: 'vault', at: [...fromRing(V.s, 0, 0), 22], lift: 5, appr: 5, under: true });
  return out;
}
