// The second lake, east of the big one, its island, and the hole in the island: where the
// abandoned accelerator is entered. Shared by the terrain bake (tools/bake.mjs shapes the basin,
// the moraine that dams it and the island) and by the game (js/sim/tunnel.js is what lies below).
export const LAKE2 = { level: 888, cx: 5215, cz: 1200, a: 370, b: 300, depth: 15, x0: 4830, x1: 5610, z0: 880, z1: 1520 };
export const ISLE = { x: 5290, z: 1185, r: 105, flat: 34, top: 9 };          // a flat-topped rock, `top` metres above the water
export const HOLE = { x: ISLE.x, z: ISLE.z, r: 9, floor: 800 };              // the shaft: straight down to the floor of the ring
export const HAMLET = { x: 4480, z: 1275 };                                  // the last houses of the valley

const clamp = (x, a, b) => Math.min(b, Math.max(a, x));
const sstep = (a, b, x) => { const t = clamp((x - a) / (b - a), 0, 1); return t * t * (3 - 2 * t); };

// how far out of the lake a point is: below 1 it is in the basin (wob, -1..1, makes the shore irregular)
export const lakeNorm = (x, z, wob = 0) => Math.hypot((x - LAKE2.cx) / LAKE2.a, (z - LAKE2.cz) / LAKE2.b) / (1 + 0.14 * clamp(wob, -1, 1));

// height of the ground at world (x, z), given the natural height h there
export function ringSiteHeight(x, z, h, wob = 0, rough = 0) {
  const L = LAKE2, dn = lakeNorm(x, z, wob);
  let out = h;
  if (dn < 1) out = Math.min(h, L.level + 0.6 - L.depth * Math.pow(1 - dn * dn, 0.8));
  else {
    // the shore, and on the low side an old moraine that holds the water back
    const dOut = (dn - 1) * 330, t = 1 - sstep(140, 380, dOut), target = L.level + 0.6 + Math.min(dOut, 80) * 0.03;
    if (target > h) out = h + (target - h) * t;
  }
  const di = Math.hypot(x - ISLE.x, z - ISLE.z);
  if (di < ISLE.r) {
    const k = sstep(ISLE.flat, ISLE.r, di);
    out = Math.max(out, L.level + ISLE.top - (ISLE.top + L.depth + 3) * k + 5 * rough * k * (1 - k));
  }
  return out;
}
