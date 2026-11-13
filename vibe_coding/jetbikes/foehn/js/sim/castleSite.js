// Where the fortress stands, and the shape of the rock it stands on. Shared by the terrain bake
// (tools/bake.mjs cuts the plateau and the moat) and by the game (js/sim/castle.js builds on it).
// Castle coordinates ("local"): metres from SITE, x east, z south, y up from the courtyard floor.
export const SITE = { x: -1644, z: -300, y: 2250 };
export const MOAT = { x0: -216, x1: -168, depth: 22, bank: 12 };   // dry ditch across the ridge, in front of the gate
export const FORECOURT = { x: -246, z: 0, r: 28 };                 // the start pad, on the far side of the bridge

const clamp = (x, a, b) => Math.min(b, Math.max(a, x));
const outside = (x, z, x0, x1, z0, z1) => Math.hypot(Math.max(x0 - x, 0, x - x1), Math.max(z0 - z, 0, z - z1));

// height of the ground at castle coordinates (lx, lz), given the natural height h there;
// wob (-1..1) roughens the width of the cliff below the walls
export function siteHeight(lx, lz, h, wob = 0) {
  const d = Math.min(outside(lx, lz, MOAT.x0, 172, -126, 126), outside(lx, lz, -276, MOAT.x0, -48, 48));
  const W = 100 * (1 + 0.3 * clamp(wob, -1, 1));
  let out = h;
  if (d < W) { const t = 1 - d / W; out = SITE.y + (h - SITE.y) * (1 - t * t); }     // a crag: steep right below the walls
  if (lx > MOAT.x0 && lx < MOAT.x1 && Math.abs(lz) < 200) {
    const m = clamp(Math.min(lx - MOAT.x0, MOAT.x1 - lx) / MOAT.bank, 0, 1);
    out = Math.min(out, SITE.y - MOAT.depth * m);
  }
  return out;
}
