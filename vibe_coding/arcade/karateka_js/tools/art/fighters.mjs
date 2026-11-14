// The fighters, drawn by this project: a jointed figure posed by where its hips, feet and hands
// are, painted in the manner of an Apple II sprite (white gi, black belt and black lines where one
// limb passes in front of another, bare skin as single coloured pixels), and cut into the pieces
// the game's drawing routine puts together: the legs, the body, the head, and a mask that clears
// the floor behind the feet. Writes src/data/art_fighters.js: the shapes and the tables that say
// which shape each pose uses and where it goes.
//
// Coordinates: x in pixels from the fighter's own column, forward = right; y in rows from the top
// of the fighters' strip. The soles are at y = 48.
// usage: node tools/art/fighters.mjs
import fs from 'node:fs';
import { Canvas, dataFile } from './canvas.mjs';

const LEN = { thigh: 11.5, shin: 11.5, upper: 7, fore: 7.5, torso: 12, neck: 5 };
const FLOOR = 46;                               // the ankles of a standing fighter

// ---- the poses -----------------------------------------------------------------------------
// legs: hip [x, y], feet [front, back] as ankle positions, back foot pointing backwards if `splay`,
//       `reach` lets a kicking leg be drawn longer than it is, as a leg coming at the viewer's side would be
// body: lean in degrees (forward positive), hands [front, back] relative to the hip, `wide` = chest turned to the viewer
const G = [[12, -13], [6, -8]];                 // the guard: front fist up, back fist at the chest
const DOWN = [[1, 3], [-1, 3]];                 // arms hanging
const stance = (hx, front, back, hy = 30) => STANCE(hx, front, back, hy - 1.5);
const STANCE = (hx, front, back, hy) => ({ hip: [hx, hy], feet: [[front, FLOOR], [back, FLOOR]], splay: true });
const LEGS = [
  /* 0 */ { hip: [12, 25], feet: [[17, 40], [11, FLOOR]] },                       // the kick: the knee comes up
  /* 1 */ { hip: [12, 25], feet: [[22, 36], [11, FLOOR]] },
  /* 2 */ { hip: [16, 27], feet: [[46, 9], [17, FLOOR]], reach: 1.6 },            // high kick
  /* 3 */ { hip: [13, 26], feet: [[45, 22], [12, FLOOR]], reach: 1.5 },           // middle kick
  /* 4 */ { hip: [12, 25], feet: [[44, 41], [11, FLOOR]], reach: 1.5 },           // low kick
  /* 5 */ stance(18, 27, 9, 28),                                                  // half-way into the stance
  /* 6 */ stance(18, 30, 4),                                                      // the fighting stance
  /* 7 */ { hip: [7, 23], feet: [[8, FLOOR], [6, FLOOR]] },                       // standing at ease
  /* 8 */ { hip: [9, 23.5], feet: [[12, FLOOR], [6, FLOOR]] },
  /* 9 */ { hip: [9, 26], feet: [[15, FLOOR], [3, FLOOR]], splay: true },
  /* 10 */ stance(16, 27, 4, 29),
  /* 11 */ { hip: [8, 23], feet: [[8, FLOOR], [6, FLOOR]] },                      // the bow
  /* 12 */ { hip: [7, 23], feet: [[8, FLOOR], [6, FLOOR]] },
  /* 13 */ { hip: [6, 23], feet: [[8, FLOOR], [6, FLOOR]] },
  /* 14 */ { hip: [8, 37], feet: [[29, FLOOR], [-1, FLOOR]], splay: true },       // struck down: sitting back...
  /* 15 */ { hip: [9, 41], feet: [[31, FLOOR], [27, 43]] },
  /* 16 */ { hip: [9, 44], feet: [[32, FLOOR], [29, 44]] },
  /* 17 */ { hip: [4, 44], feet: [[25, 37], [24, 44]] },                          // ... the legs fly up
  /* 18 */ { hip: [3, 44], feet: [[26, 42], [25, FLOOR]] },
  /* 19 */ { hip: [3, 45], feet: [[27, FLOOR], [25, FLOOR]] },                    // down for good
  /* 20 */ stance(18, 30, 4),
  /* 21 */ stance(18, 30, 4),
  /* 22 */ stance(20, 32, 3),                                                     // the lunge of a punch
  /* 23 */ stance(20, 31, 4),                                                     // 23-33: the steps of the shuffle, forward and back
  /* 24 */ stance(26, 32, 6),
  /* 25 */ stance(16, 18, 4, 29),
  /* 26 */ { hip: [11, 29], feet: [[14, FLOOR], [8, 43]] },
  /* 27 */ stance(8, 16, 3, 29),
  /* 28 */ stance(14, 18, 4, 29),
  /* 29 */ { hip: [11, 29], feet: [[15, 43], [9, FLOOR]] },
  /* 30 */ stance(6, 16, 2, 29),
  /* 31 */ stance(10, 30, 3),
  /* 32 */ stance(16, 31, 3),
  /* 33 */ stance(12, 17, 5, 28),
  /* 34 */ { hip: [20, 27], feet: [[29, FLOOR], [4, 40]] },                       // running
  /* 35 */ { hip: [23, 27], feet: [[37, 42], [5, 44]], splay: true },
  /* 36 */ { hip: [20, 26], feet: [[21, FLOOR], [9, 40]] },
  /* 37 */ { hip: [14, 25], feet: [[19, FLOOR], [5, FLOOR]], splay: true },       // walking, and coming to a stop
  /* 38 */ { hip: [14, 25], feet: [[16, FLOOR], [9, 44]] },
  /* 39 */ { hip: [13, 23], feet: [[14, FLOOR], [12, FLOOR]] },
  /* 40 */ { hip: [16, 25], feet: [[18, FLOOR], [7, 45]] },
  /* 41 */ { hip: [3, 25], feet: [[7, FLOOR], [-3, 45]] },
];
const BODY = [
  /* 0 */ { lean: -8, hands: G }, /* 1 */ { lean: -8, hands: G },
  /* 2 */ { lean: -38, hands: [[9, -9], [-7, -9]] },
  /* 3 */ { lean: -16, hands: [[10, -12], [1, -8]] },
  /* 4 */ { lean: -6, hands: G },
  /* 5 */ { lean: 0, hands: G, wide: true }, /* 6 */ { lean: 0, hands: G, wide: true },
  /* 7 */ { lean: 0, hands: DOWN }, /* 8 */ { lean: 0, hands: DOWN },
  /* 9 */ { lean: -4, hands: [[8, -6], [3, -3]] },
  /* 10 */ { lean: 0, hands: G, wide: true },
  /* 11 */ { lean: 0, hands: DOWN }, /* 12 */ { lean: 26, hands: [[4, 2], [2, 2]] }, /* 13 */ { lean: 48, hands: [[7, 1], [5, 1]] },
  /* 14 */ { lean: -14, hands: [[10, -14], [-5, -9]] },
  /* 15 */ { lean: -40, hands: [[8, -12], [-8, -2]] },
  /* 16 */ { lean: -60, hands: [[4, -10], [-10, 0]] },
  /* 17 */ { lean: -76, hands: [[-2, -8], [-14, 1]] },
  /* 18 */ { lean: -88, hands: [[-5, 1], [-13, 1]], lying: 1 },
  /* 19 */ { lean: -90, hands: [[-6, 2], [-13, 2]], lying: 2 },
  /* 20 */ { lean: 0, hands: [[11, -11], [7, -6]], wide: true },
  /* 21 */ { lean: 0, hands: [[8, -9], [5, -5]] },
  /* 22 */ { lean: 8, hands: [[23, -17], [2, -7]], legs: 22 },                    // punches from the lunge: high, middle, low
  /* 23 */ { lean: 8, hands: [[23, -11], [2, -7]], legs: 22 },
  /* 24 */ { lean: 10, hands: [[22, -4], [2, -7]], legs: 22 },
  /* 25 */ { lean: 4, hands: [[24, -18], [3, -7]], legs: 6 },                     // punches from the stance
  /* 26 */ { lean: 4, hands: [[24, -12], [3, -7]], legs: 6 },
  /* 27 */ { lean: 6, hands: [[23, -5], [3, -7]], legs: 6 },
  /* 28 */ { lean: 0, hands: [[7, -23], [6, -8]], wide: true, legs: 6 },          // the high block
  /* 29 */ { lean: 0, hands: [[13, -3], [6, -8]], wide: true, legs: 6 },          // the low block
  /* 30 */ { lean: 14, hands: [[9, -8], [-5, -3]], legs: 34 },                    // running
  /* 31 */ { lean: 14, hands: [[-4, -3], [9, -9]], legs: 34 },
  /* 32 */ { lean: 8, hands: [[5, -6], [-3, -2]], legs: 36 },
  /* 33 */ { lean: 2, hands: [[5, 1], [-4, 1]], legs: 37 },
  /* 34 */ { lean: 2, hands: [[-3, 2], [4, 1]], legs: 38 },
  /* 35 */ { lean: 0, hands: DOWN, legs: 39 },
  /* 36 */ { lean: 2, hands: [[3, 1], [-2, 2]], legs: 40 },
  /* 37 */ { lean: 0, hands: DOWN, legs: 41 },
];
// the body a pair of legs is drawn with when the two are worked out together
const BODY_OF = (l) => (l <= 21 ? l : l <= 33 ? 6 : l <= 35 ? 30 : l - 4);

// ---- the heads -----------------------------------------------------------------------------
// '#' white, 's' skin, 'k' black, 'c' a coloured line. The hero is 7 pixels wide. The others have the same face
// inside a box of 21 by 10 (the face at column 7, row 2), with room for what they wear on their heads.
export const HERO = ['.####..', '######.', 'cccccc.', '###sss.', '##sksk.', '##ssss.', '.#sss..', '..ss...'];
const FACE = ['', '', '.......######.', '.......###sss.', '.......##sksk.', '.......##ssss.', '........#sss..', '.........ss...', '', ''];   // rows 2..7 of the others' heads, to be completed by a hat
const HATS = [
  // 0 a guard: hair tied in a knot
  ['.........##..', '........####.', '.......#####.', '.......###'],
  // 1 a wide straw hat
  ['..........#..........', '........#####........', '.....###########.....', '..#################..', '.......ccccccc'],
  // 2 a helmet with horns
  ['.....#.........#.....', '.....##.......##.....', '......###...###......', '......#########......', '......#ccccccc#', '......##'],
  // 3 a hood that leaves only the eyes
  ['........####.', '.......######.', '.......######.', '.......###kkk.', '.......##sksk.', '.......######.', '.......######.', '........####'],
  // 4 a helmet with a tall crest
  ['..........c..', '.........ccc.', '.......######.', '......########', '......#ccccccc', '......##'],
  // 5 a headband with long tails
  ['.......', '.......', '........####.', '..c....######.', '.c.cccccccccc.', '....c..###'],
  // 6 the great helmet: horns and a neck guard
  ['...#.............#...', '...##...........##...', '....###.......###....', '.....###########.....', '.....#ccccccccc#.....', '.....##........', '....###.', '...####.'],
];
function grid(rows, w) { return rows.map((r) => [...r.padEnd(w, '.')]); }
function hatHead(i) {
  const g = grid(Array.from({ length: 10 }, (_, y) => FACE[y] || ''), 21);
  HATS[i].forEach((r, y) => [...r].forEach((ch, x) => { if (ch !== '.') g[y][x] = ch; }));
  return g;
}
// a head lying on its back: turned a quarter, the crown to the left and the face up
function lying(g, fx, fy, fw, fh, w, h, ox, oy) {      // the face's box in g → put at (ox, oy) of a new w by h grid
  const out = Array.from({ length: h }, () => Array(w).fill('.'));
  g.forEach((row, y) => row.forEach((ch, x) => { if (ch === '.') return; const X = (y - fy) + ox, Y = (fx + fw - 1 - x) + oy; if (out[Y]?.[X] !== undefined) out[Y][X] = ch; }));
  return out;
}
function headShape(g) {
  const c = new Canvas(g[0].length, g.length);
  g.forEach((row, y) => row.forEach((ch, x) => { if (ch === '#') c.put(x, y, 'w'); else if (ch === 's' || ch === 'c') c.put(x, y, x & 1 ? 1 : 0); else if (ch === 'k') c.put(x, y, 'k'); }));
  return c.shape(0, 0, Math.ceil(g[0].length / 7), g.length);
}

// ---- the figure ----------------------------------------------------------------------------
const add = (p, d, l = 1) => [p[0] + d[0] * l, p[1] + d[1] * l];
function ik(root, target, l1, l2, pick, reach = 1.12) {      // the middle joint of a two-bone limb; pick chooses between the two ways it can bend
  let dx = target[0] - root[0], dy = target[1] - root[1], d = Math.hypot(dx, dy) || 1e-6;
  const most = (l1 + l2) * reach;
  if (d > most) { dx *= most / d; dy *= most / d; d = most; }
  const k = d > l1 + l2 ? d / (l1 + l2) : 1; l1 *= k; l2 *= k;
  const a = Math.acos(Math.max(-1, Math.min(1, (l1 * l1 + d * d - l2 * l2) / (2 * l1 * d))));
  const way = (t) => [root[0] + (dx * Math.cos(t) - dy * Math.sin(t)) / d * l1, root[1] + (dx * Math.sin(t) + dy * Math.cos(t)) / d * l1];
  const p = way(a), q = way(-a);
  return { joint: pick(p) >= pick(q) ? p : q, end: [root[0] + dx, root[1] + dy] };
}
class Figure {
  constructor() { this.x0 = -40; this.y0 = -8; this.w = 130; this.h = 70; this.lit = new Int8Array(this.w * this.h).fill(-1); this.own = new Uint8Array(this.w * this.h); this.skin = new Uint8Array(this.w * this.h); }
  at(x, y) { return (y - this.y0) * this.w + (x - this.x0); }
  limb(a, b, w1, w2, skin, owner, edge = true, rootFree = 0) {
    const ex = b[0] - a[0], ey = b[1] - a[1], L = ex * ex + ey * ey || 1e-6, set = [];
    const inside = (x, y, grow) => { let u = ((x - a[0]) * ex + (y - a[1]) * ey) / L; u = Math.max(0, Math.min(1, u)); return Math.hypot(x - a[0] - u * ex, y - a[1] - u * ey) <= (w1 + (w2 - w1) * u) / 2 + grow; };
    for (let y = Math.floor(Math.min(a[1], b[1]) - 8); y <= Math.max(a[1], b[1]) + 8; y++) for (let x = Math.floor(Math.min(a[0], b[0]) - 8); x <= Math.max(a[0], b[0]) + 8; x++) {
      if (x < this.x0 || y < this.y0 || x >= this.x0 + this.w || y >= this.y0 + this.h) continue;
      if (inside(x, y, 0)) set.push([x, y, skin ? (x & 1 ? 1 : 0) : 1, skin]);
      else if (edge && this.lit[this.at(x, y)] === 1 && inside(x, y, 1) && Math.hypot(x - a[0], y - a[1]) > rootFree) set.push([x, y, 0]);
    }
    for (const [x, y, v, sk] of set) { const i = this.at(x, y); this.lit[i] = v; if (sk !== undefined) { this.skin[i] = sk ? 1 : 0; this.own[i] = sk ? owner : this.own[i] | owner; } }
  }
  // a black band two pixels thick across what is already painted
  belt(a, b, owner, thick = 2) {
    const n = Math.ceil(Math.hypot(b[0] - a[0], b[1] - a[1]) * 2) || 1, steep = Math.abs(b[1] - a[1]) > Math.abs(b[0] - a[0]);
    for (let i = 0; i <= n; i++) for (let t = 0; t < thick; t++) { const x = Math.round(a[0] + (b[0] - a[0]) * i / n) + (steep ? t : 0), y = Math.round(a[1] + (b[1] - a[1]) * i / n) + (steep ? 0 : t), k = this.at(x, y); if (this.lit[k] === 1) this.lit[k] = 0; }
  }
  // on this screen one unlit pixel between two lit ones shows as a colour, not as black: a black line must be two pixels wide
  finish() {
    for (let y = this.y0; y < this.y0 + this.h; y++) for (let x = this.x0 + 2; x < this.x0 + this.w - 3; x++) {
      const i = this.at(x, y), v = this.lit;
      if (this.skin[i] || this.skin[i - 1] || this.skin[i + 1]) continue;
      if (v[i] === 1 && v[i - 1] !== 1 && v[i + 1] !== 1) { v[i] = -1; continue; }             // and a lone lit pixel would be a coloured speck
      if (v[i] === 0 && v[i - 1] === 1 && v[i + 1] === 1) { if (v[i + 2] === 1) v[i + 1] = 0; else if (v[i - 2] === 1) v[i - 1] = 0; else v[i] = 1; }
    }
  }
}
const L = 1, B = 2;
// the whole fighter in one pose: legs l with body u. Returns the painted figure and where the head goes.
function figure(l, u) { return paint(LEGS[l], BODY[u]); }
// ... and in any pose at all (tools/art/scenery1.mjs draws the hero at the cliff with it)
export function paint(lp, bp) {
  const f = new Figure(), hip = lp.hip;
  const a = bp.lean * Math.PI / 180, up = [Math.sin(a), -Math.cos(a)], fwd = [Math.cos(a), Math.sin(a)];
  const shoulder = add(hip, up, LEN.torso - 1.5), neck = add(hip, up, LEN.torso), head = add(neck, up, LEN.neck - 1);
  const flat = Math.abs(bp.lean) > 70;
  const knee = (foot, front) => ik(hip, foot, LEN.thigh, LEN.shin, flat ? (p) => -p[1] : !front && lp.splay ? (p) => -p[0] : (p) => p[0], front ? (lp.reach ?? 1.12) : 1.12);
  const [front, back] = lp.feet.map((p, i) => ({ ...knee(p, i === 0), back: i === 1 }));
  const leg = (s, edge) => {
    f.limb(hip, s.joint, 7.6, 6, false, L, edge, 7); f.limb(s.joint, s.end, 5.8, 4.4, false, L, edge, 4.5);
    const toe = flat ? [0.3, -1] : [s.back && lp.splay ? -1 : 1, 0.25];
    f.limb(add(s.end, [0, 1]), add(add(s.end, [0, 1]), toe, 4.4), 3, 2.4, true, L, false);
  };
  const arm = (hand, edge) => {
    const h = [hip[0] + hand[0], hip[1] + hand[1]], e = ik(shoulder, h, LEN.upper, LEN.fore, bp.elbowUp ? (p) => -p[1] : (p) => p[1] - 0.3 * p[0], 1.5);
    f.limb(shoulder, e.joint, 4.8, 4.2, false, B, edge, 5); f.limb(e.joint, e.end, 4, 3.4, false, B, edge, 3.2);
    const d = Math.hypot(e.end[0] - e.joint[0], e.end[1] - e.joint[1]) || 1;
    f.limb(e.end, add(e.end, [(e.end[0] - e.joint[0]) / d, (e.end[1] - e.joint[1]) / d], 2.4), 4, 3.6, true, B, false);
  };
  leg(back, false);
  arm(bp.hands[1], false);
  const w = bp.wide ? 12.5 : 9.5;
  f.limb(add(hip, up, -1), add(hip, up, 0.5), w - 2, w - 2, false, L, false);                 // the seat of the trousers
  f.limb(add(hip, up, 1), shoulder, w, w + 0.6, false, B, false);                               // the jacket
  f.limb(shoulder, neck, w - 1.5, 4.5, false, B, false);
  leg(front, Math.abs(lp.feet[0][0] - lp.feet[1][0]) + Math.abs(lp.feet[0][1] - lp.feet[1][1]) > 5);
  f.limb(add(hip, up, 1.5), add(hip, up, 4.5), w, w, false, B, false);                         // the skirt of the jacket hangs over the legs
  f.belt(add(add(hip, up, 3.4), fwd, -w / 2 - 1.5), add(add(hip, up, 3.4), fwd, w / 2 + 1.5), B);
  if (bp.wide) f.belt(add(add(neck, up, -1.5), fwd, 1.5), add(add(neck, up, -6.5), fwd, -0.5), B, 1);   // the fold of the jacket
  arm(bp.hands[0], true);
  f.finish();
  for (let y = 50; y < f.y0 + f.h; y++) for (let x = f.x0; x < f.x0 + f.w; x++) f.lit[f.at(x, y)] = -1;   // nothing may reach below the strip the game repaints
  return { f, head, hip };
}

// ---- cutting the figure into the game's pieces -----------------------------------------------
// (only when this file is run, not when its figure is borrowed)
if (import.meta.url === `file://${process.argv[1]}`) {
const entries = [], table = new Map();           // address → byte, for the pose tables
let free = 0xc000;
const store = (bytes, what) => { const at = free; entries.push([at, bytes, what]); free += bytes.length; return at; };
const poke = (at, v) => table.set(at, v & 0xff);
const layer = (f, owner) => { const c = new Canvas(f.w, f.h, f.x0, f.y0); for (let y = f.y0; y < f.y0 + f.h; y++) for (let x = f.x0; x < f.x0 + f.w; x++) { const i = f.at(x, y); if (f.lit[i] === 1 && f.own[i] & owner) c.put(x, y, 1); } return c; };

const heroHead = store(headShape(grid(HERO, 7)), 'the hero\'s head');
const HERO_W = grid(HERO, 7);
const heroLying = store(headShape(lying(HERO_W, 0, 0, 7, 8, 14, 7, 3, 0)), 'the hero\'s head, lying');
const none = store([0, 0], 'nothing');
[heroHead, heroLying, heroLying].forEach((p, y) => { poke(0x61cd + y, p); poke(0x61ca + y, p >> 8); poke(0x61d9 + y, 0); poke(0x61d6 + y, 0); });
[[0x61d3, 0x61d0], [0x62eb, 0x62e8], [0x62f1, 0x62ee], [0x62f7, 0x62f4], [0x62fd, 0x62fa], [0x6aed, 0x6aea], [0x6af3, 0x6af0]].forEach(([lo, hi], i) => {
  const g = hatHead(i), up = store(headShape(g), 'opponent ' + i + ': head'), down = store(headShape(lying(g, 7, 2, 7, 8, 21, 11, 10, 2)), 'opponent ' + i + ': head, lying');
  [up, down, down].forEach((p, y) => { poke(lo + y, p); poke(hi + y, p >> 8); });
});

for (let l = 0; l < LEGS.length; l++) {
  const u = BODY_OF(l), { f, head, hip } = figure(l, u), c = layer(f, L), [col, row, wide, rows] = c.bounds();
  const p = store(c.shape(col, row, wide, rows), 'legs ' + l);
  poke(0x6000 + l, p); poke(0x602a + l, p >> 8); poke(0x6054 + l, col); poke(0x607e + l, row); poke(0x62be + l, 0);
  const anchor = Math.round(hip[0]);
  poke(0x6150 + l, Math.floor(anchor / 7)); poke(0x617a + l, anchor - 7 * Math.floor(anchor / 7));
  // the mask: black behind everything of the fighter that is over the floor (the floor starts at row 39), and a line of shadow under him
  const mask = new Canvas(f.w, f.h, f.x0, f.y0); let x1 = 1e9, x2 = -1e9;
  for (let y = 38; y < 50; y++) for (let x = f.x0 + 1; x < f.x0 + f.w - 1; x++) if (f.lit[f.at(x, y)] >= 0) { for (let d = -1; d <= 1; d++) mask.put(x + d, y, 1); mask.put(x, y - 1, 1); if (y >= 44) { x1 = Math.min(x1, x); x2 = Math.max(x2, x); } }
  if (BODY[u].lying || Math.abs(BODY[u].lean) > 55) mask.rect(Math.round(head[0]) - 6, Math.max(38, Math.round(head[1]) - 5), Math.round(head[0]) + 6, 49, 1);
  if (x2 > x1) mask.rect(x1 - 1, 49, x2 + 4, 50, 1);
  const mb = mask.bounds();
  if (mb) { const mp = store(mask.shape(mb[0], mb[1], mb[2], mb[3], { mask: true }), 'floor mask ' + l); poke(0x60a8 + l, mp); poke(0x60d2 + l, mp >> 8); poke(0x60fc + l, mb[0] - col); poke(0x6126 + l, mb[1]); }
  else { poke(0x60a8 + l, 0); poke(0x60d2 + l, 0); }
}
for (let u = 0; u < BODY.length; u++) {
  const bp = BODY[u], l = bp.legs ?? u, { f, head, hip } = figure(l, u), anchor = Math.round(hip[0]);
  const c = new Canvas(f.w, f.h, f.x0 - anchor, f.y0);                    // the same picture, counted from the hip
  for (let y = f.y0; y < f.y0 + f.h; y++) for (let x = f.x0; x < f.x0 + f.w; x++) { const i = f.at(x, y); if (f.lit[i] === 1 && f.own[i] & B) c.put(x - anchor, y, 1); }
  const [col, row, wide, rows] = c.bounds(), p = store(c.shape(col, row, wide, rows), 'body ' + u);
  poke(0x6226 + u, p); poke(0x624c + u, p >> 8); poke(0x6272 + u, col); poke(0x6298 + u, row);
  if (bp.lying) {                                                         // a lying head goes by whole bytes
    const hx = Math.round(head[0]) - 8 - anchor, hc = Math.round(hx / 7);
    poke(0x61a4 + u, 6 + bp.lying); poke(0x61da + u, hc); poke(0x6200 + u, Math.round(head[1]) - 4);
  } else {
    const hx = Math.round(head[0] - 3) - anchor, hc = Math.floor(hx / 7);
    poke(0x61a4 + u, hx - 7 * hc); poke(0x61da + u, hc); poke(0x6200 + u, Math.round(head[1] - 4));
  }
}
// the two extra masks the drawing routine lays behind the arms in some poses: ours needs none
entries.push([0x9a18, [0, 0], 'unused'], [0x9a2a, [0, 0], 'unused']);
// the pose tables, as runs of bytes
const ats = [...table.keys()].sort((a, b) => a - b);
for (let i = 0; i < ats.length;) { let j = i; while (j + 1 < ats.length && ats[j + 1] === ats[j] + 1) j++; entries.push([ats[i], ats.slice(i, j + 1).map((a) => table.get(a)), 'pose tables']); i = j + 1; }
fs.writeFileSync(new URL('../../src/data/art_fighters.js', import.meta.url), dataFile('The fighters: their shapes, and the tables that say which shape each pose uses and where it goes.', entries));
console.log('fighters:', entries.length, 'entries, shapes up to $' + free.toString(16));
}
