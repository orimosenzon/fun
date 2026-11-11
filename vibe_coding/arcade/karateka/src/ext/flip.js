// Somersaults: jump and roll through the air, forwards (F) or backwards (D), in the spirit
// of The Way of the Exploding Fist. Our second addition to the game.
//
// Like the mawashi-geri, this only watches the original program and steps in while the
// player is in the air: the game's own fighter stays in his fighting stance, we move him
// one column per frame, draw him ourselves and let the opponent's attacks pass under him.
// The figure is not taken from the game: it is a small jointed model, turned and drawn here.
import { blitArt } from './mawashi.js';

const KEY_F = 0x46, KEY_D = 0x44;
const PATIENCE = 3;              // frames a key press waits for the fighter to be ready
const Z = { POSE: 0x20, TOP: 0x21, X: 0x22, STEP: 0x27, STEP2: 0x28, ATTACK: 0x2b, FACING: 0x2c, SHIFT: 0x50, PAGE: 0x07 };
const ENEMY_X = 0x72, ENEMY_ALIVE = 0x5e, PLAYER_ALIVE = 0x5d, SWISH = 0x45, STANCE = 6;
const MIN_X = 3, MAX_X = 0x22, MIN_GAP = 8;

// One entry per frame: turn of the body (degrees, forward roll), how tightly it is tucked
// (0 standing .. 1 ball), height off the ground in pixels, and whether a column is travelled.
const FRAMES = [
  { turn: 0, tuck: 0.45, up: -3, move: 0 },   // dip
  { turn: 20, tuck: 0.25, up: 9, move: 1 },   // take off
  { turn: 95, tuck: 1, up: 19, move: 1 },
  { turn: 180, tuck: 1, up: 23, move: 1 },
  { turn: 265, tuck: 1, up: 19, move: 1 },
  { turn: 335, tuck: 0.4, up: 8, move: 1 },   // opening up
  { turn: 0, tuck: 0.5, up: -3, move: 1 },    // landing
];
const AIR = (f) => f >= 1 && f <= 5;

const W = 44, H = 62, CX = 21, GROUND = 47; // art grid; the hip is at column CX, the ground at row GROUND
const cache = new Map();
const mix = (a, b, t) => [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t];

// The model, upright and facing right; y grows downwards, the hip is the origin.
function skeleton(t) {
  const hip = [0, 0], neck = mix([1, -13], [3, -11], t), head = mix([2, -19], [7, -15], t);
  const knee = mix([2, 10], [11, -3], t), foot = mix([0, 21], [6, 7], t), toe = mix([3, 22], [10, 9], t);
  const hand = mix([10, -10], [12, 2], t), elbow = mix([5, -5], [7, -4], t);
  return { hip, neck, head, knee, foot, toe, hand, elbow };
}
export function flipArt(turn, tuck, up, dir) {
  const key = [turn, tuck, up, dir].join();
  if (cache.has(key)) return cache.get(key);
  const g = Array.from({ length: H }, () => Array(W).fill('.'));
  const a = dir * turn * Math.PI / 180, cos = Math.cos(a), sin = Math.sin(a), sk = skeleton(tuck);
  const feet = Math.max(sk.foot[1], sk.toe[1]);           // standing: keep the feet on the ground
  const oy = turn === 0 ? GROUND - feet - up : GROUND - 13 - up; // on the ground the feet stay on the floor
  const P = ([x, y]) => [CX + x * cos - y * sin, oy + x * sin + y * cos];
  const limb = (p1, p2, t1, t2, ch) => {
    const [x1, y1] = P(p1), [x2, y2] = P(p2), dx = x2 - x1, dy = y2 - y1, L = dx * dx + dy * dy || 1;
    for (let r = 0; r < H; r++) for (let c = 0; c < W; c++) {
      let u = ((c - x1) * dx + (r - y1) * dy) / L; u = Math.max(0, Math.min(1, u));
      const d = Math.hypot(c - x1 - u * dx, r - y1 - u * dy), t = (t1 + (t2 - t1) * u) / 2;
      if (d <= t && (ch !== 'o' || g[r][c] === 'W')) g[r][c] = ch;
    }
  };
  limb(sk.hip, sk.neck, 9, 11, 'W');                      // trunk (drawn first, no outline)
  { const b = mix(sk.hip, sk.neck, 0.22), n = [sk.neck[1] - sk.hip[1], sk.hip[0] - sk.neck[0]], l = Math.hypot(...n) / 7;
    limb([b[0] - n[0] / l, b[1] - n[1] / l], [b[0] + n[0] / l, b[1] + n[1] / l], 2.2, 2.2, 'o'); } // black belt
  limb(sk.neck, sk.head, 6, 7, 'W');                       // head, hair
  limb(mix(sk.neck, sk.head, 0.75), [sk.head[0] + 3.5, sk.head[1] + 2], 4.4, 4, 's'); // face
  limb(sk.hip, sk.knee, 8, 6.6, '#');                      // thigh
  limb(sk.knee, sk.foot, 6.2, 5, '#');                     // shin
  limb(sk.foot, sk.toe, 3.2, 3, 's');                      // foot
  limb(sk.neck, sk.elbow, 4.6, 4, '#'); limb(sk.elbow, sk.hand, 4, 3.4, '#'); // arm
  limb(sk.hand, [sk.hand[0] + 2, sk.hand[1] + 0.5], 3.4, 3.4, 's');           // hand
  // skin is a single-pixel stripe pattern, which the Apple II shows as a colour
  const rows = g.map((row) => row.map((ch, c) => ch === 'W' ? '#' : ch === 's' ? (c & 1 ? '#' : 'o') : ch).join(''));
  cache.set(key, rows);
  return rows;
}
export const FLIP_FRAMES = FRAMES, FLIP_GRID = { W, H, CX, GROUND };

export function installFlip(m) {
  const r = m.ram, c = m.cpu;
  const st = { want: 0, polls: 0, frame: -1, dir: 0, flips: 0, dodged: 0 };
  m.flip = st;
  const player = () => r[Z.FACING] < 0x80;
  const flying = () => st.frame >= 0;

  const pressKey = m.pressKey.bind(m);
  m.pressKey = (k) => {
    if ((k === KEY_F || k === KEY_D) && m.inGame) { st.want = k === KEY_F ? 1 : -1; st.polls = PATIENCE; } else { st.want = 0; pressKey(k); }
  };

  // the player's input for this frame
  m.hook(0x6e00, [0xa5, 0xc4, 0xd0, 0x65], () => {
    if (!flying()) {
      if (!st.want || !player()) return false;
      if (st.polls-- <= 0) { st.want = 0; return false; }
      const ready = r[Z.POSE] === STANCE && r[Z.TOP] === STANCE && r[Z.STEP] === 0 && r[Z.STEP2] === 0 && r[PLAYER_ALIVE];
      if (!ready) return false;
      st.frame = 0; st.dir = st.want; st.want = 0; st.first = true; st.flips++;
    }
    c.cycles += 60; m.rts(); return true;   // in the air the keys do nothing
  });
  // one step of the fighter's animation: here, one frame of the flight
  m.hook(0x6400, [0xa9, 0xff, 0x85, 0x2b], () => {
    if (!flying() || !player()) return false;
    if (st.first) st.first = false; else st.frame++;
    if (st.frame >= FRAMES.length) { st.frame = -1; return false; }
    if (st.frame === 1) r[SWISH] = 1;
    if (FRAMES[st.frame].move) {
      const x = r[Z.X] + st.dir, gap = r[ENEMY_X] - x;
      const blocked = st.dir > 0 ? (x > MAX_X || (r[ENEMY_ALIVE] && gap < MIN_GAP)) : x < MIN_X;
      if (!blocked) r[Z.X] = x;
    }
    return false;
  });
  // draw the fighter
  m.hook(0x6811, [0xa6, 0x20, 0xa5, 0x50], () => {
    if (!flying() || !player()) return false;
    const f = FRAMES[st.frame];
    blitArt(r, r[Z.PAGE] << 8, 7 * r[Z.X] + r[Z.SHIFT] + 7, flipArt(f.turn, f.tuck, f.up, st.dir), 161 - GROUND, CX);
    c.cycles += 17120 + 1232 * r[Z.SHIFT];
    m.rts(); return true;
  });
  // the opponent's attack on the player has just been judged (result in A): in the air it misses
  m.hook(0xb963, [0x85, 0x40, 0x20, 0x1e, 0x70], () => {
    if (flying() && AIR(st.frame) && c.a !== 0) { c.a = 0; st.dodged++; }
    return false;
  });
}
