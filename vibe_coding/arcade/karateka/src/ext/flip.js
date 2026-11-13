// Somersaults: jump and roll through the air, forwards (F) or backwards (D), in the spirit
// of The Way of the Exploding Fist. Our second addition to the game.
//
// Like the mawashi-geri, this only watches the original program and steps in while the
// player is in the air: the game's own fighter stays in his fighting stance, we move him,
// draw him ourselves and let the opponent's attacks pass under him.
//
// Every so often, when there is room for it, the somersault turns into a whole floor routine:
// a handspring into a double somersault twice as high, a stuck landing and a salute.
//
// The crane kick (C) is Arik Shemesh's idea, and it is the one from The Karate Kid (1984): the fighter
// stands on one leg with his knee up and his arms spread, hops, and snaps a front kick to the face
// with the leg he was standing on. The kick itself is judged by the game, as one of its own high
// kicks thrown from where he is, and a clean hit takes two points. On one leg he can be hit.
// The figure is not taken from the game: it is a jointed model drawn in tools/figure.mjs,
// posed after slow-motion footage of gymnasts and frozen into acro_art.js.
import { blitArt } from './mawashi.js';
import { ART } from './acro_art.js';

const KEY_F = 0x46, KEY_D = 0x44, KEY_C = 0x43;
const PATIENCE = 3;              // frames a key press waits for the fighter to be ready
const Z = { POSE: 0x20, TOP: 0x21, X: 0x22, STEP: 0x27, STEP2: 0x28, ATTACK: 0x2b, FACING: 0x2c, SHIFT: 0x50, PAGE: 0x07 };
const ENEMY_X = 0x72, ENEMY_ALIVE = 0x5e, PLAYER_ALIVE = 0x5d, ENEMY_HEALTH = 0xb7, SWISH = 0x45, STANCE = 6;
const HIGH_KICK = 0;             // the game's own number for the high kick, in its table of attacks
const MIN_X = 3, MAX_X = 0x22, MIN_GAP = 8;
const FLOOR = 161, HIP = 16;     // the floor row, and how far in front of the fighter's anchor his hips are
const SHOW_ODDS = 5;             // one somersault in so many becomes the floor routine

const travel = (seq) => ART[seq].reduce((n, f) => n + (f.move || 0), 0);

export function installFlip(m) {
  const r = m.ram, c = m.cpu;
  // `under` remembers what was on the screen above the strip the game repaints itself
  // (one list per graphics page), so a high jump leaves no trail on the scenery
  const st = { want: 0, polls: 0, frame: -1, held: 0, seq: 'front', dir: 0, inAir: false, flips: 0, shows: 0, dodged: 0, always: false, crane: false, cranes: 0, craneHits: 0, struck: false, landing: false, under: { 32: [], 64: [] } };
  m.flip = st;
  const player = () => r[Z.FACING] < 0x80;
  const flying = () => st.frame >= 0;
  const blocked = (x, dir) => dir > 0 ? (x > MAX_X || (r[ENEMY_ALIVE] && r[ENEMY_X] - x < MIN_GAP)) : x < MIN_X;
  const lucky = () => st.always || ((Math.imul(c.cycles >>> 3, 2654435761) >>> 16) % SHOW_ODDS) === 0;
  // put the scenery back, unless the game has repainted that spot itself in the meantime (it scrolls the fence)
  const restore = (page) => { const u = st.under[page]; for (let i = 0; i < u.length; i += 3) if (r[u[i]] === u[i + 2]) r[u[i]] = u[i + 1]; u.length = 0; };

  const pressKey = m.pressKey.bind(m);
  m.pressKey = (k) => {
    if ((k === KEY_F || k === KEY_D || k === KEY_C) && m.inGame) { st.want = k === KEY_D ? -1 : 1; st.crane = k === KEY_C; st.polls = PATIENCE; } else { st.want = 0; pressKey(k); }
  };

  // the player's input for this frame
  m.hook(0x6e00, [0xa5, 0xc4, 0xd0, 0x65], () => {
    // this is the first thing the game does in a frame: clean the page it is about to paint, before it scrolls the scenery
    if (st.under[r[Z.PAGE]]?.length) restore(r[Z.PAGE]);
    st.landing = false;
    if (!flying()) {
      if (!st.want || !player()) return false;
      if (st.polls-- <= 0) { st.want = 0; return false; }
      const ready = r[Z.POSE] === STANCE && r[Z.TOP] === STANCE && r[Z.STEP] === 0 && r[Z.STEP2] === 0 && r[PLAYER_ALIVE];
      if (!ready) return false;
      const plain = st.want > 0 ? 'front' : 'back', show = plain + 'Show';
      if (st.crane) { st.seq = 'crane'; st.cranes++; }
      else {
        const room = !blocked(r[Z.X] + st.want * travel(show), st.want);
        st.seq = room && lucky() ? show : plain;
        st.flips++; if (st.seq === show) st.shows++;
      }
      st.frame = 0; st.held = 0; st.dir = st.want; st.want = 0; st.first = true; st.struck = false;
    }
    c.cycles += 60; m.rts(); return true;   // in the air the keys do nothing
  });
  // one step of the fighter's animation: here, one frame of the flight
  m.hook(0x6400, [0xa9, 0xff, 0x85, 0x2b], () => {
    if (!flying() || !player()) return false;
    const seq = ART[st.seq];
    if (st.first) st.first = false;
    else if (st.held < (seq[st.frame].hold || 0)) { st.held++; return false; }
    else { st.frame++; st.held = 0; }
    if (st.frame >= seq.length) { st.frame = -1; st.inAir = false; return false; }
    const f = seq[st.frame];
    st.inAir = !!f.air;
    if (f.swish) r[SWISH] = 1;
    for (let i = 0; i < (f.move || 0); i++) { const x = r[Z.X] + st.dir; if (!blocked(x, st.dir)) r[Z.X] = x; }
    return false;
  });
  // draw the fighter
  m.hook(0x6811, [0xa6, 0x20, 0xa5, 0x50], () => {
    const page = r[Z.PAGE];
    if (st.under[page]?.length && (player() || !flying())) restore(page);
    if (!flying() || !player()) return false;
    const f = ART[st.seq][st.frame];
    blitArt(r, page << 8, 7 * r[Z.X] + r[Z.SHIFT] + HIP, f.rows, FLOOR + f.top, f.ax, st.under[page]);
    c.cycles += 17120 + 1232 * r[Z.SHIFT];
    m.rts(); return true;
  });
  // the opponent's attack on the player has just been judged (result in A): in the air it misses
  m.hook(0xb963, [0x85, 0x40, 0x20, 0x1e, 0x70], () => {
    if (flying() && st.inAir && c.a !== 0) { c.a = 0; st.dodged++; }
    return false;
  });
  // the game is about to judge the player's attack on the opponent: on the frame the crane kick
  // snaps out, it judges a high kick (the fighter himself is still in his stance, attacking nothing)
  m.hook(0xb942, [0xa4, 0x6b, 0x20, 0x66, 0x73], () => {
    if (!flying() || st.struck || !ART[st.seq][st.frame].strike) return false;
    st.struck = st.landing = true;
    c.y = HIGH_KICK; c.z = 1; c.n = 0; c.pc = 0xb944; c.cycles += 3; return true;
  });
  // the opponent takes the hit: a crane kick takes a second point
  m.hook(0x7292, [0x20, 0x0c, 0x0b], () => {
    if (st.landing) { st.landing = false; st.craneHits++; if (r[ENEMY_HEALTH] > 1) r[ENEMY_HEALTH]--; }
    return false;
  });
}
