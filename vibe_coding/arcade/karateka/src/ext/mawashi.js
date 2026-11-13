// Mawashi-geri (the roundhouse kick): our first addition to the game, on the E key.
//
// The original program keeps running untouched. This module watches a few of its
// addresses and steps in only while the player performs the new kick:
//   - the kick borrows the game's own middle-kick sequence (same reach, same hit test)
//   - it is drawn here, from our own five frames (knee up, hips turning, two of the leg
//     at full reach, snap back), instead of by the game
//   - it is slower than the other kicks (the chamber and the kick are each held one more
//     frame) and a clean hit takes two points off the opponent instead of one
// With E never pressed, nothing in here changes a single byte or cycle of the game.
// The frames are posed after a Shotokan black belt filmed from the side (tools/flip_frames.mjs).
//
// Two more kicks work the same way, both Yaniv Waldman's idea:
//   - yoko-geri kekomi (Y), the side thrust kick: the longest kick there is, it lands from one
//     step further away than the others, for the usual one point
//   - ushiro-geri (U), the back kick: he has to turn his back first, so it is the slowest, and
//     the heel takes three points
import { ART } from './acro_art.js';

const KEY_S = 0x53;
// art: the frames in acro_art.js, of which the first `pre` lead up to the kick, two are the kick and one the way back;
// chamber / kick: how many more frames than the game's own kick each part is held; extra: points on top of the usual one;
// reach: how many columns further away than the game's kicks it still lands
const KICKS = {
  0x45: { art: 'mawashi', pre: 2, chamber: 1, kick: 1, extra: 1, reach: 0 },   // E
  0x59: { art: 'yoko', pre: 2, chamber: 1, kick: 1, extra: 0, reach: 2 },      // Y
  0x55: { art: 'ushiro', pre: 3, chamber: 2, kick: 1, extra: 2, reach: 0 },    // U
};
const ARM_WINDOW = 1500000;     // cycles (about 1.5 s) the game has to accept the key press
const POSE = { STANCE: 6, STEP: 5, CHAMBER: 1, KICK: 3 };
// zero page of the fighter being worked on: pose, animation index, attack id, facing, x
const Z = { POSE: 0x20, X: 0x22, ANIM: 0x25, ATTACK: 0x2b, FACING: 0x2c, SHIFT: 0x50, PAGE: 0x07 };
const ENEMY_HEALTH = 0xb7, PLAYER_POSE = 0x60, PLAYER_ATTACK = 0x6b, GAP = 0x33;
const FLOOR = 161, HIP = 16;    // the floor row, and how far in front of the fighter's anchor his hips are
const REPAINTED = [114, 163];   // the rows the game paints afresh every frame

// Paint a frame of our art: px0 is where its `anchor` column goes, y0 its first row.
// Above the strip the game repaints, every byte we change is noted in `under` as
// address, what was there, what we left there, so that it can be put back later.
export function blitArt(mem, base, px0, rows, y0, anchor, under) {
  const touched = under && new Map();
  rows.forEach((row, i) => {
    const y = y0 + i; if (y < 0 || y > 191) return;
    const b = base + (((y & 7) << 10) | (((y >> 3) & 7) << 7) | ((y >> 6) * 40));
    const keep = touched && (y < REPAINTED[0] || y > REPAINTED[1]);
    for (let c = 0; c < row.length; c++) {
      const ch = row[c]; if (ch === '.') continue;
      const x = px0 - anchor + c; if (x < 0 || x >= 280) continue;
      const a = b + ((x / 7) | 0), bit = 1 << (x % 7);
      if (keep && !touched.has(a)) touched.set(a, mem[a]);
      mem[a] = ch === '#' ? mem[a] | bit | 0x80 : (mem[a] & ~bit) | 0x80;
    }
  });
  if (touched) for (const [a, was] of touched) under.push(a, was, mem[a]);
}

export function installMawashi(m) {
  const r = m.ram, c = m.cpu;
  // chambers / reaches: how many frames of each have been shown in this kick, which picks the art
  const st = { armedAt: -1, want: 0x45, kind: 0x45, active: false, begun: false, holdChamber: 0, holdKick: 0, chambers: 0, reaches: 0, kicks: 0, hits: 0, far: 0 };
  m.mawashi = st;
  const player = () => r[Z.FACING] < 0x80;
  const armed = () => st.armedAt >= 0 && c.cycles - st.armedAt < ARM_WINDOW;
  const kind = () => KICKS[st.kind];
  const begin = () => { st.kind = st.want; st.active = true; st.holdChamber = kind().chamber; st.holdKick = kind().kick; st.chambers = 0; st.reaches = 0; st.kicks++; };

  const pressKey = m.pressKey.bind(m);
  m.pressKey = (k) => {
    if (KICKS[k] && m.inGame) { st.armedAt = c.cycles; st.want = k; pressKey(KEY_S); } else { st.armedAt = -1; pressKey(k); }
  };

  // a kick starts from the fighting stance
  m.hook(0x6684, [0xa9, 0x07, 0x85, 0x25], () => {
    if (!player()) return false;
    if (armed()) { begin(); st.begun = false; } else st.active = false;
    st.armedAt = -1; return false;
  });
  // another kick asked for while the leg is still up
  m.hook(0x66a1, [0xe6, 0x24], () => {
    if (!player()) return false;
    if (armed()) begin(); else st.active = false;
    st.armedAt = -1; return false;
  });
  // one step of the fighter's animation
  m.hook(0x6400, [0xa9, 0xff, 0x85, 0x2b], () => {
    if (!st.active || !player()) return false;
    const pose = r[Z.POSE], anim = r[Z.ANIM];
    if (pose === POSE.STEP) st.begun = true;
    else if (st.begun && pose !== POSE.CHAMBER && pose !== POSE.KICK) { st.active = false; return false; }
    if (pose === POSE.CHAMBER && anim === 1 && st.holdChamber > 0) st.holdChamber--;
    else if (pose === POSE.KICK && anim === 2 && st.holdKick > 0) st.holdKick--;
    else return false;
    r[Z.ATTACK] = 0xff; c.cycles += 79; m.rts(); return true;
  });
  // draw the fighter
  m.hook(0x6811, [0xa6, 0x20, 0xa5, 0x50], () => {
    if (!st.active || !player()) return false;
    const pose = r[Z.POSE];
    const k = kind();
    let n;
    if (pose === POSE.CHAMBER) n = st.reaches ? k.pre + 2 : Math.min(st.chambers++, k.pre - 1);   // the way up to the kick, frame by frame; after the kick, the snap back
    else if (pose === POSE.KICK) n = k.pre + Math.min(st.reaches++, 1);
    else return false;
    const f = ART[k.art][n];
    blitArt(r, r[Z.PAGE] << 8, 7 * r[Z.X] + r[Z.SHIFT] + HIP, f.rows, FLOOR + f.top, f.ax);
    c.cycles += 17120 + 1232 * r[Z.SHIFT]; // what the game spends drawing its own middle kick
    m.rts(); return true;
  });
  // the opponent takes the hit
  m.hook(0x7292, [0x20, 0x0c, 0x0b], () => {
    if (st.active && r[PLAYER_POSE] === POSE.KICK) { st.hits++; for (let i = 0; i < kind().extra; i++) if (r[ENEMY_HEALTH] > 1) r[ENEMY_HEALTH]--; }
    return false;
  });
  // the game is about to judge the player's attack, and has just measured the distance between the two:
  // the side thrust kick is judged as if the opponent stood a step nearer
  m.hook(0xb942, [0xa4, 0x6b, 0x20, 0x66, 0x73], () => {
    const k = kind();
    if (st.active && k.reach && r[PLAYER_ATTACK] !== 0xff && r[PLAYER_POSE] === POSE.KICK && r[GAP] > 9 && r[GAP] <= 9 + k.reach) { r[GAP] -= k.reach; st.far++; }
    return false;
  });
}
