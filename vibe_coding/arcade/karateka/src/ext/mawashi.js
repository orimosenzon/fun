// Mawashi-geri (the roundhouse kick): our first addition to the game, on the E key.
//
// The original program keeps running untouched. This module watches a few of its
// addresses and steps in only while the player performs the new kick:
//   - the kick borrows the game's own middle-kick sequence (same reach, same hit test)
//   - it is drawn here, from our own two poses (chamber and kick), instead of by the game
//   - it is slower than the other kicks (the chamber and the kick are each held one more
//     frame) and a clean hit takes two points off the opponent instead of one
// With E never pressed, nothing in here changes a single byte or cycle of the game.
import { CHAMBER, KICK, ART_Y, ART_ANCHOR } from './mawashi_art.js';

const KEY_E = 0x45, KEY_S = 0x53;
const ARM_WINDOW = 400000;      // cycles (about 0.4 s) the game has to accept the key press
const POSE = { STANCE: 6, STEP: 5, CHAMBER: 1, KICK: 3 };
// zero page of the fighter being worked on: pose, animation index, attack id, facing, x
const Z = { POSE: 0x20, X: 0x22, ANIM: 0x25, ATTACK: 0x2b, FACING: 0x2c, SHIFT: 0x50, PAGE: 0x07 };
const ENEMY_HEALTH = 0xb7, PLAYER_POSE = 0x60;

export function blitArt(mem, base, px0, rows, y0 = ART_Y, anchor = ART_ANCHOR) {
  rows.forEach((row, i) => {
    const y = y0 + i; if (y < 0 || y > 191) return;
    const b = base + (((y & 7) << 10) | (((y >> 3) & 7) << 7) | ((y >> 6) * 40));
    for (let c = 0; c < row.length; c++) {
      const ch = row[c]; if (ch === '.') continue;
      const x = px0 - anchor + c; if (x < 0 || x >= 280) continue;
      const a = b + ((x / 7) | 0), bit = 1 << (x % 7);
      mem[a] = ch === '#' ? mem[a] | bit | 0x80 : (mem[a] & ~bit) | 0x80;
    }
  });
}

export function installMawashi(m) {
  const r = m.ram, c = m.cpu;
  const st = { armedAt: -1, active: false, begun: false, holdChamber: 0, holdKick: 0, kicks: 0, hits: 0 };
  m.mawashi = st;
  const player = () => r[Z.FACING] < 0x80;
  const armed = () => st.armedAt >= 0 && c.cycles - st.armedAt < ARM_WINDOW;
  const begin = () => { st.active = true; st.holdChamber = 1; st.holdKick = 1; st.kicks++; };

  const pressKey = m.pressKey.bind(m);
  m.pressKey = (k) => {
    if (k === KEY_E && m.inGame) { st.armedAt = c.cycles; pressKey(KEY_S); } else { st.armedAt = -1; pressKey(k); }
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
    const art = r[Z.POSE] === POSE.CHAMBER ? CHAMBER : r[Z.POSE] === POSE.KICK ? KICK : null;
    if (!art) return false;
    blitArt(r, r[Z.PAGE] << 8, 7 * r[Z.X] + r[Z.SHIFT], art);
    c.cycles += 17120 + 1232 * r[Z.SHIFT]; // what the game spends drawing its own middle kick
    m.rts(); return true;
  });
  // the opponent takes the hit
  m.hook(0x7292, [0x20, 0x0c, 0x0b], () => {
    if (st.active && r[PLAYER_POSE] === POSE.KICK) { st.hits++; if (r[ENEMY_HEALTH] > 1) r[ENEMY_HEALTH]--; }
    return false;
  });
}
