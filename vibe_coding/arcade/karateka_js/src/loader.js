// Building the game's memory from nothing but this project's own data. No disk, no file of the
// original: the pictures are drawn by tools/art, the tunes are written in data/tunes.js, the small
// tables the drawing routines need are computed here, and the rule tables are plain numbers.
import rules1 from './data/rules1.js';
import fighters from './data/art_fighters.js';
import scenery1 from './data/art_scenery1.js';
import { tunes, thud } from './data/tunes.js';

const SCENES = { 1: { rules: rules1, art: [...fighters, ...scenery1] } };
const bytes = (hex) => Uint8Array.from(hex.match(/../g) ?? [], (h) => parseInt(h, 16));

export function buildMemory(n = 1) {
  const scene = SCENES[n];
  if (!scene) throw new Error('scene ' + n + ' is not written yet');
  const m = new Uint8Array(0x10000);
  m.fill(0x80, 0x2000, 0x6000);                                  // both screens black
  // where each of the 192 rows of the screen starts, counted from the start of the page
  for (let y = 0; y < 192; y++) { const at = ((y & 7) << 10) | (((y >> 3) & 7) << 7) | ((y >> 6) * 40); m[0x0800 + y] = at & 0xff; m[0x08c0 + y] = at >> 8; }
  // a byte of seven pixels, turned left to right
  for (let b = 0; b < 128; b++) { let r = 0; for (let i = 0; i < 7; i++) if (b & (1 << i)) r |= 0x40 >> i; m[0x0980 + b] = r | 0x80; }
  // the pixels a shifted shape leaves alone at its two edges
  for (let s = 0; s < 7; s++) { m[0x1a34 + s] = 0x80 | ((1 << s) - 1); m[0x1a3b + s] = (0xff << s) & 0xff; }
  for (const [at, hex] of scene.rules) m.set(bytes(hex), at);
  for (const [at, hex] of scene.art) m.set(bytes(hex), at);
  for (const [at, data] of tunes(n)) m.set(data, at);
  m.set(thud(), 0x0e00); m.set([0x0e, 0xff, 0x0e], 0x0dfb);     // the sound of a blow: one-bit samples, and where they start and end
  m[0x0d5e] = 6; m[0x0d79] = 0xea; m[0x0d7a] = 0xea; m[0x0d88] = 0xc0;   // the tune player's settings
  m[0x07] = 0x40; m[0x3f] = 0x40; m[0xe4] = 0x30; m[0xe5] = 0x30;
  m[0x48] = 0x46;                                               // the seed of the opponent's dice
  m[0x4f] = 1; m[0x86] = 1;                                     // sound on
  m[0xd0] = n;                                                  // which scene
  m[0xc4] = 1;                                                  // played from the keyboard
  return m;
}
