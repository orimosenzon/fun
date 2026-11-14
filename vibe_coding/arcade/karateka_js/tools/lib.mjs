// Development helpers shared by the tools: a scene's memory (from the original disk, as a measuring
// reference, or from our own data), and a bot that plays the first scene.
import fs from 'node:fs';
import { buildMemory } from '../src/loader.js';

import scene1 from './ref/scene1.map.js';
const MAPS = { 1: scene1 };
const DISK = new URL('../../karateka/data/karateka.dsk', import.meta.url);

// ours = the game's own data; otherwise the original's memory at the start of the scene, as a reference
export function memoryFor(ours, scene = 1) {
  return ours ? buildMemory(scene) : fromDisk(new Uint8Array(fs.readFileSync(DISK)), scene);
}

// the original's memory at the start of a scene, put together from the disk image (tools/ref/*.map.js, made by tools/bake.mjs);
// the original's program is wiped, only its tables and pictures stay
export function fromDisk(dsk, scene = 1, keep) {
  const map = MAPS[scene];
  const m = new Uint8Array(0x10000);
  m.fill(0x80, 0x2000, 0x6000);
  for (const [page, sector] of map.pages) m.set(dsk.subarray(sector * 256, sector * 256 + 256), page * 256);
  for (const [addr, value] of map.patches) m[addr] = value;
  if (keep) m.set(keep.subarray(0, 0x200), 0); else m.set(map.low, 0);
  for (const [addr, len] of map.wipe) m.fill(0, addr, addr + len);
  m[0xc4] = 1;
  return m;
}
export { MAPS };

const u8 = (v) => v & 0xff;
// plays until the scene ends; opt: seed, cheat (the player cannot lose), frames, end (show the closing picture), each(game, n)
export function botPlay(g, { seed = 1, cheat = false, frames = 6000, end = false, each } = {}) {
  const raw = g.m;
  let rnd = seed * 2654435761 >>> 0; const rand = (n) => { rnd = (Math.imul(rnd, 1664525) + 1013904223) >>> 0; return (rnd >>> 8) % n; };
  const KEYS = { RIGHT: 0x95, LEFT: 0x88, SPACE: 0xa0 }, POOL = ['RIGHT', 'RIGHT', 'RIGHT', 'RIGHT', 'LEFT', 'W', 'S', 'X', 'Q', 'A', 'Z', 'SPACE', 'B', 'W', 'S', 'RIGHT'];
  g.rB700(); let n = 0, res = 'frame';
  for (; n < frames && res === 'frame'; n++) {
    if (cheat) { if (raw[0x5d]) raw[0xb6] = raw[0xb0]; if (raw[0xb7] > 1) raw[0xb7] = 1; }
    if (rand(3) === 0) {
      let k = POOL[rand(POOL.length)];
      const near = raw[0x5e] && u8(raw[0x72] - raw[0x62]) < 0x18, upright = raw[0x6f] !== 0;
      if (k === 'LEFT' && u8(0x30 - raw[0x52] + raw[0x62]) < 0x14) k = 'RIGHT';
      if (near && upright) k = 'SPACE'; else if (k === 'SPACE' && (near || rand(4))) k = 'RIGHT'; else if (k === 'B' && near) k = 'W';
      g.key = KEYS[k] ?? (k.charCodeAt(0) | 0x80);
    }
    res = g.step(); each?.(g, n); g.flips.length = 0; g.clicks.length = 0;
  }
  if (end && res === 'over') g.gameOver();
  return { frames: n, res };
}
