// The measuring instrument: the original program, running on the machine of the sibling project.
// Development only. Nothing under src/ may import this; the new game does not contain or run the original.
import fs from 'node:fs';
import path from 'node:path';
import { GameMachine } from '../../karateka/src/vm/firmware.js';
export const CPU_HZ = 1020484;
export const root = path.resolve(import.meta.dirname, '..');
export const DISK = path.resolve(root, '../karateka/data/karateka.dsk');
export const KEYS = { SPACE: 0x20, LEFT: 0x08, RIGHT: 0x15, UP: 0x0b, DOWN: 0x0a, ESC: 0x1b };
export function boot() {
  const m = new GameMachine(new Uint8Array(fs.readFileSync(DISK)));
  m.sec = (s) => m.runTo(m.cpu.cycles + Math.round(s * CPU_HZ));
  m.press = (k) => m.pressKey(KEYS[k] ?? k.charCodeAt(0));
  return m;
}
// boot, skip the title sequence, and stop when the player stands at the cliff in keyboard mode
export function startGame() { const m = boot(); m.sec(20); m.press('SPACE'); m.sec(20); return m; }
