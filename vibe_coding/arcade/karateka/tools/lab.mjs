// Small lab helpers for poking at the original while it runs on the ROM-less machine.
import fs from 'node:fs';
import path from 'node:path';
import { GameMachine } from '../src/vm/firmware.js';
import { CPU_HZ } from '../src/vm/machine.js';
import { savePagePNG } from './emu/apple2.mjs';
export const root = path.resolve(import.meta.dirname, '..');
export const KEYS = { SPACE: 0x20, LEFT: 0x08, RIGHT: 0x15, UP: 0x0b, DOWN: 0x0a };
export function newGame() {
  const m = new GameMachine(new Uint8Array(fs.readFileSync(path.join(root, 'data/karateka.dsk'))));
  m.sec = (s) => m.runTo(m.cpu.cycles + Math.round(s * CPU_HZ));
  m.press = (k) => m.pressKey(KEYS[k] ?? k.charCodeAt(0));
  m.shot = (name) => { fs.mkdirSync(path.join(root, 'reference/shots'), { recursive: true }); savePagePNG(path.join(root, 'reference/shots', name + '.png'), m.hgrPage(), 1); };
  m.t = () => m.cpu.cycles / CPU_HZ;
  return m;
}
// boot, skip the intro and stop when the player stands at the cliff
export function startGame() { const m = newGame(); m.sec(20); m.press('SPACE'); m.sec(20); return m; }
