// Shared helpers for poking at the running game from Node.
import fs from 'node:fs';
import { startGame, shot, KEYS, JOY } from './node64.mjs';
export { shot, KEYS, JOY };

export function boardGame(opts = {}) {
  const m = startGame(opts);
  const fps = m.model.hz / (m.model.lines * m.model.cpl);
  const run = (secs) => { for (let f = 0; f < secs * fps; f++) m.runFrame(); };
  const tap = (key) => { m.setKey(KEYS[key], true); run(0.15); m.setKey(KEYS[key], false); run(0.3); };
  const spriteX = (s) => m.vic.reg[s * 2] | ((m.vic.reg[0x10] >> s) & 1 ? 256 : 0);
  const spriteY = (s) => m.vic.reg[s * 2 + 1];
  const fire = (port) => { m.joy[port] = JOY.FIRE; run(0.2); m.joy[port] = 0; run(0.5); };
  function moveTo(port, x, y, sprite = 1) {
    if (sprite === 0) x += 4;
    for (let guard = 0; guard < 900; guard++) {
      let j = 0;
      if (spriteX(sprite) < x) j |= JOY.RIGHT; else if (spriteX(sprite) > x) j |= JOY.LEFT;
      if (y != null) { if (spriteY(sprite) < y) j |= JOY.DOWN; else if (spriteY(sprite) > y) j |= JOY.UP; }
      if (!j) break;
      m.joy[port] = j; m.runFrame();
    }
    m.joy[port] = 0; run(0.3);
  }
  run(12); tap('STOP'); run(2);
  for (const key of opts.setup || []) tap(key);     // e.g. F3 to hand a side to the computer
  tap('F7'); run(3);
  return { m, fps, run, tap, spriteX, spriteY, fire, moveTo };
}
export const hex = (v, n = 2) => v.toString(16).padStart(n, '0');
export function dump(m, a, n) {
  let s = '';
  for (let i = 0; i < n; i += 16) { s += hex(a + i, 4) + ':'; for (let j = 0; j < 16 && i + j < n; j++) s += ' ' + hex(m.ram[a + i + j]); s += '\n'; }
  return s;
}
