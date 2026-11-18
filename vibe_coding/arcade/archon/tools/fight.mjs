// node tools/fight.mjs [seconds of combat] [model]: start a two-player game, walk the
// light knight and the dark goblin into each other and photograph the fight.
import fs from 'node:fs';
import { startGame, shot, KEYS, JOY } from './node64.mjs';

const combatSecs = parseFloat(process.argv[2] || '12');
const m = startGame({ model: process.argv[3] || 'ntsc' });
const fps = m.model.hz / (m.model.lines * m.model.cpl);
const dir = new URL('../reference/shots/', import.meta.url).pathname;
fs.mkdirSync(dir, { recursive: true });
const run = (secs) => { for (let f = 0; f < secs * fps; f++) m.runFrame(); };
const tap = (key) => { m.setKey(KEYS[key], true); run(0.15); m.setKey(KEYS[key], false); run(0.3); };
const spriteX = (s) => m.vic.reg[s * 2] | ((m.vic.reg[0x10] >> s) & 1 ? 256 : 0);
// Hold a direction on a port until a sprite reaches x. The cursor frame is
// sprite 1; a piece being carried is sprite 0, 4 pixels to the right.
function moveTo(port, x, sprite = 1) {
  if (sprite === 0) x += 4;
  for (let guard = 0; guard < 600 && spriteX(sprite) !== x; guard++) { m.joy[port] = spriteX(sprite) < x ? JOY.RIGHT : JOY.LEFT; m.runFrame(); }
  m.joy[port] = 0; run(0.3);
  if (spriteX(sprite) !== x) throw new Error(`sprite ${sprite} stuck at ${spriteX(sprite)}, wanted ${x}`);
}
const fire = (port) => { m.joy[port] = JOY.FIRE; run(0.2); m.joy[port] = 0; run(0.5); };

run(12); tap('STOP'); run(2); tap('F7'); run(3);
moveTo(0, 96); fire(0); moveTo(0, 168, 0); fire(0); run(2);       // knight: column 1 to column 4
moveTo(1, 240); fire(1); moveTo(1, 168, 0); fire(1);              // goblin: column 7 onto the knight
shot(m, dir + 'fight_00.png');
let ss = 0, sb = 0, sidWrites = 0;
const vicRead = m.vic.read.bind(m.vic);
m.vic.read = (r) => { const v = vicRead(r); if ((r & 0x3f) === 0x1e && v) ss++; if ((r & 0x3f) === 0x1f && v) sb++; return v; };
const sidWrite = m.sid.write.bind(m.sid);
m.sid.write = (r, v) => { sidWrites++; sidWrite(r, v); };
let peak = 0;
for (let i = 1; i <= combatSecs * 2; i++) {
  // Light attacks to the right, dark walks left into it.
  m.joy[0] = (i % 4 < 2) ? JOY.FIRE | JOY.RIGHT : JOY.RIGHT;
  m.joy[1] = (i % 6 < 3) ? JOY.LEFT : JOY.FIRE | JOY.LEFT;
  run(0.5);
  for (const s of m.sid.drain()) peak = Math.max(peak, Math.abs(s));
  shot(m, `${dir}fight_${String(i).padStart(2, '0')}.png`);
}
console.log('collision reads with hits: sprite-sprite', ss, 'sprite-background', sb, '| SID writes', sidWrites, 'peak', peak.toFixed(3));
