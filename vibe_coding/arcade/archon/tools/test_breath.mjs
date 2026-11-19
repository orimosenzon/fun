// node tools/test_breath.mjs [shots]: the dragon's fire breath (src/ext/breath.js).
// Puts the dark dragon next to a light knight, lets the knight attack it, and checks
// that the breath burns, kills, hands the square to the dragon and respects its cooldown.
// With "shots" it also writes reference/shots/breath_*.png.
import fs from 'node:fs';
import { boardGame, shot, JOY, KEYS } from './lab.mjs';
import { installBreath, BREATH } from '../src/ext/breath.js';

const shots = process.argv.includes('shots');
const dir = new URL('../reference/shots/', import.meta.url).pathname;
const BOARD = 0xbd7c, LIFE = 0xbd05, X = 0xbd3e;
let failed = 0;
const check = (ok, what) => { console.log((ok ? 'ok   ' : 'FAIL ') + what); if (!ok) failed++; };

function duel(computerDark) {
  const g = boardGame(computerDark ? { setup: ['F3', 'F3'] } : {});
  const m = g.m, ext = installBreath(m);
  if (!ext) throw new Error('the game code does not match the extension');
  m.ram[BOARD + 5 * 9 + 8] = 0x80; m.ram[BOARD + 4 * 9 + 2] = 0x1d;     // dragon to row 4, column 2
  if (computerDark) { m.ram[BOARD + 4 * 9 + 1] = 0x02; m.ram[BOARD + 9] = 0x09; }   // a golem attacks instead: it lasts longer
  g.moveTo(0, 96, 136); g.fire(0); g.moveTo(0, 120, null, 0); g.fire(0);   // knight takes it on
  if (!computerDark) g.run(4);
  return { ...g, ext };
}

{
  const { m, run, ext } = duel(false);
  check(ext.state.dragon === 1, 'the dragon is recognised as the dark fighter');
  run(2);
  check(ext.ready, 'breath is ready shortly after the fight begins');
  // Knight walks towards the dragon, then the dragon breathes left at it.
  m.joy[0] = JOY.RIGHT; run(1.2); m.joy[0] = 0;
  const life0 = m.ram[LIFE], dragonX = m.ram[X + 1];
  m.joy[1] = JOY.LEFT; ext.press(1); run(0.1);
  check(ext.state.active > 0, 'second button starts the breath');
  m.joy[1] = JOY.RIGHT | JOY.FIRE;                 // try to walk and shoot while breathing
  run(0.4); if (shots) shot(m, dir + 'breath_1.png');
  check(m.ram[X + 1] === dragonX, 'the dragon stands still while breathing');
  check(m.ram[0xbd04] === 0, 'no fireball while breathing');
  check(m.ram[LIFE] < life0, `the knight burns (${life0} -> ${m.ram[LIFE]})`);
  m.joy[1] = 0; run(1.2);
  check(m.ram[LIFE] === 0, 'the knight dies in the fire');
  ext.press(1); run(0.1);
  check(!ext.state.active, 'no breath after the fight is decided');
  run(8); if (shots) shot(m, dir + 'breath_2.png');
  check(m.ram[BOARD + 4 * 9 + 2] === 0x1d && !m.ram.subarray(BOARD, BOARD + 81).includes(0x09), 'back on the board the dragon holds the square and the knight is gone');
}
{
  const { m, run, ext } = duel(false);
  run(2);
  ext.press(1); run(0.1);
  const first = ext.state.active > 0;
  run(BREATH.frames / 60 + 0.3);
  ext.press(1); run(0.1);
  check(first && !ext.state.active, 'a second breath has to wait for the cooldown');
  run(BREATH.cooldown / 60 + 0.2);
  ext.press(1); run(0.1);
  check(ext.state.active > 0, 'and is allowed after it');
}
{
  const { m, run, ext } = duel(true);
  let breathed = false;
  for (let i = 0; i < 150 && !breathed; i++) { m.joy[0] = i > 25 && i < 50 ? JOY.RIGHT : 0; run(0.1); breathed = ext.state.active > 0; }   // the knight comes closer
  m.joy[0] = 0;
  if (shots) { run(0.4); shot(m, dir + 'breath_3.png'); }
  check(breathed, 'the computer, playing the dragon, breathes on its own');
}
console.log(failed ? `${failed} failed` : 'all passed');
process.exit(failed ? 1 : 0);
