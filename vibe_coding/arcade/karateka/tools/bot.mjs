// Plays the original from start to finish with an immortal player and one-hit enemies,
// to exercise every stage (disk loads, firmware calls) on the ROM-less machine.
// Usage: node bot.mjs minutes shotEverySeconds
import { startGame } from './lab.mjs';
const minutes = +(process.argv[2] || 10), every = +(process.argv[3] || 20);
const m = startGame();
m.onWrite = (a, v) => {
  if (a === 0xb6 && v < m.ram[0xb0]) m.ram[0xb6] = m.ram[0xb0];
  if (a === 0xb7 && v > 1) m.ram[0xb7] = 1;
};
m.sec(3); m.press('K'); m.sec(0.5);
const seq = ['RIGHT', 'RIGHT', 'W', 'RIGHT', 'S', 'RIGHT', 'X']; let notStance = 0;
let i = 0, nextShot = 0; const t0 = m.t(); let lastTrack = -1, lastAlive = -1, kills = 0;
try {
  while (m.t() - t0 < minutes * 60) {
    const z = m.ram, enemy = z[0x5e] === 1 && z[0xb7] > 0, stance = z[0x61] === 6;
    if (z[0x5e] !== lastAlive) { if (lastAlive === 1 && z[0x5e] === 0) console.log(`t=${(m.t() - t0).toFixed(0)}s enemy #${++kills} down`); lastAlive = z[0x5e]; }
    notStance = stance ? 0 : notStance + 1;
    if (notStance >= 4) { m.press('SPACE'); m.sec(1.5); notStance = 0; }
    else { m.press(seq[i++ % seq.length]); m.sec(0.4); }
    if (m.drive.motor && m.drive.halfTrack !== lastTrack) { lastTrack = m.drive.halfTrack; console.log(`t=${(m.t() - t0).toFixed(0)}s disk activity, track ${lastTrack / 2}`); }
    if (m.t() - t0 >= nextShot) { m.shot('bot_' + String(Math.round(nextShot)).padStart(4, '0')); nextShot += every; }
  }
} catch (e) { console.log('STOPPED:', e.message, 'pc', m.cpu.pc.toString(16), 't', (m.t() - t0).toFixed(0)); }
console.log('done; illegal opcodes:', m.cpu.illegal, 'health', m.ram[0xb6], m.ram[0xb7]);
