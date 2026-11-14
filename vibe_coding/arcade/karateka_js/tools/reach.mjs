// Play the original with the test bot until it loads the next scene, then save what is in memory there.
// usage: node tools/reach.mjs <loads to pass> name   → re/<name>.ram and the entry address
import fs from 'node:fs';
import { boot, CPU_HZ } from './oracle.mjs';
const [passes = 1, name = 'level2'] = [+process.argv[2] || 1, process.argv[3] || 'level2'];
const m = boot(), c = m.cpu, ram = m.ram;
m.sec(20); m.press('SPACE');
const gameCode = () => ram[0xb700] === 0x20 && ram[0xb701] === 0x00 && ram[0xb702] === 0x70;
while (!(m.inGame && c.pc === 0xb700 && gameCode())) m.step();
ram[0xc4] = 1;
let rnd = 2 * 2654435761 >>> 0; const rand = (n) => { rnd = (Math.imul(rnd, 1664525) + 1013904223) >>> 0; return (rnd >>> 8) % n; };
const POOL = ['RIGHT', 'RIGHT', 'RIGHT', 'RIGHT', 'W', 'S', 'X', 'Q', 'A', 'Z', 'W', 'S', 'RIGHT'];
const KEYS = { RIGHT: 0x15, SPACE: 0x20 };
let loads = 0, wasMotor = false, frames = 0, entry = -1, lastFlip = m.video.page2;
const log = [];
// the scene number is $D2; when it reaches `passes` + 1 the scene has been loaded. Then play it for a while,
// noting which addresses are executed, so that the listing knows what is code.
const exec = new Uint32Array(0x10000); let since = -1, snap = null;
const want = passes + 1, playFrames = +(process.argv[4] || 3000);
for (let guard = 0; guard < 8e9; guard++) {
  if (since < 0 && ram[0xd2] === want && !m.drive.motor && c.pc < 0xc000 && c.pc >= 0x6000) { since = frames; snap = ram.slice(0, 0x10000); console.log(`scene ${want} is running after ${frames} frames, pc $${c.pc.toString(16)}`); }
  if (since >= 0) { if (c.pc < 0xc000) exec[c.pc]++; if (frames - since > playFrames || (m.drive.motor && frames - since > 50)) break; }
  if (m.video.page2 !== lastFlip) { lastFlip = m.video.page2; frames++;
    if (ram[0x5d]) ram[0xb6] = ram[0xb0]; if (ram[0xb7] > 1) ram[0xb7] = 1;
    if (!m.drive.motor && rand(3) === 0) { let k = POOL[rand(POOL.length)]; const near = ram[0x5e] && ((ram[0x72] - ram[0x62]) & 255) < 0x18; if (near && ram[0x6f]) k = 'SPACE'; m.pressKey(KEYS[k] ?? k.charCodeAt(0)); } }
  m.step();
}
fs.writeFileSync(new URL(`../re/${name}.cov`, import.meta.url), Buffer.from(exec.buffer));
fs.writeFileSync(new URL(`../re/${name}.start.ram`, import.meta.url), snap);
console.log(log.join('\n'));
fs.writeFileSync(new URL(`../re/${name}.ram`, import.meta.url), ram);
console.log('saved re/' + name + '.ram at pc $' + c.pc.toString(16), 'frames', frames, 'zp D0', ram[0xd0], 'D2', ram[0xd2]);
