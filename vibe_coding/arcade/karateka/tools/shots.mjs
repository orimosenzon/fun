// The additions as the player sees them: plays a move in the real game and saves every
// frame the screen showed, side by side.
// usage: node tools/shots.mjs out.png <E|Y|U|F|D|C> [show] [walk N]   (show = force the floor routine)
import { startGame } from './lab.mjs';
import { installMawashi } from '../src/ext/mawashi.js';
import { installFlip } from '../src/ext/flip.js';
import { writePNG } from './emu/apple2.mjs';
import { hgrToIndexed, PALETTE, CPU_HZ } from '../src/vm/machine.js';

const [out, key = 'F', ...opts] = process.argv.slice(2);
const m = startGame(); installMawashi(m); installFlip(m);
m.flip.always = opts.includes('show');
const walk = opts.includes('walk') ? +opts[opts.indexOf('walk') + 1] : 0;
const frames = []; let last = m.video.page2, rec = false;
const run = (s) => { const end = m.cpu.cycles + s * CPU_HZ; while (m.cpu.cycles < end) { m.runTo(m.cpu.cycles + 1500);
  if (m.video.page2 !== last) { last = m.video.page2; if (rec) frames.push(hgrToIndexed(m.hgrPage())); } } };
run(3); m.press('K'); run(0.5); m.press('SPACE'); run(3);
for (let i = 0; i < walk; i++) { m.press('RIGHT'); run(0.5); }
const busy = () => m.flip.frame >= 0 || m.mawashi.active;
const x0 = m.ram[0x62];
rec = true; for (let i = 0; i < 20 && !busy(); i++) { m.press(key); run(0.3); }
for (let i = 0; i < 200 && (m.flip.frame >= 0 || i < ('EYU'.includes(key) ? 22 : 4)); i++) run(0.1);
run(0.4);
// a window that follows the fighter, enlarged
const X0 = Math.max(0, Math.min(280 - 130, 7 * x0 - (key === 'D' ? 80 : 20))), CW = 130, Y0 = opts.includes('show') ? 50 : 84, PH = 170 - Y0, per = 6;
const shown = frames.slice(0, 60), rows = Math.ceil(shown.length / per), rgb = new Uint8Array(CW * per * PH * rows * 3);
shown.forEach((f, k) => { const ox = (k % per) * CW, oy = Math.floor(k / per) * PH;
  for (let y = 0; y < PH; y++) for (let x = 0; x < CW; x++) rgb.set(x === CW - 1 || y === PH - 1 ? [90, 90, 90] : PALETTE[f[(Y0 + y) * 280 + X0 + x]], ((oy + y) * CW * per + ox + x) * 3); });
writePNG(out, CW * per, PH * rows, rgb, 3);
console.log(frames.length, 'frames', JSON.stringify({ flips: m.flip.flips, shows: m.flip.shows, kicks: m.mawashi.kicks, x: m.ram[0x62] }));
