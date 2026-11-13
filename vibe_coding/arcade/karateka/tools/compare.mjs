// Our frames next to the game's own fighter, on one screen, to judge the drawing.
// usage: node tools/compare.mjs out.png <sequence> [first frame] [count]
import { startGame } from './lab.mjs';
import { drawPoses } from './pose.mjs';
import { blitArt } from '../src/ext/mawashi.js';
import { SEQ, packed } from './flip_frames.mjs';
import { writePNG } from './emu/apple2.mjs';
import { hgrToIndexed, PALETTE } from '../src/vm/machine.js';
const [out, name = 'mawashi', first = '0', count = '4'] = process.argv.slice(2);
const m = startGame(), r = m.ram;
drawPoses(m, [[6], [3]], { x0: 1, dx: 8 });                 // the game's stance and its middle kick
SEQ[name].slice(+first, +first + +count).forEach((f, i) => { const a = packed(f); blitArt(r, 0x2000, 7 * (15 + i * 6) + 16, a.rows, 161 + a.top, a.ax); });
const idx = hgrToIndexed(r.subarray(0x2000, 0x4000)), y0 = 70, h = 96, rgb = new Uint8Array(280 * h * 3);
for (let i = 0; i < 280 * h; i++) rgb.set(PALETTE[idx[y0 * 280 + i]], i * 3);
writePNG(out, 280, h, rgb, 7);
