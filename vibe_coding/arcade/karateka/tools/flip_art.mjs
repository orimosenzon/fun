// Preview of the somersault frames. usage: node tools/flip_art.mjs out.png
import { blitArt } from '../src/ext/mawashi.js';
import { flipArt, FLIP_FRAMES, FLIP_GRID } from '../src/ext/flip.js';
import { writePNG } from './emu/apple2.mjs';
import { hgrToIndexed, PALETTE } from '../src/vm/machine.js';
const mem = new Uint8Array(0x10000).fill(0x80), y0 = 161 - FLIP_GRID.GROUND;
FLIP_FRAMES.forEach((f, i) => blitArt(mem, 0x2000, 20 + i * 38, flipArt(f.turn, f.tuck, f.up, 1), y0, FLIP_GRID.CX));
const idx = hgrToIndexed(mem.subarray(0x2000, 0x4000)), top = 100, h = 66, rgb = new Uint8Array(280 * h * 3);
for (let i = 0; i < 280 * h; i++) rgb.set(PALETTE[idx[top * 280 + i]], i * 3);
writePNG(process.argv[2], 280, h, rgb, 7);
