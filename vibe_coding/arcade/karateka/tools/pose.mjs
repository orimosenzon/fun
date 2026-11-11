// Draw fighter poses with the original draw routine ($6811) on an empty page and save a PNG.
// usage: node tools/pose.mjs out.png "pose[:head],pose[:head],..." [scale]
import fs from 'node:fs';
import { startGame } from './lab.mjs';
import { writePNG } from './emu/apple2.mjs';
import { hgrToIndexed, PALETTE } from '../src/vm/machine.js';

export function call(m, addr) {
  const c = m.cpu; c.push(0xff); c.push(0xfe); c.pc = addr;
  let n = 0; while (c.pc !== 0xffff) { m.step(); if (++n > 2e6) throw new Error('runaway'); }
}
export function drawPoses(m, list, { mono = false, scale = 4, x0 = 4, dx = 9 } = {}) {
  const r = m.ram;
  r[7] = 0x20; r.fill(0x80, 0x2000, 0x4000);
  list.forEach(([body, head], i) => {
    r[0x20] = body; r[0x21] = head ?? body; r[0x22] = x0 + i * dx; r[0x2c] = 1; r[0x50] = 0; r[0x36] = 0; r[0x4d] = 0;
    call(m, 0x6811);
  });
  const page = r.slice(0x2000, 0x4000);
  const rgb = new Uint8Array(280 * 192 * 3);
  if (mono) {
    for (let y = 0; y < 192; y++) { const b = ((y & 7) << 10) | (((y >> 3) & 7) << 7) | ((y >> 6) * 40);
      for (let x = 0; x < 280; x++) { const on = (page[b + ((x / 7) | 0)] >> (x % 7)) & 1; const o = (y * 280 + x) * 3; const v = on ? 255 : ((x / 7 | 0) & 1 ? 40 : 0); rgb[o] = rgb[o + 1] = v; rgb[o + 2] = on ? 255 : (y % 10 === 0 ? 90 : v); } }
  } else { const idx = hgrToIndexed(page); for (let i = 0; i < idx.length; i++) { const p = PALETTE[idx[i]]; rgb.set(p, i * 3); } }
  return { rgb, page };
}
if (import.meta.url === `file://${process.argv[1]}`) {
  const [out, spec, scale = '4', mono] = process.argv.slice(2);
  const m = startGame();
  const list = spec.split(',').map((s) => s.split(':').map((v) => parseInt(v, 16)));
  const { rgb } = drawPoses(m, list, { mono: !!mono });
  // crop rows 100..192
  const y0 = 100, h = 92, crop = new Uint8Array(280 * h * 3); crop.set(rgb.subarray(y0 * 280 * 3, (y0 + h) * 280 * 3));
  writePNG(out, 280, h, crop, +scale);
}
