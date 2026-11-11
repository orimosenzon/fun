// Node wrapper around machine.mjs: loads ROMs/disk from files, writes PNGs.
import fs from 'node:fs';
import zlib from 'node:zlib';
import { Machine, hgrToIndexed, PALETTE } from './machine.mjs';
export { CPU_HZ, hgrToIndexed, hgrRowBase, PALETTE } from './machine.mjs';

function parseTsBytes(file, name) {
  let src = fs.readFileSync(file, 'utf8');
  if (name) { const i = src.indexOf(name); src = src.slice(i); src = src.slice(0, src.indexOf(']')); }
  return Uint8Array.from([...src.matchAll(/0x([0-9a-fA-F]{2})\b/g)].map((m) => parseInt(m[1], 16)));
}


export function loadRoms(romDir) {
  return { rom: parseTsBytes(`${romDir}/fpbasic.ts`, 'new Uint8Array'), diskRom: parseTsBytes(`${romDir}/disk2.ts`, 'BOOTSTRAP_ROM_16') };
}

export class Apple2 extends Machine {
  constructor({ romDir, disk }) { super({ ...loadRoms(romDir), dsk: disk ? new Uint8Array(fs.readFileSync(disk)) : null }); }
}

export function writePNG(file, width, height, rgb, scale = 1) {
  const w = width * scale, h = height * scale;
  const raw = Buffer.alloc((w * 3 + 1) * h);
  for (let y = 0; y < h; y++) {
    const sy = Math.floor(y / scale);
    let o = y * (w * 3 + 1); raw[o++] = 0;
    for (let x = 0; x < w; x++) { const s = (sy * width + Math.floor(x / scale)) * 3; raw[o++] = rgb[s]; raw[o++] = rgb[s + 1]; raw[o++] = rgb[s + 2]; }
  }
  const crcT = new Int32Array(256).map((_, n) => { let c = n; for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1; return c; });
  const crc = (buf) => { let c = -1; for (const b of buf) c = crcT[(c ^ b) & 0xff] ^ (c >>> 8); return (c ^ -1) >>> 0; };
  const chunk = (type, data) => { const len = Buffer.alloc(4); len.writeUInt32BE(data.length); const td = Buffer.concat([Buffer.from(type), data]); const cr = Buffer.alloc(4); cr.writeUInt32BE(crc(td)); return Buffer.concat([len, td, cr]); };
  const ihdr = Buffer.alloc(13); ihdr.writeUInt32BE(w, 0); ihdr.writeUInt32BE(h, 4); ihdr[8] = 8; ihdr[9] = 2;
  fs.writeFileSync(file, Buffer.concat([Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]), chunk('IHDR', ihdr), chunk('IDAT', zlib.deflateSync(raw)), chunk('IEND', Buffer.alloc(0))]));
}

export function savePagePNG(file, page, scale = 2) {
  const idx = hgrToIndexed(page);
  const rgb = new Uint8Array(280 * 192 * 3);
  for (let i = 0; i < idx.length; i++) { const p = PALETTE[idx[i]]; rgb[i * 3] = p[0]; rgb[i * 3 + 1] = p[1]; rgb[i * 3 + 2] = p[2]; }
  writePNG(file, 280, 192, rgb, scale);
}
