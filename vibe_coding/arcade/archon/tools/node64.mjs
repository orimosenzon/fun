// Node wrapper: a C64 with the ROMs and the game disk from data/, and PNG output.
import fs from 'node:fs';
import zlib from 'node:zlib';
import { C64, WIDTH, PALETTE } from '../src/vm/c64.js';
import { D64 } from '../src/vm/d64.js';
export { KEYS, JOY } from '../src/vm/c64.js';

const here = new URL('..', import.meta.url).pathname;
const bin = (f) => new Uint8Array(fs.readFileSync(here + f));

export function makeMachine({ model = 'ntsc', disk = 'data/archon.d64' } = {}) {
  const m = new C64({ kernal: bin('data/kernal.bin'), basic: bin('data/basic.bin'), chargen: bin('data/chargen.bin'), model });
  if (disk) m.disk = new D64(bin(disk));
  return m;
}

// Boot and start the first program on the disk.
export function startGame(opts) {
  const m = makeMachine(opts);
  if (!m.bootToReady()) throw new Error('BASIC did not come up');
  m.runProgram(m.disk.load(Uint8Array.of(0x2a)));
  return m;
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

export function shot(m, file, scale = 2) {
  const fb = m.vic.fb, rows = m.model.rows;
  const rgb = new Uint8Array(WIDTH * rows * 3);
  for (let i = 0; i < fb.length; i++) { const p = PALETTE[fb[i]]; rgb[i * 3] = p[0]; rgb[i * 3 + 1] = p[1]; rgb[i * 3 + 2] = p[2]; }
  writePNG(file, WIDTH, rows, rgb, scale);
}
