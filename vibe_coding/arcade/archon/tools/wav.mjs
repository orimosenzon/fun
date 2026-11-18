// node tools/wav.mjs <seconds> <out.wav> [model]: record the game's sound from power-on.
import fs from 'node:fs';
import { startGame } from './node64.mjs';
const [secs = '30', out = 'reference/title.wav', model = 'ntsc'] = process.argv.slice(2);
const m = startGame({ model });
const rate = m.sid.sampleRate, chunks = [];
const frames = parseFloat(secs) * m.model.hz / (m.model.lines * m.model.cpl);
for (let f = 0; f < frames; f++) { m.runFrame(); chunks.push(m.sid.drain()); }
const n = chunks.reduce((a, c) => a + c.length, 0);
const buf = Buffer.alloc(44 + n * 2);
buf.write('RIFF', 0); buf.writeUInt32LE(36 + n * 2, 4); buf.write('WAVEfmt ', 8); buf.writeUInt32LE(16, 16); buf.writeUInt16LE(1, 20); buf.writeUInt16LE(1, 22);
buf.writeUInt32LE(rate, 24); buf.writeUInt32LE(rate * 2, 28); buf.writeUInt16LE(2, 32); buf.writeUInt16LE(16, 34); buf.write('data', 36); buf.writeUInt32LE(n * 2, 40);
let o = 44, peak = 0, loud = 0;
for (const c of chunks) for (const s of c) { buf.writeInt16LE(Math.round(s * 32767), o); o += 2; peak = Math.max(peak, Math.abs(s)); if (Math.abs(s) > 0.02) loud++; }
fs.writeFileSync(out, buf);
console.log(`${out}: ${(n / rate).toFixed(2)} s for ${secs} s emulated, peak ${peak.toFixed(3)}, ${(100 * loud / n).toFixed(0)}% audible`);
