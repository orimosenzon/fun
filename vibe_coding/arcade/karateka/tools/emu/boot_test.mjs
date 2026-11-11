// Boot the original and dump a screenshot every N seconds, to see that the emulator works.
import fs from 'node:fs';
import path from 'node:path';
import { Apple2, savePagePNG } from './apple2.mjs';

const root = path.resolve(import.meta.dirname, '../../reference');
const disk = process.argv[2] || 'Karateka_1984_Broderbund_a3.dsk';
const total = +(process.argv[3] || 40), every = +(process.argv[4] || 4);
const outDir = process.argv[5] || path.join(root, 'shots');
fs.mkdirSync(outDir, { recursive: true });
const m = new Apple2({ romDir: path.join(root, 'roms'), disk: path.join(root, 'disks', disk) });
m.cpu.pc = 0xc600;
m.videoLog = [];
for (let t = every; t <= total; t += every) {
  m.runSeconds(every);
  const v = m.video;
  console.log(`t=${t}s pc=${m.cpu.pc.toString(16)} track=${m.drive.halfTrack / 2} motor=${m.drive.motor} video=${JSON.stringify(v)} illegal=${m.cpu.illegal} spk=${m.speaker.length} flips=${m.videoLog.length}`);
  savePagePNG(path.join(outDir, `boot_${String(t).padStart(3, '0')}.png`), m.hgrPage());
}
