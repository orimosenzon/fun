import { startGame } from './lab.mjs';
import { drawPoses } from './pose.mjs';
const m = startGame();
for (const p of process.argv.slice(2).map((v) => parseInt(v, 16))) {
  const { page } = drawPoses(m, [[p]], { x0: 10 });
  console.log('pose', p.toString(16), ' (col 0 = pixel 49, x anchor $22=10 → pixel 70 marked ^)');
  for (let y = 108; y < 164; y++) { const b = ((y & 7) << 10) | (((y >> 3) & 7) << 7) | ((y >> 6) * 40); let s = '';
    for (let x = 49; x < 140; x++) s += (page[b + ((x / 7) | 0)] >> (x % 7)) & 1 ? '#' : '.';
    console.log(String(y).padStart(3), s); }
  console.log('    ' + ' '.repeat(21) + '^');
}
