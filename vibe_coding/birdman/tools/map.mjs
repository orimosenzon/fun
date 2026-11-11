// Debug: dump a shaded top-down map of the island as PPM (stdout path arg), with POIs marked.
//   node tools/map.mjs /tmp/x.ppm && python3 -c "from PIL import Image; Image.open('/tmp/x.ppm').save('/tmp/x.png')"
import fs from 'node:fs';
import { initWorld, H, N, CELL, HALF, POI, thermals, windAt } from '../js/sim/world.js';
initWorld();
const S = 2, W = N * S, buf = Buffer.alloc(W * W * 3);
let mx = -1e9;
for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) {
  const h = H[j * N + i]; mx = Math.max(mx, h);
  const hx = H[j * N + Math.min(N - 1, i + 1)] - H[j * N + Math.max(0, i - 1)], hz = H[Math.min(N - 1, j + 1) * N + i] - H[Math.max(0, j - 1) * N + i];
  const sh = Math.max(0.25, Math.min(1.3, 0.8 + (-hx + hz * 0.4) / 40));
  let c;
  if (h < 0) c = [30, 80 + h, 150 + h * 1.5]; else if (h < 4) c = [220, 205, 150];
  else { const t = Math.min(1, h / 700); c = [70 + 130 * t, 140 - 20 * t, 60 + 90 * t]; }
  const slope = Math.hypot(hx, hz) / (2 * CELL); if (h > 4 && slope > 0.7) c = [130, 120, 110];
  if (process.argv[3] === 'lift') { const w = windAt(i * CELL - HALF, Math.max(h, 0) + 60, j * CELL - HALF)[1]; if (w > 0.7) c = [255, 255 - w * 30, 60]; if (w < -0.7) c = [80, 60, 200]; }
  if (h > 0 && Math.floor(h / 50) !== Math.floor(H[j * N + Math.max(0, i - 1)] / 50)) c = c.map((v) => v * 0.7);
  for (let a = 0; a < S; a++) for (let b = 0; b < S; b++) { const o = ((j * S + a) * W + i * S + b) * 3; buf[o] = c[0] * sh; buf[o + 1] = c[1] * sh; buf[o + 2] = c[2] * sh; }
}
const dot = (x, z, col, r = 5) => { const ci = Math.round(((x + HALF) / CELL) * S), cj = Math.round(((z + HALF) / CELL) * S); for (let a = -r; a <= r; a++) for (let b = -r; b <= r; b++) { if (Math.abs(a) + Math.abs(b) > r) continue; const o = ((cj + a) * W + ci + b) * 3; buf[o] = col[0]; buf[o + 1] = col[1]; buf[o + 2] = col[2]; } };
for (const k in POI) { dot(POI[k][0], POI[k][2], [255, 0, 0]); console.log(k, POI[k].map((v) => Math.round(v)).join(',')); }
for (const t of thermals) dot(t.x, t.z, [255, 255, 255], 3);
console.log('max h', mx);
fs.writeFileSync(process.argv[2], Buffer.concat([Buffer.from(`P6 ${W} ${W} 255\n`), buf]));
