import { heightAt, WATER_Y, valleyZ } from '../js/terrain.js';
import fs from 'fs';
const N = 600, X0 = -1500, X1 = 2500, Z0 = -2000, Z1 = 2000;
let min = 1e9, max = -1e9; const H = new Float32Array(N * N);
for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) {
  const x = X0 + (X1 - X0) * i / (N - 1), z = Z0 + (Z1 - Z0) * j / (N - 1);
  const h = heightAt(x, z); H[j * N + i] = h; min = Math.min(min, h); max = Math.max(max, h);
}
const buf = Buffer.alloc(N * N * 3);
for (let k = 0; k < N * N; k++) {
  const h = H[k];
  const i = k % N, j = (k / N) | 0;
  const hx = H[j * N + Math.min(N - 1, i + 1)] - h, hz = H[Math.min(N - 1, j + 1) * N + i] - h;
  const shade = Math.max(0, Math.min(1, 0.6 + (-hx + hz) * 0.03));
  let c = h < WATER_Y ? [40, 80, 160] : [80 + h * 0.2, 130 + h * 0.1, 60 + h * 0.2];
  c = c.map(v => Math.max(0, Math.min(255, v * shade)));
  buf[k * 3] = c[0]; buf[k * 3 + 1] = c[1]; buf[k * 3 + 2] = c[2];
}
fs.writeFileSync(process.argv[2], Buffer.concat([Buffer.from(`P6 ${N} ${N} 255\n`), buf]));
console.log('min', min.toFixed(1), 'max', max.toFixed(1));
for (const x of [0, 200, 400, 640, 900, 1080, 1300]) console.log(x, 'zc', valleyZ(x).toFixed(0), 'h', heightAt(x, valleyZ(x)).toFixed(1));
