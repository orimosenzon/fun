// Fits the cost of drawing (src/data/timing.js) to the original: least squares over the passes
// recorded by `node tools/compare.mjs --timing --dump file.json`.
// usage: node tools/fit_timing.mjs re/a.json [re/b.json ...] [--write]
import fs from 'node:fs';
const files = process.argv.slice(2).filter((a) => !a.startsWith('--'));
const rows = files.flatMap((f) => JSON.parse(fs.readFileSync(f, 'utf8')));
const N = 11;   // ten kinds of drawing work, and the frames shown
const X = rows.map((r) => [...r[2], r[3]]), y = rows.map((r) => r[0] - r[1]);
const A = Array.from({ length: N }, () => new Float64Array(N + 1));
X.forEach((x, k) => { for (let i = 0; i < N; i++) { for (let j = 0; j < N; j++) A[i][j] += x[i] * x[j]; A[i][N] += x[i] * y[k]; } });
for (let i = 0; i < N; i++) A[i][i] += 1e-6;
for (let i = 0; i < N; i++) {
  let p = i; for (let r = i + 1; r < N; r++) if (Math.abs(A[r][i]) > Math.abs(A[p][i])) p = r;
  [A[i], A[p]] = [A[p], A[i]];
  for (let r = 0; r < N; r++) if (r !== i) { const f = A[r][i] / A[i][i]; for (let j = i; j <= N; j++) A[r][j] -= f * A[i][j]; }
}
const w = A.map((r, i) => r[N] / r[i]);
const err = X.map((x, k) => Math.abs(x.reduce((s, v, i) => s + v * w[i], 0) - y[k]) / (rows[k][0]));
err.sort((a, b) => a - b);
console.log(rows.length, 'passes. cost:', w.map((v) => v.toFixed(1)).join(', '));
console.log('error per pass: median', (err[err.length >> 1] * 100).toFixed(2) + '%', '90th', (err[Math.floor(err.length * 0.9)] * 100).toFixed(2) + '%', '99th', (err[Math.floor(err.length * 0.99)] * 100).toFixed(2) + '%', 'worst', (err.at(-1) * 100).toFixed(1) + '%');
if (process.argv.includes('--write')) {
  const src = fs.readFileSync(new URL('../src/data/timing.js', import.meta.url), 'utf8').replace(/export const COST = \[.*\];/, `export const COST = [${w.map((v) => +v.toFixed(2)).join(', ')}];`);
  fs.writeFileSync(new URL('../src/data/timing.js', import.meta.url), src); console.log('written');
}
