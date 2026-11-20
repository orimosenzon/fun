// The rivers make sense: each begins at the edge of the map, at a lake or at a spring, and ends at a
// lake, in a bigger river or at the edge; the water never runs uphill, and no ground stands in it.
//   node tests/rivers.mjs
import fs from 'node:fs';
const meta = JSON.parse(fs.readFileSync(new URL('../assets/meta.json', import.meta.url)));
const buf = fs.readFileSync(new URL('../assets/terrain.bin', import.meta.url));
const H = new Uint16Array(buf.buffer, buf.byteOffset, buf.length / 2), N = meta.n, C = meta.cell, HALF = meta.half;
const g = (x, z) => { const fx = Math.min(N - 1.001, Math.max(0, (x + HALF) / C)), fz = Math.min(N - 1.001, Math.max(0, (z + HALF) / C)), i = fx | 0, j = fz | 0, u = fx - i, v = fz - j, k = j * N + i;
  return ((H[k] * (1 - u) + H[k + 1] * u) * (1 - v) + (H[k + N] * (1 - u) + H[k + N + 1] * u) * v) * meta.hscale; };
let fails = 0;
const ok = (c, msg) => { if (!c) { fails++; console.log('FAIL', msg); } };
const lakeAt = (p) => meta.lakes.find((L) => p[0] > L.x0 && p[0] < L.x1 && p[2] > L.z0 && p[2] < L.z1 && Math.abs(p[1] - L.level) < 0.2);
const atEdge = (p) => Math.max(Math.abs(p[0]), Math.abs(p[2])) > HALF - 70;
ok(meta.rivers.length >= 6, 'six rivers');
for (const r of meta.rivers) {
  const P = r.pts, a = P[0], b = P[P.length - 1];
  let buried = 0, up = 0;
  P.forEach((p, i) => {
    const q = P[Math.min(P.length - 1, i + 1)], o = P[Math.max(0, i - 1)], l = Math.hypot(q[0] - o[0], q[2] - o[2]) || 1, tx = (q[0] - o[0]) / l, tz = (q[2] - o[2]) / l;
    for (const d of [0, -(r.width / 2 + 1), r.width / 2 + 1]) if (g(p[0] - tz * d, p[2] + tx * d) > p[1] + 0.05) buried++;
    if (i && p[1] > P[i - 1][1] + 0.011) up++;
  });
  ok(!buried, `${r.name}: ground stands in the water at ${buried} places`);
  ok(!up, `${r.name}: runs uphill at ${up} places`);
  if (r.from === 'edge') ok(atEdge(a), `${r.name}: should begin at the edge of the map`);
  if (r.from === 'lake') ok(lakeAt(a) && g(a[0], a[2]) < a[1], `${r.name}: should begin in a lake`);
  if (r.from === 'spring') ok(meta.springs.some((s) => s.name === r.name), `${r.name}: a spring is marked where it begins`);
  if (r.to === 'edge') ok(atEdge(b), `${r.name}: should end at the edge of the map`);
  if (r.to === 'lake') ok(lakeAt(b) && g(b[0], b[2]) < b[1] - 0.2, `${r.name}: should end in a lake`);
  if (r.to === 'river') ok(meta.rivers.some((o) => o !== r && o.width > r.width && o.pts.some((p) => Math.hypot(p[0] - b[0], p[2] - b[2]) < o.width / 2 && Math.abs(p[1] - b[1]) < 0.3)), `${r.name}: should end in a bigger river, at its level`);
  console.log(`${r.name.padEnd(7)} ${r.from} > ${r.to}  ${(P.length * 6 / 1000).toFixed(1)} km, ${a[1]} m > ${b[1]} m`);
}
// beyond the map the main river goes on from where the map's river stops
ok(meta.far.length === 2, 'the river goes on beyond the map, both ways');
for (const [f, p] of [[meta.far[0].pts.at(-1), meta.rivers[0].pts[0]], [meta.far[1].pts[0], meta.rivers.find((r) => r.to === 'edge').pts.at(-1)]]) ok(Math.hypot(f[0] - p[0], f[2] - p[2]) < 30 && Math.abs(f[1] - p[1]) < 1, 'the far river meets the river of the map');
// every lake that a river leaves has an outlet line, and the falls are real drops
for (const r of meta.rivers) if (r.from === 'lake') ok(lakeAt(r.pts[0]).out, `${r.name}: its lake has an outlet`);
ok(meta.falls.length >= 3 && meta.falls.every((f) => f.top - f.foot >= 8), 'falls of at least 8 m');
console.log(`${meta.falls.length} falls: ${meta.falls.map((f) => `${f.river} ${(f.top - f.foot).toFixed(0)} m`).join(', ')}`);
console.log(fails ? `${fails} FAILED` : 'ALL PASS'); process.exit(fails ? 1 : 0);
