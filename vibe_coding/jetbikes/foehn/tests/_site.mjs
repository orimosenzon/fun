import { loadWorld } from './util.mjs';
import { groundAt } from '../js/sim/terrain.js';
loadWorld();
const [cx, cz] = [+process.argv[2], +process.argv[3]], st = +process.argv[4] || 60, n = 9;
for (let j = -n; j <= n; j++) { let s = ''; for (let i = -n; i <= n; i++) s += String(groundAt(cx + i * st, cz + j * st) | 0).padStart(5); console.log(s); }
// line of sight from the village / base / pier (eye 2 m) to 30 m above the site
for (const [name, p] of [['village', [520, 2040]], ['base', [-1800, 2330]], ['pier', [1500, 2260]], ['lake mid', [2600, 1500]]]) {
  const a = [p[0], groundAt(p[0], p[1]) + 2, p[1]], b = [cx, groundAt(cx, cz) - 30, cz]; let clear = 1e9;
  for (let t = 0.02; t < 0.97; t += 0.005) clear = Math.min(clear, a[1] + (b[1] - a[1]) * t - groundAt(a[0] + (b[0] - a[0]) * t, a[2] + (b[2] - a[2]) * t));
  console.log(name, 'dist', Math.hypot(a[0] - b[0], a[2] - b[2]) | 0, 'min clearance of sight line', clear | 0);
}
