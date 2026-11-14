// Development only: looks for pixels left behind by scrolling. Plays our build and the reference
// side by side (same bot, same keys) and reports passes where the reference shows plain background
// (sky, wall or floor fill) but ours does not, or with --at X,Y follows one pixel in ours.
// usage: node tools/trails.mjs [--frames N] [--at X,Y]
import { Game } from '../src/index.js';
import { Gfx } from '../src/gfx.js';
import { memoryFor, botPlay } from './lib.mjs';
const args = process.argv.slice(2), opt = (k, d) => { const i = args.indexOf('--' + k); return i < 0 ? d : args[i + 1]; };
const frames = +opt('frames', 3400);
const lit = (m, page, x, y) => { const at = (page << 8) + (((y & 7) << 10) | (((y >> 3) & 7) << 7) | ((y >> 6) * 40)) + Math.floor(x / 7); return (m[at] >> (x % 7)) & 1; };
// what plain background looks like at (x, y): 1 lit, 0 unlit, -1 not a background row we know
const plain = (x, y) => (y < 100 ? (x & 1 ? 0 : 1) : y >= 112 && y < 153 ? 0 : -1);
let log = [];
for (const k of ['blit', 'blitRight', 'blitMirror', 'blitMirrorRight']) { const f = Gfx.prototype[k]; Gfx.prototype[k] = function () { const m = this.mem, p = m[3] | (m[4] << 8); if (this.tag) log.push(p.toString(16) + ' ' + m[p] + 'x' + m[p + 1] + ' ' + k + ' col ' + (m[5] << 24 >> 24) + ' row ' + m[6] + ' shift ' + m[0x10] + ' page ' + m[7].toString(16)); return f.call(this); }; }
function run(ours) { const pages = []; const g = new Game(memoryFor(ours)); g.g.tag = ours; const logs = []; botPlay(g, { seed: 2, cheat: true, frames, each: (g, n) => { const f = g.flips.at(-1); pages.push(f ? (f.page ? f.page.slice() : g.m.slice(f.show << 8, (f.show << 8) + 0x2000)) : null); if (ours) { logs.push(log); log = []; } } }); return { pages, logs }; }
const ref = run(false), our = run(true);
const px = (page, x, y) => lit(page, 0, x, y);
if (opt('at', '')) {
  const [X, Y] = opt('at').split(',').map(Number); let was = -2;
  for (let n = 0; n < our.pages.length; n++) { const p = our.pages[n]; if (!p) continue; const v = px(p, X, Y); if (v !== was) { console.log('pass', n, 'pixel', v, 'reference', ref.pages[n] ? px(ref.pages[n], X, Y) : '-'); for (const l of our.logs[n]) { const k = /col (-?\d+) row (\d+)/.exec(l), w = /(\d+)x(\d+)/.exec(l); if (+k[2] <= Y && Y < +k[2] + +w[1]) console.log('   ', l); } was = v; } }
} else {
  // passes in which the reference shows nothing but sky above the fence (the mountain apart): ours must show nothing either
  const zone = (x, y) => !(y >= 78 && x > 80 && x < 196);
  const count = new Map(); let empty = 0, passes = 0;
  for (let n = 0; n < our.pages.length; n++) { const a = our.pages[n], b = ref.pages[n]; if (!a || !b) continue;
    let clean = true; for (let y = 0; y < 100 && clean; y++) for (let x = 0; x < 280; x++) if (zone(x, y) && px(b, x, y) !== plain(x, y)) { clean = false; break; }
    if (!clean) continue; empty++; let any = false;
    for (let y = 0; y < 100; y++) for (let x = 0; x < 280; x++) if (zone(x, y) && px(a, x, y) !== plain(x, y)) { any = true; const k = x + ',' + y, e = count.get(k) ?? [0, n, n]; e[0]++; e[2] = n; count.set(k, e); }
    if (any) passes++; }
  console.log(empty, 'passes where the reference has an empty sky;', passes, 'of them with stray pixels in ours;', count.size, 'different pixels');
  console.log([...count].sort((p, q) => q[1][0] - p[1][0]).slice(0, 30).map(([k, [c, f, l]]) => k + ' x' + c + ' passes ' + f + '-' + l).join('\n'));
}
