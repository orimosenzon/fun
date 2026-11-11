// Generator for the mawashi-geri art. It composes the two new poses from pieces of the
// existing poses (rendered by the original draw routine) plus drawn limbs, prints them as
// ASCII and writes a preview. The frozen result lives in src/ext/mawashi_art.js.
// usage: node tools/mawashi_art.mjs preview.png [--write]
import fs from 'node:fs';
import path from 'node:path';
import { startGame, root } from './lab.mjs';
import { drawPoses } from './pose.mjs';
import { writePNG } from './emu/apple2.mjs';
import { hgrToIndexed, PALETTE } from '../src/vm/machine.js';

const W = 84, H = 48, Y0 = 114, X0 = 49; // grid col 21 = the fighter's anchor pixel (7 * $22 + $50)
const m = startGame();
function grab(pose) {
  const { page } = drawPoses(m, [[pose]], { x0: 10 });
  const g = [];
  for (let y = 0; y < H; y++) { const sy = Y0 + y, b = ((sy & 7) << 10) | (((sy >> 3) & 7) << 7) | ((sy >> 6) * 40); g.push([]);
    for (let x = 0; x < W; x++) g[y].push((page[b + (((x + X0) / 7) | 0)] >> ((x + X0) % 7)) & 1 ? '#' : '.'); }
  return g;
}
const blank = () => Array.from({ length: H }, () => Array(W).fill('.'));
const R = (row) => row - Y0;
function paste(dst, src, c0, c1, r0, r1, dx = 0, dy = 0) { for (let r = R(r0); r <= R(r1); r++) for (let c = c0; c <= c1; c++) if (src[r][c] === '#') { const rr = r + dy, cc = c + dx; if (rr >= 0 && rr < H && cc >= 0 && cc < W) dst[rr][cc] = '#'; } }
function set(g, c, row, ch = '#') { const r = R(row); if (r >= 0 && r < H && c >= 0 && c < W) g[r][c] = ch; }
function limb(g, x1, y1, x2, y2, t1, t2 = t1, ch = '#') { // tapered thick segment
  for (let row = Y0; row < Y0 + H; row++) for (let c = 0; c < W; c++) {
    const dx = x2 - x1, dy = y2 - y1, L = dx * dx + dy * dy; let u = ((c - x1) * dx + (row - y1) * dy) / L; u = Math.max(0, Math.min(1, u));
    const d = Math.hypot(c - x1 - u * dx, row - y1 - u * dy); if (d <= (t1 + (t2 - t1) * u) / 2) set(g, c, row, ch);
  }
}
function rect(g, c0, c1, r0, r1, ch) { for (let r = r0; r <= r1; r++) for (let c = c0; c <= c1; c++) set(g, c, r, ch); }
function text(g, row, col, s) { [...s].forEach((ch, i) => { if (ch !== ' ') set(g, col + i, row, ch); }); }

const p1 = grab(1), p2 = grab(2), p3 = grab(3);

// ---- kick: the body turned over and leaning away, the leg coming round level, knee leading
function kick() {
  const g = blank();
  paste(g, p2, 35, 42, 141, 161);            // supporting leg
  paste(g, p2, 20, 38, 126, 140);            // head, chest
  rect(g, 29, 40, 120, 128, '.');            // drop the raised arm of the high kick
  text(g, 127, 29, '..##');                  // shoulder
  text(g, 128, 28, '######');
  limb(g, 24, 135.5, 14, 137, 4.6, 4);       // rear arm thrown back for balance
  rect(g, 8, 13, 138, 139, '.'); text(g, 135, 9, '#.#'); text(g, 136, 9, '#.#.'); text(g, 137, 9, '#.#.');
  rect(g, 37, 44, 129, 133, '.');
  limb(g, 33, 131, 37, 131.5, 5, 4);         // front arm folded, fist in front of the chest
  text(g, 130, 39, '#.#'); text(g, 131, 39, '#.#'); text(g, 132, 39, '#.#');
  limb(g, 41, 137.5, 53, 131.5, 6.5, 5.6);   // thigh
  limb(g, 53, 131.5, 64, 131, 5.2, 3.8);     // shin
  text(g, 129, 65, '..#.#.#'); text(g, 130, 65, '#.#.#.#.#'); text(g, 131, 65, '#.#.#');   // foot, instep forward
  rect(g, 39, 40, 136, 141, '.'); text(g, 135, 39, '.');   // the gap between the legs
  text(g, 161, 33, '#.#.#.#'); text(g, 160, 37, '#.#.#'); // supporting foot turned out
  return g;
}
// ---- chamber: knee lifted to the side, the shin folded back level behind it
function chamber() {
  const g = blank();
  paste(g, p1, 33, 41, 147, 161);            // supporting leg
  paste(g, p1, 22, 48, 121, 139, -2, 0);     // head, chest, arms
  rect(g, 28, 50, 137, 147, '.');
  limb(g, 36, 142, 36, 147, 7, 7);           // top of the supporting leg
  limb(g, 30, 138, 38, 138, 4, 4);           // hips
  limb(g, 39, 140.5, 45, 139.5, 8, 8.6);     // knee, pointing out of the screen
  limb(g, 43, 140.5, 28, 141.2, 4.4, 3.2);   // shin, folded back
  rect(g, 24, 41, 143, 143, 'o');            // dark line under the folded shin
  rect(g, 28, 36, 139, 139, 'o');
  text(g, 140, 21, '#.#.#'); text(g, 141, 19, '#.#.#.#'); text(g, 142, 21, '#.#.#'); // foot behind
  return g;
}
const frames = { chamber: chamber(), kick: kick() };
for (const [k, g] of Object.entries(frames)) { console.log(k); g.forEach((r, i) => console.log(String(i + Y0).padStart(3), r.join(''))); }

// preview: [pose 6] chamber kick [pose 3] in colour
import('../src/ext/mawashi.js').then(({ blitArt }) => {
  const r = m.ram; drawPoses(m, [[6]], { x0: 3 });
  const page = r.subarray(0x2000, 0x4000);
  blitArt(r, 0x2000, 7 * 12, frames.chamber.map((x) => x.join('')));
  blitArt(r, 0x2000, 7 * 20 + 3, frames.kick.map((x) => x.join('')));
  r[0x20] = r[0x21] = 3; r[0x22] = 31; r[0x2c] = 1; r[0x50] = 0;
  const idx = hgrToIndexed(page), y0 = 100, h = 70, rgb = new Uint8Array(280 * h * 3);
  for (let i = 0; i < 280 * h; i++) rgb.set(PALETTE[idx[y0 * 280 + i]], i * 3);
  writePNG(process.argv[2], 280, h, rgb, 5);
  { const c0 = 60, w = 160, big = new Uint8Array(w * h * 3); for (let y = 0; y < h; y++) big.set(rgb.subarray((y * 280 + c0) * 3, (y * 280 + c0 + w) * 3), y * w * 3); writePNG(process.argv[2].replace('.png', '_big.png'), w, h, big, 9); }
  if (process.argv.includes('--write')) {
    const out = '// Generated by tools/mawashi_art.mjs. Rows start at screen row ' + Y0 + '; column 21 is the fighter\'s anchor pixel.\n// # = pixel on, o = pixel off, . = leave the screen as it is\n' +
      'export const ART_Y = ' + Y0 + ', ART_ANCHOR = 21;\n' + Object.entries(frames).map(([k, g]) => `export const ${k.toUpperCase()} = [\n${g.map((x) => `  '${x.join('')}',`).join('\n')}\n];\n`).join('');
    fs.writeFileSync(path.join(root, 'src/ext/mawashi_art.js'), out);
  }
});
