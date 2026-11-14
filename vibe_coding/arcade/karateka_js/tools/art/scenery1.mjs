// The pictures of the first scene that are not fighters: the mountain, the fence, the cliff, the
// torii, the gate house, the warlord's hall and the warlord himself, the flash of a blow, the hero
// getting up at the cliff's edge and falling off it, the strength arrows and the closing words.
// All of it is drawn here, by rule or by hand; nothing is taken from the original's pictures.
//
// How it works: the engine puts every picture together from small shapes, each at a fixed place
// (the places are in the engine and in data/rules1.js). So each picture is painted whole on a
// canvas the size of the screen, and then cut into the boxes the engine draws, at the places
// where it draws them. `cut(canvas, address, what, x, row, bytes wide, rows)`; x is in pixels.
// usage: node tools/art/scenery1.mjs
import fs from 'node:fs';
import { Canvas, dataFile } from './canvas.mjs';
import { paint, HERO } from './fighters.mjs';

const out = [];
function grab(c, x, row, wide, rows, { mirror = false, mask = false } = {}) {
  const bytes = [rows, wide], n = wide * 7;
  for (let y = row; y < row + rows; y++) for (let b = 0; b < wide; b++) {
    let v = 0;
    for (let i = 0; i < 7; i++) { const p = b * 7 + i, cell = c.get(x + (mirror ? n - 1 - p : p), y); if (mask ? cell < 0 : cell > 0) v |= 1 << i; }
    bytes.push(v);
  }
  return bytes;
}
// mirror: the engine draws this shape turned left to right, so it is stored turned; mask: painted cells clear the screen
const cut = (c, addr, what, x, row, wide, rows, opt) => out.push([addr, grab(c, x, row, wide, rows, opt), what]);

// ---------- brushes ----------
const dist = (px, py, [x1, y1], [x2, y2]) => { const ex = x2 - x1, ey = y2 - y1, L = ex * ex + ey * ey || 1e-6; let u = ((px - x1) * ex + (py - y1) * ey) / L; u = Math.max(0, Math.min(1, u)); return Math.hypot(px - x1 - u * ex, py - y1 - u * ey); };
// a rounded stroke from a to b, w1 wide at a and w2 at b
function stroke(c, a, b, w1, w2, colour) {
  const ex = b[0] - a[0], ey = b[1] - a[1], L = ex * ex + ey * ey || 1e-6;
  c.paint((x, y) => { let u = ((x - a[0]) * ex + (y - a[1]) * ey) / L; u = Math.max(0, Math.min(1, u)); return Math.hypot(x - a[0] - u * ex, y - a[1] - u * ey) <= (w1 + (w2 - w1) * u) / 2; }, colour,
    [Math.floor(Math.min(a[0], b[0]) - w1 - 1), Math.floor(Math.min(a[1], b[1]) - w1 - 1), Math.ceil(Math.max(a[0], b[0]) + w1 + 2), Math.ceil(Math.max(a[1], b[1]) + w1 + 2)]);
}
const oval = (c, cx, cy, rx, ry, colour) => c.paint((x, y) => ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 <= 1, colour, [Math.floor(cx - rx), Math.floor(cy - ry), Math.ceil(cx + rx) + 1, Math.ceil(cy + ry) + 1]);
// wood painted in the scene's way: a row of orange, a row of black
const hatch = (x, y) => (y & 1 ? 'k' : 'o');

// ---------- what the engine's own fills leave on the screen, to paint on ----------
const RAIL = ['w', 'w', 'k', 'b', 'b', 'b', 'w', 'k'];           // the fence rail, rows 104-111: a beam, the sky through the gap, a thin lower beam
const rail = (c, x1 = 0, x2 = 280) => RAIL.forEach((colour, r) => c.rect(x1, 104 + r, x2, 105 + r, colour));
const floor = (c, x1 = 0, x2 = 280, y1 = 153, y2 = 181) => { for (let y = y1; y < y2; y++) c.rect(x1, y, x2, y + 1, y & 1 ? 'b' : 'o'); };
const sea = (c, x1, x2, y1, y2) => { for (let y = y1; y < y2; y++) c.rect(x1, y, x2, y + 1, y & 1 ? 'b' : 'k'); };
function road() {
  const c = new Canvas();
  c.rect(0, 0, 280, 104, 'b'); rail(c); c.rect(0, 112, 280, 153, 'k'); floor(c); c.rect(0, 181, 280, 192, 'k');
  return c;
}

// ---------- the strength arrows ----------
// seven rows of one byte. Only every other pixel is lit, so the arrow is orange for the player and,
// drawn mirrored one pixel over, blue for the opponent.
{
  const rows = ['#......', '#.#....', '..#.#..', '..#.#.#', '..#.#..', '#.#....', '#......'];
  out.push([0x0b12, [7, 1, ...rows.map((r) => [...r].reduce((v, ch, i) => v | (ch === '#' ? 1 << i : 0), 0))], 'a strength arrow']);
}

// ---------- the mountain and the fence ----------
{
  const c = road();
  // the mountain stands still behind the fence: a cone with hollow flanks, snow on top, haze below
  const cx = 139.5, half = (y) => 3.5 + 0.7 * (y - 81) + 0.028 * (y - 81) ** 2;
  const snow = (x) => 91 + 2.5 * Math.abs(((x + 3) % 8) - 4) - Math.abs(x - cx) * 0.08;      // the ragged lower edge of the snow
  for (let y = 81; y <= 109; y++) {
    if (y >= 104 && y <= 107) continue;                                                      // hidden by the rail
    const h = half(y);
    for (let x = Math.ceil(cx - h); x <= cx + h; x++) {
      const edge = Math.abs(x - cx) > h - 2.2;
      c.put(x, y, y < snow(x) || edge ? 'w' : y & 1 ? 'b' : 'k');
    }
  }
  c.rect(133, 81, 147, 82, 'w');                                                             // the flat of the crater
  for (const [x, y, n] of [[136, 84, 3], [143, 86, 4], [131, 89, 3], [148, 90, 4]]) c.line(x, y, x - n * 0.4, y + n, 'b');   // gullies in the snow
  RAIL.forEach((colour, r) => { if (r < 3 || r > 5) c.rect(0, 104 + r, 280, 105 + r, colour); });
  cut(c, 0xa948, 'the mountain: the peak', 18 * 7, 81, 4, 11);
  cut(c, 0xa976, 'the mountain: the snow line', 16 * 7, 92, 8, 8);
  cut(c, 0xa9b8, 'the mountain: above the fence', 15 * 7, 100, 10, 4);
  cut(c, 0xa9e2, 'the mountain: seen between the rails', 12 * 7, 108, 15, 3);
  // the rail, in tiles two bytes wide, and a post. A post is a full byte of black and white,
  // so it keeps its colours wherever the scrolling puts it.
  const plain = road();
  cut(plain, 0xaa11, 'the fence rail', 0, 104, 2, 8);
  const post = new Canvas(7, 12);
  post.rect(0, 0, 7, 12, 'k'); post.rect(2, 1, 4, 12, 'w'); post.rect(0, 4, 2, 6, 'w');
  cut(post, 0xaa23, 'a fence post', 0, 0, 1, 12);
  // ... and the piece of sky and rail that is put back where the post stood before the view moved.
  // The engine draws it one pixel over, hence the canvas that starts in an odd column.
  const gap = new Canvas(7, 12, 1, 100);
  gap.rect(1, 100, 8, 104, 'b'); RAIL.forEach((colour, r) => gap.rect(1, 104 + r, 8, 105 + r, colour));
  cut(gap, 0xaa31, 'the fence without its post', 1, 100, 1, 12);
}

// ---------- the cliff where the road begins ----------
// Left of pixel 77 there is no road: sky, then the sea in lines down to the bottom. The floor ends
// in two ledges of rock.
const WALL = 77;
const ledge = (y) => (y < 159 ? 80 : y < 167 ? 80 - (y - 158) * 3 : 56 - (y - 167) * 0.5);   // where the floor starts, row by row
function cliff() {
  const c = road();
  c.rect(0, 104, WALL, 114, 'b'); sea(c, 0, WALL, 114, 152); sea(c, 0, WALL, 152, 154); sea(c, 0, 90, 154, 181);
  // the end of the fence: a low post, and the rail starting behind it
  c.rect(WALL, 104, WALL + 5, 112, 'k'); c.rect(WALL + 1, 105, WALL + 3, 112, 'w');
  c.rect(WALL, 112, 280, 153, 'k');
  for (let y = 153; y < 181; y++) {
    const e = Math.round(ledge(y));
    c.rect(e, y, 280, y + 1, y & 1 ? 'b' : 'o');
    const nose = y < 162 ? 22 - Math.abs(y - 157) * 4.5 : y >= 167 && y < 178 ? 13 - Math.abs(y - 172) * 2.6 : 0;   // the rock under each ledge, jutting out
    c.rect(e - Math.max(2, Math.round(nose)), y, e, y + 1, 'k');
  }
  return c;
}
{
  const c = cliff();
  // this piece is always drawn at the left edge of the screen, wherever the view is, so it is the same all along
  const shore = new Canvas(84, 4, 0, 112); shore.rect(0, 112, 84, 114, 'b'); sea(shore, 0, 84, 114, 116);
  cut(shore, 0xab4a, 'the cliff: where the sky meets the sea', 0, 112, 12, 4);
  cut(c, 0xab7c, 'the cliff: the end of the fence', 70, 104, 2, 8);
  cut(c, 0xab8e, 'the cliff: the edge of the wall, one line of the sea', 70, 117, 4, 1);
  const top = new Canvas(105, 4); top.rect(7, 0, 105, 4, 'k');
  cut(top, 0xab94, 'the cliff: the sea cleared off the top of the wall', 0, 0, 15, 4, { mask: true });
  cut(c, 0xaa7d, 'the cliff: the ledges', 42, 152, 6, 29);
}

// ---------- the hero at the cliff: getting up at its edge, and falling off it ----------
// The same jointed figure as the fighters (tools/art/fighters.mjs), in poses of its own: where the
// hips and the ankles are, how far the body leans, where the hands are. Coordinates as there:
// pixels from the figure's own origin, rows from the top of the fighters' strip (row 114).
function hero(c, X, Y, legs, body, { headDown = false } = {}) {
  const { f, head } = paint(legs, body);
  const on = (x, y) => x >= f.x0 && y >= f.y0 && x < f.x0 + f.w && y < f.y0 + f.h && f.lit[f.at(x, y)] === 1;
  const cells = [];
  for (let y = f.y0; y < f.y0 + f.h; y++) for (let x = f.x0; x < f.x0 + f.w; x++) { const v = f.lit[f.at(x, y)]; if (v >= 0) cells.push([x, y, v]); }
  // the head: the hero's own, eight rows of seven pixels (standing on its crown when he tumbles)
  const hx = Math.round(head[0] - 3), hy = Math.round(head[1] - 4);
  (headDown ? [...HERO].reverse() : HERO).forEach((row, j) => [...row].forEach((ch, i) => { const x = hx + (headDown ? 6 - i : i), y = hy + j; if (ch === '#') cells.push([x, y, 1]); else if (ch === 'k') cells.push([x, y, 0]); else if (ch !== '.') cells.push([x, y, x & 1 ? 1 : 0, true]); }));
  // a dark line round him, so that he stands free of the sea and the floor
  const lit = new Set(cells.filter((e) => e[2] === 1 || e[3]).map((e) => e[0] + ',' + e[1]));
  for (const k of lit) { const [x, y] = k.split(',').map(Number); for (const [dx, dy] of [[1, 0], [-1, 0], [0, 1], [0, -1], [2, 0], [-2, 0]]) c.put(X + x + dx, Y + y + dy, 'k'); }
  for (const [x, y, v] of cells) c.put(X + x, Y + y, v);
}
{
  const FLOOR = 46, DOWN = [[1, 3], [-1, 3]];
  // six pictures of two pieces each; the pieces are the boxes the engine draws, [address, pixel, row, bytes, rows]
  const UP = [
    { legs: { hip: [4, 43], feet: [[-3, FLOOR], [-5, FLOOR]] }, body: { lean: 26, hands: [[9, 2], [7, 3]] },                 // on his knees at the very edge
      pieces: [[0xa3c5, 70, 141, 2, 17], [0xa3e9, 63, 158, 4, 8]] },
    { legs: { hip: [6, 38], feet: [[2, FLOOR], [4, FLOOR]] }, body: { lean: 62, hands: [[11, 8], [9, 8]] },                  // hands on the ground
      pieces: [[0xa40b, 77, 140, 3, 8], [0xa425, 70, 148, 3, 17]] },
    { legs: { hip: [10, 39], feet: [[14, FLOOR], [4, FLOOR]] }, body: { lean: 55, hands: [[10, 4], [8, 2]] },                // a foot under him
      pieces: [[0xa45a, 84, 139, 3, 24], [0xa4a4, 70, 143, 2, 22]] },
    { legs: { hip: [10, 39.5], feet: [[14, FLOOR], [5, FLOOR]] }, body: { lean: 40, hands: [[9, 0], [6, -1]] },              // pushing up
      pieces: [[0xa4d2, 77, 137, 5, 6], [0xa4f2, 70, 143, 4, 21]] },
    { legs: { hip: [11, 36], feet: [[14, FLOOR], [7, FLOOR]] }, body: { lean: 20, hands: [[6, -2], [3, 0]] },                // nearly up
      pieces: [[0xa548, 77, 131, 4, 10], [0xa572, 70, 141, 4, 22]] },
    { legs: { hip: [17, 25], feet: [[18, FLOOR], [16, FLOOR]] }, body: { lean: 15, hands: [[4, 2], [2, 3]] },                // straightening up
      pieces: [[0xa5cc, 84, 120, 2, 7], [0xa5dc, 77, 127, 3, 36]] },
  ];
  UP.forEach((f, i) => { const c = cliff(); hero(c, 70, 114, f.legs, f.body); for (const [addr, x, row, wide, rows] of f.pieces) cut(c, addr, `getting up at the cliff, picture ${i + 1}`, x, row, wide, rows); });
  // stepping back off the edge: two pictures. By then the view has moved one pixel, so they start in odd columns.
  const FALL = [
    { at: [70, 114], legs: { hip: [-3, 42], feet: [[16, 44], [14, FLOOR]] }, body: { lean: -35, hands: [[-2, -20], [-12, -4]], elbowUp: true },   // sitting down on nothing, arms flung up
      pieces: [[0x1e00, 49, 132, 4, 19], [0x1e4e, 49, 151, 7, 15]] },
    { at: [46, 146], legs: { hip: [0, 20], feet: [[6, 10], [9, 14]] }, body: { lean: -130, hands: [[-14, 2], [-4, 14]] }, headDown: true,        // head first
      pieces: [[0x1eb9, 42, 155, 2, 12], [0x1ed3, 28, 167, 4, 13]] },
  ];
  FALL.forEach((f, i) => { const c = cliff(); hero(c, f.at[0], f.at[1], f.legs, f.body, f); for (const [addr, x, row, wide, rows] of f.pieces) cut(c, addr, `falling off the cliff, picture ${i + 1}`, x, row, wide, rows); });
}

// ---------- the torii on the road ----------
// Seen at an angle: the near pillar on the right, tall, the far one on the left, the two beams
// running down from right to left. X = the pixel of its left edge.
{
  const c = road(), X = 0, S = 0.43, top = (x) => 30 + (76 - x) * S;
  const beam = (x1, x2, drop, white, wood) => {
    for (let x = x1; x <= x2; x++) {
      const t = Math.round(top(x) + drop), w = Math.round(white(x)), h = Math.round(wood(x));
      c.rect(X + x, t, X + x + 1, t + w, 'w');
      for (let y = t + w; y < t + w + h; y++) c.put(X + x, y, hatch(x, y));
      c.put(X + x, t + w + h, 'k');
    }
  };
  beam(7, 76, 0, (x) => 3 + x / 38, (x) => 3 + x / 25);                                     // the top beam
  for (let y = 30; y < 43; y++) for (let x = 70; x <= 76; x++) if (y >= top(x) + 1 && c.get(X + x, y) >= 0 && y < top(x) + 12) c.put(X + x, y, hatch(x, y));   // its near end, cut square
  beam(2, 62, 18, () => 2, () => 3);                                                        // the tie beam under it
  const pillar = (x1, x2, edge, y2) => {
    for (let x = x1; x <= x2; x++) for (let y = Math.round(top(x) + 4 + x / 12); y <= y2; y++) c.put(X + x, y, x < x1 + edge ? 'w' : hatch(x, y));
    for (let y = Math.round(top(x2) + 10); y <= y2; y++) { c.put(X + x2 + 1, y, 'k'); c.put(X + x2 + 2, y, 'k'); c.put(X + x1 - 1, y, 'k'); }
  };
  // the sill on the ground between the two pillars
  for (let y = 153; y <= 175; y++) { const l = 9 + (y - 153) * 1.5; c.rect(X + Math.round(l), y, X + Math.round(l) + 13, y + 1, 'k'); }
  pillar(9, 18, 4, 152);
  pillar(42, 55, 6, 171);
  c.rect(X + 40, 172, X + 59, 176, 'k');
  cut(c, 0xa707, 'the torii: the near end of the top beam', X + 42, 30, 5, 18);
  cut(c, 0xa857, 'the torii: the top beam', X + 35, 42, 1, 6);
  cut(c, 0xa82b, 'the torii: the top beam', X + 21, 48, 6, 7);
  cut(c, 0xa7d1, 'the torii: the beams', X + 7, 55, 8, 11);
  cut(c, 0xa763, 'the torii: the far end of the beams', X, 66, 6, 18);
  cut(c, 0xa85f, 'the torii: the far pillar against the sky (repeated downwards)', X + 7, 84, 2, 2);
  cut(c, 0xa865, 'the torii: the far pillar at the fence', X + 7, 104, 2, 8);
  cut(c, 0xa877, 'the torii: the far pillar, its first line on the wall', X + 7, 111, 2, 1);
  cut(c, 0xa68a, 'the torii: the far pillar against the wall (repeated)', X + 7, 111 + 2, 2, 2);
  cut(c, 0xa87b, 'the torii: the sill', X, 153, 4, 6);
  cut(c, 0xa6a6, 'the torii: the sill', X + 14, 159, 4, 6);
  cut(c, 0xa6c0, 'the torii: the sill', X + 21, 165, 3, 6);
  cut(c, 0xa6d4, 'the torii: the foot of the near pillar', X + 35, 171, 5, 5);
  cut(c, 0xa684, 'the torii: the near pillar (repeated)', X + 42, 66, 2, 2);
  cut(c, 0xa703, 'the torii: the dark side of the near pillar, against the sky (repeated)', X + 56, 66, 1, 2);
  cut(c, 0xac1a, 'the torii: the same at the fence', X + 56, 104, 2, 8);
  cut(c, 0xa6ef, 'the torii: the same on the floor', X + 56, 153, 1, 18);
}

// ---------- the gate house at the end of the road ----------
// A roofed gate, seen from the same angle: its roof sweeps up to the right and off the screen.
{
  const c = road(), X = 189;                                           // where it stands when the view has stopped (an odd pixel: that fixes its colours)
  const eave = (x) => 84 - 0.3 * x - 0.001 * x * x;                     // the lower edge of the roof
  const room = (x) => (x < 14 ? 77 : x < 28 ? 68 : x < 56 ? 54 : x < 70 ? 46 : 38);   // the highest row the engine's pieces reach above each column
  const ridge = (x) => Math.max(room(x), Math.ceil(eave(x) - (2 + 0.3 * x)));
  // the back pillar and the front one, the lintel between them, the sill on the ground
  const pillar = (x1, x2, y1, y2) => { for (let y = y1; y <= y2; y++) for (let x = x1; x <= x2; x++) c.put(X + x, y, x < x1 + 4 ? 'w' : hatch(x, y)); };
  for (let y = 153; y <= 171; y++) { const l = Math.round(31 + (y - 153) * 2.17); c.rect(X + l, y, X + l + 14, y + 1, 'k'); c.rect(X + l + 14, y, X + l + 16, y + 1, 'w'); }
  c.poly([[X + 44, 86], [X + 70, 72], [X + 70, 79], [X + 44, 93]], hatch);
  c.poly([[X + 44, 85], [X + 70, 71], [X + 70, 73], [X + 44, 87]], 'w');
  c.poly([[X + 44, 93], [X + 70, 79], [X + 70, 80], [X + 44, 94]], 'k');
  pillar(31, 44, Math.round(eave(38)) + 6, 152);
  c.rect(X + 29, 149, X + 47, 153, 'k'); c.rect(X + 30, 149, X + 46, 150, 'w');   // its foot
  pillar(70, 83, Math.round(eave(76)) + 6, 171);
  c.rect(X + 66, 172, X + 87, 176, 'k'); c.rect(X + 84, 153, X + 86, 172, 'k');
  // the roof: dark tiles, a white edge, the painted wood of the eaves under it
  // The roof starts one byte in: the engine also draws these pieces once when it comes back from the hall, far from
  // the gate, and then repaints the sky only from the next byte on. The first byte of the widest piece must stay sky.
  for (let x = 7; x <= 90; x++) {
    const e = Math.round(eave(x)), r = ridge(x);
    for (let y = r; y < e; y++) c.put(X + x, y, (x + 3 * (e - y)) % 14 < 2 && e - y > 2 && y > r + 1 ? 'w' : 'k');   // rows of tiles, catching the light
    c.rect(X + x, e, X + x + 1, e + 2, 'w');
    for (let y = e + 2; y < e + 5; y++) c.put(X + x, y, 'o');
    c.rect(X + x, e + 5, X + x + 1, e + 7, 'k');
  }
  oval(c, X + 10, 83, 3, 2.4, 'w');                                       // the upturned tip of the eaves
  const P = [[0xa006, 4, 100, 3, 12, 'the back pillar at the fence'], [0xa038, 4, 89, 8, 11, 'under the roof'], [0xa0a8, 0, 77, 13, 12, 'the tip of the roof'], [0xa146, 2, 68, 11, 9, 'the roof'],
    [0xa1b4, 4, 54, 9, 14, 'the roof'], [0xa242, 8, 46, 5, 8, 'the roof'], [0xa274, 10, 38, 3, 8, 'the roof'], [0xa2b0, 4, 112, 3, 2, 'the back pillar against the wall (repeated)'],
    [0xa2b8, 9, 172, 4, 6, 'the foot of the front pillar'], [0xa2de, 10, 149, 2, 23, 'the front pillar on the floor'], [0xa30e, 12, 153, 1, 19, 'the shadow beside it'],
    [0xa323, 6, 164, 4, 8, 'the sill'], [0xa34d, 4, 153, 6, 11, 'the sill'], [0xa39c, 4, 140, 3, 13, 'the foot of the back pillar'], [0xa000, 10, 99, 2, 2, 'the front pillar (repeated)']];
  for (const [addr, col, row, wide, rows, what] of P) cut(c, addr, 'the gate house: ' + what, X + col * 7, row, wide, rows);
}

// ---------- the flash of a blow ----------
// A star laid over the fighters; first a slightly bigger star is cleared out of them, so that it
// stands free. Two sizes.
{
  const star = (c, cx, cy, rx, ry, points, colour, turn = 0) => {
    const pts = []; for (let i = 0; i < points * 2; i++) { const a = turn + i * Math.PI / points, k = i & 1 ? 0.42 : 1; pts.push([cx + Math.sin(a) * rx * k, cy - Math.cos(a) * ry * k]); }
    c.poly(pts, colour);
  };
  for (const [maskAt, starAt, wide, rows, n] of [[0x93ab, 0x942a, 5, 20, 8], [0x9490, 0x94e6, 4, 15, 7]]) {
    const w = wide * 7, cx = w / 2 - 0.5, cy = rows / 2 - 0.5;
    const s = new Canvas(w, rows); star(s, cx, cy, w / 2 - 1, rows / 2 - 0.5, n, 'o', 0.2); star(s, cx, cy, w / 3.2, rows / 3.2, n, 'w', 0.2 + Math.PI / n); oval(s, cx, cy, 3.4, 2.2, 'w');
    const hole = new Canvas(w, rows + 5 + (wide === 4 ? 1 : 0), 0, -3);
    star(hole, cx, cy, w / 2 + 1.5, rows / 2 + 2, n, 'k', 0.2); star(hole, cx, cy, w / 3, rows / 3 + 1, n, 'k', 0.2 + Math.PI / n);
    cut(hole, maskAt, 'the flash of a blow: the room cleared for it', 0, -3, wide, rows + 5 + (wide === 4 ? 1 : 0), { mask: true });
    cut(s, starAt, 'the flash of a blow', 0, 0, wide, rows);
  }
}

// ---------- the warlord's hall ----------
// A dark room: a floor of lines running to a far wall, a doorway on each side (the left one is the
// right one mirrored by the engine), and the warlord standing in the middle with his bird.
{
  const c = new Canvas();
  for (let y = 153; y < 182; y += 2) c.rect(0, y, 280, y + 1, 'b');                         // the floor, as the engine fills it
  // the doorway on the right: a lintel that climbs towards us, a far post, a near post
  c.poly([[224, 90], [262, 84], [265, 84], [265, 88], [226, 94], [224, 94]], 'w');
  c.poly([[226, 94], [265, 88], [265, 90], [228, 95]], 'b');
  for (let y = 95; y <= 159; y++) { c.rect(232, y, 236, y + 1, 'w'); c.rect(236, y, 242, y + 1, 'b'); c.rect(242, y, 245, y + 1, 'k'); c.rect(231, y, 232, y + 1, 'k'); }
  for (let y = 95; y <= 171; y++) { c.rect(252, y, 253, y + 1, 'k'); c.rect(253, y, 257, y + 1, 'w'); c.rect(257, y, 259, y + 1, 'b'); }
  cut(c, 0x9600, 'the hall: the lintel of a doorway', 224, 83, 6, 12);
  cut(c, 0x964a, 'the hall: the far post of a doorway', 231, 95, 2, 65);
  cut(c, 0x96ce, 'the hall: the near post of a doorway', 252, 95, 1, 77);
  // the edges of the floor, which narrows towards the far wall
  const edge = new Canvas();
  for (let y = 153; y < 182; y += 2) edge.rect(0, y, 280, y + 1, 'b');
  for (let y = 153; y <= 158; y++) edge.rect(224 + Math.round((y - 153) * 1.2), y, 231, y + 1, 'k');
  for (let y = 173; y <= 181; y++) edge.rect(252 + Math.round((y - 172) * 3), y, 280, y + 1, 'k');
  cut(edge, 0x9743, 'the hall: the far corner of the floor', 224, 153, 1, 6);
  cut(edge, 0x971d, 'the hall: the near corner of the floor', 252, 173, 4, 9);
  // the door leaf, standing open: only its frame, laid over what is there
  const door = new Canvas();
  for (let x = 245; x <= 251; x++) for (let y = Math.round(96 - (x - 245) * 0.5); y <= Math.round(160 + (x - 245) * 0.9); y++) door.put(x, y, 'b');
  door.rect(245, 126, 252, 128, 'k'); door.rect(246, 110, 248, 112, 'k');
  cut(door, 0x9980, 'the hall: the open door', 238, 91, 2, 75);

  // the warlord, facing left. The engine draws all his pieces mirrored, so they are stored turned round.
  const M = { mirror: true };
  const robe = (cv, y1, y2) => { for (let y = y1; y <= y2; y++) { const h = 11 + (y - 122) * 0.16; for (let x = Math.ceil(178 - h); x <= 178 + h; x++) cv.put(x, y, 'o'); } };
  const body = new Canvas();
  robe(body, 122, 162);
  oval(body, 167, 128, 6, 6, 'w'); oval(body, 189, 128, 6, 6, 'w');                         // the shoulder guards
  body.rect(165, 133, 170, 134, 'k'); body.rect(187, 133, 192, 134, 'k');
  for (let i = 0; i < 7; i++) { body.rect(176 - i, 122 + i, 178 - i, 123 + i, 'w'); body.rect(179 + i, 122 + i, 181 + i, 123 + i, 'w'); }   // the collar
  body.rect(165, 140, 192, 142, 'k'); body.rect(174, 139, 180, 143, 'w');                   // the sash and its knot
  body.rect(176, 143, 178, 160, 'w');                                                       // the fold of the robe
  body.rect(160, 161, 197, 163, 'w');
  cut(body, 0x9eb8, 'the warlord: his robe', 161, 121, 5, 42, M);
  cut(body, 0x9858, 'the warlord: his shoulder, put back after the arm moved', 154, 121, 1, 2, M);
  const hole = new Canvas();
  for (let y = 121; y <= 161; y++) { const h = 13.5 + (y - 122) * 0.16; hole.rect(Math.ceil(178 - h), y, Math.floor(178 + h) + 1, y + 1, 'k'); }
  oval(hole, 167, 128, 8, 8, 'k'); oval(hole, 189, 128, 8, 8, 'k'); hole.rect(168, 119, 184, 122, 'k');
  cut(hole, 0x974b, 'the warlord: the room cleared for him', 154, 119, 6, 43, { mirror: true, mask: true });
  const tip = new Canvas(); tip.rect(133, 125, 147, 128, 'k');
  cut(tip, 0x984f, 'the warlord: the room cleared for his pointing hand', 133, 125, 2, 3, { mirror: true, mask: true });
  // the hem and the feet stand on the floor's lines
  const hem = new Canvas();
  for (let y = 163; y <= 171; y += 2) hem.rect(154, y, 196, y + 1, 'b');
  hem.rect(175, 169, 196, 172, 'k');
  for (let y = 163; y <= 166; y++) { const h = 17.6 + (y - 162) * 0.5; hem.rect(Math.ceil(178 - h) - 1, y, Math.floor(178 + h) + 2, y + 1, 'k'); for (let x = Math.ceil(178 - h); x <= Math.min(195, 178 + h); x++) hem.put(x, y, 'o'); }
  hem.rect(158, 167, 196, 169, 'w');
  hem.rect(163, 169, 172, 172, 'k'); hem.rect(164, 169, 171, 171, 'b'); hem.rect(183, 169, 192, 172, 'k'); hem.rect(184, 169, 191, 171, 'b');
  cut(hem, 0x9f8c, 'the warlord: the hem of his robe and his feet', 154, 163, 6, 9, M);
  // the head: a helmet with two horns, the face under it
  const head = new Canvas();
  head.rect(170, 114, 182, 116, 'w'); head.rect(172, 113, 180, 114, 'w');                   // the bowl of the helmet
  head.rect(168, 113, 170, 114, 'w'); head.rect(169, 114, 171, 115, 'w'); head.rect(182, 113, 184, 114, 'w'); head.rect(181, 114, 183, 115, 'w');   // its horns
  head.rect(171, 116, 181, 121, 'o'); head.rect(172, 117, 174, 118, 'k'); head.rect(176, 117, 178, 118, 'k'); head.rect(173, 119, 177, 120, 'k');   // the face: eyes and a hard mouth
  head.rect(180, 116, 182, 121, 'w');                                                       // the neck guard
  cut(head, 0x9a62, 'the warlord: his head', 168, 113, 2, 8, M);
  // three positions of the arm: at his side, rising, pointing
  const arm = (addr, what, x, wide, rows, shoulder, elbow, hand) => {
    const a = new Canvas();
    stroke(a, shoulder, elbow, 6, 5.4, 'w'); stroke(a, elbow, hand, 5.4, 5, 'w');
    const d = Math.hypot(hand[0] - elbow[0], hand[1] - elbow[1]), u = [(hand[0] - elbow[0]) / d, (hand[1] - elbow[1]) / d];
    stroke(a, [hand[0] + u[0] * 1.5, hand[1] + u[1] * 1.5], [hand[0] + u[0] * 6, hand[1] + u[1] * 6], 3, 2, 'o');
    cut(a, addr, what, x, 124, wide, rows, M);
  };
  arm(0x98d3, 'the warlord: his arm at his side', 147, 3, 17, [164, 128], [158, 133], [155, 137]);
  arm(0x9908, 'the warlord: his arm rising', 140, 4, 19, [164, 128], [156, 131], [149, 136]);
  arm(0x9956, 'the warlord: his arm pointing', 133, 5, 8, [164, 128], [152, 128], [143, 127.5]);
  // the bird on his shoulder: the body, and the head looking one way and the other
  const bird = new Canvas();
  oval(bird, 189, 116, 5, 4.6, 'b'); bird.rect(185, 115, 188, 119, 'w'); bird.line(191, 113, 194, 120, 'w'); bird.rect(192, 118, 196, 121, 'b');
  cut(bird, 0x9fc4, 'the bird: its body', 182, 112, 2, 9, M);
  const look = (addr, what, rows, left) => {
    const b = new Canvas(); b.rect(182, 106, 203, 112, 'k');
    b.rect(187, 110, 192, 112, 'w');                                                       // the neck, the same in both
    b.rect(186, 107, 193, 110, 'w'); b.rect(187, 106, 192, 107, 'w');
    if (left) { b.rect(183, 108, 186, 109, 'o'); b.rect(184, 109, 186, 110, 'o'); b.rect(187, 108, 189, 109, 'k'); }
    else { b.rect(193, 108, 197, 109, 'o'); b.rect(193, 109, 195, 110, 'o'); b.rect(190, 108, 192, 109, 'k'); }
    cut(b, addr, what, 182, 106, 3, rows, M);
  };
  look(0x9fd8, 'the bird: its head, looking where he points', 6, true);
  look(0x985c, 'the bird: its head, looking away', 4, false);
}

// ---------- the closing words ----------
{
  const F = {
    T: ['######', '######', '..##..', '..##..', '..##..', '..##..', '..##..', '..##..'],
    H: ['##..##', '##..##', '##..##', '######', '######', '##..##', '##..##', '##..##'],
    E: ['######', '######', '##....', '#####.', '#####.', '##....', '######', '######'],
    N: ['##..##', '###.##', '###.##', '######', '######', '##.###', '##.###', '##..##'],
    D: ['#####.', '######', '##..##', '##..##', '##..##', '##..##', '######', '#####.'],
  };
  const c = new Canvas(70, 10); let x = 6;
  for (const ch of 'THE END') { if (ch === ' ') { x += 6; continue; } c.stamp(x, 1, F[ch]); x += 8; }
  cut(c, 0x9a74, 'the closing words', 0, 0, 10, 10);
}

out.sort((a, b) => a[0] - b[0]);
fs.writeFileSync(new URL('../../src/data/art_scenery1.js', import.meta.url), dataFile('The pictures of the first scene that are not fighters: scenery, the hall and the warlord, the flash, the arrows.', out));
console.log(out.length, 'shapes,', out.reduce((n, e) => n + e[1].length, 0), 'bytes');
// development check, when the catalogue of the scene is at hand: is every shape that the scene draws (outside the fighters) here?
const cat = new URL('../../re/catalog.json', import.meta.url);
if (fs.existsSync(cat)) {
  const have = new Map(out.map((e) => [e[0], e[1]]));
  for (const [p, e] of Object.entries(JSON.parse(fs.readFileSync(cat)))) {
    const a = parseInt(p, 16);
    if (/r6783|r6971|r697C/.test(e.by) || a === 0x9a18 || a === 0x9a2a) continue;
    const mine = have.get(a);
    if (!mine) console.log('missing', p, e.h + 'x' + e.w, e.by); else if (mine[0] !== e.h || mine[1] !== e.w) console.log('size differs', p, 'here', mine[0] + 'x' + mine[1], 'catalogue', e.h + 'x' + e.w);
  }
}
