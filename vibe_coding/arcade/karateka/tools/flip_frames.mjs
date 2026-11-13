// The acrobatics, frame by frame: joint angles keyed by eye from slow-motion footage of
// gymnasts (standing back tuck, front tuck, back handspring, round-off + back tuck).
// `hip` is the height of the hip above the floor in pixels; without it the figure stands
// on whatever part of it is lowest. `move` is how many columns the fighter travels.
// usage: node tools/flip_frames.mjs preview.png [sequence ...]   |   node tools/flip_frames.mjs --write
// (--write freezes the frames into src/ext/acro_art.js, which is what the game loads)
import { draw, patch, W, H, CX, CY } from './figure.mjs';

const crouch = { T: 158, head: 8, leg: [70, 82, 90], leg2: [78, 92, 90] };
const tuck = { T: 180, head: -28, leg: [130, 140], arm: [64, 34] };

const stand = { T: 180, leg: [3, 4, 90], leg2: [-6, 2, 90] };
const air = (R, hip, more = {}) => ({ ...tuck, R, hip, air: true, ...more });
const salute = { ...stand, T: 183, head: 6, arm: [140, -6], arm2: [220, 6], order: ['arm2', 'leg2', 'torso', 'leg', 'arm', 'head'] };
const crane = { T: 180, head: 0, leg: [102, 122, 24], leg2: [0, 2, 90], arm: [122, 28], arm2: [-122, -28] };   // the crane stance: on one leg, knee up, arms out like wings
const back = { flip: true, lookBack: true };   // his back to the opponent, looking at him over the shoulder
const straight = { T: 180, head: 0, leg: [2, 3, 20], arm: [172, 4], torsoLine: false };   // stretched out, toes pointed

// mawashi-geri, after a Shotokan black belt filmed from the side in slow motion: the knee comes up
// folded, the hips turn over and the leg whips round level, the body leaning away from the kick
const support = (lean) => [-lean - 2, 5, 90];
const kick = (T, thigh, knee, more = {}) => ({ T, head: -(T - 180) * 0.8, leg: [thigh - (T - 180), knee, more.foot], leg2: support(T - 180), ...more });

export const SEQ = {
  mawashi: [
    kick(190, 108, 152, { arm: [50, 86], arm2: [26, 104], wide: 10.2 }),                              // chamber: knee up, heel to the hip
    kick(202, 102, 74, { arm: [10, 60], arm2: [36, 98], wide: 10.6, reach: 1.1 }),                    // the hips turn over, the shin swings out
    kick(214, 96, 2, { arm: [-46, 16], arm2: [26, 112], wide: 10.8, foot: 98, reach: 1.22 }),         // full extension
    kick(216, 98, 0, { arm: [-52, 12], arm2: [26, 112], wide: 10.8, foot: 102, reach: 1.22 }),
    kick(198, 98, 108, { arm: [8, 64], arm2: [36, 98], wide: 10.4 }),                                   // snapped back
  ],
  // yoko-geri kekomi, the side thrust kick (Yaniv Waldman's idea): the knee comes up high and folded, the hips
  // turn side-on, and the heel drives straight out at the ribs with the hip pushed in behind it, the foot's
  // edge leading and the body leaning away in one line with the leg
  yoko: [
    kick(188, 120, 150, { arm: [46, 92], arm2: [30, 102], wide: 10.2 }),
    kick(206, 106, 98, { arm: [24, 72], arm2: [34, 100], wide: 10.6, foot: 40 }),
    kick(226, 93, 0, { arm: [40, 6], arm2: [26, 112], wide: 10.8, foot: 24, reach: 1.3, dx: 6, leg2: [-48, 5, -80] }),   // the standing foot has pivoted, heel to the target
    kick(228, 94, 0, { arm: [42, 4], arm2: [26, 112], wide: 10.8, foot: 22, reach: 1.3, dx: 6, leg2: [-50, 5, -80] }),
    kick(200, 106, 122, { arm: [24, 72], arm2: [34, 100], wide: 10.4 }),
  ],
  // ushiro-geri, the back kick (Yaniv Waldman's idea): he turns his back on the opponent, looks over his
  // shoulder, folds the knee under him and drives the heel straight back, the body leaning forward away from it
  ushiro: [
    { T: 180, head: 0, wide: 10.6, leg: [8, 10, 90], leg2: [-8, 6, 90], arm: [34, 108], arm2: [22, 100] },                       // the pivot, feet together
    { ...back, T: 170, head: 8, leg: [92, 128], leg2: [12, 8, 90], arm: [52, 108], arm2: [44, 100] },                             // back turned, knee up
    { ...back, T: 150, head: 26, leg: [42, 112, 20], leg2: [34, 10, 90], arm: [58, 104], arm2: [50, 100] },                       // the heel cocked at the target
    { ...back, T: 128, head: 44, leg: [-40, 2, 12], leg2: [57, 9, 90], reach: 1.16, arm: [62, 100], arm2: [54, 100] },      // the heel driven straight back
    { ...back, T: 125, head: 46, leg: [-38, 0, 10], leg2: [60, 9, 90], reach: 1.16, arm: [62, 100], arm2: [54, 100] },
    { ...back, T: 162, head: 14, leg: [58, 118, 20], leg2: [22, 8, 90], arm: [54, 106], arm2: [46, 100] },                        // the leg snaps back under him
  ],
  // the crane kick of The Karate Kid (1984), Arik Shemesh's idea: the fighter stands on one leg, knee up and
  // arms spread, hops, and snaps a front kick to the face with the leg he was standing on (a stylised
  // mae-tobi-geri), landing on the other leg
  crane: [
    { ...crane, hold: 2, move: 0 },                                                                                     // the stance, held: he waits
    { T: 183, head: -2, leg: [74, 104, 30], leg2: [58, 66, 30], arm: [104, 34], arm2: [-104, -34], hip: 27, move: 1, air: true, swish: true },   // the hop: the standing leg folds, the raised one starts down
    { T: 193, head: -9, leg: [97, 2, 118], leg2: [8, 22, 30], reach: 1.2, arm: [34, 84], arm2: [-58, -26], hip: 29, move: 1, air: true, strike: true, swish: true, hold: 1 },   // front snap kick to the face
    { T: 185, head: -3, leg: [84, 112, 30], leg2: [-2, 6, 90], arm: [44, 84], arm2: [24, 92], move: 0, hold: 1 },         // down on the other leg, the kicking leg snapped back
  ],
  // standing back tuck
  back: [
    { ...crouch, arm: [-55, 12], arm2: [-48, 12], move: 0 },
    { T: 186, head: 12, leg: [4, 6, 30], leg2: [10, 12, 30], arm: [168, 6], arm2: [160, 6], hip: 30, move: 1, air: true, swish: true },
    air(62, 40, { leg: [112, 118], move: 1 }),
    air(150, 47, { move: 1 }),
    air(238, 44, { move: 1 }),
    air(318, 32, { head: -8, leg: [84, 70], leg2: [90, 80], arm: [70, 16], arm2: [64, 16], move: 1 }),
    { ...crouch, arm: [62, 22], arm2: [54, 22], move: 1 },
  ],
  // standing front tuck
  front: [
    { ...crouch, arm: [-55, 12], arm2: [-48, 12], move: 0 },
    { T: 168, head: -6, leg: [14, 6, 40], leg2: [20, 12, 40], arm: [150, 8], arm2: [142, 8], hip: 30, move: 1, air: true, swish: true },
    air(-75, 40, { leg: [112, 118], move: 1 }),
    air(-165, 47, { move: 1 }),
    air(-250, 44, { move: 1 }),
    air(-325, 32, { head: -8, leg: [72, 60], leg2: [80, 70], arm: [80, 16], arm2: [72, 16], move: 1 }),
    { ...crouch, arm: [62, 22], arm2: [54, 22], move: 1 },
  ],
  // the floor routine, backwards: back handspring, then a double back tuck twice as high, and a salute
  backShow: [
    { T: 172, head: 6, leg: [60, 70, 90], leg2: [66, 78, 90], arm: [-40, 8], arm2: [-34, 8], move: 0 },            // sit, arms swing down
    { T: 222, head: 30, leg: [-22, 14, 60], leg2: [-14, 22, 60], arm: [188, -6], arm2: [180, -6], hip: 24, move: 1, air: true, swish: true },  // jump back, arched, reaching
    { R: 180, T: 172, head: 22, leg: [-30, 10], leg2: [-8, 6], arm: [164, 6], arm2: [196, -6], move: 3, air: true },    // through the handstand
    { R: 280, T: 205, head: 0, leg: [58, 8, 80], arm: [150, 6], hip: 21, move: 2, air: true },                      // snap down, feet coming under
    { T: 184, head: 6, leg: [6, 10, 40], leg2: [12, 16, 40], arm: [176, 4], arm2: [168, 4], hip: 27, move: 0, air: true, swish: true },  // punch off the floor
    air(80, 46, { leg: [112, 118], move: 0 }),
    air(200, 62, { move: 1 }),
    air(320, 71, { move: 0 }),
    air(440, 72, { move: 1 }),
    air(560, 64, { move: 0 }),
    air(668, 44, { head: -8, leg: [86, 72], leg2: [92, 82], arm: [72, 16], arm2: [66, 16], move: 0 }),
    { ...crouch, arm: [70, 14], arm2: [62, 14], move: 0, air: true },                                                 // stuck landing
    { ...salute, hold: 2, move: 0 },                                                                                  // and the salute to the judges
  ],
  // the floor routine, forwards: front handspring, double front tuck, salute
  frontShow: [
    { T: 150, head: -4, leg: [40, 30, 90], leg2: [-34, 6, 60], arm: [150, 6], arm2: [142, 6], move: 1, air: true }, // lunge, arms up
    { R: -118, T: 180, head: 20, leg: [-64, 6], leg2: [8, 4], arm: [176, 4], move: 0, air: true },                  // hands down, back leg kicks over
    { R: -180, T: 176, head: 22, leg: [-20, 6], leg2: [10, 4], arm: [164, 6], arm2: [196, -6], move: 1, air: true }, // handstand
    { R: -292, T: 208, head: 20, leg: [-24, 12, 100], leg2: [-12, 20, 100], arm: [186, -4], hip: 20, move: 1, air: true },   // flying over, arched
    { T: 172, head: -4, leg: [10, 10, 40], leg2: [16, 16, 40], arm: [166, 6], arm2: [158, 6], hip: 27, move: 0, air: true, swish: true }, // punch
    air(-80, 46, { leg: [112, 118] }),
    air(-200, 62),
    air(-320, 71),
    air(-440, 72),
    air(-560, 64),
    air(-668, 44, { head: -8, leg: [70, 58], leg2: [78, 68], arm: [84, 16], arm2: [76, 16], move: 0 }),
    { ...crouch, arm: [70, 14], arm2: [62, 14], move: 0, air: true },
    { ...salute, hold: 2, move: 0 },
  ],
};

// the art of one frame, and where its top-left corner goes relative to the floor row
export function frameArt(f) {
  const rows = patch(draw(f, { dx: f.dx ?? 0 }), f.patch);
  let low = 0; rows.forEach((r, i) => { if (/[#o]/.test(r)) low = i; });
  const hip = f.hip ?? low - CY;            // standing: the lowest pixel is on the floor
  return { rows, top: -(hip + CY), hip };   // top = rows above the floor row where the grid starts
}
// the same, cut down to the pixels that matter: ax = the column of the hip, top = first row relative to the floor row
export function packed(f) {
  const a = frameArt(f), used = a.rows.map((r, i) => /[#o]/.test(r) ? i : -1).filter((i) => i >= 0);
  const r0 = used[0], r1 = used.at(-1), cols = a.rows.flatMap((r) => [...r].map((ch, c) => ch === '.' ? -1 : c).filter((c) => c >= 0));
  const c0 = Math.min(...cols), c1 = Math.max(...cols);
  const out = { rows: a.rows.slice(r0, r1 + 1).map((r) => r.slice(c0, c1 + 1)), ax: CX - c0, top: a.top + r0 };
  for (const k of ['move', 'air', 'hold', 'swish', 'strike']) if (f[k]) out[k] = f[k];
  return out;
}

if (import.meta.url === `file://${process.argv[1]}`) {
  const { writePNG } = await import('./emu/apple2.mjs');
  const { hgrToIndexed, PALETTE } = await import('../src/vm/machine.js');
  const { blitArt } = await import('../src/ext/mawashi.js');
  if (process.argv[2] === '--write') {
    const fs = await import('node:fs');
    const body = Object.entries(SEQ).map(([k, fr]) => `  ${k}: [\n${fr.map((f) => { const p = packed(f), { rows, ...meta } = p; return `    { ${Object.entries(meta).map(([a, b]) => `${a}: ${b}`).join(', ')}, rows: [\n${rows.map((r) => `      '${r}',`).join('\n')}\n    ] },`; }).join('\n')}\n  ],`).join('\n');
    fs.writeFileSync(new URL('../src/ext/acro_art.js', import.meta.url), `// Generated by tools/flip_frames.mjs --write. Do not edit: change the poses there.\n// # = pixel on, o = pixel off, . = leave the screen as it is. ax = the column of the hip,\n// top = the first row relative to the floor row, move = columns travelled, air = cannot be hit,\n// strike = the frame on which the kick lands.\nexport const ART = {\n${body}\n};\n`);
    console.log('wrote src/ext/acro_art.js:', Object.entries(SEQ).map(([k, v]) => `${k} ${v.length}`).join(', '));
    process.exit(0);
  }
  const [out, ...names] = process.argv.slice(2);
  const seqs = names.length ? names : Object.keys(SEQ);
  const FLOOR = 120, PH = 130, per = 5, tiles = [];
  for (const name of seqs) for (let i = 0; i < SEQ[name].length; i += per) {
    const mem = new Uint8Array(0x10000).fill(0x80);
    SEQ[name].slice(i, i + per).forEach((f, k) => { const a = frameArt(f); blitArt(mem, 0x2000, 24 + k * 54, a.rows, FLOOR + a.top, CX); });
    const idx = hgrToIndexed(mem.subarray(0x2000, 0x4000));
    for (let x = 0; x < 280; x++) if (!idx[(FLOOR + 1) * 280 + x]) idx[(FLOOR + 1) * 280 + x] = 6;
    tiles.push(idx.subarray(0, 280 * PH));
  }
  const rgb = new Uint8Array(280 * PH * tiles.length * 3), pal = [...PALETTE, [70, 70, 70]];
  tiles.forEach((t, k) => { for (let i = 0; i < t.length; i++) rgb.set(pal[t[i]], (k * 280 * PH + i) * 3); });
  writePNG(out, 280, PH * tiles.length, rgb, 7);
}
