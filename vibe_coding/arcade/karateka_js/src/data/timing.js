// How long the original's drawing routines take, in processor cycles: measured on the original
// (tools/measure_draw.mjs) and fitted with a straight-line model per kind of call.
// SHAPE[plain or mirrored][mode: OR, AND, replace] =
//   [per call, per row, per byte, more per byte when shifted, more per byte per pixel of shift, cost of a call that is entirely off the screen]
export const SHAPE = [
  [[156.01, 143.59, 57.23, 16.82, 3.54, 308.5], [178.84, 153.57, 57.54, 13.85, 4.11, 140.6], [183.72, 145.46, 56.37, 11.35, 4.87, 166.6]],
  [[289.07, 163.98, 76.6, 46.77, -5.18, 457.9], [217.67, 179.71, 75.6, 49.11, -5.58, 317.7], [228.54, 197.64, 71.24, 43.31, -4.73, 237.9]],
];
export const FILL = [[76, 48.61, 13.45], [74.97, 55.35, 13.51]];   // [per call, per row, per byte], plain and two-pattern
export const CLEAR = 90579, EDGE = 6228;                             // clearing a page; the end-of-frame touch-up of the right edge
export const PASS = 6300;                                            // everything else in a frame: the thinking, not the drawing (the median of what was measured)
// Our own fighters are cut into fewer and larger pieces than the original's (which sit below $C000), so drawing one would
// take the old processor longer. To keep the pace of the game what it was, their pieces are charged this share of their cost:
// fitted so that a whole scene lasts as long on our pictures as on the original's (tools/test_own.mjs prints the ratio).
export const OWN_FIGHTER = 0.78;
