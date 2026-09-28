// The ring course: 12 rings along the valley, over the lake, past the rock island and
// through the gorge, then a precision landing on the meadow at the far end.

import { surfaceAt, valleyZ, LAKE, HILL, LAND, PAD } from './terrain.js';

const lakeZ = valleyZ(LAKE.x) + LAKE.dz;
const hz = valleyZ(HILL.x) + HILL.dz;
// [x, z, height above the surface]
const RAW = [
  [140, valleyZ(140), 7],
  [330, valleyZ(330) - 12, 13],
  [540, lakeZ + 6, 4.5],        // skim the lake
  [720, lakeZ - 4, 4.5],
  [HILL.x - 40, hz + 150, 16],  // south of the rock island
  [1330, valleyZ(1330) + 20, 12],
  [1590, valleyZ(1590) - 10, 34],
  [1830, valleyZ(1830), 16],    // into the gorge
  [2050, valleyZ(2050) + 4, 10],
  [2290, valleyZ(2290) - 4, 22],
  [2520, valleyZ(2520) + 50, 55], // pop up high
  [2700, valleyZ(2700) - 25, 14],
];

export const RING_R = 7.5;       // inner radius, metres
export const LANDING = { x: LAND.x, z: LAND.z, r: 6 };

export function buildCourse() {
  const rings = RAW.map(([x, z, agl]) => ({ p: [x, surfaceAt(x, z) + agl, z] }));
  const pts = [[PAD[0], 3, PAD[2]], ...rings.map((r) => r.p), [LAND.x, surfaceAt(LAND.x, LAND.z) + 5, LAND.z]];
  rings.forEach((r, i) => {
    const a = pts[i], b = pts[i + 2];
    const d = [b[0] - a[0], (b[1] - a[1]) * 0.5, b[2] - a[2]];
    const l = Math.hypot(...d);
    r.n = d.map((v) => v / l);    // ring normal = direction of travel
  });
  return rings;
}
