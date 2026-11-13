// The fortress on the ridge end above the main valley. One list of solids is both what is drawn
// (js/render/castle.js) and what the bike can hit, so every wall, vault and pillar is really there.
// Castle coordinates: metres from SITE, x east (the gate looks west along the ridge), z south
// (towards the valley), y up from the courtyard floor.
import { SITE } from './castleSite.js';
import { groundAt } from './terrain.js';

export const C = { solids: [], deco: [], y0: 0, ready: false };
export const castleY = () => groundAt(SITE.x, SITE.z);

const WH = 24;           // curtain wall
const MER = 2.2;         // battlements on top of it

function plan() {
  const S = [], D = [];
  const box = (x0, x1, y0, y1, z0, z1, o = {}) => S.push({ t: 'box', x0, x1, y0, y1, z0, z1, m: 'stone', ...o });
  const cyl = (x, z, r, y0, y1, o = {}) => S.push({ t: 'cyl', x, z, r, r1: r, y0, y1, m: 'stone', ...o });
  const cone = (x, z, r, y0, y1, o = {}) => S.push({ t: 'cyl', x, z, r, r1: 0, y0, y1, m: 'roof', ...o });
  // a block with an arched tunnel right through it along `axis`; the tunnel is centred on c (the
  // other horizontal coordinate), 2*hw wide, straight up to ys and a half circle above; with yb it
  // starts at a sill instead of the bottom of the block (a window)
  const arch = (axis, x0, x1, y0, y1, z0, z1, c, hw, ys, o = {}) => S.push({ t: 'arch', axis, x0, x1, y0, y1, z0, z1, c, hw, ys, m: 'stone', ...o });
  const gable = (axis, x0, x1, y0, y1, z0, z1, o = {}) => S.push({ t: 'gable', axis, x0, x1, y0, y1, z0, z1, m: 'roof', ...o });   // ridge along axis
  const pyr = (x0, x1, y0, y1, z0, z1, o = {}) => S.push({ t: 'pyr', x0, x1, y0, y1, z0, z1, m: 'roof', ...o });
  const dbox = (x0, x1, y0, y1, z0, z1, o = {}) => D.push({ t: 'box', x0, x1, y0, y1, z0, z1, m: 'stone', ...o });
  const glow = (x, y, z, nx, nz, w = 1.5, h = 3.2) => D.push({ t: 'glow', x, y, z, nx, nz, w, h });
  const torch = (x, y, z) => D.push({ t: 'torch', x, y, z });
  const flag = (x, y, z, h = 9, s = 7) => D.push({ t: 'flag', x, y, z, h, s });
  // battlements along a straight edge (render only: the wall below is solid up to their top)
  const merlons = (x0, z0, x1, z1, y, th = 1.4) => {
    const L = Math.hypot(x1 - x0, z1 - z0), n = Math.max(1, Math.round(L / 5.2)), ux = (x1 - x0) / L, uz = (z1 - z0) / L;
    for (let k = 0; k < n; k++) {
      const s = (k + 0.5) * L / n, cx = x0 + ux * s, cz = z0 + uz * s, hx = Math.abs(ux) * 1.5 + Math.abs(uz) * th / 2, hz = Math.abs(uz) * 1.5 + Math.abs(ux) * th / 2;
      dbox(cx - hx, cx + hx, y, y + MER, cz - hz, cz + hz);
    }
  };
  const tower = (x, z, r, h, roof, o = {}) => {
    cyl(x, z, r, -30, h - 5);
    S.push({ t: 'cyl', x, z, r, r1: r + 1.6, y0: h - 8, y1: h - 5, m: 'stone' });      // corbels
    cyl(x, z, r + 1.6, h - 5, h + 1, { tone: 1.04 });
    cone(x, z, r + 2.6, h + 1, h + 1 + roof);
    flag(x, h + 1 + roof - 0.5, z, 0.32 * roof + 4, 0.2 * roof + 3);
    const rows = o.rows || [h - 2, h * 0.55];
    for (const y of rows) for (let k = 0; k < 6; k++) { const a = (k + 0.5) / 6 * 6.2832, rr = (y > h - 5 ? r + 1.6 : r) + 0.06; glow(x + Math.cos(a) * rr, y, z + Math.sin(a) * rr, Math.cos(a), Math.sin(a), 1.2, 2.6); }
  };
  const wall = (x0, x1, z0, z1, h = WH) => {
    box(x0, x1, -10, h, z0, z1, { cy1: h + MER });
    if (x1 - x0 > z1 - z0) { merlons(x0, z0 + 0.7, x1, z0 + 0.7, h); merlons(x0, z1 - 0.7, x1, z1 - 0.7, h); } else { merlons(x0 + 0.7, z0, x0 + 0.7, z1, h); merlons(x1 - 0.7, z0, x1 - 0.7, z1, h); }
  };

  // ----- curtain wall -----
  wall(-150, 150, -105, -95);
  wall(-150, -14, 95, 105); wall(4, 150, 95, 105);
  arch('z', -14, 4, -10, WH, 95, 105, -5, 6, 7, { cy1: WH + MER });                     // the postern over the abyss
  merlons(-14, 95.7, 4, 95.7, WH); merlons(-14, 104.3, 4, 104.3, WH);
  wall(140, 150, -95, 95);
  wall(-150, -140, -95, -22); wall(-150, -140, 22, 95);
  for (const sx of [-1, 1]) for (const sz of [-1, 1]) tower(150 * sx, 105 * sz, 16, 48, 42);
  for (const x of [-80, 72]) { tower(x, -107, 10, 38, 27); tower(x, 107, 10, 38, 27); }
  tower(152, -40, 10, 38, 27); tower(152, 40, 10, 38, 27);
  // buttresses on the outside, down into the rock
  const butt = (x0, x1, z0, z1) => { box(x0, x1, -30, WH - 5, z0, z1, { tone: 0.96 }); S.push({ t: 'gable', axis: x1 - x0 > z1 - z0 ? 'x' : 'z', x0, x1, y0: WH - 5, y1: WH - 1.5, z0, z1, m: 'stone', tone: 0.96 }); };
  for (let x = -126; x <= 126; x += 21) {
    if (Math.abs(x + 80) > 15 && Math.abs(x - 72) > 15) { butt(x - 1.6, x + 1.6, -109.5, -105); if (Math.abs(x + 5) > 14) butt(x - 1.6, x + 1.6, 105, 109.5); }
  }
  for (let z = -84; z <= 84; z += 21) { if (Math.abs(Math.abs(z) - 40) > 15) butt(150, 154.5, z - 1.6, z + 1.6); if (Math.abs(z) > 40) butt(-154.5, -150, z - 1.6, z + 1.6); }

  // ----- gatehouse and bridge -----
  arch('x', -162, -138, -10, 36, -22, 22, 0, 7, 8, { tone: 1.03 });
  gable('z', -164, -136, 36, 53, -24, 24);
  tower(-162, -25, 11, 46, 30); tower(-162, 25, 11, 46, 30);
  for (const z of [-13, 13]) glow(-162.06, 24, z, -1, 0, 1.6, 3.6);
  glow(-162.06, 27, 0, -1, 0, 2.2, 4.4);
  arch('z', -218, -162, -34, 0, -8, 8, -192, 13, -16);
  box(-218, -162, 0, 1.3, -8, -7.2); box(-218, -162, 0, 1.3, 7.2, 8);
  torch(-162.4, 9, -9.5); torch(-162.4, 9, 9.5); torch(-137.6, 9, -9.5); torch(-137.6, 9, 9.5);

  // ----- the wall between the two wards, a gate in each half -----
  const XH = 30;
  box(-30, -18, -10, XH, -95, -68, { cy1: XH + MER }); box(-30, -18, -10, XH, -44, 44, { cy1: XH + MER }); box(-30, -18, -10, XH, 68, 95, { cy1: XH + MER });
  for (const c of [-56, 56]) {
    arch('x', -30, -18, -10, XH, c - 12, c + 12, c, 7, 8, { cy1: XH + MER });
    tower(-31, c - 14.5, 6, 40, 18, { rows: [38] }); tower(-31, c + 14.5, 6, 40, 18, { rows: [38] });
    torch(-30.4, 9, c - 8.6); torch(-30.4, 9, c + 8.6); torch(-17.6, 9, c - 8.6); torch(-17.6, 9, c + 8.6);
  }
  merlons(-29.3, -95, -29.3, 95, XH); merlons(-18.7, -95, -18.7, 95, XH);

  // ----- the great hall: a vaulted nave 16 m wide, deep windows on both sides -----
  arch('x', 8, 14, -3, 24, 40, 72, 56, 6.5, 8); arch('x', 102, 108, -3, 24, 40, 72, 56, 6.5, 8);
  for (let k = 0; k < 8; k++) {
    const x0 = 14 + 11 * k;
    arch('z', x0, x0 + 11, -3, 13, 40, 48, x0 + 5.5, 3.2, 9, { yb: 3.5 });
    arch('z', x0, x0 + 11, -3, 13, 64, 72, x0 + 5.5, 3.2, 9, { yb: 3.5 });
    if (k > 0) { torch(x0, 6, 48.4); torch(x0, 6, 63.6); }
  }
  arch('x', 14, 102, 13, 24, 40, 72, 56, 8, 13, { tone: 0.94 });
  gable('x', 6, 110, 24, 43, 38, 74);
  for (const x of [7.94, 108.06]) glow(x, 19, 56, x < 50 ? -1 : 1, 0, 4, 4);
  tower(8, 72, 5, 58, 22, { rows: [56, 30] }); tower(108, 72, 5, 58, 22, { rows: [56, 30] });
  for (let k = 1; k < 8; k++) box(14 + 11 * k - 1.3, 14 + 11 * k + 1.3, -3, 20, 72, 75.5, { tone: 0.96 });

  // ----- east gallery: along the curtain wall, an arcade towards the yard -----
  for (let k = 0; k < 8; k++) { const z0 = -60 + 10.5 * k; arch('x', 122, 126, -3, 10, z0, z0 + 10.5, z0 + 5.25, 3.6, 5.5); torch(139.6, 6, z0); }
  arch('z', 122, 140, 10, 18, -60, 24, 133, 7, 10, { tone: 0.94 });
  box(121, 140, 18, 18.7, -61, 25, { m: 'roof' });

  // ----- north wing: the same against the north wall -----
  for (let k = 0; k < 7; k++) { const x0 = 36 + 10 * k; arch('z', x0, x0 + 10, -3, 10, -81, -77, x0 + 5, 3.6, 5.5); torch(x0, 6, -94.6); }
  arch('x', 36, 106, 10, 18, -95, -77, -88, 7, 10, { tone: 0.94 });
  box(35, 107, 18, 18.7, -95, -76, { m: 'roof' });

  // ----- the keep, a passage through its foot -----
  arch('x', 41, 81, -10, 20, -24, 16, -4, 7, 8, { tone: 0.97 });
  box(41, 81, 20, 76, -24, 16, { tone: 0.97 });
  box(39, 83, 76, 82, -26, 18, { cy1: 82 + MER });
  merlons(39, -25.3, 83, -25.3, 82); merlons(39, 17.3, 83, 17.3, 82); merlons(39.7, -26, 39.7, 18, 82); merlons(82.3, -26, 82.3, 18, 82);
  tower(84, -27, 8, 112, 38, { rows: [110, 88, 62, 36] });                                // the stair tower: the highest point
  for (const x of [41, 81]) for (const z of [-24, 16]) {
    if (x === 81 && z === -24) continue;
    S.push({ t: 'cyl', x, z, r: 0.6, r1: 6, y0: 46, y1: 54, m: 'stone' });
    cyl(x, z, 6, 54, 90); cone(x, z, 7.4, 90, 108);
    for (let k = 0; k < 4; k++) { const a = (k + 0.5) * 1.5708 + (x < 60 ? 3.1416 : 0) * 0; glow(x + Math.cos(a) * 6.06, 84, z + Math.sin(a) * 6.06, Math.cos(a), Math.sin(a), 1.1, 2.4); }
  }
  box(51, 71, 82, 100, -14, 6);
  pyr(48, 74, 100, 130, -17, 9);
  flag(61, 129.5, -4, 14, 10);
  for (const y of [30, 41, 52, 63]) for (const u of [-11, 0, 11]) {
    glow(61 + u, y, -24.06, 0, -1); glow(61 + u, y, 16.06, 0, 1); glow(40.94, y, -4 + u, -1, 0); glow(81.06, y, -4 + u, 1, 0);
  }
  for (const u of [-5, 5]) { glow(61 + u, 92, -14.06, 0, -1, 1.3, 3); glow(61 + u, 92, 6.06, 0, 1, 1.3, 3); glow(50.94, 92, -4 + u, -1, 0, 1.3, 3); glow(71.06, 92, -4 + u, 1, 0, 1.3, 3); }
  torch(40.6, 9, -12.5); torch(40.6, 9, 4.5); torch(81.4, 9, -12.5); torch(81.4, 9, 4.5); torch(61, 6, -10.7); torch(61, 6, 2.7);

  // ----- the alley between the keep and the hall, under three flying arches -----
  for (const x0 of [48, 60, 72]) { arch('x', x0, x0 + 3, 0, 20, 16, 40, 28, 9, 8); torch(x0 + 1.5, 6, 19.4); torch(x0 + 1.5, 6, 36.6); }

  // ----- the watch tower -----
  tower(126, 86, 9, 98, 36, { rows: [96, 70, 44] });

  // ----- outer ward: chapel, barracks, a well; inner ward: a fountain where the course swings round -----
  box(-110, -70, -3, 16, -90, -66, { tone: 1.05 }); gable('x', -112, -68, 16, 31, -92, -64);
  box(-70, -60, -3, 46, -84, -72, { tone: 1.05 }); pyr(-72, -58, 46, 68, -86, -70);
  for (let k = 0; k < 4; k++) glow(-102 + 8 * k, 9, -65.94, 0, 1, 1.6, 6);
  box(-125, -45, -3, 11, 72, 93, { tone: 0.92 }); gable('x', -127, -43, 11, 21, 70, 95);
  for (let k = 0; k < 7; k++) glow(-118 + 11 * k, 6, 71.94, 0, -1, 1.4, 2.4);
  cyl(-95, -25, 2.6, 0, 1.3);
  cyl(36, -46, 5, 0, 1.0); cyl(36, -46, 1, 1, 5.2); S.push({ t: 'cyl', x: 36, z: -46, r: 1, r1: 2.6, y0: 5.2, y1: 6, m: 'stone' });

  // ----- paving (render only; the ground itself is the baked plateau) -----
  D.push({ t: 'pave', x0: -138, x1: 140, z0: -95, z1: 95 }, { t: 'pave', x0: -162, x1: -138, z0: -7, z1: 7 }, { t: 'pave', x0: -14, x1: 4, z0: 95, z1: 105 });
  return [S, D];
}

export function buildCastle() {
  const [S, D] = plan();
  for (const s of S) {
    const r = s.t === 'cyl' ? Math.max(s.r, s.r1) : 0;
    s.bx0 = s.t === 'cyl' ? s.x - r : s.x0; s.bx1 = s.t === 'cyl' ? s.x + r : s.x1;
    s.bz0 = s.t === 'cyl' ? s.z - r : s.z0; s.bz1 = s.t === 'cyl' ? s.z + r : s.z1;
    s.by1 = s.cy1 !== undefined ? s.cy1 : s.y1;
  }
  C.solids = S; C.deco = D; C.y0 = castleY(); C.ready = true;
}

// is the point (castle coordinates) inside solid s, grown by pad?
function inside(s, x, y, z, pad) {
  if (x < s.bx0 - pad || x > s.bx1 + pad || z < s.bz0 - pad || z > s.bz1 + pad || y < s.y0 - pad || y > s.by1 + pad) return false;
  switch (s.t) {
    case 'box': return true;
    case 'cyl': { const t = Math.min(1, Math.max(0, (y - s.y0) / (s.y1 - s.y0))); return Math.hypot(x - s.x, z - s.z) < s.r + (s.r1 - s.r) * t + pad; }
    case 'arch': {
      const da = Math.abs((s.axis === 'x' ? z : x) - s.c), hw = s.hw - pad;
      if (da >= hw) return true;
      if (s.yb !== undefined && y < s.yb + pad) return true;
      return y > s.ys && Math.hypot(da, y - s.ys) >= hw;
    }
    case 'gable': { const t = Math.max(0, (y - s.y0) / (s.y1 - s.y0)), a = s.axis === 'x' ? z : x, c = s.axis === 'x' ? (s.z0 + s.z1) / 2 : (s.x0 + s.x1) / 2, h = s.axis === 'x' ? (s.z1 - s.z0) / 2 : (s.x1 - s.x0) / 2; return Math.abs(a - c) < h * (1 - t) + pad; }
    case 'pyr': { const t = Math.max(0, (y - s.y0) / (s.y1 - s.y0)); return Math.abs(x - (s.x0 + s.x1) / 2) < (s.x1 - s.x0) / 2 * (1 - t) + pad && Math.abs(z - (s.z0 + s.z1) / 2) < (s.z1 - s.z0) / 2 * (1 - t) + pad; }
  }
  return false;
}

// does the world point touch the fortress?
export function hitCastle(p, pad = 0.35) {
  if (!C.ready) return false;
  const x = p[0] - SITE.x, z = p[2] - SITE.z, y = p[1] - C.y0;
  if (x < -224 || x > 176 || z < -132 || z > 132 || y > 134 || y < -36) return false;
  for (const s of C.solids) if (inside(s, x, y, z, pad)) return true;
  return false;
}
export const nearCastle = (p, margin = 60) => Math.abs(p[0] - SITE.x + 24) < 200 + margin && Math.abs(p[2] - SITE.z) < 132 + margin;

// ----- the course through it: [x, z, y, ring radius or 0 for a navigation node, speed the demo rider keeps] -----
const COURSE = [
  [-150, 0, 7.5, 5, 24],                                             // 1 the gate
  [-120, 3, 7.5, 0], [-96, 18, 7.5, 0], [-72, 38, 7.5, 0], [-50, 52, 7.5, 0],
  [-24, 56, 7.5, 5, 22],                                             // 2 the inner gate
  [-4, 56, 7.5, 0],
  [35, 56, 8, 5, 22], [80, 56, 8, 5, 20],                            // 3, 4 the great hall
  [108, 56, 8, 0], [120.5, 52.7, 8, 0], [129.7, 43.5, 7.5, 0], [133, 31, 7.5, 0],
  [133, 8, 7.5, 5, 18], [133, -40, 7.5, 5, 18],                      // 5, 6 the east gallery
  [133, -60, 7.5, 0], [129.2, -74, 7.5, 0], [119, -84.2, 7.5, 0], [106, -88, 7.5, 0],
  [92, -88, 7.5, 5, 18], [50, -88, 7.5, 5, 20],                      // 7, 8 the north wing
  [36, -88, 7.5, 0], [15, -82.4, 8.5, 0], [-0.4, -67, 9.5, 0],
  [-6, -46, 9.5, 7, 20],                                              // 9 round the fountain
  [-0.4, -25, 9.5, 0], [15, -9.6, 8.5, 0], [36, -4, 7.5, 0],
  [61, -4, 7.5, 5, 16],                                              // 10 through the keep
  [81, -4, 7.5, 0], [95.3, 0.7, 8, 0], [100, 12, 8, 0], [95.3, 23.3, 8, 0], [84, 28, 7.5, 0],
  [61.5, 28, 7.5, 5, 14],                                            // 11 the alley
  [40, 28, 7.5, 0], [25, 28, 7.5, 0], [10, 28, 7.5, 0], [-0.6, 32.4, 7.5, 0], [-5, 43, 7.5, 0], [-5, 70, 7.5, 0],
  [-5, 100, 7, 5, 18],                                               // 12 the postern
  [-5, 114, 6, 0], [-16.7, 142.3, 2, 0], [-45, 154, -2, 0],
  [-100, 154, -6, 8, 32],                                            // 13 along the cliff
  [-150, 154, -8, 0], [-179.7, 141.7, -10, 0], [-192, 112, -11, 0], [-192, 60, -11, 0],
  [-192, 0, -11, 6, 30],                                             // 14 under the bridge: the finish
];
export function castleCourse() {
  const y0 = castleY(); let v = 24;
  return COURSE.map(([x, z, y, r, vr], i) => { if (vr) v = vr; return { x: SITE.x + x, z: SITE.z + z, alt: y0 + y, r: r || 30, hint: !r, finish: i === COURSE.length - 1, v: vr || v, slow: true }; });
}
