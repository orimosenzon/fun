// The fortress on the ridge end above the main valley. One list of solids is both what is drawn
// (js/render/castle.js) and what the bike can hit, so every wall, vault and pillar is really there.
// Castle coordinates: metres from SITE, x east (the gate looks west along the ridge), z south
// (towards the valley), y up from the courtyard floor.
import { SITE } from './castleSite.js';
import { groundAt } from './terrain.js';

export const C = { solids: [], deco: [], rooms: [], y0: 0, ready: false };
// the roofed spaces, as boxes [x0, x1, y0, y1, z0, z1]: daylight fades inside them and the torches take over
const ROOMS = [[8, 108, -3, 24, 40, 72], [122, 141.5, -3, 18, -60, 24], [36, 106, -3, 18, -96.5, -77], [41, 81, -3, 20, -12.4, 4.4], [-110, -70, -3, 20.5, -90, -66],
  [-162, -138, -3, 15.6, -8.4, 8.4], [-30, -18, -3, 15.6, -64.4, -47.6], [-30, -18, -3, 15.6, 47.6, 64.4]];
export const castleY = () => groundAt(SITE.x, SITE.z);

const WH = 24;           // curtain wall
const MER = 2.2;         // battlements on top of it
function rng(seed) { let s = seed >>> 0; return () => ((s = (s * 1664525 + 1013904223) >>> 0) / 4294967296); }

// Solids carry a material: stone, plaster (old lime render flaking off the stone), fach (half-timbering),
// wood, roof (slate), or hide (a collider with no face of its own: the furnishings are drawn as ornaments).
// w is how exposed to the weather a piece is: ivy and moss only grow where it is 1.
function plan() {
  const S = [], D = [], R = rng(1337), NM = [];
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
  const glow = (x, y, z, nx, nz, w = 1.5, h = 3.2, sh = 0) => D.push({ t: 'glow', x, y, z, nx, nz, w, h, sh });
  const torch = (x, y, z) => D.push({ t: 'torch', x, y, z });
  const flag = (x, y, z, h = 9, s = 7) => D.push({ t: 'flag', x, y, z, h, s });
  const hide = (x, z, r, y0, y1) => S.push({ t: 'cyl', x, z, r, r1: r, y0, y1, m: 'hide' });
  const hbox = (x0, x1, y0, y1, z0, z1) => S.push({ t: 'box', x0, x1, y0, y1, z0, z1, m: 'hide' });
  // ----- furnishings: drawn by js/render/castleProps.js, with a simple collider where they stand in the way -----
  // a figure looks along (sin yaw, cos yaw); things on a wall take its outward normal (nx, nz)
  const statue = (x, z, yaw, s, kind, o = {}) => { const y = o.y || 0, ped = o.ped !== undefined ? o.ped : s * 0.3; D.push({ t: 'statue', x, y, z, yaw, s, kind, ped }); hide(x, z, s * 0.17, y, y + ped + s); };
  const armor = (x, z, yaw, pose = 0, s = 2.7) => { D.push({ t: 'armor', x, y: 0, z, yaw, s, pose }); hide(x, z, s * 0.2, 0, s + 0.6); };
  const shield = (x, y, z, nx, nz, s, k, arms = 'swords') => D.push({ t: 'shield', x, y, z, nx, nz, s, k, arms });
  const banner = (x, y, z, nx, nz, w, h, k, free = 0) => D.push({ t: 'banner', x, y, z, nx, nz, w, h, k, free });       // y is the top edge
  const tapestry = (x, y, z, nx, nz, w, h, k) => D.push({ t: 'tapestry', x, y, z, nx, nz, w, h, k });
  const carpet = (x0, x1, z0, z1, k, ax) => D.push({ t: 'carpet', x0, x1, z0, z1, k, ax });
  const rib = (axis, w, c, hw, ys) => D.push({ t: 'rib', axis, w, c, hw, ys });
  const chandelier = (x, y, z, r, top) => { D.push({ t: 'chandelier', x, y, z, r, top }); hide(x, z, r + 0.15, y - 0.4, y + 0.6); };
  const brazier = (x, z) => { D.push({ t: 'brazier', x, y: 0, z }); hide(x, z, 0.75, 0, 2.2); };
  const barrel = (x, z, s = 1) => { D.push({ t: 'barrel', x, y: 0, z, s }); hide(x, z, 0.5 * s, 0, 1.25 * s); };
  const crate = (x, z, s = 1.2, yaw = 0) => { D.push({ t: 'crate', x, y: 0, z, s, yaw }); hbox(x - s * 0.6, x + s * 0.6, 0, s, z - s * 0.6, z + s * 0.6); };
  const chest = (x, z, yaw) => { D.push({ t: 'chest', x, y: 0, z, yaw }); hbox(x - 1, x + 1, 0, 1.1, z - 1, z + 1); };
  const rack = (x, z, nx, nz, w) => { D.push({ t: 'rack', x, y: 0, z, nx, nz, w }); hbox(x - (nz ? w / 2 : 0) - (nx < 0 ? 0.9 : 0), x + (nz ? w / 2 : 0) + (nx > 0 ? 0.9 : 0), 0, 3.6, z - (nx ? w / 2 : 0) - (nz < 0 ? 0.9 : 0), z + (nx ? w / 2 : 0) + (nz > 0 ? 0.9 : 0)); };
  const glass = (x, y, z, nx, nz, w, h, k) => D.push({ t: 'glass', x, y, z, nx, nz, w, h, k });
  // battlements along a straight edge (render only: the wall below is solid up to their top).
  // Centuries have taken some of them, and broken others
  const merlons = (x0, z0, x1, z1, y, th = 1.4) => {
    const L = Math.hypot(x1 - x0, z1 - z0), n = Math.max(1, Math.round(L / 5.2)), ux = (x1 - x0) / L, uz = (z1 - z0) / L;
    for (let k = 0; k < n; k++) {
      const s = (k + 0.5) * L / n, cx = x0 + ux * s, cz = z0 + uz * s, hx = Math.abs(ux) * 1.5 + Math.abs(uz) * th / 2, hz = Math.abs(uz) * 1.5 + Math.abs(ux) * th / 2, u = R();
      if (NM.some((q) => cx > q[0] && cx < q[1] && cz > q[2] && cz < q[3]) || u < 0.07) continue;
      dbox(cx - hx, cx + hx, y, y + (u < 0.19 ? 0.6 + R() * 1.0 : MER), cz - hz, cz + hz);
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
  // a roofed timber gallery on top of a stretch of wall (the Wehrgang of a German castle); call it before the wall
  const walk = (x0, x1, z0, z1, y) => {
    const ax = x1 - x0 > z1 - z0, L = ax ? x1 - x0 : z1 - z0, n = Math.round(L / 5), W = { m: 'wood' };
    NM.push([x0 - 1, x1 + 1, z0 - 1, z1 + 1]);
    for (let k = 0; k <= n; k++) {
      const s = k * L / n;
      if (ax) { dbox(x0 + s - 0.22, x0 + s + 0.22, y, y + 3.3, z0 + 0.1, z0 + 0.54, W); dbox(x0 + s - 0.22, x0 + s + 0.22, y, y + 3.3, z1 - 0.54, z1 - 0.1, W); dbox(x0 + s - 0.15, x0 + s + 0.15, y + 2.95, y + 3.25, z0 + 0.1, z1 - 0.1, W); }
      else { dbox(x0 + 0.1, x0 + 0.54, y, y + 3.3, z0 + s - 0.22, z0 + s + 0.22, W); dbox(x1 - 0.54, x1 - 0.1, y, y + 3.3, z0 + s - 0.22, z0 + s + 0.22, W); dbox(x0 + 0.1, x1 - 0.1, y + 2.95, y + 3.25, z0 + s - 0.15, z0 + s + 0.15, W); }
    }
    if (ax) { dbox(x0, x1, y, y + 1.3, z0 + 0.2, z0 + 0.42, W); dbox(x0, x1, y, y + 1.3, z1 - 0.42, z1 - 0.2, W); gable('x', x0 - 0.6, x1 + 0.6, y + 3.3, y + 6.6, z0 - 1.1, z1 + 1.1); }
    else { dbox(x0 + 0.2, x0 + 0.42, y, y + 1.3, z0, z1, W); dbox(x1 - 0.42, x1 - 0.2, y, y + 1.3, z0, z1, W); gable('z', x0 - 1.1, x1 + 1.1, y + 3.3, y + 6.6, z0 - 0.6, z1 + 0.6); }
  };
  // a bay turret on a corbel, hung on a wall whose outward normal is (nx, nz)
  const oriel = (x, z, nx, nz, y, r = 3.2) => {
    const cx = x + nx * r * 0.35, cz = z + nz * r * 0.35;
    S.push({ t: 'cyl', x: cx, z: cz, r: 0.5, r1: r, y0: y - 5, y1: y, m: 'stone' });
    cyl(cx, cz, r, y, y + 7.5, { m: 'plaster' }); cone(cx, cz, r + 0.8, y + 7.5, y + 15);
    for (const da of [-0.9, 0, 0.9]) { const a = Math.atan2(nz, nx) + da; glow(cx + Math.cos(a) * (r + 0.06), y + 4, cz + Math.sin(a) * (r + 0.06), Math.cos(a), Math.sin(a), 1.0, 2.4); }
  };

  // ----- curtain wall -----
  walk(-68, 60, -105, -95, WH); walk(140, 150, -27, 27, WH);
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

  // ----- gatehouse and bridge: a half-timbered storey over the gate, a murder hole, the portcullis drawn up -----
  arch('x', -162, -138, -10, 28.8, -22, 22, 0, 7, 8, { tone: 1.03 });
  box(-164.5, -135.5, 28.8, 38.4, -23, 23, { m: 'fach' });
  gable('z', -166, -134, 38.4, 55, -24.5, 24.5);
  tower(-162, -25, 11, 46, 30); tower(-162, 25, 11, 46, 30);
  for (const z of [-13, 13]) glow(-162.06, 22.5, z, -1, 0, 1.6, 3.6);
  box(-165.6, -162, 16.6, 21.4, -2.6, 2.6, { tone: 1.05 });
  dbox(-165.6, -162, 15.2, 16.6, -2.6, -1.8); dbox(-165.6, -162, 15.2, 16.6, 1.8, 2.6);
  D.push({ t: 'gable', axis: 'z', x0: -166.1, x1: -161.9, y0: 21.4, y1: 23.6, z0: -3.1, z1: 3.1, m: 'roof' });
  shield(-165.6, 19, 0, -1, 0, 2.7, 0, 'none');
  banner(-162, 24, -9.5, -1, 0, 2.6, 9.5, 2); banner(-162, 24, 9.5, -1, 0, 2.6, 9.5, 1);
  D.push({ t: 'grate', axis: 'x', w: -159.5, c: 0, hw: 7, ys: 8, yb: 12.7 });
  dbox(-161.4, -154.6, 0.1, 9.3, 6.6, 6.9, { m: 'wood' }); dbox(-161.4, -154.6, 0.1, 9.3, -6.9, -6.6, { m: 'wood' });
  arch('z', -218, -162, -34, 0, -8, 8, -192, 13, -16);
  box(-218, -162, 0, 1.3, -8, -7.2); box(-218, -162, 0, 1.3, 7.2, 8);
  torch(-162.4, 9, -9.5); torch(-162.4, 9, 9.5); torch(-137.6, 9, -9.5); torch(-137.6, 9, 9.5);
  shield(-138, 19.5, 0, 1, 0, 3, 3, 'halberds');
  brazier(-133, -11); brazier(-133, 11);

  // ----- the wall between the two wards, a gate in each half -----
  const XH = 30;
  walk(-30, -18, -34, 34, XH);
  box(-30, -18, -10, XH, -95, -68, { cy1: XH + MER }); box(-30, -18, -10, XH, -44, 44, { cy1: XH + MER }); box(-30, -18, -10, XH, 68, 95, { cy1: XH + MER });
  for (const c of [-56, 56]) {
    arch('x', -30, -18, -10, XH, c - 12, c + 12, c, 7, 8, { cy1: XH + MER });
    tower(-31, c - 14.5, 6, 40, 18, { rows: [38] }); tower(-31, c + 14.5, 6, 40, 18, { rows: [38] });
    torch(-30.4, 9, c - 8.6); torch(-30.4, 9, c + 8.6); torch(-17.6, 9, c - 8.6); torch(-17.6, 9, c + 8.6);
    D.push({ t: 'grate', axis: 'x', w: -27.5, c, hw: 7, ys: 8, yb: 12.7 });
    shield(-30, 21, c, -1, 0, 3, c < 0 ? 5 : 6, 'swords'); shield(-18, 21, c, 1, 0, 3, c < 0 ? 7 : 4, 'halberds');
  }
  merlons(-29.3, -95, -29.3, 95, XH); merlons(-18.7, -95, -18.7, 95, XH);

  // ----- the great hall (the Rittersaal): a vaulted nave 16 m wide, deep windows on both sides -----
  const IN = { w: 0.2 };
  arch('x', 8, 14, -3, 24, 40, 72, 56, 6.5, 8, IN); arch('x', 102, 108, -3, 24, 40, 72, 56, 6.5, 8, IN);
  for (let k = 0; k < 8; k++) {
    const x0 = 14 + 11 * k;
    arch('z', x0, x0 + 11, -3, 13, 40, 48, x0 + 5.5, 3.2, 9, { yb: 3.5, ...IN });
    arch('z', x0, x0 + 11, -3, 13, 64, 72, x0 + 5.5, 3.2, 9, { yb: 3.5, ...IN });
  }
  arch('x', 14, 102, 13, 24, 40, 72, 56, 8, 13, { tone: 0.94, ...IN });
  gable('x', 9.4, 106.6, 24, 43, 38, 74);
  for (let i = 0; i < 6; i++) for (const [xa, xb] of [[8, 9.4], [106.6, 108]]) box(xa, xb, 24 + 3.5 * i, 27.5 + 3.5 * i, 38 + 3 * i, 74 - 3 * i, { tone: 1.02 });   // stepped gables
  for (const x of [25, 47, 69, 91]) for (const sg of [-1, 1]) {                             // dormers
    box(x - 1.7, x + 1.7, 27.5, 33, 56 + sg * (sg > 0 ? 10 : 14.5), 56 + sg * (sg > 0 ? 14.5 : 10), { m: 'plaster' });
    gable('z', x - 2.3, x + 2.3, 33, 35.8, sg > 0 ? 62.5 : 40.8, sg > 0 ? 71.2 : 49.5);
    glow(x, 30.6, 56 + sg * 14.56, 0, sg, 1.2, 2.2, 1);
  }
  for (const x of [38, 80]) { box(x - 1.1, x + 1.1, 34, 47.5, 50, 52.2); dbox(x - 1.4, x + 1.4, 47.5, 48.1, 49.7, 52.5); }   // chimneys
  for (const x of [7.94, 108.06]) glow(x, 19, 56, x < 50 ? -1 : 1, 0, 4, 4);
  tower(8, 72, 5, 58, 22, { rows: [56, 30] }); tower(108, 72, 5, 58, 22, { rows: [56, 30] });
  for (let k = 1; k < 8; k++) box(14 + 11 * k - 1.3, 14 + 11 * k + 1.3, -3, 20, 72, 75.5, { tone: 0.96 });
  // inside: a runner down the nave, knights' armour and old kings on the piers, arms above them,
  // banners on iron rods under the vault, wheel chandeliers
  carpet(9, 107, 54, 58, 0, 'x');
  for (let k = 1; k < 8; k++) {
    const x = 14 + 11 * k;
    rib('x', x, 56, 8, 13);
    dbox(x - 0.06, x + 0.06, 17, 17.12, 49.2, 62.8, { m: 'iron' });
    banner(x, 16.95, 51.7, 1, 0, 2.3, 4, k, 1); banner(x, 16.95, 60.3, 1, 0, 2.3, 4, k + 3, 1);
    for (const sg of [-1, 1]) {
      const zf = 56 + sg * 8;
      dbox(x - 0.6, x + 0.6, 0, 13, Math.min(zf, zf - sg * 0.3), Math.max(zf, zf - sg * 0.3), { tone: 1.1, w: 0 });
      torch(x, 8, zf - sg * 0.4);
      if (k % 2) armor(x, zf - sg * 1.0, sg > 0 ? Math.PI : 0, (k + (sg > 0 ? 1 : 0)) % 2); else statue(x, zf - sg * 1.05, sg > 0 ? Math.PI : 0, 4.2, ['king', 'knight', 'bishop'][(k / 2 + (sg > 0 ? 1 : 0)) % 3]);
      shield(x, 10.9, zf, 0, -sg, 1.7, k + (sg > 0 ? 4 : 0), k % 2 ? 'swords' : 'halberds');
    }
  }
  for (const x of [30.5, 52.5, 74.5, 96.5]) chandelier(x, 16.4, 56, 2.3, 21);
  banner(8, 21, 46.5, -1, 0, 2.6, 9, 0); banner(8, 21, 65.5, -1, 0, 2.6, 9, 3);
  brazier(4.5, 46.5); brazier(4.5, 65.5); brazier(111.5, 45.5); brazier(111.5, 66.5);

  // ----- east gallery (the gallery of ancestors): along the curtain wall, an arcade towards the yard -----
  for (let k = 0; k < 8; k++) {
    const z0 = -60 + 10.5 * k, zc = z0 + 5.25;
    arch('x', 122, 126, -3, 10, z0, z0 + 10.5, zc, 3.6, 5.5, IN); torch(139.6, 6.5, z0);
    if (k) rib('z', z0, 133, 7, 10);
    if (k % 2 === 0) { tapestry(140, 9.6, zc, -1, 0, 6, 5.2, k / 2); chest(139.0, zc, -Math.PI / 2); }
    else { shield(140, 8.2, zc, -1, 0, 1.8, k + 1, k % 4 === 1 ? 'halberds' : 'swords'); armor(138.9, zc, -Math.PI / 2, (k >> 1) % 2); }
    if (k && k % 2 === 0) statue(127.0, z0, Math.PI / 2, 3.4, ['bishop', 'king', 'knight'][(k / 2) % 3]);
  }
  arch('z', 122, 140, 10, 18, -60, 24, 133, 7, 10, { tone: 0.94, ...IN });
  box(121, 140, 18, 18.7, -61, 25, { m: 'roof' });
  carpet(131, 135, -60, 24, 0, 'z');

  // ----- north wing (the armoury): the same against the north wall -----
  for (let k = 0; k < 7; k++) {
    const x0 = 36 + 10 * k, xc = x0 + 5;
    arch('z', x0, x0 + 10, -3, 10, -81, -77, xc, 3.6, 5.5, IN); torch(x0, 6.5, -94.6);
    if (k) rib('x', x0, -88, 7, 10);
    if (k % 2 === 0) { rack(xc, -95, 0, 1, 6.4); shield(xc, 8.3, -95, 0, 1, 1.6, k + 2, 'swords'); }
    else { armor(xc - 1.8, -93.9, 0, 0); armor(xc + 1.8, -93.9, 0, 1); shield(xc, 8.2, -95, 0, 1, 1.9, k, 'halberds'); }
    if (k && k % 2) { barrel(x0 - 0.62, -82.05); barrel(x0 + 0.62, -82.0); } else if (k) { crate(x0 - 0.45, -82.1, 1.25); crate(x0 + 0.95, -81.95, 0.85); }
  }
  arch('x', 36, 106, 10, 18, -95, -77, -88, 7, 10, { tone: 0.94, ...IN });
  box(35, 107, 18, 18.7, -95, -76, { m: 'roof' });

  // ----- the keep (the Bergfried), a passage through its foot lined with the kings of old -----
  arch('x', 41, 81, -10, 20, -24, 16, -4, 7, 8, { tone: 0.97, w: 0.6 });
  box(41, 81, 20, 76, -24, 16, { m: 'plaster' });
  box(39, 83, 76, 82, -26, 18, { cy1: 82 + MER });
  for (let u = -19; u <= 19; u += 3.8) { dbox(61 + u - 0.5, 61 + u + 0.5, 73.6, 76, -26, -24); dbox(61 + u - 0.5, 61 + u + 0.5, 73.6, 76, 16, 18); dbox(39, 41, 73.6, 76, -4 + u - 0.5, -4 + u + 0.5); dbox(81, 83, 73.6, 76, -4 + u - 0.5, -4 + u + 0.5); }
  merlons(39, -25.3, 83, -25.3, 82); merlons(39, 17.3, 83, 17.3, 82); merlons(39.7, -26, 39.7, 18, 82); merlons(82.3, -26, 82.3, 18, 82);
  tower(84, -27, 8, 112, 38, { rows: [110, 88, 62, 36] });                                // the stair tower: the highest point
  for (const x of [41, 81]) for (const z of [-24, 16]) {
    if (x === 81 && z === -24) continue;
    S.push({ t: 'cyl', x, z, r: 0.6, r1: 6, y0: 46, y1: 54, m: 'stone' });
    cyl(x, z, 6, 54, 90); cone(x, z, 7.4, 90, 108);
    for (let k = 0; k < 4; k++) { const a = (k + 0.5) * 1.5708; glow(x + Math.cos(a) * 6.06, 84, z + Math.sin(a) * 6.06, Math.cos(a), Math.sin(a), 1.1, 2.4); }
  }
  box(51, 71, 82, 100, -14, 6, { m: 'fach' });
  pyr(48, 74, 100, 130, -17, 9);
  flag(61, 129.5, -4, 14, 10);
  for (const y of [30, 41, 52, 63]) for (const u of [-11, 0, 11]) {
    glow(61 + u, y, -24.06, 0, -1, 1.5, 3.2, 1); glow(61 + u, y, 16.06, 0, 1, 1.5, 3.2, 1); glow(40.94, y, -4 + u, -1, 0, 1.5, 3.2, 1); glow(81.06, y, -4 + u, 1, 0, 1.5, 3.2, 1);
  }
  oriel(41, -4, -1, 0, 44); oriel(61, 16, 0, 1, 55); oriel(61, -24, 0, -1, 44);
  torch(40.6, 9, -12.5); torch(40.6, 9, 4.5); torch(81.4, 9, -12.5); torch(81.4, 9, 4.5);
  for (const x of [50.2, 61, 71.8]) { torch(x, 6.5, -10.7); torch(x, 6.5, 2.7); }
  ['king', 'knight', 'bishop', 'king'].forEach((kind, i) => { const x = [46, 55.6, 66.4, 76][i]; statue(x, -10.3, 0, 3.6, kind); statue(x, 2.3, Math.PI, 3.6, ['knight', 'king', 'king', 'bishop'][i]); });
  rib('x', 51, -4, 7, 8); rib('x', 71, -4, 7, 8);
  carpet(41, 81, -6, -2, 0, 'x');
  banner(41, 28, -9.5, -1, 0, 2.6, 10, 0); banner(41, 28, 1.5, -1, 0, 2.6, 10, 5); banner(81, 28, -9.5, 1, 0, 2.6, 10, 2); banner(81, 28, 1.5, 1, 0, 2.6, 10, 6);
  brazier(36.5, -14.5); brazier(36.5, 6.5);

  // ----- the alley between the keep and the hall, under three flying arches -----
  for (const x0 of [48, 60, 72]) { arch('x', x0, x0 + 3, 0, 20, 16, 40, 28, 9, 8, { w: 0.6 }); torch(x0 + 1.5, 6, 19.4); torch(x0 + 1.5, 6, 36.6); }
  barrel(54.6, 17.0); barrel(55.9, 17.1); barrel(55.2, 18.1, 0.8); crate(66.6, 17.2, 1.5); crate(68.4, 17.0, 1.0); barrel(54.5, 39.0); crate(66.8, 38.9, 1.3); crate(43.5, 39.0, 1.6); barrel(45.2, 39.1);

  // ----- the watch tower -----
  tower(126, 86, 9, 98, 36, { rows: [96, 70, 44] });

  // ----- outer ward -----
  // the chapel: one can fly in through the south door. Coloured glass, an altar, pews
  {
    const P = { m: 'plaster' }, K = { w: 0 };
    box(-110, -70, -3, 9, -90, -89.25, P); box(-109.25, -70.75, -3, 9, -89.25, -88.5, K);
    arch('z', -110, -70, -3, 9, -66.75, -66, -98, 4, 5, P); arch('z', -109.25, -70.75, -3, 9, -67.5, -66.75, -98, 4, 5, K);
    box(-110, -109.25, -3, 9, -89.25, -66.75, P); box(-109.25, -108.5, -3, 9, -88.5, -67.5, K);
    box(-70.75, -70, -3, 9, -89.25, -66.75, P); box(-71.5, -70.75, -3, 9, -88.5, -67.5, K);
    arch('x', -108.5, -71.5, 9, 20.5, -90, -66, -78, 10.4, 9, { m: 'plaster', w: 1 });
    box(-110, -109.25, 9, 20.5, -90, -66, P); box(-109.25, -108.5, 9, 20.5, -90, -66, K);
    box(-70.75, -70, 9, 20.5, -90, -66, P); box(-71.5, -70.75, 9, 20.5, -90, -66, K);
    gable('x', -112, -68, 20.5, 36, -92, -64);
    box(-70, -60, -3, 46, -84, -72, P); pyr(-72, -58, 46, 68, -86, -70);
    for (const a of [0, 1, 2]) glow([-65, -65, -59.94, -65][a], 40, [-84.06, -71.94, -78, -78][a], [0, 0, 1, 0][a], [-1, 1, 0, 0][a], 1.4, 3.4);
    [-103, -95, -87, -79].forEach((x, i) => { glass(x, 5.6, -88.5, 0, 1, 2.6, 6, i); glass(x, 5.6, -90, 0, -1, 2.6, 6, i); });
    [-87, -79].forEach((x, i) => { glass(x, 5.6, -67.5, 0, -1, 2.6, 6, i + 2); glass(x, 5.6, -66, 0, 1, 2.6, 6, i + 2); });
    glass(-71.5, 8.2, -78, -1, 0, 4.2, 9, 1); glass(-108.5, 12.5, -78, 1, 0, 3.4, 6.4, 2); glass(-110, 12.5, -78, -1, 0, 3.4, 6.4, 2);
    box(-76.2, -73.6, 0, 1.5, -80.6, -75.4, { tone: 1.15, w: 0 });                              // the altar
    dbox(-75.05, -74.75, 1.5, 4.1, -78.15, -77.85, { m: 'gold' }); dbox(-75.05, -74.75, 3.1, 3.4, -78.9, -77.1, { m: 'gold' });
    D.push({ t: 'candle', x: -74.9, y: 1.5, z: -80.0 }, { t: 'candle', x: -74.9, y: 1.5, z: -76.0 });
    statue(-73.6, -85.6, -Math.PI / 2, 3.2, 'bishop'); statue(-73.6, -70.4, -Math.PI / 2, 3.2, 'king');
    for (let x = -104; x <= -82; x += 2.75) {
      dbox(x, x + 0.9, 0.55, 0.7, -87.6, -81.6, { m: 'wood' }); dbox(x + 0.9, x + 1.05, 0, 1.35, -87.6, -81.6, { m: 'wood' }); dbox(x, x + 0.9, 0, 0.55, -87.6, -87.4, { m: 'wood' }); dbox(x, x + 0.9, 0, 0.55, -81.8, -81.6, { m: 'wood' });
      if (x > -91) { dbox(x, x + 0.9, 0.55, 0.7, -74.4, -68.4, { m: 'wood' }); dbox(x + 0.9, x + 1.05, 0, 1.35, -74.4, -68.4, { m: 'wood' }); dbox(x, x + 0.9, 0, 0.55, -74.4, -74.2, { m: 'wood' }); dbox(x, x + 0.9, 0, 0.55, -68.6, -68.4, { m: 'wood' }); }
    }
    hbox(-104.2, -80.6, 0, 1.4, -87.7, -81.5); hbox(-90.4, -80.6, 0, 1.4, -74.5, -68.3);
    carpet(-106, -77, -80, -76, 1, 'x');
    chandelier(-90, 14.6, -78, 2.0, 19.3);
    for (const x of [-99, -91, -83]) torch(x, 5, -88.1);
    torch(-83, 5, -67.9); torch(-75, 5, -67.9); torch(-104.5, 5, -67.9);
    brazier(-104, -63.5); brazier(-92, -63.5);
  }
  // a half-timbered house against the north wall, and the barracks along the south one
  box(-57, -39, -3, 6.4, -95, -78, { tone: 0.95 }); box(-57.7, -38.3, 6.4, 12.8, -95, -77.3, { m: 'fach' }); gable('x', -58.6, -37.4, 12.8, 23.5, -95.5, -76.2);
  box(-45, -43.2, 16, 26, -90, -88.2); glow(-52.5, 3.2, -77.94, 0, 1, 1.4, 2.6, 1); glow(-43.5, 3.2, -77.94, 0, 1, 1.4, 2.6, 1); dbox(-49.3, -46.7, 0, 4.2, -78.0, -77.82, { m: 'wood' });
  box(-125, -45, -3, 6.4, 72, 93, { tone: 0.92 }); box(-125.7, -44.3, 6.4, 12.8, 71.3, 93.5, { m: 'fach' }); gable('x', -127, -43, 12.8, 24, 70.2, 95);
  for (let k = 0; k < 7; k++) { if (k === 3) dbox(-86.4, -83.6, 0, 4.4, 71.82, 72, { m: 'wood' }); else glow(-118 + 11 * k, 3.4, 71.94, 0, -1, 1.4, 2.4, 1); }
  for (const x of [-110, -60]) { box(x - 1, x + 1, 16, 27, 80, 82); }
  // the well under its little roof, and the old linden beside it
  cyl(-95, -25, 2.6, 0, 1.3);
  dbox(-97.75, -97.45, 0, 4.2, -25.15, -24.85, { m: 'wood' }); dbox(-92.55, -92.25, 0, 4.2, -25.15, -24.85, { m: 'wood' }); dbox(-97.6, -92.4, 3.3, 3.5, -25.1, -24.9, { m: 'wood' });
  D.push({ t: 'gable', axis: 'x', x0: -98.6, x1: -91.4, y0: 4.2, y1: 6.3, z0: -27.2, z1: -22.8, m: 'roof' });
  hbox(-98.6, -91.4, 3.3, 6.3, -27.2, -22.8);
  D.push({ t: 'tree', x: -110, y: 0, z: -20, h: 17 }); hide(-110, -20, 0.9, 0, 7); hide(-110, -20, 6.2, 7, 17);
  // inner ward: a fountain where the course swings round, a stone knight on it
  cyl(36, -46, 5, 0, 1.0); cyl(36, -46, 1, 1, 5.2); S.push({ t: 'cyl', x: 36, z: -46, r: 1, r1: 2.6, y0: 5.2, y1: 6, m: 'stone' });
  statue(36, -46, -Math.PI / 2, 3.4, 'knight', { y: 6, ped: 0.5 });
  brazier(-12, 88); brazier(2, 88);

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
  // which way does each wall torch look? Away from the masonry it hangs on
  const wallAt = (x, y, z) => S.some((q) => q.m !== 'hide' && inside(q, x, y, z, 0));
  for (const d of D) if (d.t === 'torch') {
    d.nx = 0; d.nz = 0;
    for (const [ax, az] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) if (wallAt(d.x + ax * 0.7, d.y, d.z + az * 0.7)) { d.nx = -ax; d.nz = -az; break; }
  }
  C.solids = S; C.deco = D; C.rooms = ROOMS; C.y0 = castleY(); C.ready = true;
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
