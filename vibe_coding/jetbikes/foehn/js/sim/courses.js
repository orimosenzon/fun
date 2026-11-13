// The events. Check-points are rings to fly through or pads to land on; everything is placed
// relative to the baked terrain when an event is built.
import { groundAt, surfaceAt, T } from './terrain.js';
import { clamp } from './vmath.js';
import { castleCourse } from './castle.js';

// helper: point on a valley centre line at a given x (main valley) or index fraction, shifted sideways
function on(valley, t, off = 0) {
  const pts = T.meta.valleys[valley];
  const f = clamp(t, 0, 1) * (pts.length - 1), i = Math.min(pts.length - 2, Math.floor(f)), u = f - i;
  const a = pts[i], b = pts[i + 1];
  const tx = b[0] - a[0], tz = b[1] - a[1], l = Math.hypot(tx, tz) || 1;
  return [a[0] + tx * u - tz / l * off, a[1] + tz * u + tx / l * off];
}
function atX(valley, x, off = 0) {
  const pts = T.meta.valleys[valley];
  let i = pts.findIndex((p) => p[0] >= x);
  if (i < 1) i = 1;
  const a = pts[i - 1], b = pts[i], u = clamp((x - a[0]) / ((b[0] - a[0]) || 1), 0, 1);
  const tx = b[0] - a[0], tz = b[1] - a[1], l = Math.hypot(tx, tz) || 1;
  return [a[0] + tx * u - tz / l * off, a[1] + tz * u + tx / l * off];
}
const G = (p, agl, r = 11, extra = {}) => ({ x: p[0], z: p[1], agl, r, ...extra });
const L = (pad) => ({ land: pad });
const H = (p, alt) => ({ x: p[0], z: p[1], alt, r: 30, hint: true });   // invisible navigation node at an absolute altitude
// nodes along a valley centre line from fraction t0 to t1: a ring every `every` nodes, hints between
function along(valley, t0, t1, n, agl, r, every) {
  const out = [];
  for (let k = 0; k <= n; k++) out.push(G(on(valley, t0 + (t1 - t0) * k / n), agl, r, { hint: k % every !== 0 }));
  return out;
}

// name, tagline, start pad + heading (rad, 0 = east, positive = turning left/north), wind, medal times (s)
export const EVENTS = [
  {
    id: 'first', name: 'טיסת בכורה', tag: 'המראה, שש טבעות רחבות ונחיתה בכיכר הכפר',
    start: { pad: 'base', yaw: 0 }, wind: { speed: 1.5, gust: 0.4, heading: 0.3 }, medals: [58, 68, 92],
    cps: () => [G([-1480, 2330], 14, 17), G([-1050, 2220], 22, 17), G([-640, 1990], 26, 16), G([-250, 1830], 24, 16),
      G([110, 1840], 22, 16), G([340, 1940], 18, 16), L('village')],
  },
  {
    id: 'lake', name: 'ספרינט האגם', tag: 'נמוך מעל המים, סביב האי וחזרה למזח',
    start: { pad: 'pier', yaw: 0.75 }, wind: { speed: 3, gust: 0.8, heading: 0.2 }, medals: [74, 86, 112],
    cps: () => [G(atX('main', 1750, 200), 9), G(atX('main', 2150, 150), 6, 10), G(atX('main', 2500, 190), 6, 10),
      G(atX('main', 2900, 210), 7, 10), G(atX('main', 3250, 200), 7, 10), G(atX('main', 3560, 190), 8, 13),
      G(atX('main', 3800, 0), 10, 14), G(atX('main', 3560, -190), 9, 13), G(atX('main', 3200, -200), 7, 10), G([2800, 1120], 8, 11),
      G([2500, 1300], 7, 11), G(atX('main', 2200, 20), 6, 10), G(atX('main', 1850, 60), 7, 10),
      G([1560, 2150], 10, 12, { finish: true })],
  },
  {
    id: 'gorge', name: 'הקניון', tag: 'טיפוס של 900 מטר בתוך קניון צר עד האגם העליון',
    start: { pad: 'base', yaw: 0.35 }, wind: { speed: 4, gust: 1.2, heading: -1.2 }, medals: [92, 106, 138],
    cps: () => [G([-1350, 2050], 25, 14), G([-880, 1700], 30, 13), ...along('gorge', 0.02, 0.74, 24, 30, 11, 3), H(on('gorge', 0.80), 1740), H(on('gorge', 0.88), 1930),
      G([-300, -3400], 30, 15, { finish: true })],
  },
  {
    id: 'pass', name: 'מעבר ההר', tag: 'אוויר דליל: מעל הרכס ב-2,300 מטר וצלילה חזרה לעמק',
    start: { pad: 'hutTarn', yaw: 2.35 }, wind: { speed: 7, gust: 1.6, heading: 0.5 }, medals: [134, 152, 196],
    cps: () => [G(on('pass', 0.10), 40, 15), G(on('pass', 0.2), 30, 14), G(on('pass', 0.31), 14, 13), G(on('pass', 0.42), 30, 14),
      G(on('pass', 0.54), 35, 14), G(on('pass', 0.66), 35, 14), G(on('pass', 0.78), 35, 14), G(on('pass', 0.9), 30, 14),
      G([-4250, 1560], 28, 15), G([-3500, 2020], 24, 15), G([-2600, 2400], 24, 15), L('base')],
  },
  {
    id: 'courier', name: 'שליח ההרים', tag: 'שלוש נחיתות מדויקות: בקתת המדף, בקתת האלפ והכפר. המסלול לבחירתך',
    start: { pad: 'pier', yaw: 0.55 }, wind: { speed: 4, gust: 1, heading: 0.4 }, medals: [290, 335, 430],
    cps: () => [H([2400, 1700], 1150), H([3300, 1500], 1400), H([3250, 2200], 1620), H([2780, 2760], 1800), L('hutLedge'),
      H([2700, 2300], 1760), H([3350, 1050], 1500), H(on('east', 0.45), 1400), H(on('east', 0.8), 1700), L('hutAlp'),
      H(on('east', 0.6), 1620), H(atX('main', 3200, 0), 1250), H([1500, 1900], 1060), L('village')],
  },
  {
    id: 'tour', name: 'הסיבוב הגדול', tag: 'כל העמק: טיפוס למעבר ההר, ירידה מהירה דרך הקניון, הכפר וסיבוב סביב האי',
    start: { pad: 'base', yaw: 3.0 }, wind: { speed: 5, gust: 1.2, heading: 0.3 }, medals: [258, 290, 366],
    cps: () => [G([-2600, 2420], 24, 15), G([-3500, 2020], 24, 15), G([-4250, 1560], 28, 15),
      ...along('pass', 0.94, 0.06, 16, 34, 14, 2),
      G([-420, -3400], 22, 14), H(on('gorge', 0.88), 1935), H(on('gorge', 0.80), 1740), ...along('gorge', 0.72, 0.03, 24, 30, 12, 4),
      G([-150, 1700], 24, 14), G([520, 1890], 26, 14), G(atX('main', 1500, 60), 10, 12), G(atX('main', 2200, 150), 7, 11),
      G([2620, 1540], 7, 11), G([2960, 1380], 8, 12), G([2900, 1130], 8, 12), G([2560, 1160], 8, 11), G(atX('main', 2100, -60), 7, 11),
      G([1560, 2150], 10, 12, { finish: true })],
  },
  {
    id: 'castle', name: 'המבצר', tag: 'דרך השער, האולם הגדול, המסדרונות והמגדל, החוצה מעל התהום ומתחת לגשר',
    start: { pad: 'castle', yaw: 0, up: [6, 7.5, 44] }, wind: { speed: 2, gust: 0.4, heading: 0.3 }, medals: [58, 72, 100],
    cps: castleCourse,
  },
];
export const FREE = { id: 'free', name: 'טיסה חופשית', tag: 'בלי שעון. ניקוד על טיסה מהירה וקרובה לקרקע', start: { pad: 'base', yaw: 0 }, wind: { speed: 4, gust: 1, heading: 0.3 }, medals: null, cps: () => [] };

// Smooth curve through the nodes: cubic Hermite with bisector tangents scaled by each segment's
// own length (plain Catmull-Rom loops when a short segment follows a long one).
function catmull(P, step) {
  const n = P.length, dir = [];
  const unit = (a, b) => { const d = [b[0] - a[0], b[1] - a[1], b[2] - a[2]], l = Math.hypot(d[0], d[1], d[2]) || 1; return [d[0] / l, d[1] / l, d[2] / l]; };
  for (let i = 0; i < n; i++) {
    const a = i > 0 ? unit(P[i - 1], P[i]) : unit(P[0], P[1]), b = i < n - 1 ? unit(P[i], P[i + 1]) : unit(P[n - 2], P[n - 1]);
    const t = [a[0] + b[0], a[1] + b[1], a[2] + b[2]], l = Math.hypot(t[0], t[1], t[2]) || 1;
    dir.push([t[0] / l, t[1] / l, t[2] / l]);
  }
  const out = [];
  for (let i = 0; i < n - 1; i++) {
    const p1 = P[i], p2 = P[i + 1], L = Math.hypot(p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]);
    const m = Math.max(2, Math.round(L / step));
    for (let j = 0; j < m; j++) {
      const t = j / m, t2 = t * t, t3 = t2 * t;
      const h00 = 2 * t3 - 3 * t2 + 1, h10 = t3 - 2 * t2 + t, h01 = -2 * t3 + 3 * t2, h11 = t3 - t2;
      const q = [0, 1, 2].map((k) => h00 * p1[k] + h10 * L * dir[i][k] + h01 * p2[k] + h11 * L * dir[i + 1][k]);
      q.seg = i; out.push(q);
    }
  }
  const e = P[n - 1].slice(); e.seg = n - 1; out.push(e);
  return out;
}

// Build the geometry of an event: gate positions and orientations, start pose, and a smooth
// reference line through everything (used by the autopilot, the ghost of the demo, and to keep
// trees off the racing line).
export function buildEvent(ev) {
  const pads = T.meta.pads;
  const sp = pads[ev.start.pad];
  const start = { p: [sp[0], sp[1], sp[2]], yaw: ev.start.yaw, pad: ev.start.pad };
  const cps = ev.cps().map((c) => {
    if (c.land) { const p = pads[c.land]; return { type: 'land', pad: c.land, p: [p[0], p[1], p[2]], r: Math.min(p[3] * 0.55, 15) }; }
    return { type: c.hint ? 'hint' : 'ring', p: [c.x, c.alt !== undefined ? c.alt : surfaceAt(c.x, c.z) + c.agl, c.z], r: c.r, finish: !!c.finish, v: c.v, slow: !!c.slow };
  });
  // reference line: start (lifted), check-points (landings approached from above)
  const up = ev.start.up || [22, 26, 70];     // how the line leaves the pad: height above it, then height and distance of the first node
  const nodes = [[start.p[0], start.p[1] + up[0], start.p[2]], [start.p[0] + Math.cos(start.yaw) * up[2], start.p[1] + up[1], start.p[2] - Math.sin(start.yaw) * up[2]]];
  const nodeOf = [];
  for (const c of cps) { nodeOf.push(nodes.length); nodes.push(c.type === 'land' ? [c.p[0], c.p[1] + 24, c.p[2]] : c.p); }
  const path = catmull(nodes, 10);
  // Keep the line clear of the ground (12 m, less only where a low check-point asks for it),
  // and never ask for a climb or dive steeper than a loaded bike can fly.
  const lowAt = new Map();
  cps.forEach((c, k) => { if (c.type !== 'land') lowAt.set(nodeOf[k], c.p[1] - surfaceAt(c.p[0], c.p[2])); });
  const ds = (i) => Math.hypot(path[i + 1][0] - path[i][0], path[i + 1][2] - path[i][2]);
  for (let i = 0; i < path.length; i++) {
    const q = path[i];
    let clear = 12;
    for (const [node, agl] of lowAt) if (agl < clear && (q.seg === node || q.seg === node - 1)) clear = Math.min(clear, Math.max(agl, 5));
    q[1] = Math.max(q[1], surfaceAt(q[0], q[2]) + clear);
  }
  for (let pass = 0; pass < 2; pass++) {
    for (let i = path.length - 2; i >= 0; i--) path[i][1] = Math.max(path[i][1], path[i + 1][1] - 0.5 * ds(i));
    for (let i = 1; i < path.length; i++) path[i][1] = Math.max(path[i][1], path[i - 1][1] - 0.55 * ds(i - 1));
  }
  // smooth the height profile (crests make a fast bike balloon, dips need room to pull out)
  // while staying above the ground clearance and pinned to the rings
  const floor = path.map((q) => q[1]);
  const pin = new Map();
  cps.forEach((c, k) => { if (c.type !== 'land') { const i = path.findIndex((q) => q.seg >= nodeOf[k]); if (i > 0) pin.set(i, Math.max(c.p[1], floor[i])); } });
  for (let it = 0; it < 60; it++) {
    let prev = path[0][1];
    for (let i = 1; i < path.length - 1; i++) { const cur = path[i][1]; path[i][1] = Math.max(floor[i] - 4, 0.25 * prev + 0.5 * cur + 0.25 * path[i + 1][1]); prev = cur; }
    for (const [i, y] of pin) path[i][1] = y;
  }
  for (let i = 0; i < path.length; i++) path[i][1] = Math.max(path[i][1], surfaceAt(path[i][0], path[i][2]) + 5);
  // confined stretches (the fortress) carry the speed the demo rider keeps there
  const capOf = new Map();
  cps.forEach((c, k) => { if (c.slow) capOf.set(nodeOf[k], c.v); });
  for (const q of path) { const v = capOf.get(q.seg + 1) ?? capOf.get(q.seg); if (v) q.v = v; }
  let s = 0;
  path[0].s = 0;
  for (let i = 1; i < path.length; i++) { s += Math.hypot(path[i][0] - path[i - 1][0], path[i][1] - path[i - 1][1], path[i][2] - path[i - 1][2]); path[i].s = s; }
  // gate normals along the line; path index of each check-point
  cps.forEach((c, k) => {
    const node = nodeOf[k];
    let i = path.findIndex((q) => q.seg >= node);
    if (i < 0) i = path.length - 1;
    c.pathIndex = i;
    const a = path[Math.max(0, i - 2)], b = path[Math.min(path.length - 1, i + 2)];
    const d = [b[0] - a[0], b[1] - a[1], b[2] - a[2]], l = Math.hypot(d[0], d[1], d[2]) || 1;
    c.n = [d[0] / l, d[1] / l, d[2] / l];
    c.s = path[i].s;
    c.lift = path[i][1] - (c.type === 'land' ? c.p[1] + 24 : c.p[1]);
  });
  const real = cps.filter((c) => c.type !== 'hint');
  return { ev, start, cps, real, path, length: s };
}
export const eventById = (id) => (id === 'free' ? FREE : EVENTS.find((e) => e.id === id));
