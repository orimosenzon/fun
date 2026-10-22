// העולם: תוכנית הדירה, זירת הסומו, וחיתוך קרניים.
// יחידות: מטרים. y גדל כלפי מטה (כמו במסך), זווית θ גדלה עם כיוון השעון.

export const G = 9.81;

export function makeHouse() {
  const W = 12, H = 8;
  const segs = [];
  const boxes = [];
  const wallColor = [232, 223, 210];
  const wall = (x1, y1, x2, y2) => segs.push({ x1, y1, x2, y2, h: 2.5, color: wallColor, kind: 'wall' });

  wall(0, 0, W, 0); wall(W, 0, W, H); wall(W, H, 0, H); wall(0, H, 0, 0);
  wall(5, 0, 5, 3.2); wall(5, 4.2, 5, H);                       // סלון | שאר הבית
  wall(5, 4.5, 6.2, 4.5); wall(7.1, 4.5, 9.6, 4.5); wall(10.5, 4.5, W, 4.5); // מטבח | חדרים
  wall(8.5, 4.5, 8.5, H);                                        // שינה | אמבטיה

  const box = (x, y, w, d, h, color, label) => {
    const b = { x, y, w, d, h, color, label };
    boxes.push(b);
    const k = 'box';
    segs.push({ x1: x, y1: y, x2: x + w, y2: y, h, color, kind: k, box: b });
    segs.push({ x1: x + w, y1: y, x2: x + w, y2: y + d, h, color, kind: k, box: b });
    segs.push({ x1: x + w, y1: y + d, x2: x, y2: y + d, h, color, kind: k, box: b });
    segs.push({ x1: x, y1: y + d, x2: x, y2: y, h, color, kind: k, box: b });
  };
  box(0.3, 6.75, 2.4, 1.0, 0.8, [124, 92, 191], 'ספה');
  box(1.0, 5.0, 1.2, 0.6, 0.45, [160, 118, 75], 'שולחן');
  box(3.6, 6.7, 0.9, 0.9, 0.8, [92, 143, 191], 'כורסה');
  box(0.8, 0.05, 2.2, 0.45, 0.5, [70, 70, 76], 'טלוויזיה');
  box(4.3, 0.3, 0.45, 0.45, 1.0, [63, 154, 83], 'עציץ');
  box(5.05, 0.05, 5.6, 0.6, 0.9, [191, 182, 168], 'שיש');
  box(10.9, 0.05, 0.95, 0.75, 1.9, [208, 214, 220], 'מקרר');
  box(7.2, 2.0, 1.6, 0.9, 0.75, [160, 118, 75], 'שולחן אוכל');
  box(5.3, 5.7, 1.8, 2.1, 0.55, [196, 106, 106], 'מיטה');
  box(7.85, 4.7, 0.6, 1.2, 2.0, [138, 106, 74], 'ארון');
  box(8.6, 4.6, 0.6, 0.5, 0.85, [221, 221, 221], 'כיור');
  box(10.0, 6.95, 1.9, 0.9, 0.6, [159, 195, 214], 'אמבטיה');
  box(11.3, 4.7, 0.45, 0.65, 0.45, [238, 238, 238], 'אסלה');

  const rooms = [
    { x: 0, y: 0, w: 5, h: 8, tint: [233, 225, 211], name: 'סלון' },
    { x: 5, y: 0, w: 7, h: 4.5, tint: [223, 231, 226], name: 'מטבח' },
    { x: 5, y: 4.5, w: 3.5, h: 3.5, tint: [230, 221, 232], name: 'חדר שינה' },
    { x: 8.5, y: 4.5, w: 3.5, h: 3.5, tint: [219, 230, 238], name: 'אמבטיה' },
  ];
  return { kind: 'house', W, H, x0: 0, y0: 0, segs, boxes, rooms, dock: { x: 0.45, y: 1.6 } };
}

export function makeArena() {
  const R = 1.4;
  return { kind: 'arena', W: 4.5, H: 3, x0: -2.25, y0: -1.5, R, segs: [], boxes: [], rooms: [] };
}

// חיתוך קרן עם קטע. מחזיר t (מרחק לאורך הכיוון) או -1.
function hitSeg(ox, oy, dx, dy, s) {
  const sx = s.x2 - s.x1, sy = s.y2 - s.y1;
  const den = dx * sy - dy * sx;
  if (Math.abs(den) < 1e-12) return -1;
  const qx = s.x1 - ox, qy = s.y1 - oy;
  const t = (qx * sy - qy * sx) / den;
  const u = (qx * dy - qy * dx) / den;
  if (t > 1e-6 && u >= 0 && u <= 1) return t;
  return -1;
}

export function rayNearest(segs, ox, oy, dx, dy, maxD) {
  let best = maxD, seg = null;
  for (const s of segs) {
    const t = hitSeg(ox, oy, dx, dy, s);
    if (t > 0 && t < best) { best = t; seg = s; }
  }
  return { d: best, seg };
}

// כל הפגיעות לאורך הקרן, מהרחוקה לקרובה (לציור עם עומק)
export function rayAll(segs, ox, oy, dx, dy, maxD) {
  const out = [];
  for (const s of segs) {
    const t = hitSeg(ox, oy, dx, dy, s);
    if (t > 0 && t < maxD) out.push({ d: t, seg: s });
  }
  out.sort((a, b) => b.d - a.d);
  return out;
}

export function pointSegDist(px, py, s) {
  const sx = s.x2 - s.x1, sy = s.y2 - s.y1;
  const L2 = sx * sx + sy * sy;
  let u = L2 > 0 ? ((px - s.x1) * sx + (py - s.y1) * sy) / L2 : 0;
  u = Math.max(0, Math.min(1, u));
  const cx = s.x1 + u * sx, cy = s.y1 + u * sy;
  return { d: Math.hypot(px - cx, py - cy), cx, cy };
}

export function insideBox(world, x, y, pad = 0) {
  for (const b of world.boxes) {
    if (x > b.x - pad && x < b.x + b.w + pad && y > b.y - pad && y < b.y + b.d + pad) return true;
  }
  return false;
}

export function clearance(world, x, y) {
  let m = Infinity;
  for (const s of world.segs) m = Math.min(m, pointSegDist(x, y, s).d);
  return m;
}

export function randomFreePose(world, minClear = 0.35, rnd = Math.random) {
  for (let i = 0; i < 500; i++) {
    const x = world.x0 + 0.3 + rnd() * (world.W - 0.6);
    const y = world.y0 + 0.3 + rnd() * (world.H - 0.6);
    if (insideBox(world, x, y, 0.05)) continue;
    if (clearance(world, x, y) < minClear) continue;
    return { x, y, th: rnd() * Math.PI * 2 };
  }
  return { x: 2.5, y: 3, th: 0 };
}

export const wrapAngle = a => Math.atan2(Math.sin(a), Math.cos(a));
