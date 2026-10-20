// אבני הבניין של המשימות. כל אחת היא פונקציית-גנרטור: כל yield = המתנה של צעד בקרה אחד (2 מ"ש).
// כך משימה מורכבת נכתבת כרצף פשוט: לך, הושט יד, סגור אצבעות, הרם...
import { clamp, rotZYX, wrapAngle, smooth5, matMul3 } from '../ctrl/linalg.js';
import { DT } from '../sim/world.js';

// ---------- מצב הרובוט ----------
export function feetMid(r) {
  const a = r.wbc.sitePos(r.wbc.ids.l_sole), b = r.wbc.sitePos(r.wbc.ids.r_sole);
  return [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2];
}
export function bodyPos(r, name) { const d = r.w.data, i = r.w.bodyId(name); return [d.xpos[3 * i], d.xpos[3 * i + 1], d.xpos[3 * i + 2]]; }
export function bodyMat(r, name) { const i = r.w.bodyId(name); return Array.from(r.w.data.xmat.subarray(9 * i, 9 * i + 9)); }
// מנקודה במערכת של הרובוט (x קדימה, y שמאלה, מעל הרצפה) לעולם
export function robotToWorld(r, p, origin = feetMid(r), h = r.loco.heading) {
  const c = Math.cos(h), s = Math.sin(h);
  return [origin[0] + c * p[0] - s * p[1], origin[1] + s * p[0] + c * p[1], p[2]];
}

export function* wait(sec) { for (let t = 0; t < sec; t += DT) yield; }
export function* waitUntil(fn, timeout = 10) { for (let t = 0; t < timeout && !fn(); t += DT) yield; }

// ---------- ניווט: הליכה אל נקודה וכיוון ----------
// שלב א': פנייה אל היעד והליכה קדימה. שלב ב' (קרוב): צעדים קטנים הצידה/קדימה ויישור לכיוון הסופי.
export function* walkTo(r, x, y, yaw = null, { tol = 0.035, yawTol = 0.06, speed = 0.4, turn = 0.6, timeout = 40, ctx, strafe = null } = {}) {
  let t = 0, fine = 0;
  const p0 = feetMid(r);
  // מרחק קצר עם כיוון סופי: צעדים הצידה/קדימה בלי להסתובב (כמו אדם שזז לאורך שולחן)
  const holo = strafe ?? (yaw != null && Math.hypot(x - p0[0], y - p0[1]) < 0.9);
  while (t < timeout) {
    t += DT;
    const p = feetMid(r), h = r.loco.heading;
    const dx = x - p[0], dy = y - p[1], dist = Math.hypot(dx, dy);
    const bearing = Math.atan2(dy, dx);
    const yawGoal = yaw ?? bearing;
    let vx = 0, vy = 0, wz = 0;
    if (dist > 0.45 && !holo) {
      const e = wrapAngle(bearing - h);
      wz = clamp(1.6 * e, -turn, turn);
      vx = clamp(0.9 * dist, 0, speed) * Math.max(0, Math.cos(e)) ** 2;
    } else {
      const c = Math.cos(h), s = Math.sin(h);
      const ex = c * dx + s * dy, ey = -s * dx + c * dy;
      vx = clamp(1.2 * ex, -0.2, 0.25); vy = clamp(1.2 * ey, -0.13, 0.13);
      wz = clamp(1.5 * wrapAngle(yawGoal - h), -turn, turn);
      if (dist < 0.12) fine += DT;
      // אם ההתייצבות העדינה מתארכת, מקבלים סטייה קצת יותר גדולה
      const tl = fine > 3 ? tol * 2 : tol;
      if (dist < tl) { vx = 0; vy = 0; }
      if (Math.abs(wrapAngle(yawGoal - h)) < yawTol) wz = 0;
      if (!vx && !vy && !wz) break;
    }
    // פקודה קטנה מדי לא מזיזה רגל; מעגלים למינימום שימושי
    if (Math.hypot(vx, vy) > 0 && Math.hypot(vx, vy) < 0.04) { const k = 0.04 / Math.hypot(vx, vy); vx *= k; vy *= k; }
    r.loco.setCommand(vx, vy, wz);
    if (ctx) ctx.status = dist > 0.45 ? `הולך אל היעד: ${dist.toFixed(1)} מ'` : (ctx.walkLabel || 'מתייצב ליד היעד');
    yield;
    if (r.fallen) return false;
  }
  r.loco.setCommand(0, 0, 0);
  yield* waitUntil(() => r.loco.mode === 'stand', 3);
  return !r.fallen;
}

// ---------- תנועת יד חלקה ----------
// target: {pos, R?}. התנועה בפרופיל מינימום-ג'רק (חזקה חמישית), כמו תנועות יד של בני אדם
export function* moveHand(r, side, target, T = 1.2, { w = 90, kp = 180, wr = 25, loadF = null } = {}) {
  const cur = r.handTargets[side]?.pos || r.wbc.sitePos(r.wbc.ids[side + '_grip']);
  const start = cur.slice();
  const tgt = target.pos;
  for (let t = 0; t <= T; t += DT) {
    const s = smooth5(t / T);
    const pos = [0, 1, 2].map(i => start[i] + (tgt[i] - start[i]) * s);
    r.handTargets[side] = { pos, R: target.R || null, w, kp, wr };
    if (loadF) r.loads[side] = loadF;
    yield;
  }
  r.handTargets[side] = { pos: tgt.slice(), R: target.R || null, w, kp, wr };
}
export function* moveHands(r, targets, T = 1.2, opts = {}) {
  const gens = Object.entries(targets).map(([s, tg]) => moveHand(r, s, tg, T, opts[s] || opts));
  let alive = true;
  while (alive) { alive = false; for (const g of gens) if (!g.next().done) alive = true; if (alive) yield; }
}
export function releaseHands(r, sides = ['l', 'r']) { for (const s of sides) { r.handTargets[s] = null; r.loads[s] = null; } }

// אוריינטציה של כף יד: yaw של הרובוט ואז הטיות. כף היד "תלויה" (אצבעות למטה) היא rotZYX(heading)
export function handR(heading, pitch = 0, roll = 0) { return rotZYX(heading, pitch, roll); }

export function* grasp(r, side, shape, close = 1, T = 0.6) {
  r.hands.set(side, shape, close, close / Math.max(0.2, T));
  yield* wait(T + 0.1);
}

// האם גוף נוגע בגיאומטריה ששמה מתחיל במחרוזת מסוימת (למשל המגש)
export function touching(r, bodyName, prefix) {
  const { mj, model: m, data: d } = r.w, b = r.w.bodyId(bodyName);
  const cs = d.contact;
  let hit = false;
  for (let i = 0; i < d.ncon && !hit; i++) {
    const c = cs.get(i), b1 = m.geom_bodyid[c.geom1], b2 = m.geom_bodyid[c.geom2];
    if (b1 !== b && b2 !== b) continue;
    const other = b1 === b ? c.geom2 : c.geom1;
    const n = mj.mj_id2name(m, mj.mjtObj.mjOBJ_GEOM.value, other) || '';
    if (n.startsWith(prefix)) hit = true;
  }
  cs.delete();
  return hit;
}

// מיקום של גוף חופשי בסצנה
export function objPos(r, name) { return bodyPos(r, name); }
export function objMass(r, name) { const i = r.w.bodyId(name); return r.w.model.body_mass[i]; }
