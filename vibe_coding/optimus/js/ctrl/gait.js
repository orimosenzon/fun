// הליכה ושיווי משקל על בסיס ה-DCM (Divergent Component of Motion, "נקודת הלכידה הנעה").
//
// המודל: מטוטלת הפוכה לינארית (LIPM). מרכז המסה בגובה קבוע z, והרצפה דוחפת דרך נקודת ה-ZMP p:
//   c̈ = ω²(c − p),   ω = √(g/z)
// משנים משתנים ל-ξ = c + ċ/ω ומקבלים שני חלקים: c נמשך יציב אחרי ξ, ו-ξ בורח מ-p באופן אקספוננציאלי:
//   ξ̇ = ω(ξ − p)
// לכן כל השליטה היא בלהחזיק את ξ: או להזיז את p (בתוך הסוליה, "אסטרטגיית הקרסול"),
// או להזיז את הסוליה עצמה, כלומר לצעוד (step adjustment). זה גם מה שמציל מדחיפות.
import { clamp, rotZYX, smooth5, wrapAngle } from './linalg.js';
import { G } from '../robot/spec.js';

const GRAV = 9.81;

export const GAIT = {
  Tss: 0.45,      // זמן תמיכה ברגל אחת (ש')
  Tds: 0.10,      // זמן תמיכה כפולה
  Tinit: 0.30,    // העברת משקל לפני הצעד הראשון
  Trec: 0.28,     // צעד התאוששות מדחיפה: מהיר יותר
  halfW: 0.098,   // חצי הרוחב בין מרכזי כפות הרגליים
  lift: 0.075,    // גובה הנפת הרגל
  vmax: [0.62, 0.22, 0.75],   // מהירות קדימה, הצידה (מ'/ש'), סיבוב (רד'/ש')
  kDcm: 2.6,      // הגבר משוב ה-DCM
  nAhead: 4,      // כמה צעדים מתכננים קדימה
};

// ---------- גיאומטריה של תמיכה ----------
function footCorners(f, shrink = 0.9) {
  const hx = 0.5 * G.footLen * shrink, hy = 0.5 * G.footW * shrink, c = Math.cos(f.yaw), s = Math.sin(f.yaw);
  return [[hx, hy], [-hx, hy], [-hx, -hy], [hx, -hy]].map(([x, y]) => [f.p[0] + c * x - s * y, f.p[1] + s * x + c * y]);
}
function hull(pts) {
  pts = pts.slice().sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  const cr = (o, a, b) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  const lo = [], up = [];
  for (const p of pts) { while (lo.length >= 2 && cr(lo[lo.length - 2], lo[lo.length - 1], p) <= 0) lo.pop(); lo.push(p); }
  for (const p of pts.slice().reverse()) { while (up.length >= 2 && cr(up[up.length - 2], up[up.length - 1], p) <= 0) up.pop(); up.push(p); }
  return lo.slice(0, -1).concat(up.slice(0, -1));
}
// מצמיד נקודה לתוך מצולע קמור (נגד כיוון השעון)
export function clampToPoly(p, poly) {
  let inside = true;
  for (let i = 0; i < poly.length; i++) {
    const a = poly[i], b = poly[(i + 1) % poly.length];
    if ((b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) < 0) { inside = false; break; }
  }
  if (inside) return p;
  let best = null, bd = Infinity;
  for (let i = 0; i < poly.length; i++) {
    const a = poly[i], b = poly[(i + 1) % poly.length];
    const ab = [b[0] - a[0], b[1] - a[1]], L2 = ab[0] ** 2 + ab[1] ** 2;
    const t = clamp(((p[0] - a[0]) * ab[0] + (p[1] - a[1]) * ab[1]) / L2, 0, 1);
    const q = [a[0] + ab[0] * t, a[1] + ab[1] * t], dd = (q[0] - p[0]) ** 2 + (q[1] - p[1]) ** 2;
    if (dd < bd) { bd = dd; best = q; }
  }
  return best;
}
export function supportPoly(feet, stance, shrink = 0.9) {
  const pts = [];
  for (const s of ['l', 'r']) if (stance[s]) pts.push(...footCorners(feet[s], shrink));
  return hull(pts);
}
const lerp2 = (a, b, t) => [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t];
const meanYaw = (a, b, t = 0.5) => a + wrapAngle(b - a) * t;

export class Locomotion {
  constructor(wbc) {
    this.wbc = wbc;
    this.cmd = [0, 0, 0];
    this.mode = 'stand';
    this.t = 0;
    this.feet = {};
    this.readFeet();
    const c = wbc.com();
    this.zc = c[2];
    this.zTarget = c[2];
    this.omega = Math.sqrt(GRAV / this.zc);
    this.heading = meanYaw(this.feet.l.yaw, this.feet.r.yaw);
    this.phases = [];
    this.viz = {};
    this.pushTimer = 0;
    this.stepCount = 0;
    this.groundZ = { l: this.feet.l.p[2], r: this.feet.r.p[2] };
  }
  readFeet() {
    const w = this.wbc;
    for (const s of ['l', 'r']) {
      const p = w.sitePos(w.ids[s + '_sole']);
      const R = w.bodyMat(w.ids[s + '_foot']);
      this.feet[s] = { p, yaw: Math.atan2(R[3], R[0]) };
    }
  }
  // הפקודה מהמשתמש (או מהמשימה) היא יעד. הפקודה בפועל עולה בהדרגה, עם הגבלת תאוצה,
  // כמו בכל בקר הליכה אמיתי: קפיצה מאפס ל-0.6 מ'/ש' בצעד אחד מפילה את הרובוט
  setCommand(vx, vy, wz) {
    const V = GAIT.vmax;
    this.cmdTarget = [clamp(vx, -0.3, V[0]), clamp(vy, -V[1], V[1]), clamp(wz, -V[2], V[2])];
  }
  rampCommand(dt) {
    const T = this.cmdTarget || [0, 0, 0], acc = [0.5, 0.4, 1.2];
    // עצירה מהירה יותר מהאצה (אבל עדיין לא מיידית)
    for (let i = 0; i < 3; i++) {
      const d = T[i] - this.cmd[i], a = acc[i] * (Math.abs(T[i]) < Math.abs(this.cmd[i]) ? 1.6 : 1);
      this.cmd[i] += clamp(d, -a * dt, a * dt);
    }
  }
  get moving() { return Math.hypot(this.cmd[0], this.cmd[1]) > 0.02 || Math.abs(this.cmd[2]) > 0.03; }

  // ---------- תכנון צעדים ----------
  // מחזיר רשימת צעדים: {side, p:[x,y,z], yaw}, החל מהרגל firstSwing.
  // go=false: עצירה. כל צעד מניח את הרגל ליד הרגל העומדת.
  planFootholds(firstSwing, n, go = true) {
    const T = GAIT.Tss + GAIT.Tds, [vx, vy, wz] = go ? this.cmd : [0, 0, 0];
    const fl = this.feet;
    const steps = [];
    let side = firstSwing;
    if (!go) {
      let st = fl[side === 'l' ? 'r' : 'l'];
      for (let i = 0; i < n; i++) {
        const sg = side === 'l' ? 1 : -1, h = st.yaw;
        const p = [st.p[0] - Math.sin(h) * sg * 2 * GAIT.halfW, st.p[1] + Math.cos(h) * sg * 2 * GAIT.halfW, this.groundZ[side]];
        steps.push({ side, p, yaw: h });
        st = { p, yaw: h };
        side = side === 'l' ? 'r' : 'l';
      }
      return steps;
    }
    // נקודת המוצא: הרגל העומדת, מוזזת אל קו האמצע (אחרת המהירות מתכנסת ל-2/3 מהפקודה)
    const st0 = fl[side === 'l' ? 'r' : 'l'], sg0 = side === 'l' ? -1 : 1;
    let h = st0.yaw;
    let c = [st0.p[0] + Math.sin(h) * sg0 * GAIT.halfW, st0.p[1] - Math.cos(h) * sg0 * GAIT.halfW];
    for (let i = 0; i < n; i++) {
      const hNew = h + wz * T;
      const ch = Math.cos(hNew), sh = Math.sin(hNew);
      const sg = side === 'l' ? 1 : -1;
      // הליכה הצידה: הרגל המובילה יוצאת החוצה בצעד כפול, והרגל הנגררת רק חוזרת לרוחב הרגיל.
      // (אם מקדמים את המרכז שווה בכל צעד, הרגל הנגררת נוחתת צמוד לרגל המובילה)
      const dyStep = Math.sign(vy) === sg ? 2 * vy * T : 0;
      c = [c[0] + ch * vx * T - sh * dyStep, c[1] + sh * vx * T + ch * dyStep];
      // הרחבה קלה בסיבוב, כדי שהרגליים לא יתנגשו
      const w = GAIT.halfW + 0.02 * Math.abs(wz);
      steps.push({ side, p: [c[0] - sh * sg * w, c[1] + ch * sg * w, this.groundZ[side]], yaw: hNew });
      h = hNew;
      side = side === 'l' ? 'r' : 'l';
    }
    return steps;
  }

  // בונה את ציר הזמן (שלבי SS/DS) ואת מסלול ה-DCM הייחוסי, אחורה מהסוף
  buildTimeline(t0, steps, withInit, Tss = GAIT.Tss) {
    const ph = [];
    const feet = { l: { ...this.feet.l }, r: { ...this.feet.r } };
    let t = t0;
    let zmp = withInit ? lerp2(feet.l.p, feet.r.p, 0.5) : null;
    for (let i = 0; i < steps.length; i++) {
      const st = steps[i], stanceSide = st.side === 'l' ? 'r' : 'l';
      const ps = feet[stanceSide].p.slice(0, 2);
      if (i === 0 && withInit) {
        ph.push({ type: 'ds', t0: t, t1: t + GAIT.Tinit, pA: zmp, pB: ps, stance: { l: true, r: true } });
        t += GAIT.Tinit;
      }
      ph.push({ type: 'ss', t0: t, t1: t + Tss, pA: ps, pB: ps, stance: { [stanceSide]: true, [st.side]: false }, swing: st.side, to: st, stanceSide });
      t += Tss;
      feet[st.side] = { p: st.p, yaw: st.yaw };
      const last = i === steps.length - 1;
      const pNext = last ? lerp2(feet.l.p, feet.r.p, 0.5) : st.p.slice(0, 2);
      ph.push({ type: 'ds', t0: t, t1: t + GAIT.Tds * (last ? 2 : 1), pA: ps, pB: pNext, stance: { l: true, r: true }, final: last });
      t += GAIT.Tds * (last ? 2 : 1);
    }
    // DCM אחורה: בסוף עומדים, ξ = ה-ZMP האחרון
    const w = this.omega;
    let xi = ph[ph.length - 1].pB.slice();
    for (let i = ph.length - 1; i >= 0; i--) {
      const P = ph[i], T = P.t1 - P.t0;
      P.xi1 = xi.slice();
      const pd = [(P.pB[0] - P.pA[0]) / T, (P.pB[1] - P.pA[1]) / T];
      P.pd = pd;
      const C = [xi[0] - P.pB[0] - pd[0] / w, xi[1] - P.pB[1] - pd[1] / w];
      const e = Math.exp(-w * T);
      xi = [P.pA[0] + pd[0] / w + C[0] * e, P.pA[1] + pd[1] / w + C[1] * e];
      P.xi0 = xi.slice();
    }
    return ph;
  }
  refAt(P, t) {
    const w = this.omega, T = P.t1 - P.t0, s = clamp((t - P.t0) / T, 0, 1);
    const p = lerp2(P.pA, P.pB, s);
    const C = [P.xi1[0] - P.pB[0] - P.pd[0] / w, P.xi1[1] - P.pB[1] - P.pd[1] / w];
    const e = Math.exp(w * (t - P.t1));
    const xi = [p[0] + P.pd[0] / w + C[0] * e, p[1] + P.pd[1] / w + C[1] * e];
    return { p, xi };
  }

  startWalking(t, firstSwing, recovery = false) {
    this.readFeet();
    const steps = recovery ? this.planFootholds(firstSwing, 1, false) : this.planFootholds(firstSwing, GAIT.nAhead, true);
    this.phases = this.buildTimeline(t, steps, !recovery, recovery ? GAIT.Trec : GAIT.Tss);
    this.mode = 'walk';
    this.recovering = recovery;
    this.enterPhase(t);
  }
  enterPhase(t) {
    const P = this.phases[0];
    if (P.type === 'ss') {
      // תכנון מחדש מהמצב האמיתי בתחילת כל צעד
      const s = P.swing, f = this.feet[s];
      P.from = { p: f.p.slice(), yaw: f.yaw };
      P.toPlanned = { p: P.to.p.slice(), yaw: P.to.yaw };
      P.adj = [0, 0];
    }
  }

  // ---------- צעד בקרה ----------
  update(t, state) {
    this.rampCommand(t - (this.lastT ?? t));
    this.lastT = t;
    this.t = t;
    const w = this.omega;
    const c = state.com, cv = state.comVel;
    const xi = [c[0] + cv[0] / w, c[1] + cv[1] / w];
    let stance = { l: true, r: true }, feetCmd = {}, pRef, xiRef, phase = null, swingYaw = null;

    if (this.mode === 'stand') {
      this.readFeet();
      // היעד בעמידה זז בהדרגה אל האמצע בין הרגליים (ולא קופץ אליו), כדי לא להרוות את הקרסול אחרי צעד
      const mid = lerp2(this.feet.l.p, this.feet.r.p, 0.5);
      if (!this.standRef) this.standRef = clampToPoly(xi, supportPoly(this.feet, { l: true, r: true }, 0.7));
      const dx = mid[0] - this.standRef[0], dy = mid[1] - this.standRef[1], dd = Math.hypot(dx, dy), stepv = 0.25 * 0.002;
      if (dd > stepv) { this.standRef[0] += dx / dd * stepv; this.standRef[1] += dy / dd * stepv; } else this.standRef = mid;
      pRef = this.standRef; xiRef = this.standRef;
      if (this.moving) {
        const [vx, vy, wz] = this.cmd;
        const first = vy > 0.03 ? 'l' : vy < -0.03 ? 'r' : wz > 0.05 ? 'l' : wz < -0.05 ? 'r' : (this.stepCount % 2 ? 'r' : 'l');
        this.standRef = null;
        this.startWalking(t, first);
        return this.update(t, state);
      }
    } else {
      // מתקדמים בציר הזמן
      while (this.phases.length && t >= this.phases[0].t1) {
        const P = this.phases.shift();
        if (P.type === 'ss') {
          this.feet[P.swing] = { p: this.wbc.sitePos(this.wbc.ids[P.swing + '_sole']), yaw: P.to.yaw };
          this.stepCount++;
          // אחרי צעד התאוששות עוברים ישר לעמידה, שמתחילה מהמקום האמיתי של ξ
          if (this.recovering && !this.moving) { this.phases = []; break; }
        }
        if (!this.phases.length) break;
        const N = this.phases[0];
        // בתחילת כל צעד: תכנון מחדש לפי הפקודה העדכנית
        if (N.type === 'ss') {
          // אחרי צעד התאוששות לא עושים צעד נוסף: עומדים במקום, גם אם הרגליים לא צמודות
          if (this.recovering && !this.moving) { this.phases = []; break; }
          this.recovering = false;
          const steps = this.moving ? this.planFootholds(N.swing, GAIT.nAhead, true) : this.planFootholds(N.swing, 1, false);
          this.phases = this.buildTimeline(N.t0, steps, false);
          this.enterPhase(t);
        }
      }
      if (!this.phases.length) {
        this.mode = 'stand'; this.standRef = null;
        this.readFeet();
        return this.update(t, state);
      }
      phase = this.phases[0];
      const r = this.refAt(phase, t);
      pRef = r.p; xiRef = r.xi;
      stance = phase.stance;
      if (phase.type === 'ss') {
        const s = phase.swing, T = phase.t1 - phase.t0, tau = t - phase.t0, sN = tau / T;
        // התאמת צעד: מריצים קדימה את ξ עד סוף הצעד, עם אותו חוק בקרה ועם החיתוך בשולי הסוליה.
        // אם ξ צפוי לסיים במקום אחר מהמתוכנן, מזיזים את הרגל באותה מידה, וכך היחס בין ξ לרגל נשמר.
        if (sN < 0.75) {
          const e = this.predictEndError(phase, t, xi);
          phase.adj[0] += (e[0] - phase.adj[0]) * 0.3; phase.adj[1] += (e[1] - phase.adj[1]) * 0.3;
        }
        const st = this.feet[phase.stanceSide];
        let tgt = [phase.toPlanned.p[0] + phase.adj[0], phase.toPlanned.p[1] + phase.adj[1]];
        tgt = this.limitStep(st, tgt, s, phase.toPlanned.yaw);
        phase.to = { ...phase.to, p: [tgt[0], tgt[1], phase.toPlanned.p[2]] };
        const traj = (ss) => this.swingPos(phase.from, phase.to, ss);
        const dt = 0.004;
        const p0 = traj(sN), pm = traj(sN - dt / T), pp = traj(sN + dt / T);
        const vel = [0, 1, 2].map(i => (pp[i] - pm[i]) / (2 * dt));
        const acc = [0, 1, 2].map(i => (pp[i] - 2 * p0[i] + pm[i]) / (dt * dt));
        const yaw = meanYaw(phase.from.yaw, phase.to.yaw, smooth5(sN));
        swingYaw = yaw;
        feetCmd[s] = { pos: p0, vel, acc, R: rotZYX(yaw), kp: 600 };
        // נגיעה מוקדמת ברצפה
        const sole = this.wbc.sitePos(this.wbc.ids[s + '_sole']);
        if (sN > 0.75 && sole[2] < phase.to.p[2] + 0.003) {
          phase.t1 = t;  // מסיימים את הצעד עכשיו
        }
      }
    }

    // --- משוב DCM: איפה ה-ZMP צריך להיות ---
    const k = this.mode === 'stand' ? 2.0 : GAIT.kDcm;
    let p = [pRef[0] + (1 + k / w) * (xi[0] - xiRef[0]), pRef[1] + (1 + k / w) * (xi[1] - xiRef[1])];
    const poly = supportPoly(this.feet, stance);
    const pc = clampToPoly(p, poly);
    const sat = Math.hypot(p[0] - pc[0], p[1] - pc[1]);
    // דחיפה חזקה בעמידה: כש-ξ עצמו יוצא משטח התמיכה, הקרסול כבר לא יכול להציל. צריך לצעוד
    if (this.mode === 'stand') {
      const inside = clampToPoly(xi, poly);
      const out = Math.hypot(xi[0] - inside[0], xi[1] - inside[1]);
      this.pushTimer = out > 0.01 ? this.pushTimer + 0.002 : 0;
      if (this.pushTimer > 0.01 && !this.noStep) {
        // איזו רגל תצעד? זו שאזור ההגעה שלה (ביחס לרגל השנייה) מגיע הכי קרוב למקום ש-ξ יהיה בו בנחיתה
        let swing = 'l', best = Infinity;
        for (const sw of ['l', 'r']) {
          const st = this.feet[sw === 'l' ? 'r' : 'l'];
          const e = Math.exp(w * GAIT.Trec);
          const pred = [st.p[0] + (xi[0] - st.p[0]) * e, st.p[1] + (xi[1] - st.p[1]) * e];
          const reach = this.limitStep(st, pred, sw, st.yaw);
          const cost = Math.hypot(reach[0] - pred[0], reach[1] - pred[1]);
          if (cost < best) { best = cost; swing = sw; }
        }
        this.standRef = null;
        this.startWalking(t, swing, true);
        this.pushTimer = 0;
      }
    }
    // תאוצת מרכז המסה מה-LIPM, וגובה קבוע
    const zErr = this.zTarget - c[2];
    const comAcc = [w * w * (c[0] - pc[0]), w * w * (c[1] - pc[1]), 140 * zErr - 24 * cv[2]];
    // כיוון האגן: ממוצע הרגליים
    const yl = this.feet.l.yaw, yr = this.feet.r.yaw;
    let yawD;
    if (phase?.type === 'ss') yawD = meanYaw(this.feet[phase.stanceSide].yaw, swingYaw, 0.5);
    else yawD = meanYaw(yl, yr);
    this.heading = yawD;
    // חלוקת העומס בתמיכה כפולה לפי מיקום ה-ZMP
    let share = { l: 0.5, r: 0.5 };
    if (stance.l && stance.r) {
      const a = this.feet.l.p, b = this.feet.r.p, ab = [b[0] - a[0], b[1] - a[1]];
      const tt = clamp(((pc[0] - a[0]) * ab[0] + (pc[1] - a[1]) * ab[1]) / (ab[0] ** 2 + ab[1] ** 2), 0, 1);
      share = { l: 1 - tt, r: tt };
    }
    this.viz = { xi, xiRef, zmp: pc, zmpRaw: p, poly, phase: phase?.type || 'stand', swing: phase?.swing, target: phase?.to, plan: this.phases.filter(q => q.type === 'ss').map(q => q.to), sat };
    return { stance, comAcc, feet: feetCmd, yaw: yawD, share };
  }

  predictEndError(phase, t, xi0) {
    const w = this.omega, k = GAIT.kDcm, poly = supportPoly(this.feet, phase.stance);
    let x = xi0.slice();
    const dt = 0.01;
    for (let tt = t; tt < phase.t1; tt += dt) {
      const r = this.refAt(phase, tt);
      const p = clampToPoly([r.p[0] + (1 + k / w) * (x[0] - r.xi[0]), r.p[1] + (1 + k / w) * (x[1] - r.xi[1])], poly);
      const h = Math.min(dt, phase.t1 - tt);
      x = [x[0] + w * (x[0] - p[0]) * h, x[1] + w * (x[1] - p[1]) * h];
    }
    const xe = phase.xi1;
    return [x[0] - xe[0], x[1] - xe[1]];
  }

  // מגבלות קינמטיות על מיקום הרגל ביחס לרגל העומדת
  limitStep(st, tgt, side, yaw) {
    const c = Math.cos(yaw), s = Math.sin(yaw), sg = side === 'l' ? 1 : -1;
    const d = [tgt[0] - st.p[0], tgt[1] - st.p[1]];
    let fx = c * d[0] + s * d[1], fy = -s * d[0] + c * d[1];
    // אזור ההגעה: עד 50 ס"מ קדימה, 32 אחורה, ובין 15 ל-55 ס"מ הצידה. בצעד ארוך קדימה הצד מצטמצם
    fx = clamp(fx, -0.32, 0.5);
    const latMax = 0.55 - 0.25 * Math.max(0, Math.abs(fx) - 0.2);
    fy = sg > 0 ? clamp(fy, 0.15, latMax) : clamp(fy, -latMax, -0.15);
    return [st.p[0] + c * fx - s * fy, st.p[1] + s * fx + c * fy];
  }
  swingPos(from, to, s) {
    s = clamp(s, 0, 1);
    const ph = smooth5((s - 0.06) / 0.86);
    const z0 = from.p[2], z1 = to.p[2];
    const bump = 16 * s * s * (1 - s) * (1 - s);
    const lift = GAIT.lift + Math.max(0, z1 - z0);
    // ירידה רכה בסוף: מכוונים 4 מ"מ מתחת לרצפה כדי שהמגע יהיה ודאי
    const land = s > 0.9 ? -0.004 * (s - 0.9) / 0.1 : 0;
    return [from.p[0] + (to.p[0] - from.p[0]) * ph, from.p[1] + (to.p[1] - from.p[1]) * ph, z0 + (z1 - z0) * smooth5(s) + lift * bump + land];
  }
}
