// הרובוט: מחבר את ההליכה, בקר כל-הגוף, הידיים והאצבעות. צעד בקרה אחד לכל צעד סימולציה.
import { WBC } from './wbc.js';
import { Locomotion } from './gait.js';
import { PowerMeter } from './actuators.js';
import { Hands } from './hands.js';
import { rotZYX, matMul3, clamp } from './linalg.js';
import { NOMINAL, BODY } from '../robot/spec.js';
import { DT } from '../sim/world.js';

export class Robot {
  constructor(world) {
    this.w = world;
    const { mj, model: m, data: d } = world;
    mj.mj_forward(m, d);
    this.wbc = new WBC(world);
    this.loco = new Locomotion(this.wbc);
    this.power = new PowerMeter(this.wbc.joints, BODY.battery.v);
    this.hands = new Hands(world);
    const J = this.wbc.joints;
    this.jidx = Object.fromEntries(J.map((j, k) => [j.name, k]));
    // יעדי תנוחה ומשקלות
    this.post = new Float64Array(J.length);
    this.postW = new Float64Array(J.length);
    this.postDefault = new Float64Array(J.length);
    J.forEach((j, k) => { this.postDefault[k] = d.qpos[j.qadr]; });
    this.resetPosture();
    this.handTargets = { l: null, r: null };   // {pos, R, w, kp}
    this.loads = { l: null, r: null };          // משקל חפץ ביד, ניוטון
    this.squeeze = null;                        // אחיזה בלחיצה של שתי ידיים: {F, mass, k, kw}
    this.lean = 0;          // הטיית הגו קדימה (רד'), היעד
    this.crouch = 0;        // הורדת מרכז המסה (מ'), היעד
    this.leanCur = 0; this.crouchCur = 0; this.speedCrouch = 0;
    this.gaze = null;       // נקודה בעולם שהראש מסתכל עליה
    this.enabled = true;
    this.fallen = false;
    this.t = 0;
    this.tau = new Float64Array(J.length);
    this.armSwing = true;
  }
  resetPosture() {
    const J = this.wbc.joints;
    J.forEach((j, k) => {
      this.post[k] = this.postDefault[k];
      this.postW[k] = j.group === 'leg' ? 0.4 : j.group === 'torso' ? 4 : j.group === 'neck' ? 6 : 5;
    });
  }
  // מערכת הצירים "של הרובוט": ראשית באגן, x קדימה לפי כיוון ההליכה, z למעלה (גובה יחסי לאגן)
  frame() { const d = this.w.data, i = this.wbc.ids.pelvis; return { o: [d.xpos[3 * i], d.xpos[3 * i + 1], d.xpos[3 * i + 2]], h: this.loco.heading }; }
  fromLocal(l) { const { o, h } = this.frame(), c = Math.cos(h), s = Math.sin(h); return [o[0] + c * l[0] - s * l[1], o[1] + s * l[0] + c * l[1], o[2] + l[2]]; }
  toLocal(p) { const { o, h } = this.frame(), c = Math.cos(h), s = Math.sin(h), dx = p[0] - o[0], dy = p[1] - o[1]; return [c * dx + s * dy, -s * dx + c * dy, p[2] - o[2]]; }
  setJoint(name, q, w) { const k = this.jidx[name]; if (k == null) return; this.post[k] = q; if (w != null) this.postW[k] = w; }
  getJointQ(name) { const k = this.jidx[name]; return this.w.data.qpos[this.wbc.joints[k].qadr]; }

  // ---------- צעד בקרה: קוראים בין mj_step1 ל-mj_step2 ----------
  control() {
    const w = this.w, d = w.data, wbc = this.wbc;
    this.t += DT;
    const com = wbc.com();
    const comVel = wbc.out.comVel || [0, 0, 0];
    // נפילה: האגן נמוך מדי או נוטה מאוד
    const pz = d.qpos[2];
    const R = wbc.bodyMat(wbc.ids.pelvis);
    if (!this.fallen && (pz < 0.55 || R[8] < 0.5)) { this.fallen = true; this.fallT = this.t; }
    if (!this.enabled || this.fallen) {
      // רובוט נפול: המנועים רפויים עם ריסון קל, כדי שלא יתפרע
      for (let k = 0; k < wbc.joints.length; k++) {
        const j = wbc.joints[k];
        this.tau[k] = clamp(8 * (this.post[k] - d.qpos[j.qadr]) - 3 * d.qvel[j.dof], -j.torque * 0.2, j.torque * 0.2);
      }
      wbc.apply(this.tau);
      this.hands.control();
      this.power.update(d.qvel, this.tau, DT);
      return;
    }
    // כפיפה והטיה: משתנות בהדרגה (גם אם המשימה מבקשת קפיצה), כמו תנועת גוף אמיתית
    this.crouchCur += clamp(this.crouch - this.crouchCur, -0.15 * DT, 0.15 * DT);
    this.leanCur += clamp(this.lean - this.leanCur, -0.6 * DT, 0.6 * DT);
    // בהליכה מהירה הצעדים ארוכים, והרגל העומדת מתיישרת עד הסוף בסוף הצעד (סינגולריות של הברך).
    // כמו בני אדם: מורידים מעט את מרכז המסה ככל שהמהירות עולה
    const vcmd = Math.hypot(this.loco.cmd[0], this.loco.cmd[1] * 1.5);
    this.speedCrouch += ((this.loco.mode === 'walk' || vcmd > 0.02 ? 0.02 : 0) + Math.min(0.06, 0.11 * Math.max(0, vcmd - 0.2)) - this.speedCrouch) * 0.004;
    this.loco.zTarget = this.loco.zc - this.crouchCur - this.speedCrouch;
    const L = this.loco.update(this.t, { com, comVel });
    // תנופת ידיים טבעית בהליכה, נגד הרגל
    if (this.armSwing) this.swingArms(L);
    // הראש: מבט אל נקודה
    this.aimHead();
    const pelvisR = rotZYX(L.yaw);
    const torsoR = rotZYX(L.yaw, this.leanCur);
    // כשליד יש יעד, התנוחה של הזרוע כמעט משתחררת (אחרת היא "מושכת" את היד חזרה למטה)
    for (const s of ['l', 'r']) {
      const free = !this.handTargets[s];
      for (const jn of ['shoulder_pitch', 'shoulder_roll', 'shoulder_yaw', 'elbow', 'forearm_roll', 'wrist_flex', 'wrist_dev']) {
        const k = this.jidx[`${s}_${jn}`];
        this.postW[k] = free ? 5 : (jn === 'elbow' || jn === 'shoulder_yaw' ? 0.25 : 0.12);
      }
    }
    // אחיזה בשתי ידיים (ארגז): הכוח שהחפץ מפעיל על כל יד, בכיוון ציר y של הגו (כלפי חוץ), ועוד חצי מהמשקל
    if (this.squeeze) {
      const Rt = wbc.bodyMat(wbc.ids.torso), ey = [Rt[1], Rt[4], Rt[7]];
      const { F, mass, k = 1, kw = 1 } = this.squeeze;
      this.loads.l = [ey[0] * F * k, ey[1] * F * k, -mass * 9.81 / 2 * kw];
      this.loads.r = [-ey[0] * F * k, -ey[1] * F * k, -mass * 9.81 / 2 * kw];
    }
    // יעדי ידיים: בעולם, או במערכת של הרובוט (local) כשהוא נושא משהו ומתהלך
    const hands = {};
    const xp = this.w.data.xpos;
    for (const s of ['l', 'r']) {
      const T = this.handTargets[s];
      if (!T) continue;
      if (T.local) {
        // יעד צמוד לגוף: מערכת צירים של האגן עם כיוון ההליכה (רק סיבוב סביב z). כך הזרוע לא יכולה
        // "לרמות" ע"י סיבוב של המותן, והיעד זז עם הרובוט כשהוא הולך.
        const pos = this.fromLocal(T.local);
        T.vel = T.pos && T.prevT === this.t - DT ? [0, 1, 2].map(i => (pos[i] - T.pos[i]) / DT) : [0, 0, 0];
        T.pos = pos; T.prevT = this.t;
        T.R = T.Rlocal ? rotZYX(this.loco.heading + T.Rlocal[0], T.Rlocal[1], T.Rlocal[2]) : null;
      }
      hands[s] = T;
    }
    const cmd = {
      stance: L.stance, comAcc: L.comAcc, pelvisR, torsoR, torsoW: 40, feet: L.feet, share: L.share,
      hands, loads: this.loads, post: this.post, postW: this.postW,
    };
    const tau = wbc.compute(cmd);
    this.tau.set(tau);
    wbc.apply(tau);
    this.hands.control();
    this.power.update(d.qvel, tau, DT);
  }

  swingArms(L) {
    const ph = this.loco.phases[0];
    for (const s of ['l', 'r']) {
      if (this.handTargets[s]) continue;
      const k = this.jidx[s + '_shoulder_pitch'];
      let target = this.postDefault[k];
      if (ph && this.loco.mode === 'walk') {
        // היד נעה קדימה כשהרגל הנגדית קדימה. מחשבים מהמיקום היחסי של כפות הרגליים
        const f = this.loco.feet, other = s === 'l' ? 'r' : 'l';
        const h = this.loco.heading;
        const sole = (x) => this.wbc.sitePos(this.wbc.ids[x + '_sole']);
        const dx = Math.cos(h) * (sole(other)[0] - sole(s)[0]) + Math.sin(h) * (sole(other)[1] - sole(s)[1]);
        target -= clamp(dx * 0.9, -0.35, 0.35);
      }
      this.post[k] += (target - this.post[k]) * 0.05;
    }
  }
  aimHead() {
    const ky = this.jidx.neck_yaw, kp = this.jidx.neck_pitch;
    if (!this.gaze) { this.post[ky] *= 0.97; this.post[kp] += (0.12 - this.post[kp]) * 0.03; return; }
    const head = this.w.data.xpos.subarray(3 * this.wbc.ids.head, 3 * this.wbc.ids.head + 3);
    const dx = this.gaze[0] - head[0], dy = this.gaze[1] - head[1], dz = this.gaze[2] - (head[2] + 0.13);
    const torsoYaw = this.loco.heading + this.getJointQ('waist_yaw');
    const yaw = Math.atan2(Math.sin(Math.atan2(dy, dx) - torsoYaw), Math.cos(Math.atan2(dy, dx) - torsoYaw));
    const pitch = Math.atan2(-dz, Math.hypot(dx, dy)) - this.leanCur;
    this.post[ky] += (clamp(yaw, -1.1, 1.1) - this.post[ky]) * 0.06;
    this.post[kp] += (clamp(pitch, -0.45, 0.65) - this.post[kp]) * 0.06;
  }

  // נתונים לתצוגה
  telemetry() {
    const d = this.w.data, wbc = this.wbc;
    return {
      com: wbc.com(), viz: this.loco.viz, wrenches: wbc.out.wrenches, tau: this.tau,
      power: this.power.P, powerAvg: this.power.Pavg, soc: this.power.soc, hours: this.power.hoursLeft,
      mode: this.loco.mode, fallen: this.fallen, speed: Math.hypot(...(wbc.out.comVel || [0, 0]).slice(0, 2)),
    };
  }
}
