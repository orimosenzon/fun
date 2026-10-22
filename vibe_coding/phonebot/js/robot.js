// הרובוט: הנעה דיפרנציאלית (טנק), מודל מנוע DC, תקציב זרם, אנרגיה, והטלפון שעל התושבת.
import { G, pointSegDist } from './world.js';

export const PHONES = {
  mid: { name: 'בינוני: 190 ג׳, 4,500mAh', mass: 0.19, wh: 17.3 },
  small: { name: 'קטן: 150 ג׳, 4,000mAh', mass: 0.15, wh: 15.4 },
  big: { name: 'גדול: 230 ג׳, 5,000mAh', mass: 0.23, wh: 19.3 },
  old: { name: 'ישן מהמגירה: 140 ג׳, סוללה עייפה', mass: 0.14, wh: 7.5 },
};

// rpm ומומנט עצירה (N·m) על ציר הגלגל, אחרי תמסורת, במתח הנקוב V
export const MOTORS = {
  tt: { name: 'TT צהוב 1:48 (≈ $0.5 ליחידה)', V: 6, rpm: 200, stall: 0.08, Is: 1.2, Inl: 0.15, r: 0.0325, mass: 0.07 },
  n20: { name: 'N20 1:100 מתכת (≈ $1.5)', V: 6, rpm: 300, stall: 0.15, Is: 0.7, Inl: 0.06, r: 0.021, mass: 0.04 },
  ga25: { name: '25GA-370 1:34, 12V (≈ $5)', V: 12, rpm: 330, stall: 0.6, Is: 3.0, Inl: 0.2, r: 0.0325, mass: 0.2 },
};

export const POWER = {
  otg: { name: 'סוללת הטלפון דרך USB-C (OTG)', V: 5, Imax: 1.0, wh: 0, mass: 0, usesPhone: true },
  aa: { name: '4×AA אלקליין (6V)', V: 6, Imax: 2.0, wh: 12, mass: 0.1 },
  li: { name: '2×18650 ליתיום + ממיר (6V)', V: 6, Imax: 4.0, wh: 20, mass: 0.1 },
};

// כוח אחיזה של התושבת (ניוטון) ומשך התנגשות של הפגוש (שניות)
export const MOUNTS = {
  clip: { name: 'קליפס קפיצי חזק', F: 8 },
  weak: { name: 'קליפס קפיצי זול ורופף', F: 3 },
  mag: { name: 'מגנט MagSafe', F: 6 },
  screw: { name: 'מלחציים עם בורג', F: 25 },
};
export const BUMPERS = {
  hard: { name: 'פלסטיק קשיח', tau: 0.010 },
  foam: { name: 'פגוש ספוג', tau: 0.035 },
};
export const LINKS = {
  ble: { name: 'Bluetooth LE לבקר ESP32', lat: 0.03, jit: 0.015, phoneW: 0.05 },
  usb: { name: 'כבל USB-C סריאלי (אנדרואיד)', lat: 0.004, jit: 0.001, phoneW: 0.1 },
  wifi: { name: 'Wi‑Fi לבקר ESP32', lat: 0.02, jit: 0.04, phoneW: 0.35 },
};

export const BODY_R = 0.11;   // רדיוס התנגשות
const WHEELBASE = 0.14;
const CHASSIS_MASS = 0.25;
const MU = 0.7;               // חיכוך גומי על פרקט

export class Robot {
  constructor(cfg, pose = { x: 1, y: 1, th: 0 }) {
    this.cfg = cfg;
    this.x = pose.x; this.y = pose.y; this.th = pose.th;
    this.vx = 0; this.vy = 0; this.w = 0;
    this.time = 0;
    this.q = []; this.cur = [0, 0]; this.lastApply = 0;
    this.phoneOn = true; this.canFall = true;
    this.phoneE = PHONES[cfg.phone].wh;
    this.battE = POWER[cfg.power].wh;
    this.pPhone = 1; this.pMotor = 0; this.pMotorAvg = 0; this.pPhoneAvg = 1;
    this.current = 0; this.throttled = 0; this.overT = 0; this.cutoff = 0;
    this.load = 0; this.loadPeak = 0; this.aLong = 0; this.aLat = 0;
    this.impact = 0; this.events = [];
    this.color = '#0f9d8a';
    this.applyCfg(cfg);
  }

  applyCfg(cfg) {
    this.cfg = cfg;
    const m = MOTORS[cfg.motor], p = POWER[cfg.power];
    const k = p.V / m.V;
    this.w0 = (m.rpm * 2 * Math.PI / 60) * k;
    this.ts = m.stall * k;
    this.Is = m.Is * k;
    this.Inl = m.Inl * k;
    this.r = m.r;
    this.phoneMass = PHONES[cfg.phone].mass;
    this.mass = CHASSIS_MASS + m.mass + p.mass + this.phoneMass;
    this.Iz = 0.5 * this.mass * 0.09 * 0.09;
    this.h = cfg.portrait ? 0.07 : 0.035; // גובה מרכז הכובד של הטלפון מעל הקליפס
    const F = MOUNTS[cfg.mount].F;
    this.aCritLong = G * 0.004 / this.h + F * 0.015 / (this.phoneMass * this.h);
    this.aCritLat = 3 * this.aCritLong;   // במישור הטלפון: הקליפס אוחז לכל הרוחב
    this.aCritBack = 2.5 * this.aCritLong; // אחורה: משענת התושבת תומכת
    this.tau = BUMPERS[cfg.bumper].tau;
    this.link = LINKS[cfg.link];
    this.phoneE = Math.min(this.phoneE, PHONES[cfg.phone].wh);
    if (!p.usesPhone && this.battE <= 0) this.battE = p.wh;
    this.battE = Math.min(this.battE, p.wh);
    this.maxSpeed = this.w0 * this.r;
  }

  get speed() { return this.vx * Math.cos(this.th) + this.vy * Math.sin(this.th); }

  // פקודה למנועים, מגיעה אחרי השהיית הערוץ
  cmd(uL, uR) {
    const l = this.link;
    const t = Math.max(this.lastApply, this.time + l.lat + Math.random() * l.jit);
    this.lastApply = t;
    this.q.push({ t, u: [clamp(uL), clamp(uR)] });
    if (this.q.length > 400) this.q.splice(0, this.q.length - 400);
  }
  cmdNow(uL, uR) { this.q.length = 0; this.cur = [clamp(uL), clamp(uR)]; }

  resetPose(p) {
    this.x = p.x; this.y = p.y; this.th = p.th;
    this.vx = this.vy = this.w = 0; this.q.length = 0; this.cur = [0, 0];
    this.lastApply = this.time;
  }
  resetPhone() { this.phoneOn = true; this.load = this.loadPeak = 0; }
  recharge() { this.phoneE = PHONES[this.cfg.phone].wh; this.battE = POWER[this.cfg.power].wh; }

  emit(type, text) { this.events.push({ type, text }); }

  step(dt, world, phoneWatts) {
    this.time += dt;
    while (this.q.length && this.q[0].t <= this.time) this.cur = this.q.shift().u;
    let [uL, uR] = this.cur;

    const p = POWER[this.cfg.power];
    const dead = p.usesPhone ? this.phoneE <= 0 : this.battE <= 0;
    if (dead || this.cutoff > 0) { uL = 0; uR = 0; }
    if (this.cutoff > 0) { this.cutoff -= dt; if (this.cutoff <= 0) this.emit('good', 'הטלפון החזיר את המתח ליציאת ה-USB'); }

    const c = Math.cos(this.th), s = Math.sin(this.th);
    const vf = this.vx * c + this.vy * s;
    let vr = -this.vx * s + this.vy * c; // מהירות הצידה (ימינה)
    const half = WHEELBASE / 2;
    const wl = (vf + this.w * half) / this.r;
    const wr = (vf - this.w * half) / this.r;
    let tL = this.ts * Math.max(-1.5, Math.min(1.5, uL - wl / this.w0));
    let tR = this.ts * Math.max(-1.5, Math.min(1.5, uR - wr / this.w0));

    // זרם מהסוללה: PWM כופל את זרם המנוע ביחס המחזור
    const iOf = (u, t) => Math.abs(u) * (this.Inl + (this.Is - this.Inl) * Math.min(1, Math.abs(t) / this.ts));
    let I = iOf(uL, tL) + iOf(uR, tR);
    this.throttled = 0;
    if (I > p.Imax) {
      const sc = p.Imax / I;
      tL *= sc; tR *= sc; this.throttled = 1 - sc;
      if (p.usesPhone && I > 1.3 * p.Imax) {
        this.overT += dt;
        if (this.overT > 0.4 && this.cutoff <= 0) {
          this.cutoff = 1.5; this.overT = 0;
          this.emit('bad', `זרם יתר (${I.toFixed(1)}A): הטלפון ניתק את המתח ליציאת ה-USB`);
        }
      } else this.overT = Math.max(0, this.overT - dt);
      I = p.Imax;
    } else this.overT = Math.max(0, this.overT - dt);
    this.current = I;

    const N = this.mass * G / 2;
    const FL = Math.max(-MU * N, Math.min(MU * N, tL / this.r));
    const FR = Math.max(-MU * N, Math.min(MU * N, tR / this.r));
    const Froll = 0.03 * this.mass * G * Math.tanh(vf / 0.02);
    const aF = (FL + FR - Froll) / this.mass;
    const scrub = 0.15 * this.mass * G * 0.05 * Math.tanh(this.w / 0.5) + 0.002 * this.w;
    const aW = ((FL - FR) * half - scrub) / this.Iz;

    const nvf = vf + aF * dt;
    this.w += aW * dt;
    const vrNew = vr * Math.exp(-dt / 0.03);
    const aSide = (vr - vrNew) / dt;
    vr = vrNew;
    this.vx = nvf * c - vr * s;
    this.vy = nvf * s + vr * c;
    this.x += this.vx * dt; this.y += this.vy * dt; this.th += this.w * dt;

    // תאוצות שהטלפון מרגיש: לאורך (שלילי = האטה, הטלפון נוטה קדימה) וצידה
    this.load = 0;
    this.addLoad(aF, Math.abs(nvf * this.w) + Math.min(Math.abs(aSide), MU * G));
    this.impact = 0;
    if (world.segs.length) {
      for (const sg of world.segs) {
        const q = pointSegDist(this.x, this.y, sg);
        if (q.d < BODY_R && q.d > 1e-9) {
          const nx = (this.x - q.cx) / q.d, ny = (this.y - q.cy) / q.d;
          this.x += nx * (BODY_R - q.d); this.y += ny * (BODY_R - q.d);
          const vn = this.vx * nx + this.vy * ny;
          if (vn < 0) {
            const dv = -(1 + 0.2) * vn;
            this.vx += dv * nx; this.vy += dv * ny; this.w *= 0.5;
            const a = dv / (this.tau + 0.004);
            const cc = Math.cos(this.th), ss = Math.sin(this.th);
            this.addLoad(a * (nx * cc + ny * ss), Math.abs(a * (-nx * ss + ny * cc)));
            this.impact = Math.max(this.impact, dv);
          }
        }
      }
    }

    // אנרגיה
    this.pMotor = p.V * I;
    this.pPhone = phoneWatts + this.link.phoneW;
    if (p.usesPhone) this.phoneE -= (this.pPhone + this.pMotor / 0.85) * dt / 3600;
    else { this.phoneE -= this.pPhone * dt / 3600; this.battE -= this.pMotor * dt / 3600; }
    this.phoneE = Math.max(0, this.phoneE); this.battE = Math.max(0, this.battE);
    const k = Math.min(1, dt / 3);
    this.pMotorAvg += (this.pMotor - this.pMotorAvg) * k;
    this.pPhoneAvg += (this.pPhone - this.pPhoneAvg) * k;
  }

  // aLong חיובי = תאוצה קדימה: הטלפון נוטה אחורה אל המשענת (חזק).
  // aLong שלילי = האטה או מכה בחזית: הטלפון נוטה קדימה, רק הקליפס מחזיק (חלש).
  addLoad(aLong, aLat) {
    const crit = aLong < 0 ? this.aCritLong : this.aCritBack;
    const ratio = Math.hypot(aLong / crit, aLat / this.aCritLat);
    if (ratio <= this.load) return;
    this.load = ratio;
    this.loadPeak = Math.max(ratio, this.loadPeak * 0.985);
    if (ratio > 1 && this.phoneOn && this.canFall) {
      this.phoneOn = false;
      const dir = Math.abs(aLong) / crit >= aLat / this.aCritLat ? (aLong < 0 ? 'קדימה' : 'אחורה') : 'הצידה';
      const lim = dir === 'הצידה' ? this.aCritLat : crit;
      this.emit('bad', `הטלפון עף ${dir}! ${(Math.hypot(aLong, aLat) / G).toFixed(1)}g, הסף בכיוון הזה ${(lim / G).toFixed(1)}g`);
    }
  }

  // זמן נותר משוער בדקות
  runtimes() {
    const p = POWER[this.cfg.power];
    const phoneDraw = this.pPhoneAvg + (p.usesPhone ? this.pMotorAvg / 0.85 : 0);
    const phoneMin = this.phoneE / Math.max(0.05, phoneDraw) * 60;
    const battMin = p.usesPhone ? null : this.battE / Math.max(0.05, this.pMotorAvg) * 60;
    return { phoneMin, battMin, phoneDraw };
  }
}

// התנגשות בין שני רובוטים (זירת הסומו)
export function collideRobots(a, b) {
  const dx = b.x - a.x, dy = b.y - a.y, d = Math.hypot(dx, dy);
  if (d >= 2 * BODY_R || d < 1e-9) return 0;
  const nx = dx / d, ny = dy / d, pen = 2 * BODY_R - d;
  const ia = 1 / a.mass, ib = 1 / b.mass;
  a.x -= nx * pen * ia / (ia + ib); a.y -= ny * pen * ia / (ia + ib);
  b.x += nx * pen * ib / (ia + ib); b.y += ny * pen * ib / (ia + ib);
  const vrel = (a.vx - b.vx) * nx + (a.vy - b.vy) * ny;
  if (vrel <= 0) return 0;
  const j = (1 + 0.3) * vrel / (ia + ib);
  a.vx -= j * ia * nx; a.vy -= j * ia * ny;
  b.vx += j * ib * nx; b.vy += j * ib * ny;
  // מכה לא מרכזית מסובבת
  const tx = -ny, ty = nx;
  const vt = (a.vx - b.vx) * tx + (a.vy - b.vy) * ty;
  a.w += vt * 2; b.w -= vt * 2;
  const tau = a.tau + b.tau;
  for (const [r, dv, sx] of [[a, j * ia, -1], [b, j * ib, 1]]) {
    const acc = dv / tau, c = Math.cos(r.th), s = Math.sin(r.th);
    const ax = sx * nx * acc, ay = sx * ny * acc;
    r.addLoad(ax * c + ay * s, Math.abs(-ax * s + ay * c));
  }
  return vrel;
}

function clamp(u) { return Math.max(-1, Math.min(1, u)); }
