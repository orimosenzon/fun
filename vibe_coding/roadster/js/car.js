// דינמיקת הרכב: גוף קשיח ב-6 דרגות חופש, ארבעה גלגלים עם מתלים וצמיגים,
// הנעה חשמלית, מערכת יניקה, אווירודינמיקה, 10 מנועי גז קר, והתנגשויות של המרכב.

import { V3, Quat, clamp, G0 } from './math.js';
import { VARIANTS, AERO } from './params.js';
import { AirTank, Nozzle, AIR } from './gas.js';
import { tireForce, TIRE } from './tire.js';
import { FAN_DEFAULTS, solveFan } from './fan.js';
import { makeHit, makePen } from './collide.js';

const RHO = AIR.rhoAmb;

// נקודות הגוף לבדיקת פגיעות (מערכת הגוף, מטרים)
const HULL = [
  [0.92, -0.22, 2.14], [-0.92, -0.22, 2.14], [0.0, -0.18, 2.2],
  [0.72, 0.12, 1.75], [-0.72, 0.12, 1.75],
  [0.96, -0.18, -2.16], [-0.96, -0.18, -2.16],
  [0.82, 0.36, -1.95], [-0.82, 0.36, -1.95],
  [0.55, 0.72, -0.15], [-0.55, 0.72, -0.15], [0.0, 0.70, 0.55],
  [1.0, 0.0, 0.6], [-1.0, 0.0, 0.6], [1.0, 0.05, -1.1], [-1.0, 0.05, -1.1],
  // הרצפה הקשיחה (מעל החצאיות הגמישות)
  [0.95, -0.30, 1.35], [-0.95, -0.30, 1.35], [0.95, -0.30, -1.4], [-0.95, -0.30, -1.4], [0, -0.30, 0],
];

// נקודות דגימה של החצאיות: [x, z, אורך ההיקף שהן מייצגות]
const SKIRT = [
  [0, 1.72, 1.5], [0, -1.62, 1.5],
  [0.88, 0.8, 1.55], [0.88, -0.8, 1.55], [-0.88, 0.8, 1.55], [-0.88, -0.8, 1.55],
];
const SKIRT_REACH = 0.22; // החצאית יכולה לרדת עד 22 ס"מ מתחת לרצפה (90 מ"מ + 130 מ"מ אדפטיבי, כדי לעקוב אחרי רמפה מעוגלת)

export class Car {
  constructor(variantKey, world, opts = {}) {
    this.world = world;
    this.opts = { infiniteAir: false, refillBoost: 20, ...opts };
    this.setVariant(variantKey);
    this.pos = new V3(); this.vel = new V3(); this.q = new Quat(); this.w = new V3();
    this.hit = makeHit(); this.pen = makePen();
    this.events = [];
    this.reset(0, 0, 0);
  }

  setVariant(key) {
    const P = this.P = VARIANTS[key];
    this.variant = key;
    this.tank = new AirTank({ volume: P.tankVolume, pFill: P.fillPressure });
    this.thr = P.thrusters.map(t => ({
      ...t,
      posB: new V3(...t.pos), dirB: new V3(...t.dir),
      nozzle: new Nozzle({ throatArea: Nozzle.sizeFor(t.F, P.fillPressure, AIR.TAmb, 4), eps: 4 }),
      cmd: 0, u: 0, F: 0, mdot: 0, pe: 0,
    }));
    const lf = P.frontWeight, a = P.wheelbase * (1 - lf), b = P.wheelbase * lf; // מרחקי הסרנים ממרכז הכובד
    const S = P.suspension;
    const mountY = S.rest - S.staticComp + P.wheelRadius - P.cgHeight;
    this.wheels = [
      { id: 'FL', front: true, left: true, mount: new V3(P.trackF / 2, mountY, a) },
      { id: 'FR', front: true, left: false, mount: new V3(-P.trackF / 2, mountY, a) },
      { id: 'RL', front: false, left: true, mount: new V3(P.trackR / 2, mountY, -b) },
      { id: 'RR', front: false, left: false, mount: new V3(-P.trackR / 2, mountY, -b) },
    ].map(w => ({ ...w, steer: 0, omega: 0, spin: 0, comp: 0, compPrev: 0, contact: false, n: new V3(0, 1, 0), cp: new V3(),
      Fz: 0, fx: 0, fy: 0, kappa: 0, alpha: 0, slip: 0, drive: 0, brake: 0, surf: null, center: new V3() }));
    this.fan = { n: 0, nCmd: 0, dp: 0, Q: 0, power: 0, force: 0, leak: 0, gaps: SKIRT.map(() => 0.5), powerLimited: false, counter: 0 };
    this.aeroMode = 'low';
    this.updateMass();
  }

  updateMass() {
    const P = this.P;
    this.mass = P.massDry + this.tank.m;
    // אינרציה של תיבה עם מסה מרוכזת (0.8) — הערכה
    const l = P.length, wd = P.width, h = 1.0, m = this.mass, k = 0.8 / 12;
    this.I = new V3(m * (h * h + l * l) * k, m * (wd * wd + l * l) * k, m * (wd * wd + h * h) * k);
  }

  reset(x, z, yaw, y = null) {
    const P = this.P;
    const gy = y ?? 0;
    const h = P.cgHeight + 0.02;
    this.pos.set(x, gy + h, z);
    this.vel.set(0, 0, 0); this.w.set(0, 0, 0);
    this.q.setAxisAngle(new V3(0, 1, 0), yaw);
    const S0 = P.suspension.staticComp;
    for (const w of this.wheels) { w.omega = 0; w.comp = S0; w.compPrev = S0; w.steer = 0; }
    this.fan.n = 0;
    for (const t of this.thr) { t.u = 0; t.cmd = 0; }
    this.time = 0; this.airTime = 0; this.contacts = 4;
    this.gBody = new V3(0, 1, 0); this.gSmooth = new V3(0, 1, 0);
    this.accW = new V3();
    this.energyWh = 0; this.powerDraw = 0;
    this.crashed = false; this.maxImpact = 0;
    this.hullContact = false;
    this.distance = 0;
    this.thrustTotal = 0; this.mdotTotal = 0;
    this.drag = 0; this.aeroDown = 0;
    this.motorPower = 0;
    this.basis();
  }

  refillAir() { if (this.P.tankVolume > 0) this.tank.refillTo(this.P.fillPressure); }

  // ---------- בסיס הגוף ----------
  basis() {
    this.left = this.q.rotate(this._l = (this._l || new V3()).set(1, 0, 0));
    this.up = this.q.rotate(this._u = (this._u || new V3()).set(0, 1, 0));
    this.fwd = this.q.rotate(this._f = (this._f || new V3()).set(0, 0, 1));
  }
  toWorldPoint(b, out) { out.copy(b); this.q.rotate(out); return out.add(this.pos); }
  toWorldDir(b, out) { out.copy(b); return this.q.rotate(out); }

  // מוסיף כוח F (עולם) בנקודה p (עולם)
  applyForce(F, p) {
    this.F.add(F);
    const rx = p.x - this.pos.x, ry = p.y - this.pos.y, rz = p.z - this.pos.z;
    this.T.x += ry * F.z - rz * F.y; this.T.y += rz * F.x - rx * F.z; this.T.z += rx * F.y - ry * F.x;
  }
  pointVel(p, out) {
    const rx = p.x - this.pos.x, ry = p.y - this.pos.y, rz = p.z - this.pos.z, w = this.w;
    return out.set(this.vel.x + w.y * rz - w.z * ry, this.vel.y + w.z * rx - w.x * rz, this.vel.z + w.x * ry - w.y * rx);
  }

  // ---------- צעד סימולציה ----------
  // cmd: {throttle, brake, steer, handbrake, reverse, fanN, aeroMode, thr:[0..1 לכל נחיר], refill}
  step(dt, cmd) {
    const P = this.P, S = P.suspension;
    this.F = this.F || new V3(); this.T = this.T || new V3();
    this.F.set(0, 0, 0); this.T.set(0, 0, 0);
    this.basis();
    this.updateMass();
    const m = this.mass;
    const up = this.up, fwd = this.fwd, left = this.left;
    const _p = this._tp || (this._tp = new V3()), _v = this._tv || (this._tv = new V3()), _f = this._tf || (this._tf = new V3());
    const _d = this._td || (this._td = new V3()), _a = this._ta || (this._ta = new V3()), _b = this._tb || (this._tb = new V3());

    // כוח הכובד
    this.F.y -= m * G0;
    let nonGrav = this._ng || (this._ng = new V3()); nonGrav.set(0, 0, 0);

    // ---------- גלגלים: מתלים ----------
    const ray = _d.copy(up).scale(-1);
    const maxLen = S.rest + P.wheelRadius;
    let contacts = 0;
    for (const w of this.wheels) {
      this.toWorldPoint(w.mount, _p);
      w.compPrev = w.comp;
      if (this.world.raycast(_p, ray, maxLen, this.hit)) {
        w.contact = true; contacts++;
        w.comp = clamp(maxLen - this.hit.t, 0, 0.2);
        w.n.copy(this.hit.n); w.mu = this.hit.mu; w.surf = this.hit.kind;
        w.cp.copy(_p).addScaled(ray, this.hit.t);
      } else {
        w.contact = false; w.comp = 0; w.surf = null;
      }
      w.center.copy(_p).addScaled(ray, S.rest - w.comp);
    }
    // מוטות מייצבים: הפרש הדחיסה בין שמאל לימין בכל סרן
    const arbF = S.arbF * (this.wheels[0].comp - this.wheels[1].comp);
    const arbR = S.arbR * (this.wheels[2].comp - this.wheels[3].comp);

    // ---------- הנעה ----------
    const vFwd = this.vel.dot(fwd);
    const driveW = this.drivetrain(cmd, vFwd);

    for (let i = 0; i < 4; i++) {
      const w = this.wheels[i];
      const R = P.wheelRadius;
      // היגוי: גלגלים קדמיים
      if (w.front) w.steer = cmd.steerAngle || 0;
      w.drive = driveW[i];
      w.brake = this.brakeT[i];
      if (!w.contact) {
        w.Fz = 0; w.fx = 0; w.fy = 0; w.slip = 0; w.kappa = 0; w.alpha = 0;
        w.omega += (w.drive / P.wheelInertia) * dt;
        // מגבלת הסל"ד של המנוע (בלי זה גלגל באוויר מסתחרר בלי סוף)
        const wLim = P.vMax / R * 1.08;
        if (Math.abs(w.omega) > wLim) w.omega = Math.sign(w.omega) * wLim;
        const dw = (w.brake + 15) / P.wheelInertia * dt;
        w.omega = Math.abs(w.omega) <= dw ? 0 : w.omega - Math.sign(w.omega) * dw;
        w.spin += w.omega * dt;
        continue;
      }
      const compVel = (w.comp - w.compPrev) / dt;
      let Fs = S.k * w.comp + (compVel > 0 ? S.cBump : S.cReb) * compVel;
      if (w.comp > S.bumpAt) Fs += S.kBump * (w.comp - S.bumpAt);
      Fs += (i < 2 ? (i === 0 ? arbF : -arbF) : (i === 2 ? arbR : -arbR));
      Fs = Math.max(0, Fs);
      const cosN = up.dot(w.n);
      const Fn = Fs * Math.max(0, cosN);
      w.Fz = Fn;
      // כוח נורמלי
      _f.copy(w.n).scale(Fn);
      this.applyForce(_f, w.cp); nonGrav.add(_f);

      // מערכת הצירים של המגע
      const cs = Math.cos(w.steer), sn = Math.sin(w.steer);
      _a.copy(fwd).scale(cs).addScaled(left, sn).projectOnPlane(w.n).norm(); // קדימה במישור המגע
      _b.crossVectors(w.n, _a); // שמאלה
      this.pointVel(w.cp, _v);
      const vx = _v.dot(_a), vy = _v.dot(_b);
      const den = Math.max(Math.abs(vx), 3);
      w.kappa = clamp((w.omega * R - vx) / den, -1.5, 1.5);
      w.alpha = Math.atan(vy / den);
      const tf = tireForce(w.kappa, w.alpha, Fn, w.mu ?? 1);
      // סיבוב הגלגל, אינטגרציה חצי-סתומה: הצמיג קשיח מאוד ביחס לאינרציה של הגלגל,
      // ולכן אוילר מפורש מתנדנד. מלנארים את Fx(ω) ופותרים את ω החדשה.
      const dk = 0.002;
      const Sk = Math.max(0, (tireForce(w.kappa + dk, w.alpha, Fn, w.mu ?? 1).fx - tf.fx) / dk) * R / den;
      const rr = TIRE.rollRes * Fn * R;
      const Iw = P.wheelInertia;
      const w0 = w.omega;
      let wn = w0 + dt * (w.drive - tf.fx * R - Math.sign(w0) * rr) / Iw / (1 + dt * R * Sk / Iw);
      const dw = w.brake / Iw * dt;
      wn = Math.abs(wn) <= dw ? 0 : wn - Math.sign(wn) * dw;
      // הכוח שמתאים למהירות הסיבוב החדשה (באותו קירוב לינארי), חסום בשיא
      const peak = tf.mu * Fn;
      const fx = clamp(tf.fx + Sk * (wn - w0), -peak, peak);
      const wLim = P.vMax / R * 1.08;
      w.omega = Math.abs(wn) > wLim ? Math.sign(wn) * wLim : wn;
      w.fx = fx; w.fy = tf.fy; w.slip = tf.slip; w.muNow = tf.mu;
      _f.copy(_a).scale(fx).addScaled(_b, tf.fy);
      this.applyForce(_f, w.cp); nonGrav.add(_f);
      w.spin += w.omega * dt;
    }
    this.contacts = contacts;
    this.airTime = contacts === 0 ? this.airTime + dt : 0;

    // ---------- יניקה ----------
    this.fanStep(dt, cmd, up, ray, nonGrav);

    // ---------- אווירודינמיקה ----------
    const am = AERO.modes[cmd.aeroMode || this.aeroMode];
    this.aeroMode = cmd.aeroMode || this.aeroMode;
    const sp = this.vel.len();
    _f.copy(this.vel).scale(-0.5 * RHO * am.CdA * sp);
    this.drag = 0.5 * RHO * am.CdA * sp * sp;
    this.toWorldPoint(_p.set(0, 0.05, AERO.copLong), _p);
    this.applyForce(_f, _p); nonGrav.add(_f);
    const vf = Math.max(0, vFwd);
    this.aeroDown = 0.5 * RHO * am.ClA * vf * vf;
    _f.copy(up).scale(-this.aeroDown);
    this.applyForce(_f, _p); nonGrav.add(_f);
    // ריסון סיבובי אווירודינמי קטן
    this.T.addScaled(this.w, -(60 + 4 * sp));

    // ---------- מנועי גז קר ----------
    this.thrustStep(dt, cmd, nonGrav);

    // ---------- פגיעות של המרכב ----------
    this.hullStep(dt, nonGrav);

    // ---------- אינטגרציה ----------
    const ax = this.F.x / m, ay = this.F.y / m, az = this.F.z / m;
    this.vel.x += ax * dt; this.vel.y += ay * dt; this.vel.z += az * dt;
    this.pos.addScaled(this.vel, dt);
    this.distance += sp * dt;
    // משוואות אוילר במערכת הגוף
    const tb = this.q.unrotate(_a.copy(this.T));
    const wb = this.q.unrotate(_b.copy(this.w));
    const I = this.I;
    const Lx = I.x * wb.x, Ly = I.y * wb.y, Lz = I.z * wb.z;
    const gx = wb.y * Lz - wb.z * Ly, gy = wb.z * Lx - wb.x * Lz, gz = wb.x * Ly - wb.y * Lx;
    wb.x += (tb.x - gx) / I.x * dt; wb.y += (tb.y - gy) / I.y * dt; wb.z += (tb.z - gz) / I.z * dt;
    this.w.copy(this.q.rotate(wb));
    this.q.integrate(this.w, dt);

    // מה שהנוסע מרגיש: הכוח הספציפי (כל הכוחות פרט לכובד) חלקי המסה, במערכת הגוף
    this.accW.set(ax, ay, az);
    this.gBody.copy(nonGrav).scale(1 / (m * G0));
    this.q.unrotate(this.gBody);
    // מסנן של 40 מ"ש, כמו מד תאוצה אמיתי (בלי זה כל רעד של הקפיצים נראה כמו 10g)
    const kf = 1 - Math.exp(-dt / 0.04);
    this.gSmooth.lerp(this.gBody, kf);
    this.time += dt;
  }

  drivetrain(cmd, vFwd) {
    const P = this.P;
    const out = this._drv || (this._drv = [0, 0, 0, 0]);
    this.brakeT = this.brakeT || [0, 0, 0, 0];
    const wF = (this.wheels[0].omega + this.wheels[1].omega) / 2, wR = (this.wheels[2].omega + this.wheels[3].omega) / 2;
    let thr = cmd.throttle || 0;
    if (vFwd > P.vMax) thr = 0; else if (vFwd > P.vMax - 3) thr *= (P.vMax - vFwd) / 3;
    let T = thr * P.wheelTorqueMax;
    if (cmd.reverse) T = -Math.min(cmd.reverse, 1) * 2500 * (vFwd > -12 ? 1 : 0);
    // תקרת הספק: P = T·ω
    const wAvg = Math.max(Math.abs(wF * P.frontTorqueShare + wR * (1 - P.frontTorqueShare)), 8);
    const Tcap = P.motorPower / wAvg;
    if (Math.abs(T) > Tcap) T = Math.sign(T) * Tcap;
    const Tf = T * P.frontTorqueShare / 2, Tr = T * (1 - P.frontTorqueShare) / 2;
    this.tc = this.tc || [1, 1, 1, 1];
    for (let i = 0; i < 4; i++) {
      const w = this.wheels[i];
      // בקרת משיכה: מקטינה מומנט כשההחלקה עוברת 12%
      if (cmd.tc !== false && T > 0 && w.contact) {
        if (w.kappa > 0.12) this.tc[i] = Math.max(0.05, this.tc[i] - 0.02);
        else this.tc[i] = Math.min(1, this.tc[i] + 0.004);
      } else this.tc[i] = 1;
      out[i] = (i < 2 ? Tf : Tr) * this.tc[i];
      // בלמים עם ABS
      let b = (cmd.brake || 0) * P.brakeTorque * (i < 2 ? 1 : 0.75);
      if (cmd.handbrake && i >= 2) b = P.brakeTorque * 1.2;
      if (cmd.abs !== false && cmd.brake && w.kappa < -0.14 && Math.abs(vFwd) > 2) b *= 0.3;
      if (!cmd.throttle && !cmd.reverse && Math.abs(vFwd) < 0.4 && w.contact) b = Math.max(b, 900); // בלם החנייה
      this.brakeT[i] = b;
    }
    this.motorPower = Math.max(0, out.reduce((s, t, i) => s + t * this.wheels[i].omega, 0));
    return out;
  }

  fanStep(dt, cmd, up, ray, nonGrav) {
    const P = this.P, F = this.fan, FP = FAN_DEFAULTS;
    F.nCmd = P.fan ? clamp(cmd.fanN ?? 0, 0, 1) : 0;
    const dn = FP.spinRate * dt;
    F.n += clamp(F.nCmd - F.n, -dn * 1.5, dn);
    // מרווחי החצאיות: קרן מכל נקודת דגימה בהיקף
    const _p = this._fp || (this._fp = new V3());
    let leak = 0;
    for (let i = 0; i < SKIRT.length; i++) {
      const [x, z, per] = SKIRT[i];
      // הקרן יוצאת מתוך הגוף (30 ס"מ מעל הרצפה), כדי שגם ברמפה קעורה היא תתחיל מעל המשטח
      this.toWorldPoint(_p.set(x, P.floorY + 0.3, z), _p);
      const d = (this.world.raycast(_p, ray, 0.9, this.hit) ? this.hit.t : 0.9) - 0.3;
      const gap = Math.max(FP.gapMin, d - SKIRT_REACH);
      F.gaps[i] = gap;
      leak += per * gap;
    }
    F.leak = leak;
    // הפתרון של נקודת העבודה כבד יחסית: פעם בשלושה צעדים
    if (F.counter++ % 3 === 0) {
      const sol = solveFan(F.n, leak);
      F.dp = sol.dp; F.Q = sol.Q; F.power = sol.power; F.powerLimited = sol.powerLimited;
      if (sol.powerLimited) F.n = Math.min(F.n, sol.n + 0.02);
    }
    F.force = F.dp * FP.floorArea;
    if (F.force > 0) {
      const f = this._ff || (this._ff = new V3());
      f.copy(up).scale(-F.force);
      this.toWorldPoint(_p.set(0, P.floorY, -0.05), _p);
      this.applyForce(f, _p); nonGrav.add(f);
    }
  }

  thrustStep(dt, cmd, nonGrav) {
    const tank = this.tank;
    let mdot = 0, Ftot = 0;
    const _p = this._hp || (this._hp = new V3()), _f = this._hf || (this._hf = new V3());
    const tau = 0.03; // זמן התגובה של השסתום
    for (let i = 0; i < this.thr.length; i++) {
      const t = this.thr[i];
      t.cmd = clamp(cmd.thr ? cmd.thr[i] || 0 : 0, 0, 1);
      t.u += (t.cmd - t.u) * Math.min(1, dt / tau);
      if (t.u < 1e-3 || tank.m <= 0.01) { t.F = 0; t.mdot = 0; t.pe = 0; continue; }
      const p0 = t.u * tank.p;
      const perf = t.nozzle.perf(p0, tank.T);
      t.F = perf.F; t.mdot = perf.mdot; t.pe = perf.pe; t.regime = perf.regime;
      mdot += perf.mdot; Ftot += perf.F;
      this.toWorldDir(t.dirB, _f).scale(perf.F);
      this.toWorldPoint(t.posB, _p);
      this.applyForce(_f, _p); nonGrav.add(_f);
    }
    // מילוי בתחנה
    let mIn = 0;
    if (cmd.refill && tank.V > 0 && tank.p < this.P.fillPressure) mIn = 0.35 * this.opts.refillBoost;
    tank.step(mdot, dt, mIn);
    if (this.opts.infiniteAir && tank.V > 0 && tank.m < tank.mFull * 0.999) { tank.m = tank.mFull; tank.T = AIR.TAmb; tank.p = this.P.fillPressure; }
    this.mdotTotal = mdot; this.thrustTotal = Ftot;
  }

  hullStep(dt, nonGrav) {
    const _p = this._cp || (this._cp = new V3()), _v = this._cv || (this._cv = new V3()), _f = this._cf || (this._cf = new V3());
    const _b = this._cb || (this._cb = new V3());
    const k = 6e5, c = 2.5e4;
    let any = false;
    for (let i = 0; i < HULL.length; i++) {
      this.toWorldPoint(_b.set(HULL[i][0], HULL[i][1], HULL[i][2]), _p);
      if (!this.world.penetrate(_p, this.pen)) continue;
      any = true;
      const n = this.pen.n, depth = this.pen.depth;
      this.pointVel(_p, _v);
      const vn = _v.dot(n);
      if (-vn > 3) {
        this.maxImpact = Math.max(this.maxImpact, -vn);
        this.events.push({ type: 'impact', speed: -vn, pos: _p.clone() });
      }
      const Fn = Math.max(0, k * depth - c * vn);
      _f.copy(n).scale(Fn);
      // חיכוך החלקה (מגורר), מוגבל כדי לא להפוך כיוון בצעד אחד
      _v.addScaled(n, -vn);
      const vt = _v.len();
      if (vt > 1e-3) {
        const Ft = Math.min(this.pen.mu * Fn, this.mass * vt / dt * 0.2);
        _f.addScaled(_v, -Ft / vt);
        if (vt > 4 && Fn > 2000) this.events.push({ type: 'scrape', speed: vt, pos: _p.clone(), n: n.clone() });
      }
      this.applyForce(_f, _p); nonGrav.add(_f);
    }
    this.hullContact = any;
  }

  // מידע לתצוגה
  get speed() { return this.vel.len(); }
  get altitude() {
    // גובה מעל המשטח שמתחת (קרן כלפי מטה בעולם)
    const d = new V3(0, -1, 0);
    return this.world.raycast(this.pos, d, 200, this.hit) ? this.hit.t - this.P.cgHeight : 200;
  }
  get pitchDeg() {
    // זווית הנטייה של הרכב (0 = אופקי, 90 = אנכי, 180 = הפוך) בכיוון הנסיעה
    // fwd = (cosθ, sinθ), up = (−sinθ, cosθ) במישור האנכי → θ = atan2(fwd.y, up.y)
    this.basis();
    return Math.atan2(this.fwd.y, this.up.y) * 180 / Math.PI;
  }
}
