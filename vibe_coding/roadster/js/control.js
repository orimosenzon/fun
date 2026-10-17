// מהקלט של השחקן לפקודות לרכב: היגוי, הנעה, מצב מאווררים,
// ובקר הטיסה שמחלק את המומנטים בין הנחירים (control allocation).

import { V3, clamp, G0 } from './math.js';

export class Controller {
  constructor() {
    this.steer = 0;
    this.fanMode = 'auto';
    this.aeroMode = 'low';
    this.assist = true;
    this.tc = true;
    this.kick = null;
    this.hoverHold = false;
    this.cmd = { throttle: 0, brake: 0, steerAngle: 0, reverse: 0, handbrake: false, fanN: 0, aeroMode: 'low', thr: [], refill: false, tc: true, abs: true };
    this.status = '';
  }

  startKickflip(dir = 1) { this.kick = { phase: 'launch', t: 0, roll: 0, dir }; }

  // raw: {throttle, brake, steer, boost, lift, pitch, roll, yaw, handbrake}
  update(car, raw, dt, env = {}) {
    const P = car.P, cmd = this.cmd;
    car.basis();
    const vFwd = car.vel.dot(car.fwd), v = car.speed;
    const airborne = car.airTime > 0.12;
    this.airborne = airborne;

    // ---------- היגוי (בקרקע) ----------
    let target = airborne ? 0 : clamp(raw.steer, -1, 1);
    // סיוע היגוי בלולאה (כמו ההדגמה, שבה הרכב נסע לבד): רק כשהנהג לא מיגה בעצמו
    this.guiding = false;
    if (env.guide && !airborne && Math.abs(raw.steer) < 0.1 && car.wheels.some(w => w.surf === 'loop')) {
      const g = env.guide(car.pos, car.left);
      if (g) {
        const vLat = car.vel.dot(car.left);
        target = clamp(0.25 * g.err - 1.2 * vLat / Math.max(v, 3), -1, 1);
        this.guiding = true;
      }
    }
    const rate = Math.abs(target) > Math.abs(this.steer) ? 2.6 : 4.5;
    this.steer += clamp(target - this.steer, -rate * dt, rate * dt);
    const maxA = 0.56 / (1 + (v / 22) ** 2) + 0.03; // זווית היגוי מרבית קטנה עם המהירות
    cmd.steerAngle = this.steer * maxA;

    // ---------- הנעה ובלימה ----------
    cmd.throttle = clamp(raw.throttle, 0, 1);
    cmd.brake = 0; cmd.reverse = 0;
    if (raw.brake > 0) {
      if (vFwd > 0.8 || cmd.throttle > 0) cmd.brake = raw.brake;
      else cmd.reverse = raw.brake;
    }
    cmd.handbrake = !!raw.handbrake;
    cmd.tc = this.tc; cmd.abs = this.tc;
    cmd.aeroMode = this.aeroMode;
    cmd.refill = !!env.atStation && v < 0.6;

    // ---------- מאווררים ----------
    let fan = 0;
    if (this.fanMode === 'max') fan = 1;
    else if (this.fanMode === 'auto') {
      const slope = 1 - car.up.y;                  // 0 על משטח אופקי
      const lat = Math.abs(cmd.steerAngle) * v * v / 2.7; // תאוצה צידית משוערת
      fan = 0.35;
      if (slope > 0.06 || lat > 7 || (cmd.brake > 0.3 && v > 8) || (cmd.throttle > 0.6 && v < 45)) fan = 1;
      if (car.contacts === 0 && car.airTime > 0.3) fan = 0.35;
    }
    cmd.fanN = fan;

    // ---------- נחירים ----------
    const thr = cmd.thr;
    thr.length = car.thr.length;
    for (let i = 0; i < thr.length; i++) thr[i] = 0;
    if (!car.thr.length) { this.status = ''; return cmd; }
    const idx = (id) => car.thr.findIndex(t => t.id === id);
    const tank = car.tank;
    // דחף מרבי זמין עכשיו לכל נחיר (תלוי בלחץ שנותר במכל)
    const fmax = car.thr.map(t => tank.m > 0.01 ? t.nozzle.perf(tank.p, tank.T).F : 0);

    const boost = clamp(raw.boost, 0, 1);
    for (let i = 0; i < car.thr.length; i++) if (car.thr[i].group === 'main') thr[i] = boost;

    const hasLift = car.thr.some(t => t.group === 'lift');
    if (!hasLift) { this.status = boost ? 'דחף אחורי' : ''; return cmd; }

    // ---------- בקר הטיסה ----------
    const wb = car.q.unrotate(new V3().copy(car.w)); // מהירות זוויתית בגוף: x=pitch (חיובי=חרטום מטה), y=yaw (חיובי=שמאלה), z=roll (חיובי=צד שמאל עולה)
    const ub = car.q.unrotate(new V3(0, 1, 0));     // "למעלה" של העולם, כפי שהרכב רואה אותו
    let lift = clamp(raw.lift, 0, 1);
    let wDes = new V3(0, 0, 0);
    let attitude = airborne || lift > 0;
    let level = this.assist;
    const kL = 4.5;
    // פקודות המקל
    const pIn = clamp(raw.pitch, -1, 1), rIn = clamp(raw.roll, -1, 1), yIn = clamp(raw.yaw, -1, 1);
    if (airborne) { wDes.x = pIn * 3.2; wDes.z = rIn * 5.5; }
    wDes.y = yIn * 1.8;

    // יישור אוטומטי לאופק בציר שאין בו קלט
    // הסיבוב הקצר ביותר שמחזיר את "למעלה" של הרכב ל"למעלה" של העולם: ציר × זווית
    const levelErr = () => {
      const ang = Math.acos(clamp(ub.y, -1, 1));
      let ax = ub.z, az = -ub.x;
      const h = Math.hypot(ax, az);
      if (h < 1e-3) { ax = 0; az = 1; } else { ax /= h; az /= h; }
      const a = Math.min(ang, 1.3); // רוויה: לא יותר מ-1.3 רדיאן "שגיאה" כדי לא להגזים בקצב
      return [ax * a, az * a];
    };

    // ---------- קיקפליפ אוטומטי ----------
    if (this.kick) {
      const K = this.kick; K.t += dt;
      attitude = true;
      if (K.phase === 'launch') {
        lift = 1; level = true;
        if ((airborne && car.vel.y > 6) || K.t > 1.0) { K.phase = 'spin'; K.t = 0; }
        this.status = 'קיקפליפ: המראה';
      } else if (K.phase === 'spin') {
        K.roll += wb.z * dt;
        wDes.z = K.dir * 7.5; wDes.x = 0; level = false;
        lift = 0;
        if (Math.abs(K.roll) > Math.PI * 2 - 0.9 || K.t > 2.2) { K.phase = 'land'; K.t = 0; }
        this.status = `קיקפליפ: ${Math.round(Math.abs(K.roll) * 180 / Math.PI)}°`;
      } else {
        level = true;
        this.status = 'קיקפליפ: נחיתה';
        if (car.contacts >= 3 || K.t > 6) this.kick = null;
      }
    }

    if (attitude) {
      const [ex, ez] = levelErr();
      if (level && (!airborne || Math.abs(pIn) < 0.05) && !(this.kick && this.kick.phase === 'spin')) wDes.x = kL * ex;
      if (level && (!airborne || Math.abs(rIn) < 0.05) && !(this.kick && this.kick.phase === 'spin')) wDes.z = kL * ez;
    }

    // ---------- מצב ייצוב: רווח = טיפוס בקצב קבוע, J = החזקת גובה ----------
    // בלי ייצוב, רווח פותח את ארבעת נחירי הפינות עד הסוף (כ-4g כלפי מעלה). זה טוב לקפיצה, אבל אי אפשר לרחף כך.
    this.hovering = false;
    if (this.assist && !this.kick && (raw.lift > 0 || (this.hoverHold && airborne))) {
      const vyT = raw.lift > 0 ? 8 * raw.lift : 0;
      const fAll = fmax.filter((_, i) => car.thr[i].group === 'lift').reduce((a, b) => a + b, 0) * Math.max(0.3, ub.y);
      if (fAll > 0) lift = clamp(car.mass * (G0 + 2.5 * (vyT - car.vel.y)) / fAll, 0, 1);
      this.hovering = raw.lift === 0;
      if (car.tank.m < 1) this.hoverHold = false;
    }

    // ---------- סיוע נחיתה: בלימת הנפילה כמו בהדגמה ----------
    this.landing = false;
    if (this.assist && airborne && ub.y > 0.55 && car.vel.y < -2.5 && !(this.kick && this.kick.phase === 'spin')) {
      const alt = env.altitude ?? 50;
      const fl = car.thr.filter(t => t.group === 'lift');
      const fUp = fmax.filter((_, i) => car.thr[i].group === 'lift').reduce((a, b) => a + b, 0) * ub.y;
      const aUp = fUp / car.mass - G0;
      if (aUp > 0.5 && fl.length) {
        const vt = -Math.sqrt(2 * 0.4 * aUp * Math.max(alt - 0.3, 0)) - 1.0;
        const want = clamp((vt - car.vel.y) * 0.9, 0, 1);
        if (want > lift) { lift = want; this.landing = want > 0.05; }
      }
    }

    // ---------- הקצאה: מומנט רצוי → פקודות לנחירים ----------
    const I = car.I, Kd = 11;
    const tau = new V3(
      attitude ? I.x * Kd * (wDes.x - wb.x) : 0,
      I.y * 6 * (wDes.y - wb.y) * (airborne || Math.abs(yIn) > 0 ? 1 : 0),
      attitude ? I.z * Kd * (wDes.z - wb.z) : 0,
    );
    // ארבעת נחירי הפינות: τx = −Σ z·F, τz = Σ x·F
    const corners = ['FL', 'FR', 'RL', 'RR'].map(idx);
    let sz2 = 0, sx2 = 0;
    for (const i of corners) { const p = car.thr[i].posB; sz2 += p.z * p.z; sx2 += p.x * p.x; }
    const F = corners.map(i => {
      const p = car.thr[i].posB;
      return lift * fmax[i] + (-p.z) * tau.x / sz2 + p.x * tau.z / sx2;
    });
    // נחיר לא יכול "למשוך": מוסיפים לכולם באותה מידה (נשמר המומנט, נוסף עילוי)
    const mn = Math.min(...F);
    if (mn < 0) for (let k = 0; k < 4; k++) F[k] -= mn;
    // רוויה: אם צריך יותר מהמקסימום, מקטינים הכול באותו יחס
    let sc = 1;
    for (let k = 0; k < 4; k++) { const mx = fmax[corners[k]]; if (mx > 0 && F[k] > mx) sc = Math.min(sc, mx / F[k]); }
    for (let k = 0; k < 4; k++) { const mx = fmax[corners[k]]; thr[corners[k]] = mx > 0 ? clamp(F[k] * sc / mx, 0, 1) : 0; }
    // סבסוב: τy = ±z·F
    const iyl = idx('YL'), iyr = idx('YR');
    if (iyl >= 0) {
      const arm = car.thr[iyr].posB.z;
      if (tau.y > 0) thr[iyr] = clamp(tau.y / arm / (fmax[iyr] || 1), 0, 1);
      else thr[iyl] = clamp(-tau.y / arm / (fmax[iyl] || 1), 0, 1);
      // אזור מת קטן כדי לא לבזבז אוויר על רעש
      if (thr[iyl] < 0.04) thr[iyl] = 0;
      if (thr[iyr] < 0.04) thr[iyr] = 0;
    }
    if (!this.kick) this.status = this.landing ? 'בלימת נפילה' : this.hovering ? 'ריחוף (J)' : airborne ? 'טיסה' : lift > 0 ? 'המראה' : boost ? 'דחף אחורי' : '';
    // אזור מת קטן בפינות כשאין עילוי ואין צורך אמיתי
    if (lift === 0 && !airborne) for (const i of corners) thr[i] = 0;
    return cmd;
  }
}
