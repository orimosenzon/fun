// מודל המפעילים: גבול מומנט-מהירות, והספק חשמלי.
//
// מנוע חשמלי עם תמסורת נותן מומנט שיא עד מהירות מסוימת ("פינת ההספק"), ומעליה המומנט יורד
// כי הכא"מ הנגדי (back-EMF) אוכל את המתח. בבלימה (מומנט נגד כיוון התנועה) המנוע יכול לתת מומנט מלא.
// ההספק: עבודה מכנית חיובית / נצילות + הפסדי נחושת ∝ τ² + צריכה קבועה של המחשב והחיישנים.

export const ELEC = {
  base: 92,          // ואט: מחשב ה-AI, חיישנים, מצלמות, בקרים (est, כך שעמידה ≈ 100 ואט כמו ב-AI Day)
  eta: 0.82,         // נצילות מנוע+תמסורת בעבודה חיובית (est)
  regen: 0.25,       // החלק מהעבודה השלילית שחוזר לסוללה (est; תמסורת בורג לא מחזירה הרבה)
  copperAtPeak: 260, // ואט הפסדי נחושת במומנט שיא, לכל מפעיל גדול (est)
};

export function torqueLimit(j, omega, tau) {
  const peak = j.torque, wmax = j.speed;
  if (tau * omega <= 0) return peak;           // בלימה: מומנט מלא
  const w = Math.abs(omega), knee = 0.55 * wmax;
  if (w <= knee) return peak;
  return Math.max(0.05 * peak, peak * (wmax - w) / (wmax - knee));
}

// הספק חשמלי של מפרק אחד (ואט)
export function jointPower(j, omega, tau) {
  const mech = tau * omega;
  const cu = ELEC.copperAtPeak * (j.torque / 180) * (tau / j.torque) ** 2;
  return (mech > 0 ? mech / ELEC.eta : mech * ELEC.regen) + cu;
}

export class PowerMeter {
  constructor(joints, batteryKWh = 2.3) {
    this.joints = joints;
    this.capJ = batteryKWh * 3.6e6;
    this.usedJ = 0;
    this.P = 0; this.Pavg = ELEC.base;
    this.perJoint = new Float64Array(joints.length);
  }
  update(qvel, tau, dt) {
    let P = ELEC.base;
    for (let k = 0; k < this.joints.length; k++) {
      const j = this.joints[k];
      const p = jointPower(j, qvel[j.dof], tau[k]);
      this.perJoint[k] = p;
      P += p;
    }
    this.P = P;
    this.Pavg += (P - this.Pavg) * Math.min(1, dt / 1.5);
    this.usedJ += P * dt;
    return P;
  }
  get soc() { return Math.max(0, 1 - this.usedJ / this.capJ); }
  get hoursLeft() { return (this.capJ - this.usedJ) / Math.max(1, this.Pavg) / 3600; }
}
