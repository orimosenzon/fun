// מערכת היניקה: מאווררים שמוציאים אוויר מתחת לרצפה האטומה (fan car).
// נקודת העבודה היא המקום שבו עקומת המאוורר פוגשת את עקומת הדליפה מתחת לחצאיות.

import { AIR } from './gas.js';

export const FAN_DEFAULTS = {
  dp0: 12000,        // Pa, לחץ סגירה (shut-off) של כל המאווררים יחד במהירות מלאה
  qMax: 14,          // m³/s, ספיקה בלחץ אפס במהירות מלאה
  eta: 0.6,          // יעילות מאוורר+מנוע
  pMax: 80000,       // W, 4 מאווררים × 20 קילוואט
  floorArea: 3.8,    // m², שטח הרצפה האטומה
  perimeter: 9.2,    // m, היקף החצאיות
  cdLeak: 0.62,      // מקדם פריקה של חריץ דק
  gapMin: 0.003,     // m, המרווח הקטן ביותר שהחצאית שומרת
  spinRate: 2.4,     // 1/s, קצב העלייה של סל"ד המאוורר (ה"טורבו לג" החשמלי)
};

// הלחץ שהמאוורר מספק בספיקה Q כשהוא מסתובב בשבר n מהמהירות המרבית (חוקי הדמיון של מאווררים)
export function fanCurve(Q, n, P = FAN_DEFAULTS) {
  if (n <= 0) return 0;
  const qm = P.qMax * n;
  const r = Q / qm;
  return Math.max(0, P.dp0 * n * n * (1 - r * r));
}

// ספיקת הדליפה דרך חריץ בשטח leakArea בהפרש לחץ dp (ברנולי)
export function leakFlow(dp, leakArea, P = FAN_DEFAULTS) {
  return P.cdLeak * leakArea * Math.sqrt(2 * Math.max(dp, 0) / AIR.rhoAmb);
}

// פתרון נקודת העבודה: dp שבו fanCurve(Q(dp)) = dp
export function operatingPoint(n, leakArea, P = FAN_DEFAULTS) {
  if (n <= 0.001) return { dp: 0, Q: 0, power: 0 };
  let lo = 0, hi = P.dp0 * n * n;
  for (let i = 0; i < 40; i++) {
    const mid = (lo + hi) / 2;
    const Q = leakFlow(mid, leakArea, P);
    if (fanCurve(Q, n, P) > mid) lo = mid; else hi = mid;
  }
  const dp = (lo + hi) / 2;
  const Q = leakFlow(dp, leakArea, P);
  return { dp, Q, power: Q * dp / P.eta };
}

// נקודת עבודה עם תקרת הספק: אם צריך יותר מ-pMax, המנוע לא מגיע למהירות המבוקשת
export function solveFan(nCmd, leakArea, P = FAN_DEFAULTS) {
  let op = operatingPoint(nCmd, leakArea, P);
  if (op.power <= P.pMax) return { ...op, n: nCmd, powerLimited: false };
  let lo = 0, hi = nCmd;
  for (let i = 0; i < 30; i++) {
    const mid = (lo + hi) / 2;
    if (operatingPoint(mid, leakArea, P).power > P.pMax) hi = mid; else lo = mid;
  }
  op = operatingPoint(lo, leakArea, P);
  return { ...op, n: lo, powerLimited: true };
}
