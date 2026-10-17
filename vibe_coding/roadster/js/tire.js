// מודל צמיג: "הנוסחה הקסומה" של פאצ'ייקה (Pacejka Magic Formula) עם החלקה משולבת.
// מקדם החיכוך 1.26 הוא המספר שלארס מוראווי ציין לצמיגי הרודסטר בלי מאווררים.

export const TIRE = {
  mu0: 1.26,          // חיכוך שיא בעומס הנומינלי
  fz0: 5400,          // N, עומס נומינלי לגלגל
  loadSens: 0.11,     // ירידת החיכוך עם העומס (load sensitivity)
  Bx: 15, Cx: 1.6, Ex: 0,     // אורכי: שיא בהחלקה של 10%
  By: 16.6, Cy: 1.35, Ey: 0,  // רוחבי: שיא בזווית החלקה של 8°
  rollRes: 0.012,
};

export function magic(x, B, C, D, E) {
  const bx = B * x;
  return D * Math.sin(C * Math.atan(bx - E * (bx - Math.atan(bx))));
}

// מקדם חיכוך תלוי עומס
export function muAt(fz, T = TIRE, surfMu = 1) {
  const r = fz / T.fz0;
  return surfMu * T.mu0 * Math.max(0.55, 1 - T.loadSens * (r - 1));
}

// כוחות הצמיג. kappa = יחס החלקה, alpha = זווית החלקה (רדיאנים), fz = עומס.
// משתמשים בשיטת "ההחלקה המנורמלת": מצרפים את שתי ההחלקות לווקטור אחד,
// מחשבים את הכוח הכולל ומחלקים אותו בין הכיוונים. כך נשמר מעגל החיכוך.
const KX_PEAK = 0.1, AY_PEAK = 0.14;
export function tireForce(kappa, alpha, fz, surfMu = 1, T = TIRE) {
  if (fz <= 0) return { fx: 0, fy: 0, mu: 0, slip: 0 };
  const mu = muAt(fz, T, surfMu);
  const D = mu * fz;
  const sx = kappa / KX_PEAK;
  const sy = Math.tan(alpha) / Math.tan(AY_PEAK);
  const s = Math.hypot(sx, sy);
  if (s < 1e-6) return { fx: 0, fy: 0, mu, slip: 0 };
  // משקלל בין עקומת האורך לעקומת הרוחב לפי כיוון ההחלקה
  const wx = (sx * sx) / (s * s);
  const fxMag = magic(s * KX_PEAK, T.Bx, T.Cx, D, T.Ex);
  const fyMag = magic(s * Math.tan(AY_PEAK), T.By, T.Cy, D, T.Ey);
  const F = wx * fxMag + (1 - wx) * fyMag;
  return { fx: F * sx / s, fy: -F * sy / s, mu, slip: s };
}
