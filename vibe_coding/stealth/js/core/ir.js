// חתימה תרמית: פלאנק, טמפרטורת עור מחיכוך, וטווח גילוי של חיישן אינפרא-אדום (IRST)
const h = 6.62607015e-34, c = 299792458, kB = 1.380649e-23;
export const SIGMA_SB = 5.670374419e-8;

// קרינות ספקטרלית W/(m²·sr·m)
export function planck(lam, T) {
  const x = h * c / (lam * kB * T);
  if (x > 700) return 0;
  return 2 * h * c * c / (lam ** 5 * (Math.exp(x) - 1));
}

// קרינות בתחום [l1,l2] במטרים, W/(m²·sr)
export function bandRadiance(T, l1, l2, n = 200) {
  let s = 0; const d = (l2 - l1) / n;
  for (let i = 0; i < n; i++) s += planck(l1 + (i + 0.5) * d, T);
  return s * d;
}

export const MWIR = [3e-6, 5e-6], LWIR = [8e-6, 12e-6];

// אטמוספרה סטנדרטית (ISA) עד 20 ק"מ
export function isaT(hKm) { return hKm < 11 ? 288.15 - 6.5 * hKm : 216.65; }

// טמפרטורת התאוששות על העור: T = T∞(1 + r(γ-1)/2·M²)
export function skinT(M, hKm, r = 0.85) { return isaT(hKm) * (1 + r * 0.2 * M * M); }

// עוצמת קרינה (W/sr) לפי זווית מבט. aspect: 0 מלפנים, 180 מאחור
export function irIntensity(o, aspectDeg, band = MWIR) {
  const T0 = isaT(o.hKm), Ts = skinT(o.M, o.hKm);
  const a = aspectDeg * Math.PI / 180;
  // שטח עור נראה: חזית קטנה, צד גדול
  const Askin = o.Afront * Math.abs(Math.cos(a)) + o.Aside * Math.abs(Math.sin(a));
  const skin = 0.85 * Askin * Math.max(0, bandRadiance(Ts, ...band) - bandRadiance(T0, ...band));
  // חלקי מנוע לוהטים: נראים רק מאחור (קונוס של ±60°), מוסתרים בנחיר שטוח
  const rearness = Math.max(0, -Math.cos(a));
  const hot = o.hotArea * o.hotVis * Math.pow(rearness, 1.5) * 0.9 * bandRadiance(o.Thot, ...band);
  // סילון הפליטה: גז, בעיקר ברצועת CO₂ סביב 4.3 מיקרון; נראה מכל כיוון, הכי הרבה מהצד
  const plumeProj = o.plumeArea * (0.35 + 0.65 * Math.abs(Math.sin(a)));
  const plume = (band === MWIR ? 0.25 : 0.05) * plumeProj * bandRadiance(o.Tplume, ...band);
  return { total: skin + hot + plume, skin, hot, plume, Ts };
}

// טווח גילוי: J·τ(R)/R² = NEI, τ = e^{-βR}. NEI ב-W/m², β ל-ק"מ
export function irstRange(J, NEI = 3e-10, betaKm = 0.08) {
  let lo = 0.1, hi = 500;
  const f = (R) => J * Math.exp(-betaKm * R) / ((R * 1e3) ** 2) - NEI;
  if (f(hi) > 0) return hi;
  if (f(lo) < 0) return 0;
  for (let i = 0; i < 60; i++) { const m = 0.5 * (lo + hi); if (f(m) > 0) lo = m; else hi = m; }
  return 0.5 * (lo + hi);
}
