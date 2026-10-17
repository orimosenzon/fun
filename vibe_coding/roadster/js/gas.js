// תרמודינמיקה של אוויר דחוס: מכל COPV ומנועי גז קר (cold gas thrusters).
// אותו קוד משמש את הסימולטור ואת המרכז המקצועי.

export const AIR = {
  R: 287.05,        // J/(kg·K)
  gamma: 1.4,
  cp: 1004.7,
  cv: 717.6,
  pAmb: 101325,     // Pa, גובה פני הים
  TAmb: 293.15,     // K
  rhoAmb: 1.204,
};

// מקדם הדחיסות של אוויר בטמפרטורת החדר: Z ≈ 1 + b·p.
// בלחץ 379 בר (5,500 psi) Z≈1.19, כמו בטבלאות NIST/CoolProp. בלחצים כאלה אוויר
// כבר לא מתנהג כגז אידיאלי: יש בו פחות מסה ממה שהנוסחה הפשוטה מנבאת.
const ZB = 5.0e-9;
export function Z(p) { return 1 + ZB * p; }

// לחץ במכל מתוך מסה, נפח וטמפרטורה: pV = Z(p)·m·R·T → p = mRT / (V − b·mRT)
export function tankPressure(m, V, T) {
  const mrt = m * AIR.R * T;
  return mrt / Math.max(V - ZB * mrt, 1e-6);
}
// מסה במכל בלחץ וטמפרטורה נתונים
export function tankMass(p, V, T) { return p * V / (Z(p) * AIR.R * T); }

const G = AIR.gamma;
// Γ: גורם הזרימה החנוקה (choked flow), √γ·(2/(γ+1))^((γ+1)/(2(γ−1)))
export const GAMMA_FLOW = Math.sqrt(G) * Math.pow(2 / (G + 1), (G + 1) / (2 * (G - 1)));

// יחס שטחים A/A* כפונקציה של מספר מאך
export function areaRatio(M) {
  return (1 / M) * Math.pow((2 / (G + 1)) * (1 + (G - 1) / 2 * M * M), (G + 1) / (2 * (G - 1)));
}
// מאך ביציאה העל-קולית עבור יחס שטחים ε (ניוטון-רפסון)
export function exitMach(eps) {
  let M = 2 + Math.log(eps);
  for (let i = 0; i < 30; i++) {
    const f = areaRatio(M) - eps;
    const h = 1e-5, d = (areaRatio(M + h) - areaRatio(M - h)) / (2 * h);
    M -= f / d;
    if (M < 1.0001) M = 1.0001;
    if (Math.abs(f) < 1e-9) break;
  }
  return M;
}

// נחיר דה-לבאל (de Laval) קבוע. מחשב דחף וספיקת מסה לפי לחץ התא p0 וטמפרטורה T0.
export class Nozzle {
  constructor({ throatArea, eps = 4, cd = 0.97, eta = 0.96 }) {
    this.At = throatArea;
    this.eps = eps;
    this.Ae = throatArea * eps;
    this.cd = cd;     // מקדם פריקה של הגרון
    this.eta = eta;   // יעילות הנחיר (חיכוך, אובדן הטיה)
    this.Me = exitMach(eps);
    this.peRatio = Math.pow(1 + (G - 1) / 2 * this.Me * this.Me, -G / (G - 1));
    this.TeRatio = 1 / (1 + (G - 1) / 2 * this.Me * this.Me);
  }
  // מחזיר {mdot, F, pe, Te, ve, regime}
  perf(p0, T0, pa = AIR.pAmb) {
    const out = { mdot: 0, F: 0, pe: pa, Te: T0, ve: 0, regime: 'off' };
    if (p0 <= pa * 1.02) return out;
    if (p0 < pa * 1.893) {
      // זרימה תת-קולית: הגרון לא חנוק. הערכה איזנטרופית פשוטה.
      const pr = pa / p0;
      const M = Math.sqrt(2 / (G - 1) * (Math.pow(pr, -(G - 1) / G) - 1));
      const T = T0 / (1 + (G - 1) / 2 * M * M);
      const v = M * Math.sqrt(G * AIR.R * T);
      const rho = pa / (AIR.R * T);
      out.mdot = this.cd * rho * v * this.At;
      out.ve = v; out.F = out.mdot * v; out.Te = T; out.regime = 'subsonic';
      return out;
    }
    const mdot = this.cd * this.At * p0 * GAMMA_FLOW / Math.sqrt(Z(p0) * AIR.R * T0);
    const Te = T0 * this.TeRatio;
    const ve = this.Me * Math.sqrt(G * AIR.R * Te) * Math.sqrt(this.eta);
    let pe = p0 * this.peRatio;
    let F = mdot * ve + (pe - pa) * this.Ae;
    let regime = pe > pa * 1.05 ? 'under' : pe < pa * 0.95 ? 'over' : 'ideal';
    // קריטריון סאמרפילד: בהתפשטות-יתר חזקה הזרם ניתק מהדופן ולא מפסידים את כל הלחץ
    if (pe < 0.35 * pa) {
      regime = 'separated';
      F = Math.max(F, mdot * ve * 0.85);
    }
    out.mdot = mdot; out.F = Math.max(0, F); out.pe = pe; out.Te = Te; out.ve = ve; out.regime = regime;
    return out;
  }
  // גודל הגרון שנותן דחף F בלחץ ובטמפרטורה נתונים
  static sizeFor(F, p0, T0, eps = 4) {
    const probe = new Nozzle({ throatArea: 1e-4, eps });
    const f1 = probe.perf(p0, T0).F;
    return 1e-4 * F / f1;
  }
}

// מכל אוויר (אחד או יותר, מאוחדים): מסה, טמפרטורה, חילופי חום עם הדופן
export class AirTank {
  constructor({ volume, pFill, T = AIR.TAmb, hA = 1200 }) {
    this.V = volume;
    this.pFill = pFill;
    this.hA = hA;               // W/K, הסעה מדופן המכל לגז
    this.Twall = AIR.TAmb;
    this.T = T;
    this.m = volume > 0 ? tankMass(pFill, volume, T) : 0;
    this.mFull = this.m;
    this.p = volume > 0 ? pFill : 0;
  }
  get fraction() { return this.mFull > 0 ? this.m / this.mFull : 0; }
  // הוצאת ספיקה כוללת mdot במשך dt (ניפוח אדיאבטי + חום מהדופן)
  // m·cv·dT/dt = −ṁ·R·T + hA·(Tw − T)
  step(mdotOut, dt, mdotIn = 0, Tin = AIR.TAmb) {
    if (this.V <= 0) return;
    const m = Math.max(this.m, 1e-3);
    let dT = (-mdotOut * AIR.R * this.T + this.hA * (this.Twall - this.T)) / (m * AIR.cv);
    if (mdotIn > 0) dT += mdotIn * (AIR.cp * Tin - AIR.cv * this.T) / (m * AIR.cv);
    this.T = Math.max(60, this.T + dT * dt);
    this.m = Math.max(0, this.m + (mdotIn - mdotOut) * dt);
    this.p = this.m > 1e-4 ? tankPressure(this.m, this.V, this.T) : 0;
  }
  refillTo(pTarget) { this.T = AIR.TAmb; this.m = tankMass(pTarget, this.V, this.T); this.p = pTarget; }
  // אנרגיה: עבודה מועילה מרבית (אקסרגיה) בהתפשטות איזותרמית לאטמוספרה, J
  exergy() {
    if (this.m <= 0) return 0;
    const R = AIR.R, T0 = AIR.TAmb, p0 = AIR.pAmb;
    return this.m * R * T0 * (Math.log(this.p / p0) - 1 + p0 / this.p);
  }
}

export const PSI = 6894.757;
export const BAR = 1e5;
