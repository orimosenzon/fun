// מחבר את הכל: פרמטרים → רשת → ציפוי, פתחים ותא טייס → σ לכל תדר
import { buildAircraft, PART } from './geometry.js';
import * as PO from './po.js';
import { COATINGS } from './ram.js';

export const BANDS = {
  VHF: { f: 0.15e9, name: 'VHF', label: '150 מגה-הרץ', use: 'התרעה מוקדמת (ארוך גל, λ=2 מ\')' },
  L: { f: 1.3e9, name: 'L', label: '1.3 גיגה-הרץ', use: 'חיפוש ארוך טווח (λ=23 ס"מ)' },
  S: { f: 3e9, name: 'S', label: '3 גיגה-הרץ', use: 'חיפוש ובקרה (λ=10 ס"מ)' },
  X: { f: 10e9, name: 'X', label: '10 גיגה-הרץ', use: 'בקרת אש ומכ"ם קרב (λ=3 ס"מ)' },
};

// מכ"מים לדוגמה: R1 = טווח הגילוי מול מטרה של 1 מ"ר. R = R1·σ^¼
export const RADARS = {
  ew_vhf: { name: 'התרעה מוקדמת VHF', band: 'VHF', R1: 170, h: 15, guide: false },
  search_s: { name: 'מכ"ם חיפוש S', band: 'S', R1: 140, h: 20, guide: false },
  sam_x: { name: 'מכ"ם בקרת אש X (טילי קרקע-אוויר)', band: 'X', R1: 80, h: 12, guide: true },
  fighter: { name: 'מכ"ם מטוס קרב X', band: 'X', R1: 85, h: 0, guide: true },
};

export const INLETS = {
  pitot: { name: 'פתח ישיר (רואים את המדחס)', s0: 7, half: 35, ram: false },
  sduct: { name: 'צינור S בלי ציפוי', s0: 1.2, half: 45, ram: false },
  sduct_ram: { name: 'צינור S מצופה בולע', s0: 1.2, half: 45, ram: true },
  grille: { name: 'רשת מתחת לתדר הקטעון', s0: 7, half: 35, grille: 0.010, depth: 0.02 },
};
export const CANOPIES = {
  clear: { name: 'שקופה (רואים את תא הטייס)', s0: 1.5 },
  coated: { name: 'מצופה שכבת מתכת דקה (ITO)', s0: 0.0015 },
};
export const NOZZLES = {
  round: { name: 'עגול וחשוף', s0: 3, half: 35, cap: 1 },
  slot: { name: 'שטוח ומוסתר (נחיר דו-ממדי)', s0: 0.15, half: 50, cap: 0.25 },
};
export const ANTENNAS = {
  dish: { name: 'צלחת מכנית חשופה', },
  tilted: { name: 'מערך מוטה + כיפה בוררת תדרים', },
};

// דעיכת גל מתחת לתדר הקטעון בתא ריבועי ברוחב a: α = (2π/λc)√(1-(f/fc)²)
export function grilleAtten(f, a, depth) {
  const fc = PO.C0 / (2 * a);
  if (f >= fc) return 1;
  const alpha = (2 * Math.PI * fc / PO.C0) * Math.sqrt(1 - (f / fc) ** 2);
  return Math.exp(-4 * alpha * depth); // σ דועכת פי e^{-4αd}: הלוך-חזור, בעוצמה
}

const angFrom = (r, d) => Math.acos(Math.max(-1, Math.min(1, r[0] * d[0] + r[1] * d[1] + r[2] * d[2]))) * 180 / Math.PI;
const gauss = (ang, half) => Math.exp(-((ang / half) ** 2));

export function extras(p, f, opts = {}) {
  const L = p.L, out = [], lin = COATINGS.paint.g(f);
  const lam = PO.C0 / f;
  // פתחי אוויר: בצדי הגוף, כשליש מהאורך מהאף
  const inl = INLETS[p.inlet] || INLETS.pitot;
  let s0 = inl.s0;
  if (inl.ram) s0 *= lin ** 8; // ארבע החזרות בצינור, כל אחת מקטינה את העוצמה פי Γ²
  if (inl.grille) s0 = s0 * grilleAtten(f, inl.grille, inl.depth) + 0.0008;
  // בגל ארוך (VHF) הפתח קטן ביחס לגל, והחלל כבר לא מתנהג כמראה פנימית
  if (lam > 1) s0 *= 0.3;
  const xi = L / 2 - 0.32 * L, yi = p.fus.W * 0.95;
  for (const sg of [1, -1]) out.push({ p: [xi, sg * yi, 0], amp: (r) => Math.sqrt(s0 * gauss(angFrom(r, [1, 0, 0]), inl.half)) });
  // תא טייס
  const can = CANOPIES[p.canopy] || CANOPIES.clear;
  const cdir = [0.6, 0, 0.8];
  out.push({ p: [L / 2 - 0.22 * L, 0, p.fus.HU], amp: (r) => { const c = r[0] * cdir[0] + r[2] * cdir[2]; return c > 0 ? Math.sqrt(can.s0) * c : 0; } });
  // פליטה: חלל הטורבינה מאחור
  const nz = NOZZLES[p.nozzleType] || NOZZLES.round;
  out.push({ p: [-L / 2, 0, 0], amp: (r) => Math.sqrt(nz.s0 * gauss(angFrom(r, [-1, 0, 0]), nz.half)) });
  // אנטנת המכ"ם באף: בתוך התחום שלה (X) היא מראה גדולה
  const ant = p.antenna || (p.coat === 'none' ? 'dish' : 'tilted');
  const inBand = f > 7e9 && f < 13e9;
  if (ant === 'dish') {
    const s = inBand ? 6 : 2;
    out.push({ p: [L / 2 - 0.06 * L, 0, 0], amp: (r) => Math.sqrt(s * gauss(angFrom(r, [1, 0, 0]), 28)) });
  } else {
    // מערך שטוח מוטה 25° למעלה: ההחזרה הספקולרית בורחת מעל האופק. מחוץ לתחום הכיפה חוסמת
    const D = 0.85, Aap = Math.PI * D * D / 4, smax = 4 * Math.PI * Aap * Aap / (lam * lam);
    const nrm = [Math.cos(25 * Math.PI / 180), 0, Math.sin(25 * Math.PI / 180)];
    out.push({
      p: [L / 2 - 0.06 * L, 0, 0], amp: (r) => {
        if (!inBand) return Math.sqrt(0.0008);
        const th = angFrom(r, nrm) * Math.PI / 180; if (th > Math.PI / 2) return 0;
        const x = 2 * Math.PI / lam * D * Math.sin(th);
        const sinc = Math.abs(x) < 1e-6 ? 1 : Math.sin(x) / x;
        return Math.sqrt(smax * 0.002) * Math.abs(sinc) * Math.cos(th) + Math.sqrt(0.0004);
      },
    });
  }
  // דלתות תא חימוש פתוחות: חלל גדול שנראה מלמטה
  if (opts.bayOpen) out.push({ p: [0, 0, -p.fus.HL], amp: (r) => (r[2] < 0 ? Math.sqrt(12 * (-r[2]) ** 0.7) : 0) });
  // חימוש חיצוני: טילים ומכלים על הכנף
  if (opts.external) for (const sg of [1, -1]) out.push({ p: [0, sg * p.wing.span * 0.25, -0.3], amp: (r) => Math.sqrt(1.5) * (0.4 + 0.6 * Math.abs(r[1])) });
  return out;
}

export function gammaFor(p, f) {
  const g = COATINGS[p.coat] ? COATINGS[p.coat].g(f) : 1;
  const nz = NOZZLES[p.nozzleType] || NOZZLES.round;
  const arr = new Float64Array(8).fill(g);
  arr[PART.NOZZLE] = nz.cap;
  return arr;
}

export function build(p) {
  const ac = buildAircraft(p);
  const P = PO.prepare(ac);
  return { ac, P };
}

export function kOf(f) { return 2 * Math.PI * f / PO.C0; }

export function polarFor(m, bandKey, elDeg, n = 1440, opts = {}) {
  const f = BANDS[bandKey].f, k = kOf(f), p = m.ac.p;
  return PO.polar(m.P, k, elDeg, n, gammaFor(p, f), extras(p, f, opts), { edgeGamma: p.coat !== 'none' ? 0.5 : 1 });
}

export const detectRange = (R1, sigma) => R1 * Math.pow(Math.max(sigma, 1e-8), 0.25);
// אופק מכ"ם (רדיוס כדור הארץ האפקטיבי 4/3): d[ק"מ] = 4.12(√h_r + √h_t), גבהים במטרים
export const horizonKm = (hr, ht) => 4.12 * (Math.sqrt(Math.max(0, hr)) + Math.sqrt(Math.max(0, ht)));
