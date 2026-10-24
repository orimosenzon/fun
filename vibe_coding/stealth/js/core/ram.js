// חומרים בולעי מכ"ם (RAM) כקווי תמסורת, פגיעה ניצבת, מוסכמה e^{+jωt} (הפסד = חלק מדומה שלילי)
import * as C from './complex.js';
import { C0 } from './po.js';

export const ETA0 = 376.730313;

// μr(f) של פריט (Ferrite) במודל רלקסציה: μ = 1 + (μs-1)/(1 + j f/fr)
export function ferriteMu(fGHz, mus = 18, frGHz = 0.6) {
  return C.add([1, 0], C.div([mus - 1, 0], [1, fGHz / frGHz]));
}

// שכבה על מתכת (דלנבך): Zin = j η tan(k1 d), Γ = (Zin-η0)/(Zin+η0)
export function layerGamma(fHz, epsr, mur, d) {
  const k0 = 2 * Math.PI * fHz / C0;
  const n = C.sqrt(C.mul(epsr, mur));
  const k1 = C.scale(n, k0);
  const eta = C.scale(C.sqrt(C.div(mur, epsr)), ETA0);
  const zin = C.mul([0, 1], C.mul(eta, C.tan(C.scale(k1, d))));
  return C.div(C.sub(zin, [ETA0, 0]), C.add(zin, [ETA0, 0]));
}

// מסך סליסברי: יריעה התנגדותית Rs במרחק d מעל מתכת (מרווח אוויר או חומר ε)
export function salisburyGamma(fHz, Rs, d, epsSpacer = 1) {
  const k0 = 2 * Math.PI * fHz / C0, n = Math.sqrt(epsSpacer);
  const zsh = [0, ETA0 / n * Math.tan(k0 * n * d)]; // קצר בקצה המרווח, נראה כמו j η tan(kd)
  const zin = C.div(C.mul([Rs, 0], zsh), C.add([Rs, 0], zsh));
  return C.div(C.sub(zin, [ETA0, 0]), C.add(zin, [ETA0, 0]));
}

// הציפויים שבתפריט. מחזיר |Γ| (גודל, 0..1) בתדר נתון
export const COATINGS = {
  none: { name: 'מתכת חשופה', kgm2: 0, g: () => 1 },
  paint: {
    name: 'צבע פריט (MagRAM) 1 מ"מ', kgm2: 5.5, note: 'מכוון ל-6–10 גיגה-הרץ',
    g: (f) => C.abs(layerGamma(f, [8, -0.6], ferriteMu(f / 1e9, 40, 1.5), 1.0e-3)),
  },
  thick: {
    name: 'פריט עבה 2 מ"מ', kgm2: 11, note: 'מכוון לתדרים נמוכים יותר (S)',
    g: (f) => C.abs(layerGamma(f, [8, -0.6], ferriteMu(f / 1e9, 40, 0.6), 2e-3)),
  },
  foam: {
    name: 'קצף פחמן 16 מ"מ', kgm2: 1.3, note: 'קל ורחב סרט, אבל עבה',
    g: (f) => C.abs(layerGamma(f, [1.5, -1.0], [1, 0], 16e-3)),
  },
  salisbury: {
    name: 'מסך סליסברי (λ/4 ב-10 גיגה)', kgm2: 2.5, note: 'עמוק מאוד, אבל רק סביב תדר אחד',
    g: (f) => C.abs(salisburyGamma(f, ETA0, 7.5e-3, 1)),
  },
};

export const gammaDb = (g) => 20 * Math.log10(Math.max(g, 1e-6));
