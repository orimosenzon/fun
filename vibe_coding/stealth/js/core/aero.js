// אווירודינמיקה בקירוב הנדסי (שיטות של ריימר ו-DATCOM): כמה החמקנות עולה בביצועים
const D2R = Math.PI / 180;

// גרר גל בגוף דק (פון קרמן): S'(x)=Σ A_n sin nθ, x=(L/2)(1-cos θ) ⇒ D/q = (π/4) Σ n A_n²
export function waveDragDq(xs, A, L) {
  const N = xs.length - 1; // xs יורד מהאף לזנב, באורך L
  const Sx = (x) => { // אינטרפולציה של A לפי מרחק מהאף
    const t = Math.max(0, Math.min(N, x / L * N)), i = Math.min(N - 1, Math.floor(t)), f = t - i;
    return A[i] * (1 - f) + A[i + 1] * f;
  };
  const M = 240, nMax = 24, coef = new Float64Array(nMax + 1);
  for (let k = 0; k < M; k++) {
    const th = (k + 0.5) * Math.PI / M, x = L / 2 * (1 - Math.cos(th));
    const dx = L * 1e-3;
    const dS = (Sx(Math.min(L, x + dx)) - Sx(Math.max(0, x - dx))) / (Math.min(L, x + dx) - Math.max(0, x - dx));
    for (let n = 1; n <= nMax; n++) coef[n] += dS * Math.sin(n * th) * (Math.PI / M) * 2 / Math.PI;
  }
  let s = 0; for (let n = 1; n <= nMax; n++) s += n * coef[n] * coef[n];
  return Math.PI / 4 * s;
}

// סירס-האק: הגוף עם גרר הגל המינימלי לנפח ואורך נתונים
export const searsHaackDq = (V, L) => 128 * V * V / (Math.PI * L ** 4);

export function aero(ac, P) {
  const p = ac.p, w = p.wing, S = ac.area.Sref, b = w.span, AR = b * b / S;
  const lam = w.te === 'saw' ? 0.15 : w.ct / w.cr;
  const tLE = Math.tan(w.sweep * D2R);
  const tC2 = tLE - (4 / AR) * 0.5 * (1 - lam) / (1 + lam);
  const M = 0.85, beta = Math.sqrt(1 - M * M), eta = 0.95;
  const CLa = 2 * Math.PI * AR / (2 + Math.sqrt(4 + (AR * beta / eta) ** 2 * (1 + tC2 * tC2 / (beta * beta))));
  const e = w.sweep > 30
    ? 4.61 * (1 - 0.045 * AR ** 0.68) * Math.cos(w.sweep * D2R) ** 0.15 - 3.1
    : 1.78 * (1 - 0.045 * AR ** 0.68) - 0.64;
  let Swet = 0; for (let t = 0; t < P.nt; t++) Swet += P.A[t];
  // צלעות חדות ופאות שטוחות מוסיפות גרר צורה ומפרידות זרימה: קנס קטן לפי זווית הצד
  const facetPenalty = p.fus.section === 'chine' ? 1 + 0.004 * p.fus.side : 1;
  const CD0 = 0.0032 * Swet / S * facetPenalty;
  const LDmax = 0.5 * Math.sqrt(Math.PI * AR * Math.max(0.3, e) / CD0);
  const L = p.L;
  const Dq = waveDragDq(ac.area.xs, ac.area.A, L);
  const DqSH = searsHaackDq(ac.area.V, L);
  // יציבות כיוונית: זנבות (מוטים ⇒ cos²) מול הגוף המערער
  const tl = p.tail, nf = tl.type === 'none' ? 0 : tl.type === 'single' ? 1 : 2;
  const cant = tl.type === 'twin' ? tl.cant * D2R : 0;
  const Sfin = 0.5 * (tl.cr + tl.ct) * tl.h;
  const xcg = L / 2 - (w.x0 * L + 0.45 * w.cr); // בערך ברבע המיתר הממוצע של הכנף
  const xfin = L / 2 - tl.xFrac * L - 0.25 * (tl.cr + tl.ct) / 2 - 0.4 * tl.h * Math.tan(tl.sweep * D2R);
  const lv = Math.max(0, xcg - xfin);
  const CnTail = 2.0 * 0.8 * nf * Sfin * Math.cos(cant) ** 2 / S * lv / b;
  const CnFus = -0.6 * ac.area.V / (S * b);
  const Cnb = CnTail + CnFus;
  return { S, b, AR, CLa, e, Swet, CD0, LDmax, Dq, DqSH, waveRatio: Dq / DqSH, CdWave: Dq / S, Cnb, CnTail, CnFus, V: ac.area.V, lam };
}
