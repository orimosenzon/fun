// בדיקות לליבת הפיזיקה: node tools/test_core.mjs
import * as PO from '../js/core/po.js';
import { sphereSigmaNorm } from '../js/core/mie.js';
import { COATINGS, layerGamma, salisburyGamma, ETA0 } from '../js/core/ram.js';
import * as C from '../js/core/complex.js';
import { PRESETS } from '../js/core/geometry.js';
import * as M from '../js/core/model.js';
import { aero, waveDragDq, searsHaackDq } from '../js/core/aero.js';
import * as IR from '../js/core/ir.js';

let fail = 0;
const ok = (name, cond, info = '') => { console.log(`${cond ? '✓' : '✗'} ${name} ${info}`); if (!cond) fail++; };
const near = (a, b, tol) => Math.abs(a - b) <= tol;

// 1. לוח שטוח: 4πA²/λ² בניצב, ותבנית sinc² בזווית
const lam = PO.C0 / 10e9, k = 2 * Math.PI / lam;
const P = PO.prepare({ pos: [0.5, 0.5, 0, -0.5, 0.5, 0, -0.5, -0.5, 0, 0.5, -0.5, 0], tri: [0, 1, 2, 0, 2, 3] });
ok('לוח בניצב', near(PO.sigma(P, [0, 0, 1], k), 4 * Math.PI / lam ** 2, 1e-6));
for (const deg of [0.3, 2, 7, 25]) {
  const r = deg * Math.PI / 180, x = k * Math.sin(r);
  const th = 4 * Math.PI / lam ** 2 * Math.cos(r) ** 2 * (Math.sin(x) / x) ** 2;
  const s = PO.sigma(P, [Math.sin(r), 0, Math.cos(r)], k);
  ok(`לוח ב-${deg}°`, Math.abs(PO.dB(s) - PO.dB(th)) < 0.01, `${PO.dB(s).toFixed(2)} מול ${PO.dB(th).toFixed(2)} dBsm`);
}
// 2. גורדון ליד מקרה מנוון: משולש זעיר נותן בדיוק את השטח
const out = [0, 0];
PO.triIntegral(1e-7, 2e-7, -1e-7, 0.5, out);
ok('משולש מנוון', near(out[0], 0.5, 1e-6) && near(out[1], 0, 1e-6));
// רציפות סביב הסף של המקרה המנוון
const a1 = PO.triIntegral(0, 0.99e-4, 0.5e-4, 1, [0, 0]), a2 = PO.triIntegral(0, 1.01e-4, 0.5e-4, 1, [0, 0]);
ok('רציפות בסף', Math.hypot(a1[0] - a2[0], a1[1] - a2[1]) < 1e-6);
// 3. מי: ריילי, שיא התהודה, אופטי
ok('מי ריילי', near(sphereSigmaNorm(0.05) / (9 * 0.05 ** 4), 1, 0.02));
ok('מי שיא', near(sphereSigmaNorm(1.0), 3.64, 0.05), sphereSigmaNorm(1).toFixed(3));
ok('מי אופטי', near(sphereSigmaNorm(40), 1, 0.05), sphereSigmaNorm(40).toFixed(3));
// 4. סליסברי: אפס ברבע גל, מתכת מחזירה הכל
ok('סליסברי', C.abs(salisburyGamma(10e9, ETA0, lam / 4)) < 1e-3);
ok('שכבה אפסית = מתכת', near(C.abs(layerGamma(10e9, [4, -1], [1, 0], 1e-9)), 1, 1e-6));
ok('צבע פריט בולע ב-X', COATINGS.paint.g(10e9) < 0.25, (20 * Math.log10(COATINGS.paint.g(10e9))).toFixed(1) + ' dB');
ok('שום ציפוי לא עובד ב-VHF', Object.values(COATINGS).every((c) => c.g(0.15e9) > 0.5));
// 5. סירס-האק: הנוסחה הסגורה מול הסכום של פורייה
const N = 400, L = 10, Sm = 1, A = [], xs = [];
for (let i = 0; i <= N; i++) { const xi = 2 * i / N - 1; xs.push(L / 2 - i / N * L); A.push(Sm * Math.pow(1 - xi * xi, 1.5)); }
const V = 3 * Math.PI / 16 * Sm * L;
const dq = waveDragDq(xs, A, L), dqT = 9 * Math.PI * Sm * Sm / (2 * L * L);
ok('גרר גל של סירס-האק', near(dq / dqT, 1, 0.02), `${dq.toFixed(4)} מול ${dqT.toFixed(4)}`);
ok('נוסחת הנפח', near(searsHaackDq(V, L) / dqT, 1, 1e-9));
// 6. חום: סטפן-בולצמן מאינטגרל פלאנק
const T = 900, tot = IR.bandRadiance(T, 0.3e-6, 200e-6, 20000) * Math.PI;
ok('סטפן-בולצמן', near(tot / (IR.SIGMA_SB * T ** 4), 1, 0.01));
ok('חימום עור במאך 1.6', near(IR.skinT(1.6, 11), 311, 2), IR.skinT(1.6, 11).toFixed(1) + 'K');
// 7. המטוסים: החמקנים קטנים בהרבה מהקונבנציונלי מלפנים ב-X, ובשום תדר לא כמו ב-X מול VHF
const front = {};
for (const key of Object.keys(PRESETS)) {
  const m = M.build(PRESETS[key]);
  front[key] = {};
  for (const b of ['VHF', 'X']) front[key][b] = PO.dB(PO.sectorMean(M.polarFor(m, b, -5, 720), 0, 30));
  const a = aero(m.ac, m.P);
  console.log(`  ${key}: חזית X ${front[key].X.toFixed(1)} dBsm, VHF ${front[key].VHF.toFixed(1)}, L/D ${a.LDmax.toFixed(1)}, Cnβ ${a.Cnb.toFixed(3)}`);
}
for (const key of ['faceted', 'fighter', 'wing']) {
  ok(`${key} חמקן מלפנים ב-X`, front[key].X < front.conventional.X - 25);
  ok(`${key} נחשף ב-VHF`, front[key].VHF > front[key].X + 10);
}
const wing = M.build(PRESETS.wing); ok('כנף מעופפת לא יציבה כיוונית', aero(wing.ac, wing.P).Cnb < 0);
// 8. זנב אנכי מול מוטה: החזרה מהצד
const side = (cant) => { const p = JSON.parse(JSON.stringify(PRESETS.fighter)); p.tail.cant = cant; p.fus.side = 30; const m = M.build(p); return PO.dB(PO.sectorMean(M.polarFor(m, 'X', 0, 720), 90, 10)); };
const s0 = side(0), s30 = side(30);
ok('הטיית הזנבות מורידה את ההחזרה מהצד (מכ"ם באופק)', s30 < s0 - 6, `${s0.toFixed(1)} → ${s30.toFixed(1)} dBsm`);

console.log(fail ? `\n${fail} בדיקות נכשלו` : '\nהכל עבר');
process.exit(fail ? 1 : 0);
