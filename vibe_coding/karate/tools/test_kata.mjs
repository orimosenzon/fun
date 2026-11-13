// בדיקות קאטה: node tools/test_kata.mjs [-v] [id]
import { solve, sub, len, solidify, distToSeg } from '../js/body.js';
import { buildKata, timeline, kataPoseAt, moveLabel, stanceLabel } from '../js/kata.js';
import { KATAS } from '../js/katas.js';
let fails = 0;
const ok = (c, msg) => { if (!c) { fails++; console.log('  ✗ ' + msg); } };
const v = process.argv.includes('-v'), only = process.argv.slice(2).find((a) => !a.startsWith('-'));
for (const k of KATAS) {
  if (only && k.id !== only) continue;
  const K = buildKata(k), TL = timeline(K);
  const last = K.steps[K.steps.length - 1].pose;
  const drift = Math.hypot(last.pel[0] - K.open.pel[0], last.pel[2] - K.open.pel[2]);
  console.log(`${k.he}: ${K.count} תנועות, ${TL.total.toFixed(0)} שניות, סיום במרחק ${drift.toFixed(2)} מ' מההתחלה`);
  ok(drift < (k.drift ?? 1.3), `${k.id}: הקאטה לא חוזרת לנקודת ההתחלה (${drift.toFixed(2)})`);
  K.steps.forEach((s, i) => {
    for (const [nm, p] of [['pose', s.pose], ['prep', s.prep], ['air', s.air], ['post', s.post]]) {
      if (!p) continue;
      const J = solve(p);
      ok(J.legStretch <= 1.02, `${k.id}[${i + 1}] ${nm}: רגל נמתחת (${J.legStretch.toFixed(3)})`);
      for (const n of ['knL', 'knR', 'elL', 'elR', 'head', 'handL', 'handR', 'anL', 'anR']) {
        ok(J[n].every(Number.isFinite), `${k.id}[${i + 1}] ${nm} ${n} NaN`);
        if (!['fuse', 'hiza'].includes(s.m.st)) ok(J[n][1] > 0.04, `${k.id}[${i + 1}] ${nm} ${n} בתוך הרצפה (${J[n][1].toFixed(2)})`);
      }
    }
    if (v) { const l = moveLabel(s), st = stanceLabel(s); console.log(`  ${String(s.count).padStart(2)} ${s.m.sub ? '·' : ' '} [${(s.pose.pel[0]).toFixed(2)},${s.pose.pel[2].toFixed(2)}] hip ${(s.state.hip * 180 / Math.PI).toFixed(0)} ${l.he} | ${st.he}${s.m.kiai ? ' !קיאי' : ''}`); }
  });
  // רציפות: אין קפיצות בין פריימים עוקבים
  const crossAt = []; let prev = null, maxJ = 0, at = 0, pen = 0, worst = 9, worstAt = 0, cross = 0;
  for (let t = 0; t < TL.total; t += 1 / 60) {
    const J = solve(solidify(kataPoseAt(K, TL, t, {}).pose, 4, 0.12));
    // גוף מוצק: שורש כף היד לא בתוך הגו, והאמות לא חוצות זו את זו
    for (const sd of ['L', 'R']) {
      const dSpine = distToSeg(J['wr' + sd], J.pel, J.chest);
      if (dSpine < 0.11) { pen++; if (dSpine < worst) { worst = dSpine; worstAt = t; } }
    }
    for (let q = 0; q <= 4; q++) { const pt = [0, 1, 2].map((i) => J.elL[i] + (J.wrL[i] - J.elL[i]) * q / 4); const d = distToSeg(pt, J.elR, J.wrR); if (d < 0.06) { cross++; if (v) crossAt.push(t.toFixed(1)); break; } }
    if (prev) for (const n of ['pel', 'anL', 'anR', 'wrL', 'wrR']) { const d = len(sub(J[n], prev[n])); if (d > maxJ) { maxJ = d; at = t; } }
    prev = J;
  }
  ok(maxJ < 0.6, `${k.id}: קפיצה של ${maxJ.toFixed(2)} מ' בפריים אחד בזמן ${at.toFixed(2)}`);
  ok(pen === 0, `${k.id}: יד בתוך הגו ב-${pen} פריימים (הכי עמוק ${worst.toFixed(3)} בזמן ${worstAt.toFixed(2)})`);
  ok(cross <= 25, `${k.id}: אמות חוצות זו את זו ב-${cross} פריימים`);
  if (v && crossAt.length) console.log('  חציית אמות בזמנים: ' + [...new Set(crossAt)].join(' '));
  if (v) console.log(`  תזוזה מרבית בפריים: ${maxJ.toFixed(3)}`);
}
console.log(fails ? `\n${fails} כשלים` : '\nהכול עבר');
process.exit(fails ? 1 : 0);
