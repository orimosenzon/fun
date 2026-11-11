// בדיקות ליבה: node tools/test_core.mjs
import { solve, mirrorPose, shiftPose, sub, len, DIM } from '../js/body.js';
import { TECH, KAMAE, REACH, poseAt } from '../js/techniques.js';
import { Match } from '../js/fight.js';
import { makeAI } from '../js/ai.js';

let fails = 0;
const ok = (c, msg) => { if (!c) { fails++; console.log('  ✗ ' + msg); } };
const verbose = process.argv.includes('-v');

// 1. כל תנוחת מפתח ניתנת לביצוע: הרגליים מגיעות, כפות הרגליים מעל הרצפה, אין NaN
console.log('תנוחות מפתח');
for (const id in TECH) {
  const t = TECH[id];
  t.keys.forEach((k, i) => {
    for (const p of [k.pose, mirrorPose(k.pose)]) {
      const J = solve(p);
      ok(J.legStretch <= 1.0005, `${id}[${i}] רגל נמתחת מעבר לאורכה (${J.legStretch.toFixed(3)})`);
      ok(p.fL[1] >= -1e-9 && p.fR[1] >= -1e-9, `${id}[${i}] כף רגל מתחת לרצפה`);
      for (const n of ['knL', 'knR', 'elL', 'elR', 'head', 'handL', 'handR']) {
        ok(J[n].every(Number.isFinite), `${id}[${i}] ${n} NaN`);
        if (t.kind !== 'ko') ok(J[n][1] > 0.05, `${id}[${i}] ${n} בתוך הרצפה (${J[n][1].toFixed(2)})`);
      }
    }
    if (verbose) { const J = solve(k.pose); console.log(`  ${id}[${i}] leg ${J.legStretch.toFixed(2)} armL ${J.armStretchL.toFixed(2)} armR ${J.armStretchR.toFixed(2)}`); }
  });
}

// 2. טכניקה מסתיימת בדיוק בקמאה (מוזזת ומשוקפת לפי הצורך), כדי שלא תהיה קפיצה
console.log('רציפות בסוף טכניקה');
for (const id in TECH) {
  const t = TECH[id];
  if (t.stay) continue;
  const last = solve(t.keys[t.keys.length - 1].pose);
  const want = solve(shiftPose(t.sw ? mirrorPose(KAMAE) : KAMAE, t.advance));
  for (const n of ['pel', 'anL', 'anR', 'wrL', 'wrR', 'head', 'knL', 'knR']) {
    ok(len(sub(last[n], want[n])) < 1e-6, `${id}: ${n} לא חוזר לקמאה`);
  }
}

// 3. רגל תומכת לא מחליקה בזמן בעיטה
console.log('רגל תומכת נעוצה');
for (const [id, foot] of [['mae_c', 'fL'], ['mae_j', 'fL'], ['mawashi_c', 'fL'], ['mawashi_j', 'fL'], ['gyaku_c', 'fR'], ['ushiro', 'fL']]) {
  const t = TECH[id];
  const hitKeys = t.keys.filter((k) => k.t <= t.hit.to + 0.02);
  for (const k of hitKeys) ok(Math.hypot(k.pose[foot][0] - KAMAE[foot][0], k.pose[foot][2] - KAMAE[foot][2]) < 1e-9, `${id}: ${foot} זזה`);
}

console.log('טווחים (מטרים מנקודת המוצא):');
for (const id in REACH) console.log(`  ${id.padEnd(10)} x=${REACH[id].x.toFixed(2)} y=${REACH[id].y.toFixed(2)}`);
ok(REACH.gyaku_j.y > 1.4 && REACH.gyaku_c.y < 1.3, 'גובה ג׳ודאן/צ׳ודאן');
ok(REACH.oi_c.x > REACH.gyaku_c.x + 0.5, 'אוי-צוקי ארוך מגיאקו');

// 4. סימולציות קרב
const DT = 1 / 120;
const idle = () => ({ move: 0, level: 'mid', actions: [], hold: false });
function setup(D) {
  const m = new Match();
  for (let i = 0; i < 500 && m.phase !== 'fight'; i++) m.step(DT, [idle(), idle()]);
  m.f[0].x = -D / 2; m.f[1].x = D / 2; m.events.length = 0;
  return m;
}
function run(m, secs, script) {
  for (let i = 0; i < secs / DT; i++) {
    const t = i * DT;
    m.step(DT, [script(0, t, i) || idle(), script(1, t, i) || idle()]);
  }
}
const act = (a, level = 'mid', hold = false) => ({ move: 0, level, actions: a ? [a] : [], hold });

console.log('קרב');
ok(new Match().phase === 'rei', 'מתחילים בריי');
{
  const m = setup(0.95);
  run(m, 0.8, (w, t, i) => (w === 0 && i === 0 ? act('gyaku') : null));
  ok(m.f[1].hp === 90, `גיאקו-צוקי בטווח פוגע (hp=${m.f[1].hp})`);
  ok(m.events.some((e) => e.type === 'hit' && e.zone === 'C'), 'אירוע פגיעה בצ׳ודאן');
}
{
  const m = setup(1.6);
  run(m, 0.8, (w, t, i) => (w === 0 && i === 0 ? act('gyaku') : null));
  ok(m.f[1].hp === 100, 'גיאקו-צוקי מרחוק מחטיא');
}
{
  const m = setup(1.55);
  run(m, 1.2, (w, t, i) => (w === 0 && i === 0 ? act('oi', 'up') : null));
  ok(m.f[1].hp === 86, `אוי-צוקי ג׳ודאן סוגר מרחק (hp=${m.f[1].hp})`);
  ok(m.events.some((e) => e.type === 'hit' && e.zone === 'J'), 'פגיעה בראש');
  ok(m.f[0].mir === true, 'אוי-צוקי מחליף עמידה');
}
{
  const m = setup(0.95);
  run(m, 0.8, (w, t, i) => (w === 0 ? (i === 12 ? act('gyaku') : null) : (i === 0 ? act('block', 'mid', true) : act(null, 'mid', true))));
  ok(m.f[1].hp === 100, 'אודה-אוקה עוצר גיאקו צ׳ודאן');
  ok(m.events.some((e) => e.type === 'block'), 'אירוע חסימה');
}
{
  const m = setup(0.95);
  run(m, 0.8, (w, t, i) => (w === 0 ? (i === 12 ? act('gyaku') : null) : (i === 0 ? act('block', 'up', true) : act(null, 'up', true))));
  ok(m.f[1].hp === 90, 'אגה-אוקה לא עוצר צ׳ודאן');
}
{
  const m = setup(1.15);
  run(m, 0.9, (w, t, i) => (w === 0 ? (i === 12 ? act('mae') : null) : (i === 0 ? act('block', 'down', true) : act(null, 'down', true))));
  ok(m.f[1].hp === 100, `גדאן-בראי עוצר מאה-גרי (hp=${m.f[1].hp})`);
}
{
  // דה-אי: B מתחיל אוי-צוקי, A פוגע בו בקיזאמי לפני שהגיע
  const m = setup(1.45);
  run(m, 0.9, (w, t, i) => (w === 1 ? (i === 0 ? act('oi') : null) : (i === 14 ? act('kizami', 'up') : null)));
  ok(m.events.some((e) => e.type === 'hit' && e.counter && e.who === 0), 'פגיעת נגד (דה-אי) מזוהה');
  ok(m.f[1].hp === 91, `נזק נגד מוגדל (hp=${m.f[1].hp})`);
}
for (const [a, lv, D] of [['mae', 'mid', 1.2], ['mae', 'up', 1.05], ['mawashi', 'mid', 1.2], ['mawashi', 'up', 1.0], ['yoko', 'mid', 1.3], ['ushiro', 'mid', 1.4], ['kizami', 'up', 1.1]]) {
  const m = setup(D);
  run(m, 1.2, (w, t, i) => (w === 0 && i === 0 ? act(a, lv) : null));
  ok(m.f[1].hp < 100, `${a} ${lv} פוגע ממרחק ${D}`);
  ok(!m.f[0].tech && Math.abs(m.f[0].cur.pel[0]) < 1e-6, `${a}: חזרה לקמאה`);
}
{
  // נוקאאוט
  const m = setup(0.95);
  m.f[1].hp = 5;
  run(m, 3, (w, t, i) => (w === 0 && i === 0 ? act('gyaku') : null));
  ok(m.phase === 'done' && m.winner === 0, 'נוקאאוט מסיים את הקרב');
  ok(m.f[1].J.head[1] < 0.4, 'המפסיד שוכב');
}
{
  // מחשב מול מחשב: הקרב מסתיים, בלי NaN, בתוך הזירה
  let seed = 7; const rnd = () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 4294967296);
  for (let g = 0; g < 4; g++) {
    const m = new Match();
    const ai = [makeAI(g % 3, rnd), makeAI((g + 1) % 3, rnd)];
    let i = 0, minD = 9, hits = 0, blocks = 0;
    for (; i < 240 / DT && m.phase !== 'done'; i++) {
      m.step(DT, [ai[0](m, m.f[0], m.f[1], DT), ai[1](m, m.f[1], m.f[0], DT)]);
      if (m.phase === 'fight') minD = Math.min(minD, m.dist);
      for (const e of m.events) { if (e.type === 'hit') hits++; if (e.type === 'block') blocks++; }
      m.events.length = 0;
    }
    ok(m.phase === 'done', `קרב ${g} מסתיים`);
    ok(Number.isFinite(m.f[0].x) && Number.isFinite(m.f[1].x), 'מיקום תקין');
    ok(minD > 0.3, `לוחמים לא חודרים זה לזה (${minD.toFixed(2)})`);
    console.log(`  קרב ${g}: ${(i * DT).toFixed(0)} שניות, ${hits} פגיעות, ${blocks} חסימות, מנצח ${m.winner}, hp ${m.f[0].hp}/${m.f[1].hp}`);
  }
}

console.log(fails ? `\n${fails} כשלים` : '\nהכול עבר');
process.exit(fails ? 1 : 0);
