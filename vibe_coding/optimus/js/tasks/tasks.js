// המשימות האוטונומיות. כל משימה היא גנרטור שמשתמש באבני הבניין מ-prims.js.
// TaskRunner מקדם את המשימה צעד אחד בכל צעד בקרה, ומחזיק את הסטטוס לתצוגה.
import { walkTo, moveHand, moveHands, releaseHands, grasp, wait, waitUntil, feetMid, objPos, objMass, robotToWorld, handR, bodyPos, touching } from './prims.js';
import { rotZYX, clamp, smooth5 as smooth } from '../ctrl/linalg.js';
import { DT } from '../sim/world.js';
import { STATIONS } from '../sim/scenes.js';

const GRAV = 9.81;

// ---------- לך לנקודה ----------
function* gotoTask(app, ctx) {
  const r = app.robot;
  ctx.status = 'לחצו על הרצפה כדי לבחור יעד';
  ctx.wantsGoal = true;
  while (true) {
    yield* waitUntil(() => ctx.goal, 1e9);
    const g = ctx.goal;
    ctx.status = 'הולך';
    yield* walkTo(r, g[0], g[1], null, { ctx, tol: 0.06 });
    ctx.status = 'הגעתי. אפשר לבחור יעד חדש';
    ctx.goal = null;
  }
}

// ---------- נופף לשלום ----------
function* waveTask(app, ctx) {
  const r = app.robot;
  ctx.status = 'מרים יד';
  r.gaze = app.cameraTarget || null;
  const h = r.loco.heading;
  const base = robotToWorld(r, [0.12, -0.36, 1.58]);
  const R = rotZYX(h - Math.PI / 2, Math.PI, 0);   // אצבעות למעלה, כף יד קדימה
  yield* moveHand(r, 'r', { pos: base, R }, 1.1, { wr: 40 });
  r.hands.set('r', 'spread', 1, 3);
  ctx.status = 'מנופף';
  for (let t = 0; t < 4.5; t += DT) {
    const a = 0.075 * Math.sin(t * 2 * Math.PI * 1.3);
    const p = robotToWorld(r, [0.12, -0.36 + a, 1.58 - Math.abs(a) * 0.3]);
    r.handTargets.r = { pos: p, R: rotZYX(h - Math.PI / 2, Math.PI, -a * 4), w: 90, kp: 220, wr: 40 };
    yield;
  }
  r.hands.set('r', 'open', 0, 3);
  ctx.status = 'מוריד יד';
  const down = robotToWorld(r, [0.04, -0.27, 0.78]);
  yield* moveHand(r, 'r', { pos: down }, 1.0);
  releaseHands(r, ['r']);
  r.gaze = null;
  ctx.status = 'סיימתי';
}

// ---------- העברת ארגז בין שולחנות, בשתי ידיים ----------
function* boxTask(app, ctx) {
  const r = app.robot, name = ctx.args?.box || 'box1';
  const m = objMass(r, name);
  const bid = r.w.bodyId(name);
  const half = r.w.model.geom_size.subarray(3 * r.w.model.body_geomadr[bid], 3 * r.w.model.body_geomadr[bid] + 3);
  const hy = half[1];
  // 1. הליכה אל השולחן
  const A = STATIONS.tableA;
  ctx.status = 'הולך אל השולחן';
  r.gaze = objPos(r, name);
  if (!(yield* walkTo(r, A.approach[0], A.approach[1], A.approach[2], { ctx }))) return fail(ctx);
  // 2. התכופפות קלה והושטת ידיים לצדי הארגז
  ctx.status = 'מושיט ידיים';
  r.lean = 0.28; r.crouch = 0.07;
  yield* wait(0.6);
  const b = objPos(r, name);
  const h = r.loco.heading, c = Math.cos(h), s = Math.sin(h);
  // נקודת האחיזה: מרכז כף היד מול מרכז הדופן. אתר ה-grip נמצא ~8 ס"מ מתחת למרכז כף היד
  const side = (sg, out) => [b[0] - s * sg * (hy + out), b[1] + c * sg * (hy + out), b[2] - 0.05];
  const R = rotZYX(h, -0.25, 0);
  r.hands.set('l', 'flat', 1, 2); r.hands.set('r', 'flat', 1, 2);
  // מסלול: קודם למעלה (מעל גובה השולחן), אחר כך קדימה אל צדי הארגז, ואז למטה
  const up = (p, dz, dx = 0) => [p[0] + c * dx, p[1] + s * dx, p[2] + dz];
  yield* moveHands(r, { l: { pos: up(side(1, 0.1), 0.17, -0.22), R }, r: { pos: up(side(-1, 0.1), 0.17, -0.22), R } }, 1.0);
  yield* moveHands(r, { l: { pos: up(side(1, 0.09), 0.15), R }, r: { pos: up(side(-1, 0.09), 0.15), R } }, 0.8);
  yield* moveHands(r, { l: { pos: side(1, 0.09), R }, r: { pos: side(-1, 0.09), R } }, 0.7);
  ctx.status = 'סוגר על הארגז';
  yield* moveHands(r, { l: { pos: side(1, -0.01), R }, r: { pos: side(-1, -0.01), R } }, 0.7);
  // 3. לחיצה: כל יד דוחפת פנימה, והחיכוך מחזיק את המשקל. μ≈0.9, אז לחיצה של 0.6·mg בכל יד מספיקה בשוליים
  const squeeze = Math.max(40, m * GRAV * 1.1);
  r.squeeze = { F: squeeze, mass: m, k: 0, kw: 0 };
  for (let t = 0; t < 0.5; t += DT) { r.squeeze.k = t / 0.5; yield; }
  // 4. הרמה
  ctx.status = `מרים (${m.toFixed(1)} ק"ג)`;
  const lift = (sg, dz) => { const p = side(sg, -0.02); p[2] += dz; return p; };
  for (let t = 0; t < 1.2; t += DT) {
    r.squeeze.kw = Math.min(1, t / 0.6);
    const dz = 0.12 * Math.min(1, t / 1.2);
    r.handTargets.l = { pos: lift(1, dz), R, w: 120, kp: 200, wr: 25 };
    r.handTargets.r = { pos: lift(-1, dz), R, w: 120, kp: 200, wr: 25 };
    yield;
  }
  if (objPos(r, name)[2] < b[2] + 0.05) return fail(ctx, 'הארגז החליק מהידיים');
  // 5. נשיאה: יעדי הידיים צמודים לגו (במערכת הצירים שלו), כמו אדם שנושא ארגז מול החזה
  ctx.status = 'נושא את הארגז';
  r.lean = 0.05; r.crouch = 0.02;
  const gl = r.toLocal(r.handTargets.l.pos), gr = r.toLocal(r.handTargets.r.pos);
  // מקרבים את הארגז לגוף בהדרגה
  const carryL = [0.4, (gl[1] - gr[1]) / 2, gl[2] + 0.02], carryR = [0.4, -(gl[1] - gr[1]) / 2, gr[2] + 0.02];
  r.handTargets.l = { local: gl.slice(), Rlocal: [0, -0.25, 0], w: 400, kp: 320, wr: 40 };
  r.handTargets.r = { local: gr.slice(), Rlocal: [0, -0.25, 0], w: 400, kp: 320, wr: 40 };
  for (let t = 0; t < 1.0; t += DT) {
    const k = smooth(t / 1.0);
    for (let j = 0; j < 3; j++) { r.handTargets.l.local[j] = gl[j] + (carryL[j] - gl[j]) * k; r.handTargets.r.local[j] = gr[j] + (carryR[j] - gr[j]) * k; }
    yield;
  }
  r.carrying = name;
  const B = STATIONS.tableB;
  r.gaze = null;
  if (!(yield* walkTo(r, B.approach[0], B.approach[1], B.approach[2], { ctx, speed: 0.3, turn: 0.35 }))) return fail(ctx);
  if (objPos(r, name)[2] < 0.6) return fail(ctx, 'הארגז נפל בדרך');
  // 6. הנחה
  ctx.status = 'מניח על השולחן';
  r.lean = 0.15; r.crouch = 0.03;
  const tgtZ = B.top + half[2] + 0.005;
  for (let t = 0; t < 2.5; t += DT) {
    const bz = objPos(r, name)[2];
    if (bz < tgtZ + 0.004) break;
    for (const sg of [1, -1]) { const H = r.handTargets[sg > 0 ? 'l' : 'r']; H.local[2] -= 0.12 * DT; H.local[0] += 0.03 * DT; }
    yield;
  }
  // 7. שחרור
  ctx.status = 'משחרר';
  for (let t = 0; t < 0.4; t += DT) { r.squeeze.k = 1 - t / 0.4; r.squeeze.kw = 1 - t / 0.4; yield; }
  r.squeeze = null; r.loads.l = r.loads.r = null; r.carrying = null;
  for (const sg of [1, -1]) r.handTargets[sg > 0 ? 'l' : 'r'].local[1] += sg * 0.07;
  yield* wait(0.6);
  releaseHands(r);
  r.lean = 0; r.crouch = 0;
  r.hands.set('l', 'open', 0, 2); r.hands.set('r', 'open', 0, 2);
  yield* walkTo(r, B.approach[0] + 0.0, B.approach[1] + 0.3, B.approach[2], { ctx });
  ctx.status = 'הארגז על השולחן';
  ctx.done = true;
}

function fail(ctx, msg = 'נפלתי') { ctx.status = msg; ctx.failed = true; }

// ---------- מיון תאי סוללה 4680 ביד אחת ----------
// אחיזה כמו של פחית: אצבעות קדימה, כף היד לצד, אגודל למעלה, והאצבעות נכרכות סביב הגליל
function* cellsTask(app, ctx) {
  const r = app.robot, C = STATIONS.cells;
  const H = C.approach[2];
  const R = rotZYX(H, -Math.PI / 2, 0);        // "לחיצת יד": אצבעות קדימה, כף יד שמאלה, אגודל למעלה
  const fwd = [Math.cos(H), Math.sin(H)];
  const at = (x, y, z, back = 0) => [x - fwd[0] * back, y - fwd[1] * back, z];
  ctx.status = 'הולך לעמדת תאי הסוללה';
  if (!ctx.args.noWalk && !(yield* walkTo(r, C.pickStand[0], C.pickStand[1], H, { ctx }))) return fail(ctx);
  // כפיפת ברכיים: מורידה את הכתף, כך שהאמה כמעט אופקית והאצבעות לא "נוטפות" אל המגש
  r.lean = 0.06; r.crouch = 0.12;
  const zg = C.top + C.lift + 0.06;            // גובה האחיזה: מעט מעל מרכז התא
  const n = C.src.x.length;
  // עומדים כך שהתא או המשבצת יהיו תמיד באותו מקום ביחס לגוף: 28 ס"מ ימינה, מול הכתף הימנית
  const standFor = (x) => [x - 0.28, C.pickStand[1]];
  for (let i = ctx.args.first || 0; i < (ctx.args.only ? (ctx.args.first || 0) + 1 : n); i++) {
    const name = `cell${i + 1}`;
    const cp = objPos(r, name);
    r.gaze = cp;
    // התא קרוב לכתף ימין? אחרת צעד הצידה
    if (i > 0 || ctx.args.noWalk) {
      ctx.walkLabel = 'חוזר למגש המקור';
      r.crouch = 0.04;
      const st = standFor(cp[0]);
      if (!(yield* walkTo(r, st[0], st[1], H, { ctx, speed: 0.25 }))) return fail(ctx);
      ctx.walkLabel = null;
      r.crouch = 0.12;
      yield* wait(0.4);
    }
    ctx.status = `תא ${i + 1}/${n}: מושיט יד`;
    r.hands.set('r', 'open', 0, 3);
    // גישה עם מרווח של 2.5 ס"מ הצידה, כדי שהאצבעות יעברו ליד התא ולא ידחפו אותו; ואז צמוד אליו
    const right = [Math.sin(H), -Math.cos(H)];
    const side = (o) => [right[0] * o, right[1] * o, 0];
    const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
    // גישה מהצד: היד מגיעה לעומק של התא, 10 ס"מ מימינו, ואז מחליקה הצידה אליו.
    // (גישה מאחור מקפלת את המרפק לאחור, ומשם הבקר המקומי לא מוצא את הדרך קדימה)
    yield* moveHand(r, 'r', { pos: add(at(cp[0], cp[1], zg + 0.1, 0.1), side(0.13)), R }, 0.9);
    yield* moveHand(r, 'r', { pos: add(at(cp[0], cp[1], zg + 0.06, 0.02), side(0.11)), R }, 0.6);
    yield* moveHand(r, 'r', { pos: add(at(cp[0], cp[1], zg, 0.0), side(0.1)), R }, 0.5);
    yield* moveHand(r, 'r', { pos: add(at(cp[0], cp[1], zg, 0.0), side(0.026)), R }, 0.7, { w: 120, kp: 220 });
    yield* moveHand(r, 'r', { pos: add(at(cp[0], cp[1], zg, 0.0), side(-0.006)), R }, 0.35, { w: 120, kp: 220 });
    ctx.status = `תא ${i + 1}/${n}: סוגר אצבעות`;
    yield* grasp(r, 'r', 'wrap', 1, 0.6);
    r.loads.r = [0, 0, -objMass(r, name) * GRAV];
    ctx.status = `תא ${i + 1}/${n}: מרים`;
    yield* moveHand(r, 'r', { pos: at(cp[0], cp[1], zg + 0.1, 0.02), R }, 0.7, { w: 120, kp: 220 });
    if (objPos(r, name)[2] < cp[2] + 0.05) { ctx.status = `תא ${i + 1} החליק. מנסה שוב`; r.loads.r = null; r.hands.release('r', 0.4); yield* wait(0.5); i--; if ((ctx.retries = (ctx.retries || 0) + 1) > 3) return fail(ctx, 'לא הצלחתי לאחוז בתא'); continue; }
    // מחזיקים את היד צמודה לגו, מתיישרים, ונעים הצידה אל מגש היעד
    ctx.status = `תא ${i + 1}/${n}: מעביר`;
    const loc = r.toLocal(r.handTargets.r.pos);
    r.handTargets.r = { local: loc.slice(), Rlocal: [0, -Math.PI / 2, 0], w: 150, kp: 260, wr: 30 };
    const near = [loc[0] - 0.04, loc[1], loc[2] + 0.12];
    for (let t = 0; t < 0.8; t += DT) {
      const k = smooth(t / 0.8);
      for (let j = 0; j < 3; j++) r.handTargets.r.local[j] = loc[j] + (near[j] - loc[j]) * k;
      r.crouch = 0.12 - 0.08 * k;
      yield;
    }
    ctx.walkLabel = `תא ${i + 1}/${n}: צעדים הצידה`;
    const ps = standFor(C.dst.x[i]);
    if (!(yield* walkTo(r, ps[0], ps[1], H, { ctx, speed: 0.25 }))) return fail(ctx);
    ctx.walkLabel = null;
    r.crouch = 0.12;
    const slot = [C.dst.x[i], C.dst.y];
    ctx.status = `תא ${i + 1}/${n}: מניח`;
    // באחיזת כוח התא יושב עמוק בכף היד, כך שצריך להגיע רחוק יותר מאשר בהרמה: מטים את הגו קדימה
    r.lean = 0.12;
    yield* wait(0.3);
    const cur = r.handTargets.r.pos.slice();
    r.handTargets.r = { pos: cur, R, w: 150, kp: 260, wr: 30 };
    // התא לא יושב בדיוק בנקודת האחיזה: מודדים את ההיסט בפועל ומתקנים את היעד לפיו
    const off = () => { const c = objPos(r, name), g = r.wbc.sitePos(r.wbc.ids.r_grip); return [c[0] - g[0], c[1] - g[1], c[2] - g[2]]; };
    let o = off();
    const plate = C.top + C.liftDst + 0.012;
    yield* moveHand(r, 'r', { pos: [slot[0] - o[0], slot[1] - o[1], plate + 0.04 + 0.05 - o[2]], R }, 0.9, { w: 150, kp: 260 });
    yield* wait(0.2);
    o = off();
    // יורדים לאט עד שהתא נוגע במגש (כשהוא מפסיק לרדת יחד עם היד)
    const tgt = [slot[0] - o[0], slot[1] - o[1], plate + 0.04 + 0.05 - o[2]];
    r.handTargets.r = { pos: tgt, R, w: 150, kp: 260, wr: 30 };   // אותו מערך: הירידה משנה את היעד עצמו
    for (let t = 0; t < 2.5; t += DT) {
      tgt[2] -= 0.05 * DT;
      // נוגע במגש? (בודקים כל 10 מ"ש)
      if (Math.round(t / DT) % 5 === 0 && touching(r, name, 'tray_dst')) { yield* wait(0.15); break; }
      yield;
    }
    r.loads.r = null;
    // שחרור כמו של אדם: מרפים חלקית, מרימים את היד ישר למעלה כך שהאצבעות מחליקות לאורך התא, ורק אז פותחים
    { const st = r.hands.state.r; st.target = 0.55; st.speed = 1.5; }
    yield* wait(0.35);
    const up0 = r.handTargets.r.pos.slice();
    yield* moveHand(r, 'r', { pos: [up0[0], up0[1], up0[2] + 0.1], R }, 0.6, { w: 150, kp: 260 });
    r.hands.release('r', 0.4);
    yield* wait(0.4);
    r.hands.set('r', 'open', 0, 2);
    r.lean = 0.06;
    const hp2 = r.handTargets.r.pos;
    yield* moveHand(r, 'r', { pos: add(hp2, side(0.06)), R }, 0.5);
    yield* moveHand(r, 'r', { pos: add(add(hp2, side(0.08)), [0, 0, 0.08]), R }, 0.5);
    // תנוחת "מוכן": היד צמודה לגו, מעל גובה השולחן, כדי שבצעדים הצידה היא לא תיגרר מאחור
    {
      // תנוחת "מוכן": היד חוזרת קרוב לגוף ומעל גובה התאים, ונשארת שם בזמן הצעדים
      const l0 = r.toLocal(r.handTargets.r.pos), ready = [0.16, -0.4, 0.34];
      r.handTargets.r = { local: l0.slice(), Rlocal: [0, -Math.PI / 2, 0], w: 150, kp: 260, wr: 30 };
      for (let t = 0; t < 0.7; t += DT) { const k = smooth(t / 0.7); for (let j = 0; j < 3; j++) r.handTargets.r.local[j] = l0[j] + (ready[j] - l0[j]) * k; yield; }
    }
  }
  releaseHands(r, ['r']);
  r.lean = 0; r.crouch = 0; r.gaze = null;
  yield* wait(0.6);
  const ok = C.dst.x.filter((x, i) => { const p = objPos(r, `cell${i + 1}`); return Math.abs(p[0] - C.dst.tray[0]) < 0.19 && Math.abs(p[1] - C.dst.y) < 0.06 && p[2] > C.top + C.liftDst; }).length;
  ctx.status = `${ok} מתוך ${n} תאים במגש היעד`;
  ctx.done = ok === n;
  if (ok !== n) ctx.failed = true;
}

// ---------- סקוואט: כמה מומנט צריכה הברך ----------
function* squatTask(app, ctx) {
  const r = app.robot;
  r.gaze = null;
  const kn = r.jidx.l_knee;
  for (let rep = 1; rep <= 3; rep++) {
    for (const [depth, label] of [[0.12, 'חצי'], [0.24, 'עמוק']]) {
      ctx.status = `חזרה ${rep}/3: יורד (${label}, ${Math.round(depth * 100)} ס"מ)`;
      r.crouch = depth; r.lean = depth * 0.9;
      // ידיים קדימה לאיזון, כמו בסקוואט של אדם
      r.setJoint('l_shoulder_pitch', -1.2 * depth / 0.24, 6); r.setJoint('r_shoulder_pitch', -1.2 * depth / 0.24, 6);
      for (let t = 0; t < 2.2; t += DT) {
        if (Math.round(t / DT) % 50 === 0) ctx.status = `חזרה ${rep}/3: ${label} · מומנט בברך ${Math.abs(r.tau[kn]).toFixed(0)} נ"מ מתוך 300 · הספק ${Math.round(r.power.Pavg)} ואט`;
        yield;
      }
      r.crouch = 0; r.lean = 0;
      r.setJoint('l_shoulder_pitch', r.postDefault[r.jidx.l_shoulder_pitch], 5); r.setJoint('r_shoulder_pitch', r.postDefault[r.jidx.r_shoulder_pitch], 5);
      ctx.status = `חזרה ${rep}/3: עולה`;
      yield* wait(1.8);
    }
  }
  ctx.status = 'סיימתי 3 חזרות. שימו לב: ככל שהברך כפופה יותר, הזרוע של כוח המשקל ארוכה יותר';
}

// ---------- מבחן דחיפות: עד כמה אפשר לדחוף לפני שנופל ----------
function* pushTestTask(app, ctx) {
  const r = app.robot;
  let best = 0;
  for (let J = 10; J <= 70; J += 6) {
    const ang = Math.sin(J * 7.13) * Math.PI;   // כיוון "אקראי" אבל קבוע
    const dir = [Math.cos(ang), Math.sin(ang)];
    ctx.status = `דחיפה של ${J} נ"ש מכיוון ${Math.round(((ang * 180 / Math.PI) + 360) % 360)}°`;
    const tid = r.wbc.ids.torso, d = r.w.data;
    app.push = { body: tid, F: [dir[0] * J / 0.1, dir[1] * J / 0.1, 0], t: 0.1 };
    app.pushViz = { p: [d.xpos[3 * tid], d.xpos[3 * tid + 1], d.xpos[3 * tid + 2] + 0.15], F: app.push.F, t: 0.8 };
    yield* wait(2.6);
    if (r.fallen) { ctx.status = `נפל בדחיפה של ${J} נ"ש. החזיק עד ${best} נ"ש`; ctx.result = best; return; }
    best = J;
    yield* waitUntil(() => r.loco.mode === 'stand', 3);
    yield* wait(0.6);
  }
  ctx.status = `עמד בכל הדחיפות עד ${best} נ"ש`;
}

// ---------- ריקוד: העברות משקל, צעדים הצידה, סיבובים וידיים ----------
function* danceTask(app, ctx) {
  const r = app.robot;
  r.gaze = app.cameraTarget || null;
  const J = (n, q, w = 8) => r.setJoint(n, q, w);
  const arms = (t, style) => {
    if (style === 'wave') {
      J('l_shoulder_roll', 1.3 + 0.4 * Math.sin(t * 4)); J('r_shoulder_roll', -1.3 - 0.4 * Math.sin(t * 4 + Math.PI));
      J('l_elbow', -0.8 - 0.5 * Math.sin(t * 4 + 1)); J('r_elbow', -0.8 - 0.5 * Math.sin(t * 4 + 1 + Math.PI));
    } else if (style === 'pump') {
      J('l_shoulder_pitch', -1.5 - 0.6 * Math.sin(t * 5)); J('r_shoulder_pitch', -1.5 + 0.6 * Math.sin(t * 5));
      J('l_elbow', -1.6); J('r_elbow', -1.6); J('l_shoulder_roll', 0.3); J('r_shoulder_roll', -0.3);
    } else if (style === 'robot') {
      // "ריקוד רובוט": תנועות חדות, אבל לא קפיצה ביעד (מעבר של 0.15 ש' בין תנוחות)
      const ph = t * 2, k = Math.floor(ph) % 2, f = smooth(Math.min(1, (ph % 1) / 0.3));
      const a = k ? f : 1 - f;
      J('l_shoulder_pitch', -1.5 * a); J('r_shoulder_pitch', -1.5 * (1 - a));
      J('l_elbow', -0.3 - 1.27 * a); J('r_elbow', -0.3 - 1.27 * (1 - a));
    }
  };
  const seq = [
    ['העברת משקל ונדנוד', 4, [0, 0, 0], 'wave', 0.05],
    ['צעדים שמאלה', 3, [0, 0.15, 0], 'pump', 0.03],
    ['צעדים ימינה', 3, [0, -0.15, 0], 'pump', 0.03],
    ['סיבוב במקום', 4.5, [0, 0, 0.55], 'robot', 0.02],
    ['עצירה', 1.5, [0, 0, 0], 'pump', 0.02],
    ['סיבוב חזרה', 4.5, [0, 0, -0.55], 'robot', 0.02],
    ['קדימה ואחורה', 2.5, [0.3, 0, 0], 'wave', 0.0],
    ['', 2.5, [-0.25, 0, 0], 'wave', 0.0],
  ];
  let t = 0;
  for (const [name, dur, cmd, style, crouch] of seq) {
    if (name) ctx.status = `🎵 ${name}`;
    r.loco.setCommand(...cmd);
    for (let k = 0; k < dur; k += DT) {
      t += DT;
      arms(t, style);
      r.crouch = crouch + 0.05 * Math.max(0, Math.sin(t * 6));   // קפיצות קטנות בברכיים לפי הקצב
      yield;
      if (r.fallen) return fail(ctx);
    }
  }
  r.loco.setCommand(0, 0, 0); r.crouch = 0; r.resetPosture(); r.gaze = null;
  yield* waitUntil(() => r.loco.mode === 'stand', 3);
  ctx.status = 'קידה 🙇';
  r.lean = 0.5; yield* wait(1.2); r.lean = 0; yield* wait(1);
  ctx.status = 'סוף הריקוד';
}

// ---------- סיור במעבדה: ניווט בין העמדות ----------
function* tourTask(app, ctx) {
  const r = app.robot;
  // נקודות במסדרונות שבין השולחנות (הניווט כאן בלי מכשולים, אז המסלול נבחר מראש)
  const stops = [[-0.9, 0.0, 'מרכז המעבדה'], [-0.5, 1.0, 'עמדת הסוללות'], [1.8, 1.0, 'הפינה הרחוקה'], [1.8, -1.0, 'קצה המעבדה'], [0.2, -1.0, 'שולחן ב'], [0.05, 0.0, 'נקודת ההתחלה']];
  for (const [x, y, name] of stops) {
    ctx.walkLabel = `מתקרב אל ${name}`;
    ctx.goal = [x, y];
    if (!(yield* walkTo(r, x, y, null, { ctx, tol: 0.08 }))) return fail(ctx);
    ctx.status = `הגעתי אל ${name}`;
    yield* wait(0.8);
  }
  ctx.goal = null;
  ctx.status = 'סיימתי את הסיור';
}

export const TASKS = [
  { key: 'goto', icon: '📍', name: 'לך לנקודה', desc: 'לחצו על הרצפה, והרובוט מתכנן צעדים והולך לשם.', run: gotoTask, manualWalk: false },
  { key: 'wave', icon: '👋', name: 'נופף לשלום', desc: 'יד ימין למעלה, אצבעות פרושות, והראש מסתכל עליכם. שיווי המשקל מפצה על היד המורמת.', run: waveTask },
  { key: 'cells', icon: '🔋', name: 'מיין תאי סוללה', desc: 'שלושה תאי 4680 (355 גרם כל אחד): אחיזה ביד ימין, צעדים הצידה, הנחה במגש היעד. כמו בסרטון של טסלה מ-2024.', run: cellsTask },
  { key: 'squat', icon: '🏋️', name: 'סקוואט', desc: 'כפיפות ברכיים של 12 ו-24 ס"מ. הסתכלו על המומנט בברך ועל ההספק: ככל שהברך כפופה יותר, כוח המשקל פועל עם זרוע ארוכה יותר.', run: squatTask },
  { key: 'pushtest', icon: '🥊', name: 'מבחן דחיפות', desc: 'דחיפות הולכות וגדלות מכיוונים שונים, מ-10 נ"ש ועד שהרובוט נופל. כל דחיפה נבלמת באסטרטגיית הקרסול או בצעד התאוששות.', run: pushTestTask },
  { key: 'dance', icon: '💃', name: 'ריקוד', desc: 'העברות משקל, צעדים הצידה, סיבובים וידיים. כל תנועה של הידיים מזיזה את מרכז המסה, והבקר מפצה עליה.', run: danceTask },
  { key: 'tour', icon: '🧭', name: 'סיור במעבדה', desc: 'ניווט בין חמש נקודות: פנייה, הליכה, עצירה. הצעדים המתוכננים מופיעים על הרצפה בצהוב.', run: tourTask },
  { key: 'box', icon: '📦', name: 'העבר ארגז', desc: 'הולך לשולחן, תופס ארגז של 2.2 ק"ג בשתי ידיים בכוח לחיצה בלבד, ומעביר אותו לשולחן השני.', run: boxTask },
];

export class TaskRunner {
  constructor(app) {
    this.app = app;
    this.active = null; this.gen = null; this.ctx = null;
  }
  get goal() { return this.ctx?.goal || null; }
  get wantsGoal() { return !!this.ctx?.wantsGoal; }
  get status() { return this.ctx?.status || ''; }
  setGoal(g) { if (this.ctx) this.ctx.goal = g; }
  start(key, args = {}) {
    const T = TASKS.find(t => t.key === key);
    if (!T) return;
    this.stop();
    this.active = T;
    this.ctx = { status: 'מתחיל', args, t0: this.app.world.t };
    this.gen = T.run(this.app, this.ctx);
  }
  stop() {
    const r = this.app.robot;
    this.active = null; this.gen = null; this.ctx = null;
    if (r) { releaseHands(r); r.squeeze = null; r.lean = 0; r.crouch = 0; r.gaze = null; r.carrying = null; r.loco.setCommand(0, 0, 0); r.hands.set('l', 'open', 0, 2); r.hands.set('r', 'open', 0, 2); }
  }
  update() {
    if (!this.gen) return;
    if (this.app.robot.fallen) { this.ctx.status = 'נפלתי. R לאיפוס'; this.ctx.failed = true; this.gen = null; return; }
    const res = this.gen.next();
    if (res.done) { this.gen = null; if (!this.ctx.failed) this.ctx.done = true; }
  }
}
