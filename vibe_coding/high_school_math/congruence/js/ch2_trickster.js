'use strict';
/* פרק 2: משחק הרמאי. בוחרים 3 נתונים, והרמאי מנסה לבנות משולש אחר עם אותם נתונים. */

App.inits.ch2 = function () {
  const root = document.getElementById('trickster');
  root.innerHTML = `
    <svg class="board" viewBox="0 0 720 330"></svg>
    <div class="chips" id="tr-chips"></div>
    <div class="controls">
      <button class="btn primary" id="tr-go" disabled>😈 תורך, רמאי!</button>
      <button class="btn ghost" id="tr-clear">נקה בחירה</button>
    </div>
    <div id="tr-result"></div>
    <h3 class="sub">הלוח שלך: 7 סוגים של בחירות</h3>
    <div class="badges" id="tr-badges"></div>
    <div id="tr-final"></div>`;
  const $ = s => root.querySelector(s);
  const svg = root.querySelector('svg');
  const U = 34;
  const COLORS = ['#db2777', '#0d9488', '#7c3aed'];

  // המשולש של המשחק (ביחידות, y למעלה): ∠A = 40°, AB = 7, BC = 5.
  // בחרתי אותו כך שכל המקרים המעניינים יופיעו בו.
  const alpha = 40 * RAD, cc = 7, aa = 5;
  const bb = cc * Math.cos(alpha) + Math.sqrt(aa * aa - (cc * Math.sin(alpha)) ** 2);
  const SEC = { A: [0, 0], B: [cc, 0], C: [bb * Math.cos(alpha), bb * Math.sin(alpha)] };

  const SIDE_V = { a: ['B', 'C'], b: ['A', 'C'], c: ['A', 'B'] };
  const ELEMS = {
    c: { kind: 'side', name: 'AB' }, a: { kind: 'side', name: 'BC' }, b: { kind: 'side', name: 'AC' },
    A: { kind: 'ang', name: '∠A' }, B: { kind: 'ang', name: '∠B' }, C: { kind: 'ang', name: '∠C' },
  };
  const others = v => ['A', 'B', 'C'].filter(x => x !== v);
  const valueOf = (T, id) => ELEMS[id].kind === 'side'
    ? fmt(V.dist(T[SIDE_V[id][0]], T[SIDE_V[id][1]]))
    : fmtDeg(angleAt(T[others(id)[0]], T[id], T[others(id)[1]]));

  function toScreen(T, center) {
    const xs = Object.values(T).map(p => p[0]), ys = Object.values(T).map(p => p[1]);
    const cx = (Math.min(...xs) + Math.max(...xs)) / 2, cy = (Math.min(...ys) + Math.max(...ys)) / 2;
    const out = {};
    for (const k in T) out[k] = [center[0] + (T[k][0] - cx) * U, center[1] - (T[k][1] - cy) * U];
    return out;
  }
  const RIGHT = [535, 170], LEFT = [185, 170];
  const secS = toScreen(SEC, RIGHT);

  const gSec = S('g', {}, svg), gTr = S('g', {}, svg), gCap = S('g', {}, svg), gCap2 = S('g', {}, svg);
  D.text(gCap, [RIGHT[0], 22], 'המשולש שלי', 'cap c-blue');
  let sel = [], running = false;
  const found = Store.get('ch2found', {});

  // ציור משולש עם הנתונים שנבחרו מסומנים בצבע
  function drawTri(g, T, cls, prime, interactive) {
    const pts = [T.A, T.B, T.C], ctr = centroid(pts);
    D.poly(g, pts, cls);
    sel.forEach((id, i) => {
      const col = COLORS[i], e = ELEMS[id];
      if (e.kind === 'side') {
        const [p, q] = SIDE_V[id];
        D.seg(g, T[p], T[q], { class: 'selside', style: `stroke:${col}` });
        D.sideLabel(g, T[p], T[q], ctr, valueOf(SEC, id), { class: 'val', style: `fill:${col}` }, 17);
      } else {
        const [p, q] = others(id);
        D.arc(g, T[id], T[p], T[q], 1, 24, { class: 'selarc', style: `stroke:${col}` }, { style: `fill:${col};opacity:.18` });
        D.angLabel(g, T[id], T[p], T[q], 44, valueOf(SEC, id), { class: 'val', style: `fill:${col}` });
      }
    });
    ['A', 'B', 'C'].forEach(n => D.label(g, T[n], n + (prime ? '′' : ''), ctr, 20, 'lbl'));
    if (!interactive) return;
    for (const id of ['a', 'b', 'c']) {
      const [p, q] = SIDE_V[id];
      const h = D.seg(g, V.lerp(T[p], T[q], 0.2), V.lerp(T[p], T[q], 0.8), 'hit');
      h.onclick = () => toggle(id);
    }
    for (const id of ['A', 'B', 'C']) {
      const [p, q] = others(id);
      const b = V.norm(V.add(V.norm(V.sub(T[p], T[id])), V.norm(V.sub(T[q], T[id]))));
      const h = D.dot(g, V.add(T[id], V.mul(b, 26)), 26, 'hit');
      h.onclick = () => toggle(id);
    }
  }

  function renderSecret() {
    clearEl(gSec);
    drawTri(gSec, secS, 'tri-blue', false, !running);
  }
  function renderChips() {
    $('#tr-chips').innerHTML = sel.length
      ? sel.map((id, i) => `<span class="chip" style="border-color:${COLORS[i]};color:${COLORS[i]}">${M(ELEMS[id].name + ' = ' + valueOf(SEC, id))}</span>`).join('')
        + (sel.length < 3 ? `<span class="chip ghost">בחר עוד ${3 - sel.length}</span>` : '')
      : '<span class="chip ghost">עוד לא בחרת. לחץ על צלע או על זווית במשולש.</span>';
    $('#tr-go').disabled = sel.length !== 3 || running;
  }
  function toggle(id) {
    if (running) return;
    const i = sel.indexOf(id);
    if (i >= 0) sel.splice(i, 1);
    else if (sel.length < 3) sel.push(id);
    else { flash('אפשר לבחור רק 3 נתונים. לחץ על נתון שבחרת כדי לבטל אותו.'); return; }
    clearEl(gTr); clearEl(gCap2); $('#tr-result').innerHTML = '';
    renderSecret(); renderChips();
  }
  function flash(t) { $('#tr-result').innerHTML = `<div class="msg info">${t}</div>`; }

  // איזה סוג בחירה זו? ואם הרמאי יכול לנצח, איזה משולש הוא בונה
  function classify(ids) {
    const sides = ids.filter(i => 'abc'.includes(i)), angs = ids.filter(i => 'ABC'.includes(i));
    if (sides.length === 3) return { type: 'SSS' };
    if (angs.length === 3) {
      const c = centroid(Object.values(SEC)), alt = {};
      for (const k in SEC) alt[k] = V.add(c, V.mul(V.sub(SEC[k], c), 0.62));
      return { type: 'AAA', alt };
    }
    if (angs.length === 2) return { type: SIDE_V[sides[0]].every(v => angs.includes(v)) ? 'ASA' : 'AAS' };
    const v0 = angs[0], [s1, s2] = sides;
    if (SIDE_V[s1].includes(v0) && SIDE_V[s2].includes(v0)) return { type: 'SAS' };
    const adj = SIDE_V[s1].includes(v0) ? s1 : s2, opp = adj === s1 ? s2 : s1;
    const len = s => V.dist(SEC[SIDE_V[s][0]], SEC[SIDE_V[s][1]]);
    if (len(opp) >= len(adj)) return { type: 'SSA+', adj, opp, v: v0 };
    // הזווית מול הצלע הקטנה: הקודקוד השלישי יכול להחליק על הקרן לנקודה אחרת
    const W = SIDE_V[adj].find(v => v !== v0), Uv = others(v0).find(v => v !== W);
    const th = angleAt(SEC[W], SEC[v0], SEC[Uv]) * RAD, s = len(adj), t = V.dist(SEC[v0], SEC[Uv]);
    const t2 = 2 * s * Math.cos(th) - t; // השורש השני של משוואת הקוסינוסים
    if (t2 > 0.05 && Math.abs(t2 - t) > 0.05) {
      const alt = Object.assign({}, SEC);
      alt[Uv] = V.add(SEC[v0], V.mul(V.norm(V.sub(SEC[Uv], SEC[v0])), t2));
      return { type: 'SSA-', adj, opp, v: v0, alt };
    }
    return { type: 'SSA+', adj, opp, v: v0 };
  }

  const TYPES = {
    SSS: { name: 'צ.צ.צ', what: 'שלוש צלעות', ok: true },
    SAS: { name: 'צ.ז.צ', what: 'שתי צלעות והזווית <b>שביניהן</b>', ok: true },
    ASA: { name: 'ז.צ.ז', what: 'שתי זוויות והצלע <b>שביניהן</b>', ok: true },
    AAS: { name: 'ז.ז.צ', what: 'שתי זוויות וצלע <b>שלא ביניהן</b>', ok: true },
    'SSA+': { name: 'צ.צ.ז', what: 'שתי צלעות וזווית <b>מול הצלע הגדולה</b>', ok: true },
    'SSA-': { name: 'צ.צ.ז', what: 'שתי צלעות וזווית <b>מול הצלע הקטנה</b>', ok: false },
    AAA: { name: 'ז.ז.ז', what: 'שלוש זוויות', ok: false },
  };
  const ORDER = ['SSS', 'SAS', 'ASA', 'AAS', 'SSA+', 'SSA-', 'AAA'];

  function explain(cl) {
    const n = id => M(ELEMS[id].name);
    switch (cl.type) {
      case 'SSS': return 'שלוש צלעות קובעות את המשולש לגמרי. זה משפט החפיפה <b>צ.צ.צ</b>.';
      case 'SAS': return 'שתי צלעות והזווית שביניהן קובעות את המשולש. זה משפט החפיפה <b>צ.ז.צ</b>.';
      case 'ASA': return 'שתי זוויות והצלע שביניהן קובעות את המשולש. זה משפט החפיפה <b>ז.צ.ז</b>.';
      case 'AAS': return 'שתי זוויות וצלע שלא ביניהן. גם זה עובד! כי אם יודעים שתי זוויות, יודעים גם את השלישית (סכום הזוויות במשולש הוא 180°). ואז יש לנו שתי זוויות והצלע שביניהן, כלומר <b>ז.צ.ז</b> בתחפושת.';
      case 'SSA+': return `הזווית ${n(cl.v)} נמצאת מול ${n(cl.opp)}, שהיא הצלע הגדולה מבין השתיים (${M(ELEMS[cl.opp].name + ' = ' + valueOf(SEC, cl.opp))} מול ${M(ELEMS[cl.adj].name + ' = ' + valueOf(SEC, cl.adj))}). במקרה כזה המשולש נקבע. זה משפט החפיפה הרביעי, <b>צ.צ.ז</b>: שתי צלעות והזווית שמול הגדולה מביניהן.`;
      case 'SSA-': return `כאן הזווית ${n(cl.v)} נמצאת מול ${n(cl.opp)}, שהיא הצלע ה<b>קטנה</b> מבין השתיים. ואז הרמאי יכול "להחליק" קודקוד לאורך הקרן ולקבל משולש שני, שונה לגמרי. לכן במשפט צ.צ.ז הזווית חייבת להיות מול הצלע הגדולה. בפרק 3 תבין בדיוק למה.`;
      case 'AAA': return 'שלוש זוויות קובעות את <b>הצורה</b>, אבל לא את <b>הגודל</b>. הרמאי פשוט הקטין את המשולש. משולשים כאלה נקראים "דומים", ואין משפט חפיפה ז.ז.ז.';
    }
  }

  function run() {
    if (sel.length !== 3 || running) return;
    running = true; renderSecret(); renderChips();
    const cl = classify(sel);
    clearEl(gTr); clearEl(gCap2);
    D.text(gCap2, [LEFT[0], 22], 'המשולש של הרמאי', 'cap c-orange');
    const done = () => { running = false; renderSecret(); renderChips(); record(cl); };
    if (cl.alt) {
      drawTri(gTr, toScreen(cl.alt, LEFT), 'tri-orange', true, false);
      $('#tr-result').innerHTML = `<div class="card lose"><h4>😈 הרמאי ניצח!</h4>
        <p>הוא בנה משולש <b>אחר</b>, לא חופף לשלך, ובכל זאת שלושת הנתונים שבחרת זהים בשניהם (תשווה את המספרים הצבעוניים). אז הנתונים האלה <b>לא מספיקים</b>.</p><p>${explain(cl)}</p></div>`;
      done();
      return;
    }
    // הרמאי נכשל: הוא בונה העתק, הפוך ומסובב, ואז רואים אותו נוחת על המקור
    const copyU = {};
    for (const k in SEC) copyU[k] = V.rot([-SEC[k][0], SEC[k][1]], 25 * RAD);
    const cs = toScreen(copyU, LEFT);
    const from = [cs.A, cs.B, cs.C], to = [secS.A, secS.B, secS.C];
    const plan = rigidPlan(from, to);
    const frame = pts => { clearEl(gTr); drawTri(gTr, { A: pts[0], B: pts[1], C: pts[2] }, 'tri-orange ghosty', true, false); };
    frame(from);
    $('#tr-result').innerHTML = `<div class="msg info">הרמאי בונה משולש עם ${M(sel.map(id => ELEMS[id].name).join(', '))}... ועכשיו נבדוק אותו.</div>`;
    setTimeout(() => {
      animate(2000, t => frame(plan(t)), () => {
        $('#tr-result').innerHTML = `<div class="card win"><h4>🏆 ניצחת את הרמאי!</h4>
          <p>כל משולש שהוא בונה עם הנתונים האלה יוצא חופף לשלך. ראית? הוא ${plan.flip ? 'הפך, ' : ''}סובב והזיז את המשולש שלו, והוא נחת <b>בדיוק</b> על שלך. אז הנתונים האלה <b>מספיקים</b>.</p><p>${explain(cl)}</p></div>`;
        celebrate();
        done();
      });
    }, 900);
  }

  function record(cl) {
    if (!found[cl.type]) { found[cl.type] = true; Store.set('ch2found', found); }
    renderBadges();
  }
  function renderBadges() {
    $('#tr-badges').innerHTML = ORDER.map(k => {
      const t = TYPES[k], f = found[k];
      return `<div class="badge ${f ? (t.ok ? 'ok' : 'no') : ''}">
        <div class="bn">${f ? t.name : '?'}</div>
        <div class="bw">${t.what}</div>
        <div class="bv">${f ? (t.ok ? '✓ מספיק' : '✗ לא מספיק') : 'עוד לא ניסית'}</div></div>`;
    }).join('');
    if (ORDER.every(k => found[k])) {
      App.markDone('ch2');
      $('#tr-final').innerHTML = `<div class="card idea"><h4>🎓 גילית בעצמך את משפטי החפיפה!</h4>
        <p>מתוך כל הבחירות של 3 נתונים, אלה מספיקים כדי ששני משולשים יהיו חופפים:</p>
        <ul class="thm-list">
          <li><b>צ.צ.צ</b>: שלוש צלעות.</li>
          <li><b>צ.ז.צ</b>: שתי צלעות והזווית שביניהן.</li>
          <li><b>ז.צ.ז</b>: שתי זוויות והצלע שביניהן (ואם הצלע לא ביניהן, מחשבים את הזווית השלישית).</li>
          <li><b>צ.צ.ז</b>: שתי צלעות והזווית שמול הצלע הגדולה מביניהן.</li>
        </ul>
        <p>ואלה לא מספיקים: <b>ז.ז.ז</b>, ו<b>צ.צ.ז</b> כשהזווית מול הצלע הקטנה.</p>
        <p>אבל למה? למה דווקא אלה? בוא נבנה משולשים בידיים ונראה.</p>
        ${App.nextButton('ch2')}</div>`;
    }
  }

  $('#tr-go').onclick = run;
  $('#tr-clear').onclick = () => { if (running) return; sel = []; clearEl(gTr); clearEl(gCap2); $('#tr-result').innerHTML = ''; renderSecret(); renderChips(); };

  renderSecret(); renderChips(); renderBadges();
};
