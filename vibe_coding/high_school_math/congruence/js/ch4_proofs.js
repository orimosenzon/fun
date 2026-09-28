'use strict';
/* פרק 4: הוכחות אינטראקטיביות. עומר בוחר נימוק לכל שורה, והציור מתעדכן. */

App.inits.ch4 = function () {
  const host = document.getElementById('proofs');
  const TH = ['צ.צ.צ', 'צ.ז.צ', 'ז.צ.ז', 'צ.צ.ז'];
  // אותיות לטיניות בתוך משפט עברי: עוטפים כדי שהכיוון לא יתבלגן
  const bidi = t => t.replace(/[∠△]?[A-Z]+/g, x => M(x));
  const R = {
    given: 'נתון', common: 'צלע משותפת', vert: 'זוויות קודקודיות', adj: 'זוויות צמודות',
    corrS: 'צלעות מתאימות במשולשים חופפים שוות', corrA: 'זוויות מתאימות במשולשים חופפים שוות',
    mid: 'אמצע קטע',
  };

  /* סימונים: seg 'AB' עם n קווים, ang 'BAD' (הקודקוד באמצע) עם n קשתות, hl = צלע משותפת, right = זווית ישרה */
  const PROOFS = [
    {
      title: 'זוויות הבסיס במשולש שווה שוקיים', tag: 'צ.ז.צ',
      pts: { A: [230, 45], B: [80, 320], C: [380, 320], D: [230, 320] },
      segs: ['AB', 'AC', 'BC', 'AD'],
      intro: 'בטח שמעת שבמשולש שווה שוקיים זוויות הבסיס שוות. אבל למה זה נכון? הרעיון: הקטע AD חותך את המשולש לשני משולשים קטנים. אם נוכיח שהם חופפים, נקבל את מה שרצינו.',
      given: `${M('AB = AC')} (משולש שווה שוקיים). ${M('AD')} חוצה את הזווית ${M('∠BAC')}.`,
      prove: M('∠B = ∠C'),
      steps: [
        { s: 'AB = AC', ok: R.given, bad: [R.common, R.vert], mk: [{ seg: 'AB', n: 1 }, { seg: 'AC', n: 1 }] },
        { s: '∠BAD = ∠CAD', ok: 'AD חוצה זווית (נתון)', bad: [R.vert, R.adj], mk: [{ ang: 'BAD', n: 1 }, { ang: 'CAD', n: 1 }] },
        { s: 'AD = AD', ok: R.common, bad: [R.given, R.mid], mk: [{ seg: 'AD', hl: true }] },
        { s: '△ABD ≅ △ACD', cong: ['ABD', 'ACD'], ok: 'צ.ז.צ', hint: 'יש לנו שתי צלעות (AB ו-AD) וזווית אחת. איפה הזווית נמצאת ביחס לשתי הצלעות?' },
        { s: '∠B = ∠C', ok: R.corrA, bad: [R.given, R.vert], mk: [{ ang: 'ABD', n: 2 }, { ang: 'ACD', n: 2 }], final: true },
      ],
      bonus: `<p>אבל רגע, יש עוד! מהחפיפה קיבלנו <b>בחינם</b> עוד שני דברים:</p>
        <ul><li>${M('BD = CD')}: כלומר AD הוא גם <b>תיכון</b>.</li>
        <li>${M('∠ADB = ∠ADC')}: הן זוויות צמודות, סכומן 180°, אז כל אחת מהן 90°. כלומר AD הוא גם <b>גובה</b>!</li></ul>
        <p>שילמנו 3 שוויונות וקיבלנו 3 חדשים. זה הכוח של חפיפה.</p>`,
    },
    {
      title: 'שני קטעים שחוצים זה את זה', tag: 'צ.ז.צ',
      pts: { A: [80, 70], B: [380, 310], C: [60, 290], D: [400, 90], O: [230, 190] },
      segs: ['AB', 'CD', 'AC', 'BD'],
      intro: 'כאן אין צלע משותפת. אבל יש כלי אחר שמופיע המון בהוכחות: זוויות קודקודיות.',
      given: `הקטעים ${M('AB')} ו-${M('CD')} נחתכים בנקודה ${M('O')}, והיא אמצע של שניהם.`,
      prove: M('AC = BD'),
      steps: [
        { s: 'AO = BO', ok: 'O אמצע AB (נתון)', bad: [R.common, R.vert], mk: [{ seg: 'AO', n: 1 }, { seg: 'BO', n: 1 }] },
        { s: 'CO = DO', ok: 'O אמצע CD (נתון)', bad: [R.common, R.corrS], mk: [{ seg: 'CO', n: 2 }, { seg: 'DO', n: 2 }] },
        { s: '∠AOC = ∠BOD', ok: R.vert, bad: [R.adj, R.given], mk: [{ ang: 'AOC', n: 1 }, { ang: 'BOD', n: 1 }] },
        { s: '△AOC ≅ △BOD', cong: ['AOC', 'BOD'], ok: 'צ.ז.צ', hint: 'שתי צלעות וזווית. הזווית ב-O: איפה היא ביחס לשתי הצלעות שיוצאות מ-O?' },
        { s: 'AC = BD', ok: R.corrS, bad: [R.given, R.mid], mk: [{ seg: 'AC', n: 3 }, { seg: 'BD', n: 3 }], final: true },
      ],
      bonus: `<p>גם כאן יש מתנה: ${M('∠CAO = ∠DBO')}. אם כבר למדת על ישרים מקבילים, אלה זוויות מתחלפות, ולכן ${M('AC ∥ BD')}. מהנתונים לבד היה קשה מאוד לראות את זה.</p>`,
    },
    {
      title: 'אלכסון שחוצה שתי זוויות', tag: 'ז.צ.ז',
      pts: { B: [230, 40], D: [230, 345], A: [85, 150], C: [375, 150] },
      segs: ['AB', 'BC', 'CD', 'DA', 'BD'],
      intro: 'הפעם הנתונים הם זוויות. איזה משפט עובד עם שתי זוויות?',
      given: `במרובע ${M('ABCD')}, האלכסון ${M('BD')} חוצה את הזווית ${M('∠ABC')} וגם את הזווית ${M('∠ADC')}.`,
      prove: M('AB = CB'),
      steps: [
        { s: '∠ABD = ∠CBD', ok: 'BD חוצה את ∠ABC (נתון)', bad: [R.vert, R.corrA], mk: [{ ang: 'ABD', n: 1 }, { ang: 'CBD', n: 1 }] },
        { s: 'BD = BD', ok: R.common, bad: [R.given, R.mid], mk: [{ seg: 'BD', hl: true }] },
        { s: '∠ADB = ∠CDB', ok: 'BD חוצה את ∠ADC (נתון)', bad: [R.vert, R.adj], mk: [{ ang: 'ADB', n: 2 }, { ang: 'CDB', n: 2 }] },
        { s: '△ABD ≅ △CBD', cong: ['ABD', 'CBD'], ok: 'ז.צ.ז', hint: 'יש לנו שתי זוויות וצלע אחת. איפה הצלע BD ביחס לשתי הזוויות?' },
        { s: 'AB = CB', ok: R.corrS, bad: [R.given, R.common], mk: [{ seg: 'AB', n: 1 }, { seg: 'CB', n: 1 }], final: true },
      ],
      bonus: `<p>ובחינם: גם ${M('AD = CD')} וגם ${M('∠A = ∠C')}. למרובע כזה, עם שני זוגות של צלעות סמוכות שוות, קוראים <b>דלתון</b>.</p>`,
    },
    {
      title: 'האלכסון של הדלתון', tag: 'צ.צ.צ',
      pts: { A: [230, 40], B: [95, 170], C: [230, 350], D: [365, 170] },
      segs: ['AB', 'AD', 'BC', 'DC', 'AC'],
      intro: 'בנתונים אין אף זווית, רק צלעות. איזה משפט עובד רק עם צלעות?',
      given: `${M('AB = AD')} וגם ${M('CB = CD')} (דלתון).`,
      prove: `${M('AC')} חוצה את ${M('∠BAD')}, כלומר ${M('∠BAC = ∠DAC')}`,
      steps: [
        { s: 'AB = AD', ok: R.given, bad: [R.common, R.vert], mk: [{ seg: 'AB', n: 1 }, { seg: 'AD', n: 1 }] },
        { s: 'CB = CD', ok: R.given, bad: [R.common, R.mid], mk: [{ seg: 'CB', n: 2 }, { seg: 'CD', n: 2 }] },
        { s: 'AC = AC', ok: R.common, bad: [R.given, R.vert], mk: [{ seg: 'AC', hl: true }] },
        { s: '△ABC ≅ △ADC', cong: ['ABC', 'ADC'], ok: 'צ.צ.צ', hint: 'אין לנו אף זווית, רק שלוש צלעות.' },
        { s: '∠BAC = ∠DAC', ok: R.corrA, bad: [R.given, R.vert], mk: [{ ang: 'BAC', n: 1 }, { ang: 'DAC', n: 1 }], final: true },
      ],
      bonus: `<p>ובחינם: ${M('∠BCA = ∠DCA')} (האלכסון חוצה גם את הזווית ב-C), וגם ${M('∠B = ∠D')}.</p><p>אבל יש עוד משהו מפתיע בדלתון הזה... תמשיך להוכחה הבאה.</p>`,
    },
    {
      title: 'הוכחה בשתי קומות: האלכסונים של הדלתון מאונכים', tag: 'צ.צ.צ ואז צ.ז.צ',
      pts: { A: [230, 40], B: [95, 170], C: [230, 350], D: [365, 170], O: [230, 170] },
      segs: ['AB', 'AD', 'BC', 'DC', 'AC', 'BD'],
      intro: 'הנה משהו שממש לא רואים מהנתונים: בנתונים אין אף זווית, ובטח לא 90°. נצטרך שתי חפיפות. הראשונה תיתן לנו זווית, והשנייה תשתמש בה.',
      given: `${M('AB = AD')}, ${M('CB = CD')}. האלכסונים נחתכים בנקודה ${M('O')}.`,
      prove: `${M('AC ⊥ BD')} (האלכסונים מאונכים)`,
      steps: [
        { s: 'AB = AD', ok: R.given, bad: [R.common, R.vert], mk: [{ seg: 'AB', n: 1 }, { seg: 'AD', n: 1 }] },
        { s: 'CB = CD', ok: R.given, bad: [R.common, R.mid], mk: [{ seg: 'CB', n: 2 }, { seg: 'CD', n: 2 }] },
        { s: 'AC = AC', ok: R.common, bad: [R.given, R.vert], mk: [{ seg: 'AC', hl: true }] },
        { s: '△ABC ≅ △ADC', cong: ['ABC', 'ADC'], ok: 'צ.צ.צ', hint: 'רק צלעות.' },
        { s: '∠BAO = ∠DAO', ok: 'זוויות מתאימות במשולשים חופפים (מהחפיפה הראשונה)', bad: [R.given, R.vert], mk: [{ ang: 'BAO', n: 1 }, { ang: 'DAO', n: 1 }] },
        { s: 'AO = AO', ok: R.common, bad: [R.given, R.mid], mk: [{ seg: 'AO', hl: true }] },
        { s: '△ABO ≅ △ADO', cong: ['ABO', 'ADO'], refs: '1, 5, 6', ok: 'צ.ז.צ', hint: 'AB = AD, הצלע AO משותפת, והזווית שקיבלנו מהחפיפה הראשונה. איפה היא ביחס לשתי הצלעות?' },
        { s: '∠AOB = ∠AOD', ok: R.corrA, bad: [R.vert, R.given], mk: [{ ang: 'AOB', n: 2 }, { ang: 'AOD', n: 2 }] },
        { s: '∠AOB + ∠AOD = 180°', ok: R.adj, bad: [R.vert, R.common], mk: [] },
        { s: '∠AOB = ∠AOD = 90°', ok: 'שתי זוויות שוות שסכומן 180°, אז כל אחת היא חצי', bad: [R.given, R.corrA], mk: [{ right: 'AOB' }, { right: 'AOD' }], final: true },
      ],
      bonus: `<p>וגם ${M('BO = DO')}: האלכסון AC חוצה את BD.</p><p>שני משפטי חפיפה, אחד אחרי השני, הוכיחו משהו שבכלל לא הופיע בנתונים. ככה נראות רוב ההוכחות בגאומטריה: כל חפיפה נותנת "מתנות", ומשתמשים בהן כדי להגיע לחפיפה הבאה.</p>`,
    },
  ];

  host.innerHTML = `<div class="tabs" id="pf-tabs"></div><div id="pf-box"></div>`;
  const proofsDone = Store.get('proofsDone', {});
  let curP = 0;
  function renderTabs() {
    const t = host.querySelector('#pf-tabs');
    t.innerHTML = PROOFS.map((p, i) => `<button class="${i === curP ? 'active' : ''} ${proofsDone[i] ? 'done' : ''}" data-p="${i}">${proofsDone[i] ? '✓ ' : ''}הוכחה ${i + 1}<small>${p.tag}</small></button>`).join('');
    t.querySelectorAll('button').forEach(b => b.onclick = () => openProof(+b.dataset.p));
  }

  function openProof(i) {
    curP = i; renderTabs();
    const P = PROOFS[i];
    let cur = 0, animating = false;
    const chosen = [];
    const box = host.querySelector('#pf-box');
    box.innerHTML = `
      <div class="proof">
        <h3>${P.title} <span class="tag">${P.tag}</span></h3>
        <p class="muted">${P.intro}</p>
        <div class="gp"><div><b>נתון:</b> ${P.given}</div><div><b>צריך להוכיח:</b> ${P.prove}</div></div>
        <div class="proof-grid">
          <div class="proof-fig">
            <svg class="board" viewBox="0 0 460 390"></svg>
            <button class="btn ghost small" id="pf-anim" hidden>🔄 הראה איך משולש אחד נוחת על השני</button>
          </div>
          <div class="proof-side">
            <table class="steps"><thead><tr><th>#</th><th>טענה</th><th>נימוק</th></tr></thead><tbody></tbody></table>
            <div class="ask"></div>
          </div>
        </div>
        <div class="bonus"></div>
      </div>`;
    const svg = box.querySelector('svg');
    const gFig = S('g', {}, svg), gAnim = S('g', {}, svg);
    const pt = n => P.pts[n];
    const figC = centroid(Object.values(P.pts));
    const triPts = s => s.split('').map(pt);

    function lastCong(upto) { let c = null; for (let k = 0; k < upto; k++) if (P.steps[k].cong) c = P.steps[k].cong; return c; }
    function drawMark(m, cls) {
      if (m.seg) {
        const [p, q] = m.seg.split('').map(pt);
        if (m.hl) D.seg(gFig, p, q, 'mk-hl ' + cls); else D.ticks(gFig, p, q, m.n, 'mk ' + cls);
      } else if (m.ang) {
        const [p, v, q] = m.ang.split('').map(pt);
        D.arc(gFig, v, p, q, m.n, 24, 'mk ' + cls);
      } else if (m.right) {
        const [p, v, q] = m.right.split('').map(pt);
        D.right(gFig, v, p, q, 15, 'mk ' + cls);
      }
    }
    function drawFig() {
      clearEl(gFig);
      const cg = lastCong(cur);
      if (cg) { D.poly(gFig, triPts(cg[0]), 'pf-t1'); D.poly(gFig, triPts(cg[1]), 'pf-t2'); }
      for (const s of P.segs) D.seg(gFig, pt(s[0]), pt(s[1]), 'pf-seg');
      P.steps.forEach((st, k) => {
        if (!st.mk || k > cur) return;
        const cls = k === cur ? 'cur' : (st.final ? 'fin' : 'done');
        st.mk.forEach(m => drawMark(m, cls));
      });
      for (const n in P.pts) { D.dot(gFig, pt(n), 4, 'pf-dot'); D.label(gFig, pt(n), n, figC, 18, 'lbl'); }
    }

    function renderSteps() {
      const tb = box.querySelector('tbody');
      tb.innerHTML = P.steps.slice(0, Math.min(cur + 1, P.steps.length)).map((st, k) => {
        const isCur = k === cur;
        return `<tr class="${isCur ? 'cur' : ''} ${st.cong ? 'cong' : ''} ${st.final && k < cur ? 'fin' : ''}">
          <td>${k + 1}</td><td>${M(st.s)}</td><td>${isCur ? '<span class="q">?</span>' : chosen[k]}</td></tr>`;
      }).join('');
      const ask = box.querySelector('.ask');
      if (cur >= P.steps.length) { ask.innerHTML = '<div class="msg win">✔ מש"ל (מה שהיה להוכיח)</div>'; return; }
      const st = P.steps[cur];
      const opts = st.cong ? TH : shuffle([st.ok, ...st.bad]);
      ask.innerHTML = `<div class="qline">${st.cong ? 'לפי איזה משפט' : 'למה'} ${M(st.s)}?</div>
        <div class="opts">${opts.map((o, k) => `<button class="opt" data-k="${k}">${bidi(o)}</button>`).join('')}</div><div class="fb"></div>`;
      ask.querySelectorAll('.opt').forEach(b => b.onclick = () => answer(b, opts[+b.dataset.k]));
    }
    function answer(btn, txt) {
      const st = P.steps[cur], fb = box.querySelector('.fb');
      if (txt !== st.ok) {
        btn.classList.add('wrong'); btn.disabled = true;
        fb.innerHTML = st.cong ? `לא בדיוק. ${st.hint}` : 'לא בדיוק. תסתכל שוב על הציור ועל הנתונים.';
        fb.className = 'fb bad';
        return;
      }
      chosen[cur] = st.cong ? `${st.ok} (לפי ${st.refs || `${cur - 2}, ${cur - 1}, ${cur}`})` : bidi(st.ok);
      cur++;
      drawFig(); renderSteps();
      if (st.cong) { showAnim(st.cong); overlayAnim(st.cong); }
      if (cur >= P.steps.length) finish();
    }
    function showAnim(cg) {
      const b = box.querySelector('#pf-anim');
      b.hidden = false; b.onclick = () => overlayAnim(cg);
    }
    // עותק של המשולש השני טס ונוחת על הראשון, לפי סדר האותיות בחפיפה
    function overlayAnim(cg) {
      if (animating) return;
      animating = true;
      const from = triPts(cg[1]), to = triPts(cg[0]), plan = rigidPlan(from, to);
      const frame = pts => {
        clearEl(gAnim);
        D.poly(gAnim, pts, 'pf-fly');
        cg[1].split('').forEach((n, k) => D.label(gAnim, pts[k], n, centroid(pts), 16, 'lbl small c-orange'));
      };
      setTimeout(() => animate(1800, t => frame(plan(t)), () => setTimeout(() => {
        animate(500, t => { gAnim.setAttribute('opacity', 1 - t); }, () => { clearEl(gAnim); gAnim.setAttribute('opacity', 1); animating = false; });
      }, 1100)), 250);
    }
    function finish() {
      proofsDone[i] = true; Store.set('proofsDone', proofsDone); renderTabs();
      celebrate();
      const next = i + 1 < PROOFS.length
        ? `<button class="btn primary" id="pf-next">להוכחה ${i + 2} ←</button>`
        : App.nextButton('ch4');
      box.querySelector('.bonus').innerHTML = `<div class="card idea"><h4>🎁 מתנות בחינם</h4>${P.bonus}${next}</div>`;
      const nb = box.querySelector('#pf-next'); if (nb) nb.onclick = () => openProof(i + 1);
      if (PROOFS.every((_, k) => proofsDone[k])) App.markDone('ch4');
    }
    drawFig(); renderSteps();
  }
  openProof(0);
};
