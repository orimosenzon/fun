'use strict';
/* פרק 3: מעבדות בנייה. בכל מעבדה בונים משולש מחתיכות נתונות ומנסים לבנות עוד אחד, שונה. */

App.inits.ch3 = function () {
  const LABS = [
    { id: 'sss', name: 'צ.צ.צ', build: labSSS },
    { id: 'sas', name: 'צ.ז.צ', build: labSAS },
    { id: 'asa', name: 'ז.צ.ז', build: labASA },
    { id: 'ssa', name: 'צ.צ.ז', build: labSSA },
    { id: 'aaa', name: 'ז.ז.ז', build: labAAA },
  ];
  const tabs = document.getElementById('lab-tabs'), host = document.getElementById('lab-host');
  const done = Store.get('labsDone', {});
  const hosts = {};
  function markLab(id) {
    if (done[id]) return;
    done[id] = true; Store.set('labsDone', done); renderTabs();
    if (LABS.every(l => done[l.id])) App.markDone('ch3');
  }
  function renderTabs() {
    tabs.innerHTML = LABS.map(l => `<button data-lab="${l.id}" class="${l.id === cur ? 'active' : ''} ${done[l.id] ? 'done' : ''}">${done[l.id] ? '✓ ' : ''}${l.name}</button>`).join('');
    tabs.querySelectorAll('button').forEach(b => b.onclick = () => open(b.dataset.lab));
  }
  let cur = LABS[0].id;
  function open(id) {
    cur = id; renderTabs();
    for (const k in hosts) hosts[k].hidden = k !== id;
    if (!hosts[id]) {
      const h = document.createElement('div'); host.appendChild(h); hosts[id] = h;
      LABS.find(l => l.id === id).build(h, () => markLab(id));
    }
  }
  open(cur);
};

/* ---------- תשתית משותפת למעבדות ---------- */
function slider(k, label, min, max, step, val) {
  return `<label class="sl"><span>${label}</span><input type="range" data-k="${k}" min="${min}" max="${max}" step="${step}" value="${val}"><output>${val}</output></label>`;
}
function checkbox(k, label) { return `<label class="chk"><input type="checkbox" data-k="${k}"> ${label}</label>`; }

function makeLab(host, cfg) {
  host.className = 'lab';
  host.innerHTML = `
    <h3>${cfg.title}</h3>
    <div class="lab-intro">${cfg.intro}</div>
    <div class="given">${cfg.given || ''}</div>
    <svg class="board" viewBox="0 0 720 ${cfg.h}"></svg>
    <div class="readout"></div>
    <div class="lab-status msg"></div>
    <div class="lab-controls">${cfg.controls || ''}</div>
    <div class="lab-found"></div>
    <details class="lab-explain"><summary>💡 למה זה ככה? (פתח אחרי שניסית)</summary><div class="ex">${cfg.explain}</div></details>
    ${cfg.after || ''}`;
  const svg = host.querySelector('svg');
  const lab = {
    svg, U: cfg.U, found: [],
    scene: S('g', {}, svg),
    top: S('g', {}, svg),
    q: s => host.querySelector(s),
    status(html, kind) { const el = host.querySelector('.lab-status'); el.innerHTML = html; el.className = 'lab-status msg ' + (kind || ''); },
    readout(html) { host.querySelector('.readout').innerHTML = html; },
    handle(cls) { return S('circle', { r: 13, class: 'handle ' + (cls || '') }, this.top); },
    reset() { this.found = []; this.renderFound(); },
    // מתעד משולש שנבנה. מחזיר 'new' / 'same' / 'mirror'
    record(T) {
      const sides = [V.dist(T.B, T.C), V.dist(T.A, T.C), V.dist(T.A, T.B)].map(x => x / this.U);
      const o = orient(T.A, T.B, T.C);
      for (const f of this.found) if (f.sides.every((x, i) => Math.abs(x - sides[i]) < 0.12)) return f.o === o ? 'same' : 'mirror';
      this.found.push({ sides, o }); this.renderFound();
      return 'new';
    },
    renderFound() {
      const el = host.querySelector('.lab-found');
      if (!this.found.length) { el.innerHTML = ''; return; }
      const list = this.found.slice(-4).map((f, i) => `<span class="chip">${M(`BC = ${fmt(f.sides[0])},  AC = ${fmt(f.sides[1])},  AB = ${fmt(f.sides[2])}`)}</span>`).join('');
      el.innerHTML = `<div>משולשים שונים (לא חופפים) שמצאת: <b class="big-n">${this.found.length}</b></div><div class="chips">${list}</div>`;
    },
    closeMsg(kind) {
      const N = this.found.length;
      if (kind === 'new' && N === 1) this.status('✅ המשולש נסגר! עכשיו האתגר: בנה משולש <b>אחר</b>, שלא חופף לזה, מאותן חתיכות בדיוק.', 'ok');
      else if (kind === 'new') { this.status(cfg.winMsg || '😮 מצאת משולש שני, שונה מהראשון!', 'win'); cfg.done(); celebrate(); }
      else if (kind === 'same') this.status('זה בדיוק אותו משולש שכבר מצאת. נסה במקום אחר.', 'info');
      else this.status('🪞 זו <b>תמונת מראה</b> של המשולש שכבר מצאת: אותן צלעות, אותן זוויות, רק הפוך. אם תהפוך אותו הוא יכסה את הראשון בדיוק, כלומר הוא <b>חופף</b> לו. זה לא נחשב משולש שונה.', 'info');
    },
    onInput(k, fn) {
      const inp = host.querySelector(`[data-k="${k}"]`);
      inp.addEventListener(inp.type === 'checkbox' ? 'change' : 'input', () => {
        if (inp.type === 'checkbox') fn(inp.checked);
        else { inp.nextElementSibling.textContent = inp.value; fn(+inp.value); }
      });
    },
  };
  host.querySelector('details').addEventListener('toggle', e => { if (e.target.open) cfg.done(); });
  return lab;
}

const noTri = 'אין כאן משולש.';

/* ---------- צ.צ.צ ---------- */
function labSSS(host, done) {
  const U = 45, A = [195, 235], c = 6;
  let a = 4.5, b = 5, showCirc = false, thA = -118 * RAD, thB = -52 * RAD, closed = false;
  const lab = makeLab(host, {
    title: 'צ.צ.צ: שלוש צלעות', h: 470, U, done,
    intro: `יש לך שלושה מקלות: הבסיס ${M('AB = 6')} (האפור, קבוע במקום), מקל ירוק שמחובר ל-A בציר, ומקל סגול שמחובר ל-B בציר. גרור את הקצוות של המקלות. כשהקצוות נפגשים, המשולש נסגר.`,
    controls: slider('b', 'אורך המקל הירוק (AC)', 2, 9, 0.5, b) + slider('a', 'אורך המקל הסגול (BC)', 2, 9, 0.5, a)
      + checkbox('circ', 'הראה את כל המקומות שכל קצה יכול להגיע אליהם'),
    explain: `<p>הקצה של המקל הירוק יכול להגיע רק לנקודות שנמצאות במרחק ${M('AC')} מ-A. כל הנקודות האלה ביחד הן <b>מעגל</b> סביב A. הקצה של המקל הסגול: מעגל סביב B. הקודקוד C חייב להיות על שני המעגלים ביחד, כלומר בנקודת חיתוך שלהם.</p>
      <p>ושני מעגלים נחתכים <b>לכל היותר בשתי נקודות</b>: אחת מעל AB ואחת מתחתיו. והשתיים הן תמונת מראה זו של זו, כך ששני המשולשים חופפים. כלומר, שלוש הצלעות לא משאירות שום חופש. יש רק משולש אחד.</p>
      <p class="thm"><b>משפט צ.צ.צ:</b> אם שלוש הצלעות של משולש אחד שוות לשלוש הצלעות של משולש שני, המשולשים חופפים.</p>
      <p>🏗️ בגלל זה המשולש הוא הצורה ה"קשיחה": מסגרת של שלושה מקלות לא מתעוותת, ומסגרת של ארבעה מקלות (מרובע) כן מתעוותת. תסתכל על מנופים, גשרים ועמודי חשמל: משולשים בכל מקום.</p>
      <p>ומה קורה כשהמקלות קצרים מדי? נסה ${M('AC = 2, BC = 3')}. המעגלים לא נפגשים ואין משולש בכלל. כל שתי צלעות ביחד חייבות להיות ארוכות מהשלישית (אי-שוויון המשולש).</p>`,
  });
  const B = () => [A[0] + c * U, A[1]];
  const hP = lab.handle('h-green'), hQ = lab.handle('h-purple');
  const P = () => V.add(A, V.mul(V.unit(thA), b * U));
  const Q = () => V.add(B(), V.mul(V.unit(thB), a * U));

  function snap() {
    const was = closed; closed = false;
    const pts = circleCircle(A, b * U, B(), a * U), p = P(), q = Q();
    if (pts.length && V.dist(p, q) < 18) {
      const m = V.mid(p, q);
      const X = V.dist(pts[0], m) < V.dist(pts[1], m) ? pts[0] : pts[1];
      thA = V.ang(V.sub(X, A)); thB = V.ang(V.sub(X, B()));
      closed = true;
    }
    if (closed && !was) lab.closeMsg(lab.record({ A, B: B(), C: P() }));
  }
  function render() {
    const g = lab.scene; clearEl(g);
    const Bp = B(), p = P(), q = Q();
    if (showCirc) {
      S('circle', { cx: A[0], cy: A[1], r: b * U, class: 'locus green' }, g);
      S('circle', { cx: Bp[0], cy: Bp[1], r: a * U, class: 'locus purple' }, g);
      for (const X of circleCircle(A, b * U, Bp, a * U)) D.dot(g, X, 6, 'meet');
    }
    const ctr = closed ? centroid([A, Bp, p]) : [A[0] + c * U / 2, A[1] - 80];
    if (closed) D.poly(g, [A, Bp, p], 'tri-fill');
    D.seg(g, A, Bp, 'base');
    D.seg(g, A, p, 'stick green'); D.seg(g, Bp, q, 'stick purple');
    D.sideLabel(g, A, Bp, ctr, fmt(c), 'val');
    D.sideLabel(g, A, p, closed ? ctr : Bp, fmt(b), 'val c-green');
    D.sideLabel(g, Bp, q, closed ? ctr : A, fmt(a), 'val c-purple');
    D.dot(g, A, 5, 'pivot'); D.dot(g, Bp, 5, 'pivot');
    D.label(g, A, 'A', ctr, 20); D.label(g, Bp, 'B', ctr, 20);
    if (closed) D.label(g, p, 'C', ctr, 22);
    place(hP, p); place(hQ, q);
    const angs = closed ? ` · הזוויות שיצאו: ${M(`∠A = ${fmtDeg(angleAt(Bp, A, p))}, ∠B = ${fmtDeg(angleAt(A, Bp, p))}, ∠C = ${fmtDeg(angleAt(A, p, Bp))}`)}` : '';
    lab.readout(closed ? `המשולש סגור${angs}` : 'גרור את הקצוות של המקלות עד שייפגשו.');
  }
  onDrag(lab.svg, hP, { move: p => { thA = V.ang(V.sub(p, A)); snap(); render(); } });
  onDrag(lab.svg, hQ, { move: p => { thB = V.ang(V.sub(p, B())); snap(); render(); } });
  function changed() {
    closed = false; lab.reset(); lab.status('');
    if (a + b <= c) lab.status(`שני המקלות ביחד (${fmt(a + b)}) לא ארוכים מ-AB (6). הם לעולם לא ייפגשו. ${noTri}`, 'bad');
    else if (Math.abs(a - b) >= c) lab.status(`מקל אחד ארוך מדי: גם כשהקצר פתוח עד הסוף, הם לא נפגשים. ${noTri}`, 'bad');
    snap(); render();
  }
  lab.onInput('a', v => { a = v; changed(); });
  lab.onInput('b', v => { b = v; changed(); });
  lab.onInput('circ', v => { showCirc = v; render(); });
  render();
}

/* ---------- צ.ז.צ ---------- */
function labSAS(host, done) {
  const U = 45, A = [195, 235], c = 6, B = [A[0] + c * U, A[1]];
  let b = 4.5, al = 50, th = -105 * RAD, closed = false;
  const lab = makeLab(host, {
    title: 'צ.ז.צ: שתי צלעות והזווית שביניהן', h: 470, U, done,
    intro: `הבסיס ${M('AB = 6')} קבוע. המקל הירוק ${M('AC')} מחובר ל-A בציר, כמו דלת. הקו המקווקו מ-B לקצה המקל הוא <b>גומייה</b>: היא נמתחת לבד ומראה כמה ארוכה הצלע השלישית. סובב את המקל עד שהזווית ב-A תהיה בדיוק הזווית הנתונה.`,
    controls: slider('b', 'אורך המקל הירוק (AC)', 2, 8, 0.5, b) + slider('al', 'הזווית הנתונה ב-A', 20, 160, 5, al),
    explain: `<p>שים לב מה קרה כשסובבת את המקל: ככל שהזווית ב-A גדלה, הגומייה BC נמתחה. הזווית היא ה"ידית" ששולטת בצלע השלישית. לכל זווית יש אורך אחד בדיוק של BC.</p>
      <p>אז ברגע שהזווית ב-A <b>ננעלת</b>, למקל יש רק מקום אחד (או תמונת המראה שלו, מתחת לבסיס). הקצה C קבוע, והצלע BC נקבעת לבד. אין שום חופש.</p>
      <p class="thm"><b>משפט צ.ז.צ:</b> אם שתי צלעות והזווית <b>שביניהן</b> במשולש אחד שוות לשתי צלעות והזווית שביניהן במשולש שני, המשולשים חופפים.</p>
      <p>למה הזווית חייבת להיות <b>בין</b> הצלעות? כי רק אז היא הציר שמחבר אותן. זווית במקום אחר לא נועלת את הציר, ובמעבדה של צ.צ.ז תראה מה קורה אז.</p>`,
  });
  const hC = lab.handle('h-green');
  const Cp = () => V.add(A, V.mul(V.unit(th), b * U));
  const angA = () => Math.abs(normAng(th)) * DEG;
  function snap() {
    const was = closed; closed = false;
    if (Math.abs(angA() - al) < 3) { th = (normAng(th) <= 0 ? -1 : 1) * al * RAD; closed = true; }
    if (closed && !was) lab.closeMsg(lab.record({ A, B, C: Cp() }));
  }
  function render() {
    const g = lab.scene; clearEl(g);
    const C = Cp(), ctr = centroid([A, B, C]);
    if (closed) D.poly(g, [A, B, C], 'tri-fill');
    D.seg(g, A, B, 'base');
    D.seg(g, B, C, closed ? 'band closed' : 'band');
    D.seg(g, A, C, 'stick green');
    D.arc(g, A, B, C, 1, 34, closed ? 'arc-lock' : 'arc-live', closed ? 'wedge-lock' : 'wedge-live');
    D.angLabel(g, A, B, C, 56, fmtDeg(angA()), closed ? 'val c-pink' : 'val');
    D.sideLabel(g, A, B, ctr, fmt(c), 'val');
    D.sideLabel(g, A, C, ctr, fmt(b), 'val c-green');
    D.sideLabel(g, B, C, ctr, fmt(V.dist(B, C) / U), 'val c-orange');
    D.dot(g, A, 5, 'pivot'); D.dot(g, B, 5, 'pivot');
    D.label(g, A, 'A', ctr, 20); D.label(g, B, 'B', ctr, 20); D.label(g, C, 'C', ctr, 22);
    place(hC, C);
    lab.readout(`הזווית ב-A עכשיו: <b>${fmtDeg(angA())}</b> (צריך ${al}°) · הגומייה ${M('BC')}: <b>${fmt(V.dist(B, C) / U)}</b>${closed ? ' · 🔒 נעול' : ''}`);
  }
  onDrag(lab.svg, hC, { move: p => { th = V.ang(V.sub(p, A)); snap(); render(); } });
  const changed = () => { closed = false; lab.reset(); lab.status(''); snap(); render(); };
  lab.onInput('b', v => { b = v; changed(); });
  lab.onInput('al', v => { al = v; changed(); });
  render();
}

/* ---------- ז.צ.ז ---------- */
function labASA(host, done) {
  const U = 45, A = [195, 235], c = 6, B = [A[0] + c * U, A[1]], HR = 115;
  let al = 50, be = 65, thA = -100 * RAD, thB = -60 * RAD, sA = false, sB = false, closed = false, C = null;
  const lab = makeLab(host, {
    title: 'ז.צ.ז: שתי זוויות והצלע שביניהן', h: 470, U, done,
    intro: `הבסיס ${M('AB = 6')} קבוע. מכל קצה יוצאת קרן לייזר, ואתה שולט בכיוון שלה בעזרת הידית. כוון את הקרן מ-A לזווית הנתונה ב-A, ואת הקרן מ-B לזווית הנתונה ב-B. איפה שהן נפגשות, שם C.`,
    controls: slider('al', 'הזווית הנתונה ב-A', 20, 140, 5, al) + slider('be', 'הזווית הנתונה ב-B', 20, 140, 5, be),
    explain: `<p>כל זווית נותנת <b>כיוון</b>: קרן שיוצאת מהקודקוד. אחרי ששתי הזוויות ננעלות, שתי הקרניים קבועות. ושני ישרים (שאינם מקבילים) נחתכים <b>בנקודה אחת בלבד</b>. אז C נקבע, ואיתו כל המשולש. אם תנסה את שתי הקרניים מתחת לבסיס, תקבל תמונת מראה, כלומר משולש חופף.</p>
      <p class="thm"><b>משפט ז.צ.ז:</b> אם שתי זוויות והצלע <b>שביניהן</b> במשולש אחד שוות לשתי זוויות והצלע שביניהן במשולש שני, המשולשים חופפים.</p>
      <p>ומה אם הצלע <b>לא</b> בין שתי הזוויות? אין בעיה: אם יודעים שתי זוויות, יודעים גם את השלישית (${M('180° -')} שתיהן). ואז יש לנו שתי זוויות והצלע שביניהן, כלומר שוב ז.צ.ז.</p>
      <p>ולמה חייבים את הצלע? נסה במעבדה ז.ז.ז: בלי אורך אחד לפחות, אין מה שיקבע את הגודל.</p>`,
  });
  const hA = lab.handle('h-green'), hB = lab.handle('h-purple');
  const angA = () => Math.abs(normAng(thA)) * DEG;
  const angB = () => 180 - Math.abs(normAng(thB)) * DEG;
  const sgn = x => (normAng(x) <= 0 ? -1 : 1);
  function snap() {
    const was = closed;
    sA = Math.abs(angA() - al) < 3; if (sA) thA = sgn(thA) * al * RAD;
    sB = Math.abs(angB() - be) < 3; if (sB) thB = sgn(thB) * (180 - be) * RAD;
    C = null; closed = false;
    if (sA && sB && sgn(thA) === sgn(thB)) {
      const r = lineLine(A, V.unit(thA), B, V.unit(thB));
      if (r && r.t > 0 && r.u > 0) { C = V.add(A, V.mul(V.unit(thA), r.t)); closed = true; }
    }
    if (closed && !was) lab.closeMsg(lab.record({ A, B, C }));
    else if (!closed) {
      if (al + be >= 180) lab.status(`סכום שתי הזוויות הוא ${al + be}°. הקרניים לעולם לא ייפגשו (הן מקבילות או מתרחקות). ${noTri}`, 'bad');
      else if (sA && sB) lab.status('שתי הקרניים נעולות, אבל בצדדים שונים של AB. כך הן לא ייפגשו. שים את שתיהן באותו צד.', 'info');
    }
  }
  function render() {
    const g = lab.scene; clearEl(g);
    const dA = V.unit(thA), dB = V.unit(thB);
    if (closed) D.poly(g, [A, B, C], 'tri-fill');
    D.seg(g, A, V.add(A, V.mul(dA, 1500)), sA ? 'ray green lock' : 'ray green');
    D.seg(g, B, V.add(B, V.mul(dB, 1500)), sB ? 'ray purple lock' : 'ray purple');
    D.seg(g, A, B, 'base');
    const pA = V.add(A, V.mul(dA, HR)), pB = V.add(B, V.mul(dB, HR));
    D.arc(g, A, B, pA, 1, 34, sA ? 'arc-lock' : 'arc-live', sA ? 'wedge-lock' : 'wedge-live');
    D.arc(g, B, A, pB, 1, 34, sB ? 'arc-lock' : 'arc-live', sB ? 'wedge-lock' : 'wedge-live');
    D.angLabel(g, A, B, pA, 56, fmtDeg(angA()), sA ? 'val c-pink' : 'val');
    D.angLabel(g, B, A, pB, 56, fmtDeg(angB()), sB ? 'val c-pink' : 'val');
    const ctr = closed ? centroid([A, B, C]) : [A[0] + c * U / 2, A[1] - 60];
    D.sideLabel(g, A, B, ctr, fmt(c), 'val');
    D.dot(g, A, 5, 'pivot'); D.dot(g, B, 5, 'pivot');
    D.label(g, A, 'A', ctr, 20); D.label(g, B, 'B', ctr, 20);
    if (closed) {
      D.dot(g, C, 6, 'meet'); D.label(g, C, 'C', ctr, 22);
      D.sideLabel(g, A, C, ctr, fmt(V.dist(A, C) / U), 'val c-green');
      D.sideLabel(g, B, C, ctr, fmt(V.dist(B, C) / U), 'val c-purple');
    }
    place(hA, pA); place(hB, pB);
    lab.readout(`${M('∠A')}: <b>${fmtDeg(angA())}</b> (צריך ${al}°) · ${M('∠B')}: <b>${fmtDeg(angB())}</b> (צריך ${be}°)`);
  }
  onDrag(lab.svg, hA, { move: p => { thA = V.ang(V.sub(p, A)); snap(); render(); } });
  onDrag(lab.svg, hB, { move: p => { thB = V.ang(V.sub(p, B)); snap(); render(); } });
  const changed = () => { closed = false; lab.reset(); lab.status(''); snap(); render(); };
  lab.onInput('al', v => { al = v; changed(); });
  lab.onInput('be', v => { be = v; changed(); });
  render();
}

/* ---------- צ.צ.ז: המעבדה החשובה ---------- */
function labSSA(host, done) {
  const U = 40, A = [150, 335], c = 6, B = [A[0] + c * U, A[1]], al = 35;
  const dA = V.unit(-al * RAD);
  let a = 4.3, th = -60 * RAD, closed = false, behind = null, showCirc = false;
  const lab = makeLab(host, {
    title: 'צ.צ.ז: שתי צלעות וזווית שלא ביניהן', h: 420, U, done,
    intro: `נתונים: ${M('AB = 6')}, הזווית ${M('∠A = 35°')} (הקרן הירוקה, קבועה), ומקל סגול ${M('BC')} שמחובר ל-B בציר. שים לב: הזווית ב-A <b>לא</b> נמצאת בין AB ל-BC, היא נמצאת <b>מול</b> BC. גרור את קצה המקל הסגול עד שהוא נוגע בקרן, והמשולש ייסגר.`,
    controls: slider('a', 'אורך המקל הסגול (BC)', 2.5, 7.5, 0.1, a) + checkbox('circ', 'הראה את כל המקומות שהקצה יכול להגיע אליהם'),
    winMsg: '😮 <b>הצלחת!</b> שני משולשים <b>שונים</b>, ובשניהם אותם נתונים בדיוק: אותו AB, אותו BC ואותה זווית ב-A. אז כאן שתי צלעות וזווית <b>לא</b> מספיקות!<br>עכשיו הזז את המחוון כך ש-BC יהיה <b>ארוך מ-AB</b> (יותר מ-6), ונסה שוב למצוא שני משולשים שונים.',
    explain: `<p>הקודקוד C חייב להיות על הקרן (בגלל הזווית ב-A), וגם על <b>מעגל</b> סביב B (בגלל האורך של BC). וישר ומעגל יכולים להיחתך <b>בשתי נקודות</b>. בגלל זה אפשר לקבל שני משולשים שונים.</p>
      <p>אבל שים לב מתי זה קורה:</p>
      <ul>
        <li>כש-BC <b>קצר</b> מ-AB: שתי נקודות החיתוך נופלות על הקרן. שני משולשים שונים. 😈</li>
        <li>כש-BC <b>ארוך</b> מ-AB (או שווה לו): המעגל כל כך גדול, שנקודת החיתוך השנייה בורחת <b>אל מאחורי A</b>, על ההמשך המקווקו של הקרן. שם כבר אין זווית של 35° ב-A (יש 145°), אז היא לא נחשבת. נשאר רק משולש אחד. 🏆</li>
      </ul>
      <p>הזווית A נמצאת מול הצלע BC. אז התנאי "BC ארוך מ-AB" הוא בדיוק התנאי "הזווית נמצאת מול הצלע הגדולה".</p>
      <p class="thm"><b>משפט צ.צ.ז:</b> אם שתי צלעות והזווית <b>שמול הגדולה מביניהן</b> במשולש אחד שוות לאלה שבמשולש שני, המשולשים חופפים.</p>
      <p>עכשיו ברור למה המשפט נשמע כל כך מסובך. זה לא סתם תנאי שמישהו המציא: בלעדיו המשפט פשוט לא נכון.</p>`,
  });
  const hC = lab.handle('h-purple');
  const ts = () => lineCircle(A, dA, B, a * U);
  function snap() {
    const was = closed; closed = false; behind = null;
    const raw = V.add(B, V.mul(V.unit(th), a * U));
    for (const t of ts()) {
      const pt = V.add(A, V.mul(dA, t));
      if (V.dist(raw, pt) < 14) {
        th = V.ang(V.sub(pt, B));
        if (t > 8) closed = true; else if (t < -8) behind = pt;
      }
    }
    if (closed && !was) lab.closeMsg(lab.record({ A, B, C: V.add(B, V.mul(V.unit(th), a * U)) }));
    if (behind) lab.status('🔙 הקצה נוגע בהמשך של הקרן, <b>מאחורי A</b>. אבל שם הזווית ב-A היא לא 35°, אלא 145°. זה משולש עם נתונים אחרים, אז הוא לא נחשב.', 'info');
  }
  function render() {
    const g = lab.scene; clearEl(g);
    const C = V.add(B, V.mul(V.unit(th), a * U));
    if (showCirc) {
      S('circle', { cx: B[0], cy: B[1], r: a * U, class: 'locus purple' }, g);
      for (const t of ts()) D.dot(g, V.add(A, V.mul(dA, t)), 6, t > 0 ? 'meet' : 'meet off');
    }
    const ctr = closed ? centroid([A, B, C]) : [A[0] + c * U / 2, A[1] - 60];
    if (closed) D.poly(g, [A, B, C], 'tri-fill');
    D.seg(g, A, V.sub(A, V.mul(dA, 400)), 'ray back');
    D.seg(g, A, V.add(A, V.mul(dA, 1500)), 'ray green lock');
    D.seg(g, V.sub(A, [200, 0]), A, 'ray back');
    D.seg(g, A, B, 'base');
    D.seg(g, B, C, 'stick purple');
    const pA = V.add(A, V.mul(dA, 100));
    D.arc(g, A, B, pA, 1, 40, 'arc-lock', 'wedge-lock');
    D.angLabel(g, A, B, pA, 62, '35°', 'val c-pink');
    D.sideLabel(g, A, B, ctr, fmt(c), 'val');
    D.sideLabel(g, B, C, closed ? ctr : A, fmt(a), 'val c-purple');
    D.dot(g, A, 5, 'pivot'); D.dot(g, B, 5, 'pivot');
    D.label(g, A, 'A', [A[0] + 30, A[1] - 30], 20); D.label(g, B, 'B', ctr, 20);
    if (closed) D.label(g, C, 'C', ctr, 22);
    if (behind) D.dot(g, behind, 7, 'meet off');
    place(hC, C);
    const n = ts().filter(t => t > 8).length;
    lab.readout(showCirc
      ? `${M('BC = ' + fmt(a))}, ${M('AB = 6')} · המעגל חותך את הקרן ב-<b>${n}</b> ${n === 1 ? 'נקודה' : 'נקודות'}`
      : `${M('BC = ' + fmt(a))}, ${M('AB = 6')}, ${M('∠A = 35°')}`);
  }
  onDrag(lab.svg, hC, { move: p => { th = V.ang(V.sub(p, B)); snap(); render(); } });
  lab.onInput('a', v => {
    a = v; closed = false; lab.reset();
    const h = c * Math.sin(al * RAD);
    if (a < h - 0.02) lab.status(`המקל BC קצר מדי: הוא לא מגיע לקרן בכלל. ${noTri}`, 'bad');
    else if (a > c + 0.01) lab.status('עכשיו BC ארוך מ-AB. נסה למצוא שני משולשים שונים...', 'info');
    else lab.status('');
    snap(); render();
  });
  lab.onInput('circ', v => { showCirc = v; render(); });
  render();
}

/* ---------- ז.ז.ז: בונוס ---------- */
function labAAA(host, done) {
  const U = 40, A = [130, 390], al = 50, be = 60;
  let bx = A[0] + 240, ghost = null;
  const lab = makeLab(host, {
    title: 'ז.ז.ז: שלוש זוויות (בונוס)', h: 420, U, done,
    intro: `כאן כל שלוש הזוויות קבועות: ${M('∠A = 50°, ∠B = 60°')}, ולכן גם ${M('∠C = 70°')}. אבל אף צלע לא נתונה. גרור את B ימינה או שמאלה וראה מה קורה. אפשר לבנות שני משולשים שונים?`,
    winMsg: '😮 מצאת שני משולשים עם <b>אותן שלוש זוויות</b>, אבל בגדלים שונים. הם לא חופפים!',
    explain: `<p>הזוויות קובעות את <b>הצורה</b> של המשולש, אבל לא את <b>הגודל</b> שלו. כמו תמונה שעושים לה זום: הכל גדל ביחד, והזוויות נשארות אותו דבר.</p>
      <p>משולשים עם אותן זוויות נקראים <b>משולשים דומים</b>. דמיון זה נושא שלם שתלמד בהמשך. אבל חפיפה? בשביל זה חייבים לפחות צלע אחת, שתקבע את הגודל.</p>
      <p class="thm">אין משפט חפיפה ז.ז.ז.</p>`,
  });
  const hB = lab.handle('h-purple');
  const dA = V.unit(-al * RAD), dB = V.unit(-(180 - be) * RAD);
  const tri = () => {
    const B = [bx, A[1]], r = lineLine(A, dA, B, dB);
    return { A, B, C: V.add(A, V.mul(dA, r.t)) };
  };
  function render() {
    const g = lab.scene; clearEl(g);
    const T = tri(), ctr = centroid([T.A, T.B, T.C]);
    if (ghost) D.poly(g, [ghost.A, ghost.B, ghost.C], 'ghost-tri');
    D.poly(g, [T.A, T.B, T.C], 'tri-fill');
    D.seg(g, [60, A[1]], [700, A[1]], 'ray back');
    D.seg(g, T.A, T.C, 'stick green'); D.seg(g, T.B, T.C, 'stick purple'); D.seg(g, T.A, T.B, 'base');
    D.arc(g, T.A, T.B, T.C, 1, 30, 'arc-lock', 'wedge-lock'); D.angLabel(g, T.A, T.B, T.C, 50, '50°', 'val c-pink');
    D.arc(g, T.B, T.A, T.C, 1, 30, 'arc-lock', 'wedge-lock'); D.angLabel(g, T.B, T.A, T.C, 50, '60°', 'val c-pink');
    D.arc(g, T.C, T.A, T.B, 1, 26, 'arc-lock', 'wedge-lock'); D.angLabel(g, T.C, T.A, T.B, 46, '70°', 'val c-pink');
    D.sideLabel(g, T.A, T.B, ctr, fmt(V.dist(T.A, T.B) / U), 'val');
    D.sideLabel(g, T.A, T.C, ctr, fmt(V.dist(T.A, T.C) / U), 'val c-green');
    D.sideLabel(g, T.B, T.C, ctr, fmt(V.dist(T.B, T.C) / U), 'val c-purple');
    D.label(g, T.A, 'A', ctr, 20); D.label(g, T.B, 'B', ctr, 20); D.label(g, T.C, 'C', ctr, 22);
    place(hB, T.B);
    lab.readout(`${M('AB = ' + fmt(V.dist(T.A, T.B) / U))} · הזוויות לא זזות: 50°, 60°, 70°`);
  }
  onDrag(lab.svg, hB, {
    move: p => { bx = clamp(p[0], A[0] + 100, A[0] + 440); render(); },
    end: () => { const T = tri(), k = lab.record(T); if (!ghost) ghost = T; lab.closeMsg(k); render(); },
  });
  lab.status('גרור את הנקודה B.', 'info');
  render();
}
