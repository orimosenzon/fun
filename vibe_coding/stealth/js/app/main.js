import { View3D } from './view3d.js';
import { plot, fitCanvas } from './plot.js';
import { PRESETS, clonePreset } from '../core/geometry.js';
import * as M from '../core/model.js';
import * as PO from '../core/po.js';
import { COATINGS, gammaDb } from '../core/ram.js';
import { aero } from '../core/aero.js';
import * as IR from '../core/ir.js';
import { initMission } from './mission.js';

const $ = (s) => document.querySelector(s);
const state = {
  p: clonePreset('fighter'), preset: 'fighter', band: 'X', el: -3, az: 20,
  opts: { bayOpen: false, external: false }, polar: null, ref: {}, model: null,
};

// ---------- עובד רקע ----------
const worker = new Worker(new URL('../core/worker.js', import.meta.url), { type: 'module' });
let reqId = 0; const pending = new Map(), latest = {};
worker.onmessage = (e) => {
  const d = e.data, h = pending.get(d.id);
  if (!h) return;
  if (d.progress) { h.progress && h.progress(d.progress); return; }
  pending.delete(d.id); busy(-1);
  if (d.error) { console.error(d.error); return; }
  if (latest[h.type] !== d.id && !h.keep) return; // תוצאה ישנה
  h.done(d);
};
let nBusy = 0;
function busy(dn) { nBusy += dn; $('#busy').classList.toggle('on', nBusy > 0); }
export function ask(type, p, opts, done, extra = {}) {
  const id = ++reqId; latest[type + (extra.tag || '')] = id;
  pending.set(id, { type: type + (extra.tag || ''), done, keep: extra.keep, progress: extra.progress });
  busy(1);
  worker.postMessage({ id, type, p, opts });
}

// ---------- בקרות ----------
const GROUPS = [
  { title: 'גוף', open: true, items: [
    { k: 'fus.section', label: 'חתך הגוף', type: 'sel', opts: { round: 'עגול (קונבנציונלי)', chine: 'צלע חדה (chine) ודפנות מוטות' }, why: 'גליל מחזיר לכל הצדדים. פאה שטוחה מחזירה רק לכיוון אחד.' },
    { k: 'fus.side', label: 'הטיית הדפנות מהאנך', min: 0, max: 60, step: 1, unit: '°', why: 'דופן אנכית היא מראה למכ"ם קרקעי. הטיה שולחת את ההחזרה לשמיים.', show: (p) => p.fus.section === 'chine' },
    { k: 'L', label: 'אורך', min: 12, max: 30, step: 0.1, unit: ' מ\'' },
    { k: 'fus.W', label: 'חצי רוחב', min: 0.8, max: 3.5, step: 0.05, unit: ' מ\'' },
    { k: 'fus.HU', label: 'גובה מעל הצלע', min: 0.4, max: 2.2, step: 0.05, unit: ' מ\'' },
    { k: 'fus.HL', label: 'עומק מתחת לצלע', min: 0.3, max: 1.6, step: 0.05, unit: ' מ\'' },
    { k: 'fus.noseFrac', label: 'אורך האף', min: 0.15, max: 0.5, step: 0.01, unit: '', fmt: (v) => Math.round(v * 100) + '%' },
  ] },
  { title: 'כנף', open: true, items: [
    { k: 'wing.sweep', label: 'הספנה של הקצה המוביל', min: 0, max: 75, step: 0.5, unit: '°', why: 'הקוץ של הקצה המוביל יוצא בניצב אליו. ספנה של 40° ומעלה מרחיקה אותו מהאף.' },
    { k: 'wing.span', label: 'מוטה', min: 6, max: 60, step: 0.1, unit: ' מ\'' },
    { k: 'wing.cr', label: 'מיתר בשורש', min: 2, max: 24, step: 0.1, unit: ' מ\'' },
    { k: 'wing.ct', label: 'מיתר בקצה', min: 0.3, max: 6, step: 0.05, unit: ' מ\'', show: (p) => p.wing.te !== 'saw' },
    { k: 'wing.te', label: 'שפת הנגרר', type: 'sel', opts: { straight: 'ישרה', saw: 'שיני מסור (מקבילות לקצה המוביל)' } },
    { k: 'wing.teeth', label: 'מספר קטעים', min: 1, max: 6, step: 1, unit: '', show: (p) => p.wing.te === 'saw' },
    { k: 'wing.x0', label: 'מיקום הכנף מהאף', min: 0, max: 0.6, step: 0.01, fmt: (v) => Math.round(v * 100) + '%' },
    { k: 'wing.t', label: 'עובי יחסי', min: 0.02, max: 0.12, step: 0.005, fmt: (v) => (v * 100).toFixed(1) + '%' },
  ] },
  { title: 'זנבות', open: false, items: [
    { k: 'tail.type', label: 'מייצבי כיוון', type: 'sel', opts: { none: 'אין (כנף מעופפת)', single: 'זנב אנכי יחיד', twin: 'זוג זנבות' }, why: 'זנב אנכי ליד דופן אנכית יוצרים פינה של 90°: מחזיר פינתי.' },
    { k: 'tail.cant', label: 'הטיית הזנבות מהאנך', min: 0, max: 70, step: 1, unit: '°', show: (p) => p.tail.type === 'twin' },
    { k: 'tail.h', label: 'גובה הזנב', min: 1, max: 5, step: 0.05, unit: ' מ\'', show: (p) => p.tail.type !== 'none' },
    { k: 'tail.cr', label: 'מיתר הזנב', min: 1, max: 5, step: 0.05, unit: ' מ\'', show: (p) => p.tail.type !== 'none' },
    { k: 'tail.sweep', label: 'ספנת הזנב', min: 0, max: 75, step: 0.5, unit: '°', show: (p) => p.tail.type !== 'none' && !p.align },
    { k: 'hstab.on', label: 'מייצב גובה', type: 'chk' },
    { k: 'hstab.span', label: 'מוטת המייצב', min: 4, max: 12, step: 0.1, unit: ' מ\'', show: (p) => p.hstab.on },
    { k: 'align', label: 'יישור קצוות: כל הקצוות באותה זווית כמו הכנף', type: 'chk' },
  ] },
  { title: 'פתחים, ציפוי ומערכות', open: true, items: [
    { k: 'coat', label: 'ציפוי בולע (RAM)', type: 'sel', opts: Object.fromEntries(Object.entries(COATINGS).map(([k, v]) => [k, v.name])) },
    { k: 'inlet', label: 'פתחי האוויר', type: 'sel', opts: Object.fromEntries(Object.entries(M.INLETS).map(([k, v]) => [k, v.name])), why: 'פני המדחס הם מחזיר ענק. צריך להסתיר אותם מקו הראייה.' },
    { k: 'canopy', label: 'כיפת תא הטייס', type: 'sel', opts: Object.fromEntries(Object.entries(M.CANOPIES).map(([k, v]) => [k, v.name])) },
    { k: 'antenna', label: 'אנטנת המכ"ם באף', type: 'sel', opts: Object.fromEntries(Object.entries(M.ANTENNAS).map(([k, v]) => [k, v.name])) },
    { k: 'nozzleType', label: 'נחיר הפליטה', type: 'sel', opts: Object.fromEntries(Object.entries(M.NOZZLES).map(([k, v]) => [k, v.name])) },
  ] },
];

const getK = (o, k) => k.split('.').reduce((a, b) => a[b], o);
const setK = (o, k, v) => { const ks = k.split('.'); const last = ks.pop(); ks.reduce((a, b) => a[b], o)[last] = v; };

function buildControls() {
  const pr = $('#presets'); pr.innerHTML = '';
  for (const [k, v] of Object.entries(PRESETS)) {
    const b = document.createElement('button'); b.className = 'btn' + (state.preset === k ? ' on' : ''); b.textContent = v.name;
    b.onclick = () => { state.p = clonePreset(k); state.preset = k; buildControls(); changed(); };
    pr.appendChild(b);
  }
  const root = $('#groups'); root.innerHTML = '';
  for (const g of GROUPS) {
    const det = document.createElement('details'); det.className = 'grp'; det.open = g.open;
    det.ontoggle = () => { g.open = det.open; };
    det.innerHTML = `<summary>${g.title}</summary>`;
    for (const it of g.items) {
      if (it.show && !it.show(state.p)) continue;
      const v = getK(state.p, it.k);
      if (it.type === 'chk') {
        const l = document.createElement('label'); l.className = 'chk';
        l.innerHTML = `<input type="checkbox" ${v ? 'checked' : ''}> ${it.label}`;
        l.querySelector('input').onchange = (e) => { setK(state.p, it.k, e.target.checked); custom(); buildControls(); changed(); };
        det.appendChild(l); continue;
      }
      const d = document.createElement('div'); d.className = 'ctl';
      if (it.type === 'sel') {
        d.innerHTML = `<label>${it.label}</label><select>${Object.entries(it.opts).map(([k, n]) => `<option value="${k}" ${k === v ? 'selected' : ''}>${n}</option>`).join('')}</select>` + (it.why ? `<div class="why">${it.why}</div>` : '');
        d.querySelector('select').onchange = (e) => { setK(state.p, it.k, e.target.value); custom(); buildControls(); changed(); };
      } else {
        const fmt = it.fmt || ((x) => (+x).toFixed(it.step < 0.1 ? 2 : it.step < 1 ? 1 : 0) + (it.unit || ''));
        d.innerHTML = `<label>${it.label} <b>${fmt(v)}</b></label><input type="range" min="${it.min}" max="${it.max}" step="${it.step}" value="${v}">` + (it.why ? `<div class="why">${it.why}</div>` : '');
        const inp = d.querySelector('input'), lab = d.querySelector('b');
        inp.oninput = () => { setK(state.p, it.k, +inp.value); lab.textContent = fmt(+inp.value); custom(); changed(); };
      }
      det.appendChild(d);
    }
    root.appendChild(det);
  }
}
function custom() { if (state.preset) { state.preset = null; document.querySelectorAll('#presets .btn').forEach((b) => b.classList.remove('on')); } }

// ---------- תצוגה ----------
const view = new View3D($('#view'));
document.querySelectorAll('.overlay .btn').forEach((b) => {
  b.onclick = () => {
    const k = b.dataset.v;
    if (k === 'parts' || k === 'heat') {
      document.querySelectorAll('.overlay [data-v=parts],.overlay [data-v=heat]').forEach((x) => x.classList.toggle('on', x === b));
      view.mode.heat = k === 'heat';
      if (k === 'parts') view.colorParts(); else requestHeat();
      legend();
      return;
    }
    b.classList.toggle('on'); view.setMode(k, b.classList.contains('on'));
    if (k === 'bubble' && view.mode.bubble) requestBubble();
    legend();
  };
});
function legend() {
  const parts = 'כחול כהה: גוף · כחול: כנף · סגול: זנבות · טורקיז: מייצב · כתום: פתח הפליטה';
  const t = [];
  t.push(view.mode.heat ? 'צבע הפאה = כמה היא מחזירה לכיוון המכ"ם (אדום חזק, כחול חלש, אפור מוצל).' : parts);
  if (view.mode.bubble) t.push(`הבועה: σ לכל כיוון בתדר ${state.band}, סכום לא קוהרנטי של הפאות. ההיקף מראה לאן הולכת האנרגיה.`);
  if (view.mode.edges) t.push('קווים ירוקים: קצוות חדים שמפזרים (עקיפה).');
  $('#legend3d').textContent = t.join(' ');
}

function radarDir() { return PO.dirFromAzEl(state.az, state.el); }
$('#az').oninput = (e) => { state.az = +e.target.value; radarMoved(); };
$('#el').oninput = (e) => { state.el = +e.target.value; radarMoved(); requestPolar(); };
function radarMoved() {
  $('#azv').textContent = state.az + '°'; $('#elv').textContent = state.el + '°';
  $('#az').value = state.az; $('#el').value = state.el;
  view.setRadar(radarDir());
  drawPolar();
  if (view.mode.heat) requestHeat();
}

// ---------- תדרים ----------
const bandsEl = $('#bands');
for (const b of Object.keys(M.BANDS)) {
  const btn = document.createElement('button'); btn.className = 'btn' + (b === state.band ? ' on' : ''); btn.textContent = b;
  btn.title = M.BANDS[b].label + ': ' + M.BANDS[b].use;
  btn.onclick = () => { state.band = b; bandsEl.querySelectorAll('.btn').forEach((x) => x.classList.toggle('on', x === btn)); requestPolar(); if (view.mode.bubble) requestBubble(); if (view.mode.heat) requestHeat(); drawRam(); legend(); };
  bandsEl.appendChild(btn);
}

// ---------- בקשות ----------
function requestPolar() {
  ask('polar', state.p, { band: state.band, el: state.el, n: 1440, win: 10, ...state.opts }, (d) => { state.polar = d; drawPolar(); updateNumbers(); });
  const rk = state.band + '|' + state.el;
  if (!state.ref[rk]) ask('polar', PRESETS.conventional, { band: state.band, el: state.el, n: 720, win: 10 }, (d) => { state.ref[rk] = d; drawPolar(); }, { tag: 'ref' });
}
function requestBubble() { ask('bubble', state.p, { band: state.band, na: 72, ne: 37 }, (d) => view.setBubble(d.out, d.na, d.ne)); }
function requestHeat() { ask('facets', state.p, { band: state.band, r: radarDir() }, (d) => { if (view.mode.heat) view.colorHeat(d.out); }); }

let tmr = 0;
function changed() {
  clearTimeout(tmr);
  tmr = setTimeout(() => {
    state.model = M.build(state.p);
    invalidateFront(); state.below = null;
    view.setModel(state.model.ac, state.model.P);
    if (view.mode.heat) requestHeat();
    requestPolar();
    if (view.mode.bubble) requestBubble();
    drawShape(); drawRam(); drawAero(); drawIR(); drawMore();
    mission.designChanged();
  }, 90);
}

// ---------- תרשים קוטבי ----------
const DB_MIN = -45, DB_MAX = 30;
const polarCv = $('#polar');
function drawPolar() {
  const { ctx, W, H } = fitCanvas(polarCv);
  const cx = W / 2, cy = H / 2, R = Math.min(W, H) / 2 - 18;
  ctx.clearRect(0, 0, W, H);
  const rOf = (db) => R * Math.max(0, (Math.min(DB_MAX, db) - DB_MIN) / (DB_MAX - DB_MIN));
  const pt = (azDeg, r) => { const a = azDeg * Math.PI / 180; return [cx - r * Math.sin(a), cy - r * Math.cos(a)]; };
  ctx.font = '10.5px JetBrains Mono'; ctx.direction = 'ltr';
  for (let db = DB_MIN + 5; db <= DB_MAX; db += 10) {
    ctx.strokeStyle = db === 0 ? '#2e4572' : '#18243f'; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.arc(cx, cy, rOf(db), 0, 7); ctx.stroke();
    ctx.fillStyle = '#5a6d92'; ctx.textAlign = 'left'; ctx.fillText(db + ' dBsm', cx + 3, cy - rOf(db) + 11);
  }
  for (let a = 0; a < 360; a += 30) { const [x, y] = pt(a, R); ctx.strokeStyle = '#18243f'; ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(x, y); ctx.stroke(); }
  ctx.fillStyle = '#8a9bb9'; ctx.font = '12px Rubik'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.direction = 'rtl';
  [[0, 'אף'], [90, 'שמאל'], [180, 'זנב'], [270, 'ימין']].forEach(([a, t]) => { const [x, y] = pt(a, R + 10); ctx.fillText(t, x, y); });
  ctx.direction = 'ltr';
  const line = (arr, color, w, alpha = 1) => {
    if (!arr) return; const n = arr.length; ctx.strokeStyle = color; ctx.lineWidth = w; ctx.globalAlpha = alpha; ctx.beginPath();
    for (let i = 0; i <= n; i++) { const [x, y] = pt(i * 360 / n, rOf(PO.dB(arr[i % n]))); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); }
    ctx.stroke(); ctx.globalAlpha = 1;
  };
  const ref = state.ref[state.band + '|' + state.el];
  if (ref) line(ref.sm, '#ff8a5c', 1.5, 0.55);
  if (state.polar) { line(state.polar.raw, '#4fd1ff', 0.6, 0.35); line(state.polar.sm, '#7cf0c5', 2.4); }
  // סמן המכ"ם
  const [x, y] = pt(state.az, R);
  ctx.strokeStyle = '#ffd34d'; ctx.setLineDash([4, 4]); ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(x, y); ctx.stroke(); ctx.setLineDash([]);
  ctx.fillStyle = '#ffd34d'; ctx.beginPath(); ctx.arc(x, y, 5, 0, 7); ctx.fill();
  // מטוס קטן במרכז
  ctx.fillStyle = '#dfe8f7'; ctx.beginPath(); ctx.moveTo(cx, cy - 9); ctx.lineTo(cx + 7, cy + 6); ctx.lineTo(cx, cy + 3); ctx.lineTo(cx - 7, cy + 6); ctx.closePath(); ctx.fill();
  if (state.polar) {
    const n = state.polar.sm.length, i = Math.round(((state.az + 360) % 360) / 360 * n) % n;
    const s = state.polar.sm[i];
    $('#polarNote').innerHTML = `ירוק: ממוצע ב-10°, כחול חיוור: החישוב הגולמי (קוהרנטי), כתום: מטוס קונבנציונלי לייחוס. בכיוון המכ"ם: <b style="direction:ltr;display:inline-block">${PO.dB(s).toFixed(1)} dBsm = ${fmtSig(s)}</b>. ${M.BANDS[state.band].label}, ${state.polar.ms.toFixed(0)} מ"ש ל-1,440 כיוונים.`;
  }
}
polarCv.addEventListener('pointerdown', (e) => {
  const r = polarCv.getBoundingClientRect(), x = e.clientX - r.left - r.width / 2, y = e.clientY - r.top - r.height / 2;
  state.az = Math.round(Math.atan2(-x, -y) * 180 / Math.PI); radarMoved();
});

export function fmtSig(s) {
  if (s >= 100) return s.toFixed(0) + ' מ"ר';
  if (s >= 1) return s.toFixed(1) + ' מ"ר';
  if (s >= 0.01) return s.toFixed(3) + ' מ"ר';
  return s.toExponential(1) + ' מ"ר';
}
const EQUIV = [[100, 'מפציץ ישן'], [10, 'מטוס קרב'], [1, 'אדם'], [0.1, 'ציפור גדולה'], [0.01, 'ציפור קטנה'], [0.001, 'דבורה'], [0, 'גולת זכוכית']];
export const equiv = (s) => EQUIV.find(([v]) => s >= v * 0.5)[1];

function updateNumbers() {
  const pol = state.polar; if (!pol) return;
  const fr = PO.sectorMean(pol.raw, 0, 30), sd = (PO.sectorMean(pol.raw, 90, 15) + PO.sectorMean(pol.raw, 270, 15)) / 2, rr = PO.sectorMean(pol.raw, 180, 30);
  $('#aspect').innerHTML = [['חזית ±30°', fr], ['צד ±15°', sd], ['אחור ±30°', rr]].map(([t, s]) =>
    `<div><span>${t}</span><b>${PO.dB(s).toFixed(1)}</b><i>dBsm · ${equiv(s)}</i></div>`).join('');
  // טווחים: לכל מכ"ם בתדר שלו צריך σ בתדר שלו. בתדר הנוכחי מציגים את מה שיש, ואת השאר מבקשים
  rangesFor(fr);
}
function rangesFor() {
  const out = [];
  for (const [k, r] of Object.entries(M.RADARS)) {
    const key = r.band + '|' + state.el;
    const s = r.band === state.band && state.polar ? PO.sectorMean(state.polar.raw, 0, 30) : (state.front && state.front[key]);
    if (s == null) { ask('polar', state.p, { band: r.band, el: state.el, n: 720, win: 10, ...state.opts }, (d) => { state.front = state.front || {}; state.front[key] = PO.sectorMean(d.raw, 0, 30); rangesFor(); }, { tag: 'rng' + k }); continue; }
    const R = M.detectRange(r.R1, s);
    out.push(`<div class="row"><div>${r.name} <span class="tiny">(${r.band}${r.guide ? '' : ', גילוי בלבד'})</span><div class="bar"><i style="width:${Math.min(100, R / r.R1 * 100 / 1.8)}%"></i></div></div><b>${R.toFixed(0)} ק"מ</b></div>`);
  }
  $('#ranges').innerHTML = out.join('') + `<div class="tiny">R₁ = הטווח מול מטרה של 1 מ"ר. מטוס קונבנציונלי של 10 מ"ר נראה מ-1.8·R₁.</div>`;
}
function invalidateFront() { state.front = {}; }

// ---------- לשוניות ----------
document.querySelectorAll('#tabs button').forEach((b) => b.onclick = () => {
  document.querySelectorAll('#tabs button').forEach((x) => x.classList.toggle('on', x === b));
  document.querySelectorAll('.tabpane').forEach((x) => x.classList.toggle('on', x.id === 't-' + b.dataset.t));
  state.tab = b.dataset.t;
  drawShape(); drawRam(); drawAero(); drawIR(); drawMore();
  if (b.dataset.t === 'mission') mission.show();
});

// ---- צורה: מבט על וקוצי הקצוות ----
function drawShape() {
  if (!state.model || state.tab && state.tab !== 'shape') return;
  const cv = $('#plan'), { ctx, W, H } = fitCanvas(cv), P = state.model.P, p = state.model.ac.p;
  ctx.clearRect(0, 0, W, H);
  const span = Math.max(p.L, p.wing.span) * 1.08, sc = Math.min(W * 0.62, H) / span;
  const cx = W * 0.36, cy = H / 2;
  const X = (y) => cx - y * sc, Y = (x) => cy - x * sc;
  // צללית: כל המשולשים מלמעלה
  ctx.fillStyle = '#1a2846';
  for (let t = 0; t < P.nt; t++) {
    if (P.n[3 * t + 2] <= 0.05) continue;
    const v = P.v, o = 9 * t; ctx.beginPath();
    ctx.moveTo(X(v[o + 1]), Y(v[o])); ctx.lineTo(X(v[o + 4]), Y(v[o + 3])); ctx.lineTo(X(v[o + 7]), Y(v[o + 6])); ctx.fill();
  }
  // קצוות, צבע לפי כיוון הניצב במישור
  const rose = new Float64Array(72);
  for (const e of P.edges) {
    const tx = e.t[0], ty = e.t[1], h = Math.hypot(tx, ty); if (h < 0.5) continue; // רק קצוות כמעט אופקיים
    let ang = Math.atan2(tx, -ty) * 180 / Math.PI; // כיוון הניצב, 0 = אף
    ang = ((ang % 180) + 180) % 180;
    const strength = (e.w * e.L * h) ** 2;
    // הקוץ יוצא לשני צידי הקצה, אבל נראה רק מצד שבו לפחות אחת הפאות הסמוכות מוארת
    for (const a of [ang, ang + 180]) {
      const r = [Math.cos(a * Math.PI / 180), Math.sin(a * Math.PI / 180), -0.05];
      const s1 = e.n1[0] * r[0] + e.n1[1] * r[1] + e.n1[2] * r[2], s2 = e.n2[0] * r[0] + e.n2[1] * r[1] + e.n2[2] * r[2];
      if (s1 < -0.02 && s2 < -0.02) continue;
      rose[Math.floor(((a + 360) % 360) / 5) % 72] += strength;
    }
    ctx.strokeStyle = `hsl(${ang * 2},85%,62%)`; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(X(e.a[1]), Y(e.a[0])); ctx.lineTo(X(e.b[1]), Y(e.b[0])); ctx.stroke();
  }
  // פרח הקוצים
  const rx = W * 0.83, ry = H / 2, RR = Math.min(W * 0.15, H * 0.42);
  const mx = Math.max(...rose) || 1;
  ctx.strokeStyle = '#22314f'; ctx.beginPath(); ctx.arc(rx, ry, RR, 0, 7); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(rx, ry - RR - 4); ctx.lineTo(rx, ry + RR + 4); ctx.moveTo(rx - RR - 4, ry); ctx.lineTo(rx + RR + 4, ry); ctx.stroke();
  let distinct = 0;
  for (let i = 0; i < 72; i++) {
    if (rose[i] <= 0) continue;
    const u = Math.sqrt(rose[i] / mx); if (u > 0.3) distinct++;
    const a = (i * 5 + 2.5) * Math.PI / 180;
    ctx.strokeStyle = `hsl(${((i * 5 + 2.5) % 180) * 2},85%,62%)`; ctx.lineWidth = 3;
    ctx.beginPath(); ctx.moveTo(rx, ry); ctx.lineTo(rx - RR * u * Math.sin(a), ry - RR * u * Math.cos(a)); ctx.stroke();
  }
  ctx.fillStyle = '#8a9bb9'; ctx.font = '12px Rubik'; ctx.textAlign = 'center'; ctx.direction = 'rtl';
  ctx.fillText('אף ▲', rx, ry - RR - 10);
  ctx.fillText('פרח הקוצים', rx, ry + RR + 18);
  ctx.fillText('▲ אף', cx, 14);
  // כמה מהקוצים בגזרת האף
  let nose = 0, all = 0;
  for (let i = 0; i < 72; i++) { all += rose[i]; const a = i * 5 + 2.5; if (a < 30 || a > 330) nose += rose[i]; }
  ctx.fillStyle = nose / all > 0.05 ? '#ff5d73' : '#5ef0a0';
  ctx.fillText(`בגזרת האף (±30°): ${(100 * nose / (all || 1)).toFixed(1)}% מהקוצים`, rx, 16);
  ctx.direction = 'ltr';
}

function drawRam() {
  if (state.tab && state.tab !== 'shape') return;
  const fs = [], series = [];
  for (let i = 0; i <= 300; i++) fs.push(0.1 * Math.pow(180, i / 300));
  const cols = { paint: '#7cf0c5', thick: '#b59cff', foam: '#ffb547', salisbury: '#4fd1ff' };
  for (const [k, c] of Object.entries(COATINGS)) {
    if (k === 'none') continue;
    series.push({ x: fs, y: fs.map((f) => gammaDb(c.g(f * 1e9))), color: cols[k], width: k === state.p.coat ? 3 : 1.2, label: k === state.p.coat ? c.name : undefined, dash: k === state.p.coat ? [] : [4, 3] });
  }
  plot($('#ramChart'), {
    series, xLog: true, xMin: 0.1, xMax: 18, yMin: -40, yMax: 1, xLabel: 'תדר (גיגה-הרץ)', yLabel: 'החזרה (dB)',
    marks: Object.values(M.BANDS).map((b) => ({ x: b.f / 1e9, label: b.name, color: b.name === state.band ? '#ffd34d' : '#5a6d92' })),
  });
  const c = COATINGS[state.p.coat], g = c.g(M.BANDS[state.band].f);
  $('#ramNote').innerHTML = state.p.coat === 'none'
    ? 'בלי ציפוי, המתכת מחזירה הכל (0 dB). בחרו ציפוי בפאנל כדי לראות מה הוא עושה.'
    : `<b>${c.name}</b>: בתדר ${state.band} מחזיר ${gammaDb(g).toFixed(1)} dB מהשדה, כלומר σ קטן פי ${(1 / (g * g)).toFixed(0)}. ${c.note || ''}. משקל: כ-${c.kgm2} ק"ג למ"ר, ובסך הכל ${(c.kgm2 * (state.aero ? state.aero.Swet : 0)).toFixed(0)} ק"ג על המטוס. שימו לב שב-VHF אף ציפוי לא עובד: השכבה דקה מאוד ביחס לגל של שני מטרים.`;
}

function drawAero() {
  if (!state.model) return;
  const a = aero(state.model.ac, state.model.P); state.aero = a;
  if (state.tab !== 'aero') return;
  const ref = state.aeroRef || (state.aeroRef = (() => { const m = M.build(PRESETS.conventional); return aero(m.ac, m.P); })());
  const cmp = (v, r, higherGood) => { const d = (v / r - 1) * 100; const good = higherGood ? d >= 0 : d <= 0; return `<i class="${Math.abs(d) < 3 ? '' : good ? 'good' : 'bad'}">${d >= 0 ? '+' : ''}${d.toFixed(0)}% מול הקונבנציונלי</i>`; };
  $('#aeroKpis').innerHTML = [
    ['שטח כנף', a.S.toFixed(1) + ' מ"ר', ''],
    ['מנת ממדים AR', a.AR.toFixed(2), cmp(a.AR, ref.AR, true)],
    ['שיפוע העילוי CLα', a.CLa.toFixed(2) + ' /rad', cmp(a.CLa, ref.CLa, true)],
    ['גרר אפס עילוי CD₀', a.CD0.toFixed(4), cmp(a.CD0, ref.CD0, false)],
    ['עילוי/גרר מרבי', a.LDmax.toFixed(1), cmp(a.LDmax, ref.LDmax, true)],
    ['גרר גל (מאך 1.2) CDw', a.CdWave.toFixed(3), cmp(a.CdWave, ref.CdWave, false)],
    ['פי כמה מסירס-האק', a.waveRatio.toFixed(2), ''],
    ['נפח פנימי', a.V.toFixed(0) + ' מ"ק', cmp(a.V, ref.V, true)],
  ].map(([t, v, c]) => `<div class="kpi"><span>${t}</span><b>${v}</b>${c}</div>`).join('');
  const ac = state.model.ac, L = ac.p.L, xs = ac.area.xs.map((x) => L / 2 - x);
  const Smax = 16 * a.V / (3 * Math.PI * L);
  const sh = xs.map((x) => { const xi = 2 * x / L - 1; return Smax * Math.pow(Math.max(0, 1 - xi * xi), 1.5); });
  plot($('#areaChart'), { series: [{ x: xs, y: ac.area.A, color: '#7cf0c5', width: 2.5, label: 'המטוס שלך', fill: 'rgba(124,240,197,.08)' }, { x: xs, y: sh, color: '#ffb547', dash: [5, 4], label: 'סירס-האק באותו נפח' }], xMin: 0, xMax: L, yMin: 0, xLabel: 'מרחק מהאף (מ\')', yLabel: 'שטח חתך (מ"ר)' });
  // יציבות כיוונית: גרף מוטות
  const cv = $('#cnChart'), { ctx, W, H } = fitCanvas(cv);
  ctx.clearRect(0, 0, W, H);
  const items = [['זנבות', a.CnTail, '#7cf0c5'], ['גוף', a.CnFus, '#ff5d73'], ['סה"כ', a.Cnb, a.Cnb > 0.04 ? '#5ef0a0' : a.Cnb > 0 ? '#ffb547' : '#ff5d73']];
  const sc = (H * 0.38) / 0.16, y0 = H * 0.5;
  ctx.strokeStyle = '#5a6d92'; ctx.beginPath(); ctx.moveTo(30, y0); ctx.lineTo(W - 10, y0); ctx.stroke();
  items.forEach(([t, v, c], i) => {
    const x = 50 + i * (W - 80) / 3, bw = (W - 80) / 3 * 0.55, h = Math.max(-H * 0.45, Math.min(H * 0.45, v * sc));
    ctx.fillStyle = c; ctx.fillRect(x, y0 - Math.max(0, h), bw, Math.abs(h));
    ctx.fillStyle = '#dfe8f7'; ctx.font = '12px Rubik'; ctx.textAlign = 'center'; ctx.direction = 'rtl';
    ctx.fillText(t, x + bw / 2, h > 0 ? y0 + 16 : y0 - 8); ctx.direction = 'ltr'; ctx.font = '11px JetBrains Mono';
    ctx.fillText(v.toFixed(3), x + bw / 2, h > 0 ? y0 - h - 6 : y0 - h + 14);
  });
  ctx.direction = 'rtl'; ctx.textAlign = 'right'; ctx.fillStyle = '#8a9bb9'; ctx.font = '12px Rubik';
  ctx.fillText('Cnβ לרדיאן: חיובי = יציב', W - 10, 16); ctx.direction = 'ltr';
  $('#cnNote').innerHTML = a.Cnb > 0.04 ? 'יציב כיוונית בלי עזרה. הזנבות עושים את העבודה.'
    : a.Cnb > 0 ? 'יציב בקושי. הטיית הזנבות קנתה חמקנות על חשבון הזנב: כוח הצד של זנב מוטה קטן פי cos²(הטיה). מחשב טיסה (fly-by-wire) יעזור.'
      : '<b class="bad">לא יציב כיוונית.</b> בלי זנב, האף רוצה לברוח הצידה. ה-B-2 פותר את זה עם הגאים מפוצלים בקצות הכנפיים שנפתחים כבלמי אוויר בצד אחד, ומחשב טיסה שמתקן עשרות פעמים בשנייה.';
}

function irParams() {
  const ab = $('#irAB').checked, M_ = +$('#irM').value, h = +$('#irH').value, p = state.p, slot = p.nozzleType === 'slot';
  const A = state.aero || { S: 60 };
  return {
    M: M_, hKm: h,
    Afront: Math.PI * p.fus.W * (p.fus.HU + p.fus.HL) / 2 + p.wing.span * p.wing.cr * p.wing.t * 0.5,
    Aside: p.L * (p.fus.HU + p.fus.HL) * 0.75 + (p.tail.type !== 'none' ? p.tail.h * p.tail.cr * Math.cos((p.tail.cant || 0) * Math.PI / 180) : 0),
    Atop: A.S,
    hotArea: 1.1, hotVis: slot ? 0.08 : 1, Thot: ab ? 1250 : slot ? 650 : 820,
    Tplume: ab ? 1900 : slot ? 520 : 760, plumeArea: ab ? 14 : slot ? 3 : 6,
  };
}
function drawIR() {
  $('#irMv').textContent = (+$('#irM').value).toFixed(2); $('#irHv').textContent = $('#irH').value + ' ק"מ';
  if (state.tab !== 'ir') return;
  const o = irParams(), band = $('#irLW').checked ? IR.LWIR : IR.MWIR;
  const ref = { ...o, hotVis: 1, Thot: $('#irAB').checked ? 1250 : 820, Tplume: $('#irAB').checked ? 1900 : 760, plumeArea: $('#irAB').checked ? 14 : 6 };
  const asp = [], rng = [], rngRef = [];
  for (let a = 0; a <= 180; a += 2) { asp.push(a); rng.push(IR.irstRange(IR.irIntensity(o, a, band).total)); rngRef.push(IR.irstRange(IR.irIntensity(ref, a, band).total)); }
  plot($('#irChart'), { series: [{ x: asp, y: rng, color: '#ff8a5c', width: 2.5, label: 'המטוס שלך' }, { x: asp, y: rngRef, color: '#8a9bb9', dash: [5, 4], label: 'נחיר עגול חשוף' }], xMin: 0, xMax: 180, yMin: 0, xLabel: 'זווית מבט: 0 = מלפנים, 180 = מאחור', yLabel: 'טווח (ק"מ)' });
  const Ts = IR.skinT(o.M, o.hKm), Ta = IR.isaT(o.hKm);
  const lam = [], ser = [];
  for (let i = 0; i <= 200; i++) lam.push(1 + i * 0.07);
  const temps = [[o.Thot, '#ff5d73', 'חלקי מנוע ' + o.Thot.toFixed(0) + 'K'], [o.Tplume, '#ffb547', 'סילון ' + o.Tplume.toFixed(0) + 'K'], [Ts, '#7cf0c5', 'עור ' + Ts.toFixed(0) + 'K'], [Ta, '#4fd1ff', 'אוויר ' + Ta.toFixed(0) + 'K']];
  for (const [T, c, l] of temps) ser.push({ x: lam, y: lam.map((x) => Math.log10(IR.planck(x * 1e-6, T) * 1e-6 + 1e-6)), color: c, label: l, width: 2 });
  plot($('#planckChart'), { series: ser, xMin: 1, xMax: 15, yMin: -3, yMax: 5, xLabel: 'אורך גל (מיקרון)', yLabel: 'log₁₀ קרינות (W/m²·sr·µm)', marks: [{ x: 3, label: '3', color: '#5a6d92' }, { x: 5, label: '5', color: '#5a6d92' }, { x: 8, label: '8', color: '#5a6d92' }, { x: 12, label: '12', color: '#5a6d92' }], fmtY: (v) => '10^' + v });
  const f = IR.irIntensity(o, 0, band), r = IR.irIntensity(o, 180, band), s = IR.irIntensity(o, 90, band);
  $('#irKpis').innerHTML = [
    ['טמפרטורת עור', (Ts - 273.15).toFixed(0) + '°C', `אוויר בחוץ: ${(Ta - 273.15).toFixed(0)}°C`],
    ['טווח מלפנים', IR.irstRange(f.total).toFixed(0) + ' ק"מ', `${f.total.toFixed(0)} W/sr`],
    ['טווח מהצד', IR.irstRange(s.total).toFixed(0) + ' ק"מ', `${s.total.toFixed(0)} W/sr`],
    ['טווח מאחור', IR.irstRange(r.total).toFixed(0) + ' ק"מ', `${r.total.toFixed(0)} W/sr`],
  ].map(([t, v, c]) => `<div class="kpi"><span>${t}</span><b>${v}</b><i>${c}</i></div>`).join('');
}
['#irM', '#irH', '#irAB', '#irLW'].forEach((s) => $(s).addEventListener('input', drawIR));

// ---- אתגרים נוספים ----
function drawMore() {
  if (state.tab !== 'more') return;
  const a = state.aero || {};
  const pol = state.polar;
  const below = state.below;
  const coatKg = COATINGS[state.p.coat].kgm2 * (a.Swet || 0);
  $('#moreKpis').innerHTML = [
    ['σ מלפנים (תדר ' + state.band + ')', pol ? PO.dB(PO.sectorMean(pol.raw, 0, 30)).toFixed(1) + ' dBsm' : '…', state.opts.bayOpen || state.opts.external ? 'עם החימוש' : 'נקי'],
    ['σ מלמטה (מכ"ם מתחת)', below != null ? PO.dB(below).toFixed(1) + ' dBsm' : '…', 'גובה מכ"ם -60°'],
    ['משקל הציפוי', coatKg.toFixed(0) + ' ק"ג', 'כמו ' + Math.round(coatKg / 80) + ' נוסעים'],
    ['נפח פנימי לדלק וחימוש', (a.V || 0).toFixed(0) + ' מ"ק', 'אין מקום בחוץ: הכל צריך להיכנס'],
  ].map(([t, v, c]) => `<div class="kpi"><span>${t}</span><b>${v}</b><i>${c}</i></div>`).join('');
  if (below == null) ask('polar', state.p, { band: state.band, el: -60, n: 360, win: 10, ...state.opts }, (d) => { state.below = PO.sectorMean(d.raw, 0, 180); drawMore(); }, { tag: 'below' });
  lpi();
}
for (const [id, k] of [['#xBay', 'bayOpen'], ['#xExt', 'external']]) $(id).onchange = (e) => { state.opts[k] = e.target.checked; state.below = null; invalidateFront(); requestPolar(); drawMore(); };

// גילוי פסיבי: המטוס החמקן שמדליק מכ"ם צועק "אני כאן"
const LPI = { gp: 0, esm: -65, pm: false };
function lpi() {
  const el = $('#chall');
  const lam = 0.03, P = 1e4, G = 10 ** 3.3, sig = 5;
  const Sr = P * G * G * lam * lam * 1 / ((4 * Math.PI) ** 3 * (85e3) ** 4); // מכויל ל-R₁=85 ק"מ
  const Rd = Math.pow(P * G * G * lam * lam * sig / ((4 * Math.PI) ** 3 * Sr), 0.25) / 1e3;
  const Pused = LPI.pm ? P * Math.pow(60 / Rd, 4) : P; // ניהול הספק: רק מה שצריך כדי לראות מטרה ב-60 ק"מ
  const Sesm = 1e-3 * 10 ** (LPI.esm / 10) * 10 ** (LPI.gp / 10);
  const Ri = Math.sqrt(Pused * G * 1 * lam * lam / ((4 * Math.PI) ** 2 * Sesm)) / 1e3;
  const cards = [
    ['📡 להדליק מכ"ם = לצעוק', `<p>המכ"ם שלך רואה מטוס של 5 מ"ר מ-<b>${Rd.toFixed(0)} ק"מ</b> (הגל עושה הלוך וחזור, ההספק דועך כ-1/R⁴). מקלט התרעה של האויב שומע אותך מ-<b class="${Ri > Rd ? 'bad' : 'good'}">${Ri.toFixed(0)} ק"מ</b> (רק הלוך, 1/R²).</p>
      <div class="ctl"><label>רגישות מקלט ההתרעה <b>${LPI.esm} dBm</b></label><input type="range" min="-90" max="-40" step="1" value="${LPI.esm}" data-l="esm"></div>
      <div class="ctl"><label>פיזור ספקטרלי (LPI) <b>${LPI.gp} dB</b></label><input type="range" min="0" max="40" step="1" value="${LPI.gp}" data-l="gp"></div>
      <label class="chk" style="padding:2px 0"><input type="checkbox" data-l="pm" ${LPI.pm ? 'checked' : ''}> ניהול הספק: רק כמה שצריך ל-60 ק"מ</label>
      <p>מכ"ם LPI מפזר את האות על פני רוחב סרט גדול: המכ"ם עצמו יודע את הקוד ואוסף הכל בחזרה, והמאזין לא.</p>`],
    ['🔧 תחזוקה', '<p>כל בורג, פאנל ופתח שירות הם אי-רציפות שמפזרת. ה-F-117 וה-B-2 דרשו שעות של מילוי מרק וציפוי אחרי כל פתיחה, ו-B-2 צריך האנגר ממוזג. ב-F-35 הכניסו את הבולע לחומר המרוכב עצמו כדי לקצר את זה.</p>'],
    ['🔗 תפרים ופערים', '<p>פער של מילימטרים בין פאנלים הוא "חריץ" שמקרין. לכן שולי הדלתות והפאנלים חתוכים בשיני מסור (זיגזג) בזוויות של הקצוות הראשיים: כך גם הם שולחים את הקוצים לאותם כיוונים.</p>'],
    ['💣 חימוש פנימי', '<p>טיל אחד מתחת לכנף מוסיף בערך מטר רבוע: יותר מכל המטוס. לכן הכל בתא פנימי, שמגביל את מספר הטילים, ובזמן הירי הדלתות נפתחות לשניה: סמנו את התיבה למעלה ותראו מה קורה ל-σ מלמטה.</p>'],
    ['☁️ שובל, רעש ועין', '<p>מכ"ם הוא לא החיישן היחיד. שובל עיבוי בגובה רב נראה מעשרות קילומטרים, ולכן ל-B-2 יש מכ"ם (LIDAR) שמזהה שובל ומורה לטייס לשנות גובה. הצבע אפור-כהה כדי להיעלם בלילה ובגובה.</p>'],
    ['💰 מחיר', '<p>החמקנות מחייבת דיוק ייצור של עשיריות מילימטר על מטוס של 20 מטר, חומרים מיוחדים ובדיקות במתקני מדידה. B-2 עלה כ-2 מיליארד דולר ליחידה (כולל פיתוח), ולכן נבנו רק 21.</p>'],
  ];
  el.innerHTML = cards.map(([t, b]) => `<div class="card"><h4>${t}</h4>${b}</div>`).join('');
  el.querySelectorAll('[data-l]').forEach((inp) => {
    inp.oninput = inp.onchange = () => { const k = inp.dataset.l; LPI[k] = inp.type === 'checkbox' ? inp.checked : +inp.value; lpi(); };
  });
}

// ---------- מבחן חדירה ----------
const mission = initMission({ state, ask, fmtSig });

window.addEventListener('resize', () => { drawPolar(); drawShape(); drawRam(); drawAero(); drawIR(); });
state.tab = 'shape';
buildControls();
radarMoved();
legend();
changed();
window.__state = state;
