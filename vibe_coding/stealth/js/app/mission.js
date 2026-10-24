// מבחן חדירה: המטוס טס דרך מערך הגנה אווירית, וכל מכ"ם בודק בכל רגע אם הוא רואה אותו
import { fitCanvas } from './plot.js';
import * as M from '../core/model.js';
import { PRESETS } from '../core/geometry.js';

const WX = 420, WY = 236; // ק"מ
const ELS = [0, -2, -6, -15, -35];
const BANDS = ['VHF', 'S', 'X'];
const TYPES = Object.keys(M.RADARS);
const RCOL = { ew_vhf: '#b59cff', search_s: '#4fd1ff', sam_x: '#ff5d73', fighter: '#ffb547' };
const LOCK_S = 20; // שניות מעקב רצוף עד נעילה

export function initMission({ state, ask }) {
  const $ = (s) => document.querySelector(s);
  const cv = $('#map'), logEl = $('#mLog');
  const ms = {
    radars: [
      { type: 'ew_vhf', x: 205, y: 38 }, { type: 'search_s', x: 180, y: 196 }, { type: 'sam_x', x: 268, y: 96 },
      { type: 'sam_x', x: 305, y: 170 }, { type: 'fighter', x: 345, y: 52, alt: 9 },
    ],
    start: { x: 14, y: 118 }, target: { x: 398, y: 118 }, wps: [],
    alt: 9, table: null, refTable: null, running: false, s: 0, trail: [], key: '', log: [],
  };
  const route = () => [ms.start, ...ms.wps, ms.target];
  const routeLen = () => { const r = route(); let L = 0; for (let i = 1; i < r.length; i++) L += Math.hypot(r[i].x - r[i - 1].x, r[i].y - r[i - 1].y); return L; };
  function posAt(s) {
    const r = route(); let acc = 0;
    for (let i = 1; i < r.length; i++) {
      const d = Math.hypot(r[i].x - r[i - 1].x, r[i].y - r[i - 1].y);
      if (acc + d >= s || i === r.length - 1) {
        const u = Math.min(1, (s - acc) / d);
        return { x: r[i - 1].x + u * (r[i].x - r[i - 1].x), y: r[i - 1].y + u * (r[i].y - r[i - 1].y), hd: Math.atan2(r[i].y - r[i - 1].y, r[i].x - r[i - 1].x) };
      }
      acc += d;
    }
    return { ...ms.target, hd: 0 };
  }

  function need() {
    const key = JSON.stringify(state.p);
    if (key !== ms.key) {
      ms.key = key; ms.table = null;
      ask('table', state.p, { bands: BANDS, els: ELS }, (d) => { ms.table = d.res; draw(); }, { tag: 'mission', progress: () => draw() });
    }
    if ($('#mRef').checked && !ms.refTable) ask('table', PRESETS.conventional, { bands: BANDS, els: ELS }, (d) => { ms.refTable = d.res; draw(); }, { tag: 'missionRef', keep: true });
  }
  // σ מהטבלה: אזימוט במעלות (0 = אף, חיובי = שמאל), גובה במעלות
  function sig(tab, band, az, el) {
    if (!tab) return null;
    const arr = tab[band], n = arr[0].length;
    const i = Math.round((((az % 360) + 360) % 360) / 360 * n) % n;
    const e = Math.max(ELS[ELS.length - 1], Math.min(0, -Math.abs(el)));
    let k = 0; while (k + 1 < ELS.length - 1 && ELS[k + 1] > e) k++;
    const u = (ELS[k] - e) / (ELS[k] - ELS[k + 1]);
    const a = Math.log(arr[k][i] + 1e-12), b = Math.log(arr[k + 1][i] + 1e-12);
    return Math.exp(a + (b - a) * Math.max(0, Math.min(1, u)));
  }
  function detect(tab, pos, alt) {
    const res = [];
    for (const r of ms.radars) {
      const R = M.RADARS[r.type];
      const dx = r.x - pos.x, dy = r.y - pos.y, d = Math.hypot(dx, dy);
      const hr = r.type === 'fighter' ? (r.alt || 9) * 1000 : R.h;
      const hor = M.horizonKm(hr, alt * 1000);
      // ציר y במפה כלפי מטה; אזימוט חיובי = שמאל המטוס
      const brg = Math.atan2(dy, dx);
      const az = -(brg - pos.hd) * 180 / Math.PI;
      const el = Math.atan2(hr / 1000 - alt, Math.max(0.1, d)) * 180 / Math.PI;
      const s = sig(tab, R.band, az, el);
      if (s == null) { res.push({ r, seen: false }); continue; }
      const Rdet = M.detectRange(R.R1, s);
      res.push({ r, seen: d <= Rdet && d <= hor, d, Rdet, hor });
    }
    return res;
  }

  let last = 0, stats = null;
  function reset() {
    ms.running = false; ms.s = 0; ms.trail = []; ms.log = [];
    stats = { me: { lock: new Map(), locked: new Set(), seen: new Set() }, ref: { lock: new Map(), locked: new Set(), seen: new Set() } };
    $('#mPlay').textContent = '▶ טוס'; logEl.innerHTML = ''; draw();
  }
  function log(t, cls = '') { ms.log.push(`<div class="${cls}">${t}</div>`); logEl.innerHTML = ms.log.join(''); logEl.scrollTop = 1e9; }
  function step(dtReal) {
    const v = 0.28; // ק"מ לשנייה, בערך מאך 0.9
    const dt = dtReal * 50; // זמן מואץ
    ms.s = Math.min(routeLen(), ms.s + v * dt);
    const pos = posAt(ms.s), t = ms.s / v;
    const runOne = (who, tab, label) => {
      const st = stats[who]; let worst = 0;
      for (const d of detect(tab, pos, ms.alt)) {
        const id = ms.radars.indexOf(d.r), R = M.RADARS[d.r.type];
        if (d.seen) {
          worst = Math.max(worst, R.guide ? 1 : 0.5);
          if (!st.seen.has(id)) { st.seen.add(id); if (who === 'me') log(`${fmtT(t)} · ${R.name} גילה את ${label} ממרחק ${d.d.toFixed(0)} ק"מ`, 'warn'); }
          if (R.guide) {
            const l = (st.lock.get(id) || 0) + dt; st.lock.set(id, l);
            if (l >= LOCK_S && !st.locked.has(id)) { st.locked.add(id); log(`${fmtT(t)} · ${R.name} נעל על ${label}. טיל בדרך.`, 'bad'); if (who === 'me') worst = 2; }
          }
        } else st.lock.set(id, 0);
        if (st.locked.has(id)) worst = Math.max(worst, who === 'me' ? 2 : worst);
      }
      return worst;
    };
    const w = runOne('me', ms.table, 'המטוס שלך');
    if ($('#mRef').checked && ms.refTable) runOne('ref', ms.refTable, 'הקונבנציונלי');
    ms.trail.push({ x: pos.x, y: pos.y, w });
    if (ms.s >= routeLen() - 1e-6) {
      ms.running = false; $('#mPlay').textContent = '▶ טוס';
      const me = stats.me, rf = stats.ref;
      log(`<b>הגעת למטרה.</b> ${me.locked.size ? `ננעלת ${me.locked.size} פעמים: היית מיורט.` : me.seen.size ? `התגלית ע"י ${me.seen.size} מכ"מים, אבל אף אחד לא הצליח לנעול.` : 'אף מכ"ם לא ראה אותך.'}` +
        ($('#mRef').checked && ms.refTable ? ` המטוס הקונבנציונלי: התגלה ע"י ${rf.seen.size}, ננעל ${rf.locked.size} פעמים.` : ''), me.locked.size ? 'bad' : 'good');
    }
  }
  const fmtT = (t) => `${Math.floor(t / 60)}:${String(Math.floor(t % 60)).padStart(2, '0')}`;

  function loop(ts) {
    if (!ms.running) return;
    const dt = Math.min(0.05, (ts - (last || ts)) / 1000); last = ts;
    step(dt); draw();
    requestAnimationFrame(loop);
  }

  function draw() {
    if (state.tab !== 'mission') return;
    const { ctx, W, H } = fitCanvas(cv);
    const sc = Math.min(W / WX, H / WY), X = (x) => x * sc, Y = (y) => y * sc;
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = '#08111f'; ctx.fillRect(0, 0, W, H);
    ctx.strokeStyle = '#111d33'; ctx.lineWidth = 1;
    for (let x = 0; x <= WX; x += 50) { ctx.beginPath(); ctx.moveTo(X(x), 0); ctx.lineTo(X(x), Y(WY)); ctx.stroke(); }
    for (let y = 0; y <= WY; y += 50) { ctx.beginPath(); ctx.moveTo(0, Y(y)); ctx.lineTo(X(WX), Y(y)); ctx.stroke(); }
    ctx.fillStyle = '#5a6d92'; ctx.font = '11px JetBrains Mono'; ctx.direction = 'ltr'; ctx.fillText('50 km', X(5), Y(WY) - 6);
    ctx.strokeStyle = '#5a6d92'; ctx.beginPath(); ctx.moveTo(X(5), Y(WY) - 16); ctx.lineTo(X(55), Y(WY) - 16); ctx.stroke();
    const pos = posAt(ms.s);
    // טבעות גילוי
    for (const r of ms.radars) {
      const R = M.RADARS[r.type], col = RCOL[r.type];
      const hr = r.type === 'fighter' ? (r.alt || 9) * 1000 : R.h;
      const hor = M.horizonKm(hr, ms.alt * 1000);
      const front = (tab) => { if (!tab) return null; const el = Math.atan2(hr / 1000 - ms.alt, 60) * 180 / Math.PI; return M.detectRange(R.R1, sig(tab, R.band, 0, el)); };
      const f = front(ms.table);
      if (f != null) {
        ctx.fillStyle = col + '14'; ctx.strokeStyle = col; ctx.lineWidth = 1.5;
        ctx.beginPath(); ctx.arc(X(r.x), Y(r.y), X(Math.min(f, hor)), 0, 7); ctx.fill(); ctx.stroke();
      }
      if ($('#mRef').checked) { const g = front(ms.refTable); if (g != null) { ctx.strokeStyle = '#8a9bb9'; ctx.setLineDash([5, 5]); ctx.beginPath(); ctx.arc(X(r.x), Y(r.y), X(Math.min(g, hor)), 0, 7); ctx.stroke(); ctx.setLineDash([]); } }
      if (hor < 500) { ctx.strokeStyle = '#2b4f8a'; ctx.setLineDash([2, 4]); ctx.beginPath(); ctx.arc(X(r.x), Y(r.y), X(hor), 0, 7); ctx.stroke(); ctx.setLineDash([]); }
    }
    // מסלול
    const rt = route();
    ctx.strokeStyle = '#3b4d72'; ctx.setLineDash([6, 5]); ctx.lineWidth = 1.5; ctx.beginPath();
    rt.forEach((p, i) => (i ? ctx.lineTo(X(p.x), Y(p.y)) : ctx.moveTo(X(p.x), Y(p.y)))); ctx.stroke(); ctx.setLineDash([]);
    for (const w of ms.wps) { ctx.fillStyle = '#8a9bb9'; ctx.beginPath(); ctx.arc(X(w.x), Y(w.y), 4, 0, 7); ctx.fill(); }
    ctx.lineWidth = 3.5;
    for (let i = 1; i < ms.trail.length; i++) {
      const a = ms.trail[i - 1], b = ms.trail[i];
      ctx.strokeStyle = b.w >= 2 ? '#ff5d73' : b.w >= 1 ? '#ff8a5c' : b.w > 0 ? '#ffd34d' : '#5ef0a0';
      ctx.beginPath(); ctx.moveTo(X(a.x), Y(a.y)); ctx.lineTo(X(b.x), Y(b.y)); ctx.stroke();
    }
    // מכ"מים
    ctx.font = '12px Rubik'; ctx.textAlign = 'center';
    const seenNow = ms.table ? detect(ms.table, pos, ms.alt) : [];
    for (const r of ms.radars) {
      const col = RCOL[r.type], d = seenNow.find((q) => q.r === r);
      if (d && d.seen && (ms.running || ms.s > 0)) {
        ctx.strokeStyle = M.RADARS[r.type].guide ? '#ff5d73' : '#ffd34d'; ctx.lineWidth = 1.2; ctx.setLineDash([3, 3]);
        ctx.beginPath(); ctx.moveTo(X(r.x), Y(r.y)); ctx.lineTo(X(pos.x), Y(pos.y)); ctx.stroke(); ctx.setLineDash([]);
      }
      ctx.fillStyle = col; ctx.beginPath();
      if (r.type === 'fighter') { ctx.moveTo(X(r.x), Y(r.y) - 8); ctx.lineTo(X(r.x) + 7, Y(r.y) + 6); ctx.lineTo(X(r.x) - 7, Y(r.y) + 6); }
      else { ctx.rect(X(r.x) - 5, Y(r.y) - 5, 10, 10); }
      ctx.fill();
      ctx.direction = 'rtl'; ctx.fillStyle = '#dfe8f7'; ctx.fillText(M.RADARS[r.type].name.replace(' (טילי קרקע-אוויר)', ''), X(r.x), Y(r.y) - 12); ctx.direction = 'ltr';
    }
    // מטרה והתחלה
    ctx.strokeStyle = '#5ef0a0'; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(X(ms.target.x), Y(ms.target.y), 9, 0, 7); ctx.stroke();
    ctx.beginPath(); ctx.arc(X(ms.target.x), Y(ms.target.y), 3, 0, 7); ctx.stroke();
    ctx.direction = 'rtl'; ctx.fillStyle = '#5ef0a0'; ctx.fillText('מטרה', X(ms.target.x), Y(ms.target.y) + 24); ctx.direction = 'ltr';
    // המטוס
    ctx.save(); ctx.translate(X(pos.x), Y(pos.y)); ctx.rotate(pos.hd);
    ctx.fillStyle = '#ffffff'; ctx.beginPath(); ctx.moveTo(11, 0); ctx.lineTo(-7, 8); ctx.lineTo(-3, 0); ctx.lineTo(-7, -8); ctx.closePath(); ctx.fill();
    ctx.restore();
    if (!ms.table) { ctx.fillStyle = '#ffb547'; ctx.direction = 'rtl'; ctx.textAlign = 'center'; ctx.font = '14px Rubik'; ctx.fillText('מחשב את תבניות ה-RCS לשלושה תדרים ולחמש זוויות גובה…', W / 2, H / 2); ctx.direction = 'ltr'; }
  }

  // עכבר: גרירת מכ"ם, הוספת נקודת ציון, החלפת סוג
  let drag = null;
  const toW = (e) => { const r = cv.getBoundingClientRect(), sc = Math.min(r.width / WX, r.height / WY); return { x: (e.clientX - r.left) / sc, y: (e.clientY - r.top) / sc, sc }; };
  const hitRadar = (p) => ms.radars.find((r) => Math.hypot(r.x - p.x, r.y - p.y) * p.sc < 12);
  cv.addEventListener('pointerdown', (e) => {
    if (e.button !== 0) return;
    const p = toW(e), r = hitRadar(p);
    if (r) { drag = r; cv.setPointerCapture(e.pointerId); return; }
    const wi = ms.wps.findIndex((w) => Math.hypot(w.x - p.x, w.y - p.y) * p.sc < 9);
    if (wi >= 0) { ms.wps.splice(wi, 1); reset(); return; }
    // הכנסה לפי הסדר לאורך ציר x
    const i = ms.wps.findIndex((w) => w.x > p.x);
    ms.wps.splice(i < 0 ? ms.wps.length : i, 0, { x: p.x, y: p.y }); reset();
  });
  cv.addEventListener('pointermove', (e) => { if (!drag) return; const p = toW(e); drag.x = Math.max(0, Math.min(WX, p.x)); drag.y = Math.max(0, Math.min(WY, p.y)); draw(); });
  cv.addEventListener('pointerup', () => { if (drag) { drag = null; reset(); } });
  cv.addEventListener('contextmenu', (e) => {
    e.preventDefault(); const r = hitRadar(toW(e)); if (!r) return;
    r.type = TYPES[(TYPES.indexOf(r.type) + 1) % TYPES.length]; reset();
  });
  $('#mPlay').onclick = () => {
    if (!ms.table) return;
    if (ms.s >= routeLen() - 1e-6) reset();
    ms.running = !ms.running; $('#mPlay').textContent = ms.running ? '⏸ עצור' : '▶ המשך';
    if (ms.running) { last = 0; requestAnimationFrame(loop); }
  };
  $('#mReset').onclick = () => { ms.wps = []; reset(); };
  $('#mAlt').oninput = (e) => { ms.alt = +e.target.value; $('#mAltv').textContent = ms.alt < 1 ? (ms.alt * 1000).toFixed(0) + ' מ\'' : ms.alt.toFixed(1) + ' ק"מ'; reset(); };
  $('#mRef').onchange = () => { need(); reset(); };
  reset();
  return {
    show() { need(); draw(); },
    designChanged() { if (state.tab === 'mission') { need(); reset(); } else ms.key = ''; },
    draw,
  };
}
