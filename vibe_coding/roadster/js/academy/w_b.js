// ווידג'טים: נחיר דה-לבאל, אימפולס סגולי, COPV, ריקון מכל, קיקפליפ, בלימת נפילה.
import { LinePlot, SERIES, setupCanvas } from './plot.js';
import { slider, checkbox, buttons, readout, f, animate } from './ui.js';
import { Nozzle, AirTank, AIR, areaRatio, PSI, BAR, tankMass } from '../gas.js';

const G = 9.80665, M = 2237;
const GAM = AIR.gamma;

// מאך מתוך יחס שטחים, בענף התת-קולי או העל-קולי
function machFromArea(ar, supersonic) {
  if (ar <= 1.0001) return 1;
  let lo = supersonic ? 1 : 1e-4, hi = supersonic ? 10 : 1;
  for (let i = 0; i < 50; i++) {
    const mid = (lo + hi) / 2, r = areaRatio(mid);
    if (supersonic ? r < ar : r > ar) lo = mid; else hi = mid;
  }
  return (lo + hi) / 2;
}

// ================= נחיר =================
export function nozzle(root) {
  const cv = root.querySelector('canvas.nz'), pc = root.querySelector('canvas.nplot'), ctrls = root.querySelector('.ctrls');
  const st = { p0: 379, eps: 4 };
  const At = Nozzle.sizeFor(20000, 5500 * PSI, 293.15, 4);
  const plot = new LinePlot(pc, { xLabel: 'מיקום לאורך הנחיר (0 = כניסה, 1 = יציאה)', yLabel: 'יחס / מספר מאך', xMax: 1, yMax: 4, xDec: 2, yDec: 2 });
  const ro = readout(ctrls, [['F', 'דחף (נחיר ראשי)'], ['mdot', 'ספיקת מסה'], ['isp', 'אימפולס סגולי'], ['Me', 'מאך ביציאה'], ['pe', 'לחץ ביציאה'], ['Te', 'טמפ\' ביציאה'], ['reg', 'מצב הסילון']]);
  const xt = 0.38;
  const radius = (x, eps) => {
    const rt = 1, rin = 2.4, re = Math.sqrt(eps);
    if (x <= xt) { const u = (xt - x) / xt; return rt + (rin - rt) * u * u * (3 - 2 * u) * 1.0; }
    const u = (x - xt) / (1 - xt); return rt + (re - rt) * (1 - (1 - u) * (1 - u)) ;
  };
  let prof = [];
  const upd = () => {
    const nz = new Nozzle({ throatArea: At, eps: st.eps });
    const p = nz.perf(st.p0 * BAR, 293.15);
    const M_ = [], P_ = [], T_ = [];
    prof = [];
    for (let x = 0; x <= 1.0001; x += 0.01) {
      const r = radius(x, st.eps), ar = r * r;
      const m = machFromArea(ar, x > xt);
      const k = 1 + (GAM - 1) / 2 * m * m;
      M_.push([x, m]); P_.push([x, Math.pow(k, -GAM / (GAM - 1))]); T_.push([x, 1 / k]);
      prof.push({ x, r, m, t: 1 / k });
    }
    plot.vlines = [{ x: xt, label: 'גרון: מאך 1' }];
    plot.set([{ name: 'מספר מאך', data: M_ }, { name: 'לחץ / לחץ במכל', data: P_ }, { name: 'טמפ\' / טמפ\' במכל', data: T_ }], { yMax: Math.max(3.2, Math.ceil(M_[M_.length - 1][1] + 0.4)) });
    const regHe = { under: 'תת-מורחב: יהלומי מאך', over: 'מורחב-יתר', ideal: 'מותאם', separated: 'ניתוק זרימה בנחיר', subsonic: 'תת-קולי (לא חנוק)', off: 'כבוי' }[p.regime];
    ro.set({ F: `${(p.F / 1000).toFixed(1)} kN`, mdot: `${p.mdot.toFixed(1)} ק"ג/ש'`, isp: `${(p.F / Math.max(p.mdot, 1e-9) / G).toFixed(0)} ש'`, Me: nz.Me.toFixed(2), pe: `${(p.pe / BAR).toFixed(2)} בר`, Te: `${(p.Te - 273.15).toFixed(0)}°C`, reg: regHe });
    st.pe = p.pe; st.regime = p.regime; st.Me = nz.Me;
  };
  slider(ctrls, { label: 'לחץ במכל', min: 5, max: 400, step: 1, value: 379, fmt: (v) => `${v} בר`, onInput: (v) => { st.p0 = v; upd(); } });
  slider(ctrls, { label: 'יחס התרחבות A_e/A*', min: 1.5, max: 30, step: 0.5, value: 4, fmt: (v) => `${v}`, onInput: (v) => { st.eps = v; upd(); } });
  ctrls.appendChild(ctrls.querySelector('.readout'));
  upd();
  const parts = [];
  animate(root, (dt) => {
    const { c, w, h } = setupCanvas(cv);
    c.clearRect(0, 0, w, h);
    const cy = h / 2, nx0 = 30, nL = w * 0.42, rs = Math.min(18, (h * 0.42) / Math.max(2.4, Math.sqrt(st.eps)));
    const X = (x) => nx0 + x * nL;
    // מכל
    c.fillStyle = '#2a2d31'; c.fillRect(0, cy - 2.4 * rs - 10, nx0, 2 * (2.4 * rs + 10));
    // דפנות הנחיר
    c.fillStyle = '#5a5f66';
    c.beginPath(); c.moveTo(X(0), cy - 2.4 * rs - 8);
    for (const q of prof) c.lineTo(X(q.x), cy - q.r * rs);
    c.lineTo(X(1), cy - prof[prof.length - 1].r * rs - 8); c.lineTo(X(0), cy - 2.4 * rs - 14); c.fill();
    c.beginPath(); c.moveTo(X(0), cy + 2.4 * rs + 8);
    for (const q of prof) c.lineTo(X(q.x), cy + q.r * rs);
    c.lineTo(X(1), cy + prof[prof.length - 1].r * rs + 8); c.lineTo(X(0), cy + 2.4 * rs + 14); c.fill();
    // סילון: חבית עם יהלומים
    const re = prof[prof.length - 1].r * rs;
    const pr = Math.max(0.05, st.pe / AIR.pAmb);
    const cell = Math.max(10, 2 * re * 1.2 * Math.sqrt(pr));
    const bw = re * Math.min(1.7, Math.max(0.6, 0.55 * Math.sqrt(pr) + 0.3));
    const len = w - X(1) - 10;
    c.save(); c.beginPath(); c.rect(X(1), 0, len, h); c.clip();
    const grad = c.createLinearGradient(X(1), 0, X(1) + len, 0); grad.addColorStop(0, 'rgba(160,210,255,0.35)'); grad.addColorStop(1, 'rgba(160,210,255,0)');
    c.fillStyle = grad;
    c.beginPath(); c.moveTo(X(1), cy - re);
    for (let x = 0; x <= len; x += 4) { const ph = (x / cell) * Math.PI; c.lineTo(X(1) + x, cy - (bw * 0.75 + bw * 0.25 * Math.abs(Math.sin(ph))) * (1 + x / len * 0.6)); }
    for (let x = len; x >= 0; x -= 4) { const ph = (x / cell) * Math.PI; c.lineTo(X(1) + x, cy + (bw * 0.75 + bw * 0.25 * Math.abs(Math.sin(ph))) * (1 + x / len * 0.6)); }
    c.fill();
    if (st.regime === 'under' || st.regime === 'over') {
      for (let k = 0; k < 6; k++) {
        const x0 = X(1) + k * cell, a = Math.max(0, 1 - k * 0.17);
        c.strokeStyle = `rgba(255,255,255,${0.55 * a})`; c.lineWidth = 1.5;
        c.beginPath(); c.moveTo(x0, cy - bw * 0.8); c.lineTo(x0 + cell * 0.5, cy); c.lineTo(x0, cy + bw * 0.8); c.stroke();
        c.beginPath(); c.moveTo(x0 + cell * 0.5, cy); c.lineTo(x0 + cell, cy - bw * 0.8); c.moveTo(x0 + cell * 0.5, cy); c.lineTo(x0 + cell, cy + bw * 0.8); c.stroke();
        c.fillStyle = `rgba(255,255,255,${0.4 * a})`; c.beginPath(); c.ellipse(x0 + cell * 0.5, cy, 4, bw * 0.35, 0, 0, Math.PI * 2); c.fill();
      }
    }
    c.restore();
    // חלקיקים
    const spd = (q) => q.m * Math.sqrt(q.t);
    if (parts.length < 500) for (let k = 0; k < 3; k++) parts.push({ x: 0, f: (Math.random() * 2 - 1) * 0.9 });
    c.fillStyle = 'rgba(200,230,255,0.9)';
    for (let i = parts.length - 1; i >= 0; i--) {
      const p = parts[i];
      if (p.x <= 1) {
        const q = prof[Math.min(prof.length - 1, Math.max(0, Math.round(p.x * 100)))];
        p.x += (0.25 + 0.6 * Math.pow(spd(q), 0.7)) * dt; // הטווח הדינמי דחוס כדי שיראו גם את הכניסה האיטית
        c.fillRect(X(p.x) - 1, cy + p.f * q.r * rs - 1, 2.5, 2.5);
      } else {
        p.ex = (p.ex || 0) + spd(prof[prof.length - 1]) * 130 * dt;
        if (p.ex > len) { parts.splice(i, 1); continue; }
        c.fillRect(X(1) + p.ex - 1, cy + p.f * (re + p.ex * 0.12 * Math.sqrt(pr)) - 1, 2.5, 2.5);
      }
    }
    // תוויות
    c.fillStyle = '#c3c2b7'; c.font = '12px Rubik'; c.textAlign = 'center';
    c.fillText(`מכל: ${st.p0} בר, 20°C`, nx0 / 2 + 20, 16);
    c.fillText('גרון', X(xt), cy + 2.4 * rs + 26);
    c.fillText(`יציאה: מאך ${st.Me.toFixed(1)}`, X(1), cy + re + 26);
    c.textAlign = 'right'; c.fillText(st.regime === 'under' ? 'יהלומי מאך: הסילון ממשיך להתרחב ונדחס שוב ושוב' : st.regime === 'over' ? 'מורחב-יתר: הלחץ ביציאה נמוך מהאוויר שבחוץ' : '', w - 8, 16);
  });
}

// ================= אימפולס סגולי =================
export function ispBars(el) {
  const rows = [
    ['אוויר דחוס, הרודסטר (מודל)', 66, true], ['חנקן קר (פלקון 9, MMU)', 73], ['הליום קר', 165], ['הידרזין (חד-רכיבי)', 230],
    ['מנוע דלק מוצק', 265], ['Merlin 1D, גובה פני הים', 282], ['Raptor, ואקום', 380], ['מימן וחמצן (RS-25, ואקום)', 452],
  ];
  const max = 460;
  el.innerHTML = `<div class="note" style="margin-bottom:6px"><span style="display:inline-block;width:10px;height:10px;background:${SERIES[1]};border-radius:2px"></span> הרודסטר · <span style="display:inline-block;width:10px;height:10px;background:${SERIES[0]};border-radius:2px"></span> מנועים אחרים · אימפולס סגולי בשניות</div>` +
    rows.map(([n, v, me]) => `<div class="row ${me ? 'me' : ''}" title="${n}: ${v} ש'"><span class="name">${n}</span><div class="track"><div class="fill" style="width:${v / max * 100}%;background:${me ? SERIES[1] : SERIES[0]}"></div></div><span class="val">${v} s</span></div>`).join('');
}

// ================= COPV =================
export function copv(el) {
  // קפסולה עם שכבת חישוק, שכבה סלילית ומעטפת פנימית
  let s = `<svg viewBox="0 0 900 300" role="img" aria-label="מבנה מכל COPV">`;
  s += `<defs><clipPath id="cap"><rect x="150" y="70" width="600" height="160" rx="80"/></clipPath>
    <pattern id="hoop" width="8" height="8" patternUnits="userSpaceOnUse"><rect width="8" height="8" fill="#2a2d31"/><line x1="0" y1="4" x2="8" y2="4" stroke="#4a4f57" stroke-width="2"/></pattern>
    <pattern id="helix" width="14" height="14" patternUnits="userSpaceOnUse" patternTransform="rotate(32)"><rect width="14" height="14" fill="#1f2226"/><line x1="0" y1="7" x2="14" y2="7" stroke="#7a818b" stroke-width="2.5"/></pattern></defs>`;
  s += `<rect x="150" y="70" width="600" height="160" rx="80" fill="url(#helix)"/>`;
  s += `<g clip-path="url(#cap)"><rect x="330" y="70" width="280" height="160" fill="url(#hoop)"/></g>`;
  // חתך פתוח: מעטפת פנימית
  s += `<path d="M 150 150 A 80 80 0 0 1 230 70 L 330 70 L 330 230 L 230 230 A 80 80 0 0 1 150 150 Z" fill="#1a1a19"/>`;
  s += `<path d="M 162 150 A 68 68 0 0 1 230 82 L 330 82 M 162 150 A 68 68 0 0 0 230 218 L 330 218" fill="none" stroke="#9aa0a6" stroke-width="5"/>`;
  s += `<rect x="168" y="88" width="162" height="124" fill="rgba(57,135,229,0.18)"/>`;
  for (let i = 0; i < 40; i++) s += `<circle cx="${180 + Math.random() * 140}" cy="${100 + Math.random() * 100}" r="2" fill="#8ec5ff" opacity="0.8"/>`;
  s += `<rect x="740" y="135" width="40" height="30" rx="4" fill="#8a8f96"/><rect x="780" y="142" width="40" height="16" fill="#8a8f96"/>`;
  // תוויות
  const lab = (x1, y1, x2, y2, t) => `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="#c3c2b7" stroke-width="1"/><circle cx="${x1}" cy="${y1}" r="3" fill="#c3c2b7"/><text x="${x2}" y="${y2 + (y2 < y1 ? -6 : 16)}" fill="#f0efec" font-size="17" font-family="Rubik" text-anchor="middle">${t}</text>`;
  s += lab(250, 120, 210, 30, 'אוויר ב-379 בר');
  s += lab(240, 218, 240, 280, 'מעטפת פנימית אטומה (liner)');
  s += lab(470, 90, 470, 30, 'שכבת חישוק: סיבי פחמן סביב');
  s += lab(680, 215, 600, 280, 'שכבה סלילית: מחזיקה את הכיפות');
  s += lab(800, 150, 820, 60, 'שסתום ווסת');
  s += `</svg>`;
  s += `<figcaption><b>מבנה של מכל COPV.</b> הלחץ מנסה "לנפח" את המכל לשני כיוונים: היקפי (המתח בכיוון הזה כפול) ואורכי. שכבת החישוק לוקחת את ההיקפי, והסלילית את האורכי ואת הכיפות. מכל כזה במשקל של כ-50 ק"ג יכול להחזיק 300 ליטר ב-379 בר, ומכל פלדה מקביל היה שוקל פי 3–4.</figcaption>`;
  el.innerHTML = s;
}

// ================= ריקון המכל =================
export function tank(root) {
  const pc = root.querySelector('canvas.tplot'), ctrls = root.querySelector('.ctrls');
  // שני גרפים (אחוזים, טמפרטורה) זה מעל זה בעמודה אחת
  const col = document.createElement('div');
  pc.replaceWith(col); col.appendChild(pc);
  const tc = document.createElement('canvas'); tc.style.height = '170px'; tc.style.marginTop = '10px';
  col.appendChild(tc);
  const pPct = new LinePlot(pc, { xLabel: 'זמן (ש\')', yLabel: 'אחוז מההתחלה', yMax: 105, xDec: 2, yDec: 0, unitY: '%' });
  const pT = new LinePlot(tc, { xLabel: 'זמן (ש\')', yLabel: '°C', yDec: 0, unitY: '°C' });
  const st = { mode: 'main', heat: true };
  const ro = readout(ctrls, [['half', 'הדחף נחצה אחרי'], ['dur', 'דחף מעל 10%'], ['imp', 'אימפולס כולל'], ['tmin', 'טמפ\' מינימלית'], ['air', 'אוויר שיצא']]);
  const run = () => {
    const tank = new AirTank({ volume: 0.6, pFill: 5500 * PSI, hA: st.heat ? 1200 : 0 });
    const m0 = tank.m, p0 = tank.p;
    const main = new Nozzle({ throatArea: Nozzle.sizeFor(20000, 5500 * PSI, 293.15) });
    const corner = new Nozzle({ throatArea: Nozzle.sizeFor(22000, 5500 * PSI, 293.15) });
    const nz = st.mode === 'main' ? main : corner, n = st.mode === 'yaw' ? 1 : 4;
    const yawN = new Nozzle({ throatArea: Nozzle.sizeFor(4000, 5500 * PSI, 293.15) });
    const P = [], F = [], Mm = [], T = [];
    let F0 = null, half = null, dur = null, imp = 0, tmin = 20, t = 0, hoverEnd = null;
    const dt = 0.002;
    while (t < 14 && tank.m > 0.5) {
      let u = 1, perf;
      if (st.mode === 'hover') {
        const full = corner.perf(tank.p, tank.T);
        u = Math.min(1, 22000 / (4 * full.F));
        perf = corner.perf(u * tank.p, tank.T);
        if (u >= 1 && 4 * full.F < 21900 && hoverEnd === null) hoverEnd = t;
      } else perf = (st.mode === 'yaw' ? yawN : nz).perf(tank.p, tank.T);
      const Ft = n * perf.F;
      if (F0 === null) F0 = Ft;
      if (half === null && Ft < F0 / 2) half = t;
      if (Ft < 0.1 * F0 && dur === null) dur = t;
      imp += Ft * dt;
      if (Math.round(t / dt) % 10 === 0) { P.push([t, tank.p / p0 * 100]); F.push([t, Ft / F0 * 100]); Mm.push([t, tank.m / m0 * 100]); T.push([t, tank.T - 273.15]); }
      tmin = Math.min(tmin, tank.T - 273.15);
      tank.step(n * perf.mdot, dt);
      t += dt;
      if (dur !== null && t > dur + 1.5) break;
    }
    pPct.vlines = hoverEnd !== null ? [{ x: hoverEnd, label: `סוף הריחוף: ${hoverEnd.toFixed(1)} ש'` }] : [];
    pPct.set([{ name: 'לחץ', data: P }, { name: 'דחף', data: F }, { name: 'מסת אוויר', data: Mm }], { xMax: Math.max(1, t) });
    pT.set([{ name: 'טמפרטורת הגז במכל', data: T, color: SERIES[0] }], { xMax: Math.max(1, t), yMin: Math.min(-160, Math.floor(tmin / 20) * 20), yMax: 30 });
    ro.set({ half: half !== null ? `${half.toFixed(2)} ש'` : '—', dur: dur !== null ? `${dur.toFixed(2)} ש'` : `> ${t.toFixed(1)} ש'`, imp: `${f(imp / 1000)} kN·s`, tmin: `${tmin.toFixed(0)}°C`, air: `${f(m0 - tank.m)} ק"ג` });
  };
  buttons(ctrls, [['main', '4 אחוריים (80 kN)'], ['corner', '4 פינות (88 kN)'], ['hover', 'ריחוף (22 kN מווסת)'], ['yaw', 'נחיר סבסוב אחד']], { selected: 'main', onPick: (k) => { st.mode = k; run(); } });
  checkbox(ctrls, { label: 'חום מדפנות המכל', value: true, onInput: (v) => { st.heat = v; run(); } });
  ctrls.appendChild(ctrls.querySelector('.readout'));
  run();
  addEventListener('resize', () => { pPct.draw(); pT.draw(); });
}

// ================= קיקפליפ =================
export function flip(root) {
  const cv = root.querySelector('canvas.flip'), pc = root.querySelector('canvas.fplot'), ctrls = root.querySelector('.ctrls');
  const plot = new LinePlot(pc, { xLabel: 'זמן (ש\')', yLabel: 'זווית גלגול (°)', xMax: 3, yMin: -20, yMax: 400, xDec: 2, yDec: 0, unitY: '°' });
  const I = 746, arm = 0.84, FM = 44000; // שני נחירים בכל צד
  let sim = null, rate = 7.5;
  const reset = () => { sim = { t: 0, y: 0, vy: 0, x: 0, vx: 0, phi: 0, w: 0, phase: 'launch', FL: 0, FR: 0, hist: [], air: 0, done: false }; };
  const step = (dt) => {
    const s = sim; if (s.done) return;
    s.t += dt;
    const onGround = s.y <= 0 && s.t > 0.05;
    let lift = 0, wDes = 0, level = false;
    if (s.phase === 'launch') { lift = 1; level = true; if (s.vy > 6) s.phase = 'spin'; }
    else if (s.phase === 'spin') { wDes = rate; if (s.phi > Math.PI * 2 - 0.9) s.phase = 'land'; }
    else { level = true; }
    let err = ((s.phi + Math.PI) % (2 * Math.PI)) - Math.PI; if (err < -Math.PI) err += 2 * Math.PI;
    if (level) wDes = -4.5 * Math.max(-1.3, Math.min(1.3, err));
    // בלימת נחיתה
    if (s.phase === 'land' && s.vy < -2.5 && Math.cos(s.phi) > 0.55) {
      const aUp = 2 * FM / M - G, vt = -Math.sqrt(2 * 0.5 * aUp * Math.max(s.y, 0)) - 1;
      lift = Math.max(lift, Math.min(1, (vt - s.vy) * 0.9));
    }
    const tau = I * 11 * (wDes - s.w);
    let FL = lift * FM + tau / (2 * arm), FR = lift * FM - tau / (2 * arm);
    const mn = Math.min(FL, FR); if (mn < 0) { FL -= mn; FR -= mn; }
    const sc = Math.min(1, FM / Math.max(FL, FR, 1)); FL *= sc; FR *= sc;
    s.FL = FL / FM; s.FR = FR / FM;
    const T = FL + FR;
    s.air += T / 650 * dt;
    s.w += ((FL - FR) * arm) / I * dt;
    s.phi += s.w * dt;
    s.vy += (T * Math.cos(s.phi) / M - G) * dt; s.vx += (-T * Math.sin(s.phi) / M) * dt;
    s.y += s.vy * dt; s.x += s.vx * dt;
    if (s.y < 0) { s.y = 0; if (s.t > 0.3) { s.vy = 0; s.vx *= 0.9; s.w *= 0.5; if (s.phase === 'land') s.done = true; } else s.vy = Math.max(0, s.vy); }
    if (Math.round(s.t / dt) % 3 === 0) s.hist.push([s.t, s.phi * 180 / Math.PI, s.y * 20]);
  };
  const play = document.createElement('button'); play.className = 'primary'; play.textContent = '▶ הפעל קיקפליפ'; ctrls.appendChild(play);
  play.onclick = () => reset();
  const sw = document.createElement('div'); sw.style.minWidth = '220px'; ctrls.appendChild(sw);
  slider(sw, { label: 'קצב סיבוב מבוקש', min: 3, max: 10, step: 0.5, value: 7.5, fmt: (v) => `${v} רד'/ש'`, onInput: (v) => { rate = v; } });
  const ro = document.createElement('div'); ro.className = 'readout'; ro.style.flex = '1'; ctrls.appendChild(ro);
  reset();
  animate(root, (dt) => {
    for (let k = 0; k < 8; k++) step(dt / 8);
    const s = sim;
    const { c, w, h } = setupCanvas(cv);
    c.clearRect(0, 0, w, h);
    const sc = Math.min(w / 9, h / 11);
    const gy = h - 26;
    c.fillStyle = '#2b2a27'; c.fillRect(0, gy, w, h);
    // רכבים חונים ברקע (הסמי בגובה 3.9 מ')
    c.fillStyle = '#3a3a37'; c.fillRect(w / 2 - 1.3 * sc, gy - 3.9 * sc, 2.6 * sc, 3.9 * sc);
    c.fillStyle = '#8b8a83'; c.font = '11px Rubik'; c.textAlign = 'center'; c.fillText('סמי: 3.9 מ\'', w / 2, gy - 3.9 * sc - 6);
    const px = w / 2 + s.x * sc, py = gy - (s.y + 0.6) * sc;
    c.save(); c.translate(px, py); c.rotate(-s.phi);
    // להבות (למטה בגוף)
    for (const [side, cmd] of [[-1, s.FL], [1, s.FR]]) {
      if (cmd > 0.02) {
        const L = cmd * 2.6 * sc;
        const gr = c.createLinearGradient(0, 0.5 * sc, 0, 0.5 * sc + L); gr.addColorStop(0, 'rgba(200,230,255,0.95)'); gr.addColorStop(1, 'rgba(200,230,255,0)');
        c.fillStyle = gr; c.beginPath(); c.moveTo(side * 0.84 * sc - 5, 0.5 * sc); c.lineTo(side * 0.84 * sc + 5, 0.5 * sc); c.lineTo(side * 0.84 * sc + 12, 0.5 * sc + L); c.lineTo(side * 0.84 * sc - 12, 0.5 * sc + L); c.fill();
      }
    }
    c.fillStyle = '#f2f2ef'; c.beginPath(); c.roundRect(-1.0 * sc, -0.35 * sc, 2.0 * sc, 0.8 * sc, 0.25 * sc); c.fill();
    c.fillStyle = '#1d2024'; c.beginPath(); c.ellipse(0, -0.35 * sc, 0.62 * sc, 0.38 * sc, 0, Math.PI, 0); c.fill();
    c.fillStyle = '#121212'; c.fillRect(-1.0 * sc, 0.25 * sc, 0.3 * sc, 0.35 * sc); c.fillRect(0.7 * sc, 0.25 * sc, 0.3 * sc, 0.35 * sc);
    c.fillStyle = '#e53935'; c.fillRect(-0.85 * sc, 0.0, 1.7 * sc, 3);
    c.restore();
    // פקודות
    c.font = '12px Rubik'; c.textAlign = 'left'; c.fillStyle = '#c3c2b7';
    c.fillText(`שמאל ${(s.FL * 100).toFixed(0)}%`, 10, 20); c.fillText(`ימין ${(s.FR * 100).toFixed(0)}%`, 10, 36);
    c.fillStyle = '#3a3a37'; c.fillRect(80, 12, 100, 8); c.fillRect(80, 28, 100, 8);
    c.fillStyle = SERIES[0]; c.fillRect(80, 12, 100 * s.FL, 8); c.fillRect(80, 28, 100 * s.FR, 8);
    c.textAlign = 'right'; c.fillStyle = '#f0efec'; c.font = 'bold 13px Rubik';
    c.fillText({ launch: 'המראה: כל הנחירים', spin: 'סיבוב: צד אחד דוחף', land: s.done ? 'נחת' : 'יישור ונחיתה' }[s.phase], w - 10, 20);
    plot.set([{ name: 'זווית גלגול', data: s.hist.map(p => [p[0], p[1]]) }]);
    ro.innerHTML = `<div class="kv"><span>גובה מרבי</span><b>${f(Math.max(0, ...s.hist.map(p => p[2] / 20)), 1)} מ'</b></div><div class="kv"><span>אוויר</span><b>~${f(s.air)} ק"ג</b></div>`;
  });
}

// ================= בלימת נפילה =================
export function land(root) {
  const pc = root.querySelector('canvas.lplot'), ctrls = root.querySelector('.ctrls');
  const plot = new LinePlot(pc, { xLabel: 'גובה (מ\')', yLabel: 'מהירות נפילה (מ\'/ש\')', xDec: 1, yDec: 1, unitX: ' מ\'', unitY: ' מ\'/ש\'' });
  const st = { h0: 21, fill: 1, margin: 0.4 };
  const ro = readout(ctrls, [['vt', 'מהירות פגיעה'], ['vf', 'בלי בלימה'], ['hs', 'הבלימה מתחילה ב'], ['g', 'שיא g (מורגש)'], ['air', 'אוויר']]);
  const run = () => {
    const tank = new AirTank({ volume: 0.6, pFill: 5500 * PSI * st.fill });
    const m0 = tank.m;
    const nz = new Nozzle({ throatArea: Nozzle.sizeFor(22000, 5500 * PSI, 293.15) });
    let y = st.h0, vy = 0, t = 0, hs = null, gmax = 0, u = 0;
    const dt = 0.002, free = [], br = [];
    for (let hh = st.h0; hh >= 0; hh -= st.h0 / 100) free.push([hh, Math.sqrt(2 * G * (st.h0 - hh))]);
    while (y > 0 && t < 10) {
      const full = 4 * nz.perf(tank.p, tank.T).F;
      const aUp = full / M - G;
      const vt = -Math.sqrt(2 * st.margin * Math.max(aUp, 0.1) * Math.max(y - 0.3, 0)) - 1;
      const want = Math.max(0, Math.min(1, (vt - vy) * 0.9));
      u += (want - u) * Math.min(1, dt / 0.03);
      const perf = nz.perf(u * tank.p, tank.T);
      const F = u > 0.002 ? 4 * perf.F : 0;
      if (F > 1000 && hs === null) hs = y;
      gmax = Math.max(gmax, F / (M * G));
      tank.step(u > 0.002 ? 4 * perf.mdot : 0, dt);
      vy += (F / M - G) * dt; y += vy * dt; t += dt;
      if (Math.round(t / dt) % 5 === 0) br.push([Math.max(0, y), -vy]);
    }
    br.push([0, -vy]);
    plot.markers = hs !== null ? [{ x: hs, y: Math.sqrt(2 * G * (st.h0 - hs)), label: 'הבלימה מתחילה', color: SERIES[1] }] : [];
    plot.set([{ name: 'נפילה חופשית', data: free, color: '#8b8a83', dash: [5, 4] }, { name: 'עם בלימה', data: br, color: SERIES[1] }], { xMax: st.h0, yMax: Math.ceil(Math.sqrt(2 * G * st.h0) + 2) });
    const vtd = -vy;
    ro.set({ vt: `<span style="color:${vtd < 3.5 ? '#7fd9b0' : vtd < 7 ? '#f2c46d' : '#ff9b97'}">${vtd.toFixed(1)} מ'/ש'</span>`, vf: `${Math.sqrt(2 * G * st.h0).toFixed(1)} מ'/ש'`, hs: hs !== null ? `${hs.toFixed(1)} מ'` : '—', g: `${gmax.toFixed(1)}g`, air: `${f(m0 - tank.m)} מתוך ${f(m0)} ק"ג` });
  };
  slider(ctrls, { label: 'גובה הנפילה', min: 4, max: 35, step: 1, value: 21, fmt: (v) => `${v} מ'`, onInput: (v) => { st.h0 = v; run(); } });
  slider(ctrls, { label: 'מילוי המכל', min: 0.15, max: 1, step: 0.05, value: 1, fmt: (v) => `${Math.round(v * 100)}%`, onInput: (v) => { st.fill = v; run(); } });
  slider(ctrls, { label: 'מרווח ביטחון בבקר', min: 0.2, max: 0.9, step: 0.05, value: 0.4, fmt: (v) => v < 0.35 ? `${v} (שמרני)` : v > 0.65 ? `${v} (נועז)` : `${v}`, onInput: (v) => { st.margin = v; run(); } });
  ctrls.appendChild(ctrls.querySelector('.readout'));
  run();
  addEventListener('resize', () => plot.draw());
}
