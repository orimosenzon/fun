// ווידג'טים: אנטומיה (תלת-ממד), מעגל החיכוך, מערכת היניקה, הקיר.
import { LinePlot, SERIES, setupCanvas } from './plot.js';
import { slider, checkbox, readout, f, animate } from './ui.js';
import { muAt, TIRE } from '../tire.js';
import { solveFan, fanCurve, leakFlow, FAN_DEFAULTS } from '../fan.js';
import { wallProfile } from '../track.js';
import { AIR } from '../gas.js';
import { VARIANTS } from '../params.js';

const G = 9.80665;
const M = 2237; // מסת הרוקט עם מכל מלא ונהג

// ================= אנטומיה =================
export async function anatomy(root) {
  const THREE = await import('three');
  const { OrbitControls } = await import('three/addons/controls/OrbitControls.js');
  const { buildCarModel } = await import('../render/carmodel.js');
  const cv = root.querySelector('canvas.three');
  const renderer = new THREE.WebGLRenderer({ canvas: cv, antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  const scene = new THREE.Scene();
  const cam = new THREE.PerspectiveCamera(36, 2, 0.1, 100);
  cam.position.set(-3.9, 2.3, 4.0);
  const ctl = new OrbitControls(cam, cv); ctl.enableDamping = true; ctl.autoRotate = true; ctl.autoRotateSpeed = 0.8;
  ctl.target.set(0, 0.1, 0); ctl.minDistance = 3; ctl.maxDistance = 14;
  scene.add(new THREE.HemisphereLight(0xdde8ff, 0x332a20, 1.4));
  const dl = new THREE.DirectionalLight(0xffffff, 2.2); dl.position.set(3, 6, 4); scene.add(dl);
  const P = VARIANTS.rocket;
  const model = buildCarModel(P);
  model.setXray(true);
  scene.add(model.root);
  // רצפה עם רשת
  const grid = new THREE.GridHelper(10, 20, 0x444444, 0x2a2a2a); grid.position.y = -P.cgHeight; scene.add(grid);
  // מצב לכל רכב: גלגלים במקום
  const fakeCar = { pos: { x: 0, y: 0, z: 0 }, q: { x: 0, y: 0, z: 0, w: 1 }, P, wheels: [0, 1, 2, 3].map(() => ({ mount: { y: P.suspension.rest - P.suspension.staticComp + P.wheelRadius - P.cgHeight }, comp: P.suspension.staticComp, steer: 0, spin: 0 })), aeroMode: 'high', fan: { n: 0.6 }, thr: P.thrusters.map(() => ({ u: 0 })) };
  const INFO = {
    copv: { name: 'מכלי COPV', col: '#9aa0a6', text: 'שני מכלי לחץ מסיבי פחמן, יחד 600 ליטר אוויר ב-379 בר (כ-227 ק"ג). אחד לרוחב מאחורי המושב, ואחד במקום המושב השני (לכן לרוקט יש מושב אחד). זה ה"דלק" של כל הנחירים. פרק 6.' },
    pipe: { name: 'צנרת לחץ גבוה', col: '#9fd8ff', text: 'קווים מהמכלים אל עשרת הנחירים. בכל נחיר שסתום חשמלי מהיר שנפתח ונסגר באלפיות שנייה, כך שהמחשב יכול לווסת את הדחף בכל רגע. פרק 7.' },
    fan: { name: 'מאווררי יניקה', col: '#3ec2a0', text: 'ארבעה מאווררים של כ-20 קילוואט מאחור, שיונקים אוויר מהחלל האטום מתחת לרצפה (המלבן הירוק השקוף) ופולטים אותו דרך הדיפיוזר. הפרש של 11% מהאטמוספרה על 3.8 מ"ר נותן 4.5 טון. פרק 3.' },
    battery: { name: 'סוללה', col: '#2f6fd6', text: 'תאי 4680 במגש שטוח ונמוך ברצפה, בין הסרנים. מיקום נמוך = מרכז כובד נמוך = פחות נטייה בפניות. קיבולת לא פורסמה; טווח מוצהר של יותר מ-300 מייל.' },
    motor: { name: 'מנועים חשמליים', col: '#c77d2a', text: 'שלושה מנועים (הערכה): אחד קדמי ושניים אחוריים, יחד יותר ממגה-ואט. בלי תיבת הילוכים: מומנט מלא מאפס. פרק 10.' },
  };
  const list = root.querySelector('.comp-list'), info = root.querySelector('.comp-info');
  const keys = Object.keys(INFO);
  const btns = keys.map(k => {
    const b = document.createElement('button');
    b.innerHTML = `<span class="sw" style="background:${INFO[k].col}"></span>${INFO[k].name}`;
    b.onclick = () => pick(k); list.appendChild(b); return b;
  });
  const nzBtn = document.createElement('button'); nzBtn.innerHTML = `<span class="sw" style="background:#ff8a65"></span>נחירים`; list.appendChild(nzBtn);
  nzBtn.onclick = () => pick('nozzles');
  INFO.nozzles = { name: 'עשרה נחירים', text: 'ארבעה אחוריים בדיפיוזר (כ-80 קילוניוטון יחד, לזינוק), ארבעה בפינות התחתונות שנושפים למטה (להמראה, גלגול ובלימת נפילה), ושניים בחרטום שנושפים לצדדים (סבסוב). הם זוהרים כשבוחרים אותם. פרק 5.' };
  function pick(k) {
    btns.forEach((b, i) => b.classList.toggle('on', keys[i] === k)); nzBtn.classList.toggle('on', k === 'nozzles');
    for (const c of model.components) c.mesh.visible = (k === 'nozzles') ? true : c.key === k || k === null;
    for (const c of model.components) if (c.mesh.material.emissive) c.mesh.material.emissiveIntensity = c.key === k ? 0.6 : 0.0;
    fakeCar.thr.forEach(t => t.u = k === 'nozzles' ? 1 : 0);
    info.innerHTML = `<b>${INFO[k].name}</b><p>${INFO[k].text}</p>`;
  }
  // הדגשה: כל רכיב מקבל חומר משלו כדי שאפשר יהיה להאיר אותו לבד
  for (const c of model.components) { c.mesh.material = c.mesh.material.clone(); if (c.mesh.material.emissive) c.mesh.material.emissive.set(c.mesh.material.color); }
  const resize = () => { const r = cv.getBoundingClientRect(); renderer.setSize(r.width, r.height, false); cam.aspect = r.width / r.height; cam.updateProjectionMatrix(); };
  addEventListener('resize', resize); resize();
  cv.addEventListener('pointerdown', () => { ctl.autoRotate = false; });
  animate(root, (dt) => { model.update(fakeCar, dt, 'orbit'); ctl.update(); renderer.render(scene, cam); });
}

// ================= מעגל החיכוך =================
export function tires(root) {
  const cv = root.querySelector('canvas.circle'), ctrls = root.querySelector('.ctrls');
  let suction = 0;
  const s = slider(ctrls, { label: 'יניקה (תוספת למשקל)', min: 0, max: 2, step: 0.05, value: 0, fmt: (v) => `${v.toFixed(2)}g`, onInput: (v) => { suction = v; } });
  const ro = readout(ctrls, [['mu', 'μ בעומס הזה'], ['g', 'אחיזה מרבית'], ['t60', '0–96 קמ"ש (מינימום)'], ['skid', 'מהירות במעגל R=30 מ\'']]);
  let t = 0;
  const trail = [];
  animate(root, (dt) => {
    t += dt;
    const { c, w, h } = setupCanvas(cv);
    c.clearRect(0, 0, w, h);
    const fz = M * G * (1 + suction) / 4;
    const mu = muAt(fz);
    const gmax = mu * (1 + suction);
    const R = Math.min(w, h) * 0.42 * Math.min(1, gmax / 3.2) / (gmax / 3.2) / 3.2; // פיקסלים ל-1g, כך ש-3.2g נכנס
    const cx = w / 2, cy = h / 2, ppg = Math.min(w, h) * 0.44 / 3.2;
    // טבעות g
    c.strokeStyle = 'rgba(255,255,255,0.08)'; c.fillStyle = '#8b8a83'; c.font = '11px Rubik'; c.textAlign = 'center';
    for (let g = 1; g <= 3; g++) { c.beginPath(); c.arc(cx, cy, g * ppg, 0, Math.PI * 2); c.stroke(); c.fillText(`${g}g`, cx + g * ppg - 10, cy - 4); }
    c.beginPath(); c.moveTo(cx - 3.2 * ppg, cy); c.lineTo(cx + 3.2 * ppg, cy); c.moveTo(cx, cy - 3.2 * ppg); c.lineTo(cx, cy + 3.2 * ppg); c.stroke();
    c.fillText('האצה', cx, cy - 3.2 * ppg + 2); c.fillText('בלימה', cx, cy + 3.2 * ppg + 10);
    c.textAlign = 'left'; c.fillText('פנייה ימינה', cx + 3.2 * ppg - 64, cy + 16); c.textAlign = 'right'; c.fillText('שמאלה', cx - 3.2 * ppg + 40, cy + 16);
    // מעגל ללא יניקה (להשוואה)
    const g0 = muAt(M * G / 4);
    c.setLineDash([5, 5]); c.strokeStyle = 'rgba(255,255,255,0.35)'; c.beginPath(); c.arc(cx, cy, g0 * ppg, 0, Math.PI * 2); c.stroke(); c.setLineDash([]);
    // מעגל החיכוך
    c.fillStyle = 'rgba(25,158,112,0.13)'; c.strokeStyle = SERIES[2]; c.lineWidth = 2;
    c.beginPath(); c.arc(cx, cy, gmax * ppg, 0, Math.PI * 2); c.fill(); c.stroke(); c.lineWidth = 1;
    // "נהג" שמבקש כוחות: ליסז'ו שחורג לפעמים
    const want = 2.2 * Math.max(gmax, 1.3) / 1.5;
    const ax = want * 0.75 * Math.sin(t * 0.9), ay = -want * 0.62 * Math.sin(t * 1.7 + 0.6);
    const mag = Math.hypot(ax, ay), slip = mag > gmax;
    const sx = slip ? ax * gmax / mag : ax, sy = slip ? ay * gmax / mag : ay;
    trail.push([sx, sy]); if (trail.length > 90) trail.shift();
    c.strokeStyle = 'rgba(57,135,229,0.45)'; c.lineWidth = 2; c.beginPath();
    trail.forEach(([x, y], i) => i ? c.lineTo(cx + x * ppg, cy + y * ppg) : c.moveTo(cx + x * ppg, cy + y * ppg)); c.stroke();
    if (slip) { c.strokeStyle = 'rgba(230,103,103,0.7)'; c.setLineDash([3, 3]); c.beginPath(); c.moveTo(cx + sx * ppg, cy + sy * ppg); c.lineTo(cx + ax * ppg, cy + ay * ppg); c.stroke(); c.setLineDash([]); }
    c.fillStyle = slip ? '#e66767' : SERIES[0]; c.beginPath(); c.arc(cx + sx * ppg, cy + sy * ppg, 7, 0, Math.PI * 2); c.fill();
    c.fillStyle = '#f0efec'; c.font = '13px Rubik'; c.textAlign = 'right';
    c.fillText(slip ? 'החלקה! הנהג ביקש יותר ממה שיש' : 'בתוך המעגל: הצמיג מחזיק', w - 10, 20);
    c.fillStyle = '#c3c2b7'; c.font = '11px Rubik'; c.fillText('מקווקו: בלי יניקה', w - 10, h - 10);
    const t60 = 26.82 / (gmax * G);
    ro.set({ mu: mu.toFixed(2), g: `${gmax.toFixed(2)}g`, t60: `${t60.toFixed(2)} ש'`, skid: `${f(Math.sqrt(gmax * G * 30) * 3.6)} קמ"ש` });
  });
}

// ================= מערכת היניקה =================
export function fan(root) {
  const sec = root.querySelector('canvas.section'), cv = root.querySelector('canvas.curve'), ctrls = root.querySelector('.ctrls');
  const st = { gap: 3, n: 1 };
  const plot = new LinePlot(cv, { xLabel: 'ספיקה (מ"ק/ש\')', yLabel: 'הפרש לחץ (kPa)', xMax: 14, yMax: 13, xDec: 1, yDec: 1, unitX: '', unitY: ' kPa' });
  const ro = readout(ctrls, [['dp', 'הפרש לחץ'], ['pct', 'מהאטמוספרה'], ['F', 'כוח הצמדה'], ['g', 'ביחס למשקל'], ['Q', 'ספיקה'], ['P', 'הספק']]);
  const upd = () => {
    const P = FAN_DEFAULTS, leakA = P.perimeter * st.gap / 1000;
    const sol = solveFan(st.n, leakA);
    const fanS = [], leakS = [], powS = [];
    for (let q = 0; q <= 14; q += 0.1) {
      fanS.push([q, fanCurve(q, sol.n) / 1000]);
      const dp = (q / (P.cdLeak * leakA)) ** 2 * AIR.rhoAmb / 2;
      leakS.push([q, dp / 1000]);
      powS.push([q, q > 0.5 ? P.pMax * P.eta / q / 1000 : NaN]);
    }
    plot.markers = [{ x: sol.Q, y: sol.dp / 1000, label: 'נקודת העבודה', color: '#f0efec' }];
    plot.set([{ name: 'עקומת המאוורר', data: fanS }, { name: 'דליפה מתחת לחצאיות', data: leakS }, { name: 'תקרת הספק 80 kW', data: powS, dash: [5, 4], color: '#8b8a83' }]);
    const F = sol.dp * P.floorArea;
    st.sol = sol;
    ro.set({ dp: `${(sol.dp / 1000).toFixed(1)} kPa`, pct: `${(sol.dp / 1013).toFixed(1)}%`, F: `${f(F / G)} ק"ג`, g: `${(F / (M * G)).toFixed(2)}g`, Q: `${sol.Q.toFixed(1)} מ"ק/ש'`, P: `${f(sol.power / 1000)} kW${sol.powerLimited ? ' (תקרה)' : ''}` });
  };
  slider(ctrls, { label: 'מרווח חצאית', min: 1, max: 30, step: 0.5, value: 3, fmt: (v) => `${v} מ"מ`, onInput: (v) => { st.gap = v; upd(); } });
  slider(ctrls, { label: 'מהירות מאוורר', min: 0, max: 1, step: 0.01, value: 1, fmt: (v) => `${Math.round(v * 100)}%`, onInput: (v) => { st.n = v; upd(); } });
  upd();
  // אנימציית חתך
  const parts = [];
  animate(root, (dt) => {
    const { c, w, h } = setupCanvas(sec);
    c.clearRect(0, 0, w, h);
    const dp = st.sol.dp, frac = dp / 12000;
    const gy = h - 30;                  // קרקע
    const x0 = w * 0.1, x1 = w * 0.9;   // קצוות הרצפה
    const fy = gy - 34;                 // רצפה
    const gapPx = Math.min(26, 3 + st.gap * 0.8);
    // קרקע
    c.fillStyle = '#2b2a27'; c.fillRect(0, gy, w, h - gy);
    // החלל מתחת: כחול לפי הירידה בלחץ
    c.fillStyle = `rgba(57,135,229,${0.08 + 0.4 * frac})`; c.fillRect(x0, fy, x1 - x0, gy - fy);
    // מרכב
    c.fillStyle = '#d9d8d2';
    c.beginPath(); c.moveTo(x0 - 10, fy); c.lineTo(x1 + 20, fy); c.quadraticCurveTo(x1 + 40, fy - 30, x1 - 10, fy - 50);
    c.lineTo(w * 0.62, fy - 64); c.quadraticCurveTo(w * 0.45, fy - 110, w * 0.3, fy - 70); c.lineTo(x0 + 10, fy - 50); c.quadraticCurveTo(x0 - 30, fy - 30, x0 - 10, fy); c.fill();
    // מאוורר מאחור (צד ימין = אחורה? בעברית: הרכב נוסע שמאלה)
    const fx = x1 - 30;
    c.fillStyle = '#199e70'; c.beginPath(); c.arc(fx, fy - 18, 14, 0, Math.PI * 2); c.fill();
    c.strokeStyle = '#121211'; c.lineWidth = 2; const ang = performance.now() / 1000 * st.sol.n * 30;
    for (let k = 0; k < 4; k++) { c.beginPath(); c.moveTo(fx, fy - 18); c.lineTo(fx + 12 * Math.cos(ang + k * 1.57), fy - 18 + 12 * Math.sin(ang + k * 1.57)); c.stroke(); }
    // חצאיות
    c.fillStyle = '#3a3a37'; c.fillRect(x0 - 4, fy, 4, gy - fy - gapPx); c.fillRect(x1, fy, 4, gy - fy - gapPx);
    // חיצי לחץ אטמוספרי מעל הרכב
    c.strokeStyle = 'rgba(240,239,236,0.5)'; c.fillStyle = 'rgba(240,239,236,0.5)'; c.lineWidth = 1.5;
    for (let x = x0 + 30; x < x1; x += 46) { arrow(c, x, 8, x, 30); }
    c.font = '12px Rubik'; c.textAlign = 'center'; c.fillStyle = '#c3c2b7'; c.fillText('לחץ אטמוספרי: 101 kPa', w / 2, 46 + 0);
    c.fillStyle = '#8ec5ff'; c.fillText(`מתחת לרצפה: ${(101.3 - dp / 1000).toFixed(1)} kPa`, w / 2, gy - 10);
    // חלקיקים: נכנסים מתחת לחצאיות, נשאבים למאוורר
    const leakV = Math.sqrt(2 * dp / AIR.rhoAmb); // מ'/ש' בחריץ
    const rate = st.sol.Q * 6 * dt;
    for (let k = 0; k < rate; k++) {
      const left = Math.random() < 0.5;
      parts.push({ x: left ? x0 - 30 : x1 + 30, y: gy - Math.random() * gapPx * 0.9 - 1, vx: (left ? 1 : -1) * (40 + leakV * 1.4), vy: 0, life: 4 });
    }
    c.fillStyle = 'rgba(160,210,255,0.85)';
    for (let i = parts.length - 1; i >= 0; i--) {
      const p = parts[i];
      const inside = p.x > x0 && p.x < x1 && p.y > fy;
      if (inside) { // נמשכים אל פתח המאוורר
        const dx = fx - p.x, dy = (fy - 4) - p.y, d = Math.hypot(dx, dy) || 1;
        p.vx += (dx / d * 220 - p.vx * 1.2) * dt; p.vy += (dy / d * 220 - p.vy * 1.2) * dt;
        if (d < 10) { p.x = fx; p.y = fy - 18; p.vx = 160; p.vy = -60; p.out = true; }
      }
      p.x += p.vx * dt; p.y += p.vy * dt; p.life -= dt;
      if (p.life < 0 || p.x > w + 10 || p.x < -10) { parts.splice(i, 1); continue; }
      c.globalAlpha = Math.min(1, p.life); c.fillRect(p.x - 1.5, p.y - 1.5, 3, 3);
    }
    c.globalAlpha = 1;
    // כוח
    const F = dp * 3.8 / G;
    c.fillStyle = '#7fd9b0'; c.font = 'bold 13px Rubik'; c.textAlign = 'right';
    c.fillText(`כוח הצמדה: ${f(F)} ק"ג`, w - 10, 20);
    c.font = '11px Rubik'; c.fillStyle = '#8b8a83'; c.fillText(`המרווח מוגדל פי ~10 בציור`, w - 10, 36);
  });
}
function arrow(c, x1, y1, x2, y2) {
  c.beginPath(); c.moveTo(x1, y1); c.lineTo(x2, y2); c.stroke();
  const a = Math.atan2(y2 - y1, x2 - x1);
  c.beginPath(); c.moveTo(x2, y2); c.lineTo(x2 - 7 * Math.cos(a - 0.4), y2 - 7 * Math.sin(a - 0.4)); c.lineTo(x2 - 7 * Math.cos(a + 0.4), y2 - 7 * Math.sin(a + 0.4)); c.closePath(); c.fill();
}

// ================= הקיר =================
export function wall(root) {
  const cv = root.querySelector('canvas.ramp'), pc = root.querySelector('canvas.nplot'), ctrls = root.querySelector('.ctrls');
  const pr = wallProfile();
  const st = { v: 11, s: 2, fall: null, t: 0, showF: true };
  const plot = new LinePlot(pc, { xLabel: 'זווית הנטייה (°)', yLabel: 'ביחידות משקל הרכב', xMax: 170, yMin: -2, yMax: 6, xDec: 0, yDec: 2, unitX: '°', unitY: '' });
  const ro = readout(ctrls, []);
  const roEl = ctrls.lastChild;
  const calc = () => {
    // לאורך המסלול: N/mg = v²k/g + cosθ + s ; חיכוך נדרש ≈ |sinθ| (מהירות קבועה) ; זמין ≈ μN
    const N = [], fric = [], need = [];
    let fall = null;
    for (let s = 0; s <= pr.total; s += 0.25) {
      const q = pr.at(s), deg = q.th * 180 / Math.PI;
      const n = st.v * st.v * q.k / G + Math.cos(q.th) + st.s;
      N.push([deg, n]); need.push([deg, Math.abs(Math.sin(q.th))]);
      const mu = muAt(Math.max(1, n) * M * G / 4);
      fric.push([deg, Math.max(0, n) * mu]);
      if (fall === null && (n <= 0 || mu * n < Math.abs(Math.sin(q.th)) * 0.98)) fall = { s, deg, why: n <= 0 ? 'normal' : 'grip' };
    }
    st.fall = fall;
    plot.markers = fall ? [{ x: fall.deg, y: 0, label: fall.why === 'normal' ? 'ניתוק' : 'החלקה', color: '#e66767', align: fall.deg > 120 ? 'right' : 'left' }] : [];
    plot.set([{ name: 'כוח נורמלי', data: N }, { name: 'חיכוך זמין (μN)', data: fric, color: SERIES[2] }, { name: 'חיכוך נדרש', data: need, color: SERIES[1] }]);
    roEl.innerHTML = fall ? `<div class="kv"><span>תוצאה</span><b style="color:#ff9b97">${fall.why === 'normal' ? 'ניתוק' : 'החלקה'} ב-${fall.deg.toFixed(0)}°</b></div><div class="note">${fall.why === 'normal' ? 'הכוח הנורמלי ירד לאפס: הכובד ניצח את היניקה והצנטריפוגה.' : 'הצמיגים לא מחזיקים: החיכוך הנדרש כדי לא להחליק אחורה גדול ממה שיש.'}</div>`
      : `<div class="kv"><span>תוצאה</span><b style="color:#7fd9b0">מגיע ל-170° ✓</b></div><div class="note">נורמלי מינימלי: ${Math.min(...N.map(p => p[1])).toFixed(2)} משקל.</div>`;
  };
  slider(ctrls, { label: 'מהירות', min: 0.5, max: 25, step: 0.5, value: 11, fmt: (v) => `${(v * 3.6).toFixed(0)} קמ"ש`, onInput: (v) => { st.v = v; st.t = 0; calc(); } });
  slider(ctrls, { label: 'יניקה', min: 0, max: 2, step: 0.1, value: 2, fmt: (v) => `${v.toFixed(1)}g`, onInput: (v) => { st.s = v; st.t = 0; calc(); } });
  ctrls.appendChild(roEl);
  checkbox(ctrls, { label: 'הצג חיצי כוח', value: true, onInput: (v) => { st.showF = v; } });
  calc();
  let pos = 0, fallState = null;
  animate(root, (dt) => {
    const { c, w, h } = setupCanvas(cv);
    c.clearRect(0, 0, w, h);
    // קנה מידה: h בין −2 ל-30 מ', v בין 0 ל-23
    const sc = Math.min((w - 40) / 38, (h - 30) / 24);
    const X = (hh) => w - 20 - (hh + 12) * sc; // הרכב נוסע שמאלה בציור (מימין לשמאל, כמו הקריאה)
    const Y = (vv) => h - 20 - vv * sc;
    c.fillStyle = '#2b2a27'; c.fillRect(0, Y(0), w, h - Y(0));
    // הרמפה
    c.strokeStyle = '#d9d8d2'; c.lineWidth = 5; c.beginPath();
    c.moveTo(X(-12), Y(0));
    for (let s = 0; s <= pr.total; s += 0.3) { const q = pr.at(s); c.lineTo(X(q.h), Y(q.v)); }
    c.stroke(); c.lineWidth = 1;
    // פס אדום
    c.strokeStyle = '#d32f2f'; c.lineWidth = 1.5; c.beginPath();
    for (let s = 0; s <= pr.total; s += 0.3) { const q = pr.at(s), nx = -Math.sin(q.th), ny = Math.cos(q.th); const px = X(q.h + nx * 0.25), py = Y(q.v + ny * 0.25); s ? c.lineTo(px, py) : c.moveTo(px, py); }
    c.stroke(); c.lineWidth = 1;
    // תוויות זוויות
    c.fillStyle = '#8b8a83'; c.font = '11px Rubik'; c.textAlign = 'center';
    for (const d of [45, 90, 135, 170]) { let s = 0; for (; s < pr.total; s += 0.2) if (pr.at(s).th * 180 / Math.PI >= d) break; const q = pr.at(s); c.fillText(`${d}°`, X(q.h - Math.sin(q.th) * -2.2), Y(q.v + Math.cos(q.th) * -2.2)); }
    c.fillText('21 מ\'', X(pr.at(pr.total).h) - 0, Y(21.6));
    // תנועה
    const sEnd = st.fall ? st.fall.s : pr.total;
    if (!fallState) {
      pos += st.v * dt;
      if (pos > sEnd + 0.01) { fallState = { t: 0, ...posState(pr, Math.min(pos, sEnd), st.v) }; }
    } else {
      fallState.t += dt;
      fallState.vy -= G * dt; fallState.x += fallState.vx * dt; fallState.y += fallState.vy * dt; fallState.rot += fallState.w * dt;
      if (fallState.y < 0.4 || fallState.t > 3.5) { fallState = null; pos = -14; }
    }
    let cx, cy, ang;
    if (pos < 0 && !fallState) { cx = pos; cy = 0; ang = 0; }
    else if (!fallState) { const q = pr.at(pos); cx = q.h - Math.sin(q.th) * 0.6; cy = q.v + Math.cos(q.th) * 0.6; ang = q.th; }
    else { cx = fallState.x; cy = fallState.y; ang = fallState.rot; }
    if (pos < 0 && !fallState) cy = 0.6;
    // מכונית
    c.save(); c.translate(X(cx), Y(cy)); c.rotate(ang);
    c.fillStyle = '#f2f2ef'; c.beginPath(); c.roundRect(-2.2 * sc, -0.5 * sc, 4.4 * sc, 1.0 * sc, 0.4 * sc); c.fill();
    c.fillStyle = '#1d2024'; c.beginPath(); c.ellipse(-0.3 * sc, -0.55 * sc, 1.0 * sc, 0.35 * sc, 0, Math.PI, 0); c.fill();
    c.restore();
    // חיצי כוח
    if (st.showF && !fallState && pos >= 0) {
      const q = pr.at(pos), nx = -Math.sin(q.th), ny = Math.cos(q.th);
      const L = sc * 1.6; // פיקסלים ליחידת משקל
      const n = st.v * st.v * q.k / G + Math.cos(q.th) + st.s;
      const px = X(cx), py = Y(cy);
      c.lineWidth = 2.5;
      // במסך: x הפוך (הרכב נוסע שמאלה) ו-y הפוך (למעלה), לכן וקטור פיזיקלי (a,b) הוא (−a,−b) בפיקסלים
      c.strokeStyle = c.fillStyle = '#e66767'; arrow(c, px, py, px, py + L);
      c.strokeStyle = c.fillStyle = '#199e70'; if (st.s > 0) arrow(c, px, py, px + nx * st.s * L, py + ny * st.s * L);
      c.strokeStyle = c.fillStyle = '#3987e5'; if (n > 0) arrow(c, px, py, px - nx * n * L, py - ny * n * L);
      c.lineWidth = 1;
      c.font = '12px Rubik'; c.textAlign = 'left'; c.fillStyle = '#f0efec';
      c.fillText(`זווית ${(q.th * 180 / Math.PI).toFixed(0)}° · נורמלי ${n.toFixed(2)} משקל · הנוסע מרגיש ${(n - st.s + 0).toFixed(1)}g`, 12, 20);
    }
    // מקרא
    c.font = '12px Rubik'; c.textAlign = 'left';
    [['כובד', '#e66767'], ['יניקה', '#199e70'], ['נורמלי', '#3987e5']].forEach(([t, col], i) => { c.fillStyle = col; c.fillRect(12, 34 + i * 16, 12, 3); c.fillStyle = '#c3c2b7'; c.fillText(t, 30, 38 + i * 16); });
    if (fallState) { c.fillStyle = '#ff9b97'; c.font = 'bold 14px Rubik'; c.fillText(st.fall ? (st.fall.why === 'normal' ? 'ניתוק מהמשטח!' : 'החלקה!') : 'שחרור בקצה', 12, 100); }
  });
}
function posState(pr, s, v) {
  const q = pr.at(s);
  return { x: q.h - Math.sin(q.th) * 0.6, y: q.v + Math.cos(q.th) * 0.6, vx: Math.cos(q.th) * v, vy: Math.sin(q.th) * v, rot: q.th, w: 1.2 };
}
