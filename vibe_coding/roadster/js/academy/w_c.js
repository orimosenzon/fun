// ווידג'טים: תקציב אוויר, מחשבון ריחוף, הגוף, השוואת g, מנוע, אווירודינמיקה, ציר זמן, מונחים.
import { LinePlot, SERIES, setupCanvas } from './plot.js';
import { slider, buttons, readout, f } from './ui.js';
import { Nozzle, AirTank, PSI, BAR } from '../gas.js';
import { muAt } from '../tire.js';
import { AERO } from '../params.js';

const G = 9.80665, M = 2237;

function barsHTML(rows, max, unit, legend) {
  return (legend ? `<div class="note" style="margin-bottom:6px">${legend}</div>` : '') + rows.map(([n, v, me, label]) =>
    `<div class="row ${me ? 'me' : ''}" title="${n}: ${label ?? v} ${unit}"><span class="name">${n}</span><div class="track"><div class="fill" style="width:${Math.min(100, v / max * 100)}%;background:${me ? SERIES[1] : SERIES[0]}"></div></div><span class="val">${label ?? v}${unit}</span></div>`).join('');
}
const sw = (c) => `<span style="display:inline-block;width:10px;height:10px;background:${c};border-radius:2px;vertical-align:middle"></span>`;

// ================= תקציב =================
export function budget(root) {
  const cv = root.querySelector('canvas.bud');
  const d = document.createElement('div'); d.className = 'bars'; cv.replaceWith(d);
  const rows = [
    ['המכל המלא (רוקט)', 227, false],
    ['ירידה מהקיר: היפוך ובלימה', 120, true],
    ['קפיצה ל-14 מ\' (0.7 ש\' דחף מלא)', 132, true],
    ['קיקפליפ מעל השורה', 106, true],
    ['0–96 קמ"ש עם הסילונים', 61, true],
    ['שנייה אחת של ריחוף', 34, true],
  ];
  d.innerHTML = barsHTML(rows, 240, ' ק"ג', `${sw(SERIES[0])} מה שיש · ${sw(SERIES[1])} מה שעולה כל תרגיל (ק"ג אוויר)`);
}

// ================= מחשבון ריחוף =================
export function hoverCalc(el) {
  el.innerHTML = `<h4 style="margin:0 0 8px">מחשבון: כמה זמן אפשר לרחף?</h4><div class="wgrid" style="margin-top:0"><div class="out readout"></div><div class="ctrls"></div></div>`;
  const ctrls = el.querySelector('.ctrls'), out = el.querySelector('.out');
  const st = { m: 2237, V: 600, p: 379 };
  const run = () => {
    const tank = new AirTank({ volume: st.V / 1000, pFill: st.p * BAR });
    const nz = new Nozzle({ throatArea: Nozzle.sizeFor(22000, 5500 * PSI, 293.15) });
    const need = st.m * G;
    let t = 0; const dt = 0.002, m0 = tank.m;
    if (4 * nz.perf(tank.p, tank.T).F < need) {
      out.innerHTML = `<div class="kv"><span>אוויר במכל</span><b>${f(m0)} ק"ג</b></div><div class="kv"><span>תוצאה</span><b style="color:#ff9b97">לא ממריא</b></div><div class="note">בלחץ הזה ארבעת הנחירים לא נותנים דחף ששווה למשקל.</div>`;
      return;
    }
    while (t < 60) {
      const full = 4 * nz.perf(tank.p, tank.T).F;
      if (full < need) break;
      const u = need / full;
      const perf = nz.perf(u * tank.p, tank.T);
      tank.step(4 * perf.mdot, dt); t += dt;
    }
    out.innerHTML = `<div class="kv"><span>אוויר במכל</span><b>${f(m0)} ק"ג</b></div><div class="kv"><span>זמן ריחוף</span><b style="color:#7fd9b0;font-size:20px">${t.toFixed(1)} ש'</b></div><div class="kv"><span>צריכה ממוצעת</span><b>${f((m0 - tank.m) / t, 1)} ק"ג/ש'</b></div><div class="kv"><span>נשאר כשנגמר הדחף</span><b>${f(tank.m)} ק"ג (${(tank.p / BAR).toFixed(0)} בר)</b></div><div class="note">הריחוף נגמר כשהלחץ יורד עד שגם בפתיחה מלאה הנחירים לא מחזיקים את המשקל. האוויר שנשאר במכל "תקוע".</div>`;
  };
  slider(ctrls, { label: 'מסת הרכב', min: 1200, max: 3000, step: 10, value: 2237, fmt: (v) => `${f(v)} ק"ג`, onInput: (v) => { st.m = v; run(); } });
  slider(ctrls, { label: 'נפח המכלים', min: 100, max: 2000, step: 50, value: 600, fmt: (v) => `${v} ל'`, onInput: (v) => { st.V = v; run(); } });
  slider(ctrls, { label: 'לחץ מילוי', min: 100, max: 700, step: 10, value: 379, fmt: (v) => `${v} בר`, onInput: (v) => { st.p = v; run(); } });
  run();
}

// ================= הגוף =================
export function body(root) {
  const svg = root.querySelector('svg.person'), ctrls = root.querySelector('.ctrls');
  const S = {
    rest: ['ישיבה', 0, 1, 'המושב דוחף למעלה ב-1g. זה מה שאנחנו מכירים כ"משקל".'],
    launch: ['זינוק (רוקט)', 4.6, 1, 'המשענת דוחפת קדימה ב-4–5g. הגוף נמעך לאחור, הראש נצמד למשענת ושוקל 23 ק"ג. קצר מדי (חצי שנייה) בשביל שהראייה תיפגע.'],
    brake: ['בלימה חזקה', -1.4, 1, 'החגורות מחזיקות אתכם. ברודסטר עם יניקה מלאה הבלימה יכולה להגיע ל-2.5g, כמו בפורמולה 1 במהירות נמוכה.'],
    curve: ['כניסה לקיר', 0, 2.6, 'הרמפה מתעקלת והמושב דוחף למעלה ב-2.6g: אתם כבדים פי 2.6. הדם נוטה לרדת לרגליים, אבל זה עדיין רחוק מהגבול.'],
    top: ['תלוי הפוך', 0, -1, 'בראש הקיר הרכב הפוך ואיטי. החגורות מחזיקות אתכם מ"ליפול לתקרה". 1g שלילי: הדם זורם לראש והפנים מתמלאות.'],
    fall: ['נפילה', 0, 0, 'אחרי השחרור ולפני שהנחירים נדלקים: חוסר משקל מוחלט, כמו ברגע העליון של רכבת הרים. הקיבה "עולה".'],
    landing: ['בלימת הנפילה', 0, 3.3, 'ארבעת נחירי הפינות דוחפים למעלה, ואתם נלחצים למושב ב-3.3g (הקריאה מההדגמה). הראש שוקל 16 ק"ג.'],
    kick: ['קיקפליפ', 0, 0.1, 'כמעט חוסר משקל, והעולם מסתובב פעם אחת בשנייה. מחלת תנועה כמעט מובטחת בפעם הראשונה.'],
    fighter: ['טייס קרב', 0, 9, 'לשם השוואה: 9g בפנייה חדה. בלי חליפת לחץ ותרגול נשימה, מאבדים הכרה בתוך שניות.'],
  };
  const ro = document.createElement('div'); ro.className = 'readout';
  let cur = { gx: 0, gy: 1 }, target = { gx: 0, gy: 1 };
  buttons(ctrls, Object.entries(S).map(([k, v]) => [k, v[0]]), { selected: 'rest', onPick: (k) => { target = { gx: S[k][1], gy: S[k][2] }; ro.innerHTML = `<div class="kv"><span>הכוח מהמושב</span><b>${Math.hypot(S[k][1], S[k][2]).toFixed(1)}g</b></div><div class="kv"><span>משקל הראש</span><b>${(5 * Math.hypot(S[k][1], S[k][2])).toFixed(0)} ק"ג</b></div><p style="margin:6px 0 0;line-height:1.6">${S[k][3]}</p>`; } });
  ctrls.appendChild(ro);
  ro.innerHTML = `<p style="margin:0">${S.rest[3]}</p>`;
  const draw = () => {
    cur.gx += (target.gx - cur.gx) * 0.12; cur.gy += (target.gy - cur.gy) * 0.12;
    const gx = cur.gx, gy = cur.gy;
    // הגוף זז מעט נגד כיוון הכוח (קפיץ של המושב), והראש יותר
    const dx = -gx * 3, dy = (gy - 1) * 2.5;
    const hx = 150 + dx * 2.2, hy = 92 + dy * 2.2;
    // דם: +Gz → לרגליים, −Gz → לראש
    const blood = Math.max(-1, Math.min(1, (gy - 1) / 4));
    const headCol = blood < 0 ? `rgb(${200 + (-blood) * 55},${150 - (-blood) * 80},${140 - (-blood) * 80})` : `rgb(${200 - blood * 60},${170 - blood * 30},${150 - blood * 10})`;
    const L = Math.min(110, Math.hypot(gx, gy) * 16);
    const ang = Math.atan2(-gy, gx);
    let s = `<rect x="0" y="0" width="320" height="300" fill="#161615"/>`;
    s += `<text x="16" y="24" fill="#8b8a83" font-size="12" font-family="Rubik">כיוון הנסיעה →</text>`;
    // מושב
    s += `<path d="M 95 215 L 205 215 L 205 228 L 95 228 Z" fill="#3a3a37"/><path d="M 95 228 L 80 90 L 96 86 L 112 215 Z" fill="#3a3a37"/>`;
    // גוף
    s += `<g stroke="#e8e6e0" stroke-width="16" stroke-linecap="round" fill="none">`;
    s += `<line x1="${120 + dx}" y1="${200 + dy}" x2="${132 + dx * 1.6}" y2="${118 + dy * 1.6}"/>`; // גו
    s += `<line x1="${120 + dx}" y1="${200 + dy}" x2="${205 + dx * 0.5}" y2="${205 + dy * 0.6}"/><line x1="${205 + dx * 0.5}" y1="${205 + dy * 0.6}" x2="${232}" y2="${262}"/>`; // רגל
    s += `<line x1="${132 + dx * 1.6}" y1="${128 + dy * 1.6}" x2="${180 + dx}" y2="${160 + dy}"/><line x1="${180 + dx}" y1="${160 + dy}" x2="215" y2="150"/>`; // יד
    s += `</g>`;
    s += `<circle cx="${hx - 12}" cy="${hy}" r="21" fill="${headCol}" stroke="#e8e6e0" stroke-width="3"/>`;
    s += `<rect x="208" y="136" width="10" height="30" rx="4" fill="#5a5f66"/>`;
    // חץ הכוח
    const ax = 150, ay = 160, ex = ax + Math.cos(ang) * L, ey = ay + Math.sin(ang) * L;
    if (L > 4) s += `<line x1="${ax}" y1="${ay}" x2="${ex}" y2="${ey}" stroke="${SERIES[2]}" stroke-width="5"/><polygon points="${ex},${ey} ${ex - 12 * Math.cos(ang - 0.4)},${ey - 12 * Math.sin(ang - 0.4)} ${ex - 12 * Math.cos(ang + 0.4)},${ey - 12 * Math.sin(ang + 0.4)}" fill="${SERIES[2]}"/>`;
    s += `<text x="300" y="24" fill="#f0efec" font-size="22" font-family="JetBrains Mono" text-anchor="end">${Math.hypot(gx, gy).toFixed(1)}g</text>`;
    // מד דם
    s += `<text x="300" y="70" fill="#8b8a83" font-size="11" font-family="Rubik" text-anchor="end">הדם</text>`;
    s += `<rect x="286" y="80" width="12" height="150" rx="6" fill="#2a2a28"/>`;
    const by = 155 + blood * 65;
    s += `<circle cx="292" cy="${by}" r="9" fill="#c62828"/><text x="280" y="86" fill="#8b8a83" font-size="10" font-family="Rubik" text-anchor="end">ראש</text><text x="280" y="232" fill="#8b8a83" font-size="10" font-family="Rubik" text-anchor="end">רגליים</text>`;
    svg.innerHTML = s;
    requestAnimationFrame(draw);
  };
  draw();
}

// ================= השוואת g =================
export function gbars(root) {
  const cv = root.querySelector('canvas.gb');
  const d = document.createElement('div'); d.className = 'bars'; cv.replaceWith(d);
  const rows = [
    ['מעלית שמתחילה לעלות (Gz)', 1.2], ['מעבורת חלל, שיגור', 3], ['רכבת הרים, שיא (Gz)', 4.5], ['דראגסטר Top Fuel', 5], ['פורמולה 1, בלימה', 5.5], ['טייס קרב עם חליפה (Gz)', 9],
    ['רודסטר: תלוי הפוך בראש הקיר (−Gz)', 1, true, '−1'], ['רודסטר: כניסה לקיר (Gz)', 2.6, true], ['רודסטר: בלימת הנפילה (Gz)', 3.3, true], ['רודסטר רוקט: זינוק', 4.9, true],
  ];
  d.innerHTML = barsHTML(rows, 9.5, 'g', `${sw(SERIES[0])} לשם השוואה · ${sw(SERIES[1])} הרודסטר (סימולטור והדגמה)`);
}

// ================= מנוע וצמיגים =================
export function motor(root) {
  const pc = root.querySelector('canvas.mplot'), ctrls = root.querySelector('.ctrls');
  const plot = new LinePlot(pc, { xLabel: 'מהירות (קמ"ש)', yLabel: 'כוח (kN)', xMax: 380, yMax: 100, xDec: 0, yDec: 1, unitX: ' קמ"ש', unitY: ' kN' });
  const st = { wing: 'high' };
  const run = () => {
    const am = AERO.modes[st.wing];
    const tire = [], tireFan = [], mot = [], drag = [], thr = [];
    for (let k = 0; k <= 380; k += 4) {
      const v = k / 3.6;
      const down = 0.5 * 1.204 * am.ClA * v * v;
      const N0 = M * G + down, N1 = N0 + 44000;
      tire.push([k, muAt(N0 / 4) * N0 / 1000]);
      tireFan.push([k, muAt(N1 / 4) * N1 / 1000]);
      mot.push([k, Math.min(12000 / 0.345, 1e6 / Math.max(v, 0.1)) / 1000]);
      drag.push([k, 0.5 * 1.204 * am.CdA * v * v / 1000]);
      thr.push([k, 80]);
    }
    plot.set([{ name: 'צמיגים', data: tire }, { name: 'צמיגים + יניקה', data: tireFan }, { name: 'מנועים', data: mot }, { name: 'גרר', data: drag }, { name: 'סילונים', data: thr, dash: [5, 4] }]);
  };
  buttons(ctrls, [['road', 'כנף: כביש'], ['low', 'כנף: גרר נמוך'], ['high', 'כנף: הצמדה']], { selected: 'high', onPick: (k) => { st.wing = k; run(); } });
  const n = document.createElement('p'); n.className = 'note'; n.style.margin = '0';
  n.textContent = 'עד כ-130 קמ"ש המנועים חזקים מהצמיגים בלי יניקה, ולכן היניקה עוזרת כבר בזינוק. מעל זה ההספק (מגה-ואט) הוא הגבול. הגרר עולה עם ריבוע המהירות, אבל במגה-ואט הוא היה משתווה למנועים רק בכ-450 קמ"ש. לכן המהירות המרבית המוצהרת (370 קמ"ש) היא כנראה מגבלה של המנועים או הצמיגים, לא של האוויר.';
  ctrls.appendChild(n);
  run();
  addEventListener('resize', () => plot.draw());
}

// ================= אווירודינמיקה =================
export function aero(root) {
  const pc = root.querySelector('canvas.aplot'), ctrls = root.querySelector('.ctrls');
  const plot = new LinePlot(pc, { xLabel: 'מהירות (קמ"ש)', yLabel: 'כוח הצמדה (ק"ג)', xMax: 360, yMax: 7000, xDec: 0, yDec: 0, unitX: ' קמ"ש', unitY: ' ק"ג' });
  const kg = (ClA, v) => 0.5 * 1.204 * ClA * v * v / G;
  const ser = ['road', 'low', 'high'].map((k) => ({ name: `כנף: ${AERO.modes[k].nameHe}`, data: [] }));
  const fanS = { name: 'מאווררים (Max Grip)', data: [], color: SERIES[2] };
  for (let k = 0; k <= 360; k += 4) { const v = k / 3.6; ['road', 'low', 'high'].forEach((m, i) => ser[i].data.push([k, kg(AERO.modes[m].ClA, v)])); fanS.data.push([k, 4515]); }
  ser[0].color = SERIES[3]; ser[1].color = SERIES[0]; ser[2].color = SERIES[1];
  plot.vlines = [{ x: 40, label: 'הקיר' }];
  plot.set([...ser, fanS]);
  const n = document.createElement('p'); n.className = 'note'; n.style.margin = '0';
  n.innerHTML = 'הכנף והגוף נותנים כוח ∝ v², והמאווררים כוח קבוע. רק מעל ~300 קמ"ש הכנף מתחילה להתחרות במאווררים. ערכי ClA הם הערכה שלנו.';
  ctrls.appendChild(n);
  addEventListener('resize', () => plot.draw());
}

// ================= ציר זמן =================
export function timeline(root) {
  const svg = root.querySelector('svg.tl'), info = root.querySelector('.tl-info');
  const CAT = { gas: ['גז דחוס, טילים ומכלים', SERIES[0]], fan: ['יניקה', SERIES[2]], ctl: ['בקרה ממוחשבת', SERIES[3]], tesla: ['טסלה', SERIES[1]] };
  const E = [
    ['1888', 'gas', 'הנחיר של דה-לבאל', 'גוסטף דה-לבאל, מהנדס שוודי, ממציא את הנחיר המתכנס-מתבדר לטורבינת קיטור. 40 שנה אחר כך רוברט גודארד משתמש בו בטילים הראשונים.'],
    ['1970', 'fan', 'Chaparral 2J', 'מכונית ה-Can-Am הראשונה עם מאווררי יניקה וחצאיות. נאסרה אחרי עונה אחת.'],
    ['1978', 'fan', 'Brabham BT46B', 'מכונית פורמולה 1 עם מאוורר. ניצחה במרוץ היחיד שלה (שבדיה, ניקי לאודה) והוסרה.'],
    ['1979', 'ctl', 'F-16: היגוי בחוט', 'מטוס הקרב המבצעי הראשון בלי חיבור מכני בין ההגאים להגאי הטיסה: מחשב מתרגם את תנועות הטייס.'],
    ['1984', 'gas', 'MMU: ריחוף על חנקן', 'ברוס מקנדלס עוזב את מעבורת החלל בלי כבל, על כיסא עם 24 נחירי חנקן קר. אותו עיקרון כמו נחירי הרודסטר.'],
    ['1988', 'ctl', 'איירבוס A320', 'מטוס הנוסעים הראשון עם היגוי בחוט דיגיטלי מלא.'],
    ['2008', 'tesla', 'הרודסטר הראשון', 'המכונית הראשונה של טסלה, על שלדה של לוטוס אליס.'],
    ['2014', 'gas', 'טויוטה מיראי', 'מכונית מימן סדרתית עם מכלי COPV ב-700 בר: הלחץ גבוה כמעט פי 2 מזה של הרודסטר.'],
    ['2015', 'gas', 'נחיתת פלקון 9', 'לראשונה, שלב ראשון של טיל חוזר ונוחת. נחירי חנקן קר מסובבים אותו אחרי ההיפרדות.'],
    ['2016', 'gas', 'AMOS-6', 'מכל COPV של הליום כשל בתוך מכל החמצן של פלקון 9 על כן השיגור, והלוויין הישראלי הושמד. שיעור בבטיחות של מכלי לחץ.'],
    ['2017', 'tesla', 'הרודסטר 2 מוצג', 'הבטחה: 0–60 ב-1.9 ש', 'tesla'],
    ['2018', 'tesla', 'סטארמן וחבילת SpaceX', 'פברואר: הרודסטר הראשון נשלח לחלל על פלקון הבי, עם בובה בשם סטארמן. יוני: מאסק מודיע על "חבילת SpaceX" עם כעשרה נחירי גז קר לרודסטר החדש.'],
    ['2022', 'fan', 'McMurtry בגודווד', 'מכונית יניקה חשמלית שוברת את שיא גבעת גודווד.'],
    ['2023', 'ctl', 'סייברטראק', 'היגוי בחוט ראשון ברכב סדרתי של טסלה. אותה ארכיטקטורה ברודסטר, עם עול במקום הגה.'],
    ['2025', 'fan', 'נסיעה הפוכה ופטנט', 'אפריל: מקמרטרי נוסעת הפוך על מתקן מסתובב. אוגוסט: פטנט של טסלה על מאווררים עם חצאיות אדפטיביות (US 12,377,920, לפי דיווח).'],
    ['2026', 'tesla', 'ההדגמה', '15 באוקטובר: טיפוס על קיר עד 170°, נפילה מ-21 מ\' ונחיתה, וקיקפליפ מעל חמש מכוניות.'],
    ['2028', 'tesla', 'ייצור (מתוכנן)', 'שלושה דגמים: Performance, Thrust, Rocket.'],
  ];
  E[10][3] = 'נובמבר: טסלה מציגה את הדור השני עם הבטחה ל-0–60 ב-1.9 ש\', 1,000 ק"מ טווח ומסירות ב-2020. אחרי עיכובים ב-2020, 2021, 2023 ו-2024 הוא הוצג מחדש ב-2026.';
  const W = 1000, n = E.length, x0 = 75, x1 = 925, y = 150;
  svg.style.direction = 'ltr'; // כדי ש-text-anchor יתנהג פיזית (ימין/שמאל) גם עם טקסט עברי
  let s = `<line x1="${x0}" y1="${y}" x2="${x1}" y2="${y}" stroke="rgba(255,255,255,0.25)" stroke-width="2"/>`;
  E.forEach((e, i) => {
    const x = x1 - (x1 - x0) * i / (n - 1); // מימין לשמאל, כמו הקריאה
    const up = i % 2 === 0, col = CAT[e[1]][1];
    const ly = up ? y - 34 - (i % 4 === 0 ? 30 : 0) : y + 34 + (i % 4 === 1 ? 30 : 0);
    s += `<g class="ev" data-i="${i}" style="cursor:pointer"><line x1="${x}" y1="${y}" x2="${x}" y2="${ly + (up ? 8 : -14)}" stroke="${col}" stroke-opacity="0.5"/>`;
    s += `<circle cx="${x}" cy="${y}" r="9" fill="#161615"/><circle cx="${x}" cy="${y}" r="6.5" fill="${col}"/>`;
    s += `<text x="${x}" y="${ly}" fill="#f0efec" font-size="13" font-family="Rubik" text-anchor="middle">${e[2]}</text>`;
    s += `<text x="${x}" y="${ly + (up ? -16 : 16)}" fill="${col}" font-size="12" font-family="JetBrains Mono" text-anchor="middle">${e[0]}</text>`;
    s += `<circle cx="${x}" cy="${y}" r="18" fill="transparent"/></g>`;
  });
  // מקרא
  Object.values(CAT).forEach(([name, col], i) => { s += `<circle cx="${W - 20 - i * 170}" cy="18" r="6" fill="${col}"/><text x="${W - 32 - i * 170}" y="22" fill="#c3c2b7" font-size="13" font-family="Rubik" text-anchor="end">${name}</text>`; });
  svg.innerHTML = s;
  const show = (i) => { const e = E[i]; info.innerHTML = `<span class="yr">${e[0]}</span> · <b>${e[2]}</b><br>${e[3]}`; };
  svg.querySelectorAll('.ev').forEach((g) => { g.addEventListener('mouseenter', () => show(+g.dataset.i)); g.addEventListener('click', () => show(+g.dataset.i)); });
  show(15);
}

// ================= מונחים =================
export const TERMS = {
  skirt: ['skirt', 'חצאית: לוח גמיש בשולי הרצפה שכמעט נוגע בכביש ואוטם את החלל מתחת לרכב. השם מהאנגלית, כמו שולי בגד.'],
  jerk: ['jerk', 'ג\'רק: קצב השינוי של התאוצה (מטר לשנייה בשלישית). באנגלית "טלטלה פתאומית". ג\'רק גבוה = מכה, גם כשהתאוצה עצמה קטנה.'],
  clothoid: ['clothoid / Euler spiral', 'קלותואידה, או ספירלת אוילר: עקומה שהעקמומיות שלה גדלה לינארית עם האורך. מהיוונית klōthō, "לטוות", על שם קלותו, אלת הגורל שטוותה את חוט החיים: העקומה מתלפפת כמו חוט על פלך.'],
  throat: ['throat', 'גרון: החתך הצר ביותר בנחיר. שם הזרימה מגיעה בדיוק למהירות הקול ("זרימה חנוקה"), ולכן הוא שקובע כמה גז יוצא בשנייה.'],
  isp: ['specific impulse, Isp', 'אימפולס סגולי: הדחף לכל יחידת ספיקת משקל, בשניות. "סגולי" (specific) = ליחידת מסה. Isp × 9.81 = מהירות הפליטה האפקטיבית במטר לשנייה.'],
  copv: ['Composite Overwrapped Pressure Vessel', 'כלי לחץ עטוף בחומר מרוכב. composite = מרוכב, מהלטינית componere, "להרכיב". מעטפת דקה עטופה בסיבי פחמן ושרף.'],
  inertia: ['moment of inertia', 'מומנט התמד: ההתנגדות של גוף לשינוי במהירות הסיבוב שלו, כמו מסה בתנועה ישרה. מהלטינית inertia, "עצלות". תלוי במרחק המסה מהציר בריבוע.'],
  allocation: ['control allocation', 'הקצאת בקרה: חלוקת כוח או מומנט רצוי בין כמה מפעילים (כאן: נחירים), כשלכל אחד מגבלות משלו, כמו "רק לדחוף" ו"עד 22 קילוניוטון".'],
  pd: ['proportional controller', 'בקר יחסי: מפעיל תיקון שגדל ביחס לשגיאה. כאן השגיאה היא ההפרש בין קצב הסיבוב הרצוי לקצב בפועל, והתיקון הוא מומנט.'],
  fbw: ['fly-by-wire / steer-by-wire', 'היגוי בחוט: הפקודות עוברות כאות חשמלי ("דרך חוט", by wire) ולא דרך מוט, כבל או צינור הידראולי. מחשב יושב באמצע ויכול לשנות אותן.'],
};
export function terms() {
  const pop = document.getElementById('term-pop');
  document.querySelectorAll('.term').forEach((el) => {
    const t = TERMS[el.dataset.t]; if (!t) return;
    const show = () => {
      pop.innerHTML = `<span class="en">${t[0]}</span>${t[1]}`;
      pop.classList.remove('hidden');
      const r = el.getBoundingClientRect();
      const pw = Math.min(340, innerWidth - 20);
      let left = Math.max(10, Math.min(innerWidth - pw - 10, r.left + r.width / 2 - pw / 2));
      pop.style.left = left + 'px'; pop.style.top = (r.bottom + 8) + 'px';
      if (r.bottom + 8 + pop.offsetHeight > innerHeight) pop.style.top = (r.top - pop.offsetHeight - 8) + 'px';
    };
    el.addEventListener('mouseenter', show); el.addEventListener('click', show);
    el.addEventListener('mouseleave', () => pop.classList.add('hidden'));
  });
  addEventListener('scroll', () => pop.classList.add('hidden'), { passive: true });
}
