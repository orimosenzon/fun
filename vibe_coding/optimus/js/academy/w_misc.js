// ווידג'טים נוספים: למידת חיזוק (מטוטלת על עגלה), כוח מכה מול ISO/TS 15066, עלות תנועה, זמן עבודה וציר זמן.
import { LinePlot, SERIES, setupCanvas } from './plot.js';
import { slider, readout, buttons, f, animate } from './ui.js';
import { OPPOINTS } from './oppoints.js';
import { BODY } from '../robot/spec.js';

// ================= למידת חיזוק: מטוטלת הפוכה על עגלה =================
function cartStep(s, u) {
  // המודל הקלאסי של ברטו, סאטון ואנדרסון (1983)
  const g = 9.8, mc = 1, mp = 0.1, l = 0.5, dt = 0.02, F = 10 * Math.max(-1, Math.min(1, u));
  const [x, xd, th, thd] = s, ct = Math.cos(th), sn = Math.sin(th);
  const tmp = (F + mp * l * thd * thd * sn) / (mc + mp);
  const thacc = (g * sn - ct * tmp) / (l * (4 / 3 - mp * ct * ct / (mc + mp)));
  const xacc = tmp - mp * l * thacc * ct / (mc + mp);
  return [x + dt * xd, xd + dt * xacc, th + dt * thd, thd + dt * thacc];
}
function episode(wts, seed, maxT = 500) {
  let r = seed;
  const rnd = () => { r = (r * 16807) % 2147483647; return (r / 2147483647 - 0.5) * 0.1; };
  let s = [rnd(), rnd(), rnd(), rnd()], t = 0;
  for (; t < maxT; t++) {
    const u = wts[0] * s[0] + wts[1] * s[1] + wts[2] * s[2] + wts[3] * s[3];
    s = cartStep(s, u);
    if (Math.abs(s[0]) > 2.4 || Math.abs(s[2]) > 0.21) break;
  }
  return t;
}
export function rl(root) {
  const cv = root.querySelector('canvas.anim'), pcv = root.querySelector('canvas.plot'), ctrls = root.querySelector('.ctrls');
  const plot = new LinePlot(pcv, { xLabel: 'איטרציה', yLabel: 'צעדים עד נפילה', xMin: 0, xMax: 60, yMin: 0, yMax: 520, xDec: 0, yDec: 0 });
  let wts = [0, 0, 0, 0], hist = [], it = 0, training = false, seedBase = 7;
  const ro = readout(ctrls, [['w', 'המשקלות'], ['r', 'ציון נוכחי'], ['i', 'איטרציה']]);
  const row = document.createElement('div'); row.className = 'btns'; ctrls.appendChild(row);
  const mk = (t, fn) => { const b = document.createElement('button'); b.textContent = t; b.onclick = fn; row.appendChild(b); return b; };
  const trainBtn = mk('▶ אמן', () => { training = !training; trainBtn.textContent = training ? '⏸ עצור' : '▶ אמן'; });
  mk('התחל מחדש', () => { wts = [0, 0, 0, 0]; hist = []; it = 0; plot.set([]); seedBase++; });
  const score = (w) => (episode(w, 11 + seedBase) + episode(w, 23 + seedBase) + episode(w, 37 + seedBase)) / 3;
  let r = 12345; const gauss = () => { r = (r * 16807) % 2147483647; const u = r / 2147483647; r = (r * 16807) % 2147483647; const v = r / 2147483647; return Math.sqrt(-2 * Math.log(u + 1e-12)) * Math.cos(2 * Math.PI * v); };
  function trainStep() {
    // ARS: 8 כיוונים אקראיים, מנסים ±, ומתקדמים בכיוון הממוצע המשוקלל
    const nu = 0.3, alpha = 0.6, n = 8;
    const step = [0, 0, 0, 0], rs = [];
    for (let k = 0; k < n; k++) {
      const d = [gauss(), gauss(), gauss(), gauss()];
      const rp = score(wts.map((w, i) => w + nu * d[i])), rm = score(wts.map((w, i) => w - nu * d[i]));
      rs.push(rp, rm);
      for (let i = 0; i < 4; i++) step[i] += (rp - rm) * d[i];
    }
    const sd = Math.sqrt(rs.reduce((a, b) => a + b * b, 0) / rs.length - (rs.reduce((a, b) => a + b, 0) / rs.length) ** 2) || 1;
    wts = wts.map((w, i) => w + alpha / (n * sd) * step[i]);
    it++;
    hist.push([it, score(wts)]);
    plot.set([{ name: 'ציון', data: hist }], { xMax: Math.max(60, it + 5) });
  }
  let s = [0.02, 0, 0.05, 0], tAcc = 0, uiT = 0, demoT = 0;
  animate(root, (dt) => {
    if (training && (uiT += dt) > 0.12) { uiT = 0; trainStep(); }
    // הדגמה חיה של המדיניות הנוכחית
    tAcc += dt;
    while (tAcc > 0.02) {
      tAcc -= 0.02; demoT++;
      const u = wts[0] * s[0] + wts[1] * s[1] + wts[2] * s[2] + wts[3] * s[3];
      s = cartStep(s, u);
      if (Math.abs(s[0]) > 2.4 || Math.abs(s[2]) > 0.21 || demoT > 500) { s = [(Math.random() - 0.5) * 0.1, 0, (Math.random() - 0.5) * 0.1, 0]; demoT = 0; }
    }
    ro.set({ w: wts.map(w => f(w, 2)).join(', '), r: hist.length ? `${f(hist[hist.length - 1][1], 0)} / 500` : '–', i: it });
    const { c, w, h } = setupCanvas(cv);
    c.clearRect(0, 0, w, h);
    const S = w / 6, gy = h * 0.72, cx = w / 2 + s[0] * S;
    c.strokeStyle = 'rgba(255,255,255,.25)'; c.beginPath(); c.moveTo(w / 2 - 2.4 * S, gy + 16); c.lineTo(w / 2 + 2.4 * S, gy + 16); c.stroke();
    c.fillStyle = '#d6b980'; c.fillRect(cx - 34, gy - 10, 68, 26);
    const L = 0.5 * 2 * S;
    c.strokeStyle = '#f0efec'; c.lineWidth = 7; c.lineCap = 'round';
    c.beginPath(); c.moveTo(cx, gy); c.lineTo(cx + Math.sin(s[2]) * L, gy - Math.cos(s[2]) * L); c.stroke();
    c.fillStyle = '#8b8a83'; c.font = '12px Rubik'; c.textAlign = 'center'; c.fillText(`צעד ${demoT}`, w / 2, 18);
  });
}

// ================= כוח מכה מול ISO/TS 15066 =================
// גבולות (מגע קצר, בערך פי 2 מהגבול הקוואזי-סטטי) וקבועי קפיץ של רקמה, לפי טבלאות התקן (מעוגל)
const REGIONS = { face: ['פנים', 130, 75], chest: ['חזה', 280, 25], hand: ['יד ואצבעות', 280, 75], upperArm: ['זרוע', 300, 30], thigh: ['ירך', 440, 50] };
export function impact(root) {
  const cv = root.querySelector('canvas.plot'), ctrls = root.querySelector('.ctrls');
  let m = 3, reg = 'chest';
  const plot = new LinePlot(cv, { xLabel: 'מהירות המגע (מ\'/ש\')', yLabel: 'כוח שיא (ניוטון)', xMin: 0, xMax: 2, yMin: 0, yMax: 900, xDec: 2, yDec: 0, unitY: ' נ\'' });
  slider(ctrls, { label: 'מסה אפקטיבית', min: 0.5, max: 20, step: 0.5, value: m, fmt: (v) => `${v} ק"ג`, onInput: (v) => { m = v; draw(); } });
  buttons(ctrls, Object.entries(REGIONS).map(([k, v]) => [k, v[0]]), { selected: reg, onPick: (k) => { reg = k; draw(); } });
  const ro = readout(ctrls, [['v', 'מהירות מרבית מותרת'], ['k', 'קשיחות הרקמה']]);
  function draw() {
    const [name, Fmax, kmm] = REGIONS[reg];
    const k = kmm * 1000;
    // שני גופים עם קפיץ ביניהם: אנרגיה קינטית של המסה המצומצמת הופכת לאנרגיה בקפיץ
    const mh = 40;   // מסה אפקטיבית של איבר האדם (ק"ג), מקורב
    const mu = 1 / (1 / m + 1 / mh);
    const F = (v) => v * Math.sqrt(k * mu);
    const data = Array.from({ length: 81 }, (_, i) => [i * 0.025, F(i * 0.025)]);
    plot.vlines = [];
    plot.markers = [{ x: Math.min(2, Fmax / Math.sqrt(k * mu)), y: Fmax, label: `גבול: ${Fmax} נ'` }];
    plot.set([{ name: `מכה ב${name}`, data }, { name: 'גבול התקן', data: [[0, Fmax], [2, Fmax]], dash: [5, 4], color: '#ff4d3d' }]);
    ro.set({ v: `${f(Fmax / Math.sqrt(k * mu), 2)} מ'/ש'`, k: `${kmm} נ'/מ"מ` });
  }
  draw();
}

// ================= עלות תנועה =================
export function cotBars(el) {
  const m = BODY.mass.v, P = OPPOINTS.walk?.power || 250, v = 0.45;
  const mine = P / (m * 9.81 * v);
  const rows = [
    ['אדם (מטבולי, נטו)', 0.2, SERIES[2]],
    ['Cornell Ranger (2011)', 0.28, SERIES[3]],
    ['אופטימוס בסימולטור, 0.45 מ\'/ש\'', mine, '#d6b980', true],
    ['ASIMO (הערכה)', 3.2, SERIES[1]],
  ];
  const mx = Math.max(...rows.map(r => r[1]));
  el.innerHTML = rows.map(([n, val, col, me]) => `<div class="row${me ? ' me' : ''}"><span class="name">${n}</span><div class="track"><div class="fill" style="width:${val / mx * 100}%;background:${col}"></div></div><span class="val">${val.toFixed(2)}</span></div>`).join('');
}

// ================= מחשבון זמן עבודה =================
export function runtimeCalc(el) {
  el.innerHTML = '<b>כמה זמן מחזיקה הסוללה?</b><p class="note" style="margin:4px 0 10px">חלוקה של יום עבודה בין פעילויות. ההספקים נמדדו בסימולטור (עמידה, הליכה, סקוואט כ"עבודה פיזית").</p><div class="wgrid" style="margin:0"><div class="ctrls"></div><div class="rt-out"></div></div>';
  const ctrls = el.querySelector('.ctrls');
  const P = { stand: OPPOINTS.stand?.power || 97, walk: OPPOINTS.walk?.power || 250, work: Math.max(160, OPPOINTS.squat?.power || 134) * 1.6 };
  const mix = { stand: 50, walk: 30, work: 20 };
  const ro = readout(el.querySelector('.rt-out'), [['p', 'הספק ממוצע'], ['h', 'זמן עד סוללה ריקה'], ['e', 'אנרגיה ליום של 8 שעות']]);
  const upd = () => {
    const tot = mix.stand + mix.walk + mix.work || 1;
    const avg = (mix.stand * P.stand + mix.walk * P.walk + mix.work * P.work) / tot;
    ro.set({ p: `${f(avg, 0)} ואט`, h: `${f(BODY.battery.v * 1000 / avg, 1)} שעות`, e: `${f(avg * 8 / 1000, 2)} קוט"ש` });
  };
  slider(ctrls, { label: `עמידה ומניפולציה קלה (${Math.round(P.stand)} ואט)`, min: 0, max: 100, value: mix.stand, fmt: (v) => `${v}%`, onInput: (v) => { mix.stand = v; upd(); } });
  slider(ctrls, { label: `הליכה (${Math.round(P.walk)} ואט)`, min: 0, max: 100, value: mix.walk, fmt: (v) => `${v}%`, onInput: (v) => { mix.walk = v; upd(); } });
  slider(ctrls, { label: `עבודה פיזית (~${Math.round(P.work)} ואט)`, min: 0, max: 100, value: mix.work, fmt: (v) => `${v}%`, onInput: (v) => { mix.work = v; upd(); } });
  upd();
}

// ================= ציר זמן =================
const EVENTS = [
  [1206, 'אל-ג\'זארי', 'המהנדס אל-ג\'זארי מדיארבכר מתאר בספרו אוטומטים: להקת נגנים מכנית על סירה, ומשרתת שמגישה מים לרחיצה. התנועה מתוכנתת בגלגלי זיזים, הרעיון שיחזור בכל מכונה מתוכנתת.'],
  [1495, 'האביר של לאונרדו', 'לאונרדו דה וינצ\'י משרטט אביר מכני בשריון, שמזיז זרועות, ראש ולסת בעזרת כבלים וגלגלות. דגם שנבנה לפי השרטוטים ב-2002 עבד.'],
  [1774, 'הכותב של ז\'אקה-דרו', 'בובה שכותבת טקסט של עד 40 תווים בעט ודיו. הטקסט "מתוכנת" בדיסק של זיזים שאפשר להחליף: אחד המחשבים הראשונים, ועדיין עובד במוזיאון בנויישאטל.'],
  [1920, 'R.U.R.', 'המחזה של קארל צ\'אפק טובע את המילה "רובוט", מהצ\'כית robota: עבודת כפייה. הרובוטים במחזה מורדים בבני האדם.'],
  [1939, 'אלקטרו', 'רובוט של וסטינגהאוס בתערוכה העולמית בניו יורק: 2.1 מטר, הולך (על גלגלים מוסתרים), מדבר 700 מילים ומעשן סיגריות.'],
  [1961, 'יונימייט', 'הזרוע התעשייתית הראשונה, בפס של ג\'נרל מוטורס. לא דומה לאדם בכלל, וזה בדיוק הלקח: רובוט לא חייב להיראות כמונו כדי להחליף עבודה.'],
  [1972, 'ה-ZMP', 'ווקוברטוביץ\' וסטפננקו מפרסמים את התנאי לשיווי משקל של הולך דו-רגלי. בסיס לכל הרובוטים ההולכים.'],
  [1973, 'WABOT-1', 'אוניברסיטת ואסדה (פרופ\' איצ\'ירו קאטו): הרובוט ההומנואידי השלם הראשון. הולך (לאט, 45 שניות לצעד), אוחז, ומדבר יפנית.'],
  [1983, 'הקופץ של רייברט', 'מארק רייברט (אחר כך מייסד בוסטון דיינמיקס) בונה רובוט קופץ על רגל אחת שמאזן את עצמו בתנועה. הלידה של שיווי משקל דינמי.'],
  [1996, 'הונדה P2', 'אחרי עשור בסוד: הרובוט הדו-רגלי הראשון שהולך בלי כבלים, עם הסוללה והמחשב בגוף. 210 ק"ג.'],
  [2000, 'ASIMO', 'הונדה מציגה את ASIMO: 1.2 מטר, 52 ק"ג. רץ, עולה במדרגות, ונהיה סמל עולמי. פרש ב-2018.'],
  [2006, 'נקודת הלכידה', 'ג\'רי פראט וצוותו ב-IHMC מגדירים את נקודת הלכידה, והופכים התאוששות מדחיפה לבעיה פתירה.'],
  [2013, 'Atlas ו-DRC', 'בוסטון דיינמיקס בונה את Atlas ההידראולי לתחרות של DARPA. בגמר, ב-2015, הרובוטים נופלים שוב ושוב. KAIST הקוריאנית מנצחת.'],
  [2021, 'Tesla Bot', 'ב-AI Day טסלה מכריזה על הרובוט, עם רקדן בחליפה על הבמה. 173 ס"מ, 57 ק"ג, "יעשה עבודות מסוכנות, חוזרות ומשעממות".'],
  [2022, 'Bumblebee', 'ב-AI Day 2 עולה לבמה אב-טיפוס שהולך לבד בפעם הראשונה, ולידו דגם יפה יותר שעוד לא יודע ללכת. טסלה מציגה את המפעילים: 28, משישה סוגים.'],
  [2023, 'Gen 2', 'סרטון של הדור השני: 10 ק"ג פחות, הליכה מהירה ב-30%, ידיים עם חיישני מגע שמרימות ביצה.'],
  [2024, 'We, Robot', 'רובוטים מגישים משקאות ומשוחחים עם אורחים. מתברר שחלק גדול מהם הופעל מרחוק. במקביל, במפעל, אופטימוס ממיין תאי 4680 עם רשת נוירונים. בוסטון דיינמיקס מחליפה את Atlas ההידראולי בגרסה חשמלית.'],
  [2025, 'V2.5 הזהוב', 'מאסק מציג רובוט בצבע זהב ומבהיר שזה V2.5: "את V3 עוד לא הראינו". מגבלות ייצוא של מגנטים מסין פוגעות בייצור. ראש הפרויקט, מילן קובאץ\', עוזב.'],
  [2026, 'Optimus 3', 'אפריל: פטנטים של יד עם 22 דרגות חופש. יולי: פסי ייצור מותקנים בפרמונט. ספטמבר: הדמיות של V3 נמצאות באפליקציה של טסלה. החשיפה הרשמית: "קרוב לתחילת הייצור".'],
];
export function timeline(root) {
  const cv = root.querySelector('canvas.anim'), info = root.querySelector('.tl-info');
  let sel = EVENTS.length - 1, hover = -1;
  // ציר לא לינארי: המאות הראשונות דחוסות, העשורים האחרונים פרושים
  const pos = (y) => y < 1900 ? (y - 1200) / 700 * 0.22 : y < 2000 ? 0.22 + (y - 1900) / 100 * 0.38 : 0.6 + (y - 2000) / 26 * 0.4;
  const pts = () => { const r = cv.getBoundingClientRect(); return EVENTS.map((e, i) => [24 + pos(e[0]) * (r.width - 48), i % 2 ? 0.62 : 0.38]); };
  const show = () => { const e = EVENTS[sel]; info.innerHTML = `<span class="yr">${e[0]}</span> · <b>${e[1]}</b><br>${e[2]}`; };
  cv.addEventListener('pointermove', (ev) => { const r = cv.getBoundingClientRect(); const P = pts(); hover = P.findIndex(([x, y]) => Math.hypot(x - (ev.clientX - r.left), y * r.height - (ev.clientY - r.top)) < 14); cv.style.cursor = hover >= 0 ? 'pointer' : 'default'; draw(); });
  cv.addEventListener('click', () => { if (hover >= 0) { sel = hover; show(); draw(); } });
  function draw() {
    const { c, w, h } = setupCanvas(cv);
    c.clearRect(0, 0, w, h);
    const P = pts();
    c.strokeStyle = 'rgba(255,255,255,.25)'; c.lineWidth = 2; c.beginPath(); c.moveTo(24, h / 2); c.lineTo(w - 24, h / 2); c.stroke();
    c.font = '11px Rubik'; c.textAlign = 'center';
    for (const y of [1200, 1500, 1800, 1900, 1950, 2000, 2010, 2020]) { const x = 24 + pos(y) * (w - 48); c.fillStyle = '#5c5b55'; c.fillRect(x - 0.5, h / 2 - 5, 1, 10); c.fillText(y, x, h / 2 + 18); }
    EVENTS.forEach((e, i) => {
      const [x, fy] = P[i], y = fy * h, on = i === sel || i === hover;
      c.strokeStyle = 'rgba(255,255,255,.12)'; c.lineWidth = 1; c.beginPath(); c.moveTo(x, h / 2); c.lineTo(x, y); c.stroke();
      c.fillStyle = e[0] >= 2021 ? '#d6b980' : on ? '#f0efec' : '#8b8a83';
      c.beginPath(); c.arc(x, y, on ? 7 : 5, 0, Math.PI * 2); c.fill();
      if (on || e[0] >= 2021 || i % 3 === 0) { c.fillStyle = on ? '#f0efec' : '#a3a29a'; c.fillText(e[1], x, fy < 0.5 ? y - 12 : y + 20); }
    });
  }
  show(); draw();
  addEventListener('resize', draw);
}
