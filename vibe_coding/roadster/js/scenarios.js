// תרחישים: כל אחד עם נקודת התחלה, הוראות, מעקב ותוצאה.
import { LAYOUT, SPAWNS, loopGuide } from './track.js';
import { G0 } from './math.js';

const kmh = (v) => (v * 3.6).toFixed(0);
const f1 = (x) => x.toFixed(1), f2 = (x) => x.toFixed(2);

export const SCENARIOS = [
  {
    key: 'free', num: 1, nameHe: 'נסיעה חופשית', icon: '🏁',
    blurb: 'כל אתר הניסויים פתוח: הקיר, הלולאה, הצינור, קיר הרכיבה, רמפות ומסלול תאוצה.',
    howto: 'נסו הכול. מקשים 1–8 מעבירים ישר לכל תרחיש. כשנגמר האוויר, עצרו בעיגול הכחול של תחנת האוויר.',
    spawn: SPAWNS.free,
  },
  {
    key: 'wall', num: 2, nameHe: 'הקיר מההדגמה', icon: '🧗',
    blurb: 'הרמפה המעוקלת מהאירוע: עולה עד 170°, בגובה 21 מ\', הרכב תלוי הפוך, משתחרר ונוחת.',
    howto: 'סעו ישר אל הקיר ב-40–60 קמ"ש (W). המאווררים יעברו ל-Max Grip. בקצה הרכב יתנתק, יתהפך באוויר, ומנועי הפינות יבלמו את הנפילה.',
    spawn: SPAWNS.wall, fan: 'max',
    init: (s) => { s.maxPitch = 0; s.hAt = 0; s.dfMax = 0; s.released = false; s.landed = false; s.peakG = 0; s.fallG = 0; s.hFallG = 0; },
    update(car, s) {
      const p = car.pitchDeg, pp = p < -100 ? 360 + p : p;
      if (car.contacts > 0 && car.pos.z > LAYOUT.wall.z && pp > s.maxPitch && car.pos.y > 1) { s.maxPitch = pp; s.hAt = car.pos.y; }
      if (car.contacts > 0 && car.pos.y > 3) s.dfMax = Math.max(s.dfMax, car.fan.force / G0);
      if (s.maxPitch > 100 && car.contacts === 0 && car.pos.y > 4) s.released = true;
      if (s.released && car.contacts === 0) { const g = car.gSmooth.len(); if (g > s.fallG) { s.fallG = g; s.hFallG = car.altitude; } }
      if (s.released && !s.landed && car.contacts >= 3 && car.pos.y < 3) {
        s.landed = true; s.upright = car.up.y > 0.7;
        return { done: true, ok: s.upright && s.maxPitch > 150, html: this.result(s) };
      }
      return { live: `זווית ${pp.toFixed(0)}° · שיא ${s.maxPitch.toFixed(0)}° · גובה ${f1(car.pos.y)} מ' · יניקה ${(car.fan.force / G0).toFixed(0)} ק"ג` };
    },
    result: (s) => `
      <table><tr><th></th><th>אתם</th><th>ההדגמה</th></tr>
      <tr><td>זווית מרבית</td><td>${s.maxPitch.toFixed(0)}°</td><td>170°</td></tr>
      <tr><td>גובה בזווית הזו</td><td>${f1(s.hAt)} מ'</td><td>~21 מ'</td></tr>
      <tr><td>כוח יניקה מרבי</td><td>${s.dfMax.toFixed(0)} ק"ג</td><td>4,700 ק"ג (קריאה במסך)</td></tr>
      <tr><td>שיא G בבלימת הנפילה</td><td>${f1(s.fallG)}g בגובה ${f1(s.hFallG)} מ'</td><td>3.3g בגובה ~6.5 מ'</td></tr>
      <tr><td>נחיתה</td><td>${s.upright ? 'על הגלגלים ✓' : 'הפוך ✗'}</td><td>על הגלגלים, בענן אדים</td></tr></table>`,
  },
  {
    key: 'kickflip', num: 3, nameHe: 'קיקפליפ מעל המכוניות', icon: '🛹',
    blurb: 'קפיצה עם סיבוב שלם סביב ציר האורך מעל סייברקאב, סייברטראק, סמי, מודל Y ומודל 3.',
    howto: 'האיצו ל-60–70 קמ"ש ולחצו K על הקו הצהוב (קיקפליפ אוטומטי, כמו בהדגמה). או בעצמכם: רווח להמראה, ובאוויר A/D לסיבוב.',
    spawn: SPAWNS.kickflip, needs: 'lift',
    init: (s) => { s.touched = false; s.maxH = 0; s.roll = 0; s.over = false; },
    update(car, s) {
      const L = LAYOUT.lineup;
      const inZone = Math.abs(car.pos.x - L.x) < 5 && car.pos.z > L.z - 1 && car.pos.z < s.end + 1;
      if (inZone) { s.over = true; s.maxH = Math.max(s.maxH, car.pos.y); if (car.hullContact && car.pos.y > 0.7) s.touched = true; }
      if (car.contacts === 0) { const wb = car.q.unrotate({ x: car.w.x, y: car.w.y, z: car.w.z }); s.roll += wb.z * s.dt; }
      if (s.over && car.pos.z > s.end + 4 && car.contacts >= 3) {
        const ok = !s.touched && car.up.y > 0.7;
        return { done: true, ok, html: `
          <table><tr><th></th><th>אתם</th><th>ההדגמה</th></tr>
          <tr><td>גובה מרבי (מרכז הכובד)</td><td>${f1(s.maxH)} מ'</td><td>~7 מ'</td></tr>
          <tr><td>סיבוב באוויר</td><td>${Math.abs(s.roll * 180 / Math.PI).toFixed(0)}°</td><td>360°</td></tr>
          <tr><td>נגיעה ברכבים</td><td>${s.touched ? 'כן ✗' : 'לא ✓'}</td><td>לא</td></tr>
          <tr><td>אוויר שנשאר</td><td>${car.tank.m.toFixed(0)} ק"ג מתוך ${car.tank.mFull.toFixed(0)}</td><td>לא נמסר</td></tr></table>` };
      }
      return { live: `מהירות ${kmh(car.speed)} קמ"ש · מרחק לקו ${Math.max(0, L.z - 16 - car.pos.z).toFixed(0)} מ' · גובה ${f1(car.pos.y)} מ'` };
    },
  },
  {
    key: 'drag', num: 4, nameHe: 'מסלול תאוצה', icon: '⏱️',
    blurb: '0–100 קמ"ש, רבע מייל וחצי מייל. כמה מהר באמת אפשר כשהצמיגים מוגבלים ל-1.26g?',
    howto: 'החזיקו W, ובדגמי ת\'ראסט ורוקט גם Shift לדחף. המאווררים ב-Max Grip מוסיפים אחיזה בזינוק.',
    spawn: SPAWNS.drag, fan: 'max',
    init: (s) => { s.t = null; s.marks = {}; s.z0 = LAYOUT.drag.z0; s.peak = 0; s.air0 = null; },
    update(car, s) {
      const v = car.vel.z, d = car.pos.z - s.z0;
      if (s.t === null && v > 0.3) { s.t = 0; s.air0 = car.tank.m; }
      if (s.t !== null) {
        s.t += s.dt; s.peak = Math.max(s.peak, car.gSmooth.z);
        const M = s.marks;
        if (!M.m60 && v >= 26.8224) M.m60 = s.t, M.air60 = s.air0 - car.tank.m;
        if (!M.k100 && v >= 27.7778) M.k100 = s.t;
        if (!M.k200 && v >= 55.5556) M.k200 = s.t;
        if (!M.q && d >= 402.336) { M.q = s.t; M.qv = v; }
        if (!M.h && d >= 804.672) { M.h = s.t; M.hv = v;
          const claim = car.P.claimed060;
          return { done: true, ok: true, html: `
            <table><tr><th></th><th>אתם</th><th>טענת טסלה</th></tr>
            <tr><td>0–60 מייל/ש' (96.6 קמ"ש)</td><td>${M.m60 ? f2(M.m60) : '—'} ש'</td><td>פחות מ-${claim} ש'</td></tr>
            <tr><td>0–100 קמ"ש</td><td>${M.k100 ? f2(M.k100) : '—'} ש'</td><td></td></tr>
            <tr><td>0–200 קמ"ש</td><td>${M.k200 ? f2(M.k200) : '—'} ש'</td><td></td></tr>
            <tr><td>רבע מייל</td><td>${f2(M.q)} ש' ב-${kmh(M.qv)} קמ"ש</td><td rowspan=2>"חצי מייל בזמן שאחרים עושים רבע"</td></tr>
            <tr><td>חצי מייל</td><td>${f2(M.h)} ש' ב-${kmh(M.hv)} קמ"ש</td></tr>
            <tr><td>שיא תאוצה</td><td>${f1(s.peak)}g</td><td>3.4g ממוצע בהדגמה החיה</td></tr>
            <tr><td>אוויר עד 60</td><td>${(M.air60 || 0).toFixed(0)} ק"ג</td><td></td></tr></table>
            <p class="note">הזמנים נמדדים מהתזוזה הראשונה, בלי "רול-אאוט" של רגל (30 ס"מ) שיצרנים נוהגים להחסיר.</p>` };
        }
      }
      const M = s.marks;
      return { live: s.t === null ? 'מוכנים? W להתחלה' : `${f2(s.t)} ש' · ${kmh(v)} קמ"ש · ${d.toFixed(0)} מ'${M.m60 ? ` · 0–60 ב-${f2(M.m60)} ש'` : ''}${M.q ? ` · רבע מייל ב-${f2(M.q)} ש'` : ''}` };
    },
  },
  {
    key: 'tunnel', num: 5, nameHe: 'הצינור', icon: '🌀',
    blurb: 'צינור בקוטר 16 מ\'. המאווררים מצמידים את הרכב לדופן, אז אפשר לטפס עליה ולנסוע על התקרה.',
    howto: 'היכנסו ב-50 קמ"ש ויותר והיגו בעדינות, כך שתנסעו בספירלה לאורך הצינור. אם תפנו ישר אל הדופן, החרטום ייתקע: רכב באורך 4.35 מ\' לא יכול להתעקל ברדיוס 8 מ\'. היעד: 360° סביב הציר.',
    spawn: SPAWNS.tunnel, fan: 'max',
    init: (s) => { s.ang = 0; s.prev = null; s.max = 0; s.ceil = 0; },
    update(car, s) {
      const L = LAYOUT.tunnel;
      const inside = Math.abs(car.pos.x - L.x) < L.r && car.pos.z > L.z0 && car.pos.z < L.z1;
      if (inside) {
        const a = Math.atan2(car.pos.x - L.x, L.r - car.pos.y);
        if (s.prev !== null) { let d = a - s.prev; if (d > Math.PI) d -= 2 * Math.PI; if (d < -Math.PI) d += 2 * Math.PI; s.ang += d; }
        s.prev = a; s.max = Math.max(s.max, Math.abs(s.ang));
        if (car.up.y < -0.8 && car.contacts > 0) s.ceil += s.dt;
        if (s.max >= Math.PI * 2) return { done: true, ok: true, html: `<p>סיבוב שלם סביב ציר הצינור! זמן על התקרה: ${f1(s.ceil)} ש'.</p><p class="note">על התקרה המאווררים מושכים את הרכב למעלה ב-2g, והכובד מושך למטה ב-1g. נשאר 1g "למעלה", אז הצמיגים עדיין נלחצים לדופן.</p>` };
      } else s.prev = null;
      return { live: `זווית סביב הציר ${(s.ang * 180 / Math.PI).toFixed(0)}° · שיא ${(s.max * 180 / Math.PI).toFixed(0)}° · על התקרה ${f1(s.ceil)} ש'` };
    },
  },
  {
    key: 'loop', num: 6, nameHe: 'הלולאה', icon: '➰',
    blurb: 'לולאה בקוטר 17 מ\' עם כניסה ויציאה הדרגתיות (קלותואידה), כמו ברכבות הרים.',
    howto: 'סעו ישר ב-50–70 קמ"ש. ההיגוי האוטומטי שומר על מרכז המסלול (כי היציאה מוסטת הצידה). נסו גם לאט מאוד, רק עם היניקה!',
    spawn: SPAWNS.loop, fan: 'max',
    init: (s) => { s.minUp = 1; s.vTop = null; s.prog = 0; },
    update(car, s) {
      const g = loopGuide(car.pos, car.left);
      if (g && car.wheels.some(w => w.surf === 'loop')) { s.prog = Math.max(s.prog, g.s / g.total); if (car.up.y < s.minUp) { s.minUp = car.up.y; s.vTop = car.speed; s.gTop = car.gSmooth.y; } }
      if (s.prog > 0.95 && car.contacts >= 3 && car.pos.y < 1) {
        return { done: true, ok: s.minUp < -0.9, html: `<p>לולאה שלמה! מהירות בנקודה העליונה: ${kmh(s.vTop || 0)} קמ"ש.</p><p class="note">בלי יניקה, רכב צריך לפחות √(g·R) ≈ ${kmh(Math.sqrt(9.81 * 8.5))} קמ"ש בפסגה כדי לא ליפול. עם 2g של יניקה אפשר לעבור אותה כמעט בעמידה.</p>` };
      }
      return { live: `התקדמות ${(s.prog * 100).toFixed(0)}% · ${kmh(car.speed)} קמ"ש` };
    },
  },
  {
    key: 'wallride', num: 7, nameHe: 'נסיעה על קיר אנכי', icon: '🧱',
    blurb: 'קיר בגובה 20 מ\'. נוסעים עליו לאורכו, במקביל לקרקע, כשהגלגלים "דורכים" על קיר.',
    howto: 'סעו לאורך הקיר, היגו ימינה כדי לעלות עליו ושמרו על גובה מעל 5 מ\'. היעד: 150 מ\' על הקיר.',
    spawn: SPAWNS.wallride, fan: 'max',
    init: (s) => { s.dist = 0; },
    update(car, s) {
      if (car.pos.y > 5 && car.wheels.some(w => w.surf === 'wallride')) s.dist += car.speed * s.dt;
      if (s.dist >= 150) return { done: true, ok: true, html: `<p>150 מ' על קיר אנכי!</p><p class="note">על הקיר הכובד מושך את הרכב במקביל למשטח. הצמיגים מחזיקים אותו בחיכוך, וכוח החיכוך המרבי = μ × כוח הלחיצה. בלי מאווררים הלחיצה היא רק הכוח הצנטריפטלי, ועם מאווררים נוספים עוד ~4.5 טון.</p>` };
      return { live: `על הקיר: ${s.dist.toFixed(0)} / 150 מ' · גובה ${f1(car.pos.y)} מ'` };
    },
  },
  {
    key: 'hover', num: 8, nameHe: 'מבחן ריחוף', icon: '🛸',
    blurb: 'כמה זמן אפשר לרחף על 600 ליטר אוויר? המספרים מפתיעים.',
    howto: 'רווח מטפס (בקצב קבוע כשהייצוב פעיל), ו-J מחזיק גובה. עלו לגובה 5 מ\', לחצו J, וכוונו לטבעת הכתומה עם W/S ו-A/D (הטיית הרכב מזיזה אותו). הזמן נספר כשאתם בתוך הטבעת.',
    spawn: SPAWNS.hover, needs: 'lift',
    init: (s) => { s.inRing = 0; s.air0 = null; s.tAir = 0; },
    update(car, s) {
      const H = LAYOUT.hover;
      if (s.air0 === null) s.air0 = car.tank.m;
      if (car.contacts === 0) s.tAir += s.dt;
      const dh = Math.hypot(car.pos.x - H.x, car.pos.z - H.z);
      if (dh < 4 && car.pos.y > 3.5 && car.pos.y < 7) s.inRing += s.dt;
      if (car.tank.m < 3 && car.contacts >= 3) return { done: true, ok: s.inRing > 0.5, html: `<p>זמן בטבעת: <b>${f1(s.inRing)} ש'</b>. זמן באוויר: ${f1(s.tAir)} ש'.</p><p class="note">לריחוף צריך דחף ששווה למשקל, כ-22 קילוניוטון. בנחיר של אוויר דחוס כל קילוגרם אוויר נותן רק ~650 N·s, אז הריחוף "שורף" 30–35 ק"ג אוויר בשנייה. המכל כולו (227 ק"ג) מספיק לשניות ספורות, ועוד פחות מזה כי הלחץ יורד.</p>` };
      return { live: `בטבעת ${f1(s.inRing)} ש' · באוויר ${f1(s.tAir)} ש' · אוויר ${car.tank.m.toFixed(0)} ק"ג · ${((car.tank.p) / 1e5).toFixed(0)} בר` };
    },
  },
];

export function scenarioByKey(k) { return SCENARIOS.find(s => s.key === k) || SCENARIOS[0]; }
