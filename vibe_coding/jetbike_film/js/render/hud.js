// On-screen telemetry and titles (HTML overlay, captured together with the canvas).

import { LIFT_TMAX, MAIN_TMAX, G } from '../vehicle.js';
import { surfaceAt } from '../terrain.js';

const el = (tag, cls, parent, html = '') => { const e = document.createElement(tag); if (cls) e.className = cls; if (html) e.innerHTML = html; parent.appendChild(e); return e; };

const fx = (v, d) => (Math.abs(v) < 0.5 * 10 ** -d ? 0 : v).toFixed(d);
const ramp = (t, a, b, fi = 0.8, fo = 0.8) => Math.max(0, Math.min(1, (t - a) / fi, (b - t) / fo));

export function makeHud(root, marks, events, tEnd) {
  const hud = el('div', 'hud', root);
  const row = (label) => { const r = el('div', 'row', hud); el('span', 'lab', r, label); return el('span', 'val', r); };
  const spd = row('מהירות'), alt = row('גובה'), bank = row('הטיה'), pitch = row('עלרוד'), gl = row('עומס'), fuel = row('דלק');
  const bars = el('div', 'bars', hud);
  const names = ['שמ׳ קד׳', 'ימ׳ קד׳', 'שמ׳ אח׳', 'ימ׳ אח׳', 'אחורי'];
  const bar = names.map((n) => {
    const b = el('div', 'bar', bars);
    const tr = el('div', 'track', b);
    const fill = el('div', 'fill', tr);
    const vane = el('div', 'vane', b);
    el('div', 'bn', b, n);
    return { fill, vane };
  });
  el('div', 'bcap', hud, 'דחף מנועים · הסטת כנפונים');

  const title = el('div', 'title', root, `<div class="t1">MJ-5</div><div class="t2">אופנוע סילון · טיסת ניסוי</div>
    <div class="t3">כל תנועה בסרט מחושבת בסימולציה פיזיקלית: גוף קשיח בשש דרגות חופש, חמישה מנועי סילון עם השהיית סחרור, כנפוני הסטה, גרר, רוח ומערבולות</div>`);
  const cap = el('div', 'caption', root);
  const end = el('div', 'endcard', root, `<div class="t2">MJ-5</div><div class="t3">4 מנועי עילוי × 1,300 ניוטון · מנוע שיוט 1,600 ניוטון · 325 ק״ג בהמראה<br>
    בקר טיסה ב-200 הרץ · פיזיקה ב-2,000 הרץ</div>`);

  const captions = [
    [1.0, 6.2, 'הנעה: ארבעת מנועי העילוי ניצתים זה אחר זה'],
    [8.0, events.liftoff + 0.3, 'הגברת דחף: הטורבינות מאיצות עד שהדחף עולה על המשקל'],
    [events.liftoff + 0.5, events.liftoff + 5.5, 'המראה: הגזים הנפלטים כלפי מטה מעיפים את חצץ הרחבה'],
    [19.5, 25.5, 'מנוע השיוט דוחף קדימה, מנועי העילוי נושאים את כל המשקל'],
    [marks.tLake - 2.5, marks.tLake + 3.5, 'חמישה מטרים מעל האגם: הסילון מרסס את המים'],
    [marks.tHill + 7, marks.tHill + 13, 'פנייה בהטיה: כמו כל כלי טיס, מטים את וקטור הדחף'],
    [marks.tBack + 12, marks.tBack + 18, 'בלימה: האף מתרומם והדחף של מנועי העילוי בולם את התנועה'],
    [events.touchdown - 1.2, events.touchdown + 3.5, 'נגיעה בקרקע, הורדת דחף וכיבוי'],
  ];
  let last = '';
  return {
    update(s, t, show = true) {
      hud.style.opacity = show ? ramp(t, 7, tEnd - 3) : 0;
      const v = Math.hypot(...s.v) * 3.6;
      spd.textContent = `${v.toFixed(0)} קמ״ש`;
      alt.textContent = `${fx(Math.max(0, s.p[1] - 0.86 - surfaceAt(s.p[0], s.p[2])), 1)} מ׳`;
      const [w, x, y, z] = s.q;
      // body axes in world
      const fy = 2 * (x * y + w * z);                    // forward.y
      const rzY = 2 * (y * z - w * x);                   // right.y
      bank.textContent = `${fx(-Math.asin(Math.max(-1, Math.min(1, rzY))) * 57.3, 0)}°`;
      pitch.textContent = `${fx(Math.asin(Math.max(-1, Math.min(1, fy))) * 57.3, 0)}°`;
      const ay = s.acc[1] + G, load = Math.hypot(s.acc[0], ay, s.acc[2]) / G;
      gl.textContent = `${load.toFixed(2)} g`;
      fuel.textContent = `${s.fuel.toFixed(1)} ק״ג`;
      for (let i = 0; i < 5; i++) {
        const T = i < 4 ? s.T[i] : s.mainT;
        const tm = (i < 4 ? LIFT_TMAX : MAIN_TMAX) * s.sigma;
        bar[i].fill.style.height = `${Math.min(100, (T / tm) * 100).toFixed(1)}%`;
        if (i < 4) {
          const a = s.vanes[i][0] * 57.3, b = s.vanes[i][1] * 57.3;
          bar[i].vane.style.transform = `translate(${(b * 0.8).toFixed(1)}px, ${(-a * 0.8).toFixed(1)}px)`;
        } else bar[i].vane.style.opacity = 0;
      }
      title.style.opacity = ramp(t, 0.8, 5.8, 1.2, 1.2);
      end.style.opacity = ramp(t, tEnd - 5.5, tEnd - 0.6, 1.2, 1.0);
      let c = '', op = 0;
      for (const [a, b, txt] of captions) if (t >= a && t < b) { c = txt; op = ramp(t, a, b, 0.6, 0.6); }
      if (c && c !== last) { cap.textContent = c; last = c; }
      cap.style.opacity = op;
    },
  };
}
