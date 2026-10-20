// האקדמיה: תוכן עניינים חי, טעינה עצלה של כל ווידג'ט כשהוא מתקרב למסך, מונחים וטבלת המקורות.
import * as B from './w_body.js';
import * as A from './w_act.js';
import * as H from './w_hand.js';
import * as L from './w_balance.js';
import * as M from './w_misc.js';
import { TERMS } from './terms.js';
import { BODY, JOINTS, MASS, G, ACTUATOR_TYPES } from '../robot/spec.js';

const WIDGETS = {
  'w-anatomy': B.anatomy, 'w-lab': B.lab,
  'w-gear': A.gear, 'w-screw': A.screw, 'w-ts': A.torqueSpeed,
  'w-tendon': H.tendon, 'w-grip': H.grip,
  'w-pendulum': L.pendulum, 'w-lipm': L.lipm, 'w-imu': L.imu,
  'w-rl': M.rl, 'w-impact': M.impact, 'w-timeline': M.timeline,
};
const io = new IntersectionObserver((entries) => {
  for (const e of entries) {
    if (!e.isIntersecting) continue;
    io.unobserve(e.target);
    const fn = WIDGETS[e.target.id];
    Promise.resolve().then(() => fn?.(e.target)).catch((err) => { console.error(e.target.id, err); e.target.insertAdjacentHTML('beforeend', `<p class="note">שגיאה בטעינת ההדמיה: ${err.message}</p>`); });
  }
}, { rootMargin: '400px' });
for (const id of Object.keys(WIDGETS)) { const el = document.getElementById(id); if (el) io.observe(el); }

M.cotBars(document.getElementById('cot-bars'));
M.runtimeCalc(document.getElementById('runtime-calc'));

// ---------- מונחים ----------
const pop = document.getElementById('term-pop');
document.querySelectorAll('.term').forEach((el) => {
  const t = TERMS[el.dataset.t]; if (!t) return;
  const show = () => {
    pop.innerHTML = `<span class="en">${t[0]}</span>${t[1]}`;
    pop.classList.remove('hidden');
    const r = el.getBoundingClientRect();
    const pw = Math.min(340, innerWidth - 20);
    const left = Math.max(10, Math.min(innerWidth - pw - 10, r.left + r.width / 2 - pw / 2));
    pop.style.left = left + 'px'; pop.style.top = (r.bottom + 8) + 'px';
    if (r.bottom + 8 + pop.offsetHeight > innerHeight) pop.style.top = (r.top - pop.offsetHeight - 8) + 'px';
  };
  el.addEventListener('mouseenter', show); el.addEventListener('click', show);
  el.addEventListener('mouseleave', () => pop.classList.add('hidden'));
});
addEventListener('scroll', () => pop.classList.add('hidden'), { passive: true });

// ---------- טבלת המקורות, ישר מתוך spec.js ----------
{
  const SRC = { tesla: ['tesla', 'טסלה'], report: ['report', 'דיווח'], est: ['est', 'הערכה'] };
  const tag = (s) => `<span class="src ${SRC[s][0]}">${SRC[s][1]}</span>`;
  const rows = [
    ['גובה', `${BODY.height.v} מ'`, BODY.height.src, BODY.height.note],
    ['מסה', `${BODY.mass.v} ק"ג`, BODY.mass.src, BODY.mass.note + '. במודל: 57.0 ק"ג, מחולק לפי ההערכה שלנו'],
    ['נשיאה', `${BODY.payload.v} ק"ג`, BODY.payload.src, BODY.payload.note],
    ['סוללה', `${BODY.battery.v} קוט"ש, ${BODY.busVoltage.v} וולט`, BODY.battery.src, BODY.battery.note],
    ['הספק', `~${BODY.powerIdle.v} / ~${BODY.powerWalk.v} ואט`, BODY.powerIdle.src, 'בישיבה / בהליכה מהירה. בסימולטור: 97 / 250–350, ממודל הפסדים שלנו'],
    ['מפעילים בגוף', `${BODY.actuators.v}`, BODY.actuators.src, BODY.actuators.note],
    ['סוגי מפעילים', Object.values(ACTUATOR_TYPES).filter(a => a.src === 'tesla').map(a => a.label).join(', '), 'tesla', 'AI Day 2022. ההקצאה של כל סוג למפרק היא הערכה שלנו'],
    ['מפעילים ביד ובאמה', `${BODY.handActuators.v}`, BODY.handActuators.src, BODY.handActuators.note],
    ['דרגות חופש ביד', `${BODY.handDof.v}`, BODY.handDof.src, BODY.handDof.note],
    ['חלקים', `~${BODY.parts.v.toLocaleString('en-US')}`, BODY.parts.src, BODY.parts.note],
    ['מראה', 'זהב שמפניה, שחור', 'report', 'הדמיות מאפליקציית טסלה, ספטמבר 2026. לא אושר רשמית'],
    ['אורכי איברים', `ירך ${G.thigh}, שוק ${G.shin}, זרוע ${G.upperArm}, אמה ${G.forearm} מ'`, 'est', 'פרופורציות אנושיות לגובה 173 ס"מ, מותאמות לתמונות'],
    ['מסות איברים', `גו ${MASS.torso}, ירך ${MASS.thigh}, שוק ${MASS.shin}, אמה ${MASS.forearm} ק"ג`, 'est', 'האמה כבדה יחסית בגלל מנועי האצבעות'],
    ['מומנטים במפרקים', 'ראו טבלה בפרק 1', 'est', 'מתוך דירוגי המפעילים וזרועות מנוף סבירות'],
    ['אינרציית רוטור מוחזרת', '0.003–0.18 ק"ג·מ"ר', 'est', 'armature ב-MuJoCo. משפיעה מאוד על היציבות'],
    ['חיכוך במפרקים', '0.02–0.5 נ"מ', 'est', 'frictionloss ב-MuJoCo, עם פיצוי בבקר'],
    ['כוח אצבע', '~22 ניוטון בקצה', 'est', 'מפעילי מיקום עם מגבלת כוח'],
    ['מודל ההספק', 'בסיס 92 ואט, נצילות 82%, הפסדי נחושת', 'est', 'כויל כך שעמידה ≈ 100 ואט כמו ב-AI Day'],
  ];
  document.getElementById('src-table').innerHTML = '<tr><th>מה</th><th>ערך</th><th>מקור</th><th>הערה</th></tr>' + rows.map(r => `<tr><td>${r[0]}</td><td>${r[1]}</td><td>${tag(r[2])}</td><td>${r[3] || ''}</td></tr>`).join('');
}

// ---------- תוכן עניינים ----------
const links = [...document.querySelectorAll('#toc a[href^="#"]')];
const spy = new IntersectionObserver((entries) => {
  for (const e of entries) if (e.isIntersecting) links.forEach(a => a.classList.toggle('on', a.getAttribute('href') === '#' + e.target.id));
}, { rootMargin: '-40% 0px -55% 0px' });
document.querySelectorAll('main section').forEach(s => spy.observe(s));
const toc = document.getElementById('toc');
document.getElementById('toc-btn').onclick = () => toc.classList.toggle('open');
links.forEach(a => a.addEventListener('click', () => toc.classList.remove('open')));
