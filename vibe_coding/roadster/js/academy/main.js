// המרכז המקצועי: תוכן עניינים חי, טעינה עצלה של כל ווידג'ט כשהוא מתקרב למסך.
import * as A from './w_a.js';
import * as B from './w_b.js';
import * as C from './w_c.js';

const WIDGETS = {
  'w-anatomy': A.anatomy, 'w-tires': A.tires, 'w-fan': A.fan, 'w-wall': A.wall,
  'w-nozzle': B.nozzle, 'w-tank': B.tank, 'w-flip': B.flip, 'w-land': B.land,
  'w-budget': C.budget, 'w-body': C.body, 'w-gbars': C.gbars, 'w-motor': C.motor, 'w-aero': C.aero, 'w-timeline': C.timeline,
};

const io = new IntersectionObserver((entries) => {
  for (const e of entries) {
    if (!e.isIntersecting) continue;
    io.unobserve(e.target);
    const fn = WIDGETS[e.target.id];
    try { fn?.(e.target); } catch (err) { console.error(e.target.id, err); e.target.insertAdjacentHTML('beforeend', `<p class="note">שגיאה בטעינת ההדמיה: ${err.message}</p>`); }
  }
}, { rootMargin: '400px' });
for (const id of Object.keys(WIDGETS)) { const el = document.getElementById(id); if (el) io.observe(el); }

B.ispBars(document.getElementById('isp-bars'));
B.copv(document.getElementById('w-copv'));
C.hoverCalc(document.getElementById('hover-calc'));
C.terms();

// תוכן עניינים: הדגשת הפרק הנוכחי
const links = [...document.querySelectorAll('#toc a[href^="#"]')];
const spy = new IntersectionObserver((entries) => {
  for (const e of entries) if (e.isIntersecting) {
    links.forEach(a => a.classList.toggle('on', a.getAttribute('href') === '#' + e.target.id));
  }
}, { rootMargin: '-40% 0px -55% 0px' });
document.querySelectorAll('main section').forEach(s => spy.observe(s));
const toc = document.getElementById('toc');
document.getElementById('toc-btn').onclick = () => toc.classList.toggle('open');
links.forEach(a => a.addEventListener('click', () => toc.classList.remove('open')));
