import { redrawAll } from './plot.js';
import { initGeo } from './demos/geo.js';
import { initFence, initBead, initEntropy } from './demos/meaning.js';
import { initKKT } from './demos/kkt.js';
import { initDuality } from './demos/duality.js';
import { initLP } from './demos/lp.js';

for (const init of [initGeo, initFence, initBead, initEntropy, initKKT, initDuality, initLP]) {
  try { init(); } catch (e) { console.error(init.name, e); }
}

// theme toggle
const root = document.documentElement, btn = document.getElementById('theme');
const dark = () => root.dataset.theme ? root.dataset.theme === 'dark' : matchMedia('(prefers-color-scheme: dark)').matches;
const sync = () => { btn.textContent = dark() ? '☀' : '☾'; };
try { const saved = localStorage.getItem('lagrange-theme'); if (saved) root.dataset.theme = saved; } catch (e) { /* storage blocked */ }
sync();
btn.addEventListener('click', () => {
  root.dataset.theme = dark() ? 'light' : 'dark';
  try { localStorage.setItem('lagrange-theme', root.dataset.theme); } catch (e) { /* storage blocked */ }
  sync(); redrawAll();
});
matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => { sync(); redrawAll(); });

// highlight the current chapter in the table of contents
const links = [...document.querySelectorAll('nav.toc a')];
const io = new IntersectionObserver(entries => {
  for (const e of entries) if (e.isIntersecting) {
    links.forEach(a => a.classList.toggle('on', a.getAttribute('href') === '#' + e.target.id));
  }
}, { rootMargin: '-30% 0px -60% 0px' });
document.querySelectorAll('main section[id]').forEach(s => io.observe(s));
