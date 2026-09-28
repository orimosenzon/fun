// Game HUD (Hebrew): telemetry panel in the film's style, race info, next-ring marker,
// engine temperature for emergency power, messages and the start / finish screens.

import { LIFT_TMAX, MAIN_TMAX, G } from './vehicle.js';

const el = (tag, cls, parent, html = '') => { const e = document.createElement(tag); if (cls) e.className = cls; if (html) e.innerHTML = html; parent.appendChild(e); return e; };
const fx = (v, d) => (Math.abs(v) < 0.5 * 10 ** -d ? 0 : v).toFixed(d);
export const fmtTime = (s) => `${Math.floor(s / 60)}:${(s % 60).toFixed(2).padStart(5, '0')}`;

export function makeHud(root) {
  const panel = el('div', 'hud', root);
  const row = (label) => { const r = el('div', 'row', panel); el('span', 'lab', r, label); return el('span', 'val', r); };
  const spd = row('מהירות'), alt = row('גובה'), vs = row('אנכי'), bank = row('הטיה'), gl = row('עומס'), fuel = row('דלק');
  const bars = el('div', 'bars', panel);
  const names = ['שמ׳ קד׳', 'ימ׳ קד׳', 'שמ׳ אח׳', 'ימ׳ אח׳', 'שיוט'];
  const bar = names.map((n) => {
    const b = el('div', 'bar', bars);
    const tr = el('div', 'track', b);
    const fill = el('div', 'fill', tr);
    const cmd = el('div', 'cmd', tr);
    const vane = el('div', 'vane', b);
    el('div', 'bn', b, n);
    return { fill, cmd, vane };
  });
  const heatRow = el('div', 'heat', panel, '<span>טמפ׳ מנוע</span><div class="htrack"><div class="hfill"></div></div>');
  const heatFill = heatRow.querySelector('.hfill');
  const mode = el('div', 'mode', panel);

  const race = el('div', 'race', root);
  const rings = el('div', 'rings', race), time = el('div', 'time', race), best = el('div', 'best', race);
  const marker = el('div', 'marker', root, '<div class="mk"></div><div class="md"></div>');
  const mDist = marker.querySelector('.md');
  const msg = el('div', 'msg', root);
  const warn = el('div', 'warn', root);
  let msgUntil = 0;

  const screen = el('div', 'screen', root);
  return {
    screen,
    message(text, ms = 1600, cls = '') { msg.innerHTML = text; msg.className = `msg show ${cls}`; msgUntil = performance.now() + ms; },
    warning(text) { warn.textContent = text; warn.style.opacity = text ? 1 : 0; },
    showScreen(html) { screen.innerHTML = html; screen.style.display = html ? 'flex' : 'none'; },
    update(s) {
      const b = s.bike;
      spd.textContent = `${fx(Math.hypot(...b.v) * 3.6, 0)} קמ״ש`;
      alt.textContent = `${fx(Math.max(0, s.agl), 1)} מ׳`;
      vs.textContent = `${fx(b.v[1], 1)} מ׳/ש`;
      bank.textContent = `${fx(s.bankDeg, 0)}°`;
      const load = Math.hypot(b.acc[0], b.acc[1] + G, b.acc[2]) / G;
      gl.textContent = `${load.toFixed(2)} g`;
      fuel.textContent = `${b.fuel.toFixed(1)} ק״ג`;
      fuel.style.color = b.fuel < 8 ? '#ff6b4a' : '';
      for (let i = 0; i < 5; i++) {
        const e = i < 4 ? b.lift[i] : b.main;
        const tm = (i < 4 ? LIFT_TMAX : MAIN_TMAX) * (b.sigma || 0.9);
        bar[i].fill.style.height = `${Math.min(100, (e.T / tm) * 100).toFixed(1)}%`;
        const c = i < 4 ? s.cmd.lift[i] : s.cmd.main;
        bar[i].cmd.style.bottom = `${Math.min(100, (c / tm) * 100).toFixed(1)}%`;
        if (i < 4) {
          const [a, bb] = b.vanes[i];
          bar[i].vane.style.transform = `translate(${(bb * 57.3 * 0.8).toFixed(1)}px, ${(-a * 57.3 * 0.8).toFixed(1)}px)`;
        } else bar[i].vane.style.opacity = 0;
      }
      heatFill.style.width = `${(s.heat * 100).toFixed(0)}%`;
      heatFill.style.background = s.heat > 0.8 ? '#ff4a3a' : s.heat > 0.5 ? '#ffb347' : '#7fd8ff';
      mode.textContent = `${s.assist ? (s.radar ? 'עזר · גובה מעל הקרקע' : 'בקר טיסה: עזר') : 'בקר טיסה: ידני'} · ${s.camName}`;
      rings.textContent = s.ringIndex < s.ringCount ? `טבעת ${s.ringIndex + 1}/${s.ringCount}` : s.finished ? 'סיום' : 'נחיתה במטרה';
      time.textContent = fmtTime(s.raceTime);
      best.textContent = s.best ? `שיא ${fmtTime(s.best)}` : '';
      // next target marker
      if (s.marker) {
        marker.style.display = 'block';
        marker.style.transform = `translate(${s.marker.x}px, ${s.marker.y}px)`;
        marker.classList.toggle('edge', s.marker.edge);
        marker.querySelector('.mk').style.transform = `rotate(${s.marker.rot}rad)`;
        mDist.textContent = `${s.marker.dist.toFixed(0)} מ׳`;
      } else marker.style.display = 'none';
      if (performance.now() > msgUntil) msg.className = 'msg';
    },
    setVisible(v) { panel.style.display = race.style.display = v ? '' : 'none'; if (!v) marker.style.display = 'none'; },
  };
}
