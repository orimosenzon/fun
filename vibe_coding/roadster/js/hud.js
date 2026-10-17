// תצוגת הנתונים: מד G, תחושת הגוף, מאווררים, מכל האוויר, מפת נחירים, מפה קטנה, טלמטריה ומסך הרכב.
import { G0 } from './math.js';
import { FAN_MODES, AERO } from './params.js';
import { AIR, PSI } from './gas.js';
import { LAYOUT } from './track.js';

const $ = (id) => document.getElementById(id);
const fmt = (x, d = 0) => x.toLocaleString('en-US', { minimumFractionDigits: d, maximumFractionDigits: d });

export class HUD {
  constructor() {
    this.el = {};
    for (const id of ['hud-variant', 'hud-scn', 'hud-cam', 'banner-status', 'banner-live', 'g-total', 'feel', 'head-w', 'alt', 'pitch',
      'fan-bar', 'fan-mode', 'fan-dp', 'fan-force', 'fan-gap', 'fan-pw', 'tank-fill', 'tank-p', 'tank-psi', 'tank-m', 'tank-t', 'thrust',
      'spd', 'pow-bar', 'pow', 'aero-mode', 'assist', 'tc', 'air-block', 'nozmap', 'toast']) this.el[id] = $(id);
    this.gm = $('gmeter').getContext('2d');
    this.mm = $('minimap').getContext('2d');
    this.tele = $('tele-c').getContext('2d');
    this.hist = [];
    this.trail = [];
    this.t = 0; this.slow = 0;
    this.toastT = 0;
  }

  setVariant(car) {
    this.el['hud-variant'].textContent = car.P.nameHe;
    this.el['air-block'].style.display = car.thr.length ? '' : 'none';
    // מפת נחירים במבט על
    const svg = this.el.nozmap;
    let s = `<path d="M -38 -88 Q 0 -100 38 -88 L 42 70 Q 0 96 -42 70 Z" fill="rgba(255,255,255,0.06)" stroke="rgba(255,255,255,0.3)"/>`;
    s += `<text x="0" y="-60" fill="#9aa5b1" font-size="11" text-anchor="middle">קדימה</text>`;
    this.noz = [];
    car.thr.forEach((t, i) => {
      const x = -t.pos[0] * 42, y = -t.pos[2] * 40; // מבט מלמעלה: x שמאלה בגוף ⇒ ימין במסך הפוך
      const r = t.group === 'main' ? 7 : t.group === 'lift' ? 8 : 5;
      const col = t.group === 'main' ? '#ff8a65' : t.group === 'lift' ? '#7cc8ff' : '#c5e1a5';
      s += `<circle id="nz${i}" cx="${x}" cy="${y}" r="${r}" fill="${col}" fill-opacity="0.12" stroke="${col}" stroke-width="1.5"/>`;
      if (t.group === 'yaw') s += `<line x1="${x}" y1="${y}" x2="${x - Math.sign(-t.dir[0]) * 14}" y2="${y}" stroke="${col}" stroke-width="1"/>`;
    });
    s += `<text x="0" y="95" fill="#9aa5b1" font-size="10" text-anchor="middle">${car.thr.length} נחירים</text>`;
    svg.innerHTML = s;
    this.noz = car.thr.map((_, i) => svg.querySelector('#nz' + i));
  }

  toast(msg, ms = 1800) {
    const t = this.el.toast; t.textContent = msg; t.classList.add('on');
    clearTimeout(this._tt); this._tt = setTimeout(() => t.classList.remove('on'), ms);
  }

  update(car, ctl, cam, live, dt) {
    this.t += dt; this.slow += dt;
    const E = this.el;
    // מהיר: כל פריים
    E.spd.textContent = Math.round(car.speed * 3.6);
    const st = ctl.status;
    E['banner-status'].textContent = st; E['banner-status'].classList.toggle('on', !!st);
    for (let i = 0; i < this.noz.length; i++) {
      const u = car.thr[i].u;
      this.noz[i].setAttribute('fill-opacity', (0.12 + 0.88 * Math.min(1, u)).toFixed(2));
    }
    this.drawG(car);
    // היסטוריה לגרף
    this.hist.push({ t: this.t, v: car.speed * 3.6, g: car.gSmooth.len(), df: car.fan.force / G0, p: car.tank.p / 1e5, th: car.thrustTotal / 1000, h: car.pos.y });
    while (this.hist.length && this.hist[0].t < this.t - 12) this.hist.shift();
    if (this.slow < 0.1) return;
    this.slow = 0;

    // איטי: 10 פעמים בשנייה
    const g = car.gSmooth, gt = g.len();
    E['g-total'].textContent = gt.toFixed(1);
    E['head-w'].textContent = `${(5 * gt).toFixed(1)} ק"ג`;
    E.feel.textContent = feeling(car);
    E.alt.textContent = `${Math.max(0, car.altitude).toFixed(1)} מ'`;
    E.pitch.textContent = `${car.pitchDeg.toFixed(0)}°`;
    E['fan-bar'].style.width = `${car.fan.n * 100}%`;
    E['fan-mode'].textContent = FAN_MODES[ctl.fanMode].nameHe;
    E['fan-dp'].textContent = `${(car.fan.dp / 1000).toFixed(1)} kPa`;
    E['fan-force'].textContent = `${fmt(car.fan.force / G0)} ק"ג`;
    const gapAvg = car.fan.gaps.reduce((a, b) => a + b, 0) / car.fan.gaps.length;
    E['fan-gap'].textContent = gapAvg > 0.3 ? 'פתוח' : `${(gapAvg * 1000).toFixed(0)} מ"מ`;
    E['fan-pw'].textContent = `${(car.fan.power / 1000).toFixed(0)} kW${car.fan.powerLimited ? ' ⚠' : ''}`;
    if (car.thr.length) {
      const tk = car.tank;
      E['tank-fill'].style.height = `${Math.max(0, Math.min(1, tk.p / car.P.fillPressure)) * 100}%`;
      E['tank-p'].textContent = `${(tk.p / 1e5).toFixed(0)} בר`;
      E['tank-psi'].textContent = `${fmt(tk.p / PSI)} psi`;
      E['tank-m'].textContent = `${tk.m.toFixed(0)} ק"ג`;
      const tc = tk.T - 273.15;
      E['tank-t'].textContent = `${tc.toFixed(0)}°C`;
      E['tank-t'].style.color = tc < -30 ? '#7cc8ff' : tc > 45 ? '#ffb300' : '';
      E.thrust.textContent = `${(car.thrustTotal / 1000).toFixed(0)} kN`;
    }
    E['pow-bar'].style.width = `${Math.min(100, car.motorPower / car.P.motorPower * 100)}%`;
    E.pow.textContent = `${fmt(car.motorPower / 1000)} kW`;
    E['aero-mode'].textContent = `כנף: ${AERO.modes[ctl.aeroMode].nameHe} (V)`;
    E.assist.textContent = `ייצוב: ${ctl.assist ? 'פעיל' : 'כבוי'} (L)`; E.assist.classList.toggle('on', ctl.assist);
    E.tc.textContent = `בקרת משיכה: ${ctl.tc ? 'פעילה' : 'כבויה'} (T)`; E.tc.classList.toggle('on', ctl.tc);
    E['hud-cam'].textContent = `מצלמה: ${{ chase: 'עוקבת', cockpit: 'מתוך הרכב', orbit: 'חופשית' }[cam.mode]} (C)`;
    E['banner-live'].textContent = live || '';
    this.drawMinimap(car);
    if (!document.getElementById('tele').classList.contains('hidden')) this.drawTele();
  }

  drawG(car) {
    const c = this.gm, W = 190, R = 80, cx = W / 2, cy = W / 2;
    c.clearRect(0, 0, W, W);
    c.strokeStyle = 'rgba(255,255,255,0.15)'; c.lineWidth = 1;
    for (const r of [1, 2, 3]) { c.beginPath(); c.arc(cx, cy, R * r / 3, 0, Math.PI * 2); c.stroke(); }
    c.beginPath(); c.moveTo(cx - R, cy); c.lineTo(cx + R, cy); c.moveTo(cx, cy - R); c.lineTo(cx, cy + R); c.stroke();
    c.fillStyle = 'rgba(255,255,255,0.4)'; c.font = '10px Rubik'; c.textAlign = 'center';
    c.fillText('1g', cx + R / 3 + 8, cy - 3); c.fillText('2g', cx + 2 * R / 3 + 8, cy - 3); c.fillText('3g', cx + R + 0, cy - 3);
    c.fillText('האצה', cx, cy + R + 10); c.fillText('בלימה', cx, cy - R - 3);
    // הנוסע מרגיש את הכוח הספציפי: האצה קדימה = נלחץ אחורה (מוצג למטה)
    const g = car.gSmooth;
    const sx = Math.max(-3.3, Math.min(3.3, g.x)), sz = Math.max(-3.3, Math.min(3.3, g.z));
    const px = cx + (sx) * R / 3, py = cy + sz * R / 3;
    // שובל
    this.trail.push([px, py]); if (this.trail.length > 40) this.trail.shift();
    c.strokeStyle = 'rgba(229,57,53,0.35)'; c.lineWidth = 2; c.beginPath();
    this.trail.forEach(([x, y], i) => (i ? c.lineTo(x, y) : c.moveTo(x, y))); c.stroke();
    c.fillStyle = '#e53935'; c.beginPath(); c.arc(px, py, 7, 0, Math.PI * 2); c.fill();
    // אנכי: פס בצד
    const gy = g.y;
    const h = Math.max(-1, Math.min(4, gy)) / 4 * R;
    c.fillStyle = gy < 0 ? '#ffb300' : gy < 0.3 ? '#7cc8ff' : '#4caf50';
    c.fillRect(W - 10, cy - Math.max(0, h), 6, Math.abs(h));
    c.fillStyle = 'rgba(255,255,255,0.5)'; c.fillText(`${gy.toFixed(1)}`, W - 8, cy + R + 10);
  }

  drawMinimap(car) {
    const c = this.mm, W = 170, s = W / 1100, cx = W / 2, cy = W / 2;
    const P = (x, z) => [cx - x * s, cy - z * s]; // צפון = +z למעלה, x שמאלה
    c.clearRect(0, 0, W, W);
    c.fillStyle = 'rgba(185,182,174,0.35)';
    const rect = (x0, x1, z0, z1) => { const [a, b] = P(x1, z1), [d, e] = P(x0, z0); c.fillRect(a, b, d - a, e - b); };
    rect(-150, 95, -230, 290); rect(-460, -385, -340, 250); rect(225, 295, -230, 60);
    c.fillStyle = 'rgba(60,60,60,0.8)'; rect(LAYOUT.drag.x - 11, LAYOUT.drag.x + 11, -330, 660); rect(LAYOUT.tunnel.x - 9, LAYOUT.tunnel.x + 9, -230, 210);
    const dot = (x, z, col, r = 3) => { const [a, b] = P(x, z); c.fillStyle = col; c.beginPath(); c.arc(a, b, r, 0, Math.PI * 2); c.fill(); };
    dot(LAYOUT.wall.x, LAYOUT.wall.z + 8, '#e53935', 4);
    dot(LAYOUT.loop.x, LAYOUT.loop.z + 12, '#1e88e5', 4);
    dot(LAYOUT.lineup.x, LAYOUT.lineup.z + 8, '#cfd8dc', 4);
    dot(LAYOUT.station.x, LAYOUT.station.z, '#29b6f6', 3);
    dot(LAYOUT.hover.x, LAYOUT.hover.z, '#ff7043', 3);
    c.strokeStyle = '#ffb300'; c.lineWidth = 3; { const [a, b] = P(LAYOUT.tunnel.x, LAYOUT.tunnel.z0), [d, e] = P(LAYOUT.tunnel.x, LAYOUT.tunnel.z1); c.beginPath(); c.moveTo(a, b); c.lineTo(d, e); c.stroke(); }
    c.strokeStyle = '#2e7d32'; { const [a, b] = P(LAYOUT.wallRide.x, LAYOUT.wallRide.z0), [d, e] = P(LAYOUT.wallRide.x, LAYOUT.wallRide.z1); c.beginPath(); c.moveTo(a, b); c.lineTo(d, e); c.stroke(); }
    // הרכב
    const [x, y] = P(car.pos.x, car.pos.z);
    const yaw = Math.atan2(car.fwd.x, car.fwd.z);
    c.save(); c.translate(x, y); c.rotate(-yaw); // במפה x שמאלה, וקנבס מסובב עם כיוון השעון
    c.fillStyle = '#fff'; c.beginPath(); c.moveTo(0, -7); c.lineTo(4.5, 5); c.lineTo(-4.5, 5); c.closePath(); c.fill();
    c.restore();
  }

  drawTele() {
    const c = this.tele, W = 420, H = 230;
    c.clearRect(0, 0, W, H);
    const series = [
      { k: 'v', name: 'מהירות קמ"ש', max: 400, col: '#fff' },
      { k: 'g', name: 'G', max: 6, col: '#e53935' },
      { k: 'df', name: 'יניקה ק"ג', max: 5000, col: '#3ec2a0' },
      { k: 'p', name: 'לחץ מכל בר', max: 400, col: '#7cc8ff' },
      { k: 'th', name: 'דחף kN', max: 180, col: '#ff8a65' },
      { k: 'h', name: 'גובה מ\'', max: 25, col: '#ffd54f' },
    ];
    const h = this.hist; if (h.length < 2) return;
    const t0 = this.t - 12;
    c.font = '11px Rubik'; c.textAlign = 'right';
    series.forEach((s, i) => {
      c.strokeStyle = s.col; c.lineWidth = 1.6; c.beginPath();
      h.forEach((p, j) => { const x = (p.t - t0) / 12 * (W - 10) + 5, y = H - 8 - Math.min(1, p[s.k] / s.max) * (H - 40); j ? c.lineTo(x, y) : c.moveTo(x, y); });
      c.stroke();
      c.fillStyle = s.col; c.fillText(`${s.name}: ${fmt(h[h.length - 1][s.k], s.k === 'g' ? 1 : 0)}`, W - 6 - (i % 3) * 138, 14 + Math.floor(i / 3) * 14);
    });
  }

  // המסך המרכזי ברכב: "מצב מסלול", כמו בתמונות הרשמיות
  drawScreen(model, car, ctl) {
    const cv = model.screenCanvas, c = cv.getContext('2d'), W = cv.width, H = cv.height;
    c.fillStyle = '#05070a'; c.fillRect(0, 0, W, H);
    c.fillStyle = '#e53935'; c.font = 'bold 22px Rubik'; c.textAlign = 'left'; c.fillText('TRACK MODE', 20, 34);
    c.fillStyle = '#fff'; c.font = 'bold 92px JetBrains Mono, monospace'; c.fillText(Math.round(car.speed * 2.23694), 20, 140);
    c.font = '20px Rubik'; c.fillStyle = '#9aa5b1'; c.fillText('MPH', 26, 168);
    const rows = [
      ['G', car.gSmooth.len().toFixed(2)],
      ['DOWNFORCE', `${fmt(car.fan.force / G0 + car.aeroDown / G0)} kg`],
      ['THRUSTER', car.thr.length ? `${fmt(car.tank.p / PSI)} psi` : '—'],
      ['FAN', `${Math.round(car.fan.n * 100)}%`],
    ];
    c.font = '20px JetBrains Mono, monospace';
    rows.forEach(([k, v], i) => { c.fillStyle = '#9aa5b1'; c.fillText(k, 270, 60 + i * 46); c.fillStyle = '#fff'; c.fillText(v, 270, 84 + i * 46); });
    // מד G קטן
    c.strokeStyle = '#333'; c.beginPath(); c.arc(120, 240, 40, 0, Math.PI * 2); c.stroke();
    c.fillStyle = '#e53935'; c.beginPath(); c.arc(120 + car.gSmooth.x * 13, 240 + car.gSmooth.z * 13, 6, 0, Math.PI * 2); c.fill();
    if (ctl.status) { c.fillStyle = '#7cc8ff'; c.font = 'bold 20px Rubik'; c.fillText(ctl.status, 200, 270); }
    model.screenTex.needsUpdate = true;
  }
}

// תיאור מילולי של מה שהגוף מרגיש
export function feeling(car) {
  const g = car.gSmooth, tot = g.len();
  const air = car.contacts === 0;
  if (tot < 0.25) return 'חוסר משקל. הגוף "מרחף" ברצועות והקיבה מתרוממת, כמו ברגע העליון של רכבת הרים.';
  if (g.y < -0.3) return `הפוך: הרצועות מחזיקות אתכם ב-${(-g.y).toFixed(1)}g, והדם זורם לראש. ${car.contacts ? 'הרכב עצמו לא נופל כי המאווררים מושכים אותו לקיר.' : ''}`;
  if (g.z > 1.8) return `נדחסים למושב ב-${g.z.toFixed(1)}g. קשה להרים את הראש מהמשענת, והנשימה נעשית כבדה.`;
  if (g.z > 0.9) return `הגב נלחץ למושב (${g.z.toFixed(1)}g אחורה). הראש מרגיש כבד פי ${tot.toFixed(1)}.`;
  if (g.z < -1.2) return `נזרקים קדימה אל החגורות (${(-g.z).toFixed(1)}g). כך מרגישה בלימה של מכונית מרוץ.`;
  if (Math.abs(g.x) > 1.3) return `הגוף נזרק ${g.x > 0 ? 'ימינה' : 'שמאלה'} ב-${Math.abs(g.x).toFixed(1)}g. בלי מושב צמוד לא הייתם נשארים במקום.`;
  if (g.y > 2.2) return `נמחצים למושב ב-${g.y.toFixed(1)}g כלפי מטה. הידיים כבדות פי ${g.y.toFixed(1)}, והראייה עלולה להצטמצם.`;
  if (g.y > 1.4) return `כבדים פי ${g.y.toFixed(1)}: הרכב מסתובב בקשת והכוח הצנטריפטלי לוחץ אתכם למושב.`;
  if (air) return 'באוויר. התחושה תלויה בדחף: כשהמנועים כבויים, זה חוסר משקל.';
  if (Math.abs(g.x) > 0.6) return `נטייה לצד ב-${Math.abs(g.x).toFixed(1)}g, כמו בפנייה חדה במכונית ספורט.`;
  return 'נסיעה רגילה. בערך 1g כלפי מטה, כמו בכל מכונית.';
}
