// Heads-up display (DOM overlay) and the mini-map.
import { T } from '../sim/terrain.js';
import { G, FUEL0, TILT_MAX } from '../sim/bike.js';
import { V_MAX } from '../sim/fcs.js';
import { len, yprOf } from '../sim/vmath.js';

const $ = (id) => document.getElementById(id);
export const fmtTime = (t) => { const m = Math.floor(t / 60), s = t - m * 60; return `${m}:${s < 10 ? '0' : ''}${s.toFixed(2)}`; };
const REASON = { ground: 'פגעת בקרקע', hard: 'נחיתה קשה מדי', water: 'נפלת למים', tree: 'עץ', building: 'בניין', ring: 'פגעת בטבעת', bounds: 'יצאת מגבולות העמק', fuel: 'נגמר הדלק' };

export class Hud {
  constructor() {
    this.el = $('hud'); this.msgT = 0; this.warnT = 0; this.acc = 0;
    this.mapC = document.querySelector('#map canvas'); this.mapG = this.mapC.getContext('2d');
    this.base = this.makeMap();
  }
  makeMap() {
    const n = T.n, s = 2, w = (n - 1) / s, c = document.createElement('canvas'); c.width = c.height = w;
    const g = c.getContext('2d'), img = g.createImageData(w, w), H = T.H, lakes = T.lakes;
    for (let j = 0; j < w; j++) for (let i = 0; i < w; i++) {
      const k = j * s * n + i * s, h = H[k];
      const sh = Math.max(0, Math.min(1, 0.62 + ((H[k] - H[k + s]) * 0.6 + (H[k] - H[k + s * n]) * 0.5) / (T.cell * s)));
      let col = h < 1500 ? [58, 92, 50] : h < 2100 ? [96, 104, 68] : h < 2380 ? [118, 114, 106] : [226, 232, 240];
      const x = -T.half + i * s * T.cell, z = -T.half + j * s * T.cell;
      for (const L of lakes) if (h < L.level && x > L.x0 && x < L.x1 && z > L.z0 && z < L.z1) col = [44, 120, 150];
      const q = (j * w + i) * 4; img.data[q] = col[0] * (0.35 + 0.8 * sh); img.data[q + 1] = col[1] * (0.35 + 0.8 * sh); img.data[q + 2] = col[2] * (0.35 + 0.8 * sh); img.data[q + 3] = 255;
    }
    g.putImageData(img, 0, 0);
    g.strokeStyle = '#3d93b8'; g.lineWidth = 1.5;
    for (const r of T.meta.rivers) { g.beginPath(); r.pts.forEach((p, i) => { const x = (p[0] + T.half) / (T.cell * s), y = (p[2] + T.half) / (T.cell * s); i ? g.lineTo(x, y) : g.moveTo(x, y); }); g.stroke(); }
    g.fillStyle = '#ffffff';
    for (const p of Object.values(T.meta.pads)) { g.beginPath(); g.arc((p[0] + T.half) / (T.cell * s), (p[2] + T.half) / (T.cell * s), 2.5, 0, 7); g.fill(); }
    return c;
  }
  show(on) { this.el.classList.toggle('on', on); }
  setEvent(game) {
    const ev = game.ev, m = ev.medals;
    let h = `<b>${ev.name}</b><br>`;
    if (m) h += `<span class="m"><i class="medal g3"></i> <span class="num">${fmtTime(m[0])}</span></span><span class="m"><i class="medal g2"></i> <span class="num">${fmtTime(m[1])}</span></span><span class="m"><i class="medal g1"></i> <span class="num">${fmtTime(m[2])}</span></span><br>`;
    if (game.best) h += `השיא שלך: <span class="num">${fmtTime(game.best.time)}</span>`;
    else if (m) h += 'עוד אין לך שיא כאן';
    $('evt').innerHTML = h;
    $('top').style.display = m ? '' : 'none'; $('score').style.display = m ? 'none' : '';
    $('delta').textContent = '';
  }
  message(text, sub = '', dur = 1.6) { const e = $('msg'); e.innerHTML = text + (sub ? `<small>${sub}</small>` : ''); e.classList.add('on'); this.msgT = dur; }
  warn(text) { const e = $('warn'); e.textContent = text; e.classList.add('on'); this.warnT = 0.5; }
  crash(reason) { this.message('💥', REASON[reason] || reason, 2.2); }
  delta(d) {
    const e = $('delta');
    if (d === null || d === undefined) { e.textContent = ''; return; }
    e.textContent = (d <= 0 ? '−' : '+') + Math.abs(d).toFixed(2); e.className = 'num ' + (d <= 0 ? 'good' : 'bad');
    e.style.opacity = 1; clearTimeout(this._dt); this._dt = setTimeout(() => (e.style.opacity = 0), 2600);
  }
  hint(html) { $('hint').innerHTML = html || ''; }

  update(dt, game, cam, viewW, viewH) {
    if (this.msgT > 0 && (this.msgT -= dt) <= 0) $('msg').classList.remove('on');
    if (this.warnT > 0 && (this.warnT -= dt) <= 0) $('warn').classList.remove('on');
    const b = game.bike;
    // next check-point arrow (every frame, it has to be smooth)
    const cp = game.nextCp, ar = $('arrow');
    if (cp && (game.state === 'fly' || game.state === 'countdown')) {
      const v = cam.worldToScreen(cp.p[0], cp.p[1] + (cp.type === 'land' ? 2 : 0), cp.p[2]);
      const dist = Math.hypot(cp.p[0] - b.p[0], cp.p[1] - b.p[1], cp.p[2] - b.p[2]);
      const off = v.behind || Math.abs(v.x) > 0.92 || Math.abs(v.y) > 0.88;
      if (off) {
        let x = v.x, y = v.y; if (v.behind) { x = -x; y = -y; if (Math.abs(x) < 0.2 && Math.abs(y) < 0.2) y = -1; }
        const k = 0.86 / Math.max(Math.abs(x), Math.abs(y), 1e-3); x *= k; y *= k;
        ar.style.opacity = 1; ar.style.left = (x * 0.5 + 0.5) * viewW + 'px'; ar.style.top = (-y * 0.5 + 0.5) * viewH + 'px';
        ar.firstElementChild.style.transform = `rotate(${Math.atan2(-y, x) * 180 / Math.PI + 45}deg)`;
        ar.lastElementChild.textContent = dist > 999 ? (dist / 1000).toFixed(1) + ' km' : dist.toFixed(0) + ' m';
      } else ar.style.opacity = 0;
    } else ar.style.opacity = 0;

    this.acc += dt; if (this.acc < 0.066) return; this.acc = 0;
    const V = len(b.v);
    $('kmh').textContent = (V * 3.6).toFixed(0);
    $('agl').textContent = Math.max(0, b.agl).toFixed(b.agl < 20 ? 1 : 0);
    $('vs').textContent = (b.v[1] >= 0 ? '+' : '') + b.v[1].toFixed(0);
    $('gg').textContent = (Math.hypot(b.acc[0], b.acc[1] + G, b.acc[2]) / G).toFixed(1);
    const vmax = V_MAX + 30;
    $('vt').firstElementChild.style.width = Math.min(100, V / vmax * 100) + '%';
    $('vt').lastElementChild.style.left = Math.max(0, Math.min(100, game.vt / vmax * 100)) + '%';
    for (let i = 0; i < 4; i++) { const e = b.eng[i]; $('e' + i).style.height = Math.min(100, e.T / 21.5) + '%'; $('e' + i).style.background = e.ab > 0.3 ? 'linear-gradient(#bfe8ff,#7a6bff)' : ''; }
    $('nozA').setAttribute('transform', `rotate(${b.tilt * 180 / Math.PI})`);
    $('heat').firstElementChild.style.width = b.heat * 100 + '%'; $('heat').classList.toggle('lock', b.abLock);
    $('fuel').firstElementChild.style.width = b.fuel / FUEL0 * 100 + '%';
    if (game.ev.medals) {
      $('timer').textContent = fmtTime(game.total);
      $('cpn').innerHTML = `נקודה <span class="num">${Math.min(game.cp + 1, game.cps.length)} / ${game.cps.length}</span>` + (game.penalty ? ` · עונשין <span class="num">+${game.penalty}</span>` : '');
    } else $('score').innerHTML = `${Math.floor(game.score).toLocaleString('en')} נק'` + (game.combo > 1 ? ` · <span style="color:#ffc440">×${game.combo}</span>` : '');
    this.drawMap(game);
  }
  drawMap(game) {
    const g = this.mapG, W = this.mapC.width, b = game.bike, s = this.base.width / (2 * T.half);
    const span = 2600, px = (b.p[0] + T.half) * s, pz = (b.p[2] + T.half) * s, r = span * s / 2;
    g.fillStyle = '#10161d'; g.fillRect(0, 0, W, W);
    g.drawImage(this.base, px - r, pz - r, 2 * r, 2 * r, 0, 0, W, W);
    const k = W / span, tx = (x) => (x - b.p[0]) * k + W / 2, tz = (z) => (z - b.p[2]) * k + W / 2;
    game.cps.forEach((c, i) => {
      if (i < game.cp) return;
      const x = tx(c.p[0]), y = tz(c.p[2]); if (x < -8 || y < -8 || x > W + 8 || y > W + 8) return;
      g.beginPath(); g.arc(x, y, i === game.cp ? 9 : 5, 0, 7);
      g.fillStyle = i === game.cp ? '#ffc440' : 'rgba(120,225,255,.85)'; g.fill();
      if (c.type === 'land') { g.strokeStyle = '#fff'; g.lineWidth = 2; g.stroke(); }
    });
    const [yaw] = yprOf(b.q);
    g.save(); g.translate(W / 2, W / 2); g.rotate(-yaw);
    g.beginPath(); g.moveTo(15, 0); g.lineTo(-10, -9); g.lineTo(-5, 0); g.lineTo(-10, 9); g.closePath(); g.fillStyle = '#fff'; g.strokeStyle = '#000'; g.lineWidth = 2; g.stroke(); g.fill(); g.restore();
  }
}
