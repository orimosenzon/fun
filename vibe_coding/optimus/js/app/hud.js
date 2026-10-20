// התצוגה: לוחות המידע, חלון המשימות, מסך הפתיחה והמקשים.
import { TASKS } from '../tasks/tasks.js';
import { G } from '../robot/spec.js';

const $ = (id) => document.getElementById(id);
const JOINT_HE = {
  hip_yaw: 'ירך (סיבוב)', hip_roll: 'ירך (צד)', hip_pitch: 'ירך', knee: 'ברך', ankle_pitch: 'קרסול', ankle_roll: 'קרסול (צד)',
  waist_yaw: 'מותן (סיבוב)', waist_pitch: 'מותן', neck_yaw: 'צוואר', neck_pitch: 'צוואר (הנהון)',
  shoulder_pitch: 'כתף', shoulder_roll: 'כתף (צד)', shoulder_yaw: 'כתף (סיבוב)', elbow: 'מרפק', forearm_roll: 'אמה (סיבוב)', wrist_flex: 'שורש כף היד', wrist_dev: 'שורש (צד)',
};
export function jointHe(name) {
  const side = name.startsWith('l_') ? ' שמאל' : name.startsWith('r_') ? ' ימין' : '';
  const base = name.replace(/^[lr]_/, '');
  return (JOINT_HE[base] || base) + side;
}
// מיקומי המפרקים על "צללית" הגוף בחלון החום (מבט מלפנים: ימין הרובוט משמאל לצופה)
const HEAT_POS = {
  neck: [75, 30], waist: [75, 98],
  l_shoulder: [104, 52], r_shoulder: [46, 52], l_elbow: [112, 90], r_elbow: [38, 90], l_wrist: [116, 124], r_wrist: [34, 124],
  l_hip: [90, 120], r_hip: [60, 120], l_knee: [92, 165], r_knee: [58, 165], l_ankle: [93, 210], r_ankle: [57, 210],
};
function heatKey(n) {
  if (n.startsWith('neck')) return 'neck';
  if (n.startsWith('waist')) return 'waist';
  const s = n[0], b = n.slice(2);
  if (b.startsWith('shoulder')) return s + '_shoulder';
  if (b.startsWith('elbow') || b.startsWith('forearm')) return s + '_elbow';
  if (b.startsWith('wrist')) return s + '_wrist';
  if (b.startsWith('hip')) return s + '_hip';
  if (b.startsWith('knee')) return s + '_knee';
  if (b.startsWith('ankle')) return s + '_ankle';
}
function heatColor(x) {
  x = Math.max(0, Math.min(1, x));
  const stops = [[0, [43, 58, 74]], [0.33, [35, 196, 131]], [0.66, [246, 195, 59]], [1, [255, 77, 61]]];
  for (let i = 1; i < stops.length; i++) if (x <= stops[i][0]) {
    const [a, ca] = stops[i - 1], [b, cb] = stops[i], t = (x - a) / (b - a);
    return `rgb(${ca.map((c, k) => Math.round(c + (cb[k] - c) * t)).join(',')})`;
  }
  return 'rgb(255,77,61)';
}

export class HUD {
  constructor() {
    this.toastT = 0;
    this.t = 0;
    this.heatAvg = {};
  }
  toast(msg, ms = 2400) {
    const el = $('toast'); el.textContent = msg; el.classList.add('on');
    clearTimeout(this.toastT); this.toastT = setTimeout(() => el.classList.remove('on'), ms);
  }
  setScene() {}
  setPushArmed(v) { $('push-arm').classList.toggle('on', v); }

  bind(app, ctx) {
    this.app = app; this.ctx = ctx;
    const { rig, input } = ctx;
    // מסך פתיחה
    const intro = $('intro');
    const close = () => { intro.classList.add('hidden'); app.paused = false; };
    $('go').onclick = close;
    intro.querySelectorAll('.scard').forEach(c => c.onclick = () => {
      const a = c.dataset.act;
      if (a === 'academy') location.href = 'academy.html';
      else { close(); if (a === 'tasks') this.openTasks(); }
    });
    $('btn-menu').onclick = () => { intro.classList.remove('hidden'); };
    $('opt-overlay').onchange = (e) => { app.overlay.setVisible(e.target.checked); };
    $('opt-pip').onchange = (e) => { this.setPip(e.target.checked); };
    $('opt-quality').onchange = (e) => this.setQuality(e.target.value);
    this.setQuality(ctx.store.get('quality', 'med'));
    $('opt-quality').value = ctx.store.get('quality', 'med');
    // משימות
    const list = $('task-list');
    list.innerHTML = TASKS.map(t => `<div class="scard" data-k="${t.key}"><h4>${t.icon} ${t.name}</h4><p>${t.desc}</p>${t.meta ? `<div class="meta">${t.meta}</div>` : ''}</div>`).join('');
    list.querySelectorAll('.scard').forEach(c => c.onclick = () => { $('tasks-modal').classList.add('hidden'); this.startTask(c.dataset.k); });
    $('btn-tasks').onclick = () => this.openTasks();
    $('tasks-close').onclick = () => $('tasks-modal').classList.add('hidden');
    $('tc-stop').onclick = () => { app.tasks.stop(); };
    // כלים
    $('push-arm').onclick = () => { input.pushArmed = !input.pushArmed; this.setPushArmed(input.pushArmed); if (input.pushArmed) this.toast('לחצו על הרובוט כדי לדחוף אותו'); };
    $('push-strength').oninput = (e) => { input.pushStrength = +e.target.value; $('push-val').textContent = `${e.target.value} נ"ש`; };
    $('btn-pause').onclick = () => this.togglePause();
    $('btn-slow').onclick = () => this.toggleSlow();
    $('btn-reset').onclick = () => this.reset();
    // מקשים
    input.onKey = (e) => this.onKey(e);
    app.paused = true;
  }
  openTasks() { $('tasks-modal').classList.remove('hidden'); }
  startTask(k) {
    const app = this.app;
    // משימות מתחילות מעולם נקי: הרובוט עומד בנקודת ההתחלה והחפצים במקומם
    const T = TASKS.find(t => t.key === k);
    if (T?.fresh !== false) this.ctx.loadScene(T?.scene || app.sceneKey);
    app.tasks.start(k);
    app.paused = false;
  }
  togglePause() { this.app.paused = !this.app.paused; $('btn-pause').classList.toggle('on', this.app.paused); $('btn-pause').textContent = this.app.paused ? '▶' : '⏸'; }
  toggleSlow() { this.app.slow = this.app.slow === 1 ? 0.25 : 1; $('btn-slow').classList.toggle('on', this.app.slow !== 1); this.toast(this.app.slow === 1 ? 'מהירות רגילה' : 'הילוך איטי ×¼'); }
  reset() { const k = this.app.tasks.active?.key; this.ctx.loadScene(this.app.sceneKey); if (k && k !== 'goto') this.app.tasks.start(k); this.toast('איפוס'); }
  setPip(on) { this.app.pip = on; $('pip').classList.toggle('hidden', !on); }
  setQuality(q) {
    const { renderer } = this.ctx;
    renderer.setPixelRatio(q === 'high' ? Math.min(devicePixelRatio, 2) : q === 'med' ? Math.min(devicePixelRatio, 1.5) : 1);
    renderer.shadowMap.enabled = q !== 'low';
    this.ctx.store.set('quality', q);
    dispatchEvent(new Event('resize'));
  }
  onKey(e) {
    const app = this.app, { rig, input } = this.ctx, r = app.robot;
    switch (e.code) {
      case 'KeyC': { const m = rig.cycle(); $('hud-cam').textContent = 'מצלמה: ' + ({ follow: 'עוקבת', eyes: 'עיני הרובוט', free: 'חופשית' })[m]; break; }
      case 'KeyX': { const v = !app.overlay.visible; app.overlay.setVisible(v); $('opt-overlay').checked = v; break; }
      case 'KeyP': this.setPip(!app.pip); $('opt-pip').checked = app.pip; break;
      case 'Space': e.preventDefault(); this.togglePause(); break;
      case 'KeyM': this.toggleSlow(); break;
      case 'KeyR': this.reset(); break;
      case 'KeyB': input.pushFromCamera(); break;
      case 'Escape': $('intro').classList.toggle('hidden'); break;
      case 'KeyH': $('intro').classList.remove('hidden'); document.querySelector('details.controls').open = true; break;
      case 'Digit1': input.setHandMode('l'); this.toast('טלאופרציה: יד שמאל'); break;
      case 'Digit2': input.setHandMode('r'); this.toast('טלאופרציה: יד ימין'); break;
      case 'Digit3': input.setHandMode('both'); this.toast('טלאופרציה: שתי הידיים (מראה)'); break;
      case 'Digit0': input.setHandMode(null); this.toast('הידיים חזרו למצב רגיל'); break;
      case 'KeyG': if (input.handMode) input.toggleGrip(input.handMode); break;
      case 'KeyT': {
        const shapes = [['power', 'כוח'], ['pinch', 'צביטה'], ['cylinder', 'גלילית'], ['point', 'הצבעה'], ['peace', 'וי'], ['hook', 'וו']];
        const i = (shapes.findIndex(s => s[0] === input.gripShape) + 1) % shapes.length;
        input.gripShape = shapes[i][0]; $('grip-shape').textContent = shapes[i][1];
        for (const s of ['l', 'r']) if (input.grip[s]) r.hands.set(s, input.gripShape, 1, 3);
        break;
      }
      case 'KeyZ': r.crouch = r.crouch > 0.05 ? 0 : 0.16; this.toast(r.crouch ? 'כפיפת ברכיים: מרכז המסה יורד 16 ס"מ' : 'זקוף'); break;
      case 'KeyV': r.lean = r.lean > 0.1 ? 0 : 0.35; break;
    }
  }

  // ---------- עדכון בכל פריים ----------
  update(app, dt) {
    this.t += dt;
    const r = app.robot;
    if (!r) return;
    const tm = r.telemetry();
    // משימה
    const T = app.tasks.active, ctx = app.tasks.ctx;
    const card = $('task-card');
    if (T || ctx) {
      card.classList.remove('hidden');
      $('tc-icon').textContent = (T || {}).icon || '🤖';
      $('tc-name').textContent = (T || {}).name || '';
      $('tc-desc').textContent = (T || {}).desc || '';
      const st = $('tc-status');
      st.textContent = app.tasks.status;
      st.className = ctx?.done ? 'ok' : ctx?.failed ? 'bad' : '';
      $('hud-task').textContent = (T ? T.icon + ' ' + T.name : 'שליטה ידנית');
    } else { card.classList.add('hidden'); $('hud-task').textContent = 'שליטה ידנית'; }
    if (r.fallen && !this.fallShown) { this.fallShown = true; this.toast('הרובוט נפל. R לאיפוס', 4000); }
    if (!r.fallen) this.fallShown = false;
    // הידיים
    const input = this.ctx.input;
    $('hand-hud').classList.toggle('hidden', !input.handMode);
    if (input.handMode) $('hand-which').textContent = ({ l: 'יד שמאל', r: 'יד ימין', both: 'שתי הידיים' })[input.handMode];
    // ההספק מתעדכן בקצב נמוך יותר
    if ((this.acc = (this.acc || 0) + dt) < 0.1) return;
    this.acc = 0;
    $('b-mode').textContent = r.fallen ? 'נפל' : r.loco.mode === 'stand' ? 'עמידה' : r.loco.recovering ? 'צעד התאוששות' : 'הליכה';
    $('b-speed').textContent = tm.speed.toFixed(2);
    $('b-comz').textContent = tm.com[2].toFixed(2);
    $('b-steps').textContent = r.loco.stepCount;
    $('e-power').textContent = Math.round(tm.powerAvg);
    $('e-pbar').style.width = Math.min(100, tm.powerAvg / 10) + '%';
    $('e-soc').textContent = `${(tm.soc * 100).toFixed(1)}%`;
    $('e-hours').textContent = tm.hours.toFixed(1);
    this.drawBalance(r);
    this.drawHeat(r);
  }

  drawBalance(r) {
    const cv = $('balance'), c = cv.getContext('2d'), W = cv.width, H = cv.height;
    c.clearRect(0, 0, W, H);
    const v = r.loco.viz, com = r.wbc.com();
    if (!v.poly) return;
    // מבט מלמעלה, מסובב כך שהרובוט תמיד "פונה למעלה"
    const h = r.loco.heading, ch = Math.cos(h), sh = Math.sin(h);
    const S = 340, cx = W / 2, cy = H / 2 + 10;
    const mid = [(r.loco.feet.l.p[0] + r.loco.feet.r.p[0]) / 2, (r.loco.feet.l.p[1] + r.loco.feet.r.p[1]) / 2];
    const P = (p) => { const dx = p[0] - mid[0], dy = p[1] - mid[1]; const fx = ch * dx + sh * dy, fy = -sh * dx + ch * dy; return [cx - fy * S, cy - fx * S]; };
    // רשת
    c.strokeStyle = 'rgba(255,255,255,.06)'; c.lineWidth = 1;
    for (let i = -3; i <= 3; i++) { c.beginPath(); c.moveTo(cx + i * 34, 0); c.lineTo(cx + i * 34, H); c.stroke(); c.beginPath(); c.moveTo(0, cy + i * 34); c.lineTo(W, cy + i * 34); c.stroke(); }
    // כפות רגליים
    for (const s of ['l', 'r']) {
      const id = r.wbc.ids[s + '_sole'], p = r.wbc.sitePos(id), R = r.wbc.bodyMat(r.wbc.ids[s + '_foot']);
      const yaw = Math.atan2(R[3], R[0]), hx = G.footLen / 2, hy = G.footW / 2, cc = Math.cos(yaw), ss = Math.sin(yaw);
      const corners = [[hx, hy], [-hx, hy], [-hx, -hy], [hx, -hy]].map(([x, y]) => P([p[0] + cc * x - ss * y, p[1] + ss * x + cc * y]));
      const stance = (r.wbc.out.contacts || []).includes(s);
      c.fillStyle = stance ? 'rgba(214,195,155,.35)' : 'rgba(255,255,255,.07)';
      c.strokeStyle = stance ? 'rgba(214,195,155,.9)' : 'rgba(255,255,255,.35)';
      c.beginPath(); corners.forEach((q, i) => i ? c.lineTo(...q) : c.moveTo(...q)); c.closePath(); c.fill(); c.stroke();
    }
    // מצולע תמיכה
    c.strokeStyle = '#23c483'; c.lineWidth = 1.5; c.setLineDash([4, 3]);
    c.beginPath(); v.poly.forEach((q, i) => { const pp = P(q); i ? c.lineTo(...pp) : c.moveTo(...pp); }); c.closePath(); c.stroke(); c.setLineDash([]);
    // צעדים מתוכננים
    c.strokeStyle = 'rgba(255,194,61,.7)';
    for (const st of (v.plan || []).slice(0, 3)) { const q = P(st.p); c.beginPath(); c.arc(q[0], q[1], 5, 0, Math.PI * 2); c.stroke(); }
    // ZMP, מרכז מסה, DCM
    const z = P(v.zmp), m = P(com), x = P(v.xi);
    c.strokeStyle = 'rgba(47,140,255,.6)'; c.beginPath(); c.moveTo(...m); c.lineTo(...x); c.stroke();
    c.fillStyle = '#ff4d3d'; c.beginPath(); c.arc(z[0], z[1], 5, 0, Math.PI * 2); c.fill();
    c.strokeStyle = '#fff'; c.lineWidth = 2; c.beginPath(); c.arc(m[0], m[1], 6, 0, Math.PI * 2); c.stroke();
    c.fillStyle = '#2f8cff'; c.beginPath(); c.arc(x[0], x[1], 4.5, 0, Math.PI * 2); c.fill();
    c.fillStyle = 'rgba(255,255,255,.4)'; c.font = '10px Rubik'; c.textAlign = 'center'; c.fillText('קדימה ▲', cx, 12);
  }

  drawHeat(r) {
    const cv = $('body-heat'), c = cv.getContext('2d'), W = cv.width, H = cv.height;
    c.clearRect(0, 0, W, H);
    // צללית פשוטה
    c.fillStyle = 'rgba(214,195,155,.10)'; c.strokeStyle = 'rgba(214,195,155,.35)'; c.lineWidth = 1;
    const sil = new Path2D('M75 8 a16 18 0 1 1 0.1 0 Z M50 44 L100 44 Q112 46 114 60 L120 128 L110 130 L104 66 L100 112 L96 222 L82 222 L76 130 L74 130 L68 222 L54 222 L50 112 L46 66 L40 130 L30 128 L36 60 Q38 46 50 44 Z');
    c.fill(sil); c.stroke(sil);
    // ערך לכל מפרק: המקסימום של |τ|/τmax בקבוצה, עם החלקה
    const groups = {};
    r.wbc.joints.forEach((j, k) => {
      const key = heatKey(j.name); if (!key) return;
      groups[key] = Math.max(groups[key] || 0, Math.abs(r.tau[k]) / j.torque);
    });
    for (const [k, val] of Object.entries(groups)) {
      this.heatAvg[k] = (this.heatAvg[k] ?? val) * 0.6 + val * 0.4;
      const p = HEAT_POS[k]; if (!p) continue;
      const x = this.heatAvg[k];
      c.fillStyle = heatColor(x);
      c.beginPath(); c.arc(p[0], p[1], 6 + x * 5, 0, Math.PI * 2); c.fill();
    }
    // המפרקים העמוסים ביותר
    const rows = r.wbc.joints.map((j, k) => [j, Math.abs(r.tau[k])]).sort((a, b) => b[1] / b[0].torque - a[1] / a[0].torque).slice(0, 4);
    $('top-joints').innerHTML = rows.map(([j, t]) => `<div class="kv"><span>${jointHe(j.name)}</span><b>${t.toFixed(0)}/${j.torque} N·m</b></div>`).join('');
  }
}
