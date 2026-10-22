// חיבור הכול: מצבים, לולאת סימולציה, קלט, טלמטריה.
import { makeHouse, makeArena, randomFreePose, G } from './world.js';
import { Robot, collideRobots, PHONES, MOTORS, POWER, MOUNTS, BUMPERS, LINKS } from './robot.js';
import { OccGrid, Explorer, Cat, Follower, SumoAI, QLearner, seeTarget, RL_ACTIONS } from './modes.js';
import { MapView, CamView, Face, readTheme } from './render.js';

const $ = id => document.getElementById(id);
const DT = 1 / 120;

const HELP = {
  drive: 'נוהגים בעצמכם: חיצים או WASD (בטלפון: הג׳ויסטיק). נסו להתנגש בקיר במהירות מלאה ותראו מה קורה לטלפון. סמנו "שליטה מרחוק" כדי להרגיש איך החותנת מקליפורניה נוהגת עם השהיית רשת.',
  map: 'הרובוט ממפה את הבית לבד. אין לו לידאר, רק מצלמה ורשת שמעריכה עומק מתמונה אחת, עם שדה ראייה של 70°. אפור = לא ידוע, שקוף = פנוי, כהה = מכשול. האלגוריתם: נוסעים תמיד אל הגבול הקרוב בין הידוע ללא ידוע (frontier).',
  follow: 'רשת זיהוי עצמים רצה על הטלפון ומחפשת את החתול. כשהוא בשדה הראייה הרובוט מתקרב עד 70 ס״מ; כשהוא נעלם, מסתובבים לצד שבו נראה לאחרונה. אפשר לגרור את החתול במפה.',
  sumo: '"אנדרואיד הזועם": מפילים את הטלפון של היריב או דוחפים אותו מהשולחן. לתושבת יש משענת מאחור, אז הטלפון עף בקלות קדימה (מכה בחזית, בלימה חדה) וקשה אחורה. מכה חזיתית מפילה את שניכם. הטריק המלוכלך: לנגח ברוורס בפרצוף של היריב.',
  rl: 'למידת חיזוק (Q-learning) מאפס: 5 קרני עומק, כל אחת "קרוב / בינוני / רחוק", ושלוש פעולות. פרס על נסיעה קדימה, עונש של 10 על התנגשות. בהתחלה הוא אקראי לגמרי; הגבירו מהירות וצפו בגרף.',
};

const state = {
  mode: 'drive', world: makeHouse(), arena: makeArena(), robots: [], cat: null, grid: null, explorer: null,
  follower: new Follower(), ai: null, ai2: null, q: new QLearner(), rlSpeed: 1, showMap: true,
  keys: new Set(), joy: { x: 0, y: 0, on: false }, poseHist: [], netQ: [], score: [0, 0], roundOver: 0,
  camMode: 'rgb', fallT: 0, ctrlAcc: 0, gridAcc: 0, simT: 0, deadSaid: false,
};

const cfg = () => ({
  phone: $('cfg-phone').value, motor: $('cfg-motor').value, power: $('cfg-power').value,
  mount: $('cfg-mount').value, bumper: $('cfg-bumper').value, link: $('cfg-link').value,
  portrait: $('cfg-portrait').checked,
});
const AI_CFG = { phone: 'mid', motor: 'tt', power: 'li', mount: 'clip', bumper: 'hard', link: 'ble', portrait: false };

function fillSelect(id, obj) {
  const el = $(id);
  for (const [k, v] of Object.entries(obj)) { const o = document.createElement('option'); o.value = k; o.textContent = v.name; el.append(o); }
}
fillSelect('cfg-phone', PHONES); fillSelect('cfg-motor', MOTORS); fillSelect('cfg-power', POWER);
fillSelect('cfg-mount', MOUNTS); fillSelect('cfg-bumper', BUMPERS); fillSelect('cfg-link', LINKS);
$('cfg-power').value = 'li';

const mapView = new MapView($('map'));
const cam = new CamView($('cam'));
const face = new Face($('face'));

const currentWorld = () => state.mode === 'sumo' ? state.arena : state.world;

function log(text, type = '') {
  const li = document.createElement('li'); li.textContent = text; if (type) li.className = type;
  $('log').prepend(li);
  while ($('log').children.length > 30) $('log').lastChild.remove();
}
let bannerT = 0;
function banner(text, type = '', secs = 3) {
  const b = $('banner'); b.textContent = text; b.className = 'banner show ' + type; bannerT = secs;
}

function setMode(m) {
  state.mode = m;
  document.querySelectorAll('.modes button').forEach(b => b.classList.toggle('on', b.dataset.mode === m));
  document.querySelectorAll('[data-for]').forEach(el => { el.hidden = !el.dataset.for.split(' ').includes(m); });
  $('modehelp').textContent = HELP[m];
  resetMode();
  const coarse = matchMedia('(pointer: coarse)').matches;
  $('joy').classList.toggle('show', coarse && (m === 'drive' || m === 'sumo'));
}

function resetMode() {
  const m = state.mode, c = cfg();
  state.cat = null; state.grid = null; state.explorer = null; state.path = null;
  state.poseHist = []; state.netQ = []; state.roundOver = 0; state.simT = 0; state.deadSaid = false;
  if (m === 'sumo') {
    const a = new Robot(c, { x: -0.6, y: 0, th: 0 });
    const b = new Robot(AI_CFG, { x: 0.6, y: 0, th: Math.PI });
    b.color = '#d64545';
    state.robots = [a, b];
    state.ai = new SumoAI(state.arena.R, $('sumo-foe').value === 'dirty'); state.ai2 = new SumoAI(state.arena.R);
  } else {
    const start = m === 'map' ? { x: 1.0, y: 1.6, th: 0 } : m === 'rl' ? randomFreePose(state.world, 0.4) : { x: 2.5, y: 3.2, th: 0 };
    const r = new Robot(c, start);
    state.robots = [r];
    if (m === 'map') { state.grid = new OccGrid(state.world); state.explorer = new Explorer(state.grid); }
    if (m === 'follow' || m === 'drive') state.cat = new Cat(state.world);
    if (m === 'follow') state.follower = new Follower();
    if (m === 'rl') { r.canFall = false; state.q.state = -1; }
  }
  state.robots[0].canFall = m !== 'rl' && $('cfg-fall').checked;
  if (state.robots[1]) state.robots[1].canFall = $('cfg-fall').checked;
  mapView.fit(currentWorld());
}

// ---------- קלט ----------
const KEYMAP = { ArrowUp: 'u', KeyW: 'u', ArrowDown: 'd', KeyS: 'd', ArrowLeft: 'l', KeyA: 'l', ArrowRight: 'r', KeyD: 'r', Space: 'b' };
function simVisible() { const r = $('sim').getBoundingClientRect(); return r.top < innerHeight * 0.8 && r.bottom > innerHeight * 0.2; }
addEventListener('keydown', e => {
  if (/INPUT|SELECT|TEXTAREA/.test(e.target.tagName)) return;
  const k = KEYMAP[e.code]; if (!k) return;
  if ((state.mode === 'drive' || state.mode === 'sumo') && simVisible()) { e.preventDefault(); state.keys.add(k); }
});
addEventListener('keyup', e => { const k = KEYMAP[e.code]; if (k) state.keys.delete(k); });
addEventListener('blur', () => state.keys.clear());

function manualCmd() {
  let t = 0, s = 0;
  if (state.keys.has('u')) t += 1; if (state.keys.has('d')) t -= 1;
  if (state.keys.has('r')) s += 1; if (state.keys.has('l')) s -= 1;
  if (state.joy.on) { t = -state.joy.y; s = state.joy.x; }
  if (state.keys.has('b')) return [0, 0];
  const turn = s * (Math.abs(t) > 0.1 ? 0.45 : 0.6);
  let uL = t + turn, uR = t - turn;
  const m = Math.max(1, Math.abs(uL), Math.abs(uR));
  return [uL / m, uR / m];
}

const joy = $('joy'), knob = joy.querySelector('i');
joy.addEventListener('pointerdown', e => { joy.setPointerCapture(e.pointerId); state.joy.on = true; moveJoy(e); });
joy.addEventListener('pointermove', e => { if (state.joy.on) moveJoy(e); });
const endJoy = () => { state.joy = { x: 0, y: 0, on: false }; knob.style.transform = ''; };
joy.addEventListener('pointerup', endJoy); joy.addEventListener('pointercancel', endJoy);
function moveJoy(e) {
  const r = joy.getBoundingClientRect();
  let x = (e.clientX - r.left - r.width / 2) / (r.width / 2), y = (e.clientY - r.top - r.height / 2) / (r.height / 2);
  const m = Math.hypot(x, y); if (m > 1) { x /= m; y /= m; }
  state.joy.x = x; state.joy.y = y;
  knob.style.transform = `translate(${x * 36}px, ${y * 36}px)`;
}

// גרירת החתול
let dragCat = false;
$('map').addEventListener('pointerdown', e => {
  if (!state.cat) return;
  const p = mapView.toWorld(e.clientX, e.clientY);
  if (Math.hypot(p.x - state.cat.x, p.y - state.cat.y) < 0.45) { dragCat = true; state.cat.dragged = true; $('map').setPointerCapture(e.pointerId); }
});
$('map').addEventListener('pointermove', e => {
  if (!dragCat) return;
  const p = mapView.toWorld(e.clientX, e.clientY);
  state.cat.x = Math.max(0.2, Math.min(11.8, p.x)); state.cat.y = Math.max(0.2, Math.min(7.8, p.y));
});
const endDrag = () => { if (dragCat) { dragCat = false; state.cat.dragged = false; state.cat.goal = null; state.cat.wait = 2; } };
$('map').addEventListener('pointerup', endDrag); $('map').addEventListener('pointercancel', endDrag);

// ---------- הספק הטלפון לפי מצב ----------
function phoneWatts() {
  const base = 0.9, camera = 1.3;
  switch (state.mode) {
    case 'drive': return base + camera + ($('cfg-remote').checked ? 2.2 : 0);
    case 'map': return base + camera + 1.8;
    case 'follow': return base + camera + 1.6;
    case 'sumo': return base + camera + 0.8;
    default: return base + camera + 0.3;
  }
}

// ---------- צעד סימולציה ----------
function stepSim(dt) {
  const m = state.mode, world = currentWorld(), R = state.robots, me = R[0];
  state.simT += dt;
  if (m === 'rl') {
    state.q.tick(me, world, dt);
    me.step(dt, world, phoneWatts());
    me.recharge();
    return;
  }
  state.ctrlAcc += dt;
  const ctrl = state.ctrlAcc >= 1 / 30;
  if (ctrl) state.ctrlAcc = 0;

  if (m === 'drive' && ctrl) {
    const u = manualCmd();
    if ($('cfg-remote').checked) {
      const net = +$('cfg-net').value / 1000;
      state.netQ.push({ t: state.simT + net, u });
      while (state.netQ.length && state.netQ[0].t <= state.simT) { const c = state.netQ.shift(); me.cmd(...c.u); }
    } else me.cmd(...u);
  }
  if (m === 'map') {
    state.gridAcc += dt;
    if (state.gridAcc > 0.1) { state.gridAcc = 0; state.grid.integrate(me, world); }
    if (ctrl) {
      const was = state.explorer.done;
      me.cmd(...state.explorer.control(me, world, 1 / 30));
      state.path = state.explorer.path;
      if (!was && state.explorer.done) { log(`המיפוי הושלם: ${state.grid.areaKnown().toFixed(0)} מ״ר ב-${(state.simT / 60).toFixed(1)} דקות`, 'good'); banner('המיפוי הושלם 🎉', 'good'); }
    }
  }
  if (state.cat) state.cat.update(dt);
  if (m === 'follow' && ctrl) me.cmd(...state.follower.control(me, world, state.cat));
  if (m === 'sumo' && ctrl && !state.roundOver) {
    const [a, b] = R;
    if ($('sumo-aivai').checked) a.cmd(...state.ai2.control(a, b, 1 / 30)); else a.cmd(...manualCmd());
    b.cmd(...state.ai.control(b, a, 1 / 30));
  }
  const pw = phoneWatts();
  for (const r of R) r.step(dt, world, pw);
  if (m === 'sumo') {
    collideRobots(R[0], R[1]);
    if (!state.roundOver) {
      const off = R.map(r => Math.hypot(r.x, r.y) > state.arena.R + 0.03);
      const lost = R.map((r, i) => off[i] || !r.phoneOn);
      if (lost[0] || lost[1]) {
        R.forEach((r, i) => { if (off[i]) r.phoneOn = false; });
        state.roundOver = 2.5;
        const why = i => off[i] ? 'נפל מהשולחן' : 'איבד את הטלפון';
        if (lost[0] && lost[1]) banner(`תיקו: שניהם ${off[0] || off[1] ? 'בחוץ' : 'איבדו את הטלפון'} · ${state.score[0]} : ${state.score[1]}`, '');
        else {
          const i = lost[0] ? 0 : 1;
          state.score[1 - i]++;
          banner(`${i === 0 ? 'הכחול' : 'האדום'} ${why(i)}! ${state.score[0]} : ${state.score[1]}`, i === 0 ? 'bad' : 'good');
        }
      }
    }
  }
  state.poseHist.push({ t: state.simT, x: me.x, y: me.y, th: me.th });
  while (state.poseHist.length && state.poseHist[0].t < state.simT - 2.5) state.poseHist.shift();
}

function poseAt(delay) {
  const h = state.poseHist, t = state.simT - delay;
  for (let i = h.length - 1; i >= 0; i--) if (h[i].t <= t) return h[i];
  return h[0] || state.robots[0];
}

// ---------- טלמטריה ----------
function setBar(id, frac, txt, cls) {
  const el = $(id);
  const i = el.querySelector('i');
  i.style.width = Math.max(0, Math.min(100, frac * 100)) + '%';
  i.className = cls || (frac > 0.9 ? 'bad' : frac > 0.65 ? 'warn' : '');
  el.parentElement.querySelector('.v').textContent = txt;
}
const fmtMin = m => m == null ? '' : m > 600 ? `${(m / 60).toFixed(0)} שעות` : m > 90 ? `${(m / 60).toFixed(1)} שעות` : `${m.toFixed(0)} דק׳`;

function telemetry() {
  const r = state.robots[0], p = POWER[r.cfg.power];
  $('t-speed').textContent = `${Math.abs(r.speed).toFixed(2)} / ${r.maxSpeed.toFixed(2)} m/s`;
  $('t-mass').textContent = `${(r.mass * 1000).toFixed(0)} g`;
  setBar('b-current', r.current / p.Imax, `${r.current.toFixed(2)}A מתוך ${p.Imax}A`, r.throttled > 0.05 ? 'bad' : '');
  setBar('b-load', r.loadPeak, r.phoneOn ? `${Math.round(r.loadPeak * 100)}% · סף ${(r.aCritLong / G).toFixed(1)}g קדימה` : 'הטלפון נפל');
  const rt = r.runtimes();
  const pf = r.phoneE / PHONES[r.cfg.phone].wh;
  setBar('b-phone', pf, `${Math.round(pf * 100)}% · ${rt.phoneDraw.toFixed(1)}W · עוד ${fmtMin(rt.phoneMin)}`, pf < 0.15 ? 'bad' : 'good');
  const bWrap = $('b-batt').parentElement;
  if (p.usesPhone) { bWrap.style.opacity = 0.45; setBar('b-batt', 0, 'אין: המנועים שותים מהטלפון', ''); }
  else {
    bWrap.style.opacity = 1;
    const bf = r.battE / p.wh;
    setBar('b-batt', bf, `${Math.round(bf * 100)}% · ${r.pMotorAvg.toFixed(1)}W · עוד ${fmtMin(rt.battMin)}`, bf < 0.15 ? 'bad' : 'good');
  }
  let html = '';
  if (state.mode === 'map') {
    html = `שטח ממופה: <b class="num">${state.grid.areaKnown().toFixed(1)} m²</b> מתוך כ-86 · זמן: <span class="num">${state.simT.toFixed(0)}s</span>${state.explorer.done ? ' · ✅ הושלם' : ''}`;
  } else if (state.mode === 'sumo') {
    html = `ניקוד: <b style="color:#0f9d8a">כחול ${state.score[0]}</b> : <b style="color:#d64545">${state.score[1]} אדום</b> · משקל היריב ${(state.robots[1].mass * 1000).toFixed(0)} g`;
  } else if (state.mode === 'follow') {
    const s = state.follower.seen;
    html = s ? `👁️ רואה את החתול: ${s.d.toFixed(1)} מ׳, ${(s.bearing * 180 / Math.PI).toFixed(0)}°` : '🔍 מחפש את החתול';
  } else if (state.mode === 'rl') {
    const q = state.q, ep = q.episodes, last = ep.slice(-30);
    const avg = last.length ? last.reduce((a, e) => a + e.dist, 0) / last.length : 0;
    html = `אפיזודות: <b class="num">${ep.length}</b> · החלטות: <span class="num">${q.steps}</span> · ε = <span class="num">${q.eps.toFixed(2)}</span><br>ממוצע 30 אחרונות: <b class="num">${avg.toFixed(1)} m</b> עד התנגשות · פעולה: ${RL_ACTIONS[q.action].name}`;
    drawRLChart();
  } else if (state.mode === 'drive') {
    html = $('cfg-remote').checked ? `שליטה מרחוק: הפקודה מגיעה אחרי <span class="num">${$('cfg-net').value}ms</span>, הווידאו אחרי <span class="num">${+$('cfg-net').value + 120}ms</span>` : 'נהיגה מקומית (ערוץ: ' + LINKS[r.cfg.link].name + ')';
  }
  $('t-mode').innerHTML = html;
}

function drawRLChart() {
  const cv = $('rl-chart'), ctx = cv.getContext('2d'), W = cv.width, H = cv.height;
  const T = readThemeCached();
  ctx.clearRect(0, 0, W, H);
  const ep = state.q.episodes; if (ep.length < 2) return;
  const win = 20, pts = [];
  let sum = 0;
  for (let i = 0; i < ep.length; i++) { sum += ep[i].dist; if (i >= win) sum -= ep[i - win].dist; pts.push(sum / Math.min(i + 1, win)); }
  const mx = Math.max(2, ...pts);
  ctx.strokeStyle = T.muted; ctx.lineWidth = 1; ctx.globalAlpha = 0.4;
  ctx.beginPath(); ctx.moveTo(0, H - 1); ctx.lineTo(W, H - 1); ctx.stroke(); ctx.globalAlpha = 1;
  ctx.strokeStyle = T.accent; ctx.lineWidth = 2; ctx.beginPath();
  pts.forEach((v, i) => { const x = i / (pts.length - 1) * (W - 4) + 2, y = H - 4 - v / mx * (H - 18); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
  ctx.stroke();
  ctx.fillStyle = T.muted; ctx.font = '11px Heebo'; ctx.textAlign = 'left'; ctx.direction = 'ltr';
  ctx.fillText(`max ${mx.toFixed(1)} m`, 4, 11);
}
let themeCache = null;
const readThemeCached = () => themeCache || (themeCache = readTheme());

// ---------- ציור ----------
function draw(dtReal) {
  const r = state.robots[0], world = currentWorld();
  mapView.fit(world);
  mapView.draw({
    robots: state.robots, cat: state.cat, grid: state.grid, showMap: state.showMap,
    path: state.mode === 'map' ? state.path : null, rays: state.mode === 'rl' ? state.q.rays : null, showFov: true,
  });
  const remote = state.mode === 'drive' && $('cfg-remote').checked;
  const delay = remote ? +$('cfg-net').value / 1000 + 0.12 : 0;
  const pose = delay ? poseAt(delay) : r;
  const sprites = [];
  if (state.cat) sprites.push({ kind: 'cat', x: state.cat.x, y: state.cat.y, w: 0.4, h: 0.26 });
  if (state.mode === 'sumo') { const o = state.robots[1]; sprites.push({ kind: 'robot', x: o.x, y: o.y, w: 0.22, h: 0.17, color: o.color, phoneOn: o.phoneOn }); }
  let detect = null;
  if (state.mode === 'follow' && state.follower.seen) detect = { sprite: sprites[0], label: 'cat', conf: Math.max(0.4, 0.97 - 0.06 * state.follower.seen.d), d: state.follower.seen.d };
  cam.draw(world, pose, { camH: r.r + 0.09, depth: state.camMode === 'depth', sprites, detect, delay, fallen: !r.phoneOn });

  let mood = 'calm', look = 0;
  if (!r.phoneOn) mood = 'fallen';
  else if (r.loadPeak > 0.7) mood = 'scared';
  else if (state.mode === 'sumo') mood = 'angry';
  else if (r.phoneE / PHONES[r.cfg.phone].wh < 0.15) mood = 'sleepy';
  else if (state.mode === 'follow' && state.follower.seen) { mood = 'happy'; look = -state.follower.seen.bearing * 2; }
  else if (state.mode === 'map' && state.explorer?.done) mood = 'happy';
  if (mood !== 'fallen' && Math.abs(r.w) > 0.3) look = -Math.sign(r.w) * 0.8;
  face.draw(dtReal, mood, look);
  $('phonebox').classList.toggle('fallen', !r.phoneOn && state.fallT > 0);
}

// ---------- לולאה ----------
let last = performance.now(), telT = 0;
function frame(now) {
  const dtReal = Math.min(0.05, (now - last) / 1000); last = now;
  const n = state.mode === 'rl' ? 2 * state.rlSpeed : Math.round(dtReal / DT);
  for (let i = 0; i < n; i++) stepSim(DT);

  for (const r of state.robots) {
    for (const ev of r.events) { log((state.robots.length > 1 ? (r === state.robots[0] ? 'כחול: ' : 'אדום: ') : '') + ev.text, ev.type); if (ev.type === 'bad' && state.mode !== 'sumo') banner(ev.text, 'bad'); }
    r.events.length = 0;
  }
  const me = state.robots[0];
  if (state.mode !== 'sumo' && state.mode !== 'rl' && !me.phoneOn) {
    state.fallT += dtReal;
    if (state.fallT > 2.5) { me.resetPhone(); state.fallT = 0; log('מישהו הרים את הטלפון והחזיר אותו לתושבת', ''); }
  } else state.fallT = 0;
  if (state.mode === 'sumo' && state.roundOver > 0) {
    state.roundOver -= dtReal;
    if (state.roundOver <= 0) {
      const [a, b] = state.robots;
      a.resetPose({ x: -0.6, y: 0, th: 0 }); b.resetPose({ x: 0.6, y: 0, th: Math.PI });
      a.resetPhone(); b.resetPhone(); state.roundOver = 0;
    }
  }
  const p = POWER[me.cfg.power];
  const dead = p.usesPhone ? me.phoneE <= 0 : me.battE <= 0 || me.phoneE <= 0;
  if (dead && !state.deadSaid) { state.deadSaid = true; log(me.phoneE <= 0 ? 'הטלפון כבה: נגמרה הסוללה' : 'סוללת המנועים ריקה', 'bad'); }

  if (bannerT > 0) { bannerT -= dtReal; if (bannerT <= 0) $('banner').classList.remove('show'); }
  draw(dtReal);
  telT -= dtReal;
  if (telT <= 0) { telT = 0.1; telemetry(); }
  requestAnimationFrame(frame);
}

// ---------- חיווט ממשק ----------
document.querySelectorAll('.modes button').forEach(b => b.addEventListener('click', () => setMode(b.dataset.mode)));
for (const id of ['cfg-phone', 'cfg-motor', 'cfg-power', 'cfg-mount', 'cfg-bumper', 'cfg-link', 'cfg-portrait']) {
  $(id).addEventListener('change', () => {
    const c = cfg(); state.robots[0].applyCfg(c);
    if (id === 'cfg-power' || id === 'cfg-phone') state.robots[0].recharge();
    log(`הגדרה: ${$(id).selectedOptions ? $(id).selectedOptions[0]?.textContent || '' : ''}${id === 'cfg-portrait' ? ($('cfg-portrait').checked ? 'טלפון לאורך' : 'טלפון לרוחב') : ''}`);
  });
}
$('cfg-fall').addEventListener('change', () => { for (const r of state.robots) r.canFall = state.mode !== 'rl' && $('cfg-fall').checked; });
$('cfg-net').addEventListener('input', () => { $('net-v').textContent = $('cfg-net').value; });
$('btn-reset').addEventListener('click', () => { resetMode(); log('איפוס'); });
$('btn-mapview').addEventListener('click', e => { state.showMap = !state.showMap; e.target.classList.toggle('on', state.showMap); });
document.querySelectorAll('[data-rlspeed]').forEach(b => b.addEventListener('click', () => {
  state.rlSpeed = +b.dataset.rlspeed;
  document.querySelectorAll('[data-rlspeed]').forEach(x => x.classList.toggle('on', x === b));
}));
$('rl-greedy').addEventListener('click', e => { state.q.greedy = !state.q.greedy; e.target.classList.toggle('on', state.q.greedy); });
$('rl-reset').addEventListener('click', () => { state.q.reset(); state.robots[0].resetPose(randomFreePose(state.world, 0.4)); log('הסוכן שכח הכול'); });
$('sumo-reset').addEventListener('click', () => { state.score = [0, 0]; resetMode(); });
$('sumo-foe').addEventListener('change', () => { state.ai.dirty = $('sumo-foe').value === 'dirty'; });
document.querySelectorAll('[data-cam]').forEach(b => b.addEventListener('click', () => {
  state.camMode = b.dataset.cam;
  document.querySelectorAll('[data-cam]').forEach(x => x.classList.toggle('on', x === b));
}));
$('theme').addEventListener('click', () => {
  const root = document.documentElement;
  const dark = root.dataset.theme ? root.dataset.theme === 'dark' : matchMedia('(prefers-color-scheme: dark)').matches;
  root.dataset.theme = dark ? 'light' : 'dark';
  try { localStorage.setItem('phonebot-theme', root.dataset.theme); } catch {}
  themeCache = null; readTheme();
});
matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => { themeCache = null; readTheme(); });
addEventListener('resize', () => mapView.fit(currentWorld()));

readTheme();
setMode('drive');
log('ברוכים הבאים. הטלפון מחובר, המנועים מחכים.', 'good');
requestAnimationFrame(frame);
