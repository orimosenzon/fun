// הלולאה הראשית: פיזיקה בצעדים קבועים (600 בשנייה), ציור, קלט, קול ותצוגה.
import * as THREE from 'three';
import { World } from './collide.js';
import { buildTrack, LAYOUT, loopGuide } from './track.js';
import { Car } from './car.js';
import { Controller } from './control.js';
import { VARIANTS, AERO, SOURCES } from './params.js';
import { createRenderer, buildEnvironment } from './render/env.js';
import { buildCarModel } from './render/carmodel.js';
import { FX } from './render/fx.js';
import { CameraRig } from './camera.js';
import { Input } from './input.js';
import { Audio } from './audio.js';
import { HUD } from './hud.js';
import { SCENARIOS, scenarioByKey } from './scenarios.js';

const DT = 1 / 600;
const $ = (id) => document.getElementById(id);
const store = {
  get(k, d) { try { const v = localStorage.getItem('roadster.' + k); return v === null ? d : JSON.parse(v); } catch { return d; } },
  set(k, v) { try { localStorage.setItem('roadster.' + k, JSON.stringify(v)); } catch { /* מצב פרטי */ } },
};
const params = new URLSearchParams(location.search);

// ---------- עולם ----------
const world = new World();
const track = buildTrack(world);
const container = $('view');
const renderer = createRenderer(container);
const env = buildEnvironment(renderer, track);
const scene = env.scene;
const camera = new THREE.PerspectiveCamera(62, container.clientWidth / container.clientHeight, 0.1, 9000);
const rig = new CameraRig(camera, renderer.domElement);
const input = new Input();
const audio = new Audio();
const hud = new HUD();

const state = {
  variant: params.get('v') || store.get('variant', 'rocket'),
  scenario: params.get('s') || store.get('scenario', 'free'),
  running: false, paused: true,
  infinite: store.get('infinite', false), assist: store.get('assist', true), sound: store.get('sound', true),
  quality: store.get('quality', 'med'),
  scnState: {}, finished: false, acc: 0,
};
if (!VARIANTS[state.variant]) state.variant = 'rocket';

let car, ctl, model, fx;
function makeCar() {
  if (model) scene.remove(model.root);
  if (fx) { scene.remove(fx.vapor.points); scene.remove(fx.sparks.points); }
  car = new Car(state.variant, world, { infiniteAir: state.infinite });
  ctl = ctl || new Controller();
  ctl.assist = state.assist;
  model = buildCarModel(car.P);
  scene.add(model.root);
  fx = new FX(scene, model);
  hud.setVariant(car);
}

function startScenario(key) {
  const scn = scenarioByKey(key);
  state.scenario = scn.key; store.set('scenario', scn.key);
  if (scn.needs === 'lift' && !car.thr.some(t => t.group === 'lift')) {
    hud.toast('התרחיש הזה דורש את דגם הרוקט (מנועי פינות). עוברים לרוקט.', 2600);
    state.variant = 'rocket'; store.set('variant', 'rocket'); makeCar();
  }
  const [x, z, yaw] = scn.spawn;
  car.reset(x, z, yaw);
  car.refillAir();
  ctl.kick = null; ctl.hoverHold = false;
  ctl.fanMode = scn.fan || 'auto';
  state.scnState = { dt: 0, end: track.lineupEnd };
  scn.init?.(state.scnState);
  state.finished = false;
  rig.snap();
  $('hud-scn').textContent = `${scn.icon} ${scn.nameHe}`;
  hud.toast(scn.howto, 5200);
  $('result').classList.add('hidden');
}

// ---------- מסך פתיחה ----------
function buildIntro() {
  const vc = $('variants'); vc.innerHTML = '';
  for (const [k, V] of Object.entries(VARIANTS)) {
    const d = document.createElement('div');
    d.className = 'vcard' + (k === state.variant ? ' sel' : '');
    const src = (s) => `<span class="src ${s}">${SOURCES[s]}</span>`;
    const lift = V.thrusters.filter(t => t.group === 'lift').length;
    d.innerHTML = `<h4>${V.nameHe} <span class="dim" style="font-weight:400">${V.name}</span></h4>
      <div class="tag">${V.tagline}</div>
      <div class="spec"><span>0–60 מייל/ש' ${src('tesla')}</span><b>&lt;${V.claimed060} s</b></div>
      <div class="spec"><span>מושבים ${src('tesla')}</span><b>${V.seats}</b></div>
      <div class="spec"><span>אוויר דחוס ${src(V.tankVolume ? (k === 'rocket' ? 'tesla' : 'report') : 'tesla')}</span><b>${V.tankVolume ? V.tankVolume * 1000 + ' L' : '—'}</b></div>
      <div class="spec"><span>נחירים</span><b>${V.thrusters.length || '—'}${lift ? ' (כולל המראה)' : ''}</b></div>
      <div class="spec"><span>מסה (עם נהג) ${src('est')}</span><b>${V.massDry + Math.round(V.tankVolume * 378)} kg</b></div>`;
    d.onclick = () => { state.variant = k; store.set('variant', k); buildIntro(); };
    vc.appendChild(d);
  }
  const sc = $('scenarios'); sc.innerHTML = '';
  for (const S of SCENARIOS) {
    const d = document.createElement('div');
    const dis = S.needs === 'lift' && !VARIANTS[state.variant].thrusters.some(t => t.group === 'lift');
    d.className = 'scard' + (S.key === state.scenario ? ' sel' : '') + (dis ? ' disabled' : '');
    d.innerHTML = `<h4>${S.icon} ${S.num}. ${S.nameHe}</h4><p>${S.blurb}${dis ? '<br><b>דורש רוקט</b>' : ''}</p>`;
    d.onclick = () => { state.scenario = S.key; store.set('scenario', S.key); buildIntro(); };
    sc.appendChild(d);
  }
  $('opt-infinite').checked = state.infinite;
  $('opt-assist').checked = state.assist;
  $('opt-sound').checked = state.sound;
  $('opt-quality').value = state.quality;
}
$('opt-infinite').onchange = (e) => { state.infinite = e.target.checked; store.set('infinite', state.infinite); if (car) car.opts.infiniteAir = state.infinite; };
$('opt-assist').onchange = (e) => { state.assist = e.target.checked; store.set('assist', state.assist); if (ctl) ctl.assist = state.assist; };
$('opt-sound').onchange = (e) => { state.sound = e.target.checked; store.set('sound', state.sound); audio.setMuted(!state.sound); };
$('opt-quality').onchange = (e) => { state.quality = e.target.value; store.set('quality', state.quality); applyQuality(); };

function applyQuality() {
  const dpr = window.devicePixelRatio || 1;
  const q = state.quality;
  renderer.setPixelRatio(q === 'high' ? Math.min(dpr, 2) : q === 'med' ? Math.min(dpr, 1.5) : 1);
  renderer.shadowMap.enabled = q !== 'low';
  env.sun.shadow.mapSize.set(q === 'high' ? 4096 : 2048, q === 'high' ? 4096 : 2048);
  if (env.sun.shadow.map) { env.sun.shadow.map.dispose(); env.sun.shadow.map = null; }
  scene.traverse(o => { if (o.material) o.material.needsUpdate = true; });
  resize();
}

function go() {
  if (!car || car.variant !== state.variant) makeCar();
  ctl.assist = state.assist; car.opts.infiniteAir = state.infinite;
  if (!state.running || state.scenario !== state._lastScn || state._lastVar !== state.variant) startScenario(state.scenario);
  state._lastScn = state.scenario; state._lastVar = state.variant;
  state.running = true; state.paused = false;
  $('intro').classList.add('hidden'); $('hud').classList.remove('hidden');
  if (state.sound) { audio.start(); audio.setMuted(false); } else audio.setMuted(true);
}
$('go').onclick = go;
function openMenu() { state.paused = true; buildIntro(); $('go').textContent = state.running ? 'המשך ▶' : 'סעו! ▶'; $('intro').classList.remove('hidden'); }
$('btn-menu').onclick = openMenu;
$('btn-academy').onclick = () => { location.href = 'academy.html'; };
for (const a of document.querySelectorAll('a.help')) a.onclick = () => { location.href = 'academy.html#' + a.dataset.ch; };
$('result-again').onclick = () => { startScenario(state.scenario); state.paused = false; };
$('result-close').onclick = () => { $('result').classList.add('hidden'); state.paused = false; state.finished = true; };

// מגע
if ('ontouchstart' in window) {
  $('touch').classList.remove('hidden');
  for (const b of document.querySelectorAll('#touch button')) {
    const k = b.dataset.t;
    const set = (v) => {
      if (k === 'steerL') input.touch.steer = v ? 1 : 0;
      else if (k === 'steerR') input.touch.steer = v ? -1 : 0;
      else input.touch[k] = v ? 1 : 0;
    };
    b.addEventListener('touchstart', (e) => { e.preventDefault(); set(1); });
    b.addEventListener('touchend', (e) => { e.preventDefault(); set(0); });
  }
}

function resize() {
  renderer.setSize(container.clientWidth, container.clientHeight);
  camera.aspect = container.clientWidth / container.clientHeight; camera.updateProjectionMatrix();
}
addEventListener('resize', resize);

// ---------- מקשים חד-פעמיים ----------
function handleKeys() {
  const h = (c) => input.hit(c);
  if (h('Escape')) { if (!$('intro').classList.contains('hidden')) go(); else openMenu(); }
  if (state.paused) return;
  if (h('KeyC')) { const m = rig.cycle(); hud.toast({ chase: 'מצלמה עוקבת', cockpit: 'מבט מתוך הרכב', orbit: 'מצלמה חופשית: גררו בעכבר' }[m]); }
  if (h('KeyF')) { const order = ['auto', 'max', 'off']; ctl.fanMode = order[(order.indexOf(ctl.fanMode) + 1) % 3]; hud.toast(`מאווררים: ${{ auto: 'אוטומטי', max: 'Max Grip', off: 'כבויים' }[ctl.fanMode]}`); }
  if (h('KeyV')) { const order = ['low', 'high', 'road']; ctl.aeroMode = order[(order.indexOf(ctl.aeroMode) + 1) % 3]; hud.toast(`כנף: ${AERO.modes[ctl.aeroMode].nameHe}`); }
  if (h('KeyL')) { ctl.assist = !ctl.assist; state.assist = ctl.assist; store.set('assist', ctl.assist); hud.toast(ctl.assist ? 'ייצוב ובלימת נפילה: פעילים' : 'ייצוב כבוי: אתם שולטים בנחירים לבד'); }
  if (h('KeyT')) { ctl.tc = !ctl.tc; hud.toast(ctl.tc ? 'בקרת משיכה ו-ABS: פעילים' : 'בקרת משיכה כבויה'); }
  if (h('KeyG')) $('tele').classList.toggle('hidden');
  if (h('KeyJ')) {
    if (!car.thr.some(t => t.group === 'lift')) hud.toast('ריחוף דורש את דגם הרוקט');
    else if (!ctl.assist) hud.toast('החזקת גובה עובדת רק עם ייצוב (L)');
    else { ctl.hoverHold = !ctl.hoverHold; hud.toast(ctl.hoverHold ? 'החזקת גובה: פעילה. רווח לטיפוס, J לביטול' : 'החזקת גובה: כבויה'); }
  }
  if (h('KeyX')) { model.xrayOn = !model.xrayOn; model.setXray(model.xrayOn); hud.toast(model.xrayOn ? 'רנטגן: מכלי אוויר (אפור), סוללה (כחול), מנועים (כתום), מאווררים (ירוק)' : 'רנטגן כבוי', 3000); }
  if (h('KeyM')) { state.sound = !state.sound; audio.setMuted(!state.sound); }
  if (h('KeyR')) startScenario(state.scenario);
  if (h('KeyH')) { openMenu(); document.querySelector('.controls').open = true; }
  if (h('KeyK')) {
    if (!car.thr.some(t => t.group === 'lift')) hud.toast('קיקפליפ דורש את דגם הרוקט');
    else if (car.tank.m < 40) hud.toast('אין מספיק אוויר. סעו לתחנת האוויר (עיגול כחול)');
    else ctl.startKickflip(input.k('KeyA', 'ArrowLeft') ? -1 : 1);
  }
  for (let i = 1; i <= 8; i++) if (h('Digit' + i)) { const S = SCENARIOS.find(s => s.num === i); if (S) startScenario(S.key); }
}

// ---------- לולאה ----------
let last = performance.now(), screenT = 0, frames = 0, fpsT = 0;
function frame(now) {
  requestAnimationFrame(frame);
  let dt = Math.min(0.05, (now - last) / 1000); last = now;
  input.read();
  handleKeys();
  const raw = input.read();
  if (state.running && !state.paused) {
    const st = LAYOUT.station;
    const atStation = Math.hypot(car.pos.x - st.x, car.pos.z - st.z) < st.r;
    car._refilling = atStation && car.speed < 0.6 && car.tank.V > 0 && car.tank.p < car.P.fillPressure * 0.995;
    ctl.update(car, raw, dt, { altitude: car.altitude, atStation, guide: loopGuide });
    state.acc += dt;
    let n = 0;
    while (state.acc >= DT && n < 60) { car.step(DT, ctl.cmd); state.acc -= DT; n++; }
    // תרחיש
    const scn = scenarioByKey(state.scenario);
    let live = '';
    if (scn.update && !state.finished) {
      state.scnState.dt = dt;
      const r = scn.update(car, state.scnState);
      if (r?.live) live = r.live;
      if (r?.done) {
        state.finished = true;
        $('result-title').textContent = r.ok ? `${scn.icon} הצלחה!` : `${scn.icon} כמעט…`;
        $('result-body').innerHTML = r.html;
        $('result').classList.remove('hidden');
        audio.beep(r.ok ? 1320 : 440, 0.25);
        setTimeout(() => { if (!$('result').classList.contains('hidden')) state.paused = true; }, 1600);
      }
    }
    if (car._refilling) live = `ממלאים אוויר: ${(car.tank.p / 1e5).toFixed(0)} בר · ${car.tank.m.toFixed(0)} ק"ג${state.infinite ? '' : ' (מואץ פי 20 ממדחס אמיתי)'}`;
    if (car.tank.V > 0 && car.tank.m < 15 && !car._refilling) live = live || 'המכל כמעט ריק. סעו לתחנת האוויר (העיגול הכחול במפה)';
    // פגיעה קשה
    if (car.maxImpact > 14 && !state._crashT) { state._crashT = 2; hud.toast(`💥 פגיעה ב-${(car.maxImpact * 3.6).toFixed(0)} קמ"ש. R לאיפוס`, 2500); }
    if (state._crashT) { state._crashT -= dt; if (state._crashT <= 0) { state._crashT = 0; car.maxImpact = 0; } }
    // תקוע הפוך
    if (car.up.y < -0.2 && car.contacts === 0 && car.hullContact && car.speed < 1) { state._flipT = (state._flipT || 0) + dt; if (state._flipT > 2) { hud.toast('הרכב הפוך. R לאיפוס'); state._flipT = -5; } }
    else if (state._flipT > 0) state._flipT = 0;

    model.update(car, dt, rig.mode);
    fx.update(car, dt, world);
    audio.update(car, rig.mode, dt);
    car.events.length = 0;
    hud.update(car, ctl, rig, live, dt);
    screenT += dt; if (screenT > 0.1) { screenT = 0; hud.drawScreen(model, car, ctl); }
  }
  if (car) {
    rig.update(car, dt, world, model);
    // הצל עוקב אחרי הרכב
    const sd = env.sun.position.clone().sub(env.sun.target.position).normalize();
    env.sun.target.position.set(car.pos.x, car.pos.y, car.pos.z);
    env.sun.position.copy(env.sun.target.position).addScaledVector(sd, 200);
  }
  renderer.render(scene, camera);
  input.endFrame();
  frames++; fpsT += dt; if (fpsT > 1) { window.__fps = frames / fpsT; frames = 0; fpsT = 0; }
}

// ---------- התחלה ----------
makeCar();
buildIntro();
applyQuality();
{ const [x, z, yaw] = scenarioByKey(state.scenario).spawn; car.reset(x, z, yaw); }
rig.update(car, 0.016, world, model);
$('loading').classList.add('hidden');
if (params.get('cam')) rig.setMode(params.get('cam'));
if (params.get('auto')) go();
requestAnimationFrame(frame);
window.__app = { car: () => car, ctl: () => ctl, rig, scene, camera, renderer, model: () => model, state, startScenario, input, world, hud };
