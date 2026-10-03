// main.js — אתחול, לולאת רינדור, מעבר בין לשוניות ושליטה בזמן
import * as THREE from 'three';
import { createWorld, RE_KM, RM_KM } from './world.js';
import { $, $$ } from './util.js';
import { settings, setSetting, onSettings, clockStr } from './settings.js';
import { orbitsMode } from './mode_orbits.js';
import { compareMode } from './mode_compare.js';
import { launchMode } from './mode_launch.js';
import { missionsMode } from './mode_missions.js';
import { museumMode } from './mode_museum.js';
import { aboutMode } from './mode_about.js';

// מסך הטעינה: פס התקדמות והסבר על כל שלב
const LOAD_STEPS = {
  'earth_day': 'תמונת לוויין של כדור הארץ ביום (NASA Blue Marble)',
  'earth_night': 'אורות הערים בצד הלילה (NASA Black Marble)',
  'clouds': 'שכבת העננים',
  'moon': 'מפת פני הירח (LRO)',
};
const ldFill = $('#ld-fill'), ldMsg = $('#ld-msg'), ldSteps = $('#ld-steps');
const setProgress = (p, msg) => { ldFill.style.width = `${Math.round(p)}%`; if (msg) ldMsg.textContent = msg; };
const doneStep = text => { const li = document.createElement('li'); li.textContent = text; ldSteps.appendChild(li); };
doneStep('מנוע התלת־ממד (Three.js)');
setProgress(12, 'טוען תמונות של כדור הארץ והירח');
const labelFor = url => Object.entries(LOAD_STEPS).find(([k]) => url.includes(k))?.[1] ?? url;
THREE.DefaultLoadingManager.onProgress = (url, loaded, total) => {
  doneStep(labelFor(url));
  const next = Object.entries(LOAD_STEPS).map(([, v]) => v).find(v => ![...ldSteps.children].some(li => li.textContent === v));
  setProgress(12 + 70 * loaded / total, next ? 'טוען: ' + next : 'הכנת הסצנה');
};
let texturesReady = false;
THREE.DefaultLoadingManager.onLoad = () => { texturesReady = true; };
function finishLoading() {
  setProgress(100, 'מוכן');
  setTimeout(() => $('#loading').classList.add('done'), 300);
}

const world = createWorld($('#c'));
const panel = $('#panel-body');

const MODES = { orbits: orbitsMode, compare: compareMode, launch: launchMode, missions: missionsMode, museum: museumMode, about: aboutMode };
let mode = null, modeName = null;

// בקרת קצב הזמן; כל לשונית מגדירה אילו קצבים מוצעים
const warpEl = $('#warp');
world.setWarps = (list, current) => {
  warpEl.innerHTML = '';
  for (const w of list) {
    const b = document.createElement('button');
    b.textContent = w === 0 ? '❚❚' : (w >= 3600 ? `${w / 3600}h/s` : `${w}×`);
    b.dataset.w = w;
    b.onclick = () => { world.clock.rate = w; [...warpEl.children].forEach(c => c.classList.toggle('on', c === b)); };
    if (w === current) b.classList.add('on');
    warpEl.appendChild(b);
  }
  world.clock.rate = current;
};
world.timeDisplay = null; // לשונית יכולה להחליף את תצוגת השעון

// ---------- מבט נקי: מסתיר את כל הממשק ומשאיר רק את הסצנה ----------
let cleanTimer = 0;
function wakeCleanbar() {
  const bar = $('#cleanbar');
  bar.classList.remove('idle'); document.body.classList.remove('idle');
  clearTimeout(cleanTimer);
  cleanTimer = setTimeout(() => { bar.classList.add('idle'); document.body.classList.add('idle'); }, 2200);
}
function setClean(on) {
  document.body.classList.toggle('clean', on);
  $('#cleanbar').hidden = !on;
  $('#cleanGo').hidden = !mode?.launch;
  mode?.onClean?.(on);
  if (on) wakeCleanbar(); else { clearTimeout(cleanTimer); document.body.classList.remove('idle'); }
}
world.setClean = setClean;
$('#cleanExit').addEventListener('click', () => setClean(false));
$('#cleanBtn').addEventListener('click', () => setClean(true));
$('#cleanGo').addEventListener('click', () => mode?.launch?.());
window.addEventListener('keydown', e => { if (e.key === 'Escape' && document.body.classList.contains('clean')) setClean(false); });
for (const ev of ['pointermove', 'pointerdown', 'wheel']) window.addEventListener(ev, () => { if (document.body.classList.contains('clean')) wakeCleanbar(); }, { passive: true });

function setMode(name) {
  if (name === modeName) return;
  if (document.body.classList.contains('clean')) setClean(false);
  if (mode?.exit) mode.exit(world);
  modeName = name;
  mode = MODES[name];
  $$('#tabs button').forEach(b => b.classList.toggle('on', b.dataset.tab === name));
  panel.innerHTML = '';
  panel.scrollTop = 0;
  $('#hud').hidden = true;
  world.timeDisplay = null;
  mode.enter(world, panel);
  if (location.hash.slice(1).split('/')[0] !== name) history.replaceState(null, '', '#' + name);
}
world.setMode = setMode;
window.__app = { world, MODES };

$$('#tabs button[data-tab]').forEach(b => b.addEventListener('click', () => setMode(b.dataset.tab)));
$('#sheet-toggle').addEventListener('click', () => $('#panel').classList.toggle('collapsed'));

// ---------- הגדרות ----------
const SETTINGS_UI = [
  ['speedUnit', 'יחידות מהירות', [['kmh', 'קמ"ש'], ['kms', 'ק"מ לשנייה'], ['ms', 'מטר לשנייה']], 'דלתא־וי נשאר בק"מ לשנייה, היחידה המקובלת בתחום.'],
  ['timeZone', 'שעון', [['utc', 'UTC'], ['israel', 'שעון ישראל']]],
  ['satSize', 'גודל הלוויינים על המסך', [[0.5, 'קטן'], [1, 'בינוני'], [2, 'גדול']], 'הלוויינים תמיד מוגדלים, אחרת לא היו נראים.'],
  ['labels', 'שמות על הסצנה', [[true, 'להציג'], [false, 'להסתיר']]],
  ['clouds', 'עננים', [[true, 'להציג'], [false, 'להסתיר']]],
  ['quality', 'איכות גרפיקה', [['high', 'גבוהה'], ['saver', 'חסכונית']], 'במחשבים ובטלפונים חלשים, "חסכונית" חלקה יותר.'],
];
function renderSettings() {
  $('#settings').innerHTML = `<h3>הגדרות <button id="set-close" aria-label="סגור">✕</button></h3>` + SETTINGS_UI.map(([k, title, opts, note]) => `
    <div class="set"><span>${title}</span><div class="seg">${opts.map(([v, l]) => `<button data-k="${k}" data-v='${JSON.stringify(v)}' class="${settings[k] === v ? 'on' : ''}">${l}</button>`).join('')}</div>
    ${note ? `<span class="small" style="margin-top:4px">${note}</span>` : ''}</div>`).join('');
}
$('#gear').addEventListener('click', e => { e.stopPropagation(); renderSettings(); $('#settings').hidden = !$('#settings').hidden; });
$('#settings').addEventListener('click', e => {
  e.stopPropagation();
  if (e.target.closest('#set-close')) { $('#settings').hidden = true; return; }
  const b = e.target.closest('button[data-k]');
  if (!b) return;
  setSetting(b.dataset.k, JSON.parse(b.dataset.v));
  renderSettings();
});
document.addEventListener('click', () => { $('#settings').hidden = true; });
function applySettings() {
  $('#labels').style.display = settings.labels ? '' : 'none';
  world.clouds.visible = settings.clouds;
  world.renderer.setPixelRatio(settings.quality === 'saver' ? 1 : Math.min(window.devicePixelRatio, 2));
  resize();
}
onSettings(k => { applySettings(); mode?.onSettings?.(k); });

function resize() {
  const w = window.innerWidth, h = window.innerHeight;
  world.renderer.setSize(w, h, false);
  world.camera.aspect = w / h;
  world.camera.updateProjectionMatrix();
  mode?.resize?.(w, h);
}
window.addEventListener('resize', resize);
applySettings();

// כשהמצלמה מסתובבת סביב מרכז כדור הארץ (או הירח), קרוב לפני השטח כל מעלה של סיבוב
// מזיזה מאות קילומטרים, וכל צעד זום קופץ באחוז מהמרחק למרכז. לכן מקטינים את
// מהירויות הסיבוב, ההזזה והזום לפי הגובה מעל פני השטח, כך שהקרקע "נצמדת" לעכבר.
const ORIGIN = new THREE.Vector3();
function adaptControls() {
  const c = world.controls, cam = world.camera;
  let R = null, center = null;
  if (!mode?.scene && c.target.length() < RE_KM) { R = RE_KM; center = ORIGIN; }
  else if (!mode?.scene && c.target.distanceTo(world.moon.position) < RM_KM) { R = RM_KM; center = world.moon.position; }
  if (!R) { c.rotateSpeed = 1; c.panSpeed = 1; c.zoomSpeed = 1.2; return; }
  const d = cam.position.distanceTo(center), h = Math.max(d - R, 0.001);
  const x = h / R;
  c.rotateSpeed = Math.min(1, x / 6 + (x / 3) ** 2);
  c.panSpeed = c.rotateSpeed;
  c.zoomSpeed = Math.max(0.04, Math.min(1.2, 2 * h / d));
}

let last = performance.now();
function frame(now) {
  const dtReal = Math.min(0.1, (now - last) / 1000);
  last = now;
  const dtSim = world.clock.paused ? 0 : dtReal * world.clock.rate;
  world.clock.time += dtSim * 1000;
  world.updateCelestial();
  mode?.update?.(world, dtReal, dtSim);
  adaptControls();
  world.controls.update();
  const sc = mode?.scene ?? world.scene;
  const cam = mode?.camera ?? world.camera;
  cam.updateMatrixWorld();
  world.renderer.render(sc, cam);
  world.updateLabels(window.innerWidth, window.innerHeight);
  const d = new Date(world.clock.time);
  $('#clock').textContent = world.timeDisplay ? world.timeDisplay() : clockStr(d.getTime());
  requestAnimationFrame(frame);
}

requestAnimationFrame(frame);
// אחרי שהתמונות נטענו: בונים את הלשונית הראשונה (כולל סימולציית שיגור קצרה לחישוב ההפסדים)
// ומציגים רק אחרי שהפריים הראשון צויר
const waitTextures = setInterval(() => {
  if (!texturesReady) return;
  clearInterval(waitTextures);
  setProgress(88, 'מחשב סימולציית שיגור לתקציב הדלק ובונה את הסצנה');
  setTimeout(() => {
    const start = location.hash.slice(1).split('/')[0];
    setMode(MODES[start] ? start : 'orbits');
    doneStep('סימולציית שיגור ובניית הסצנה');
    requestAnimationFrame(() => requestAnimationFrame(finishLoading));
  }, 50);
}, 50);
// גיבוי: אם משהו נתקע, לא להשאיר את המשתמש מול מסך טעינה
setTimeout(() => { if (!modeName) setMode('orbits'); finishLoading(); }, 25000);
