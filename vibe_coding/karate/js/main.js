// main.js: סצנה, דוג'ו, מצלמה, קלט, ממשק וצליל.

import * as THREE from 'three';
import { solve, clonePose, solidify } from './body.js';
import { TECH, KAMAE, poseAt } from './techniques.js';
import { Match, ARENA } from './fight.js';
import { makeAI } from './ai.js';
import { Rig } from './rig.js';
import { buildKata, timeline, kataPoseAt, moveLabel, stanceLabel, moveHead } from './kata.js';
import { KATAS } from './katas.js';

const Q = new URLSearchParams(location.search);
const $ = (id) => document.getElementById(id);

// ---------- סצנה ----------
const renderer = new THREE.WebGLRenderer({ antialias: true, canvas: $('c') });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.05;
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x1a1512);
scene.fog = new THREE.Fog(0x1a1512, 14, 30);
const camera = new THREE.PerspectiveCamera(36, 1, 0.1, 60);

function canvasTex(w, h, draw) {
  const c = document.createElement('canvas'); c.width = w; c.height = h;
  draw(c.getContext('2d'), w, h);
  const t = new THREE.CanvasTexture(c); t.colorSpace = THREE.SRGBColorSpace; t.anisotropy = 8;
  return t;
}

function buildDojo() {
  // רצפת עץ
  const floorTex = canvasTex(1024, 1024, (g, w, h) => {
    g.fillStyle = '#b98a55'; g.fillRect(0, 0, w, h);
    const n = 16, pw = w / n;
    for (let i = 0; i < n; i++) {
      const v = 0.86 + Math.random() * 0.2;
      g.fillStyle = `rgb(${185 * v | 0},${138 * v | 0},${85 * v | 0})`;
      g.fillRect(i * pw, 0, pw, h);
      for (let k = 0; k < 26; k++) {
        g.strokeStyle = `rgba(90,58,28,${0.05 + Math.random() * 0.09})`; g.lineWidth = 1 + Math.random();
        const x = i * pw + Math.random() * pw;
        g.beginPath(); g.moveTo(x, 0); g.bezierCurveTo(x + 5, h * 0.3, x - 5, h * 0.6, x + 2, h); g.stroke();
      }
      g.fillStyle = 'rgba(50,30,14,0.55)'; g.fillRect(i * pw, 0, 2, h);
      const off = Math.random() * h;
      g.fillRect(i * pw, off, pw, 2); g.fillRect(i * pw, (off + h * 0.47) % h, pw, 2);
    }
  });
  floorTex.wrapS = floorTex.wrapT = THREE.RepeatWrapping; floorTex.repeat.set(7, 4);
  const floor = new THREE.Mesh(new THREE.PlaneGeometry(22, 14), new THREE.MeshStandardMaterial({ map: floorTex, roughness: 0.55, metalness: 0.02 }));
  floor.rotation.x = -Math.PI / 2; floor.receiveShadow = true; scene.add(floor);

  // גבולות השיאי-ג'ו
  const tape = new THREE.MeshBasicMaterial({ color: 0xf2efe6, transparent: true, opacity: 0.75 });
  const line = (w, d, x, z) => { const m = new THREE.Mesh(new THREE.PlaneGeometry(w, d), tape); m.rotation.x = -Math.PI / 2; m.position.set(x, 0.004, z); scene.add(m); };
  const A = ARENA + 0.5;
  line(2 * A, 0.05, 0, -3.2); line(2 * A, 0.05, 0, 3.2); line(0.05, 6.4, -A, 0); line(0.05, 6.4, A, 0);
  line(0.05, 0.6, -1.2, 0); line(0.05, 0.6, 1.2, 0);

  // קירות
  const wallMat = new THREE.MeshStandardMaterial({ color: 0xe6dcc6, roughness: 0.95 });
  const wood = new THREE.MeshStandardMaterial({ color: 0x4a301c, roughness: 0.7 });
  const box = (w, h, d, mat, x, y, z) => { const m = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), mat); m.position.set(x, y, z); m.receiveShadow = true; scene.add(m); return m; };
  box(22, 4.4, 0.2, wallMat, 0, 2.2, -6.6);
  box(0.2, 4.4, 14, wallMat, -11, 2.2, 0); box(0.2, 4.4, 14, wallMat, 11, 2.2, 0);
  box(22, 0.9, 0.24, wood, 0, 0.45, -6.56);          // לוח עץ תחתון
  box(22, 0.22, 0.3, wood, 0, 3.5, -6.5);            // קורה
  for (let x = -10; x <= 10; x += 4) box(0.2, 4.4, 0.3, wood, x, 2.2, -6.5);

  // חלונות שוג'י
  const shoji = canvasTex(256, 384, (g, w, h) => {
    g.fillStyle = '#f7efd9'; g.fillRect(0, 0, w, h);
    g.strokeStyle = '#3d2817'; g.lineWidth = 7;
    for (let i = 0; i <= 4; i++) { g.beginPath(); g.moveTo(i * w / 4, 0); g.lineTo(i * w / 4, h); g.stroke(); }
    for (let j = 0; j <= 6; j++) { g.beginPath(); g.moveTo(0, j * h / 6); g.lineTo(w, j * h / 6); g.stroke(); }
  });
  for (const x of [-8, -4, 4, 8]) {
    const m = new THREE.Mesh(new THREE.PlaneGeometry(3.4, 2.3), new THREE.MeshStandardMaterial({ map: shoji, emissive: 0xfff0cc, emissiveMap: shoji, emissiveIntensity: 0.55, roughness: 1 }));
    m.position.set(x, 2.15, -6.48); scene.add(m);
  }
  // שומן: מגילת קליגרפיה
  const scroll = canvasTex(256, 640, (g, w, h) => {
    g.fillStyle = '#efe6cf'; g.fillRect(0, 0, w, h);
    g.fillStyle = '#8a1f1a'; g.fillRect(0, 0, w, 26); g.fillRect(0, h - 26, w, 26);
    g.fillStyle = '#17120e'; g.textAlign = 'center'; g.textBaseline = 'middle';
    g.font = '150px "Noto Serif CJK JP","Noto Sans CJK JP","Yu Mincho","Hiragino Mincho ProN",serif';
    ['松', '濤', '館'].forEach((ch, i) => g.fillText(ch, w / 2, 130 + i * 185));
    g.fillStyle = '#b3261e'; g.fillRect(w - 62, h - 96, 34, 34);
  });
  const sc = new THREE.Mesh(new THREE.PlaneGeometry(0.9, 2.25), new THREE.MeshStandardMaterial({ map: scroll, roughness: 0.9 }));
  sc.position.set(0, 2.2, -6.46); scene.add(sc);
  box(1.05, 0.06, 0.06, wood, 0, 3.36, -6.44); box(1.05, 0.06, 0.06, wood, 0, 1.05, -6.44);

  // תאורה
  scene.add(new THREE.HemisphereLight(0xfff3dc, 0x5c4630, 1.15));
  const sun = new THREE.DirectionalLight(0xfff1d6, 2.3);
  sun.position.set(3.5, 8, 5.5); sun.castShadow = true;
  sun.shadow.mapSize.set(2048, 2048);
  const sc2 = sun.shadow.camera; sc2.left = -7; sc2.right = 7; sc2.top = 5; sc2.bottom = -5; sc2.near = 1; sc2.far = 22;
  sun.shadow.bias = -0.0004; sun.shadow.normalBias = 0.02;
  scene.add(sun);
  const fill = new THREE.DirectionalLight(0xcfe0ff, 0.5); fill.position.set(-5, 4, -3); scene.add(fill);
}
buildDojo();

// ---------- לוחמים ----------
const match = new Match();
const rigs = [
  new Rig({ skin: 0xd8a076, hair: 0x15100c, ribbon: 0xc8281e }),
  new Rig({ skin: 0xc68a5e, hair: 0x2a1c12, ribbon: 0xffffff }),
];
rigs[1].group.rotation.y = Math.PI;
for (const r of rigs) scene.add(r.group);

// ניצוץ פגיעה
const sparks = [];
function spark(at, color, size) {
  const m = new THREE.Mesh(new THREE.SphereGeometry(1, 12, 8), new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.9, depthWrite: false }));
  m.position.set(at[0], at[1], at[2]); m.scale.setScalar(0.02);
  scene.add(m); sparks.push({ m, t: 0, size });
}

// ---------- צליל ----------
let actx = null, muted = false;
function audio() {
  if (!actx) { try { actx = new (window.AudioContext || window.webkitAudioContext)(); } catch (e) { /* אין צליל */ } }
  if (actx && actx.state === 'suspended') actx.resume();
  return actx;
}
function noise(dur) {
  const n = actx.sampleRate * dur | 0, b = actx.createBuffer(1, n, actx.sampleRate), d = b.getChannelData(0);
  for (let i = 0; i < n; i++) d[i] = Math.random() * 2 - 1;
  const s = actx.createBufferSource(); s.buffer = b; return s;
}
function env(g, t, peak, dur, attack = 0.005) {
  g.gain.setValueAtTime(0.0001, t); g.gain.exponentialRampToValueAtTime(peak, t + attack);
  g.gain.exponentialRampToValueAtTime(0.0001, t + dur);
}
const SND = {
  whoosh(v = 0.12) {
    const t = actx.currentTime, s = noise(0.2), f = actx.createBiquadFilter(), g = actx.createGain();
    f.type = 'bandpass'; f.Q.value = 1.2; f.frequency.setValueAtTime(500, t); f.frequency.exponentialRampToValueAtTime(2600, t + 0.14);
    env(g, t, v, 0.19, 0.06); s.connect(f).connect(g).connect(actx.destination); s.start(t);
  },
  hit(strong) {
    const t = actx.currentTime, o = actx.createOscillator(), g = actx.createGain();
    o.frequency.setValueAtTime(strong ? 150 : 190, t); o.frequency.exponentialRampToValueAtTime(45, t + 0.16);
    env(g, t, strong ? 0.9 : 0.6, 0.22); o.connect(g).connect(actx.destination); o.start(t); o.stop(t + 0.25);
    const s = noise(0.1), f = actx.createBiquadFilter(), g2 = actx.createGain();
    f.type = 'lowpass'; f.frequency.value = 1800; env(g2, t, 0.5, 0.09, 0.002);
    s.connect(f).connect(g2).connect(actx.destination); s.start(t);
  },
  block() {
    const t = actx.currentTime, o = actx.createOscillator(), g = actx.createGain();
    o.type = 'triangle'; o.frequency.setValueAtTime(420, t); o.frequency.exponentialRampToValueAtTime(200, t + 0.06);
    env(g, t, 0.35, 0.09, 0.002); o.connect(g).connect(actx.destination); o.start(t); o.stop(t + 0.1);
    const s = noise(0.05), f = actx.createBiquadFilter(), g2 = actx.createGain();
    f.type = 'highpass'; f.frequency.value = 1500; env(g2, t, 0.25, 0.05, 0.002);
    s.connect(f).connect(g2).connect(actx.destination); s.start(t);
  },
  kiai(pitch = 1) {
    // קיאי: צעקה קצרה, מסור דרך שני פורמנטים
    const t = actx.currentTime, o = actx.createOscillator(), g = actx.createGain();
    o.type = 'sawtooth'; o.frequency.setValueAtTime(250 * pitch, t); o.frequency.linearRampToValueAtTime(185 * pitch, t + 0.3);
    env(g, t, 0.35, 0.36, 0.02);
    for (const [fq, q] of [[760, 6], [1250, 8]]) {
      const f = actx.createBiquadFilter(); f.type = 'bandpass'; f.frequency.setValueAtTime(fq, t); f.frequency.linearRampToValueAtTime(fq * 0.8, t + 0.3); f.Q.value = q;
      o.connect(f).connect(g);
    }
    g.connect(actx.destination); o.start(t); o.stop(t + 0.4);
  },
  kime(strong) {
    // נשיפה חדה מהבטן וחבטת הגי
    const t = actx.currentTime;
    const s = noise(0.12), f = actx.createBiquadFilter(), g = actx.createGain();
    f.type = 'bandpass'; f.frequency.setValueAtTime(3200, t); f.frequency.exponentialRampToValueAtTime(1500, t + 0.09); f.Q.value = 0.9;
    env(g, t, strong ? 0.30 : 0.18, 0.10, 0.004); s.connect(f).connect(g).connect(actx.destination); s.start(t);
    const s2 = noise(0.03), f2 = actx.createBiquadFilter(), g2 = actx.createGain();
    f2.type = 'highpass'; f2.frequency.value = 2500; env(g2, t, strong ? 0.35 : 0.2, 0.03, 0.001);
    s2.connect(f2).connect(g2).connect(actx.destination); s2.start(t);
  },
  stomp() {
    const t = actx.currentTime, o = actx.createOscillator(), g = actx.createGain();
    o.frequency.setValueAtTime(95, t); o.frequency.exponentialRampToValueAtTime(38, t + 0.12);
    env(g, t, 0.55, 0.16, 0.003); o.connect(g).connect(actx.destination); o.start(t); o.stop(t + 0.2);
  },
  clap() {
    const t = actx.currentTime;
    for (const dt of [0, 0.012]) { const s = noise(0.06), f = actx.createBiquadFilter(), g = actx.createGain(); f.type = 'bandpass'; f.frequency.value = 1400; f.Q.value = 0.8; env(g, t + dt, 0.5, 0.06, 0.001); s.connect(f).connect(g).connect(actx.destination); s.start(t + dt); }
  },
};
function play(name, ...a) { if (muted || !audio()) return; try { SND[name](...a); } catch (e) { /* מתעלמים */ } }

// ---------- קלט ----------
const keys = new Set();
const pressed = [[], []];
// פעולה לכל מקש: [שחקן, סוג, ערך]
const MAP1 = { KeyF: 'kizami', KeyG: 'gyaku', KeyH: 'oi', KeyR: 'mae', KeyT: 'mawashi', KeyY: 'yoko', KeyV: 'ushiro', KeyC: 'block', KeyX: 'shuto' };
const MAP2 = { KeyJ: 'kizami', KeyK: 'gyaku', KeyL: 'oi', KeyU: 'mae', KeyI: 'mawashi', KeyO: 'yoko', KeyP: 'ushiro', KeyM: 'block', KeyN: 'shuto' };
let mode = 'cpu', difficulty = 1, ai = [null, makeAI(1)], started = false, paused = false;

function keyDown(code) {
  if (keys.has(code)) return;
  keys.add(code);
  if (MAP1[code]) pressed[0].push(MAP1[code]);
  if (MAP2[code]) pressed[mode === '2p' ? 1 : 0].push(MAP2[code]);
}
addEventListener('keydown', (e) => {
  if (kata.on) { kataKey(e); return; }
  if (e.code === 'Escape') { togglePause(); return; }
  if (e.code.startsWith('Arrow') || e.code === 'Space') e.preventDefault();
  if (!e.repeat) keyDown(e.code);
});
addEventListener('keyup', (e) => keys.delete(e.code));
addEventListener('blur', () => keys.clear());

function readInput(p) {
  const two = mode === '2p';
  const k = (c) => keys.has(c);
  let fwd, back, up, down, hold;
  if (p === 0) {
    fwd = k('KeyD') || (!two && k('ArrowRight')); back = k('KeyA') || (!two && k('ArrowLeft'));
    up = k('KeyW') || (!two && k('ArrowUp')); down = k('KeyS') || (!two && k('ArrowDown'));
    hold = k('KeyC') || k('KeyX') || (!two && (k('KeyM') || k('KeyN')));
  } else {
    fwd = k('ArrowLeft'); back = k('ArrowRight'); up = k('ArrowUp'); down = k('ArrowDown');
    hold = k('KeyM') || k('KeyN');
  }
  return { move: fwd ? 1 : back ? -1 : 0, level: up ? 'up' : down ? 'down' : 'mid', actions: pressed[p].splice(0), hold };
}

// כפתורי מגע
for (const b of document.querySelectorAll('[data-k]')) {
  const code = b.dataset.k;
  const on = (e) => { e.preventDefault(); audio(); b.classList.add('on'); keyDown(code); };
  const off = (e) => { e.preventDefault(); b.classList.remove('on'); keys.delete(code); };
  b.addEventListener('pointerdown', on); b.addEventListener('pointerup', off);
  b.addEventListener('pointercancel', off); b.addEventListener('pointerleave', off);
  b.addEventListener('contextmenu', (e) => e.preventDefault());
}
if (matchMedia('(pointer: coarse)').matches) document.body.classList.add('touch');

// ---------- מצלמה ----------
let az = 0, userAz = 0, zoom = 1, camX = 0, shake = 0, camDist = 6;
{
  let drag = null;
  const c = $('c');
  c.addEventListener('pointerdown', (e) => { drag = { x: e.clientX, a: userAz }; c.setPointerCapture(e.pointerId); });
  c.addEventListener('pointermove', (e) => { if (!drag) return; const a = drag.a - (e.clientX - drag.x) * 0.006; userAz = kata.on ? a : Math.max(-1.3, Math.min(1.3, a)); });
  c.addEventListener('pointerup', () => { drag = null; });
  c.addEventListener('wheel', (e) => { zoom = Math.max(0.6, Math.min(1.8, zoom * (e.deltaY > 0 ? 1.08 : 0.93))); e.preventDefault(); }, { passive: false });
  c.addEventListener('dblclick', () => { userAz = 0; zoom = 1; });
}
function resize() {
  const w = innerWidth, h = innerHeight;
  renderer.setSize(w, h, false); camera.aspect = w / h; camera.updateProjectionMatrix();
}
addEventListener('resize', resize); resize();

function updateCamera(dt, time) {
  const [a, b] = match.f;
  const mid = (a.x + b.x) / 2, D = Math.abs(b.x - a.x);
  camX += (mid - camX) * Math.min(1, dt * 4);
  const narrow = camera.aspect < 1 ? 1.0 / camera.aspect : 1;
  const want = (4.3 + D * 0.95) * zoom * Math.min(narrow, 2.4);
  camDist += (want - camDist) * Math.min(1, dt * 3);
  az = userAz + Math.sin(time * 0.17) * 0.07;
  shake = Math.max(0, shake - dt * 4);
  const sx = (Math.random() - 0.5) * shake * 0.08, sy = (Math.random() - 0.5) * shake * 0.08;
  camera.position.set(camX + Math.sin(az) * camDist + sx, 1.45 + sy, Math.cos(az) * camDist);
  camera.lookAt(camX, 1.02, 0);
}

// ---------- ממשק ----------
const NAMES = ['אָקָא', 'שִׁירוֹ'];
const LV = { j: 'ג׳ודאן', c: 'צ׳ודאן' };
let bannerT = 0;
function banner(text, sub = '', dur = 1.3) {
  $('banner').innerHTML = `<b>${text}</b>${sub ? `<span>${sub}</span>` : ''}`;
  $('banner').classList.add('show'); bannerT = dur;
}
function techLabel(f) {
  const t = f.last;
  if (!t || f.lastAge > 1.3) return '';
  const lv = t.kind === 'attack' && TECH[t.id.replace(/_[cj]$/, '') + '_j'] ? ' ' + LV[t.lv] : '';
  return `${t.he}${lv} <i>${t.name}</i>`;
}
const ghost = [100, 100];
function updateHud(dt) {
  match.f.forEach((f, i) => {
    $('hp' + i).style.width = f.hp + '%';
    if (ghost[i] < f.hp) ghost[i] = f.hp;
    ghost[i] += (f.hp - ghost[i]) * Math.min(1, dt * 2.5);
    $('gh' + i).style.width = ghost[i] + '%';
    const l = techLabel(f), el = $('tech' + i);
    if (el.dataset.v !== l) { el.dataset.v = l; el.innerHTML = l; }
  });
  if (bannerT > 0) { bannerT -= dt; if (bannerT <= 0) $('banner').classList.remove('show'); }
}

function handleEvents() {
  for (const e of match.events) {
    if (e.type === 'rei') banner('רֵיי', 'rei · קידה', 2.4);
    else if (e.type === 'hajime') { banner('הַגִ׳ימֶה!', 'hajime', 1.0); play('clap'); }
    else if (e.type === 'kime') { play('kime', e.kind === 'attack'); if (e.id.startsWith('oi')) play('stomp'); }
    else if (e.type === 'hit') {
      play('hit', e.dmg >= 14); if (e.kiai || e.counter) play('kiai', e.who ? 0.88 : 1);
      spark(e.at, e.counter ? 0xffd27a : 0xffffff, 0.16 + e.dmg * 0.008);
      shake = Math.min(1, 0.35 + e.dmg * 0.03);
      if (e.counter) banner('דֶה־אַי!', 'deai · פגיעה בתוך התקפת היריב', 0.9);
      const v = $('bar' + (1 - e.who)); v.classList.remove('flash'); void v.offsetWidth; v.classList.add('flash');
    } else if (e.type === 'block') { play('block'); spark(e.at, 0x9fd0ff, 0.09); }
    else if (e.type === 'ko') banner('יָאמֶה!', 'yame', 1.6);
    else if (e.type === 'yame') {
      $('result').textContent = e.winner < 0 ? 'אָי־אוּצִ׳י: שניהם נפלו' : `הניצחון ל${NAMES[e.winner]}`;
      $('end').classList.add('show');
    }
  }
  match.events.length = 0;
}

// הפעלת צליל הנפה בתחילת התקפה
const seq = [0, 0];
function watchSwings() {
  match.f.forEach((f, i) => {
    if (f.techSeq !== seq[i]) { seq[i] = f.techSeq; if (f.tech && f.tech.kind === 'attack') play('whoosh', 0.07); }
  });
}

function startMatch() {
  audio();
  ai = [null, mode === 'cpu' ? makeAI(difficulty) : null];
  if (Q.get('auto')) ai[0] = makeAI(2);
  match.events.length = 0; match.restart();
  ghost[0] = ghost[1] = 100;
  $('menu').classList.remove('show'); $('end').classList.remove('show');
  started = true; paused = false;
}
function togglePause() {
  if (!started) return;
  paused = !paused;
  $('menu').classList.toggle('show', paused);
  $('start').textContent = paused ? 'קרב חדש' : 'הַגִ׳ימֶה';
  $('resume').style.display = paused ? '' : 'none';
}
$('start').onclick = startMatch;
$('resume').onclick = togglePause;
$('again').onclick = startMatch;
$('toMenu').onclick = () => { $('end').classList.remove('show'); paused = false; togglePause(); };
$('pause').onclick = togglePause;
$('mute').onclick = () => { muted = !muted; $('mute').textContent = muted ? '🔇' : '🔊'; };
for (const b of document.querySelectorAll('[data-mode]')) b.onclick = () => {
  mode = b.dataset.mode; if (b.dataset.diff) difficulty = +b.dataset.diff;
  document.querySelectorAll('[data-mode]').forEach((x) => x.classList.toggle('sel', x === b));
  document.body.classList.toggle('two', mode === '2p');
};
$('helpBtn').onclick = () => $('help').classList.toggle('show');


// ---------- מצב קאטה ----------
const kata = { on: false, def: null, K: null, TL: null, t: 0, playing: false, stepMode: false, stepIdx: -1, speed: 1, shown: -9, view: 0, pose: {}, pos: [0, 0], cam: [0, 0], kimeAge: 9 };
const VIEWS = [['מלפנים', 0], ['מאחור', Math.PI], ['מימין', -Math.PI / 2], ['משמאל', Math.PI / 2]];
const GROUPS = { heian: 'הייאן', tekki: 'טקי', adv: 'קאטות מתקדמות', basic: 'יסוד' };
function buildKataMenu() {
  const g = $('kGrid'); g.innerHTML = '';
  let last = null;
  for (const k of KATAS) {
    if (k.group !== last) { last = k.group; const h = document.createElement('h3'); h.textContent = GROUPS[k.group] || ''; if (h.textContent) g.appendChild(h); }
    const b = document.createElement('button');
    b.innerHTML = `<b>${k.he}</b><small>${k.ja} · ${k.mean || ''}</small>`;
    b.onclick = () => openKata(k.id);
    g.appendChild(b);
  }
}
function kataWorld(p) { return [kata.pos[0] - p[2], kata.pos[1] + p[0]]; }   // מערכת הקאטה -> עולם: חזית הקאטה אל המצלמה
function openKata(id) {
  const def = KATAS.find((k) => k.id === id); if (!def) return;
  audio();
  const k = kata; k.def = def; k.K = buildKata(def); k.TL = timeline(k.K);
  // ממרכזים את האמבוסן (תוואי הקאטה) בדוג'ו
  let x0 = 0, x1 = 0, z0 = 0, z1 = 0;
  for (const s of k.K.steps) { x0 = Math.min(x0, s.pose.pel[0]); x1 = Math.max(x1, s.pose.pel[0]); z0 = Math.min(z0, s.pose.pel[2]); z1 = Math.max(z1, s.pose.pel[2]); }
  k.pos = [(z0 + z1) / 2, -0.6 - (x0 + x1) / 2];
  k.t = 0; k.playing = false; k.stepIdx = -1; k.shown = -9; k.on = true; started = false; paused = false;
  k.cam = kataWorld(k.K.open.pel);
  document.body.classList.add('kata');
  for (const o of ['menu', 'end', 'kataMenu']) $(o).classList.remove('show');
  rigs[1].group.visible = false; rigs[0].group.rotation.y = -Math.PI / 2;
  $('kName').innerHTML = `${def.he}<small>${def.ja}</small>`;
  const L = $('kList'); L.innerHTML = '';
  k.K.steps.forEach((s, i) => {
    const d = document.createElement('div'), l = moveLabel(s), st = stanceLabel(s);
    if (s.m.sub) d.className = 'sub';
    if (s.m.via) d.style.display = 'none';
    d.innerHTML = `<em>${s.m.sub ? '' : s.count}</em><span>${l.he || '(מעבר)'}${s.m.kiai ? ' · קיאי' : ''}<small>${s.m.yame ? '' : st.he}</small></span>`;
    d.onclick = () => kataJump(i);
    L.appendChild(d);
  });
  const P = $('kProg'); P.querySelectorAll('i').forEach((x) => x.remove());
  for (const m of k.TL.marks) { const i = document.createElement('i'); i.style.left = (m.tk / k.TL.total * 100) + '%'; P.appendChild(i); }
  kataUI();
}
function closeKata(toKumite) {
  kata.on = false; kata.playing = false;
  document.body.classList.remove('kata', 'list');
  rigs[1].group.visible = true; rigs[0].group.rotation.y = 0;
  $(toKumite ? 'menu' : 'kataMenu').classList.add('show');
}
function kataJump(i) {
  const k = kata, n = k.K.steps.length; i = moveHead(k.K, Math.max(0, Math.min(n - 1, i)));
  k.t = k.TL.marks[i].t0 + 1e-4; k.stepIdx = i; k.playing = true; kataUI();
}
function kataCur() { return kataPoseAt(kata.K, kata.TL, kata.t, {}).i; }
function kataNext() {
  const k = kata, i = kataCur();
  if (k.stepMode) { if (k.t >= k.TL.marks[k.TL.marks.length - 1].t1 - 0.01) return; let j = k.t <= k.TL.marks[0].t0 && !k.playing ? 0 : i + 1; while (j < k.K.steps.length - 1 && k.K.steps[j].m.via && moveHead(k.K, j) <= i) j++; kataJump(j); }
  else kataJump(i + 1);
}
function kataPrev() { const i = moveHead(kata.K, Math.max(0, kataCur())); kataJump(kata.playing && kata.t - kata.TL.marks[i].t0 > 0.5 ? i : i - 1); }
function kataAgain() { kataJump(Math.max(0, kataCur())); }
function kataToggle() {
  const k = kata; audio();
  if (k.playing) { k.playing = false; }
  else {
    if (k.t >= k.TL.total - 0.05) { k.t = 0; k.stepIdx = -1; }
    if (k.stepMode) { kataNext(); return; }
    k.playing = true;
  }
  kataUI();
}
function kataUI() {
  const k = kata;
  $('kPlay').textContent = k.playing ? 'עצור' : (k.stepMode ? 'בצע' : (k.t >= k.TL.total - 0.05 ? 'מהתחלה' : 'נגן'));
  $('kStep').classList.toggle('sel', k.stepMode);
  $('kListBtn').classList.toggle('sel', document.body.classList.contains('list'));
  $('kView').textContent = VIEWS[k.view][0];
  $('kSpeedV').textContent = '×' + k.speed.toFixed(2).replace(/0$/, '');
}
function kataKey(e) {
  if (e.code === 'Space') { e.preventDefault(); kataToggle(); }
  else if (e.code === 'ArrowLeft' || e.code === 'KeyN') { e.preventDefault(); kataNext(); }
  else if (e.code === 'ArrowRight' || e.code === 'KeyP') { e.preventDefault(); kataPrev(); }
  else if (e.code === 'KeyR') kataAgain();
  else if (e.code === 'Escape') closeKata(false);
}
$('kataBtn').onclick = () => { $('menu').classList.remove('show'); buildKataMenu(); $('kataMenu').classList.add('show'); };
$('kToKumite').onclick = () => { $('kataMenu').classList.remove('show'); $('menu').classList.add('show'); };
$('kMenu').onclick = () => { closeKata(false); buildKataMenu(); };
$('kPlay').onclick = kataToggle; $('kNext').onclick = kataNext; $('kPrev').onclick = kataPrev; $('kAgain').onclick = kataAgain;
$('kStep').onclick = () => { kata.stepMode = !kata.stepMode; if (kata.stepMode && kata.playing) kata.stepIdx = Math.max(0, kataCur()); kataUI(); };
$('kSpeed').oninput = (e) => { kata.speed = +e.target.value; kataUI(); };
$('kView').onclick = () => { kata.view = (kata.view + 1) % VIEWS.length; userAz = 0; kataUI(); };
$('kListBtn').onclick = () => { document.body.classList.toggle('list'); kataUI(); };
$('kMute').onclick = () => { muted = !muted; $('kMute').textContent = $('mute').textContent = muted ? '🔇' : '🔊'; };
$('kProg').onclick = (e) => { const r = $('kProg').getBoundingClientRect(); const t = (e.clientX - r.left) / r.width * kata.TL.total; const i = kata.TL.marks.findIndex((m) => t < m.t1); kataJump(i < 0 ? kata.TL.marks.length - 1 : i); };

let kAz = 0;
function kataFrame(dt) {
  const k = kata, marks = k.TL.marks, n = marks.length;
  if (k.playing) {
    const t0 = k.t; k.t += dt * k.speed;
    for (let i = 0; i < n; i++) {
      if (t0 < marks[i].tk && k.t >= marks[i].tk) {
        const m = k.K.steps[i].m;
        if (!m.soft) { play('kime', true); if (m.stomp) play('stomp'); }
        if (m.kiai) play('kiai');
      }
    }
    while (k.stepMode && k.stepIdx >= 0 && k.stepIdx < n - 1 && k.K.steps[k.stepIdx].m.via && k.t >= marks[k.stepIdx].t1) k.stepIdx++;
    if (k.stepMode && k.stepIdx >= 0 && k.t >= marks[k.stepIdx].t1 - 0.002) { k.t = marks[k.stepIdx].t1 - 0.002; k.playing = false; kataUI(); }
    if (k.t >= k.TL.total) { k.t = k.TL.total; k.playing = false; kataUI(); }
  }
  const r = kataPoseAt(k.K, k.TL, k.t, k.pose);
  if (r.i !== k.shown) {
    k.shown = r.i;
    if (r.i < 0) { $('kMove').innerHTML = 'יוֹי'; $('kJa').textContent = 'yoi'; $('kStance').textContent = stanceLabel({ m: { st: k.def.start || 'hachiji' } }).he; }
    else {
      const s = k.K.steps[r.i], l = moveLabel(s), st = stanceLabel(s);
      $('kMove').innerHTML = `${s.m.yame ? '' : `<em>${s.count}/${k.K.count}</em>`}${l.he || '(מעבר)'}${s.m.kiai ? '<u>קיאי!</u>' : ''}`;
      $('kJa').textContent = l.ja || '';
      $('kStance').textContent = (s.m.yame ? '' : st.he) + (s.m.note ? (s.m.yame ? '' : ' · ') + s.m.note : '');
    }
    const L = $('kList').children;
    const hi = r.i < 0 ? -1 : (() => { let j = r.i; while (k.K.steps[j].m.via && j < L.length - 1) j++; return j; })();
    for (let i = 0; i < L.length; i++) L[i].classList.toggle('cur', i === hi);
    if (hi >= 0 && document.body.classList.contains('list')) L[hi].scrollIntoView({ block: 'nearest' });
  }
  $('kFill').style.width = (k.t / k.TL.total * 100) + '%';
  // קימה: שקיעה ורעד קצר, כמו בקומיטה
  const p = clonePose(r.pose), a = r.kimeAge, soft = r.i >= 0 && k.K.steps[r.i].m.soft;
  if (a < 0.4 && !soft) {
    const lock = Math.exp(-a / 0.11), ring = Math.exp(-a / 0.05) * Math.cos(a * 70);
    p.pel[1] -= 0.014 * lock; p.rot[1] += 0.02 * ring; p.hL[1] += 0.006 * ring; p.hR[1] += 0.006 * ring;
  } else if (!k.playing || a > 0.4) { p.pel[1] += Math.sin(time * 1.9) * 0.004; }
  rigs[0].update(solve(solidify(p, 4, 0.12)));
  rigs[0].group.position.set(k.pos[0], 0, k.pos[1]);
  const w = kataWorld(r.pose.pel), f = Math.min(1, dt * 3.5);
  k.cam[0] += (w[0] - k.cam[0]) * f; k.cam[1] += (w[1] - k.cam[1]) * f;
  let want = VIEWS[k.view][1] + userAz; kAz += (want - kAz) * Math.min(1, dt * 6);
  const narrow = camera.aspect < 1 ? Math.min(1 / camera.aspect, 2.2) : 1;
  const d = 5.0 * zoom * (0.55 + 0.45 * narrow);
  camera.position.set(k.cam[0] + Math.sin(kAz) * d, 1.5, k.cam[1] + Math.cos(kAz) * d);
  camera.lookAt(k.cam[0], camera.aspect < 1 ? 0.78 : 0.95, k.cam[1]);
  renderer.render(scene, camera);
}
window.__kata = kata;

// שכבת חיים מעל התנוחה המחושבת. לא משפיעה על הפגיעות, רק על מה שרואים.
// בקמאה: נשימה מהבטן, לחץ קל קדימה ואחורה. ברגע הקימה: כל הגוף ננעל ושוקע, עם רעד קצר שדועך.
const tmpPose = [null, null];
function alive(f, i, time) {
  const p = tmpPose[i] = clonePose(f.cur);
  if (f.down) return p;
  const ph = i * 2.1;
  if (!f.tech) {
    const br = Math.sin(time * 1.9 + ph);              // נשימה
    const sw = Math.sin(time * 0.83 + ph * 1.7);        // העברת משקל איטית
    p.pel[1] += br * 0.006 - 0.004;
    p.pel[0] += sw * 0.012;
    p.rot[1] += br * 0.006;
    p.hL[1] += br * 0.007; p.hR[1] += br * 0.009;
    p.hL[0] += Math.sin(time * 1.3 + ph) * 0.008;
  }
  const a = f.kimeAge;
  if (a < 0.4) {
    const lock = Math.exp(-a / 0.11);                   // שקיעה אל תוך העמידה
    const ring = Math.exp(-a / 0.05) * Math.cos(a * 70); // רעד הגי והשרירים
    p.pel[1] -= 0.016 * lock;
    p.rot[1] += 0.022 * ring;
    p.tw += 0.03 * ring * (f.mir ? -1 : 1);
    p.head[1] -= 0.03 * lock;                            // הסנטר נאסף
    p.hL[1] += 0.006 * ring; p.hR[1] += 0.006 * ring;
  }
  return p;
}

// ---------- לולאה ----------
const DT = 1 / 120;
let acc = 0, last = performance.now(), time = 0, frames = 0, fpsT = 0;
const debugPose = Q.get('pose');

function frame(now) {
  requestAnimationFrame(frame);
  const dt = Math.min(0.05, (now - last) / 1000); last = now; time += dt;
  frames++; fpsT += dt; if (fpsT > 1) { window.__fps = Math.round(frames / fpsT); frames = 0; fpsT = 0; }

  if (debugPose) {
    // מצב בדיקה: ?pose=gyaku_c&t=0.16&mir=1 מקפיא טכניקה לצילום
    const f = match.f[0], tk = TECH[debugPose];
    const mir = Q.get('mir') === '1';
    const t = Q.has('t') ? +Q.get('t') : (time * (+Q.get('speed') || 0.5)) % (tk.dur + 0.4);
    const from = tk.kind === 'cine' ? tk.keys[0].pose : KAMAE;
    f.cur = poseAt(tk, Math.min(t, tk.dur), from, mir);
    rigs[0].update(solve(f.cur)); rigs[0].group.position.set(f.x, 0, 0);
    rigs[1].group.visible = Q.get('opp') === '1'; match.f[1].x = f.x + 0.8;
    if (rigs[1].group.visible) { const o = match.f[1]; o.x = f.x + (+Q.get('d') || 1.2); o.cur = KAMAE; rigs[1].update(solve(KAMAE)); rigs[1].group.position.set(o.x, 0, 0); }
    userAz = Q.has('az') ? +Q.get('az') : userAz; zoom = +Q.get('zoom') || 0.75;
    updateCamera(dt, 0);
    renderer.render(scene, camera);
    return;
  }

  if (kata.on) { kataFrame(dt); return; }

  if (started && !paused) {
    acc += dt;
    while (acc >= DT) {
      const inputs = [0, 1].map((i) => (ai[i] ? ai[i](match, match.f[i], match.f[1 - i], DT) : readInput(i)));
      match.step(DT, inputs);
      watchSwings(); handleEvents();
      acc -= DT;
    }
  }
  match.f.forEach((f, i) => {
    rigs[i].update(solve(alive(f, i, time))); rigs[i].group.position.set(f.x, 0, 0);
  });
  for (let i = sparks.length - 1; i >= 0; i--) {
    const s = sparks[i]; s.t += dt;
    const u = s.t / 0.22;
    s.m.scale.setScalar(s.size * (0.3 + u)); s.m.material.opacity = 0.9 * (1 - u);
    if (u >= 1) { scene.remove(s.m); s.m.geometry.dispose(); s.m.material.dispose(); sparks.splice(i, 1); }
  }
  updateCamera(dt, time);
  updateHud(dt);
  renderer.render(scene, camera);
}
window.__match = match;
if (debugPose) { document.body.classList.add('debug'); for (const f of match.f) { f.tech = null; } }
if (Q.get('auto')) startMatch();
if (Q.get('kata')) {
  // ?kata=heian1 פותח קאטה; &step=7 מקפיא בשיא התנועה השביעית (&ph=prep להכנה), &view=0..3, &az=, &zoom=, &play=1
  openKata(Q.get('kata'));
  if (Q.has('view')) kata.view = +Q.get('view');
  if (Q.has('az')) userAz = +Q.get('az');
  if (Q.has('zoom')) zoom = +Q.get('zoom');
  if (Q.has('step')) { const m = kata.TL.marks[+Q.get('step') - 1]; kata.t = Q.get('ph') === 'prep' ? m.tp : m.tk + 0.6 * (m.t1 - m.tk) * 0 + 0.001; kAz = VIEWS[kata.view][1] + userAz; kata.cam = kataWorld(kataPoseAt(kata.K, kata.TL, kata.t, {}).pose.pel); }
  if (Q.get('ui') === '0') document.body.classList.add('debug');
  if (Q.get('play')) kata.playing = true;
}
requestAnimationFrame(frame);
