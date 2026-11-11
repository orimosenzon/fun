// Bird-Man: wiring. World, renderer, camera, HUD and the loop.
import * as THREE from '../lib/three.module.min.js';
import { initWorld, groundAt, POI } from './sim/world.js';
import { Bird } from './sim/bird.js';
import { clamp, sstep, qrot } from './sim/vmath.js';
import { U } from './render/env.js';
import { buildSky, buildTerrain, buildWater } from './render/land.js';
import { buildTrees, buildProps, buildClouds, buildBirds, buildMarkers } from './render/scenery.js';
import { buildBirdModel } from './render/birdModel.js';
import { buildFx } from './render/fx.js';
import { Game, PROLOGUE } from './story.js';
import { Input } from './ui/input.js';
import { Audio } from './ui/audio.js';
import { pilot } from './autopilot.js';

const $ = (id) => document.getElementById(id);
const params = new URLSearchParams(location.search);
const canvas = $('c');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, powerPreference: 'high-performance' });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(62, 1, 0.3, 40000);
function resize() { renderer.setSize(innerWidth, innerHeight, false); camera.aspect = innerWidth / innerHeight; camera.updateProjectionMatrix(); }
addEventListener('resize', resize); resize();

initWorld();
buildSky(scene); buildTerrain(scene); buildWater(scene);
const trees = buildTrees(scene), props = buildProps(scene); buildClouds(scene);
const flock = buildBirds(scene), markers = buildMarkers(scene);
const bird = new Bird(), model = buildBirdModel(scene), fx = buildFx(scene);
const input = new Input(), audio = new Audio();
const isTouch = matchMedia('(pointer: coarse)').matches;
if (isTouch) document.body.classList.add('touch');
input.bindTouch($('stick'), $('knob'), $('bflap'), $('btuck'));

// ---------- UI ----------
let mode = 'menu', time = 0, flashT = 0, titleT = 0, punch = 0, slow = 0, white = 0, cardI = 0, curMarkers = [];
const show = (id, on) => $(id).classList.toggle('on', on);
const ui = {
  markers(list) { curMarkers = list; markers.set(list.map((s, i) => ({ ...s, dir: list[i + 1] ? [list[i + 1].p[0] - s.p[0], list[i + 1].p[1] - s.p[1], list[i + 1].p[2] - s.p[2]] : list[i - 1] ? [s.p[0] - list[i - 1].p[0], s.p[1] - list[i - 1].p[1], s.p[2] - list[i - 1].p[2]] : [1, 0, 0] }))); },
  title(t, sub) { $('titlecard').innerHTML = `<small>${sub}</small><b>${t}</b>`; show('titlecard', true); titleT = 3.2; $('chap').firstChild.textContent = t; },
  objective(html, big) { $('obj').innerHTML = html; $('obj').classList.toggle('big', !!big); },
  flash(t, sub) { $('flash').innerHTML = `${t}<small>${sub || ''}</small>`; show('flash', true); flashT = 2.4; },
  chime(last) { audio.chime(last); },
  respawned() { camSnap = true; fx.reset(model); },
  chapterDone(g) {
    const ch = g.chapter, last = g.ci >= g.chapters.length - 2;
    $('done').querySelector('h2').textContent = ch.title; $('done').querySelector('p').textContent = ch.done || '';
    $('done').querySelector('.time').textContent = ch.timed ? `זמן: ${fmt(g.time)}   שיא: ${fmt(g.best[g.ci])}` : '';
    $('bnext').textContent = last ? 'טיסה חופשית' : 'הפרק הבא';
    setMode('done');
  },
};
const fmt = (s) => (s == null ? '' : `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, '0')}`);
const game = new Game(bird, ui);
{ const c = game.chapters[0].start; bird.placeGround(c.x, c.z, c.heading); }

function setMode(m) {
  mode = m;
  show('menu', m === 'menu'); show('card', m === 'card'); show('done', m === 'done'); show('pause', m === 'pause');
  show('hud', m === 'play'); show('touch', m === 'play');
  if (m === 'menu') buildMenu();
}
function buildMenu() {
  const box = $('chapters'); box.innerHTML = '';
  $('bstart').textContent = game.unlocked > 0 ? 'המשך את הסיפור' : 'התחל את הסיפור';
  game.chapters.forEach((ch, i) => {
    const b = document.createElement('button'); b.innerHTML = `<b>${ch.title}</b>${ch.sub}`;
    b.disabled = !(ch.free || i <= game.unlocked); b.onclick = () => { audio.start(); audio.click(); begin(i); };
    box.appendChild(b);
  });
}
function begin(i) { game.start(i); setMode('play'); camSnap = true; }
function nextCard() {
  cardI++;
  if (cardI >= PROLOGUE.length) { begin(0); return; }
  $('card').querySelector('p').textContent = PROLOGUE[cardI];
  if (cardI === PROLOGUE.length - 1) { game.start(0); game.state = 'idle'; show('titlecard', false); titleT = 0; }
}
$('bstart').onclick = () => {
  audio.start(); audio.click();
  if (game.unlocked > 0) return begin(Math.min(game.unlocked, game.chapters.length - 1));
  cardI = -1; setMode('card'); nextCard();
};
$('card').onclick = () => { audio.click(); nextCard(); };
$('bnext').onclick = () => begin(Math.min(game.ci + 1, game.chapters.length - 1));
$('bagain').onclick = () => begin(game.ci);
$('bfly').onclick = () => { game.state = 'free'; ui.objective('', false); markers.set([]); curMarkers = []; setMode('play'); };
$('bmenu').onclick = $('bmenu2').onclick = () => setMode('menu');
$('bresume').onclick = () => setMode('play');
$('brestart').onclick = () => begin(game.ci);
$('bpause').onclick = () => setMode('pause');
$('oinv').checked = input.invert; $('oinv').onchange = (e) => input.setInvert(e.target.checked);
$('omute').onchange = (e) => audio.setMuted(e.target.checked);
$('ovario').onchange = (e) => { audio.varioOn = e.target.checked; };
input.on('any', () => audio.start());
input.on('Space', () => { if (mode === 'card') nextCard(); });
input.on('Enter', () => { if (mode === 'card') nextCard(); else if (mode === 'menu') $('bstart').click(); else if (mode === 'done') $('bnext').click(); });
input.on('KeyP', () => { if (mode === 'play') setMode('pause'); else if (mode === 'pause') setMode('play'); });
input.on('Escape', () => { if (mode === 'play') setMode('pause'); else if (mode === 'pause') setMode('play'); });
input.on('KeyR', () => { if (mode === 'play') { game.spawn(); } });
input.on('KeyM', () => { $('omute').checked = !audio.muted; audio.setMuted(!audio.muted); });
input.on('KeyH', () => { $('keys').style.display = $('keys').style.display === 'none' ? '' : 'none'; });

// ---------- camera ----------
const camPos = new THREE.Vector3(), camLook = new THREE.Vector3(), tmp = new THREE.Vector3(), tmp2 = new THREE.Vector3(), camUp = new THREE.Vector3(0, 1, 0), dirS = new THREE.Vector3(1, 0, 0);
let camSnap = true, fov = 62;
function chaseCam(dt) {
  const b = bird, V = b.airspeed;
  const f = qrot(b.q, [1, 0, 0]), r = qrot(b.q, [0, 0, 1]);
  const dir = tmp.set(f[0], f[1], f[2]);
  if (b.mode === 'air') { const sp = Math.hypot(b.v[0], b.v[1], b.v[2]); if (sp > 4) dir.multiplyScalar(0.3).add(tmp2.set(b.v[0], b.v[1] * 0.75, b.v[2]).multiplyScalar(0.7 / sp)); }
  else if (b.mode === 'ground') dir.set(Math.cos(b.heading), -0.12, -Math.sin(b.heading));
  dir.normalize(); dir.y = clamp(dir.y, -0.62, 0.5); dir.normalize();
  dirS.lerp(dir, camSnap ? 1 : 1 - Math.exp(-dt * 3.2)).normalize();
  const up = sstep(0, 1, b.poseUp), dist = 8.6 + 1.5 * sstep(18, 70, V) - 2.2 * up, height = 2.5 - 0.6 * up;
  const want = tmp2.set(b.p[0], b.p[1], b.p[2]).addScaledVector(dirS, -dist); want.y += height + Math.max(0, -dirS.y) * 2.5;
  camPos.lerp(want, camSnap ? 1 : 1 - Math.exp(-dt * (7 + V * 0.35)));
  const g = Math.max(groundAt(camPos.x, camPos.z), 0) + 0.9; if (camPos.y < g) camPos.y = g;
  const look = tmp2.set(b.p[0], b.p[1] + 0.7 + 0.5 * up, b.p[2]).addScaledVector(dirS, 7);
  camLook.lerp(look, camSnap ? 1 : 1 - Math.exp(-dt * 10));
  // lean the horizon a little into the turn, and let the air shake the view when he is fast or stalling
  const lean = b.mode === 'air' ? clamp(b.bank, -1.3, 1.3) * 0.32 : 0;
  camUp.lerp(tmp.set(r[0] * Math.sin(lean), 1, r[2] * Math.sin(lean)).normalize(), camSnap ? 1 : 1 - Math.exp(-dt * 4));
  const shake = b.mode === 'air' ? 0.012 * sstep(40, 85, V) + 0.03 * b.stall * sstep(5, 12, V) : 0;
  camera.position.copy(camPos);
  camera.up.copy(camUp);
  camera.lookAt(camLook.x + (Math.random() - 0.5) * shake * 8, camLook.y + (Math.random() - 0.5) * shake * 8, camLook.z);
  const fv = 60 + 20 * sstep(13, 85, V) + punch;
  fov += (fv - fov) * (camSnap ? 1 : Math.min(1, dt * 5));
  camSnap = false;
}
function orbitCam(t, hero) {
  camera.up.set(0, 1, 0);
  if (hero) { const a = t * 0.12 + 2.2; camera.position.set(bird.p[0] + Math.cos(a) * 7.5, bird.p[1] + 1.6 + Math.sin(t * 0.2) * 0.5, bird.p[2] + Math.sin(a) * 7.5); camera.lookAt(bird.p[0], bird.p[1] + 0.5, bird.p[2]); fov = 50; }
  else { const a = 0.55 + t * 0.018; camera.position.set(-300 + Math.cos(a) * 4300, 520 + 120 * Math.sin(t * 0.05), 300 + Math.sin(a) * 4000); camera.lookAt(-500, 130, 200); fov = 50; }
  camPos.copy(camera.position); camSnap = true;
}

// ---------- HUD ----------
const el = { kmh: $('kmh'), alt: $('alt'), vario: $('vario'), stam: $('stam'), pw: $('pw'), root: $('root'), warn: $('warn'), tgt: $('tgt'), vig: $('vig'), sv: $('speedvig'), white: $('white'), chapT: $('chap').lastChild };
let hudT = 0, varioS = 0, pwS = 0, gS = 1;
function hud(dt) {
  const b = bird;
  varioS += (b.climb - varioS) * Math.min(1, dt * 3); pwS += (Math.max(b.power, 0) - pwS) * Math.min(1, dt * 2.5); gS += (b.gload - gS) * Math.min(1, dt * 6);
  el.vig.style.opacity = clamp((Math.abs(gS) - 3.2) / 3.5, 0, 0.85);
  el.sv.style.opacity = b.mode === 'air' ? 0.5 * sstep(45, 85, b.airspeed) : 0;
  el.white.style.opacity = white;
  hudT -= dt; if (hudT > 0) return; hudT = 0.08;
  el.kmh.textContent = Math.round(b.airspeed * 3.6);
  el.alt.textContent = Math.round(b.p[1] - (b.mode === 'ground' ? 1 : 0));
  el.vario.textContent = (varioS >= 0 ? '▲ +' : '▼ ') + varioS.toFixed(1);
  el.vario.className = 'num ' + (varioS > 0.3 ? 'up' : varioS < -2 ? 'dn' : '');
  el.stam.style.transform = `scaleX(${b.stamina.toFixed(3)})`;
  el.root.classList.toggle('low', b.stamina < 0.15);
  el.pw.textContent = pwS > 150 ? `${(pwS / 1000).toFixed(1)} קילוואט` : '';
  el.warn.classList.toggle('on', b.mode === 'air' && b.stall > 0.45 && b.fold > 0.8 && b.agl > 8);
  el.chapT.textContent = game.chapter.timed && game.si > 0 ? fmt(game.time) : '';
}
const tv = new THREE.Vector3();
function target() {
  const s = game.state === 'play' ? game.step : null;
  if (!s || !s.p) { el.tgt.style.opacity = 0; return; }
  tv.set(s.p[0], s.p[1] + (s.type === 'land' ? 2 : 0), s.p[2]);
  const d = tv.distanceTo(camera.position);
  tv.applyMatrix4(camera.matrixWorldInverse); const behind = tv.z > 0; tv.applyMatrix4(camera.projectionMatrix);
  let x = tv.x, y = tv.y; if (behind) { x = -x; y = -y; }
  const off = behind || Math.abs(x) > 0.92 || Math.abs(y) > 0.88;
  if (off) { const k = 1 / Math.max(Math.abs(x) / 0.92, Math.abs(y) / 0.86, 1e-3); if (behind || k < 1) { x *= k; y *= k; } if (behind && Math.hypot(x, y) < 0.5) { const m = 0.86 / Math.max(Math.hypot(x, y), 1e-3); x *= m; y *= m; } }
  el.tgt.style.opacity = 1; el.tgt.classList.toggle('off', off);
  el.tgt.style.transform = `translate(${((x + 1) / 2) * innerWidth}px, ${((1 - y) / 2) * innerHeight}px)`;
  el.tgt.firstChild.style.transform = off ? `rotate(${Math.atan2(-y, x) + Math.PI / 4}rad)` : '';
  el.tgt.lastChild.textContent = d > 1000 ? `${(d / 1000).toFixed(1)} ק"מ` : `${Math.round(d)} מ'`;
}

// ---------- loop ----------
let last = performance.now(), fps = 60, stepT = 0, slowT = 0, pr = Math.min(devicePixelRatio, 2);
function frame(now) {
  requestAnimationFrame(frame);
  const rdt = Math.min((now - last) / 1000, 0.05); last = now; fps += (1 / Math.max(rdt, 1e-3) - fps) * 0.05;
  let dt = rdt;
  if (mode === 'play') {
    game.update(rdt);
    slow = Math.max(0, slow - rdt);
    dt = rdt * game.timeScale * (slow > 0 ? 0.45 : 1);
    const inp = params.has('auto') ? pilot(bird, game, time) : input.read(rdt);
    bird.step(dt, inp, time);
    for (const e of bird.events) {
      if (e.type === 'spread') { audio.spread(e.speed); punch = 5 + Math.min(e.speed, 70) * 0.22; if (e.speed > 28) { slow = 0.35; white = 0.25; } fx.burst(bird.p, 6 + Math.min(e.speed, 60) / 4, 0xfff6e0, 5, 3, 1.6, 2.5); }
      if (e.type === 'tuck') audio.tuck();
      if (e.type === 'land') { audio.land(e.q); fx.burst([bird.p[0], bird.p[1] - 1, bird.p[2]], 26, 0xcdbb94, 6, 2.5, 1.0, 5); }
      if (e.type === 'crash') { audio.crash(); punch = -8; white = 0.5; fx.burst(bird.p, 60, 0xcdbb94, 12, 7, 1.6, 9); fx.burst(bird.p, 24, 0xfff6e0, 8, 6, 2.6, 2.5); }
      if (e.type === 'splash') { audio.splash(); fx.burst([bird.p[0], 0.2, bird.p[2]], 110, 0xeaf6ff, 9 + e.speed * 0.2, 9 + e.speed * 0.15, 1.5, 9.8); }
      if (e.type === 'liftoff') punch = 4;
    }
    bird.events.length = 0;
    if (bird.mode === 'ground' && Math.abs(bird.gspeed) > 1) { stepT -= dt * Math.abs(bird.gspeed); if (stepT < 0) { stepT = 1.6; audio.step(); } }
    time += dt;
    punch *= Math.exp(-rdt * 2.5); white *= Math.exp(-rdt * 5);
    chaseCam(rdt);
    hud(rdt);
    if (flashT > 0 && (flashT -= rdt) <= 0) show('flash', false);
    if (titleT > 0 && (titleT -= rdt) <= 0) show('titlecard', false);
  } else if (mode === 'pause' || mode === 'done') { dt = 0; } else {
    time += dt;
    orbitCam(time, mode === 'card' && cardI === PROLOGUE.length - 1);
    if (titleT > 0 && (titleT -= rdt) <= 0) show('titlecard', false);
  }
  U.uTime.value = time;
  camera.fov = fov; camera.updateProjectionMatrix();
  model.update(bird, dt);
  camera.updateMatrixWorld();
  fx.update(dt, time, bird, model, camera);
  flock.update(time); props.update(time);
  markers.update(game.state === 'play' ? game.markerIndex : 99, time);
  if (mode === 'play') target();
  audio.frame(bird, rdt, camera.position.y);
  renderer.render(scene, camera);
  game.fps = fps;
  // weak GPU: trade sharpness for smoothness
  if (fps < 38 && pr > 0.75) { slowT += rdt; if (slowT > 3) { pr = Math.max(0.75, pr * 0.8); renderer.setPixelRatio(pr); resize(); slowT = 0; fps = 60; } } else slowT = 0;
}

// first frame compiles the shaders; then open the menu
renderer.compile(scene, camera);
$('loading').style.display = 'none'; $('bstart').style.display = '';
setMode('menu');
fx.reset(model);
window.game = game; window.dbg = { renderer, bird, input, camera, scene, audio, begin, setMode, trees, POI };
if (params.has('ch')) begin(+params.get('ch'));
requestAnimationFrame(frame);
