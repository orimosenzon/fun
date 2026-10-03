// main.js — אתחול, לולאת רינדור, מעבר בין לשוניות ושליטה בזמן
import * as THREE from 'three';
import { createWorld } from './world.js';
import { $, $$ } from './util.js';
import { orbitsMode } from './mode_orbits.js';
import { compareMode } from './mode_compare.js';
import { launchMode } from './mode_launch.js';
import { missionsMode } from './mode_missions.js';
import { museumMode } from './mode_museum.js';
import { aboutMode } from './mode_about.js';

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

function setMode(name) {
  if (name === modeName) return;
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

$$('#tabs button').forEach(b => b.addEventListener('click', () => setMode(b.dataset.tab)));
$('#sheet-toggle').addEventListener('click', () => $('#panel').classList.toggle('collapsed'));

function resize() {
  const w = window.innerWidth, h = window.innerHeight;
  world.renderer.setSize(w, h, false);
  world.camera.aspect = w / h;
  world.camera.updateProjectionMatrix();
  mode?.resize?.(w, h);
}
window.addEventListener('resize', resize);
resize();

let last = performance.now();
function frame(now) {
  const dtReal = Math.min(0.1, (now - last) / 1000);
  last = now;
  const dtSim = world.clock.paused ? 0 : dtReal * world.clock.rate;
  world.clock.time += dtSim * 1000;
  world.updateCelestial();
  mode?.update?.(world, dtReal, dtSim);
  world.controls.update();
  const sc = mode?.scene ?? world.scene;
  const cam = mode?.camera ?? world.camera;
  cam.updateMatrixWorld();
  world.renderer.render(sc, cam);
  world.updateLabels(window.innerWidth, window.innerHeight);
  const d = new Date(world.clock.time);
  $('#clock').textContent = world.timeDisplay ? world.timeDisplay() : d.toISOString().slice(0, 16).replace('T', ' ') + ' UTC';
  requestAnimationFrame(frame);
}

const start = location.hash.slice(1).split('/')[0];
setMode(MODES[start] ? start : 'orbits');
requestAnimationFrame(frame);
THREE.DefaultLoadingManager.onLoad = () => $('#loading').classList.add('done');
setTimeout(() => $('#loading').classList.add('done'), 6000);
