// Film player / renderer. Viewer mode plays in real time with scrubbing and a free camera;
// render mode (?render) exposes renderFrame(i) for tools/render.py, which composites
// several motion-blurred, jittered sub-frames per video frame.

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { runSimulation } from '../sim.js';
import { buildWorld, SUN_DIR } from './world.js';
import { buildBike } from './bikeModel.js';
import { Particles, Plumes } from './fx.js';
import { Post } from './post.js';
import { makeShots } from './shots.js';
import { makeHud } from './hud.js';
import { surfaceAt, WATER_Y } from '../terrain.js';

const params = new URLSearchParams(location.search);
const RENDER = params.has('render');
const FPS = +(params.get('fps') || 30);
const SUB = +(params.get('sub') || (RENDER ? 8 : 1));
const statusEl = document.getElementById('status');
const say = (m) => { statusEl.textContent = m; };
const frame = () => new Promise((r) => requestAnimationFrame(() => r()));

say('מריץ סימולציה פיזיקלית…');
await frame();
const t0 = performance.now();
const rec = runSimulation();
console.log('simulation', ((performance.now() - t0) / 1000).toFixed(1), 's', rec.events);

const W = RENDER ? +(params.get('w') || 1920) : innerWidth;
const H = RENDER ? +(params.get('h') || 1080) : innerHeight;
const renderer = new THREE.WebGLRenderer({ antialias: false, preserveDrawingBuffer: RENDER, powerPreference: 'high-performance' });
renderer.setPixelRatio(1);
renderer.setSize(W, H);
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
if (RENDER && !params.has("shadowAlways")) renderer.shadowMap.autoUpdate = false;
renderer.autoClear = true;
document.getElementById('stage').appendChild(renderer.domElement);

say('בונה עמק…');
await frame();
const scene = new THREE.Scene();
const hazeScene = new THREE.Scene();
const shots = makeShots(rec);
const world = buildWorld(scene, renderer, shots.clearings);
const bike = buildBike(world.envTex);
scene.add(bike.root);
const particles = new Particles(scene, SUN_DIR);
particles.hAt = world.hAt;
const plumes = new Plumes(scene, hazeScene);
const post = new Post(renderer, W, H);
const camera = new THREE.PerspectiveCamera(40, W / H, 0.2, 40000);
const hud = makeHud(document.getElementById('overlay'), shots.marks, rec.events, shots.tEnd);
const EXPOSURE = 0.64;

// ---- effects clock (particles integrate forward only)
let fxT = 0;
function advanceFx(t) {
  if (t < fxT - 1e-6 || t - fxT > 4) { particles.reset(); fxT = Math.max(0, t - 4); }
  while (fxT < t - 1e-9) {
    const dt = Math.min(1 / 240, t - fxT);
    const s = rec.at(fxT);
    bike.update(s, 0);
    particles.step(dt, bike.exhausts(s), s.wind);
    fxT += dt;
  }
}

// jet blast on the grass: impingement points of the lift jets
function blastUniforms(ex) {
  const arr = world.grassU.uBlast.value;
  ex.forEach((e, i) => {
    if (e.d.y > -0.2 || e.T < 30) { arr[i].set(0, 0, 0, 1); return; }
    const h = (e.p.y - surfaceAt(e.p.x, e.p.z)) / -e.d.y;
    const x = e.p.x + e.d.x * h, z = e.p.z + e.d.z * h;
    const U = 500 * Math.min(1, 0.84 / Math.max(h, 0.1)) * Math.sqrt(e.T / 1250);
    arr[i].set(x, z, Math.min(3.5, U / 25), 1.2 + 0.35 * h);
  });
}

let lastSub = 0;
let freeCam = null;
function drawSub(t, jitter, camT) {
  const s = rec.at(t);
  bike.update(s, Math.max(0, t - lastSub));
  lastSub = t;
  const ex = bike.exhausts(s);
  advanceFx(t);
  bike.update(s, 0);
  particles.upload();
  world.grassU.uTime.value = t;
  world.grassU.uWind.value.set(...s.wind);
  blastUniforms(ex);
  world.water.material.uniforms.time.value = t * 0.45;
  // camera
  if (freeCam) {
    freeCam.update();
  } else {
    const c = shots.at(camT ?? t);
    const cc = c.shot.cam(t);
    camera.position.copy(cc.pos);
    camera.up.copy(cc.up || new THREE.Vector3(0, 1, 0));
    camera.lookAt(cc.look);
    camera.fov = cc.fov;
    camera.near = cc.near || 0.2;
    camera.updateProjectionMatrix();
  }
  if (jitter) camera.setViewOffset(W, H, jitter[0], jitter[1], W, H); else camera.clearViewOffset();
  plumes.update(ex, t, post, camera);
  // shadow frustum follows the bike
  const b = bike.root.position;
  world.sun.target.position.copy(b);
  world.sun.position.copy(b).addScaledVector(SUN_DIR, 400);
  world.sun.target.updateMatrixWorld();
  post.renderFrame(scene, camera, hazeScene);
  return s;
}

const halton = (i, b) => { let f = 1, r = 0; while (i > 0) { f /= b; r += f * (i % b); i = Math.floor(i / b); } return r; };
const fadeAt = (t) => Math.min(1, t / 1.2) * Math.min(1, Math.max(0, (shots.tEnd - t) / 1.6));

// ---- render API (tools/render.py)
window.film = {
  fps: FPS, duration: shots.tEnd, frames: Math.floor(shots.tEnd * FPS), events: rec.events, marks: shots.marks,
  shots: shots.shots.map((s) => ({ name: s.name, t0: s.t0, t1: s.t1 })),
  async renderFrame(i) {
    const tc = i / FPS, shutter = 0.5 / FPS;
    post.clearAccum();
    for (let k = 0; k < SUB; k++) {
      const t = Math.max(0, tc + ((k + 0.5) / SUB - 0.5) * shutter);
      const j = SUB > 1 ? [halton(k + 1, 2) - 0.5, halton(k + 1, 3) - 0.5] : null;
      if (k === 0) renderer.shadowMap.needsUpdate = true;
      drawSub(t, j, tc);
      post.accumulate(1 / SUB);
    }
    post.toScreen(EXPOSURE, tc, fadeAt(tc));
    hud.update(rec.at(tc), tc);
    await frame();
    return true;
  },
  // sound data: engines at 120 Hz, and each shot's camera over its time span (+margins)
  audioData(rate = 120) {
    const n = Math.floor(shots.tEnd * rate) + 1;
    const eng = [];
    for (let i = 0; i < n; i++) {
      const t = i / rate, s = rec.at(t);
      bike.update(s, 0);
      const ex = bike.exhausts(s);
      // where each jet hits the ground/water and how hard (same model as the particles)
      const hits = ex.map((e) => {
        if (e.d.y > -0.12 || e.T < 30) return [0, 0, 0, 0, 0];
        let th = (e.p.y - surfaceAt(e.p.x, e.p.z)) / -e.d.y;
        const hx = e.p.x + e.d.x * th, hz = e.p.z + e.d.z * th;
        th = (e.p.y - surfaceAt(hx, hz)) / -e.d.y;
        if (th < 0 || th > 25) return [0, 0, 0, 0, 0];
        const x = e.p.x + e.d.x * th, z = e.p.z + e.d.z * th, y = surfaceAt(x, z);
        const Ve = 560 * Math.sqrt(Math.min(1, e.T / 1250));   // T ~ mdot*V with mdot ~ V  =>  V ~ sqrt(T)
        const U = Ve * Math.min(1, 6 * 2 * e.r / Math.max(th, 0.1));
        return [x, y, z, U, y <= WATER_Y + 0.01 ? 1 : 0];
      });
      eng.push({ t, v: s.v, e: ex.map((e) => [e.p.x, e.p.y, e.p.z, e.d.x, e.d.y, e.d.z, e.T, e.N, e.egt]), hit: hits, air: s.airspeed, contact: s.contact });
    }
    const cams = shots.shots.map((sh) => {
      const list = [];
      for (let t = Math.max(0, sh.t0 - 0.25); t <= sh.t1 + 0.25; t += 1 / rate) {
        const c = sh.cam(Math.min(t, shots.tEnd));
        const m = new THREE.Matrix4().lookAt(c.pos, c.look, c.up || new THREE.Vector3(0, 1, 0));
        const right = new THREE.Vector3().setFromMatrixColumn(m, 0);
        list.push([t, c.pos.x, c.pos.y, c.pos.z, right.x, right.y, right.z, sh.onboard ? 1 : 0]);
      }
      return { name: sh.name, t0: sh.t0, t1: sh.t1, onboard: !!sh.onboard, cam: list };
    });
    return { rate, duration: shots.tEnd, eng, cams, events: rec.events };
  },
};

say('');
if (RENDER) {
  window.filmReady = true;
} else {
  // ---- interactive player
  const ui = document.getElementById('ui');
  ui.hidden = false;
  const slider = document.getElementById('scrub');
  const playBtn = document.getElementById('play');
  const shotLbl = document.getElementById('shot');
  const camSel = document.getElementById('cam');
  slider.max = shots.tEnd;
  let t = +(params.get('t') || 0), playing = !params.has('t'), lastNow = performance.now();
  playBtn.onclick = () => { playing = !playing; playBtn.textContent = playing ? '❚❚' : '▶'; };
  playBtn.textContent = playing ? '❚❚' : '▶';
  slider.oninput = () => { t = +slider.value; };
  addEventListener('keydown', (e) => { if (e.code === 'Space') { playBtn.onclick(); e.preventDefault(); } });
  let controls = null;
  camSel.onchange = () => {
    if (camSel.value === 'orbit') {
      controls = new OrbitControls(camera, renderer.domElement);
      controls.target.copy(bike.root.position);
      let prev = bike.root.position.clone();
      freeCam = { update() { const d = bike.root.position.clone().sub(prev); camera.position.add(d); controls.target.add(d); prev = bike.root.position.clone(); controls.update(); } };
      camera.near = 0.1; camera.fov = 45; camera.updateProjectionMatrix();
    } else { controls?.dispose(); controls = null; freeCam = null; }
  };
  const loop = () => {
    const now = performance.now();
    if (playing) t = Math.min(shots.tEnd, t + (now - lastNow) / 1000);
    lastNow = now;
    slider.value = t;
    drawSub(t, null, t);
    post.toScreen(EXPOSURE, t, 1, false);
    hud.update(rec.at(t), t);
    shotLbl.textContent = `${t.toFixed(1)} s · ${shots.at(t).shot.name}`;
    requestAnimationFrame(loop);
  };
  loop();
}
