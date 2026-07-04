import * as THREE from 'three';
import { buildWorld, terrainHeight, terrainNormal, WATER_Y, SPAWN } from './world.js';
import { buildBike } from './bike.js';
import { BikePhysics, TUNE } from './physics.js';
import { makeJetFX } from './jetfx.js';
import { makeGrass } from './grass.js';
import { makeAnimals } from './animals.js';
import { makeExplosion } from './explosion.js';
import { HUD } from './hud.js';
import { JetAudio } from './audio.js';
import { makeTouchControls } from './touch.js';
import { t, setLang } from './i18n.js';

const clamp = (v, a, b) => Math.min(b, Math.max(a, v));

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.setSize(innerWidth, innerHeight);
document.body.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.fog = new THREE.FogExp2(0xc6d8e8, 0.00042);
const camera = new THREE.PerspectiveCamera(68, innerWidth / innerHeight, 0.1, 9000);

const world = buildWorld(scene);
const bike = buildBike();
scene.add(bike.group);
const fx = makeJetFX(scene, bike.group, camera, renderer);
const boomFx = makeExplosion(scene, camera, renderer);
const grass = makeGrass(scene);
const animals = makeAnimals(scene);
const phys = new BikePhysics();
const hud = new HUD();
const audio = new JetAudio();

// soft blob shadow on the terrain — crucial for judging altitude
const shadow = new THREE.Mesh(
  new THREE.CircleGeometry(1.7, 24),
  new THREE.MeshBasicMaterial({ color: 0x000000, transparent: true, opacity: 0.35, depthWrite: false })
);
scene.add(shadow);

// floating arrow pointing at the next ring
const arrow = new THREE.Mesh(
  new THREE.ConeGeometry(0.32, 1.1, 10),
  new THREE.MeshBasicMaterial({ color: 0xff9020 })
);
scene.add(arrow);

let camMode = 0; // 0 = chase, 1 = onboard
let started = false;

// ?demo — auto-takeoff, for headless smoke tests and screenshots
// ?pose=cruise — start mid-air at cruise speed with full afterburner (visual debug)
// ?touch=1 — force the mobile touch controls on a desktop browser
// ?selftest=touch — headless check that the touch controls actually move phys state
const PARAMS = new URLSearchParams(location.search);
const DEMO = PARAMS.has('demo') || PARAMS.has('pose');
const POSE = PARAMS.get('pose');
const TOUCH_MODE = matchMedia('(pointer: coarse)').matches || PARAMS.has('touch');
document.body.classList.toggle('touch-mode', TOUCH_MODE);
if (PARAMS.get('lang')) { setLang(PARAMS.get('lang')); hud.renderStatic(); }
hud.setTouchMode(TOUCH_MODE);

// shared so both the keyboard shortcuts and the touch menu buttons trigger the same actions
function toggleCamera() { camMode = (camMode + 1) % 2; }
function doReset() { phys.reset(); hud.message(t('reset'), 900); }
function toggleAssist() {
  phys.assist = !phys.assist;
  hud.message(phys.assist ? t('assistMsgOn') : t('assistMsgOff'), 1600);
}
function toggleMute() { hud.message(audio.toggleMute() ? t('muted') : t('soundOn'), 1000); }
function toggleHelp() { hud.toggleHelp(); }
function firePulse(which) { if (phys.firePulse(which)) audio.pulse(); }

const touch = TOUCH_MODE ? makeTouchControls({
  onCamera: toggleCamera, onReset: doReset, onAssist: toggleAssist, onMute: toggleMute, onHelp: toggleHelp,
  onLang: () => hud.toggleLang(), onPulse: firePulse,
}) : null;

if (PARAMS.has('selftest') && touch) {
  const dispatch = (el, type, id, x, y) => el.dispatchEvent(new PointerEvent(type, {
    pointerId: id, clientX: x, clientY: y, bubbles: true, cancelable: true,
  }));
  setTimeout(() => {
    dispatch(document.body, 'pointerdown', 1, 5, 5); // first tap: start the game
    setTimeout(() => {
      const joy = document.querySelector('.tjoy');
      const r = joy.getBoundingClientRect(), cx = r.left + r.width / 2, cy = r.top + r.height / 2;
      dispatch(joy, 'pointerdown', 2, cx, cy);
      dispatch(joy, 'pointermove', 2, cx + 40, cy + 40); // right + down -> steer>0, pitch>0
      const boost = document.querySelector('.tboost');
      dispatch(boost, 'pointerdown', 3, 1, 1);
      setTimeout(() => {
        document.body.dataset.ttSteer = touch.steer.toFixed(2);
        document.body.dataset.ttPitch = touch.pitch.toFixed(2);
        document.body.dataset.ttBoost = String(touch.boostRear);
        document.body.dataset.ttStarted = String(started);
        document.body.dataset.ttHelpBefore = hud.help.style.display;
        document.querySelectorAll('.tmenu button')[4].click(); // '?' help toggle
        document.body.dataset.ttHelpAfter = hud.help.style.display;
        document.body.dataset.ttDone = '1';
      }, 100);
    }, 100);
  }, 300);
}

if (DEMO) {
  started = true;
  setTimeout(() => hud.hideStart(), 0);
}
if (POSE === 'cruise') {
  phys.pos.set(SPAWN.x, SPAWN.y + 55, SPAWN.z + 120);
  phys.vel.set(0, 0, 52);
  phys.throttle = 1;
  phys.collective = 0.55;
}
// ?pose=boom — slam into the ground on the first frames to showcase the
// crash explosion (starts just above the terrain so headless shots catch it)
if (POSE === 'boom') {
  phys.pos.set(SPAWN.x, SPAWN.y + 5, SPAWN.z + 40);
  phys.vel.set(0, -30, 3);
}

// ?pose=show — bike pinned mid-air with everything firing, close static camera
function holdShowPose() {
  phys.pos.set(SPAWN.x, SPAWN.y + 8, SPAWN.z);
  phys.vel.set(0, 0, 0);
  phys.quat.identity();
  phys.angVel.set(0, 0, 0);
  phys.angMom.set(0, 0, 0);
  phys.throttle = 1;
  phys.collective = 0.9;
  phys.input.boostRear = true;
  phys.firePulse('C');   // rhythmic center pulses (auto-limited by the gap)
}
let ringIndex = 0, runStart = 0, finished = false, finishTimer = 0, bestTime = null, lastRunMs = 0;
const totalRings = world.rings.length;

const keys = {};
addEventListener('keydown', e => {
  if (['Space', 'ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight'].includes(e.code)) e.preventDefault();
  if (e.repeat) return;
  if (!started) {
    if (e.code === 'KeyL') { hud.toggleLang(); return; } // language toggle without launching
    started = true;
    hud.hideStart();
    audio.init();
    return;
  }
  keys[e.code] = true;
  switch (e.code) {
    case 'KeyZ': firePulse('L'); break;
    case 'KeyX': firePulse('R'); break;
    case 'ShiftLeft':
    case 'ShiftRight': firePulse('C'); break;
    case 'KeyR': doReset(); break;
    case 'KeyC': toggleCamera(); break;
    case 'KeyH': toggleHelp(); break;
    case 'KeyM': toggleMute(); break;
    case 'KeyT': toggleAssist(); break;
    case 'KeyL': hud.toggleLang(); break;
  }
});
addEventListener('keyup', e => { keys[e.code] = false; });
addEventListener('blur', () => { for (const k in keys) keys[k] = false; });
if (TOUCH_MODE) {
  addEventListener('pointerdown', e => {
    if (started || e.target.closest?.('.lang-btn')) return;
    started = true;
    hud.hideStart();
    audio.init();
  });
}
addEventListener('resize', () => {
  camera.aspect = innerWidth / innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
});

function resetCourse() {
  ringIndex = 0;
  runStart = 0;
  finished = false;
  for (const r of world.rings) {
    r.passed = false;
    r.mesh.material.emissive.setHex(0xff8c1a);
    r.mesh.scale.setScalar(1);
  }
}

function applyInput(dt) {
  if (DEMO) {
    if (POSE === 'boom') {
      phys.throttle = 0;
      phys.collective = 0;
      return;
    }
    if (POSE === 'cruise') {
      phys.throttle = 1;
      phys.collective = 0.55;
      phys.input.boostRear = true;
    } else {
      phys.throttle = 0.65;
      phys.collective = 0.8;
    }
    return;
  }
  if (keys['KeyW'] || touch?.throttleUp) phys.throttle = clamp(phys.throttle + dt * 0.6, 0, 1);
  if (keys['KeyS'] || touch?.throttleDown) phys.throttle = clamp(phys.throttle - dt * 0.8, 0, 1);
  if (keys['KeyE'] || touch?.collUp) phys.collective = clamp(phys.collective + dt * 0.55, 0, 1.25);
  if (keys['KeyD'] || touch?.collDown) phys.collective = clamp(phys.collective - dt * 0.55, 0, 1.25);
  const steerTarget = clamp((keys['ArrowRight'] ? 1 : 0) - (keys['ArrowLeft'] ? 1 : 0) + (touch?.steer || 0), -1, 1);
  phys.steer += (steerTarget - phys.steer) * Math.min(1, dt * 8);
  // airplane-yoke convention: pull back (arrow down / joystick down) raises the nose
  const pitchTarget = clamp((keys['ArrowDown'] ? 1 : 0) - (keys['ArrowUp'] ? 1 : 0) + (touch?.pitch || 0), -1, 1);
  phys.pitch += (pitchTarget - phys.pitch) * Math.min(1, dt * 6);
  phys.input.boostRear = !!keys['Space'] || !!touch?.boostRear;
}

const _fh = new THREE.Vector3(), _look = new THREE.Vector3(), _tmp = new THREE.Vector3();
const _n = new THREE.Vector3(), _Z = new THREE.Vector3(0, 0, 1), _Y = new THREE.Vector3(0, 1, 0);
const FLIP = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), Math.PI);
camera.position.set(SPAWN.x, SPAWN.y + 4, SPAWN.z - 14);

function updateCamera(dt) {
  // debug cameras for the living scenery: park on the first herd / flock
  if (POSE === 'horses' && animals.horses.length) {
    const h = animals.horses[0];
    const hy = terrainHeight(h.hx, h.hz);
    camera.position.set(h.hx + 14, hy + 4, h.hz + 14);
    camera.lookAt(h.hx, hy + 1, h.hz);
    camera.fov = 55;
    camera.updateProjectionMatrix();
    return;
  }
  if (POSE === 'birds' && animals.flocks.length) {
    const f = animals.flocks[0];
    camera.position.set(f.cx - Math.sin(f.heading) * 16, f.y + 3, f.cz - Math.cos(f.heading) * 16);
    camera.lookAt(f.cx, f.y, f.cz);
    camera.fov = 60;
    camera.updateProjectionMatrix();
    return;
  }
  if (POSE === 'show') {
    camera.position.set(phys.pos.x + 5.5, phys.pos.y + 1.0, phys.pos.z - 6.5);
    camera.lookAt(phys.pos.x, phys.pos.y - 0.6, phys.pos.z - 1);
    camera.fov = 55;
    camera.updateProjectionMatrix();
    return;
  }
  if (POSE === 'boom') {
    // static wide view of the impact zone
    camera.position.set(SPAWN.x + 22, SPAWN.y + 10, SPAWN.z + 16);
    camera.lookAt(SPAWN.x, SPAWN.y + 3, SPAWN.z + 42);
    camera.fov = 55;
    camera.updateProjectionMatrix();
    return;
  }
  const speed = phys.vel.length();
  if (camMode === 0) {
    _fh.set(0, 0, 1).applyQuaternion(phys.quat);
    _fh.y *= 0.25;
    _fh.normalize();
    const dist = 8.5 + speed * 0.05;
    _look.copy(phys.pos).addScaledVector(_fh, -dist);
    _look.y += 2.6 + speed * 0.012;
    const minY = Math.max(terrainHeight(_look.x, _look.z), WATER_Y) + 1.0;
    if (_look.y < minY) _look.y = minY;
    camera.position.lerp(_look, 1 - Math.exp(-5 * dt));
    _look.copy(phys.pos).addScaledVector(phys.vel, 0.1);
    _look.y += 0.8;
    camera.lookAt(_look);
    camera.fov = 68 + clamp(speed * 0.22, 0, 22) + (phys.input.boostRear ? 3 : 0);
  } else {
    _look.set(0, 0.95, 0.3).applyQuaternion(phys.quat).add(phys.pos);
    camera.position.copy(_look);
    camera.quaternion.copy(phys.quat).multiply(FLIP);
    camera.fov = 75;
  }
  camera.updateProjectionMatrix();
}

function updateRings(dt) {
  if (finished) {
    finishTimer -= dt;
    if (finishTimer <= 0) resetCourse();
    return;
  }
  if (ringIndex >= totalRings) return;
  const r = world.rings[ringIndex];
  const s = 1 + 0.07 * Math.sin(performance.now() * 0.005);
  r.mesh.scale.setScalar(s);
  if (phys.pos.distanceTo(r.pos) < 8.5) {
    r.passed = true;
    r.mesh.material.emissive.setHex(0x30ff70);
    r.mesh.scale.setScalar(1);
    ringIndex++;
    audio.beep();
    if (ringIndex === 1) runStart = performance.now();
    if (ringIndex === totalRings) {
      finished = true;
      finishTimer = 5;
      lastRunMs = performance.now() - runStart;
      const secs = lastRunMs / 1000;
      let msg = t('allRingsTime') + secs.toFixed(1) + ' ' + t('seconds');
      if (bestTime === null || secs < bestTime) { bestTime = secs; msg += t('newRecord'); }
      hud.message(msg, 4500);
    } else {
      hud.message(t('ring') + ringIndex + '/' + totalRings, 900);
    }
  }
}

function fmtTime(ms) {
  const s = ms / 1000;
  return (s / 60 | 0) + ':' + (s % 60).toFixed(1).padStart(4, '0');
}

// wind readout: speed + an arrow showing where the wind pushes, in bike frame
// (screen orientation: body +X is screen-LEFT, hence the sign flip)
const WIND_ARROWS = ['↑', '↗', '→', '↘', '↓', '↙', '←', '↖'];
function windLabel() {
  const W = phys.wind.W;
  if (phys.wind.speed < 0.01) return '';
  _tmp.set(0, 0, 1).applyQuaternion(phys.quat);
  const rel = Math.atan2(W.x, W.z) - Math.atan2(_tmp.x, _tmp.z);
  const idx = ((Math.round(-rel / (Math.PI / 4)) % 8) + 8) % 8;
  return t('wind') + ' ' + (phys.wind.speed * 3.6).toFixed(0) + ' ' + t('kmh') + ' ' + WIND_ARROWS[idx];
}

const clock = new THREE.Clock();
const STEP = 1 / 120;
let acc = 0, hudAcc = 0, firstFrame = true, frameNo = 0;

function animate() {
  requestAnimationFrame(animate);
  // idle throttling: on the start screen render at ~1/4 rate to save CPU
  // (hidden tabs already pause rAF entirely, courtesy of the browser)
  frameNo++;
  if (!started && frameNo % 4 !== 0) return;
  const dt = Math.min(clock.getDelta(), 0.05);

  if (started) applyInput(dt);
  acc += dt;
  while (acc > STEP) { phys.step(STEP); acc -= STEP; }
  if (POSE === 'show') holdShowPose();

  const crashReason = phys.consumeCrashEvent();
  if (crashReason) {
    hud.showCrash(crashReason);
    const gy = Math.max(terrainHeight(phys.pos.x, phys.pos.z), WATER_Y);
    boomFx.trigger(phys.pos, gy, crashReason === 'water' ? 'water' : 'fire');
    audio.boom();
    bike.group.visible = false;   // the wreck IS the debris until respawn
  }
  if (!phys.crashed && !bike.group.visible) bike.group.visible = true;

  updateRings(dt);

  bike.group.position.copy(phys.pos);
  bike.group.quaternion.copy(phys.quat);

  // blob shadow
  const gh = Math.max(terrainHeight(phys.pos.x, phys.pos.z), WATER_Y);
  const agl = phys.pos.y - gh;

  fx.update(dt, {
    jets: {
      // flames follow the ACTUAL spool state, so they lag the levers like the thrust does
      rear: phys.spoolRear + phys.ab * 0.9,
      // steer > 0 (screen-right) lifts the body +X side, i.e. the 'right' jet at x=+0.75
      left: phys.spoolLift * 0.7 + (phys.pulses.L > 0 ? 1.3 : 0) + Math.max(0, -phys.steer) * 0.25,
      right: phys.spoolLift * 0.7 + (phys.pulses.R > 0 ? 1.3 : 0) + Math.max(0, phys.steer) * 0.25,
      center: phys.spoolLift * 0.7 + (phys.pulses.C > 0 ? 1.3 : 0),
    },
    pos: phys.pos, quat: phys.quat, vel: phys.vel,
    vec: phys.nozzle,
    agl, groundY: gh,
  });
  grass.update(phys.pos, dt);
  animals.update(dt, phys.pos);
  boomFx.update(dt);
  terrainNormal(phys.pos.x, phys.pos.z, _n);
  shadow.position.set(phys.pos.x, gh + 0.08, phys.pos.z);
  shadow.quaternion.setFromUnitVectors(_Z, _n);
  shadow.scale.setScalar(1 + agl * 0.02);
  shadow.material.opacity = 0.38 * clamp(1 - agl / 70, 0, 1);
  shadow.visible = shadow.material.opacity > 0.02;

  // next-ring arrow
  if (!finished && ringIndex < totalRings) {
    arrow.visible = true;
    arrow.position.copy(phys.pos).add(_tmp.set(0, 2.1, 0));
    _tmp.copy(world.rings[ringIndex].pos).sub(arrow.position).normalize();
    arrow.quaternion.setFromUnitVectors(_Y, _tmp);
  } else {
    arrow.visible = false;
  }

  updateCamera(dt);
  // gust buffet: a whisper of camera shake proportional to local turbulence
  const buffet = Math.min(0.12, phys.wind.gustMag * 0.03) * clamp(phys.vel.length() * 0.02, 0, 1);
  const shake = Math.max(boomFx.shake, buffet);
  if (shake > 0.01) {
    camera.position.x += (Math.random() - 0.5) * shake;
    camera.position.y += (Math.random() - 0.5) * shake;
    camera.position.z += (Math.random() - 0.5) * shake;
  }

  const speed = phys.vel.length();
  const power = clamp(phys.jetRear / 8400 * 1.2 + phys.jetLift / 7800 * 0.8, 0, 1.6);
  audio.update(power, speed);

  hudAcc += dt;
  if (hudAcc > 0.1) {
    hudAcc = 0;
    hud.setStats({
      speedKmh: speed * 3.6,
      agl,
      alt: phys.pos.y,
      throttle: phys.throttle,
      collective: phys.collective,
      spoolRear: phys.spoolRear + phys.ab * 0.2,
      spoolLift: phys.spoolLift,
      assist: phys.assist,
      rings: ringIndex + '/' + totalRings,
      wind: windLabel(),
      time: runStart && !finished ? fmtTime(performance.now() - runStart) : (finished ? fmtTime(lastRunMs) : '—'),
      best: bestTime,
    });
  }

  renderer.render(scene, camera);
  if (firstFrame) {
    firstFrame = false;
    document.body.dataset.ok = '1'; // headless smoke-test marker
  }
}
animate();
