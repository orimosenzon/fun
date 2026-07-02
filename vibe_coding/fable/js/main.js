import * as THREE from 'three';
import { buildWorld, terrainHeight, terrainNormal, WATER_Y, SPAWN } from './world.js';
import { buildBike } from './bike.js';
import { BikePhysics, TUNE } from './physics.js';
import { HUD } from './hud.js';
import { JetAudio } from './audio.js';

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
const DEMO = new URLSearchParams(location.search).has('demo');
if (DEMO) {
  started = true;
  setTimeout(() => hud.hideStart(), 0);
}
let ringIndex = 0, runStart = 0, finished = false, finishTimer = 0, bestTime = null, lastRunMs = 0;
const totalRings = world.rings.length;

const keys = {};
addEventListener('keydown', e => {
  if (['Space', 'ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight'].includes(e.code)) e.preventDefault();
  if (e.repeat) return;
  if (!started) {
    started = true;
    hud.hideStart();
    audio.init();
    return;
  }
  keys[e.code] = true;
  switch (e.code) {
    case 'KeyR': phys.reset(); hud.message('איפוס', 900); break;
    case 'KeyC': camMode = (camMode + 1) % 2; break;
    case 'KeyH': hud.toggleHelp(); break;
    case 'KeyM': hud.message(audio.toggleMute() ? 'שקט' : 'סאונד פועל', 1000); break;
    case 'KeyT':
      phys.assist = !phys.assist;
      hud.message(phys.assist ? 'מייצב טיסה: פועל' : 'מייצב טיסה: כבוי — טיסה חופשית!', 1600);
      break;
  }
});
addEventListener('keyup', e => { keys[e.code] = false; });
addEventListener('blur', () => { for (const k in keys) keys[k] = false; });
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
    phys.throttle = 0.65;
    phys.collective = 0.8;
    return;
  }
  if (keys['ArrowUp']) phys.throttle = clamp(phys.throttle + dt * 0.6, 0, 1);
  if (keys['ArrowDown']) phys.throttle = clamp(phys.throttle - dt * 0.8, 0, 1);
  if (keys['KeyE']) phys.collective = clamp(phys.collective + dt * 0.55, 0, 1.25);
  if (keys['KeyD']) phys.collective = clamp(phys.collective - dt * 0.55, 0, 1.25);
  const steerTarget = (keys['ArrowRight'] ? 1 : 0) - (keys['ArrowLeft'] ? 1 : 0);
  phys.steer += (steerTarget - phys.steer) * Math.min(1, dt * 8);
  phys.input.boostRear = !!keys['Space'];
  phys.input.boostL = !!keys['KeyZ'];
  phys.input.boostR = !!keys['KeyX'];
  phys.input.boostC = !!keys['KeyS'];
}

const _fh = new THREE.Vector3(), _look = new THREE.Vector3(), _tmp = new THREE.Vector3();
const _n = new THREE.Vector3(), _Z = new THREE.Vector3(0, 0, 1), _Y = new THREE.Vector3(0, 1, 0);
const FLIP = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), Math.PI);
camera.position.set(SPAWN.x, SPAWN.y + 4, SPAWN.z - 14);

function updateCamera(dt) {
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
      const t = lastRunMs / 1000;
      let msg = 'כל הטבעות! זמן: ' + t.toFixed(1) + ' שניות';
      if (bestTime === null || t < bestTime) { bestTime = t; msg += ' — שיא חדש!'; }
      hud.message(msg, 4500);
    } else {
      hud.message('טבעת ' + ringIndex + '/' + totalRings, 900);
    }
  }
}

function fmtTime(ms) {
  const s = ms / 1000;
  return (s / 60 | 0) + ':' + (s % 60).toFixed(1).padStart(4, '0');
}

const clock = new THREE.Clock();
const STEP = 1 / 120;
let acc = 0, hudAcc = 0, firstFrame = true;

function animate() {
  requestAnimationFrame(animate);
  const dt = Math.min(clock.getDelta(), 0.05);

  if (started) applyInput(dt);
  acc += dt;
  while (acc > STEP) { phys.step(STEP); acc -= STEP; }

  const crashReason = phys.consumeCrashEvent();
  if (crashReason) hud.showCrash(crashReason);

  updateRings(dt);

  bike.group.position.copy(phys.pos);
  bike.group.quaternion.copy(phys.quat);
  bike.setJets({
    rear: phys.throttle + (phys.input.boostRear ? 0.9 : 0),
    left: phys.collective * 0.7 + (phys.input.boostL ? 0.9 : 0) + Math.max(0, phys.steer) * 0.25,
    right: phys.collective * 0.7 + (phys.input.boostR ? 0.9 : 0) + Math.max(0, -phys.steer) * 0.25,
    center: phys.collective * 0.7 + (phys.input.boostC ? 0.9 : 0),
  });

  // blob shadow
  const gh = Math.max(terrainHeight(phys.pos.x, phys.pos.z), WATER_Y);
  const agl = phys.pos.y - gh;
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

  const speed = phys.vel.length();
  const power = clamp(phys.jetRear / 5600 * 1.2 + phys.jetLift / 7800 * 0.8, 0, 1.6);
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
      assist: phys.assist,
      rings: ringIndex + '/' + totalRings,
      time: runStart && !finished ? fmtTime(performance.now() - runStart) : (finished ? fmtTime(lastRunMs) : '—'),
      best: bestTime !== null ? bestTime.toFixed(1) + ' שנ\'' : '',
    });
  }

  renderer.render(scene, camera);
  if (firstFrame) {
    firstFrame = false;
    document.body.dataset.ok = '1'; // headless smoke-test marker
  }
}
animate();
