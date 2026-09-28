// MJ-5 — the game. Real-time version of the film's engine: the same rigid-body physics,
// turbojets and vane allocation, flown through PilotAssist from keyboard / gamepad / touch.

import * as THREE from 'three';
import { JetBike, makeWind, PROBES } from './physics.js';
import { PilotAssist } from './pilot.js';
import { buildCourse, RING_R, LANDING } from './course.js';
import { heightAt, surfaceAt, WATER_Y, PAD, valleyZ } from './terrain.js';
import { G, IDLE_N, LIFT_TMAX, MAIN_TMAX, MAIN_DIR } from './vehicle.js';
import { qrot, len } from './vmath.js';
import { buildWorld, SUN_DIR } from './render/world.js';
import { buildBike } from './render/bikeModel.js';
import { Particles, Plumes } from './render/fx.js';
import { Post } from './render/post.js';
import { buildRings } from './render/rings.js';
import { Explosion } from './render/explosion.js';
import { makeHud, fmtTime } from './hud.js';
import { Input, makeTouch } from './input.js';
import { JetAudio } from './audio.js';

const params = new URLSearchParams(location.search);
const DEMO = params.has('demo');
const QUALITY = +(params.get('q') || (matchMedia('(pointer: coarse)').matches ? 0.6 : 1));
let renderScale = +(params.get('scale') || (QUALITY < 1 ? 0.7 : 1));
const frame = () => new Promise((r) => requestAnimationFrame(() => r()));
// loading bar (index.html): 0–12 % module download, 12–90 % world, then bike and GPU warm-up
const load = (f, label) => window.loader?.set(f, label);
const T_LOAD = performance.now();

load(0.12, 'מתחיל לבנות את העמק');
await frame();
const renderer = new THREE.WebGLRenderer({ antialias: false, powerPreference: 'high-performance', preserveDrawingBuffer: params.has('shots') });
renderer.setPixelRatio(1);
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
document.getElementById('stage').appendChild(renderer.domElement);
let W = innerWidth, H = innerHeight;
renderer.setSize(W, H);

const scene = new THREE.Scene();
const hazeScene = new THREE.Scene();
const rings = buildCourse();
const corridor = [];
{ const pts = [[PAD[0], 2, PAD[2]], ...rings.map((r) => r.p), [LANDING.x, heightAt(LANDING.x, LANDING.z) + 3, LANDING.z]];
  for (let i = 0; i < pts.length - 1; i++) corridor.push([pts[i], pts[i + 1]]); }
const world = await buildWorld(scene, renderer, { quality: QUALITY, clearings: [], corridor }, (f, label) => load(0.12 + 0.78 * f, label));
load(0.9, 'מרכיב את האופנוע');
await frame();
const model = buildBike(world.envTex);
scene.add(model.root);
const particles = new Particles(scene, SUN_DIR);
particles.hAt = world.hAt;
const plumes = new Plumes(scene, hazeScene);
const ringFx = buildRings(scene, rings);
const boom = new Explosion(scene, world.hAt);
const post = new Post(renderer, Math.round(W * renderScale), Math.round(H * renderScale));
post.blur = 0.55;
const camera = new THREE.PerspectiveCamera(60, W / H, 0.2, 40000);
const hud = makeHud(document.getElementById('overlay'));
const input = new Input();
if (matchMedia('(pointer: coarse)').matches || params.has('touch')) makeTouch(document.getElementById('overlay'), input);
const audio = new JetAudio();
const EXPOSURE = 0.64;

// tree collision grid (20 m cells)
const treeGrid = new Map();
for (const t of world.trees) {
  const k = `${Math.floor(t[0] / 20)},${Math.floor(t[2] / 20)}`;
  if (!treeGrid.has(k)) treeGrid.set(k, []);
  treeGrid.get(k).push(t);
}

// ---------------- simulation state
const wind = makeWind(11);
const bike = new JetBike([PAD[0], PAD[1], PAD[2]], 0);
const pilot = new PilotAssist();
const PHYS_DT = 1 / 1000;
let acc = 0, simT = 0, stepN = 0;
let cmd = pilot.cmd;
let state = 'menu';          // menu | fly | crashed | finished
let ringIndex = 0, raceT = 0, raceOn = false, checkpoint = null, crashT = 0, heat = 0, overheated = false;
let best = +(localStorage.getItem('mj5-best') || 0) || null;
let assist = true, prevContact = 0, lastVy = 0, finishInfo = null;
const CAMS = ['chase', 'far', 'cockpit', 'tv'];
const CAM_NAMES = { chase: 'מצלמה: מרדף', far: 'מצלמה: רחוקה', cockpit: 'מצלמה: עיני הרוכב', tv: 'מצלמה: שידור' };
let camMode = CAMS.includes(params.get('cam')) ? params.get('cam') : 'chase';
const initialYaw = Math.atan2(-(rings[0].p[2] - PAD[2]), rings[0].p[0] - PAD[0]);

function resetAt(pos, yaw, running) {
  bike.reset(pos, yaw, 0);
  pilot.reset(yaw);
  for (const e of [...bike.lift, bike.main]) {
    if (running) { e.phase = 'run'; e.N = e === bike.main ? IDLE_N : 0.86; } else { e.phase = 'off'; e.N = 0; }
    e.T = e.Tcore = 0;
  }
  input.state.throttle = 0;
  particles.reset();
  heat = 0; overheated = false;
  post.cut = true;
}
function startEngines() {
  bike.lift.forEach((e, i) => setTimeout(() => e.start(), 150 + i * 650));
  setTimeout(() => bike.main.start(), 150 + 4 * 650);
}
// debug: ?pose=final starts hovering near the landing target with all rings done
function applyPose() {
  if (params.get('pose') !== 'final') return;
  ringIndex = rings.length; raceOn = true;
  const y = heightAt(LANDING.x - 30, LANDING.z) + 8;
  resetAt([LANDING.x - 30, y, LANDING.z], 0, true);
  checkpoint = { p: [LANDING.x - 30, y, LANDING.z], yaw: 0 };
}
function newRace() {
  ringIndex = 0; raceT = 0; raceOn = false; checkpoint = null; finishInfo = null;
  resetAt([PAD[0], PAD[1] + bike.mp.cg[1] - 0.012, PAD[2]], initialYaw, false);
  boom.clear();
  ringFx.reset();
  state = 'fly';
  hud.showScreen('');
  hud.setVisible(true);
  startEngines();
  hud.message('הנעת מנועים… המתן לסיבוב סרק ואז <b>E</b> להמראה', 3500);
  applyPose();
}
resetAt([PAD[0], PAD[1] + bike.mp.cg[1] - 0.012, PAD[2]], initialYaw, false);

// ---------------- autopilot (demo / attract mode): steer toward the next target
function autopilot() {
  const landing = ringIndex >= rings.length;
  const target = !landing ? rings[ringIndex].p : [LANDING.x, heightAt(LANDING.x, LANDING.z) + 1, LANDING.z];
  const prev = ringIndex > 0 ? rings[ringIndex - 1].p : [PAD[0], 3, PAD[2]];
  const f = qrot(bike.q, [1, 0, 0]);
  const yaw = Math.atan2(-f[2], f[0]);
  const dx = target[0] - bike.p[0], dz = target[2] - bike.p[2], dist = Math.hypot(dx, dz);
  let err = Math.atan2(-dz, dx) - yaw;
  err = Math.atan2(Math.sin(err), Math.cos(err));
  const V = Math.hypot(bike.v[0], bike.v[2]);
  const onPad = bike.contactN > 0.2 * bike.mp.m * G;
  const inp = { stickX: 0, stickY: 0, rudder: 0, climb: 0, throttle: 0, boost: false, assist: true };
  if (onPad && !raceOn) { inp.climb = bike.lift.every((e) => e.phase === 'run') ? 1 : 0; return inp; }
  inp.stickX = Math.max(-1, Math.min(1, -err * 2.2)) * (V > 8 ? 1 : 0);
  inp.rudder = V < 8 ? Math.max(-1, Math.min(1, -err * 2)) : 0;
  // altitude: follow the straight line between rings, never below the terrain ahead + margin
  const legL = Math.hypot(target[0] - prev[0], target[2] - prev[2]) || 1;
  const u = Math.max(0, Math.min(1, 1 - dist / legL));
  let altDes = prev[1] + (target[1] - prev[1]) * Math.min(1, u * 1.25);
  let ahead = -1e9;
  for (let k = 0; k <= 6; k++) {
    const la = k * (6 + V * 0.35);
    ahead = Math.max(ahead, surfaceAt(bike.p[0] + bike.v[0] / (V || 1) * la, bike.p[2] + bike.v[2] / (V || 1) * la));
  }
  if (!landing) {
    altDes = Math.max(altDes, ahead + (dist < 60 ? 3.5 : 6));
    const vWant = Math.abs(err) > 0.35 ? 22 : 40;
    inp.throttle = Math.max(0, Math.min(1, 0.55 + (vWant - V) * 0.08));
    if (V > vWant + 6) inp.stickY = -Math.min(0.6, (V - vWant) * 0.04);
  } else if (dist > 90) {
    inp.throttle = V < 30 ? 0.6 : 0.1;
    altDes = Math.max(heightAt(bike.p[0], bike.p[2]) + 12, ahead + 6);
  } else {
    // precision landing: velocity command toward the target, pitch/bank as acceleration
    const fwd = [Math.cos(yaw), -Math.sin(yaw)], right = [Math.sin(yaw), Math.cos(yaw)];   // (x, z)
    const vWant = Math.min(22, dist * 0.35);
    const ux = dx / (dist || 1), uz = dz / (dist || 1);
    const wx = ux * vWant, wz = uz * vWant;
    const ef = (wx - bike.v[0]) * fwd[0] + (wz - bike.v[2]) * fwd[1];
    const er = (wx - bike.v[0]) * right[0] + (wz - bike.v[2]) * right[1];
    inp.stickY = Math.max(-1, Math.min(1, ef * 0.14));
    inp.stickX = Math.max(-1, Math.min(1, er * 0.14));
    inp.rudder = 0; inp.throttle = 0;
    const settled = dist < 2.5 && V < 1.2;
    altDes = heightAt(bike.p[0], bike.p[2]) + 0.86 + (settled ? -1.5 : dist < 12 ? 4 : 9);
    altDes = Math.max(altDes, settled ? -1e9 : ahead + 3);
  }
  inp.climb = Math.max(-1, Math.min(1, (altDes - bike.p[1]) * 0.3 - bike.v[1] * 0.15));
  if (landing && dist < 2.5 && V < 1.2) inp.climb = Math.max(inp.climb, -0.3);   // gentle touchdown
  return inp;
}

// ---------------- collisions
function crash(reason, water) {
  if (state !== 'fly') return;
  state = 'crashed'; crashT = 0;
  console.log('crash', reason, bike.p.map((v) => v.toFixed(2)).join(','), 'v', bike.v.map((v) => v.toFixed(2)).join(','), 'contact', bike.contactN.toFixed(0));
  boom.trigger(bike.p, bike.v, water);
  audio.boom(water); audio.silenceEngines();
  model.root.visible = false;
  const txt = { ground: 'התרסקות בקרקע', water: 'נפילה למים', tree: 'פגיעה בעץ', ring: 'פגיעה בטבעת', hard: 'נחיתה קשה מדי', fuel: 'נגמר הדלק' }[reason];
  hud.message(`${txt}<br><small>חוזרים לנקודת הביקורת…</small>`, 3200, 'bad');
}
function checkCollisions() {
  const W = bike.mp.m * G;
  // airframe / rider touching ground or water
  for (const d of PROBES) {
    const p = bike.toWorld(d);
    const h = heightAt(p[0], p[2]);
    if (h < WATER_Y && p[1] < WATER_Y) return crash('water', true);
    if (p[1] < h - 0.03) return crash('ground', false);
  }
  // landing gear: too fast a touchdown collapses it
  if (bike.contactN > 0.05 * W && prevContact <= 0.05 * W) {
    const vh = Math.hypot(bike.v[0], bike.v[2]);
    if (lastVy < -5.5 || vh > 16) return crash('hard', false);
  }
  // trees
  const cx = Math.floor(bike.p[0] / 20), cz = Math.floor(bike.p[2] / 20);
  for (let i = -1; i <= 1; i++) for (let j = -1; j <= 1; j++) {
    const list = treeGrid.get(`${cx + i},${cz + j}`);
    if (!list) continue;
    for (const [x, y, z, s, r, sp] of list) {
      const hgt = bike.p[1] - y;
      if (hgt < 0 || hgt > s) continue;
      const dxz = Math.hypot(bike.p[0] - x, bike.p[2] - z);
      let rad;
      if (sp === 2) {
        const wdt = 1.2 + 0.5 * r, u = (hgt - 0.68 * s) / (0.3 * s);
        rad = Math.abs(u) < 1 ? 0.3 * s * wdt * Math.sqrt(1 - u * u) : 0.04 * s;
      } else {
        const wdt = 0.75 + 0.35 * r, u = (hgt / s - 0.1) / 0.8;
        rad = u < 0 ? 0.03 * s : (0.3 * Math.pow(Math.max(0, 1 - u), 0.9) + 0.035) * s * wdt * 0.9;
      }
      if (dxz < rad + 0.7) return crash('tree', false);
    }
  }
  // ring rims (torus centre circle radius RING_R + 0.55, tube 0.42)
  for (const rg of rings.slice(ringIndex)) {   // passed rings are hidden and no longer solid
    const dx = bike.p[0] - rg.p[0], dy = bike.p[1] - rg.p[1], dz = bike.p[2] - rg.p[2];
    if (dx * dx + dy * dy + dz * dz > 400) continue;
    const along = dx * rg.n[0] + dy * rg.n[1] + dz * rg.n[2];
    const rx = dx - along * rg.n[0], ry = dy - along * rg.n[1], rz = dz - along * rg.n[2];
    const radial = Math.hypot(rx, ry, rz);
    if (Math.hypot(radial - (RING_R + 0.55), along) < 0.42 + 0.75) return crash('ring', false);
  }
}

// ring crossing: sign change of the distance to the ring plane, inside the radius
function checkRing(p0) {
  if (ringIndex >= rings.length) return;
  const rg = rings[ringIndex];
  const s0 = (p0[0] - rg.p[0]) * rg.n[0] + (p0[1] - rg.p[1]) * rg.n[1] + (p0[2] - rg.p[2]) * rg.n[2];
  const p1 = bike.p;
  const s1 = (p1[0] - rg.p[0]) * rg.n[0] + (p1[1] - rg.p[1]) * rg.n[1] + (p1[2] - rg.p[2]) * rg.n[2];
  if (s0 < 0 && s1 >= 0) {
    const u = s0 / (s0 - s1);
    const x = p0[0] + (p1[0] - p0[0]) * u - rg.p[0], y = p0[1] + (p1[1] - p0[1]) * u - rg.p[1], z = p0[2] + (p1[2] - p0[2]) * u - rg.p[2];
    if (Math.hypot(x, y, z) < RING_R) {
      ringIndex++;
      checkpoint = { p: rg.p.slice(), yaw: Math.atan2(-rg.n[2], rg.n[0]) };
      audio.chime(ringIndex * 1);
      hud.message(ringIndex < rings.length ? `טבעת ${ringIndex}/${rings.length} · ${fmtTime(raceT)}` : 'כל הטבעות! עכשיו נחיתה מדויקת במטרה', 1800, 'good');
    }
  }
}

function checkFinish() {
  if (ringIndex < rings.length || state !== 'fly') return;
  const W = bike.mp.m * G;
  const d = Math.hypot(bike.p[0] - LANDING.x, bike.p[2] - LANDING.z);
  if (bike.contactN > 0.85 * W && len(bike.v) < 0.6 && d < LANDING.r) {
    state = 'finished';
    raceOn = false;
    const bonus = Math.max(0, (LANDING.r - d) / LANDING.r) * 5;   // up to 5 s off for a bullseye
    const total = raceT - bonus;
    const isBest = !params.has('pose') && (!best || total < best);
    if (isBest) { best = total; localStorage.setItem('mj5-best', String(total)); }
    finishInfo = { time: raceT, d, total, isBest };
    audio.chime(12);
    hud.setVisible(false);
    hud.showScreen(`<div class="card"><h1>נחיתה!</h1>
      <div class="big">${fmtTime(total)}</div>
      <p>זמן טיסה ${fmtTime(raceT)} · מרחק ממרכז המטרה ${d.toFixed(2)} מ׳ (בונוס ${bonus.toFixed(1)} ש׳)</p>
      ${isBest ? '<p class="gold">שיא חדש!</p>' : `<p>השיא: ${fmtTime(best)}</p>`}
      <p class="hint">Enter / לחיצה לטיסה נוספת</p></div>`);
    bike.lift.forEach((e) => e.stop()); bike.main.stop();
  }
}

// ---------------- cameras
const cam = { pos: new THREE.Vector3(-8, 3, 8), look: new THREE.Vector3(), up: new THREE.Vector3(0, 1, 0), fov: 60, tv: null, shake: 0 };
function updateCamera(dt, t) {
  const b = new THREE.Vector3(...bike.p);
  const q = new THREE.Quaternion(bike.q[1], bike.q[2], bike.q[3], bike.q[0]);
  const v = new THREE.Vector3(...bike.v);
  const sp = v.length();
  const fwdB = new THREE.Vector3(1, 0, 0).applyQuaternion(q);
  const heading = new THREE.Vector3(fwdB.x, 0, fwdB.z).normalize();
  if (sp > 6) heading.lerp(new THREE.Vector3(v.x, v.y * 0.3, v.z).normalize(), Math.min(1, (sp - 6) / 10));
  let near = 0.2;
  model.head.visible = true;
  const k = 1 - Math.exp(-dt * 7);
  if (state === 'menu') {
    const a = t * 0.12 + 2.2;
    cam.pos.set(b.x + Math.cos(a) * 9, b.y + 1.2 + Math.sin(t * 0.2) * 0.4, b.z + Math.sin(a) * 9);
    cam.look.copy(b).add(new THREE.Vector3(0, 0.3, 0)); cam.up.set(0, 1, 0); cam.fov = 42;
  } else if (state === 'crashed' || state === 'finished') {
    const o = new THREE.Vector3(...(boom.origin || bike.p));
    cam.look.lerp(state === 'finished' ? b : o, k);
    const want = cam.look.clone().add(new THREE.Vector3(Math.cos(t * 0.3) * 18, 7, Math.sin(t * 0.3) * 18));
    cam.pos.lerp(want, 1 - Math.exp(-dt * 1.2));
    cam.up.set(0, 1, 0);
  } else if (camMode === 'cockpit') {
    model.head.visible = false;
    model.root.updateMatrixWorld(true);
    const eye = new THREE.Vector3(); model.head.getWorldPosition(eye);
    cam.pos.copy(eye).addScaledVector(fwdB, 0.08).addScaledVector(new THREE.Vector3(0, 1, 0).applyQuaternion(q), 0.09);
    cam.look.copy(cam.pos).addScaledVector(fwdB, 10).addScaledVector(new THREE.Vector3(0, 1, 0).applyQuaternion(q), -1.2);
    cam.up.set(0, 1, 0).applyQuaternion(q);
    cam.fov = 78; near = 0.05;
  } else if (camMode === 'tv') {
    if (!cam.tv || b.distanceTo(cam.tv) > 140) {
      const side = new THREE.Vector3(-heading.z, 0, heading.x).multiplyScalar(Math.random() < 0.5 ? 1 : -1);
      const p = b.clone().addScaledVector(heading, 60 + sp * 1.5).addScaledVector(side, 14 + Math.random() * 10);
      p.y = Math.max(surfaceAt(p.x, p.z) + 1.5, b.y - 4 + Math.random() * 6);
      // keep the camera clear of the rings (a ring right in front of the lens fills the frame)
      for (const rg of rings) {
        const d = p.distanceTo(new THREE.Vector3(...rg.p));
        if (d < 25) p.addScaledVector(new THREE.Vector3(p.x - rg.p[0], 0, p.z - rg.p[2]).normalize(), 25 - d + 5);
      }
      cam.tv = p;
    }
    cam.pos.copy(cam.tv);
    cam.look.lerp(b, 1 - Math.exp(-dt * 10));
    cam.up.set(0, 1, 0);
    cam.fov = THREE.MathUtils.clamp(2 * Math.atan(7 / b.distanceTo(cam.tv)) * 57.3, 8, 55);
  } else {
    const far = camMode === 'far';
    const dist = (far ? 20 : 5.6) + sp * (far ? 0.08 : 0.035);
    const want = b.clone().addScaledVector(heading, -dist).add(new THREE.Vector3(0, (far ? 6.5 : 1.75) + sp * 0.006, 0));
    cam.pos.lerp(want, 1 - Math.exp(-dt * (far ? 3 : 5)));
    cam.look.lerp(b.clone().addScaledVector(heading, 4).add(new THREE.Vector3(0, 0.9, 0)), 1 - Math.exp(-dt * 12));
    const upB = new THREE.Vector3(0, 1, 0).applyQuaternion(q);
    cam.up.set(0, 1, 0).lerp(upB, 0.22).normalize();
    cam.fov = 58 + THREE.MathUtils.clamp(sp - 12, 0, 45) * 0.32;
  }
  // never inside the ground
  const gy = surfaceAt(cam.pos.x, cam.pos.z) + 0.6;
  if (cam.pos.y < gy) cam.pos.y = gy;
  // shake: g-load changes and jet blast near the ground
  const load = Math.hypot(bike.acc[0], bike.acc[1] + G, bike.acc[2]) / G;
  const agl = bike.p[1] - surfaceAt(bike.p[0], bike.p[2]);
  const blast = state === 'fly' ? Math.max(0, 1 - agl / 8) * (bike.lift.reduce((a, e) => a + e.T, 0) / 3500) : 0;
  const shakeAmt = (camMode === 'cockpit' ? 0.004 : 0.012) * (Math.abs(load - 1) * 2 + blast * 1.5 + (state === 'crashed' ? Math.max(0, 4 - crashT * 3) : 0) + sp / 80);
  const sh = new THREE.Vector3(Math.sin(t * 41) + Math.sin(t * 67), Math.sin(t * 53) + Math.sin(t * 29), Math.sin(t * 47)).multiplyScalar(shakeAmt);
  camera.position.copy(cam.pos).add(sh);
  camera.up.copy(cam.up);
  camera.lookAt(cam.look);
  camera.fov += (cam.fov - camera.fov) * (1 - Math.exp(-dt * 4));
  camera.near = near;
  camera.updateProjectionMatrix();
}

// ---------------- UI screens
function showMenu() {
  hud.setVisible(false);
  hud.showScreen(`<div class="card wide"><h1>MJ-5</h1><h2>אופנוע סילון · מסלול העמק</h2>
    <p>12 טבעות לאורך העמק, מעל האגם, סביב הסלע ודרך הקניון, ואז נחיתה מדויקת במטרה. הפיזיקה אמיתית: חמישה מנועי סילון עם השהיית סחרור, כנפוני הסטה, גרר, אפקט קרקע, רוח ומערבולות.</p>
    <table class="keys">
      <tr><td><b>E / D</b></td><td>עלייה / ירידה (בלי מגע: שמירת גובה)</td></tr>
      <tr><td><b>← →</b></td><td>הטיה ופנייה</td></tr>
      <tr><td><b>↑ ↓</b></td><td>אף למטה / למעלה: בריחוף נע קדימה/אחורה, במהירות צולל/מטפס. ↓ חזק = בלימה</td></tr>
      <tr><td><b>W / S</b></td><td>מצערת מנוע השיוט</td></tr>
      <tr><td><b>Z / X</b></td><td>הגה כיוון (סבסוב)</td></tr>
      <tr><td><b>רווח</b></td><td>הספק חירום (המנוע מתחמם)</td></tr>
      <tr><td><b>C</b> מצלמה · <b>T</b> בקר ידני · <b>R</b> חזרה לביקורת · <b>M</b> שקט</td><td>שלט משחק נתמך</td></tr>
    </table>
    ${best ? `<p class="gold">השיא שלך: ${fmtTime(best)}</p>` : ''}
    <button id="go">התחלה (Enter)</button></div>`);
  document.getElementById('go').onclick = () => begin();
}
function begin() {
  audio.start();
  newRace();
}
showMenu();

// ---------------- main loop
const prof = { phys: 0, fx: 0, render: 0, n: 0 };
let last = performance.now(), t = 0, fpsAcc = 0, fpsN = 0, prevWarn = '';
const tmpV = new THREE.Vector3();
function loop(now) {
  const dt = Math.min(0.05, (now - last) / 1000);
  last = now; t += dt;
  const inp = DEMO ? { ...autopilot() } : input.update(dt);
  if (input.pressed('KeyC') || input.padPressed(3)) { camMode = CAMS[(CAMS.indexOf(camMode) + 1) % CAMS.length]; cam.tv = null; post.cut = true; }
  if (!DEMO) {
    if ((state === 'menu' || state === 'finished') && (input.pressed('Enter') || input.padPressed(9) || input.padPressed(0))) begin();
    if (input.pressed('KeyT') || input.padPressed(2)) { assist = !assist; pilot.lever = Math.min(1, bike.lift.reduce((a, e) => a + e.T, 0) / (4 * LIFT_TMAX * 0.9)); hud.message(assist ? 'בקר עזר: מייצב, שומר גובה, פנייה מתואמת' : 'בקר ידני: הסטיק שולט בקצבים, E/D ידית עילוי', 2200); }
    if (input.pressed('KeyM')) audio.setMuted(!audio.muted);
    if ((input.pressed('KeyR') || input.padPressed(8)) && state === 'fly') respawn();
    if (input.pressed('KeyH')) showMenu();
  } else if (state === 'menu' && t > 1) { newRace(); }
  inp.assist = DEMO ? true : assist;

  // emergency power heats the cruise engine; at the limit the FADEC cuts it until it cools
  if (inp.boost && !overheated && state === 'fly') heat = Math.min(1, heat + dt * 0.14); else heat = Math.max(0, heat - dt * 0.09);
  if (heat >= 1) overheated = true;
  if (overheated && heat < 0.35) overheated = false;
  inp.boost = inp.boost && !overheated;

  const tP0 = performance.now();
  // physics
  if (state === 'fly' || state === 'menu' || state === 'finished') {
    acc += dt;
    const p0 = bike.p.slice();
    let steps = 0;
    while (acc >= PHYS_DT && steps < 60) {
      if (stepN % 5 === 0) cmd = pilot.update(PHYS_DT * 5, bike, state === 'fly' ? inp : { stickX: 0, stickY: 0, rudder: 0, climb: 0, throttle: 0, boost: false, assist: true });
      lastVy = bike.v[1];
      prevContact = bike.contactN;
      bike.step(PHYS_DT, cmd, wind);
      simT += PHYS_DT; acc -= PHYS_DT; stepN++; steps++;
      if (state === 'fly') { checkCollisions(); if (state !== 'fly') break; }
    }
    if (steps >= 60) acc = 0;
    if (state === 'fly') {
      checkRing(p0);
      if (!raceOn && bike.contactN < 1 && ringIndex === 0 && bike.p[1] > PAD[1] + 1.2) raceOn = true;
      if (raceOn) raceT += dt;
      checkFinish();
      if (bike.fuel <= 0 && bike.contactN < 1 && bike.lift[0].N < 0.3) crash('fuel', false);
    }
  } else if (state === 'crashed') {
    crashT += dt;
    if (raceOn) raceT += dt;
    if (crashT > 3.6) respawn();
  }

  const tP1 = performance.now();
  // visuals
  const s = {
    p: bike.p, q: bike.q, v: bike.v, w: bike.w, acc: bike.acc, T: bike.lift.map((e) => e.T), N: bike.lift.map((e) => e.N),
    vanes: bike.vanes, mainT: bike.main.T, mainN: bike.main.N, egt: [...bike.lift.map((e) => e.egt), bike.main.egt],
    fuel: bike.fuel, airspeed: bike.airspeed || 0,
  };
  model.update(s, dt);
  const ex = state === 'crashed' ? [] : model.exhausts(s);
  const windNow = wind(bike.p, simT);
  particles.step(dt, ex, windNow);
  particles.upload();
  boom.step(dt, windNow);
  ringFx.update(ringIndex, t, state === 'finished');
  world.grassU.uTime.value = t;
  world.grassU.uWind.value.set(...windNow);
  const blastArr = world.grassU.uBlast.value;
  let blastSum = 0, wet = 0;
  ex.forEach((e, i) => {
    if (e.d.y > -0.2 || e.T < 30) { blastArr[i].set(0, 0, 0, 1); return; }
    const h = (e.p.y - surfaceAt(e.p.x, e.p.z)) / -e.d.y;
    const x = e.p.x + e.d.x * h, z = e.p.z + e.d.z * h;
    const U = 560 * Math.sqrt(Math.min(1, e.T / 1250)) * Math.min(1, 0.84 / Math.max(h, 0.1));
    blastArr[i].set(x, z, Math.min(3.5, U / 25), 1.2 + 0.35 * h);
    if (i < 4) { blastSum += (U / 260) ** 2; if (heightAt(x, z) < WATER_Y) wet = 1; }
  });
  world.water.material.uniforms.time.value = t * 0.45;
  const tP2 = performance.now();
  updateCamera(dt, t);
  world.updateLOD(camera.position);
  plumes.update(ex.length ? ex : model.exhausts(s).map((e) => ({ ...e, T: 0, egt: 300 })), t, post, camera);
  const bp = model.root.position;
  world.sun.target.position.copy(bp);
  world.sun.position.copy(bp).addScaledVector(SUN_DIR, 400);
  world.sun.target.updateMatrixWorld();
  post.renderFrame(scene, camera, hazeScene);
  post.toScreen(EXPOSURE, t, 1, false);
  const tP3 = performance.now();
  prof.phys += tP1 - tP0; prof.fx += tP2 - tP1; prof.render += tP3 - tP2; prof.n++;

  // sound
  if (audio.ctx) {
    const right = new THREE.Vector3().setFromMatrixColumn(camera.matrixWorld, 0);
    const camVel = tmpV.copy(camera.position).sub(cam.prevPos || camera.position).divideScalar(Math.max(dt, 1e-3));
    cam.prevPos = camera.position.clone();
    const engines = state === 'crashed' ? [] : [...bike.lift.map((e, i) => ({ e, p: ex[i]?.p })), { e: bike.main, p: ex[4]?.p }].map(({ e, p }, i) => ({
      N: e.N, T: e.T, Tmax: (i < 4 ? LIFT_TMAX : MAIN_TMAX) * 0.9, pos: p ? [p.x, p.y, p.z] : bike.p,
    }));
    if (engines.length) audio.update({
      engines, camPos: camera.position.toArray(), camRight: right.toArray(), camVel: camVel.toArray(), bikeVel: bike.v,
      onboard: camMode === 'cockpit' || camMode === 'chase', airspeed: bike.airspeed || 0, blast: blastSum, wet, fuelFlow: bike.fuelFlow || 0,
    });
  }

  // HUD
  if (state === 'fly') {
    const f = qrot(bike.q, [0, 0, 1]);
    const target = ringIndex < rings.length ? new THREE.Vector3(...rings[ringIndex].p) : new THREE.Vector3(LANDING.x, heightAt(LANDING.x, LANDING.z) + 1, LANDING.z);
    const pr = target.clone().project(camera);
    const behind = pr.z > 1;
    let mx = (pr.x * 0.5 + 0.5) * W, my = (-pr.y * 0.5 + 0.5) * H;
    if (behind) { mx = W - mx; my = H - my; }
    const edge = behind || mx < 30 || mx > W - 30 || my < 30 || my > H - 30;
    let rot = 0;
    if (edge) {
      const cx = W / 2, cy = H / 2, dx = mx - cx, dy = my - cy;
      const sc = Math.min((W / 2 - 40) / Math.abs(dx || 1e-6), (H / 2 - 40) / Math.abs(dy || 1e-6));
      mx = cx + dx * Math.min(1, sc); my = cy + dy * Math.min(1, sc);
      rot = Math.atan2(dy, dx) + Math.PI / 2;
      if (behind && sc > 1) { mx = cx + dx * sc; my = cy + dy * sc; }
    }
    hud.update({
      bike, cmd, agl: bike.p[1] - 0.86 - surfaceAt(bike.p[0], bike.p[2]), bankDeg: -Math.asin(Math.max(-1, Math.min(1, f[1]))) * 57.3,
      heat, assist: inp.assist, radar: pilot.tel.radar, camName: CAM_NAMES[camMode], ringIndex, ringCount: rings.length, raceTime: raceT, best, finished: false,
      marker: { x: mx, y: my, edge, rot, dist: target.distanceTo(new THREE.Vector3(...bike.p)) },
    });
    let w = '';
    const aglNow = bike.p[1] - 0.86 - surfaceAt(bike.p[0], bike.p[2]);
    const vhNow = Math.hypot(bike.v[0], bike.v[2]);
    if (raceOn && aglNow < 3 && vhNow > 14 && bike.contactN < 1) w = 'גובה נמוך!';
    else if (overheated) w = 'מנוע השיוט התחמם: הספק חירום מושבת';
    else if (heat > 0.75) w = 'טמפרטורת מנוע גבוהה';
    else if (bike.fuel < 6) w = 'דלק נמוך';
    else if (bike.lift.some((e) => e.phase === 'start')) w = 'הנעת מנועים…';
    if (w !== prevWarn) { hud.warning(w); prevWarn = w; if (w && !w.includes('הנעת')) audio.beep(); }
    document.body.dataset.radar = pilot.tel.radar ? 1 : 0;
  } else hud.warning('');

  // keep the frame rate up: adapt the render scale
  fpsAcc += dt; fpsN++;
  if (fpsAcc > 2) {
    const fps = fpsN / fpsAcc;
    document.body.dataset.fps = fps.toFixed(0);
    document.body.dataset.prof = `phys ${(prof.phys / prof.n).toFixed(1)} fx ${(prof.fx / prof.n).toFixed(1)} render ${(prof.render / prof.n).toFixed(1)} scale ${renderScale.toFixed(2)}`;
    prof.phys = prof.fx = prof.render = prof.n = 0;
    if (!params.has('scale') && state === 'fly') {
      const target = fps < 42 ? Math.max(0.55, renderScale - 0.1) : fps > 58 && renderScale < 1 ? Math.min(1, renderScale + 0.05) : renderScale;
      if (target !== renderScale) { renderScale = target; resize(); }
    }
    fpsAcc = 0; fpsN = 0;
  }
  input.endFrame();
  requestAnimationFrame(loop);
}

function respawn() {
  model.root.visible = true;
  boom.clear();
  const cp = checkpoint || { p: [PAD[0], PAD[1] + bike.mp.cg[1], PAD[2]], yaw: initialYaw };
  const running = !!checkpoint;
  resetAt(cp.p, cp.yaw, running);
  if (!running) startEngines();
  state = 'fly';
  hud.message(checkpoint ? 'ממשיכים מהטבעת האחרונה' : 'חזרה לרחבת ההמראה', 1500);
}

function resize() {
  W = innerWidth; H = innerHeight;
  renderer.setSize(W, H);
  post.setSize(Math.round(W * renderScale), Math.round(H * renderScale));
  post.blur = 0.55;
  camera.aspect = W / H;
  camera.updateProjectionMatrix();
}
addEventListener('resize', resize);
addEventListener('click', () => { if (state === 'menu') begin(); else if (state === 'finished') begin(); else audio.start(); });

// GPU warm-up behind the loading screen: shader compilation would otherwise freeze the first frames
load(0.93, 'מכין את כרטיס הגרפיקה');
await frame();
updateCamera(1 / 60, 0);
try { await renderer.compileAsync(scene, camera); } catch (e) { console.warn('compileAsync', e); }
load(0.97, 'מכין את כרטיס הגרפיקה');
await frame();
last = performance.now();
loop(last);          // first real frame (post passes, water mirror, shadow map) still under the loader
await frame();
console.log(`loaded in ${((performance.now() - T_LOAD) / 1000).toFixed(1)} s after modules`);
window.loader?.done();

window.game = { get state() { return state; }, get ringIndex() { return ringIndex; }, get raceT() { return raceT; }, bike, get fps() { return document.body.dataset.fps; }, begin };
