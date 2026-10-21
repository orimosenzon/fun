// הלולאה הראשית: טוענים את MuJoCo, בונים עולם ורובוט, ומריצים פיזיקה בצעדים קבועים של 2 מ"ש.
import * as THREE from 'three';
import loadMujoco from 'mujoco';
import { World, DT } from './sim/world.js';
import { Robot } from './ctrl/robot.js';
import { createRenderer, buildEnvironment, buildSceneObjects } from './render/env.js';
import { buildRobot } from './render/robotmodel.js';
import { CameraRig } from './app/camera.js';
import { Overlay } from './app/overlay.js';
import { Input } from './app/input.js';
import { HUD } from './app/hud.js';
import { SCENES } from './sim/scenes.js';
import { TaskRunner, TASKS } from './tasks/tasks.js';

const $ = (id) => document.getElementById(id);
const params = new URLSearchParams(location.search);
const store = {
  get(k, d) { try { const v = localStorage.getItem('optimus.' + k); return v === null ? d : JSON.parse(v); } catch { return d; } },
  set(k, v) { try { localStorage.setItem('optimus.' + k, JSON.stringify(v)); } catch { /* מצב פרטי */ } },
};

const mj = await loadMujoco();
const container = $('view');
const renderer = createRenderer(container);
const env = buildEnvironment(renderer);
const scene = env.scene;
const camera = new THREE.PerspectiveCamera(45, container.clientWidth / container.clientHeight, 0.02, 200);
const rig = new CameraRig(camera, renderer.domElement);
const input = new Input();
const hud = new HUD();

// מערכת הצירים של MuJoCo היא z למעלה. עוטפים הכול בקבוצה מסובבת, כדי לעבוד בפנים בקואורדינטות של הסימולציה
const simRoot = new THREE.Group();
simRoot.rotation.x = -Math.PI / 2;
scene.add(simRoot);

const app = {
  mj, world: null, robot: null, model: null, objects: null, overlay: null, tasks: null,
  sceneKey: params.get('scene') || store.get('scene', 'lab'),
  paused: false, slow: 1, acc: 0, push: null, xray: false,
};
window.__app = app;
app.rig = rig;

function loadScene(key) {
  if (!SCENES[key]) key = 'lab';
  app.sceneKey = key; store.set('scene', key);
  if (app.model) simRoot.remove(app.model.root);
  if (app.objects) simRoot.remove(app.objects.group);
  if (app.overlay) app.overlay.dispose();
  if (app.world) app.world.dispose();
  const def = SCENES[key].build();
  // לבדיקות: נקודת התחלה אחרת לרובוט מה-URL
  if (params.get('start')) { const [x, y, yaw] = params.get('start').split(',').map(Number); def.start = { x, y, yaw }; }
  app.world = new World(mj, def);
  app.robot = new Robot(app.world);
  app.model = buildRobot(app.world);
  simRoot.add(app.model.root);
  app.objects = buildSceneObjects(app.world, def.looks);
  simRoot.add(app.objects.group);
  app.overlay = new Overlay(simRoot, app);
  app.tasks = new TaskRunner(app);
  app.def = def;
  hud.setScene(SCENES[key], def);
  rig.init = false;
  render();
}

// ---------- צעד סימולציה ----------
function simStep() {
  const { mj, world, robot } = app;
  const m = world.model, d = world.data;
  mj.mj_step1(m, d);
  app.tasks.update(DT);
  robot.control();
  d.xfrc_applied.fill(0);
  if (app.push && app.push.t > 0) {
    const b = app.push.body;
    for (let i = 0; i < 3; i++) d.xfrc_applied[6 * b + i] = app.push.F[i];
    app.push.t -= DT;
  }
  mj.mj_step2(m, d);
  world.t += DT;
}

// ---------- ציור ----------
const eye = { pos: new THREE.Vector3(), quat: new THREE.Quaternion() };
const EYE_TILT = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1, 0, 0), -15 * Math.PI / 180);
function render() {
  const d = app.world.data;
  app.model.update(d);
  app.objects.update(d);
  app.overlay.update();
  const p = d.xpos;
  const pelvis = new THREE.Vector3(p[3], p[5], -p[4]);
  // מצלמת העיניים: הגוף 'head', בכיוון x שלו
  const hid = app.robot.wbc.ids.head;
  const hp = new THREE.Vector3(p[3 * hid] , p[3 * hid + 1], p[3 * hid + 2]);
  const R = d.xmat.subarray(9 * hid, 9 * hid + 9);
  const fwd = new THREE.Vector3(R[0], R[3], R[6]), up = new THREE.Vector3(R[2], R[5], R[8]);
  const eyeSim = hp.clone().addScaledVector(fwd, 0.115).addScaledVector(up, 0.135);
  const toThree = (v) => new THREE.Vector3(v.x, v.z, -v.y);
  eye.pos.copy(toThree(eyeSim));
  const f3 = toThree(fwd), u3 = toThree(up);
  const m4 = new THREE.Matrix4().lookAt(new THREE.Vector3(), f3, u3);
  // המצלמות מוטות 15° למטה בתוך הראש (הנחה שלנו, כמו ברוב הרובוטים): בלי זה, ראש זקוף רואה רק מ-2.3 מ' והלאה,
  // והשולחן שהרובוט עובד עליו נשאר מחוץ לתמונה. ההטיה היא בתצוגה בלבד ולא משנה את הפיזיקה
  eye.quat.setFromRotationMatrix(m4).multiply(EYE_TILT);
  app.model.headVisible(rig.mode !== 'eyes');
  rig.update(pelvis, eye, 1 / 60);
  // המצלמה בקואורדינטות הסימולציה (למשימות שמסתכלות על הצופה)
  app.cameraTarget = [camera.position.x, -camera.position.z, camera.position.y];
  renderer.setScissorTest(false);
  renderer.setViewport(0, 0, container.clientWidth, container.clientHeight);
  renderer.render(scene, camera);
  // חלון הפינה: רינדור שני לתוך המלבן של החלון. במבט רגיל הוא מראה מה הרובוט רואה,
  // ובמבט מהעיניים הוא מתהפך ומראה את הרובוט מבחוץ
  if (app.pip) {
    const el = $('pip').getBoundingClientRect(), H = container.clientHeight;
    if (el.width > 0) {
      const eyes = rig.mode === 'eyes', cam = eyes ? rig.view : eyeCam;
      if (!eyes) { eyeCam.position.copy(eye.pos); eyeCam.quaternion.copy(eye.quat); }
      cam.aspect = el.width / el.height; cam.updateProjectionMatrix();
      app.model.headVisible(eyes);
      renderer.setScissorTest(true);
      renderer.setScissor(el.left, H - el.bottom, el.width, el.height);
      renderer.setViewport(el.left, H - el.bottom, el.width, el.height);
      renderer.render(scene, cam);
      renderer.setScissorTest(false);
      app.model.headVisible(!eyes);
    }
  }
}
const eyeCam = new THREE.PerspectiveCamera(70, 1.6, 0.03, 100);
window.__simStep = simStep; window.__render = render;

// ---------- לולאה ----------
let last = performance.now(), fpsAcc = 0, fpsN = 0;
function frame(now) {
  requestAnimationFrame(frame);
  const dt = Math.min(0.05, (now - last) / 1000); last = now;
  fpsAcc += dt; fpsN++;
  if (fpsAcc > 1) { window.__fps = fpsN / fpsAcc; fpsAcc = 0; fpsN = 0; }
  if (!app.world) return;
  input.apply(app, dt);
  if (!app.paused) {
    app.acc += dt * app.slow;
    let n = 0;
    const t0 = performance.now();
    while (app.acc >= DT && n < 40) {
      simStep(); app.acc -= DT; n++;
      if (performance.now() - t0 > 22) { app.acc = 0; break; }   // מחשב איטי: מאטים את הזמן במקום לקפוא
    }
  }
  render();
  hud.update(app, dt);
}

addEventListener('resize', () => {
  camera.aspect = container.clientWidth / container.clientHeight; camera.updateProjectionMatrix();
  renderer.setSize(container.clientWidth, container.clientHeight);
});

hud.bind(app, { loadScene, rig, input, store, camera, renderer });
input.bind(app, { rig, hud, loadScene, camera, renderer, simRoot });
loadScene(app.sceneKey);
$('loading').classList.add('hidden');
if (params.get('clean')) { $('hud').classList.add('hidden'); app.overlay.setVisible(false); }
if (params.get('auto')) { $('intro').classList.add('hidden'); app.paused = false; }
if (params.get('task')) app.tasks.start(params.get('task'), params.get('targs') ? JSON.parse(params.get('targs')) : {});
if (params.get('pip')) hud.setPip(true);
if (params.get('cam')) rig.setMode(params.get('cam'));
requestAnimationFrame(frame);
