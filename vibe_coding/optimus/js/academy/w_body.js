// ווידג'טים שמשתמשים במנוע הפיזיקה עצמו: אנטומיה (המודל של הסימולטור) ומעבדה חיה עם הבקר האמיתי.
import { animate, slider, f } from './ui.js';
import { World, DT } from '../sim/world.js';
import { Robot } from '../ctrl/robot.js';
import { buildRobot } from '../render/robotmodel.js';

let mjPromise = null;
const loadMj = () => (mjPromise ||= import('mujoco').then(m => m.default()));

async function threeSetup(cv, { fov = 32, pos = [2.4, 1.6, 2.6], target = [0, 0.95, 0] } = {}) {
  const THREE = await import('three');
  const { OrbitControls } = await import('three/addons/controls/OrbitControls.js');
  const { RoomEnvironment } = await import('three/addons/environments/RoomEnvironment.js');
  const renderer = new THREE.WebGLRenderer({ canvas: cv, antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  const scene = new THREE.Scene();
  scene.environment = new THREE.PMREMGenerator(renderer).fromScene(new RoomEnvironment(), 0.03).texture;
  scene.environmentIntensity = 0.8;
  scene.add(new THREE.HemisphereLight(0xffffff, 0x404040, 0.8));
  const dl = new THREE.DirectionalLight(0xffffff, 2.2); dl.position.set(3, 6, 4); scene.add(dl);
  const cam = new THREE.PerspectiveCamera(fov, 2, 0.05, 50);
  cam.position.set(...pos);
  const ctl = new OrbitControls(cam, cv); ctl.enableDamping = true; ctl.target.set(...target); ctl.enablePan = false;
  ctl.minDistance = 0.8; ctl.maxDistance = 7;
  const grid = new THREE.GridHelper(4, 16, 0x4a4a46, 0x2a2a28); scene.add(grid);
  const sim = new THREE.Group(); sim.rotation.x = -Math.PI / 2; scene.add(sim);
  const resize = () => { const r = cv.getBoundingClientRect(); renderer.setSize(r.width, r.height, false); cam.aspect = r.width / r.height; cam.updateProjectionMatrix(); };
  addEventListener('resize', resize); resize();
  return { THREE, renderer, scene, cam, ctl, sim };
}

// ================= אנטומיה =================
const PARTS = {
  head: { name: 'ראש', bodies: ['head'], text: 'קסדה שחורה מבריקה, ומאחורי מגן הפנים המראתי: המצלמות. אופטימוס רואה רק עם מצלמות, כמו המכוניות של טסלה, בלי LIDAR. בצוואר שני מפרקים (Gen 2 הוסיף אותם), כך שהראש יכול להסתכל על מה שהידיים עושות.' },
  torso: { name: 'גו', bodies: ['torso'], text: 'בחזה: הסוללה (2.3 קוט"ש), המחשב (היום מחשב FSD, בהמשך שבב AI5) וחלק מהקירור. 17 ק"ג, כמעט שליש מהמסה. ממוקם גבוה, וזה בכוונה: מטוטלת הפוכה עם מסה גבוהה נופלת לאט יותר ונותנת לבקר יותר זמן לתקן.' },
  pelvis: { name: 'אגן ומותן', bodies: ['pelvis', 'abdomen'], text: 'הגוף ה"צף" של הרובוט: אין לו מנוע שמחבר אותו לעולם, ולכן הוא נשלט רק דרך הרגליים. במותן שני מפעילים סיבוביים של 180 נ"מ, שמאפשרים להתכופף ולהסתובב בלי להזיז את הרגליים.' },
  arm: { name: 'זרועות', bodies: ['l_upper_arm', 'r_upper_arm', 'l_forearm', 'r_forearm'], text: 'בכל זרוע: 3 מפרקים בכתף, מרפק, וסיבוב של האמה. ב-V3 האמה עבה יותר, כי בתוכה יושבים 25 המנועים שמושכים את גידי האצבעות.' },
  hand: { name: 'ידיים', bodies: ['l_hand', 'r_hand'], prefix: ['l_thumb', 'l_index', 'l_middle', 'l_ring', 'l_little', 'r_thumb', 'r_index', 'r_middle', 'r_ring', 'r_little'], text: '22 דרגות חופש בכל צד: 4 בכל אצבע ו-2 בשורש. קצות אצבעות עם חיישני מגע. הכי פחות ברור מה ייכנס בסוף ל-V3: מאסק אמר באפריל 2026 שהתכנון שבפטנט הוחלף.' },
  leg: { name: 'רגליים', bodies: ['l_thigh', 'r_thigh', 'l_shin', 'r_shin'], text: 'שלושה מפרקים בירך (מפעילים סיבוביים), וברך עם מפעיל קווי של 8,000 ניוטון. ב-V3 השוק מכוסה ב"מגף" שחור, ומתחתיו שני מפעילים קוויים שמניעים את הקרסול, כמו שרירי השוק של אדם.' },
  foot: { name: 'כפות רגליים', bodies: ['l_foot', 'r_foot'], text: 'סוליה של 24.5 על 10 ס"מ: זה כל שטח התמיכה. Gen 2 הוסיף חיישני כוח ומומנט בכף הרגל ובהונות מפרקיות. בסימולטור כף הרגל קשיחה.' },
};
export async function anatomy(root) {
  const cv = root.querySelector('canvas.three');
  const mj = await loadMj();
  const T = await threeSetup(cv, { fov: 32, pos: [2.0, 1.3, 2.0], target: [0, 0.86, 0] });
  const { THREE } = T;
  const world = new World(mj, {});
  mj.mj_forward(world.model, world.data);
  const model = buildRobot(world);
  T.sim.add(model.root);
  model.update(world.data);
  // חומרים נפרדים לכל חלק, כדי שאפשר יהיה להאיר אותו
  const byBody = {};
  for (const p of model.parts) {
    const name = world.names.body[p.id];
    p.obj.traverse(o => { if (o.isMesh) { o.material = o.material.clone(); o.userData.base = o.material.emissive?.clone(); } });
    byBody[name] = p.obj;
  }
  // צירי המפרקים: נקודות לבנות
  const d = world.data, m = world.model;
  for (let j = 1; j < m.njnt; j++) {
    if (m.jnt_type[j] !== 3) continue;
    const name = world.names.joint[j] || '';
    if (/thumb|index|middle|ring|little/.test(name)) continue;
    const s = new THREE.Mesh(new THREE.SphereGeometry(0.014, 10, 8), new THREE.MeshBasicMaterial({ color: 0xffffff, depthTest: false, transparent: true, opacity: 0.85 }));
    s.renderOrder = 5;
    s.position.set(d.xanchor[3 * j], d.xanchor[3 * j + 1], d.xanchor[3 * j + 2]);
    T.sim.add(s);
  }
  const list = root.querySelector('.comp-list'), info = root.querySelector('.comp-info');
  const keys = Object.keys(PARTS);
  const btns = keys.map(k => { const b = document.createElement('button'); b.textContent = PARTS[k].name; b.onclick = () => pick(k); list.appendChild(b); return b; });
  function objsOf(k) {
    const P = PARTS[k], out = [];
    for (const [name, o] of Object.entries(byBody)) if (P.bodies.includes(name) || P.prefix?.some(pf => name.startsWith(pf))) out.push(o);
    return out;
  }
  function pick(k) {
    btns.forEach((b, i) => b.classList.toggle('on', keys[i] === k));
    for (const o of Object.values(byBody)) o.traverse(x => { if (x.isMesh && x.material.emissive) { x.material.emissive.set(0x000000); x.material.opacity = 1; } });
    for (const o of objsOf(k)) o.traverse(x => { if (x.isMesh && x.material.emissive) x.material.emissive.set(0x3a2a0a); });
    info.innerHTML = `<b>${PARTS[k].name}</b><p>${PARTS[k].text}</p>`;
  }
  let auto = true;
  cv.addEventListener('pointerdown', () => { auto = false; });
  animate(root, (dt) => {
    if (auto) { const a = 0.25 * dt; const p = T.cam.position; const x = p.x, z = p.z; p.x = x * Math.cos(a) - z * Math.sin(a); p.z = x * Math.sin(a) + z * Math.cos(a); }
    T.ctl.update(); T.renderer.render(T.scene, T.cam);
  });
}

// ================= מעבדה חיה =================
const GROUPS = [
  ['ברכיים', ['l_knee', 'r_knee']], ['ירכיים', ['l_hip_pitch', 'r_hip_pitch', 'l_hip_roll', 'r_hip_roll']], ['קרסוליים', ['l_ankle_pitch', 'r_ankle_pitch']],
  ['מותן', ['waist_pitch', 'waist_yaw']], ['כתף ימין', ['r_shoulder_pitch', 'r_shoulder_roll']], ['מרפק ימין', ['r_elbow']],
];
export async function lab(root) {
  const cv = root.querySelector('canvas.three');
  const mj = await loadMj();
  const T = await threeSetup(cv, { fov: 30, pos: [2.0, 1.3, 2.4], target: [0.1, 0.85, 0] });
  root.querySelector('.lab-loading')?.remove();
  let world, robot, model;
  const make = () => {
    if (model) T.sim.remove(model.root);
    world?.dispose();
    world = new World(mj, {});
    robot = new Robot(world);
    model = buildRobot(world);
    T.sim.add(model.root);
  };
  make();
  const st = { crouch: 0, lean: 0, reach: 0, height: 1.0, push: null };
  const ctrls = root.querySelector('.ctrls');
  slider(ctrls, { label: 'כפיפת ברכיים', min: 0, max: 0.22, step: 0.01, value: 0, fmt: (v) => `${Math.round(v * 100)} ס"מ`, onInput: (v) => { st.crouch = v; } });
  slider(ctrls, { label: 'הטיית גו', min: 0, max: 0.6, step: 0.02, value: 0, fmt: (v) => `${Math.round(v * 57.3)}°`, onInput: (v) => { st.lean = v; } });
  slider(ctrls, { label: 'יד ימין קדימה', min: 0, max: 0.55, step: 0.01, value: 0, fmt: (v) => `${Math.round(v * 100)} ס"מ`, onInput: (v) => { st.reach = v; } });
  slider(ctrls, { label: 'גובה היד', min: 0.6, max: 1.5, step: 0.01, value: 1.0, fmt: (v) => `${v.toFixed(2)} מ'`, onInput: (v) => { st.height = v; } });
  const row = document.createElement('div'); row.className = 'btns'; ctrls.appendChild(row);
  const btn = (label, fn) => { const b = document.createElement('button'); b.textContent = label; b.onclick = fn; row.appendChild(b); };
  btn('דחיפה מלפנים', () => { st.push = { F: [-180, 0], t: 0.1 }; });
  btn('דחיפה מהצד', () => { st.push = { F: [0, 160], t: 0.1 }; });
  btn('איפוס', () => make());
  const bars = root.querySelector('.lab-bars');
  bars.innerHTML = GROUPS.map(([n]) => `<div class="lb"><span>${n}</span><div class="tr"><div class="fl"></div></div><b>0%</b></div>`).join('');
  const fills = [...bars.querySelectorAll('.fl')], vals = [...bars.querySelectorAll('b')];
  let acc = 0, ui = 0;
  animate(root, (dt) => {
    const { model: m, data: d } = world;
    robot.crouch = st.crouch; robot.lean = st.lean;
    if (st.reach > 0.02) {
      const p = robot.fromLocal([0.1 + st.reach, -0.22, 0]); p[2] = st.height;
      robot.handTargets.r = { pos: p, w: 90, kp: 160 };
    } else robot.handTargets.r = null;
    acc += Math.min(dt, 0.05);
    let n = 0;
    while (acc >= DT && n < 30) {
      mj.mj_step1(m, d); robot.control();
      d.xfrc_applied.fill(0);
      if (st.push && st.push.t > 0) { const ti = robot.wbc.ids.torso; d.xfrc_applied[6 * ti] = st.push.F[0]; d.xfrc_applied[6 * ti + 1] = st.push.F[1]; st.push.t -= DT; }
      mj.mj_step2(m, d);
      acc -= DT; n++;
    }
    model.update(d);
    // המצלמה עוקבת אחרי הרובוט
    const px = d.qpos[0], py = d.qpos[1];
    T.ctl.target.lerp(new T.THREE.Vector3(px, 0.85, -py), 0.05);
    T.ctl.update(); T.renderer.render(T.scene, T.cam);
    if ((ui += dt) > 0.12) {
      ui = 0;
      GROUPS.forEach(([, js], i) => {
        const v = Math.max(...js.map(jn => { const k = robot.jidx[jn]; return Math.abs(robot.tau[k]) / robot.wbc.joints[k].torque; }));
        fills[i].style.width = Math.min(100, v * 100) + '%'; vals[i].textContent = Math.round(v * 100) + '%';
      });
      if (robot.fallen) { robot.fallen = false; make(); }
    }
  });
}
