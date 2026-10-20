// הסביבה: מעבדה בהירה בסגנון אולפן של טסלה. רצפת בטון מוחלק, קיר רקע מעוגל, תאורה רכה וצללים.
// כל החפצים של הסצנה (שולחנות, ארגזים, תאי סוללה) מצוירים ישירות מהגיאומטריה של MuJoCo.
import * as THREE from 'three';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';

export function createRenderer(container) {
  const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(container.clientWidth, container.clientHeight);
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 0.95;
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  container.appendChild(renderer.domElement);
  return renderer;
}

function concreteTexture() {
  const s = 512, c = document.createElement('canvas'); c.width = c.height = s;
  const g = c.getContext('2d');
  g.fillStyle = '#8d8f91'; g.fillRect(0, 0, s, s);
  const img = g.getImageData(0, 0, s, s);
  for (let i = 0; i < img.data.length; i += 4) {
    const n = (Math.random() - 0.5) * 16;
    img.data[i] += n; img.data[i + 1] += n; img.data[i + 2] += n;
  }
  g.putImageData(img, 0, 0);
  // כתמים רכים
  for (let k = 0; k < 40; k++) {
    const x = Math.random() * s, y = Math.random() * s, r = 20 + Math.random() * 80;
    const gr = g.createRadialGradient(x, y, 0, x, y, r);
    const a = Math.random() * 0.06;
    gr.addColorStop(0, `rgba(${Math.random() < 0.5 ? '255,255,255' : '40,40,40'},${a})`); gr.addColorStop(1, 'rgba(0,0,0,0)');
    g.fillStyle = gr; g.fillRect(0, 0, s, s);
  }
  // קווי תפר של לוחות בטון כל 2 מטר (הטקסטורה חוזרת כל 4 מ')
  g.strokeStyle = 'rgba(40,40,40,0.35)'; g.lineWidth = 2;
  for (const p of [0, s / 2]) { g.beginPath(); g.moveTo(p, 0); g.lineTo(p, s); g.moveTo(0, p); g.lineTo(s, p); g.stroke(); }
  const t = new THREE.CanvasTexture(c);
  t.wrapS = t.wrapT = THREE.RepeatWrapping; t.anisotropy = 8;
  t.colorSpace = THREE.SRGBColorSpace;
  return t;
}

export function buildEnvironment(renderer) {
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0xdfe3e7);
  scene.fog = new THREE.Fog(0xdfe3e7, 18, 60);
  const pmrem = new THREE.PMREMGenerator(renderer);
  scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.03).texture;
  scene.environmentIntensity = 0.85;

  const hemi = new THREE.HemisphereLight(0xffffff, 0x9a958c, 0.9);
  scene.add(hemi);
  const key = new THREE.DirectionalLight(0xfff4e8, 2.4);
  key.position.set(4, 9, 5);
  key.castShadow = true;
  key.shadow.mapSize.set(2048, 2048);
  const sc = key.shadow.camera; sc.left = -6; sc.right = 6; sc.top = 6; sc.bottom = -6; sc.near = 1; sc.far = 30;
  key.shadow.bias = -0.0003; key.shadow.normalBias = 0.02;
  scene.add(key, key.target);
  const rimL = new THREE.DirectionalLight(0xdfe8ff, 1.1);
  rimL.position.set(-6, 5, -4); scene.add(rimL);

  // רצפה
  const tex = concreteTexture(); tex.repeat.set(15, 15);
  const floor = new THREE.Mesh(new THREE.PlaneGeometry(60, 60), new THREE.MeshStandardMaterial({ map: tex, roughness: 0.62, metalness: 0.05 }));
  floor.rotation.x = -Math.PI / 2; floor.receiveShadow = true;
  scene.add(floor);
  // קיר רקע מעוגל (סייקלורמה) מסביב
  const wallMat = new THREE.MeshStandardMaterial({ color: 0xf0f2f4, roughness: 0.9, side: THREE.BackSide });
  const wall = new THREE.Mesh(new THREE.CylinderGeometry(24, 24, 14, 64, 1, true), wallMat);
  wall.position.y = 7; scene.add(wall);
  // פסי תאורה בתקרה
  const stripMat = new THREE.MeshBasicMaterial({ color: 0xffffff });
  for (let i = -3; i <= 3; i++) {
    const s = new THREE.Mesh(new THREE.BoxGeometry(0.12, 0.04, 16), stripMat);
    s.position.set(i * 2.2, 9, 0); scene.add(s);
  }
  return { scene, key };
}

// ---------- ציור החפצים של הסצנה מתוך המודל של MuJoCo ----------
const LOOKS = {
  table: { color: 0xe9e6df, roughness: 0.5, metalness: 0.0 },
  tableLeg: { color: 0x303236, roughness: 0.4, metalness: 0.7 },
  shelf: { color: 0x3a3d42, roughness: 0.45, metalness: 0.6 },
  box: { color: 0xc49a6c, roughness: 0.85, metalness: 0.0 },
  tote: { color: 0x2f6fb8, roughness: 0.6, metalness: 0.0 },
  cell: { color: 0x9aa3ad, roughness: 0.25, metalness: 0.9 },
  cellTop: { color: 0x1a1a1a, roughness: 0.5, metalness: 0.2 },
  tray: { color: 0x1e2226, roughness: 0.7, metalness: 0.1 },
  cup: { color: 0xf3efe6, roughness: 0.25, metalness: 0.0 },
  red: { color: 0xd23a32, roughness: 0.5 }, green: { color: 0x2f9d58, roughness: 0.5 }, blue: { color: 0x2e6fd0, roughness: 0.5 }, yellow: { color: 0xe7b416, roughness: 0.5 },
  step: { color: 0xbfc3c8, roughness: 0.7 },
  ball: { color: 0xff7a1a, roughness: 0.45 },
  marker: { color: 0x7a8088, roughness: 0.6 },
  dumbbell: { color: 0x2b2d31, roughness: 0.35, metalness: 0.8 },
};
export function buildSceneObjects(world, looks = {}) {
  const { model: m, mj } = world;
  const group = new THREE.Group();
  const items = [];
  const robotRoot = world.bodyId('pelvis');
  // מזהים את הגופים של הרובוט (תת-העץ של האגן) כדי לא לצייר אותם כאן
  const isRobot = new Uint8Array(m.nbody);
  for (let b = 1; b < m.nbody; b++) { let p = b; while (p > 0 && p !== robotRoot) p = m.body_parentid[p]; isRobot[b] = p === robotRoot ? 1 : 0; }
  for (let gi = 0; gi < m.ngeom; gi++) {
    const b = m.geom_bodyid[gi];
    if (isRobot[b]) continue;
    const name = mj.mj_id2name(m, mj.mjtObj.mjOBJ_GEOM.value, gi) || '';
    if (name === 'floor') continue;
    const type = m.geom_type[gi], sz = m.geom_size.subarray(3 * gi, 3 * gi + 3);
    let geo;
    if (type === 6) geo = new THREE.BoxGeometry(2 * sz[0], 2 * sz[1], 2 * sz[2]);
    else if (type === 5) { geo = new THREE.CylinderGeometry(sz[0], sz[0], 2 * sz[1], 28); geo.rotateX(Math.PI / 2); }
    else if (type === 3) { geo = new THREE.CapsuleGeometry(sz[0], 2 * sz[1], 8, 16); geo.rotateX(Math.PI / 2); }
    else if (type === 2) geo = new THREE.SphereGeometry(sz[0], 24, 16);
    else if (type === 4) { geo = new THREE.SphereGeometry(1, 24, 16); geo.scale(sz[0], sz[1], sz[2]); }
    else continue;
    const lookName = looks[name] || Object.keys(LOOKS).find(k => name.startsWith(k)) || 'marker';
    const L = LOOKS[lookName] || LOOKS.marker;
    const mat = new THREE.MeshStandardMaterial({ color: L.color, roughness: L.roughness ?? 0.6, metalness: L.metalness ?? 0 });
    const mesh = new THREE.Mesh(geo, mat);
    mesh.castShadow = true; mesh.receiveShadow = true;
    mesh.matrixAutoUpdate = false;
    group.add(mesh);
    items.push({ gi, mesh, name, dynamic: m.body_dofnum ? m.body_dofnum[b] > 0 : true });
  }
  const mat4 = new THREE.Matrix4();
  return {
    group, items,
    update(d) {
      const xp = d.geom_xpos, xm = d.geom_xmat;
      for (const it of items) {
        const i = it.gi, R = xm.subarray(9 * i, 9 * i + 9);
        mat4.set(R[0], R[1], R[2], xp[3 * i], R[3], R[4], R[5], xp[3 * i + 1], R[6], R[7], R[8], xp[3 * i + 2], 0, 0, 0, 1);
        it.mesh.matrix.copy(mat4);
      }
    },
  };
}
