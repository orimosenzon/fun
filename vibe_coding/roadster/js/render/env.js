// הסביבה: רנדרר, שמיים, תאורה, קרקע, המבנים (מאותם נתונים כמו הפיזיקה), ותפאורה של אתר הניסויים.
import * as THREE from 'three';
import { Sky } from 'three/addons/objects/Sky.js';
import { groundHeight } from '../collide.js';
import { LAYOUT, SPAWNS, loopShift } from '../track.js';
import { concreteTexture, asphaltTexture, dirtTexture, rampTexture, steelTexture, labelTexture } from './textures.js';
import { buildParkedVehicle } from './vehicles.js';

export function createRenderer(container) {
  const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(container.clientWidth, container.clientHeight);
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 0.82;
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  container.appendChild(renderer.domElement);
  return renderer;
}

export function buildEnvironment(renderer, track) {
  const scene = new THREE.Scene();
  const fogColor = new THREE.Color(0xc9d3dc);
  scene.fog = new THREE.Fog(fogColor, 400, 3200);

  // שמיים ושמש (אחר הצהריים בטקסס)
  const sky = new Sky();
  sky.scale.setScalar(20000);
  const u = sky.material.uniforms;
  u.turbidity.value = 6; u.rayleigh.value = 1.6; u.mieCoefficient.value = 0.004; u.mieDirectionalG.value = 0.82;
  const sunDir = new THREE.Vector3().setFromSphericalCoords(1, THREE.MathUtils.degToRad(90 - 28), THREE.MathUtils.degToRad(215));
  u.sunPosition.value.copy(sunDir);
  scene.add(sky);
  // מפת סביבה להשתקפויות על הצבע
  const pmrem = new THREE.PMREMGenerator(renderer);
  const envScene = new THREE.Scene(); const sky2 = new Sky(); sky2.scale.setScalar(1000);
  Object.assign(sky2.material.uniforms.sunPosition.value, sunDir);
  for (const k of ['turbidity', 'rayleigh', 'mieCoefficient', 'mieDirectionalG']) sky2.material.uniforms[k].value = u[k].value;
  envScene.add(sky2);
  const gnd = new THREE.Mesh(new THREE.CircleGeometry(900, 16), new THREE.MeshBasicMaterial({ color: 0x8a7c5e }));
  gnd.rotation.x = -Math.PI / 2; gnd.position.y = -2; envScene.add(gnd);
  scene.environment = pmrem.fromScene(envScene, 0.02).texture;
  scene.environmentIntensity = 0.75;

  const hemi = new THREE.HemisphereLight(0xcfe3ff, 0x8a7a58, 0.9);
  scene.add(hemi);
  const sun = new THREE.DirectionalLight(0xfff1dc, 2.6);
  sun.position.copy(sunDir).multiplyScalar(200);
  sun.castShadow = true;
  sun.shadow.mapSize.set(2048, 2048);
  const sc = sun.shadow.camera; sc.left = -45; sc.right = 45; sc.top = 45; sc.bottom = -45; sc.near = 10; sc.far = 500;
  sun.shadow.bias = -0.0004; sun.shadow.normalBias = 0.03;
  scene.add(sun, sun.target);

  // ---------- קרקע ----------
  const dirt = dirtTexture(); dirt.repeat.set(160, 160);
  const groundMat = new THREE.MeshStandardMaterial({ map: dirt, roughness: 0.95, color: 0xffffff });
  const ground = new THREE.Mesh(new THREE.PlaneGeometry(2400, 2400, 1, 1), groundMat);
  ground.rotation.x = -Math.PI / 2; ground.receiveShadow = true;
  scene.add(ground);
  // גבעות מסביב (מאותה פונקציית גובה כמו הפיזיקה)
  {
    const g = new THREE.RingGeometry(650, 6000, 160, 40);
    const pos = g.attributes.position, colors = [];
    const c1 = new THREE.Color(0x9a8a62), c2 = new THREE.Color(0x6f7550);
    for (let i = 0; i < pos.count; i++) {
      const x = pos.getX(i), y = pos.getY(i);
      const h = groundHeight(x, -y);
      pos.setZ(i, h);
      const t = Math.min(1, h / 80);
      const c = c1.clone().lerp(c2, t * 0.8 + 0.1 * Math.sin(x * 0.01));
      colors.push(c.r, c.g, c.b);
    }
    g.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
    g.computeVertexNormals();
    const hills = new THREE.Mesh(g, new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 1 }));
    hills.rotation.x = -Math.PI / 2; hills.position.y = -0.05; hills.receiveShadow = true;
    scene.add(hills);
  }

  // משטחי בטון ואספלט
  const conc = concreteTexture(), asph = asphaltTexture();
  const pad = (x0, x1, z0, z1, map, tile = 12, y = 0.015) => {
    const t = map.clone(); t.needsUpdate = true; t.repeat.set((x1 - x0) / tile, (z1 - z0) / tile);
    const m = new THREE.Mesh(new THREE.PlaneGeometry(x1 - x0, z1 - z0), new THREE.MeshStandardMaterial({ map: t, roughness: 0.92, polygonOffset: true, polygonOffsetFactor: -1, polygonOffsetUnits: -1 }));
    m.rotation.x = -Math.PI / 2; m.position.set((x0 + x1) / 2, y, (z0 + z1) / 2); m.receiveShadow = true;
    scene.add(m); return m;
  };
  pad(-150, 95, -230, 290, conc);                                   // רחבה מרכזית
  pad(LAYOUT.drag.x - 11, LAYOUT.drag.x + 11, -330, 660, asph, 10, 0.02); // מסלול תאוצה
  pad(-460, -385, -340, 250, conc);                                  // קיר הרכיבה
  pad(LAYOUT.tunnel.x - 9, LAYOUT.tunnel.x + 9, -230, 210, asph, 10, 0.02);
  pad(225, 295, -230, 60, conc);                                     // לולאה
  pad(-385, -150, -250, -230, asph, 10, 0.02); pad(95, 225, -250, -230, asph, 10, 0.02); // דרכי חיבור
  pad(-385, 225, -270, -250, asph, 10, 0.021);

  // סימונים: קו זינוק ומרחקים במסלול התאוצה
  const lineMat = new THREE.MeshBasicMaterial({ color: 0xffffff });
  const yellow = new THREE.MeshBasicMaterial({ color: 0xffd400 });
  const mark = (x, z, w, d, mat) => { const m = new THREE.Mesh(new THREE.PlaneGeometry(w, d), mat); m.rotation.x = -Math.PI / 2; m.position.set(x, 0.04, z); scene.add(m); return m; };
  mark(LAYOUT.drag.x, LAYOUT.drag.z0 + 2.5, 20, 0.5, lineMat);
  for (const [d, txt] of [[100, '100 מ\''], [201.17, '⅛ מייל'], [402.34, '¼ מייל'], [804.67, '½ מייל']]) {
    mark(LAYOUT.drag.x, LAYOUT.drag.z0 + 2.5 + d, 20, 0.35, yellow);
    signpost(scene, LAYOUT.drag.x + 13, LAYOUT.drag.z0 + 2.5 + d, txt, 6, 1.6, -Math.PI / 2);
  }
  for (let z = LAYOUT.drag.z0; z < 650; z += 12) mark(LAYOUT.drag.x, z, 0.18, 5, lineMat);
  // סמן המראה לקיקפליפ
  const kz = LAYOUT.lineup.z - 16;
  mark(LAYOUT.lineup.x, kz, 8, 0.6, yellow);
  for (let i = -3; i <= 3; i += 2) { const a = mark(LAYOUT.lineup.x + i, kz - 3, 0.5, 2.4, yellow); a.rotation.z = 0; }
  signpost(scene, LAYOUT.lineup.x + 7, kz, 'נקודת המראה (K)', 7, 1.4, Math.PI / 2);
  // תחנת אוויר
  {
    const S = LAYOUT.station;
    const ringM = new THREE.Mesh(new THREE.RingGeometry(S.r - 0.4, S.r, 48), new THREE.MeshBasicMaterial({ color: 0x29b6f6 }));
    ringM.rotation.x = -Math.PI / 2; ringM.position.set(S.x, 0.05, S.z); scene.add(ringM);
    const comp = new THREE.Mesh(new THREE.BoxGeometry(3, 2.2, 2), new THREE.MeshStandardMaterial({ color: 0x1e88e5, roughness: 0.5, metalness: 0.3 }));
    comp.position.set(S.x + S.r + 2.5, 1.1, S.z); comp.castShadow = true; scene.add(comp);
    for (let i = 0; i < 4; i++) {
      const b = new THREE.Mesh(new THREE.CylinderGeometry(0.35, 0.35, 2.6, 16), new THREE.MeshStandardMaterial({ color: 0x2a2d31, metalness: 0.6, roughness: 0.4 }));
      b.position.set(S.x + S.r + 5 + i * 0.8, 1.3, S.z + 0.5); b.castShadow = true; scene.add(b);
    }
    signpost(scene, S.x + S.r + 2.5, S.z, 'תחנת אוויר דחוס', 6, 1.4, -Math.PI / 2, 3.2);
  }

  // ---------- המבנים מהפיזיקה ----------
  const mats = {
    ramp: new THREE.MeshStandardMaterial({ map: rampTexture('#ecebe6', '#d32f2f'), roughness: 0.55, metalness: 0.15, side: THREE.DoubleSide }),
    loop: new THREE.MeshStandardMaterial({ map: rampTexture('#e7e9ee', '#1565c0'), roughness: 0.5, metalness: 0.2, side: THREE.DoubleSide }),
    tunnel: new THREE.MeshStandardMaterial({ map: rampTexture('#cfd3d8', '#ffb300'), roughness: 0.5, metalness: 0.3, side: THREE.DoubleSide }),
    wallride: new THREE.MeshStandardMaterial({ map: rampTexture('#e0ddd5', '#2e7d32'), roughness: 0.6, metalness: 0.1, side: THREE.DoubleSide }),
    kicker: new THREE.MeshStandardMaterial({ map: rampTexture('#f2d14b', '#222'), roughness: 0.6, side: THREE.DoubleSide }),
  };
  for (const part of track.parts) {
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(part.geo.positions, 3));
    g.setAttribute('normal', new THREE.BufferAttribute(part.geo.normals, 3));
    // UV במטרים חלקי 10
    const uv = new Float32Array(part.geo.uvs.length);
    const su = part.name === 'tunnel' ? part.layout.r / 10 : 0.1, sv = 0.1;
    for (let i = 0; i < uv.length; i += 2) { uv[i] = part.geo.uvs[i] * su; uv[i + 1] = part.geo.uvs[i + 1] * sv; }
    g.setAttribute('uv', new THREE.BufferAttribute(uv, 2));
    g.setIndex(new THREE.BufferAttribute(part.geo.indices, 1));
    const mat = mats[part.mat].clone();
    mat.map = mats[part.mat].map.clone(); mat.map.needsUpdate = true;
    if (part.name === 'wall' || part.name === 'kicker') mat.map.repeat.set(1, 1);
    const m = new THREE.Mesh(g, mat);
    m.castShadow = true; m.receiveShadow = true;
    scene.add(m);
  }
  buildSupports(scene, track);

  // רכבים חונים
  for (const v of track.parked) { const m = buildParkedVehicle(v); m.traverse(o => { if (o.isMesh) o.castShadow = true; }); scene.add(m); }

  // תפאורה
  buildScenery(scene);

  // שלטים מעל כל אזור
  signpost(scene, LAYOUT.wall.x + 9, LAYOUT.wall.z - 20, 'הקיר מההדגמה', 9, 2.0, 0, 5);
  signpost(scene, LAYOUT.lineup.x - 16, LAYOUT.lineup.z - 6, 'קיקפליפ מעל השורה', 9, 2.0, 0, 5.5);
  signpost(scene, LAYOUT.tunnel.x, LAYOUT.tunnel.z0 - 8, 'הצינור: נסיעה על התקרה', 11, 2.0, 0, 13.5);
  signpost(scene, LAYOUT.loop.x + 12, LAYOUT.loop.z - 12, 'הלולאה', 6, 2.0, 0, 5);
  signpost(scene, LAYOUT.wallRide.x + 18, LAYOUT.wallRide.z0 - 15, 'קיר אנכי: נסיעה לאורך הקיר', 12, 2.0, 0, 5);
  signpost(scene, LAYOUT.drag.x - 14, LAYOUT.drag.z0, 'מסלול תאוצה', 8, 2.0, Math.PI / 2, 4);
  signpost(scene, LAYOUT.hover.x + 9, LAYOUT.hover.z, 'מבחן ריחוף', 7, 2.0, -Math.PI / 2, 4);
  // טבעת הריחוף
  {
    const H = LAYOUT.hover;
    const tor = new THREE.Mesh(new THREE.TorusGeometry(4, 0.12, 10, 48), new THREE.MeshStandardMaterial({ color: 0xff7043, emissive: 0x551800 }));
    tor.rotation.x = Math.PI / 2; tor.position.set(H.x, 5, H.z); scene.add(tor);
    for (const sx of [-1, 1]) {
      const p = new THREE.Mesh(new THREE.CylinderGeometry(0.1, 0.1, 5, 8), new THREE.MeshStandardMaterial({ color: 0x555555 }));
      p.position.set(H.x + sx * 4.1, 2.5, H.z); scene.add(p);
    }
    const c = new THREE.Mesh(new THREE.RingGeometry(3.6, 4, 48), new THREE.MeshBasicMaterial({ color: 0xff7043 }));
    c.rotation.x = -Math.PI / 2; c.position.set(H.x, 0.05, H.z); scene.add(c);
  }

  return { scene, sun, sky, hemi };
}

// שלט על עמודים
export function signpost(scene, x, z, text, w, h, yaw = 0, height = 3) {
  const t = labelTexture(text, { w: 1024, h: Math.round(1024 * h / w) });
  // שני פנים גב אל גב, כדי שהטקסט ייקרא נכון משני הצדדים
  const board = new THREE.Group();
  const mat = new THREE.MeshBasicMaterial({ map: t });
  const f1 = new THREE.Mesh(new THREE.PlaneGeometry(w, h), mat);
  const f2 = new THREE.Mesh(new THREE.PlaneGeometry(w, h), mat); f2.rotation.y = Math.PI;
  board.add(f1, f2);
  board.position.set(x, height, z); board.rotation.y = yaw;
  scene.add(board);
  const pm = new THREE.MeshStandardMaterial({ color: 0x333333 });
  for (const s of [-1, 1]) {
    const p = new THREE.Mesh(new THREE.CylinderGeometry(0.06, 0.06, height, 6), pm);
    p.position.set(x + Math.cos(yaw) * s * w * 0.4, height / 2 - h / 2, z - Math.sin(yaw) * s * w * 0.4);
    scene.add(p);
  }
  return board;
}

// קורת פלדה בין שתי נקודות
const beamGeo = new THREE.BoxGeometry(1, 1, 1);
function beams(list, mat, thick = 0.25) {
  const m = new THREE.InstancedMesh(beamGeo, mat, list.length);
  const o = new THREE.Object3D(), a = new THREE.Vector3(), b = new THREE.Vector3();
  list.forEach(([p, q, t], i) => {
    a.set(...p); b.set(...q);
    const len = a.distanceTo(b);
    o.position.copy(a).add(b).multiplyScalar(0.5);
    o.lookAt(b);
    o.scale.set(t || thick, t || thick, len);
    o.updateMatrix(); m.setMatrixAt(i, o.matrix);
  });
  m.castShadow = true; m.receiveShadow = true;
  return m;
}

function buildSupports(scene, track) {
  const steel = new THREE.MeshStandardMaterial({ map: steelTexture(), color: 0x9aa1a8, roughness: 0.5, metalness: 0.7 });
  const red = new THREE.MeshStandardMaterial({ color: 0xb71c1c, roughness: 0.5, metalness: 0.5 });
  for (const part of track.parts) {
    const list = [];
    if (part.name === 'wall') {
      // מסבך פלדה בשני הצדדים, מחוץ לרוחב המסלול
      const L = part.layout, pr = part.profile;
      for (const sx of [1, -1]) {
        const x = L.x + sx * (L.width / 2 + 0.6);
        let prev = null;
        for (let s = 0; s <= pr.total + 0.01; s += pr.total / 24) {
          const q = pr.at(s);
          const nY = Math.cos(q.th), nZ = -Math.sin(q.th);
          const p = [x, q.v - nY * 0.7, L.z + q.h - nZ * 0.7];
          if (prev) list.push([prev, p, 0.35]);
          if (p[1] > 1) list.push([[x, 0, p[2]], p, 0.3]);
          if (prev && p[1] > 1) list.push([[x, 0, prev[2]], p, 0.12]);
          // קורה רוחבית מתחת למשטח, רק כשהמשטח לא מעל אזור הנסיעה
          if (q.th < Math.PI / 2 + 0.3 || q.th > 2.6) list.push([[L.x - L.width / 2 - 0.6, p[1], p[2]], [L.x + L.width / 2 + 0.6, p[1], p[2]], 0.18]);
          prev = p;
        }
      }
      // מגדל אדום (כמו המגדל בהדגמה), לצד הקיר
      scene.add(beams(towerBeams(L.x + 26, L.z + 14, 46, 5), red));
    } else if (part.name === 'loop') {
      // גשר עליון עם כבלים, שני מגדלים מחוץ לרוחב
      const L = part.layout, pr = part.profile;
      const xA = L.x + L.width / 2 + 3, xB = L.x - L.shift - L.width / 2 - 3;
      let top = 0, zTop = L.z; for (const p of pr.pts) if (p.v > top) { top = p.v; zTop = L.z + p.h; }
      const H = top + 4;
      for (const x of [xA, xB]) for (const dz of [-3, 3]) list.push([[x, 0, zTop + dz], [x, H, zTop + dz], 0.5]);
      list.push([[xA, H, zTop - 3], [xB, H, zTop - 3], 0.5], [[xA, H, zTop + 3], [xB, H, zTop + 3], 0.5]);
      list.push([[xA, H, zTop - 3], [xA, H, zTop + 3], 0.4], [[xB, H, zTop - 3], [xB, H, zTop + 3], 0.4]);
      for (let s = pr.total * 0.28; s <= pr.total * 0.72; s += pr.total / 22) {
        const q = pr.at(s), sh = loopShift(s, pr, L);
        for (const u of [-L.width / 2 - 0.2, L.width / 2 + 0.2]) {
          const p = [L.x - u - sh, q.v + 0.5 * Math.sign(Math.cos(q.th)) * 0, L.z + q.h];
          list.push([p, [p[0], H, Math.min(Math.max(p[2], zTop - 3), zTop + 3)], 0.05]);
        }
      }
    } else if (part.name === 'tunnel') {
      const L = part.layout;
      const ringMat = new THREE.MeshStandardMaterial({ color: 0x455a64, metalness: 0.7, roughness: 0.4 });
      for (let z = L.z0; z <= L.z1 + 0.1; z += 7) {
        const r = new THREE.Mesh(new THREE.TorusGeometry(L.r + 0.5, 0.18, 8, 48), ringMat);
        r.position.set(L.x, L.r, z); r.castShadow = true; scene.add(r);
        list.push([[L.x - L.r - 0.6, 0, z], [L.x - L.r - 0.6, L.r, z], 0.3], [[L.x + L.r + 0.6, 0, z], [L.x + L.r + 0.6, L.r, z], 0.3]);
      }
    } else if (part.name === 'wallride') {
      const L = part.layout, pr = part.profile;
      const back = L.x - pr.at(pr.total).h - 1.2;
      for (let z = L.z0; z <= L.z1; z += 8) {
        list.push([[back, 0, z], [back, L.height + 6.5, z], 0.4]);
        list.push([[back, 2, z], [back - 5, 0, z], 0.25], [[back, L.height, z], [back - 7, 0, z], 0.25]);
      }
      list.push([[back, L.height + 6.5, L.z0], [back, L.height + 6.5, L.z1], 0.4]);
    }
    if (list.length) scene.add(beams(list, steel));
  }
}

// מגדל סריג: 4 עמודים, טבעות ואלכסונים
function towerBeams(x, z, H, w) {
  const L = [];
  const c = [[-1, -1], [1, -1], [1, 1], [-1, 1]].map(([a, b]) => [x + a * w / 2, z + b * w / 2]);
  for (const [cx, cz] of c) L.push([[cx, 0, cz], [cx, H, cz], 0.45]);
  for (let y = 0; y <= H; y += 4) for (let i = 0; i < 4; i++) {
    const a = c[i], b = c[(i + 1) % 4];
    L.push([[a[0], y, a[1]], [b[0], y, b[1]], 0.22]);
    if (y + 4 <= H) L.push([[a[0], y, a[1]], [b[0], y + 4, b[1]], 0.14]);
  }
  // זרועות בראש
  L.push([[x - w / 2, H - 6, z], [x - w / 2 - 14, H - 6, z], 0.6], [[x + w / 2, H - 6, z], [x + w / 2 + 14, H - 6, z], 0.6]);
  return L;
}

function buildScenery(scene) {
  const white = new THREE.MeshStandardMaterial({ color: 0xeeeeee, roughness: 0.6, metalness: 0.2 });
  const grey = new THREE.MeshStandardMaterial({ color: 0x8e959c, roughness: 0.7, metalness: 0.3 });
  const dark = new THREE.MeshStandardMaterial({ color: 0x2b2f34, roughness: 0.8 });
  // האנגרים
  for (const [x, z, w, d, h] of [[-60, -330, 70, 40, 18], [40, -330, 50, 36, 14], [330, 120, 40, 60, 16]]) {
    const g = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), grey); g.position.set(x, h / 2, z); g.castShadow = true; g.receiveShadow = true; scene.add(g);
    const roof = new THREE.Mesh(new THREE.CylinderGeometry(d / 2, d / 2, w, 24, 1, false, 0, Math.PI), white);
    roof.rotation.z = Math.PI / 2; roof.position.set(x, h, z); roof.scale.set(1, 0.35, 1); roof.castShadow = true; scene.add(roof);
  }
  // חוות מכלים (כדורים וגלילים לבנים)
  for (let i = 0; i < 4; i++) {
    const s = new THREE.Mesh(new THREE.SphereGeometry(7, 24, 16), white); s.position.set(420 + i * 18, 8, -260); s.castShadow = true; scene.add(s);
    for (const [dx, dz] of [[-5, -5], [5, -5], [5, 5], [-5, 5]]) { const l = new THREE.Mesh(new THREE.CylinderGeometry(0.4, 0.4, 3, 6), grey); l.position.set(420 + i * 18 + dx, 1.5, -260 + dz); scene.add(l); }
  }
  for (let i = 0; i < 5; i++) {
    const c = new THREE.Mesh(new THREE.CylinderGeometry(3, 3, 28, 24), white); c.position.set(-420 + i * 9, 14, 380); c.castShadow = true; scene.add(c);
  }
  // דוכן בדיקה (test stand) מבטון
  const stand = new THREE.Mesh(new THREE.BoxGeometry(18, 30, 14), new THREE.MeshStandardMaterial({ color: 0xa8a49a, roughness: 0.95 }));
  stand.position.set(520, 15, 140); stand.castShadow = true; scene.add(stand);
  // יציעים עם קהל, במרחק בטוח (כמו בהדגמה: מאות מטרים)
  const standMat = new THREE.MeshStandardMaterial({ color: 0x5f6670, roughness: 0.8 });
  const crowdColors = [0x263238, 0xeceff1, 0x37474f, 0xb0bec5, 0xd32f2f, 0x1565c0, 0xffffff, 0x212121];
  for (const [x, z, yaw] of [[-210, 330, Math.PI], [-130, 330, Math.PI], [190, -420, 0]]) {
    const g = new THREE.Group();
    for (let r = 0; r < 8; r++) {
      const step = new THREE.Mesh(new THREE.BoxGeometry(60, 0.6, 1.4), standMat);
      step.position.set(0, 0.3 + r * 0.6, r * 1.4); step.castShadow = true; step.receiveShadow = true; g.add(step);
    }
    const n = 8 * 70;
    const people = new THREE.InstancedMesh(new THREE.BoxGeometry(0.45, 1.1, 0.35), new THREE.MeshStandardMaterial({ roughness: 0.9 }), n);
    const o = new THREE.Object3D(), col = new THREE.Color();
    let k = 0;
    for (let r = 0; r < 8; r++) for (let i = 0; i < 70; i++) {
      if (Math.random() < 0.2) { o.scale.set(0, 0, 0); } else o.scale.set(1, 0.9 + Math.random() * 0.25, 1);
      o.position.set(-29 + i * 0.84 + Math.random() * 0.2, 0.6 + r * 0.6 + 0.55, r * 1.4 + 0.1);
      o.updateMatrix(); people.setMatrixAt(k, o.matrix); people.setColorAt(k, col.setHex(crowdColors[(Math.random() * crowdColors.length) | 0])); k++;
    }
    g.add(people);
    const roof = new THREE.Mesh(new THREE.BoxGeometry(62, 0.4, 14), dark); roof.position.set(0, 9, 5); roof.castShadow = true; g.add(roof);
    g.position.set(x, 0, z); g.rotation.y = yaw; scene.add(g);
  }
  // תורני תאורה
  const poleM = new THREE.MeshStandardMaterial({ color: 0x777777, metalness: 0.6, roughness: 0.4 });
  for (const [x, z] of [[-150, -230], [95, -230], [-150, 290], [95, 290], [-20, 300], [230, 60], [-460, 250], [-460, -340]]) {
    const p = new THREE.Mesh(new THREE.CylinderGeometry(0.25, 0.4, 26, 8), poleM); p.position.set(x, 13, z); p.castShadow = true; scene.add(p);
    const h = new THREE.Mesh(new THREE.BoxGeometry(4, 1.5, 0.6), dark); h.position.set(x, 26, z); scene.add(h);
  }
  // שיחים ועצי מסקיט פזורים בשטח
  const bush = new THREE.InstancedMesh(new THREE.IcosahedronGeometry(1, 0), new THREE.MeshStandardMaterial({ color: 0x5f6b3a, roughness: 1, flatShading: true }), 700);
  const o = new THREE.Object3D(); let n = 0;
  for (let i = 0; i < 1400 && n < 700; i++) {
    const a = Math.random() * Math.PI * 2, r = 330 + Math.random() * 1300;
    const x = Math.cos(a) * r, z = Math.sin(a) * r;
    if (Math.abs(x - 120) < 30 && z > -350 && z < 680) continue;
    const s = 0.7 + Math.random() * 2.6;
    o.position.set(x, groundHeight(x, z) + s * 0.5, z); o.scale.set(s * (1 + Math.random()), s * (0.6 + Math.random() * 0.6), s * (1 + Math.random()));
    o.rotation.y = Math.random() * 6;
    o.updateMatrix(); bush.setMatrixAt(n++, o.matrix);
  }
  bush.count = n; bush.castShadow = true;
  scene.add(bush);
  // גדר היקפית נמוכה
  const fence = new THREE.InstancedMesh(new THREE.BoxGeometry(0.12, 1.6, 0.12), poleM, 400);
  for (let i = 0; i < 400; i++) { const a = i / 400 * Math.PI * 2; o.position.set(Math.cos(a) * 690, 0.8, Math.sin(a) * 690); o.scale.set(1, 1, 1); o.rotation.set(0, 0, 0); o.updateMatrix(); fence.setMatrixAt(i, o.matrix); }
  scene.add(fence);
}

export { SPAWNS };
