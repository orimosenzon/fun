// world.js — סצנת הבסיס: כדור הארץ, ירח, שמש, כוכבים, מצלמה ותוויות.
// יחידות הסצנה: קילומטרים. מערכת אינרציאלית משוונית (ECI) ממופה ל-three כך: x→x, z(צפון)→y, y→-z.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import { gmst, sunDirECI, moonPosECI } from './physics.js';

export const RE_KM = 6378.137;
export const RM_KM = 1737.4;

// המרות קואורדינטות
export const eciToThree = (v, out = new THREE.Vector3()) => out.set(v[0], v[2], -v[1]);
export const ecefDir = (latDeg, lonDeg) => {
  const la = latDeg * Math.PI / 180, lo = lonDeg * Math.PI / 180;
  return [Math.cos(la) * Math.cos(lo), Math.cos(la) * Math.sin(lo), Math.sin(la)];
};
// ECEF → ECI לפי זמן כוכבי
export const ecefToEci = (v, theta) => [v[0] * Math.cos(theta) - v[1] * Math.sin(theta), v[0] * Math.sin(theta) + v[1] * Math.cos(theta), v[2]];
export const eciToEcef = (v, theta) => [v[0] * Math.cos(theta) + v[1] * Math.sin(theta), -v[0] * Math.sin(theta) + v[1] * Math.cos(theta), v[2]];

const earthVS = `
  #include <common>
  #include <logdepthbuf_pars_vertex>
  varying vec2 vUv; varying vec3 vN; varying vec3 vP;
  void main(){
    vUv = uv;
    vN = normalize(mat3(modelMatrix) * normal);
    vec4 wp = modelMatrix * vec4(position,1.0); vP = wp.xyz;
    gl_Position = projectionMatrix * viewMatrix * wp;
    #include <logdepthbuf_vertex>
  }`;
const earthFS = `
  #include <common>
  #include <logdepthbuf_pars_fragment>
  uniform sampler2D dayTex; uniform sampler2D nightTex; uniform vec3 sunDir;
  varying vec2 vUv; varying vec3 vN; varying vec3 vP;
  void main(){
    #include <logdepthbuf_fragment>
    vec3 N = normalize(vN);
    vec3 V = normalize(cameraPosition - vP);
    float d = dot(N, sunDir);
    vec3 day = texture2D(dayTex, vUv).rgb;
    vec3 night = texture2D(nightTex, vUv).rgb;
    float dayMix = smoothstep(-0.06, 0.10, d);
    vec3 col = day * (max(d, 0.0) * 1.35 + 0.012);
    // אורות ערים בצד הלילה
    col += night * night * vec3(1.0, 0.82, 0.55) * 1.6 * (1.0 - dayMix);
    // ברק השמש על האוקיינוסים
    float water = smoothstep(0.02, 0.10, day.b - day.r);
    vec3 H = normalize(sunDir + V);
    col += vec3(1.0, 0.95, 0.85) * pow(max(dot(N, H), 0.0), 60.0) * water * 0.6 * step(0.0, d);
    // גוון שקיעה בקו הדמדומים
    col += day * vec3(1.0, 0.45, 0.2) * 0.35 * exp(-pow((d - 0.02) * 22.0, 2.0));
    // אובך כחול בשוליים
    float rim = pow(1.0 - max(dot(N, V), 0.0), 2.5);
    col += vec3(0.25, 0.5, 1.0) * rim * (dayMix * 0.55 + 0.02);
    gl_FragColor = vec4(col, 1.0);
    #include <tonemapping_fragment>
    #include <colorspace_fragment>
  }`;
const atmoFS = `
  #include <common>
  #include <logdepthbuf_pars_fragment>
  uniform vec3 sunDir; varying vec3 vN; varying vec3 vP;
  void main(){
    #include <logdepthbuf_fragment>
    vec3 N = normalize(vN);
    vec3 V = normalize(cameraPosition - vP);
    float f = pow(clamp(1.0 - abs(dot(N, V)), 0.0, 1.0), 5.0);
    float lit = smoothstep(-0.35, 0.4, dot(N, sunDir));
    gl_FragColor = vec4(vec3(0.35, 0.6, 1.0) * f * lit * 1.6, f * lit);
    #include <colorspace_fragment>
  }`;
const cloudFS = `
  #include <common>
  #include <logdepthbuf_pars_fragment>
  uniform sampler2D cloudTex; uniform vec3 sunDir; varying vec2 vUv; varying vec3 vN; varying vec3 vP;
  void main(){
    #include <logdepthbuf_fragment>
    float c = texture2D(cloudTex, vUv).r;
    float d = dot(normalize(vN), sunDir);
    float l = max(d, 0.0) * 1.2 + 0.01;
    gl_FragColor = vec4(vec3(l), smoothstep(0.15, 0.9, c) * 0.85 * smoothstep(-0.2, 0.05, d));
    #include <colorspace_fragment>
  }`;

export function createWorld(canvas) {
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, logarithmicDepthBuffer: true, preserveDrawingBuffer: false });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.0;
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x000000);
  const pmrem = new THREE.PMREMGenerator(renderer);
  scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
  scene.environmentIntensity = 0.45;

  const camera = new THREE.PerspectiveCamera(45, 1, 0.0005, 5e9);
  camera.position.set(0, 8000, 26000);
  const controls = new OrbitControls(camera, canvas);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.minDistance = RE_KM * 1.05;
  controls.maxDistance = 2e6;
  controls.zoomSpeed = 1.2;

  const loader = new THREE.TextureLoader();
  const tex = (url) => { const t = loader.load(url); t.colorSpace = THREE.SRGBColorSpace; t.anisotropy = renderer.capabilities.getMaxAnisotropy(); return t; };
  const dayTex = tex('textures/earth_day.jpg');
  const nightTex = tex('textures/earth_night.jpg');
  const cloudTex = loader.load('textures/clouds.jpg');
  const moonTex = tex('textures/moon.jpg');

  const sunDir = new THREE.Vector3(1, 0, 0);
  const earthMat = new THREE.ShaderMaterial({ uniforms: { dayTex: { value: dayTex }, nightTex: { value: nightTex }, sunDir: { value: sunDir } }, vertexShader: earthVS, fragmentShader: earthFS });

  // כדור הארץ מסתובב בתוך קבוצה (מערכת ECEF)
  const earth = new THREE.Group();
  scene.add(earth);
  const earthMesh = new THREE.Mesh(new THREE.SphereGeometry(RE_KM, 256, 128), earthMat);
  earth.add(earthMesh);
  const clouds = new THREE.Mesh(new THREE.SphereGeometry(RE_KM + 12, 128, 64), new THREE.ShaderMaterial({
    uniforms: { cloudTex: { value: cloudTex }, sunDir: { value: sunDir } }, vertexShader: earthVS, fragmentShader: cloudFS, transparent: true, depthWrite: false,
  }));
  earth.add(clouds);
  const atmo = new THREE.Mesh(new THREE.SphereGeometry(RE_KM + 90, 128, 64), new THREE.ShaderMaterial({
    uniforms: { sunDir: { value: sunDir } }, vertexShader: earthVS, fragmentShader: atmoFS, transparent: true, depthWrite: false, side: THREE.BackSide, blending: THREE.AdditiveBlending,
  }));
  scene.add(atmo);

  // ירח (נעול גאות: הצד הקרוב פונה לכדור הארץ)
  // הירח מואר רק מהשמש, כדי שהמופע שלו בכל תאריך יהיה נכון
  const moon = new THREE.Mesh(new THREE.SphereGeometry(RM_KM, 96, 48), new THREE.ShaderMaterial({
    uniforms: { dayTex: { value: moonTex }, sunDir: { value: sunDir } }, vertexShader: earthVS,
    fragmentShader: `
      #include <common>
      #include <logdepthbuf_pars_fragment>
      uniform sampler2D dayTex; uniform vec3 sunDir; varying vec2 vUv; varying vec3 vN; varying vec3 vP;
      void main(){
        #include <logdepthbuf_fragment>
        vec3 c = texture2D(dayTex, vUv).rgb;
        float d = max(dot(normalize(vN), sunDir), 0.0);
        gl_FragColor = vec4(c * (d * 1.5 + 0.015), 1.0);
        #include <tonemapping_fragment>
        #include <colorspace_fragment>
      }` }));
  scene.add(moon);

  // שמש
  const sun = new THREE.DirectionalLight(0xffffff, 3.2);
  scene.add(sun); scene.add(sun.target);
  scene.add(new THREE.AmbientLight(0x6a7890, 0.55));
  // אור מילוי חלש מכיוון המצלמה, כדי שגם צד הצל של כלים יהיה קריא
  const fill = new THREE.DirectionalLight(0x9fb4d8, 0.45);
  camera.add(fill); camera.add(fill.target); fill.position.set(0.3, 0.4, 1); fill.target.position.set(0, 0, -1); scene.add(camera);
  const sunSprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: glowTexture(), color: 0xfff2d0, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending }));
  sunSprite.scale.setScalar(2.2e7);
  scene.add(sunSprite);

  // כוכבים: התפלגות אקראית עם בהירויות שונות (לא קטלוג אמיתי)
  scene.add(makeStars());

  // תוויות HTML
  const labelLayer = document.getElementById('labels');
  const labels = new Set();
  function addLabel(text, cls = '') {
    const el = document.createElement('div');
    el.className = 'label ' + cls;
    el.textContent = text;
    labelLayer.appendChild(el);
    const L = { el, pos: new THREE.Vector3(), visible: true, occlude: true, obj: null };
    labels.add(L);
    return L;
  }
  function removeLabel(L) { if (!L) return; L.el.remove(); labels.delete(L); }
  const tmp = new THREE.Vector3(), tmp2 = new THREE.Vector3();
  function updateLabels(w, h) {
    for (const L of labels) {
      if (L.obj) L.obj.getWorldPosition(L.pos);
      tmp.copy(L.pos).project(camera);
      let show = L.visible && tmp.z < 1 && Math.abs(tmp.x) < 1.2 && Math.abs(tmp.y) < 1.2;
      if (show && L.occlude && L.occluder !== false) {
        // האם כדור הארץ מסתיר את הנקודה?
        const c = camera.position;
        tmp2.copy(L.pos).sub(c);
        const len = tmp2.length(); tmp2.divideScalar(len);
        const b = c.dot(tmp2), cc = c.lengthSq() - (RE_KM * 0.995) ** 2;
        const disc = b * b - cc;
        if (disc > 0) { const t = -b - Math.sqrt(disc); if (t > 0 && t < len) show = false; }
      }
      L.el.style.display = show ? 'block' : 'none';
      if (show) L.el.style.transform = `translate(${(tmp.x * 0.5 + 0.5) * w}px, ${(-tmp.y * 0.5 + 0.5) * h}px)`;
    }
  }

  // זמן הסימולציה
  const clock = { time: Date.now(), rate: 1, paused: false };

  const tmpE = new THREE.Vector3();
  function updateCelestial() {
    const date = new Date(clock.time);
    earth.rotation.y = gmst(date);
    eciToThree(sunDirECI(date), sunDir).normalize();
    sun.position.copy(sunDir).multiplyScalar(1e6);
    sunSprite.position.copy(sunDir).multiplyScalar(1.4e8);
    if (!world.moonOverride) {
      const mp = moonPosECI(date);
      eciToThree([mp[0] / 1000, mp[1] / 1000, mp[2] / 1000], moon.position);
    }
    orientMoon();
  }
  function orientMoon() {
    const x = tmpE.copy(moon.position).negate().normalize();
    const y = new THREE.Vector3(0, 1, 0);
    y.sub(x.clone().multiplyScalar(y.dot(x))).normalize();
    const z = new THREE.Vector3().crossVectors(x, y);
    moon.quaternion.setFromRotationMatrix(new THREE.Matrix4().makeBasis(x, y, z));
  }

  // קטע קרקע ברזולוציה גבוהה סביב נקודה (לשיגור), עם UV תואם לטקסטורה הגלובלית
  function groundPatch(latDeg, lonDeg, radiusKm = 60) {
    const ang = radiusKm / RE_KM;
    const geo = new THREE.SphereGeometry(RE_KM + 0.002, 160, 160, 0, Math.PI * 2, 0, ang);
    // סיבוב כך שהקוטב של הקטע יעמוד על הנקודה
    const dir = ecefDir(latDeg, lonDeg);
    const target = new THREE.Vector3(dir[0], dir[2], -dir[1]);
    const q = new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 1, 0), target);
    geo.applyQuaternion(q);
    const pos = geo.attributes.position, uv = geo.attributes.uv, nrm = geo.attributes.normal;
    for (let i = 0; i < pos.count; i++) {
      const x = pos.getX(i), y = pos.getY(i), z = pos.getZ(i), r = Math.hypot(x, y, z);
      let u = Math.atan2(z, -x) / (2 * Math.PI); if (u < 0) u += 1;
      uv.setXY(i, u, 1 - Math.acos(y / r) / Math.PI);
      nrm.setXYZ(i, x / r, y / r, z / r);
    }
    const m = new THREE.Mesh(geo, earthMat);
    earth.add(m);
    return m;
  }

  const world = {
    renderer, scene, camera, controls, earth, earthMesh, earthMat, clouds, atmo, moon, sun, sunDir, clock,
    addLabel, removeLabel, updateLabels, updateCelestial, orientMoon, groundPatch, moonOverride: false,
  };
  return world;
}

function glowTexture() {
  const c = document.createElement('canvas'); c.width = c.height = 128;
  const g = c.getContext('2d');
  const gr = g.createRadialGradient(64, 64, 0, 64, 64, 64);
  gr.addColorStop(0, 'rgba(255,255,255,1)'); gr.addColorStop(0.08, 'rgba(255,245,220,0.9)');
  gr.addColorStop(0.25, 'rgba(255,220,160,0.25)'); gr.addColorStop(1, 'rgba(255,200,120,0)');
  g.fillStyle = gr; g.fillRect(0, 0, 128, 128);
  return new THREE.CanvasTexture(c);
}

function makeStars() {
  const n = 7000, pos = new Float32Array(n * 3), col = new Float32Array(n * 3);
  let seed = 7;
  const rnd = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
  for (let i = 0; i < n; i++) {
    const u = rnd() * 2 - 1, t = rnd() * Math.PI * 2, s = Math.sqrt(1 - u * u);
    const R = 3e9;
    pos.set([R * s * Math.cos(t), R * u, R * s * Math.sin(t)], i * 3);
    const b = Math.pow(rnd(), 6) * 0.9 + 0.08;
    const tint = rnd();
    col.set([b * (tint > 0.8 ? 1 : 0.85), b * 0.9, b * (tint < 0.2 ? 0.75 : 1)], i * 3);
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  g.setAttribute('color', new THREE.BufferAttribute(col, 3));
  return new THREE.Points(g, new THREE.PointsMaterial({ size: 1.6, sizeAttenuation: false, vertexColors: true, depthWrite: false }));
}

// קו מסלול מנקודות ECI (ק"מ)
export function makeLine(points, color = 0x6fc3ff, opacity = 0.9, dashed = false) {
  const g = new THREE.BufferGeometry().setFromPoints(points);
  const m = dashed
    ? new THREE.LineDashedMaterial({ color, transparent: true, opacity, dashSize: 600, gapSize: 400, depthWrite: false })
    : new THREE.LineBasicMaterial({ color, transparent: true, opacity, depthWrite: false });
  const l = new THREE.Line(g, m);
  if (dashed) l.computeLineDistances();
  return l;
}
