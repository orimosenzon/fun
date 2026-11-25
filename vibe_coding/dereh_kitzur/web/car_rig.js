/* The flying car over the map: what is drawn.
 *
 * The flight is car_core.js, which is FÖHN's F-5 with a car's manners. This
 * file is the rig explore.js holds, with the handles of the two bikes'
 * (bike.js makeRig, f5.js): one scene with only the vehicle in it, on a
 * transparent canvas over MapLibre's picture, and a camera set every frame
 * from the map's own.
 *
 * Nothing here is modelled twice. The body, the canopy, the wheels, the
 * wing, the nozzles and the whole cockpit are the roadster project's
 * (../../roadster/js/render/carmodel.js), the Rocket variant: one seat in
 * the middle, four nozzles at the back, one at each corner where the F-5
 * has its lift pipes, two at the nose. The flames and the dust are FÖHN's
 * (fx.js), hung on those nozzles. Both are imported, so the car here is the
 * one that project has today.
 *
 * `three` is a bare name in the roadster's files. index.html maps it to
 * FÖHN's copy of three.js, the one fx.js and f5.js already load, so the car,
 * its flames and the F-5 all share one library.
 *
 * Two views (Ori, 25/11/2026: "from behind, where the whole car is seen,
 * and from inside the car, as if sitting in it"):
 *
 *   chase    the bikes' chase camera, a little further back for a wider
 *            thing. The scene is drawn from the map's camera, so the car
 *            sits on the photograph.
 *   inside   the driver's seat. Here there are two lenses, as in the
 *            balloon's basket (balloon.js): the world outside is the map,
 *            through the map's narrow lens, and the cockpit is drawn around
 *            it through a wide one of its own, from a camera that rides in
 *            the car. A cockpit through the map's 42 degrees is a letterbox
 *            of dashboard; through its own lens it is a seat. What ties the
 *            two pictures together is that they move as one: the map looks
 *            where the nose points and rolls with the bank, and the cockpit
 *            stays still around you, which is what a cockpit does.
 *
 * Loaded on the first take-off, by car.js, and not before.
 */
import * as THREE from 'three';
import { buildCarModel } from '../../roadster/js/render/carmodel.js';
import { VARIANTS } from '../../roadster/js/params.js';
import { FX } from '../../jetbikes/foehn/js/render/fx.js';
import { U } from '../../jetbikes/foehn/js/render/env.js';
import { ENG, SIDE_MAX, G } from '../../jetbikes/foehn/js/sim/bike.js';
import { qrot, qinvrot } from '../../jetbikes/foehn/js/sim/vmath.js';
import { CarFlight, yawFor, V_MAX } from './car_core.js';

export { yawFor };

const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
const RAD = Math.PI / 180;
const _Y = new THREE.Vector3(0, 1, 0), _tmp2 = new THREE.Vector3();

/* The three layers of the scene. In the chase view one camera sees all of
 * them. Inside, the map's camera draws the world (the dust, the grain on
 * the ground) and the cockpit's camera draws the car; the flames belong to
 * neither, because from the seat they are under the floor. */
const L_WORLD = 0, L_CAR = 1, L_FLAME = 2;

/* The cockpit's lens and where the head is. */
const CAB = {
  HFOV: 98,            // degrees across, on a wide screen; the vertical field follows from it
  VFOV_MIN: 60,
  VFOV_MAX: 98,        // a phone held upright: as wide as is still a windscreen and not a fisheye
  EYE: [0, 0.50, -0.56],   // the driver's eyes in the car's frame: x left, y up, z forward
  DASH: 9.5,           // degrees below the nose that the top of the dashboard is seen at, and
  DASH_Y: 0.24,        // where it is put: this share of half the screen below the middle, on any screen
  DIP: 9,              // degrees below the nose the map is asked to look: the horizon sits in the upper third
  NOSE_K: 0.55,        // how much of a nose the map cannot follow is shown by the cockpit rising instead
  LOOK_MIN: -9, LOOK_MAX: 23      // how far the nose may carry the cockpit from where it rests
};

const WINDS = ['צפון', 'צפון־מזרח', 'מזרח', 'דרום־מזרח', 'דרום', 'דרום־מערב', 'מערב', 'צפון־מערב'];

/** A sky for the paint and the glass to reflect: pale at the horizon, blue
 *  overhead, the ground's colour below. */
function skyEnv(renderer) {
  const geo = new THREE.SphereGeometry(100, 24, 12);
  const col = new Float32Array(geo.attributes.position.count * 3);
  const top = new THREE.Color(0x4f86c8), hor = new THREE.Color(0xdfe9f2), gnd = new THREE.Color(0x6f7359);
  const c = new THREE.Color();
  for (let i = 0; i < geo.attributes.position.count; i++) {
    const y = geo.attributes.position.getY(i) / 100;
    if (y >= 0) c.copy(hor).lerp(top, Math.pow(y, 0.6)); else c.copy(hor).lerp(gnd, Math.min(1, -y * 4));
    col[i * 3] = c.r; col[i * 3 + 1] = c.g; col[i * 3 + 2] = c.b;
  }
  geo.setAttribute('color', new THREE.BufferAttribute(col, 3));
  const s = new THREE.Scene();
  s.add(new THREE.Mesh(geo, new THREE.MeshBasicMaterial({ vertexColors: true, side: THREE.BackSide })));
  const pm = new THREE.PMREMGenerator(renderer);
  const tex = pm.fromScene(s, 0, 1, 500).texture;
  pm.dispose();
  return tex;
}

/** `TUNE` is the first bike's (Bike.TUNE): the chase camera's numbers and
 *  the grain on the ground are shared, so the three vehicles sit in the
 *  same picture. */
export function makeRig(canvas, TUNE) {
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true, premultipliedAlpha: true });
  renderer.setClearColor(0x000000, 0);
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.0;
  renderer.autoClear = false;
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(42, 1, 0.1, 6000);
  const cab = new THREE.PerspectiveCamera(70, 1, 0.04, 80);
  cab.layers.set(L_CAR);

  const sun = new THREE.DirectionalLight(0xfff2dd, Math.PI * 1.1);
  sun.position.set(700, 1000, -400);
  const hemi = new THREE.HemisphereLight(0xbdd7f2, 0x62705a, Math.PI * 0.6);
  scene.add(sun, hemi);
  scene.environment = skyEnv(renderer);
  scene.environmentIntensity = 0.6;

  const P = VARIANTS.rocket;
  const model = buildCarModel(P);
  const phys = new CarFlight();
  const b = phys.b;

  /* The frames, from the outside in. FÖHN's body frame is x forward, y up,
   * z to the right, about the bike's centre of gravity; its design frame
   * has the origin on the ground under the middle. The car stands in that
   * design frame with its wheels on the ground where the F-5 has its skids,
   * and the roadster's own frame (x left, y up, z forward, about the car's
   * centre of gravity) is a quarter turn inside it. */
  const holder = new THREE.Group();          // FÖHN's body frame
  const design = new THREE.Group();          // FÖHN's design frame
  const fo = new THREE.Group();              // the same axes, at the car's centre of gravity
  const mount = new THREE.Group();           // the roadster's frame
  fo.position.set(0, P.cgHeight, 0);
  mount.rotation.y = Math.PI / 2;
  holder.add(design); design.add(fo); fo.add(mount); mount.add(model.root);
  scene.add(holder);

  /* Where FÖHN's flames hang. fx.js wants the F-5's nozzles: four lift
   * pipes (front left, front right, rear left, rear right), the rear
   * nozzle, and four side exits; each a group whose local -x is the way the
   * gas leaves. The car's are the Rocket's own nozzles, in the roadster's
   * numbers (params.js), turned into FÖHN's axes: forward is the car's z,
   * right is minus its x. */
  const byId = {};
  for (const t of P.thrusters) byId[t.id] = t.pos;
  const fAxes = (p) => [p[2], p[1], -p[0]];
  const lifts = ['FL', 'FR', 'RL', 'RR'].map((id) => {
    const g = new THREE.Group();
    g.position.set(...fAxes(byId[id]));
    const noz = new THREE.Group();
    noz.rotation.z = Math.PI / 2;            // local -x = straight down
    g.add(noz); fo.add(g);
    return { g, noz };
  });
  const rears = ['M2', 'M1', 'M3', 'M4'].map((id) => {
    const g = new THREE.Group();
    const p = fAxes(byId[id]);
    g.position.set(p[0] - 0.06, p[1], p[2]);
    fo.add(g);
    return g;
  });
  const sides = [];
  for (const [pair, x, y, z] of [[0, 1.86, -0.1, 0.84], [1, -1.72, -0.06, 0.99]]) {
    for (const sgn of [-1, 1]) {
      const g = new THREE.Group();
      g.position.set(x, y, sgn * z);
      g.rotation.y = sgn * Math.PI / 2;      // local -x = outward
      const noz = new THREE.Group();
      g.add(noz); fo.add(g);
      sides.push({ noz, pair, sgn });
    }
  }
  const fx = new FX(scene, {
    nozzles: [...lifts.map((l) => ({ noz: l.noz, r: 0.066 })), { noz: rears[0], r: 0.07 },
      ...sides.map((s) => ({ noz: s.noz, r: 0.045 }))],
    sides
  });
  // One flame from FÖHN, four nozzles on the car: the other three are the
  // same flame again, each following the first.
  const rearPlume = fx.plumes[4].mesh;
  const rearTwins = rears.slice(1).map((g) => {
    const m = new THREE.Mesh(rearPlume.geometry, rearPlume.material);
    m.renderOrder = rearPlume.renderOrder;
    g.add(m);
    return m;
  });

  /* The flames, the glows and the sparks are additive, and FÖHN draws them
   * over its own opaque world. Here the canvas is transparent and the world
   * is the map underneath, and three.js's additive blending also adds to the
   * canvas's alpha: a faint blue flame became a dark, nearly solid column
   * over the photograph (f5.js, 20/11/2026). So anything additive adds its
   * light and leaves the alpha alone, which is what light over a picture is. */
  const lightOnly = () => scene.traverse((o) => {
    for (const m of [].concat(o.material || [])) {
      if (m.blending !== THREE.AdditiveBlending) continue;
      m.blending = THREE.CustomBlending;
      m.blendEquation = THREE.AddEquation;
      m.blendSrc = THREE.SrcAlphaFactor; m.blendDst = THREE.OneFactor;
      m.blendSrcAlpha = THREE.ZeroFactor; m.blendDstAlpha = THREE.OneFactor;
      m.needsUpdate = true;
    }
  });
  lightOnly();

  // Who draws what: see the note at L_WORLD.
  model.root.traverse((o) => o.layers.set(L_CAR));
  for (const pl of fx.plumes) pl.mesh.layers.set(L_FLAME);
  for (const m of rearTwins) m.layers.set(L_FLAME);
  for (const l of [sun, hemi, fx.light, fx.flash]) l.layers.enableAll();
  camera.layers.enable(L_CAR);
  camera.layers.enable(L_FLAME);

  // The cockpit's camera rides in the car, in the roadster's frame.
  mount.add(cab);

  /* What the roadster's cockpit does not have, because its own inside view
   * never looks down: a lining for the tub the seat stands in (it is the
   * body's own white paint, and from the seat a white bathtub), and the
   * driver's own arms. The roadster hides its driver when you sit in the
   * seat, which is right for the helmet and leaves the yoke turning by
   * itself; so the forearms are put back, from the bottom of the picture to
   * the grips, and they follow the yoke round. (The legs were tried too.
   * Seen from above, two white thighs between two white arms were four
   * stripes across the bottom of the screen.) All of it is drawn only from
   * the seat. */
  const cabin = new THREE.Group();
  {
    const trim = new THREE.MeshStandardMaterial({ color: 0x17191d, roughness: 0.85, side: THREE.DoubleSide });
    const wall = (w, h, d, x, y, z) => {
      const m = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), trim);
      m.position.set(x, y, z);
      cabin.add(m);
    };
    wall(1.16, 0.02, 1.6, 0, -0.085, -0.2);            // the floor
    for (const sx of [1, -1]) wall(0.02, 0.36, 1.6, sx * 0.575, 0.09, -0.2);   // the sides, up to the sill
    wall(1.16, 0.4, 0.02, 0, 0.1, 0.56);               // under the dashboard
    cabin.visible = false;
    model.cockpit.add(cabin);
  }
  // The yoke is not among the things the roadster's model hands back; it is
  // the one group that stands where the yoke stands.
  const yoke = model.cockpit.children.find((o) => o.isGroup && o !== cabin
    && Math.abs(o.position.y - 0.25) < 1e-3 && Math.abs(o.position.z - 0.1) < 1e-3) || null;
  const arms = yoke ? [1, -1].map((sx) => {
    const fore = new THREE.Mesh(new THREE.CylinderGeometry(0.036, 0.046, 1, 12), model.materials.suit);
    const hand = new THREE.Mesh(new THREE.SphereGeometry(0.047, 14, 10),
      new THREE.MeshStandardMaterial({ color: 0x1d1f23, roughness: 0.6 }));
    hand.scale.set(1, 1.15, 1.25);
    cabin.add(fore, hand);
    return { sx, fore, hand, elbow: new THREE.Vector3(sx * 0.33, 0.07, -0.36), grip: new THREE.Vector3() };
  }) : [];
  cabin.traverse((o) => o.layers.set(L_CAR));
  function poseArms() {
    for (const a of arms) {
      a.grip.set(a.sx * 0.17, 0.035, -0.012).applyEuler(yoke.rotation).add(yoke.position);
      a.hand.position.copy(a.grip);
      a.hand.quaternion.copy(yoke.quaternion);
      _tmp2.copy(a.grip).sub(a.elbow);
      const len = _tmp2.length();
      a.fore.scale.set(1, len, 1);
      a.fore.position.copy(a.elbow).addScaledVector(_tmp2, 0.5);
      a.fore.quaternion.setFromUnitVectors(_Y, _tmp2.normalize());
    }
  }

  // A soft shadow on the ground, the car's shape and not a disc: it is what
  // the height is judged by, and it turns with the car.
  const shadowTex = (() => {
    const c = document.createElement('canvas');
    c.width = c.height = 128;
    const ctx = c.getContext('2d');
    const g = ctx.createRadialGradient(64, 64, 0, 64, 64, 64);
    g.addColorStop(0, 'rgba(0,0,0,1)');
    g.addColorStop(0.55, 'rgba(0,0,0,0.85)');
    g.addColorStop(1, 'rgba(0,0,0,0)');
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, 128, 128);
    return new THREE.CanvasTexture(c);
  })();
  const shadow = new THREE.Mesh(
    new THREE.PlaneGeometry(2, 2),
    new THREE.MeshBasicMaterial({ color: 0x000000, map: shadowTex, transparent: true, opacity: 0.35, depthWrite: false, toneMapped: false })
  );
  shadow.rotation.order = 'YXZ';
  shadow.rotation.x = -Math.PI / 2;
  scene.add(shadow);

  // The ground, up close: see the long note at the same place in bike.js.
  const grain = (() => {
    const c = document.createElement('canvas');
    c.width = c.height = 256;
    const ctx = c.getContext('2d');
    const img = ctx.createImageData(256, 256);
    for (let i = 0; i < img.data.length; i += 4) {
      const v = 80 + Math.random() * 110;
      img.data[i] = img.data[i + 1] = img.data[i + 2] = v;
      img.data[i + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
    const tex = new THREE.CanvasTexture(c);
    tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
    const TILE = 4, SIZE = 240;
    const mat = new THREE.ShaderMaterial({
      uniforms: { uTex: { value: tex }, uOpacity: { value: 0 }, uRepeat: { value: SIZE / TILE } },
      vertexShader: `
        varying vec2 vUv; varying vec2 vPos;
        void main() { vUv = uv; vPos = position.xy; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`,
      fragmentShader: `
        uniform sampler2D uTex; uniform float uOpacity; uniform float uRepeat;
        varying vec2 vUv; varying vec2 vPos;
        void main() {
          float g = texture2D(uTex, vUv * uRepeat).r;
          float fade = 1.0 - smoothstep(50.0, 110.0, length(vPos));
          gl_FragColor = vec4(vec3(g), uOpacity * fade);
        }`,
      transparent: true, depthWrite: false, toneMapped: false,
    });
    const mesh = new THREE.Mesh(new THREE.PlaneGeometry(SIZE, SIZE), mat);
    mesh.rotation.x = -Math.PI / 2;
    mesh.position.y = 0.02;
    mesh.renderOrder = -1;
    scene.add(mesh);
    return {
      update() {
        mesh.position.x = Math.round(phys.pos.x / TILE) * TILE;
        mesh.position.z = Math.round(phys.pos.z / TILE) * TILE;
        mat.uniforms.uOpacity.value = TUNE.grain * clamp(1 - (phys.pos.y - 4) / 36, 0, 1);
        mesh.visible = mat.uniforms.uOpacity.value > 0.01;
      }
    };
  })();

  const _look = new THREE.Vector3(), _tmp = new THREE.Vector3();
  const _up = new THREE.Vector3(0, 1, 0), _q = new THREE.Quaternion();
  const camPos = new THREE.Vector3();
  let camInit = false, camMode = 0, time = 0;
  let aspect = 1, mapFov = 42;
  let lookRest = 1;                             // degrees the cockpit's camera rests below the nose: see resize
  let lookNow = lookRest;                       // and where it is now, eased
  const head = new THREE.Vector3(), headV = new THREE.Vector3();

  // The car's axes in the world: the nose, and the driver's left.
  const nose = () => qrot(b.q, [1, 0, 0]);
  const left = () => qrot(b.q, [0, 0, -1]);
  const bankDeg = () => Math.asin(clamp(left()[1], -1, 1)) / RAD;
  const noseDeg = () => Math.asin(clamp(nose()[1], -1, 1)) / RAD;
  const headingDeg = () => { const n = nose(); return (Math.atan2(n[0], -n[2]) / RAD + 360) % 360; };

  /* The roadster's model is moved by the roadster's own update, which reads
   * a car of that project. This is the little of one it reads, filled from
   * the F-5: the wheels hang on their springs once off the ground and turn
   * with the stick (so the yoke does), the wing comes up at speed and
   * stands up on the brake, the fans turn with the core, and each nozzle
   * glows with the air that leaves it. */
  const yW = -P.cgHeight + P.wheelRadius;
  const car = {
    pos: { x: 0, y: 0, z: 0 }, q: { x: 0, y: 0, z: 0, w: 1 },
    P: { suspension: { rest: 0 } },
    wheels: [0, 1, 2, 3].map(() => ({ mount: { y: yW }, comp: 0, steer: 0, spin: 0 })),
    aeroMode: 'road',
    fan: { n: 0 },
    thr: P.thrusters.map(() => ({ u: 0 }))
  };
  let droop = 0;

  function sync(dt) {
    U.uTime.value = time;
    holder.position.set(b.p[0], b.p[1], b.p[2]);
    holder.quaternion.set(b.q[1], b.q[2], b.q[3], b.q[0]);
    design.position.set(-b.mp.cg[0], -b.mp.cg[1], -b.mp.cg[2]);

    const speed = phys.vel.length();
    droop += ((b.contactN > 0 ? 0 : 1) - droop) * Math.min(1, dt * 5);
    for (let i = 0; i < 4; i++) {
      const w = car.wheels[i];
      w.comp = -0.07 * droop;
      w.steer = i < 2 ? -phys.lean * 0.11 : 0;      // 6 degrees of wheel, 25 of yoke
      // rolling on the ground they turn; in the air they run down
      if (b.contactN > 0) w.spin += Math.hypot(b.v[0], b.v[2]) / P.wheelRadius * dt;
    }
    car.aeroMode = phys.brake ? 'high' : speed > 28 ? 'low' : 'road';
    car.fan.n = clamp((b.eng.N - 0.25) / 0.75, 0, 1) * 0.7;
    const rear = clamp(b.R / (ENG.tmax * 0.6), 0, 1) * 0.7 + clamp(b.abOut, 0, 1) * 0.5;
    for (let i = 0; i < 4; i++) car.thr[i].u = phys.alive ? rear : 0;
    for (let i = 0; i < 4; i++) car.thr[4 + i].u = phys.alive ? clamp(b.L[i] / 1600, 0, 1) * 0.7 : 0;
    // the nose pair: the one on the left blows left and swings the nose right
    car.thr[8].u = clamp(b.sd[0] / (0.6 * SIDE_MAX), 0, 1);
    car.thr[9].u = clamp(-b.sd[0] / (0.6 * SIDE_MAX), 0, 1);
    model.update(car, dt, camMode ? 'cockpit' : 'chase');
    cabin.visible = !!camMode;
    if (camMode) poseArms();
    holder.visible = phys.alive;

    // The flames' nozzles, moved as FÖHN's model moves its own.
    const sweep = Math.max(b.tilt, Math.PI / 2) - Math.PI / 2;
    for (let i = 0; i < 4; i++) lifts[i].g.rotation.z = sweep + b.va[i];
    for (const g of rears) g.rotation.set(0, -b.ry, b.rp);
  }

  /* The screen beside the yoke. The roadster's shows a track mode; this one
   * shows what this car is: the speed and the height, where it is headed,
   * and the fusion core's five exits with the air each is passing. Painted
   * a few times a second and only while somebody is in the seat. */
  let screenT = 0;
  function paintScreen() {
    const cv = model.screenCanvas, c = cv.getContext('2d'), W = cv.width, H = cv.height;
    const speed = phys.vel.length() * 3.6, alt = phys.pos.y;
    c.fillStyle = '#05070a'; c.fillRect(0, 0, W, H);
    c.textBaseline = 'alphabetic';
    c.direction = 'ltr';
    c.fillStyle = '#4fd1ff'; c.font = '600 20px system-ui, sans-serif'; c.textAlign = 'left';
    c.fillText('FUSION DRIVE', 22, 34);
    c.fillStyle = '#fff'; c.font = '700 96px ui-monospace, Menlo, Consolas, monospace';
    c.fillText(String(Math.round(speed)), 18, 138);
    c.fillStyle = '#9aa5b1'; c.font = '22px system-ui, sans-serif';
    c.fillText('km/h', 24, 168);

    c.textAlign = 'right';
    c.fillStyle = '#9aa5b1'; c.font = '20px system-ui, sans-serif';
    c.fillText('ALT m', W - 22, 34);
    c.fillStyle = '#fff'; c.font = '700 44px ui-monospace, Menlo, Consolas, monospace';
    c.fillText(String(Math.round(alt)), W - 22, 80);
    c.fillStyle = '#9aa5b1'; c.font = '20px system-ui, sans-serif';
    c.fillText('HDG', W - 22, 116);
    c.fillStyle = '#fff'; c.font = '700 34px system-ui, sans-serif';
    c.direction = 'rtl';
    c.fillText(WINDS[Math.round(headingDeg() / 45) % 8], W - 22, 154);
    c.direction = 'ltr';

    // The car from above, nose up, and its nozzles.
    const cx = W / 2 + 20, cy = 232, hw = 34, hl = 54;
    c.strokeStyle = '#3a4653'; c.lineWidth = 2;
    c.beginPath();
    c.roundRect(cx - hw, cy - hl, hw * 2, hl * 2, 16);
    c.stroke();
    const dot = (x, y, u, r) => {
      c.fillStyle = `rgba(79, 209, 255, ${0.18 + 0.82 * clamp(u, 0, 1)})`;
      c.beginPath(); c.arc(x, y, r * (0.7 + 0.5 * clamp(u, 0, 1)), 0, Math.PI * 2); c.fill();
    };
    const L = b.L;
    dot(cx - hw, cy - hl + 14, L[0] / 1600, 9); dot(cx + hw, cy - hl + 14, L[1] / 1600, 9);
    dot(cx - hw, cy + hl - 14, L[2] / 1600, 9); dot(cx + hw, cy + hl - 14, L[3] / 1600, 9);
    const r = b.R / (ENG.tmax * 0.6) + b.abOut;
    for (const dx of [-21, -7, 7, 21]) dot(cx + dx, cy + hl + 6, r, 5);
    // the core
    const n = clamp((b.eng.N - ENG.idleN) / (1 - ENG.idleN), 0, 1);
    c.fillStyle = '#9aa5b1'; c.font = '18px system-ui, sans-serif'; c.textAlign = 'left';
    c.fillText('CORE', 22, 214);
    c.fillStyle = '#1b222b'; c.fillRect(22, 226, 150, 12);
    c.fillStyle = b.abOut > 0.1 ? '#ffb066' : '#4fd1ff'; c.fillRect(22, 226, 150 * n, 12);
    c.fillStyle = '#9aa5b1';
    c.fillText('LIFT', 22, 264);
    const lift = clamp((L[0] + L[1] + L[2] + L[3]) / (b.mp.m * G) / 1.6, 0, 1);
    c.fillStyle = '#1b222b'; c.fillRect(22, 274, 150, 12);
    c.fillStyle = '#4fd1ff'; c.fillRect(22, 274, 150 * lift, 12);
    model.screenTex.needsUpdate = true;
  }

  function place(x, z, yaw) {
    phys.place(x, z, yaw);
    camInit = false;
    head.set(0, 0, 0); headV.set(0, 0, 0);
    droop = 0;
    sync(1 / 60);
  }

  /** Advance the flight by `dt` seconds: the physics (FÖHN's own fixed
   *  steps), then the model, the flames, the dust, the shadow and the crash.
   *  Returns the crash reason if one happened this frame. */
  function advance(dt) {
    time += dt;
    phys.step(dt);
    const crashReason = phys.consumeCrashEvent();
    if (crashReason) { fx.explode(b.p, b.v, false); lightOnly(); }
    sync(dt);
    fx.update(dt, b, phys.sim.windW || [0, 0, 0], phys.alive);
    for (const m of rearTwins) { m.scale.copy(rearPlume.scale); m.visible = rearPlume.visible; }

    const agl = phys.pos.y;
    const n = nose();
    shadow.position.set(b.p[0], 0.08, b.p[2]);
    shadow.rotation.y = Math.atan2(n[0], n[2]);
    const grow = 1 + agl * 0.02;
    shadow.scale.set(1.45 * grow, 2.8 * grow, 1);
    shadow.material.opacity = 0.5 * clamp(1 - agl / 70, 0, 1);
    shadow.visible = phys.alive && !camMode && shadow.material.opacity > 0.02;
    grain.update();

    // The head, a mass on a spring: pressed back as the car pulls away,
    // thrown to the outside of a turn. A centimetre or two for every g.
    if (camMode) {
      const f = qinvrot(b.q, [b.acc[0], b.acc[1] + G, b.acc[2]]);   // what the seat pushes with, FÖHN's axes
      const gx = -f[2] / G, gy = f[1] / G, gz = f[0] / G;           // the roadster's: left, up, forward
      _tmp.set(clamp(-gx * 0.018, -0.05, 0.05), clamp(-(gy - 1) * 0.012, -0.04, 0.04), clamp(-gz * 0.022, -0.06, 0.06));
      _look.copy(_tmp).sub(head).multiplyScalar(140).addScaledVector(headV, -18);
      headV.addScaledVector(_look, dt);
      head.addScaledVector(headV, dt);
      head.clampLength(0, 0.08);
      screenT -= dt;
      if (screenT <= 0) { screenT = 0.12; paintScreen(); }
    }
    return crashReason;
  }

  /** Where the camera wants to be: behind the car, or in its seat. Same
   *  contract as the bikes': `out` gets a position, a unit look direction,
   *  and a roll in degrees, right side down positive. */
  function wantCamera(dt, out) {
    const speed = phys.vel.length(), n = nose();
    if (camMode === 0) {
      const hl = Math.hypot(n[0], n[1] * 0.25, n[2]) || 1;
      const fx_ = n[0] / hl, fy_ = n[1] * 0.25 / hl, fz_ = n[2] / hl;
      const dist = TUNE.camBack * 1.2 + speed * TUNE.camBackV;
      _look.set(b.p[0] - fx_ * dist, b.p[1] - fy_ * dist + TUNE.camUp * 1.1 + speed * TUNE.camUpV, b.p[2] - fz_ * dist);
      _tmp.set(b.p[0] + b.v[0] * 0.1, b.p[1] + b.v[1] * 0.1 + 0.3, b.p[2] + b.v[2] * 0.1);
      // the map cannot look up: the camera is raised, not tilted
      const hd = Math.hypot(_look.x - _tmp.x, _look.z - _tmp.z);
      const minY = _tmp.y + hd * Math.tan(TUNE.camDip * RAD);
      if (_look.y < minY) _look.y = minY;
      if (_look.y < 1.0) _look.y = 1.0;
      if (!camInit) { camPos.copy(_look); camInit = true; }
      else camPos.lerp(_look, 1 - Math.exp(-TUNE.camFollow * dt));
      const shake = Math.max(fx.shake * 0.5, b.abOut * 0.03);
      out.x = camPos.x; out.y = camPos.y; out.z = camPos.z;
      if (shake > 0.01) {
        out.x += (Math.random() - 0.5) * shake;
        out.y += (Math.random() - 0.5) * shake;
        out.z += (Math.random() - 0.5) * shake;
      }
      _tmp.sub(camPos).normalize();
      out.dx = _tmp.x; out.dy = _tmp.y; out.dz = _tmp.z;
    } else {
      // The eyes, in FÖHN's design frame: the car's forward is its x.
      const e = b.toWorld([CAB.EYE[2], P.cgHeight + CAB.EYE[1], -CAB.EYE[0]]);
      out.x = e[0]; out.y = e[1]; out.z = e[2];
      const dip = CAB.DIP * RAD;
      const d = qrot(b.q, [Math.cos(dip), -Math.sin(dip), 0]);
      out.dx = d[0]; out.dy = d[1]; out.dz = d[2];
      camInit = false;
    }
    out.roll = bankDeg();
    return out;
  }

  function setCamera(x, y, z, dx, dy, dz, roll, vfov) {
    camera.position.set(x, y, z);
    _tmp.set(x + dx, y + dy, z + dz);
    camera.up.copy(_up);
    camera.lookAt(_tmp);
    if (roll) {
      _tmp.set(dx, dy, dz).normalize();
      _q.setFromAxisAngle(_tmp, roll * RAD);
      camera.quaternion.premultiply(_q);
    }
    if (Math.abs(camera.fov - vfov) > 0.01) {
      camera.fov = vfov;
      camera.updateProjectionMatrix();
    }
    mapFov = vfov;
    if (!camMode) return;

    /* The cockpit. It looks a little below the nose, which puts the top of
     * the dashboard under the middle of the screen and leaves the upper
     * half to the windscreen. The map follows the nose down and cannot
     * follow it up (it has no pitch above the horizon); when the nose is
     * higher than the map could go, the cockpit rises over the picture
     * instead, by a share of the difference, which is what a climbing nose
     * looks like from the seat. */
    const mapDip = Math.asin(clamp(-dy, -1, 1)) / RAD;            // where the map really looks, below the horizon
    const over = (noseDeg() + mapDip) - CAB.DIP;                    // the nose above where the map shows it
    const want = clamp(lookRest + over * CAB.NOSE_K, lookRest + CAB.LOOK_MIN, lookRest + CAB.LOOK_MAX);
    lookNow += (want - lookNow) * 0.2;
    cab.position.set(CAB.EYE[0] + head.x, CAB.EYE[1] + head.y, CAB.EYE[2] + head.z);
    cab.rotation.order = 'YXZ';
    cab.rotation.set(-lookNow * RAD, Math.PI, 0);
  }

  function resize(w, h, dpr) {
    renderer.setPixelRatio(dpr);
    renderer.setSize(w, h, false);
    aspect = w / h;
    camera.aspect = aspect;
    camera.updateProjectionMatrix();
    cab.aspect = aspect;
    cab.fov = clamp(2 * Math.atan(Math.tan(CAB.HFOV / 2 * RAD) / aspect) / RAD, CAB.VFOV_MIN, CAB.VFOV_MAX);
    cab.updateProjectionMatrix();
    // The wider the lens is top to bottom, the more of the car's floor it
    // takes in, and a phone held upright was half legs. So the eyes rest
    // wherever puts the top of the dashboard at the same place on the
    // screen: a degree below the nose on a wide screen, six above it on a
    // tall one.
    lookRest = CAB.DASH - Math.atan(CAB.DASH_Y * Math.tan(cab.fov / 2 * RAD)) / RAD;
  }

  function render() {
    renderer.clear();
    if (!camMode) {
      renderer.render(scene, camera);
      return;
    }
    camera.layers.set(L_WORLD);
    renderer.render(scene, camera);
    camera.layers.enable(L_CAR);
    camera.layers.enable(L_FLAME);
    renderer.clearDepth();
    renderer.render(scene, cab);
  }

  return {
    phys, bike: { group: holder }, model, fx, shadow, camera, cab, renderer, scene,
    place, advance, wantCamera, setCamera, resize, render,
    setCamMode(m) { camMode = m ? 1 : 0; camInit = false; lookNow = lookRest; screenT = 0; },
    getCamMode: () => camMode,
    /** Degrees clockwise from north; north is -z. */
    heading: headingDeg,
    bank: bankDeg,
    nose: noseDeg,
    cabLook: () => lookNow,
    /** For the tests: how opaque the picture is at a point of the screen,
     *  0..1 from the top left. Drawn and read in one go, because the
     *  drawing buffer does not outlive the frame. */
    alphaAt(nx, ny) {
      render();
      const gl = renderer.getContext(), px = new Uint8Array(4);
      gl.readPixels(Math.round(nx * (gl.drawingBufferWidth - 1)), Math.round((1 - ny) * (gl.drawingBufferHeight - 1)), 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, px);
      return px[3] / 255;
    },
    V_MAX,
  };
}
