/* The F-5 of FÖHN over the map: what is drawn.
 *
 * The flight is f5_core.js; this file is the rig explore.js holds, with the
 * same handles as the first bike's (bike.js, makeRig): one scene with only
 * the bike in it, on a transparent canvas over MapLibre's picture, and a
 * camera that is set every frame from the map's own.
 *
 * The model, its flames and its dust are FÖHN's files, imported like the
 * flight. FÖHN draws through a post-processing pass that this page has no
 * use for (the world here is a photograph underneath), so the tone mapping
 * is the renderer's own and the sky the metal reflects is a plain gradient.
 *
 * Loaded on the first take-off, by bike5.js, and not before: nobody who came
 * for a trail should pay for a motorcycle they may never ride.
 */
import * as THREE from '../../jetbikes/foehn/lib/three.module.min.js';
import { buildBike } from '../../jetbikes/foehn/js/render/bikeModel.js';
import { FX } from '../../jetbikes/foehn/js/render/fx.js';
import { U } from '../../jetbikes/foehn/js/render/env.js';
import { qrot } from '../../jetbikes/foehn/js/sim/vmath.js';
import { F5, yawFor } from './f5_core.js';

export { yawFor };

const clamp = (v, a, b) => Math.min(b, Math.max(a, v));

/** A sky for the paint and the metal to reflect: pale at the horizon, blue
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

/** `TUNE` is the first bike's (Bike.TUNE): the chase camera and the grain on
 *  the ground are the same for both, so that swapping bikes does not move
 *  the picture. */
export function makeRig(canvas, TUNE) {
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true, premultipliedAlpha: true });
  renderer.setClearColor(0x000000, 0);
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.0;
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(42, 1, 0.1, 6000);

  const sun = new THREE.DirectionalLight(0xfff2dd, Math.PI * 1.1);
  sun.position.set(700, 1000, -400);
  scene.add(sun);
  scene.add(new THREE.HemisphereLight(0xbdd7f2, 0x62705a, Math.PI * 0.6));
  scene.environment = skyEnv(renderer);
  scene.environmentIntensity = 0.6;

  const model = buildBike();
  scene.add(model.root);
  const fx = new FX(scene, model);
  const phys = new F5();
  const b = phys.b;

  /* The flames, the glows and the sparks are additive, and FÖHN draws them
   * over its own opaque world. Here the canvas is transparent and the world
   * is the map underneath, and three.js's additive blending also adds to the
   * canvas's alpha: a faint blue flame became a dark, nearly solid column
   * over the photograph (Ori, 20/11/2026). So anything additive adds its
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

  // soft blob shadow on the ground - crucial for judging altitude
  const shadow = new THREE.Mesh(
    new THREE.CircleGeometry(1.2, 24),
    new THREE.MeshBasicMaterial({ color: 0x000000, transparent: true, opacity: 0.35, depthWrite: false })
  );
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

  // The bike's axes in the world: the nose, and the rider's left.
  const nose = () => qrot(b.q, [1, 0, 0]);
  const left = () => qrot(b.q, [0, 0, -1]);
  const bankDeg = () => Math.asin(clamp(left()[1], -1, 1)) * 180 / Math.PI;

  function sync(dt) {
    U.uTime.value = time;
    model.update(b, time, phys.lean, dt);
    model.root.visible = phys.alive;
  }

  function place(x, z, yaw) {
    phys.place(x, z, yaw);
    camInit = false;
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

    const agl = phys.pos.y;
    shadow.position.set(b.p[0], 0.08, b.p[2]);
    shadow.scale.setScalar(1 + agl * 0.02);
    shadow.material.opacity = 0.38 * clamp(1 - agl / 70, 0, 1);
    shadow.visible = phys.alive && shadow.material.opacity > 0.02;
    grain.update();
    return crashReason;
  }

  /** Where the camera wants to be: the first bike's chase camera, or the
   *  rider's eyes. Same contract as bike.js: `out` gets a position, a unit
   *  look direction, and a roll in degrees, right side down positive. */
  function wantCamera(dt, out) {
    const speed = phys.vel.length(), n = nose();
    if (camMode === 0) {
      const hl = Math.hypot(n[0], n[1] * 0.25, n[2]) || 1;
      const fx_ = n[0] / hl, fy_ = n[1] * 0.25 / hl, fz_ = n[2] / hl;
      const dist = TUNE.camBack + speed * TUNE.camBackV;
      _look.set(b.p[0] - fx_ * dist, b.p[1] - fy_ * dist + TUNE.camUp + speed * TUNE.camUpV, b.p[2] - fz_ * dist);
      _tmp.set(b.p[0] + b.v[0] * 0.1, b.p[1] + b.v[1] * 0.1 + 0.3, b.p[2] + b.v[2] * 0.1);
      // the map cannot look up: the camera is raised, not tilted
      const hd = Math.hypot(_look.x - _tmp.x, _look.z - _tmp.z);
      const minY = _tmp.y + hd * Math.tan(TUNE.camDip * Math.PI / 180);
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
      const e = b.toWorld([0.16, 1.5, 0]);
      out.x = e[0]; out.y = e[1]; out.z = e[2];
      const dip = TUNE.eyeDip * Math.PI / 180;
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
      _q.setFromAxisAngle(_tmp, roll * Math.PI / 180);
      camera.quaternion.premultiply(_q);
    }
    if (Math.abs(camera.fov - vfov) > 0.01) {
      camera.fov = vfov;
      camera.updateProjectionMatrix();
    }
  }

  function resize(w, h, dpr) {
    renderer.setPixelRatio(dpr);
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  }

  return {
    phys, bike: { group: model.root }, fx, shadow, camera, renderer, scene,
    place, advance, wantCamera, setCamera, resize,
    render() { renderer.render(scene, camera); },
    setCamMode(m) { camMode = m; camInit = false; },
    getCamMode: () => camMode,
    /** Degrees clockwise from north; north is -z. */
    heading() { const n = nose(); return (Math.atan2(n[0], -n[2]) * 180 / Math.PI + 360) % 360; },
    bank: bankDeg,
    nose() { return Math.asin(clamp(nose()[1], -1, 1)) * 180 / Math.PI; },
  };
}
