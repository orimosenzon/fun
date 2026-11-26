/* The F-16's cockpit, built: what is drawn around the pilot (26/11/2026).
 *
 * The first cockpit here (cockpit.js, 18/9/2026) was flat on purpose: an SVG
 * HUD and a strip of panel along the bottom of the screen. Ori asked for one
 * much closer to the real aircraft's, and for the flight to feel like a
 * modern simulator's, and the thing those have in common is that the cockpit
 * is a place: it has depth, the head can turn in it, the sun moves across
 * it in a turn, and the HUD is a glass you look through and not a sticker
 * on the screen. So this is three-dimensional, on a transparent canvas over
 * the map, the way the balloon's basket and the car's cabin are.
 *
 * One lens, not two. The balloon and the car draw their cabins through a
 * wide lens of their own over the map's narrow one, because MapLibre 4
 * could not open past sixty degrees nor roll. MapLibre 5 can do both and
 * look above the horizon, so the map is given the pilot's own eye: the same
 * position, direction, roll and field of view as the camera in here, and
 * the world outside the canopy is simply correct.
 *
 * Everything is in the aircraft's axes, metres from the pilot's eye:
 * forward is -z, right is +x, up is +y. The cockpit never moves; the head
 * turns in it, and the sun is brought into its axes every frame.
 *
 * The dimensions are the real cockpit's, near enough: the HUD's glass half a
 * metre from the eye, the panel seventy centimetres, the sills thirty below
 * the eye and ninety apart. The single-piece canopy has no frame in front,
 * which is the F-16's famous view and the reason so little is drawn above
 * the glare shield.
 *
 * The drawing of the glass and the dials is f16_draw.js; the flight is
 * f16_core.js. Loaded on the first take-off, by f16.js, and not before.
 */
import * as THREE from 'three';
import { HUD, PANEL, PARTS, ATLAS, drawHud, drawPanel, drawDyn } from './f16_draw.js';

export * from './f16_core.js';

const RAD = Math.PI / 180;
const clamp = (v, a, b) => Math.min(b, Math.max(a, v));

const GLASS = { w: 0.19, h: 0.155, z: -0.50, tilt: 22 };
const HUD_D = 40;                    // metres ahead that the symbols are hung: far enough to be infinity
export const LOOK_PITCH = -12 * RAD;  // where the eyes rest: a little down, on the top of the panel

function canvasOf(w, h) {
  const c = document.createElement('canvas');
  c.width = w; c.height = h;
  return c;
}

function tex(canvas, renderer) {
  const t = new THREE.CanvasTexture(canvas);
  t.colorSpace = THREE.SRGBColorSpace;
  t.anisotropy = renderer.capabilities.getMaxAnisotropy();
  t.generateMipmaps = true;
  t.minFilter = THREE.LinearMipmapLinearFilter;
  return t;
}

/** The plate under the HUD: the threat scope's ring, the ICP's keys, the
 *  DED's frame. 0.56 x 0.105 metres. */
function drawUpfront(c, k) {
  const W = 0.56 * k, H = 0.105 * k;
  const X = (m) => (m + 0.28) * k, Y = (m) => m * k;
  const g = c.createLinearGradient(0, 0, 0, H);
  g.addColorStop(0, '#26292c'); g.addColorStop(1, '#3d4145');
  c.fillStyle = g; c.fillRect(0, 0, W, H);
  const key = (x, y, w, h, t, s = 0.0052) => {
    c.fillStyle = '#17181a'; c.beginPath(); c.roundRect(X(x - w / 2), Y(y - h / 2), w * k, h * k, 0.002 * k); c.fill();
    c.strokeStyle = 'rgba(255,255,255,.16)'; c.lineWidth = 1.2; c.stroke();
    c.fillStyle = '#eef1f3'; c.font = `700 ${Math.round(s * k)}px ui-monospace, "DejaVu Sans Mono", monospace`;
    c.textAlign = 'center'; c.textBaseline = 'middle';
    String(t).split('\n').forEach((ln, i, a) => c.fillText(ln, X(x), Y(y + (i - (a.length - 1) / 2) * s * 1.05)));
  };
  // The ICP, 17 cm across: the override keys, the number pad, the rockers.
  c.fillStyle = '#1e2023'; c.beginPath(); c.roundRect(X(-0.088), Y(0.004), 0.176 * k, 0.097 * k, 0.004 * k); c.fill();
  c.strokeStyle = '#000'; c.lineWidth = 2; c.stroke();
  ['COM\n1', 'COM\n2', 'IFF', 'LIST', 'A-A', 'A-G'].forEach((t, i) => key(-0.066 + i * 0.0265, 0.017, 0.022, 0.017, t, 0.0044));
  const pad = [['T-ILS\n1', 'ALOW\n2', '3', 'RCL'], ['STPT\n4', 'CRUS\n5', 'TIME\n6', 'ENTR'], ['MARK\n7', 'FIX\n8', 'A-CAL\n9', 'M-SEL\n0']];
  pad.forEach((row, j) => row.forEach((t, i) => key(-0.045 + i * 0.0265, 0.041 + j * 0.0205, 0.022, 0.017, t, 0.0042)));
  key(-0.074, 0.051, 0.014, 0.036, '▲\n\n▼', 0.006);
  key(0.071, 0.04, 0.016, 0.016, 'WX', 0.0044); key(0.071, 0.062, 0.016, 0.016, 'FLIR', 0.004);
  key(0.071, 0.084, 0.016, 0.014, 'DRIFT', 0.0034);
  // The threat scope's bezel, left; the DED's frame, right.
  c.fillStyle = '#131416'; c.beginPath(); c.arc(X(-0.185), Y(0.0525), 0.043 * k, 0, 7); c.fill();
  c.strokeStyle = '#5b6066'; c.lineWidth = 2; c.stroke();
  c.fillStyle = '#0c0d0e'; c.beginPath(); c.roundRect(X(0.105), Y(0.012), 0.168 * k, 0.056 * k, 0.003 * k); c.fill();
  c.strokeStyle = '#5b6066'; c.lineWidth = 1.5; c.stroke();
  c.fillStyle = '#c9ced4'; c.font = `700 ${Math.round(0.0042 * k)}px ui-monospace, monospace`; c.textAlign = 'center';
  c.fillText('THREAT WARNING', X(-0.185), Y(0.0995)); c.fillText('DATA ENTRY DISPLAY', X(0.19), Y(0.079));
}

/** A side console's top: rows of panels with knobs and switches. */
function drawConsole(c, w, h, seed) {
  let r = seed;
  const rnd = () => { r = (r * 16807) % 2147483647; return r / 2147483647; };
  c.fillStyle = '#3f4347'; c.fillRect(0, 0, w, h);
  const rows = 9;
  for (let i = 0; i < rows; i++) {
    const y = 6 + i * (h - 12) / rows, ph = (h - 12) / rows - 8;
    c.fillStyle = '#1c1e21'; c.beginPath(); c.roundRect(8, y, w - 16, ph, 8); c.fill();
    c.strokeStyle = 'rgba(255,255,255,.08)'; c.lineWidth = 2; c.stroke();
    const n = 2 + Math.floor(rnd() * 3);
    for (let j = 0; j < n; j++) {
      const x = 8 + (w - 16) * (j + 0.5) / n, yy = y + ph * (0.35 + rnd() * 0.3);
      if (rnd() < 0.5) {
        c.fillStyle = '#0c0d0e'; c.beginPath(); c.arc(x, yy, 13, 0, 7); c.fill();
        c.strokeStyle = '#70767d'; c.lineWidth = 2; c.stroke();
        c.strokeStyle = '#eef1f3'; c.lineWidth = 3; c.beginPath(); c.moveTo(x, yy); c.lineTo(x + 9 * Math.cos(rnd() * 6), yy + 9 * Math.sin(rnd() * 6)); c.stroke();
      } else {
        c.fillStyle = '#0c0d0e'; c.beginPath(); c.arc(x, yy, 8, 0, 7); c.fill();
        c.strokeStyle = '#d7dadd'; c.lineWidth = 5; c.lineCap = 'round';
        c.beginPath(); c.moveTo(x, yy); c.lineTo(x, yy + (rnd() < 0.5 ? -14 : 14)); c.stroke();
      }
      c.fillStyle = 'rgba(238,241,243,.8)'; c.fillRect(x - 16, y + ph - 12, 32, 3);
    }
  }
}

function glow(size, stops) {
  const c = canvasOf(size, size), g = c.getContext('2d');
  const gr = g.createRadialGradient(size / 2, size / 2, 0, size / 2, size / 2, size / 2);
  for (const [o, col] of stops) gr.addColorStop(o, col);
  g.fillStyle = gr; g.fillRect(0, 0, size, size);
  return c;
}

export function makePit(canvas, opts = {}) {
  const renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: opts.antialias !== false, stencil: true, premultipliedAlpha: true, powerPreference: 'high-performance' });
  renderer.setClearColor(0x000000, 0);
  renderer.outputColorSpace = THREE.SRGBColorSpace;

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(60, 1.6, 0.03, 160);
  camera.rotation.order = 'YXZ';
  const pit = new THREE.Group();          // everything that is the aircraft
  scene.add(pit);

  const hemi = new THREE.HemisphereLight(0xdfeaff, 0x5d5a52, 1.25);
  const sun = new THREE.DirectionalLight(0xfff1da, 2.4);
  scene.add(hemi, sun, sun.target);

  const M = (color, extra = {}) => new THREE.MeshLambertMaterial({ color, ...extra });
  const mDark = M(0x1d1f22), mGrey = M(0x4a4f54), mBlack = M(0x101112), mSkin = M(0x8d949b), mSkinD = M(0x767d85);
  const box = (w, h, d, mat, x, y, z, parent = pit) => {
    const m = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), mat);
    m.position.set(x, y, z); parent.add(m); return m;
  };

  /* ---------- the glare shield ---------- */
  // An arch across the cockpit, highest in the middle, falling away forward
  // and down so that nothing of it stands above its own lip.
  const lipY = (x) => -0.112 - 0.11 * Math.pow(Math.abs(x) / 0.46, 3);
  {
    const N = 28, pos = [], idx = [];
    for (let i = 0; i <= N; i++) {
      const x = -0.47 + 0.94 * i / N, y = lipY(x);
      pos.push(x, y, -0.47, x * 1.04, y - 0.30, -1.30, x, y - 0.022, -0.475, x, y - 0.03, -0.52);
    }
    for (let i = 0; i < N; i++) {
      const a = i * 4, b = a + 4;
      idx.push(a, b, a + 1, b, b + 1, a + 1);          // the top
      idx.push(a, a + 2, b, b, a + 2, b + 2);          // the lip's edge
      idx.push(a + 2, a + 3, b + 2, b + 2, a + 3, b + 3);   // and its underside
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
    g.setIndex(idx); g.computeVertexNormals();
    pit.add(new THREE.Mesh(g, M(0x17181a, { side: THREE.DoubleSide })));
  }

  /* ---------- the HUD ---------- */
  const tilt = GLASS.tilt * RAD;
  const glassY = GLASS.z * Math.tan(HUD.DEP * RAD);       // the glass's centre: on the HUD's own axis
  const glassMat = new THREE.MeshBasicMaterial({
    color: 0x9fe0b4, transparent: true, opacity: 0.085, depthWrite: false, side: THREE.DoubleSide,
    stencilWrite: true, stencilFunc: THREE.AlwaysStencilFunc, stencilRef: 1, stencilZPass: THREE.ReplaceStencilOp
  });
  const glass = new THREE.Mesh(new THREE.PlaneGeometry(GLASS.w, GLASS.h), glassMat);
  glass.position.set(0, glassY, GLASS.z); glass.rotation.x = tilt; glass.renderOrder = 5;
  pit.add(glass);
  // A second, smaller plate behind the first, as the real combiner has: it
  // is only seen as a line of light along its top edge.
  {
    const edge = new THREE.Mesh(new THREE.PlaneGeometry(GLASS.w, 0.004), new THREE.MeshBasicMaterial({ color: 0xcfeedd, transparent: true, opacity: 0.35 }));
    edge.position.set(0, glassY + GLASS.h / 2 * Math.cos(tilt), GLASS.z + GLASS.h / 2 * Math.sin(tilt)); edge.rotation.x = tilt;
    pit.add(edge);
  }
  for (const sx of [-1, 1]) {
    // The arms that hold the glass, and the feet they stand on.
    const arm = box(0.007, GLASS.h + 0.02, 0.016, mBlack, sx * (GLASS.w / 2 + 0.0035), glassY - 0.006, GLASS.z);
    arm.rotation.x = tilt;
    box(0.022, 0.03, 0.05, mBlack, sx * (GLASS.w / 2 + 0.004), -0.118, GLASS.z - 0.045);
  }
  // The projector's housing, on the glare shield behind the glass.
  box(0.21, 0.05, 0.3, mDark, 0, -0.135, -0.69);
  box(0.13, 0.012, 0.11, mBlack, 0, -0.106, -0.60);
  {
    const lens = new THREE.Mesh(new THREE.CircleGeometry(0.045, 24), new THREE.MeshBasicMaterial({ color: 0x0c1f14 }));
    lens.position.set(0, -0.0995, -0.60); lens.rotation.x = -Math.PI / 2; pit.add(lens);
  }

  const hudCanvas = canvasOf(HUD.N, HUD.N);
  // Drawn on the processor, not the graphics card: a canvas that lives on the
  // card has to be read back from it before it can be a texture, every
  // frame, and that read-back cost half the frame rate.
  const hudCtx = hudCanvas.getContext('2d', { willReadFrequently: true });
  const hudTex = new THREE.CanvasTexture(hudCanvas);
  hudTex.colorSpace = THREE.SRGBColorSpace;
  hudTex.generateMipmaps = false; hudTex.minFilter = THREE.LinearFilter;
  const hudMat = new THREE.MeshBasicMaterial({
    map: hudTex, transparent: true, depthTest: false, depthWrite: false, toneMapped: false,
    stencilWrite: true, stencilFunc: THREE.EqualStencilFunc, stencilRef: 1
  });
  const hudSize = 2 * HUD_D * Math.tan(HUD.HALF * RAD);
  const hud = new THREE.Mesh(new THREE.PlaneGeometry(hudSize, hudSize), hudMat);
  hud.position.set(0, -HUD_D * Math.tan(HUD.DEP * RAD), -HUD_D);
  hud.renderOrder = 6;
  scene.add(hud);

  /* ---------- the live parts' atlas ---------- */
  const dynCanvas = canvasOf(ATLAS.N, ATLAS.N);
  const dynCtx = dynCanvas.getContext('2d', { willReadFrequently: true });
  const dynTex = new THREE.CanvasTexture(dynCanvas);
  dynTex.colorSpace = THREE.SRGBColorSpace;
  dynTex.generateMipmaps = false; dynTex.minFilter = THREE.LinearFilter;
  const dynMat = new THREE.MeshBasicMaterial({ map: dynTex, transparent: true, toneMapped: false });
  /** A rectangle of the atlas as a plane w x h metres. */
  const part = (key, w, h, round) => {
    const [ax, ay, aw, ah] = ATLAS[key];
    const g = round ? new THREE.CircleGeometry(w / 2, 40) : new THREE.PlaneGeometry(w, h);
    const uv = g.attributes.uv;
    for (let i = 0; i < uv.count; i++) {
      uv.setXY(i, (ax + uv.getX(i) * aw) / ATLAS.N, 1 - (ay + (1 - uv.getY(i)) * ah) / ATLAS.N);
    }
    return new THREE.Mesh(g, dynMat);
  };
  const ROUND = { adi: 1, hsi: 1, asi: 1, altm: 1, oil: 1, noz: 1, rpm: 1, ftit: 1, fuel: 1, rwr: 1 };

  /* ---------- under the HUD: the threat scope, the ICP, the DED ---------- */
  const up = new THREE.Group();
  up.position.set(0, -0.204, -0.545); up.rotation.x = -10 * RAD;
  pit.add(up);
  {
    const k = 3000, c = canvasOf(0.56 * k, 0.105 * k);
    drawUpfront(c.getContext('2d'), k);
    up.add(new THREE.Mesh(new THREE.PlaneGeometry(0.56, 0.105), M(0xffffff, { map: tex(c, renderer) })));
    const rwr = part('rwr', 0.078, 0.078, true); rwr.position.set(-0.185, 0, 0.002); up.add(rwr);
    const ded = part('ded', 0.158, 0.046); ded.position.set(0.189, 0.0085, 0.002); up.add(ded);
  }
  // The warning lights on the glare shield's lip, either side of the HUD.
  for (const [key, x, w] of [['browL', -0.20, 0.07], ['browR', 0.235, 0.14]]) {
    const ph = w * ATLAS[key][3] / ATLAS[key][2] * 0.62;
    box(w + 0.006, ph + 0.006, 0.03, mBlack, x, lipY(x) - 0.004, -0.487);
    const p = part(key, w, ph); p.position.set(x, lipY(x) - 0.004, -0.4715); pit.add(p);
  }
  // The indexers, on the HUD's arms: angle of attack left, nosewheel steering right.
  for (const [key, sx] of [['aoaIdx', -1], ['nwsIdx', 1]]) {
    box(0.018, 0.042, 0.02, mBlack, sx * 0.118, -0.062, -0.50);
    const p = part(key, 0.0144, 0.036); p.position.set(sx * 0.118, -0.062, -0.4895); pit.add(p);
  }
  // The underside between the plate and the panel.
  {
    const s = new THREE.Mesh(new THREE.PlaneGeometry(0.9, 0.19), mBlack);
    s.position.set(0, -0.272, -0.625); s.rotation.x = Math.PI / 2 - 16 * RAD; pit.add(s);
  }

  /* ---------- the instrument panel ---------- */
  const panel = new THREE.Group();
  const PT = 14 * RAD;
  panel.position.set(0, -0.30 - PANEL.H / 2 * Math.cos(PT), -0.70 + PANEL.H / 2 * Math.sin(PT));
  panel.rotation.x = -PT;
  pit.add(panel);
  {
    const c = canvasOf(Math.round(PANEL.W * PANEL.PPM), Math.round(PANEL.H * PANEL.PPM));
    drawPanel(c.getContext('2d'));
    panel.add(new THREE.Mesh(new THREE.PlaneGeometry(PANEL.W, PANEL.H), M(0xffffff, { map: tex(c, renderer), alphaTest: 0.5, transparent: false })));
    for (const key in PARTS) {
      const [x, y, w, h] = PARTS[key];
      const p = part(key, w, h, ROUND[key]);
      p.position.set(x, PANEL.H / 2 - y, 0.0015);
      panel.add(p);
    }
  }
  // The gear handle: a lever with a wheel on its end, up or down in its slot.
  const gearH = new THREE.Group();
  {
    const arm = new THREE.Mesh(new THREE.CylinderGeometry(0.004, 0.004, 0.04, 8), mGrey);
    arm.rotation.x = Math.PI / 2; arm.position.z = 0.02; gearH.add(arm);
    const wheel = new THREE.Mesh(new THREE.CylinderGeometry(0.014, 0.014, 0.012, 20), M(0xd8dbde));
    wheel.rotation.z = Math.PI / 2; wheel.position.z = 0.045; gearH.add(wheel);
    const hub = new THREE.Mesh(new THREE.CylinderGeometry(0.006, 0.006, 0.014, 12), M(0xc63b2e, { emissive: 0x000000 }));
    hub.rotation.z = Math.PI / 2; hub.position.z = 0.045; gearH.add(hub);
    gearH.userData.hub = hub;
    panel.add(gearH);
  }

  /* ---------- the tub: walls, sills, consoles, floor ---------- */
  for (const sx of [-1, 1]) {
    const wall = new THREE.Mesh(new THREE.PlaneGeometry(2.0, 0.62), mGrey);
    wall.position.set(sx * 0.445, -0.61, -0.2); wall.rotation.y = -sx * Math.PI / 2; pit.add(wall);
    // The sill, and the canopy's own frame on top of it.
    box(0.06, 0.035, 2.3, mDark, sx * 0.47, -0.30, -0.25);
    box(0.03, 0.022, 2.3, mBlack, sx * 0.468, -0.275, -0.25);
    // The console.
    const cc = canvasOf(256, 1024); drawConsole(cc.getContext('2d'), 256, 1024, sx < 0 ? 7 : 23);
    const con = new THREE.Mesh(new THREE.PlaneGeometry(0.19, 1.0), M(0xffffff, { map: tex(cc, renderer) }));
    con.position.set(sx * 0.348, -0.50, -0.07); con.rotation.x = -Math.PI / 2; pit.add(con);
    const face = new THREE.Mesh(new THREE.PlaneGeometry(1.0, 0.42), mDark);
    face.position.set(sx * 0.253, -0.71, -0.07); face.rotation.y = -sx * Math.PI / 2; pit.add(face);
  }
  {
    const floor = new THREE.Mesh(new THREE.PlaneGeometry(1.0, 2.0), mBlack);
    floor.position.set(0, -0.92, -0.2); floor.rotation.x = -Math.PI / 2; pit.add(floor);
    // The seat: its pan, its back, its headrest and the handle between the knees.
    box(0.42, 0.06, 0.42, mDark, 0, -0.62, 0.02);
    box(0.44, 0.75, 0.08, mDark, 0, -0.30, 0.30).rotation.x = -0.28;
    box(0.26, 0.20, 0.09, M(0x2a2c2f), 0, 0.03, 0.25);
    box(0.05, 0.035, 0.05, M(0xd9b21c), 0, -0.575, -0.17);
  }
  // The throttle, on the left console: it slides with the lever.
  const throttle = new THREE.Group();
  {
    const grip = new THREE.Mesh(new THREE.CapsuleGeometry(0.024, 0.07, 6, 12), mBlack);
    grip.rotation.z = Math.PI / 2; grip.position.y = 0.075; throttle.add(grip);
    box(0.02, 0.075, 0.03, mDark, 0, 0.035, 0, throttle);
    throttle.position.set(-0.335, -0.50, -0.18);
    pit.add(throttle);
    const slot = new THREE.Mesh(new THREE.PlaneGeometry(0.03, 0.2), mBlack);
    slot.position.set(-0.335, -0.498, -0.26); slot.rotation.x = -Math.PI / 2; pit.add(slot);
  }
  // The side stick, on the right: it barely moves in the real aircraft, a
  // few millimetres against force sensors; here it leans enough to be seen.
  const stick = new THREE.Group();
  {
    box(0.07, 0.03, 0.08, mDark, 0, 0.015, 0, stick);
    const grip = new THREE.Mesh(new THREE.CapsuleGeometry(0.019, 0.095, 6, 12), mBlack);
    grip.position.y = 0.095; grip.rotation.x = -0.22; stick.add(grip);
    const head = new THREE.Mesh(new THREE.SphereGeometry(0.024, 14, 10), mBlack);
    head.position.set(0, 0.155, 0.014); stick.add(head);
    const hat = new THREE.Mesh(new THREE.CylinderGeometry(0.007, 0.007, 0.012, 10), mGrey);
    hat.position.set(0, 0.181, 0.012); stick.add(hat);
    stick.position.set(0.345, -0.50, -0.34);
    pit.add(stick);
    // An arm rest behind it, where the forearm lies at nine g.
    box(0.09, 0.03, 0.2, mDark, 0.36, -0.445, -0.12);
  }
  // The canopy's rear frame, an arch behind the pilot's shoulders.
  {
    const bow = new THREE.Mesh(new THREE.TorusGeometry(0.50, 0.028, 8, 28, Math.PI), mDark);
    bow.position.set(0, -0.30, 0.42); pit.add(bow);
  }

  /* ---------- the aircraft outside: seen over a shoulder ---------- */
  {
    const flat = (pts, y, mat) => {
      const s = new THREE.Shape(pts.map(([x, z]) => new THREE.Vector2(x, z)));
      const m = new THREE.Mesh(new THREE.ExtrudeGeometry(s, { depth: 0.07, bevelEnabled: false }), mat);
      m.rotation.x = Math.PI / 2; m.position.y = y; pit.add(m); return m;
    };
    for (const sx of [-1, 1]) {
      // The strake, then the wing: forty degrees of sweep, a missile on the tip.
      flat([[sx * 0.55, -0.9], [sx * 1.0, 1.9], [sx * 0.55, 1.9]], -0.72, mSkinD);
      flat([[sx * 0.95, 1.6], [sx * 4.72, 4.1], [sx * 4.72, 5.1], [sx * 0.95, 5.4]], -0.72, mSkin);
      const msl = new THREE.Mesh(new THREE.CylinderGeometry(0.065, 0.065, 2.9, 10), M(0xdfe2e5));
      msl.rotation.x = Math.PI / 2; msl.position.set(sx * 4.82, -0.74, 4.2); pit.add(msl);
      const nose = new THREE.Mesh(new THREE.ConeGeometry(0.065, 0.32, 10), M(0x44474b));
      nose.rotation.x = -Math.PI / 2; nose.position.set(sx * 4.82, -0.74, 2.59); pit.add(nose);
      box(0.012, 0.36, 0.3, M(0xdfe2e5), sx * 4.82, -0.74, 5.4);
      box(0.36, 0.012, 0.3, M(0xdfe2e5), sx * 4.82, -0.74, 5.4);
      // The tailplane.
      flat([[sx * 0.8, 7.6], [sx * 2.75, 8.9], [sx * 2.75, 9.6], [sx * 0.8, 9.7]], -0.80, mSkinD);
    }
    // The spine, and the fin on it.
    const spine = new THREE.Mesh(new THREE.CylinderGeometry(0.62, 0.72, 9.2, 14), mSkin);
    spine.rotation.x = Math.PI / 2; spine.position.set(0, -0.95, 5.3); spine.scale.set(1.25, 1, 0.8); pit.add(spine);
    const fs = new THREE.Shape([[6.2, 0], [9.9, 0], [10.3, 2.75], [9.2, 2.75]].map(([z, y]) => new THREE.Vector2(z, y)));
    const fin = new THREE.Mesh(new THREE.ExtrudeGeometry(fs, { depth: 0.1, bevelEnabled: false }), mSkinD);
    fin.rotation.y = -Math.PI / 2; fin.position.set(0.05, -0.45, 0); pit.add(fin);
    // The nose, for the eye that leans forward: grey, then the radome's darker cone.
    const fore = new THREE.Mesh(new THREE.CylinderGeometry(0.30, 0.52, 2.2, 14), mSkin);
    fore.rotation.x = -Math.PI / 2; fore.position.set(0, -0.98, -2.2); pit.add(fore);
    const rad = new THREE.Mesh(new THREE.ConeGeometry(0.30, 1.6, 14), M(0x5a5f65));
    rad.rotation.x = -Math.PI / 2; rad.position.set(0, -0.98, -4.1); pit.add(rad);
  }

  /* ---------- the sun ---------- */
  const sunSprite = new THREE.Sprite(new THREE.SpriteMaterial({
    map: new THREE.CanvasTexture(glow(256, [[0, 'rgba(255,255,245,1)'], [0.07, 'rgba(255,252,225,1)'], [0.12, 'rgba(255,240,190,.55)'], [0.35, 'rgba(255,230,170,.14)'], [1, 'rgba(255,230,170,0)']])),
    transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, toneMapped: false
  }));
  sunSprite.scale.setScalar(70);
  scene.add(sunSprite);

  /* ---------- the frame ---------- */
  let mode = 'pit';
  let turn = 0, frame = 0;
  const v3 = new THREE.Vector3(), qc = new THREE.Quaternion();
  const head = { x: 0, y: 0, z: 0 };
  let lastFpm = null;

  function resize(w, h, dpr) {
    renderer.setPixelRatio(dpr);
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  }

  /** 'pit' is the cockpit; 'hud' is the symbols alone on the open sky, for
   *  when the panel is in the way of the town. */
  function setMode(m) {
    mode = m === 'hud' ? 'hud' : 'pit';
    pit.visible = mode === 'pit';
    hudMat.stencilWrite = mode === 'pit';
    hudMat.stencilFunc = mode === 'pit' ? THREE.EqualStencilFunc : THREE.AlwaysStencilFunc;
    hudMat.needsUpdate = true;
  }

  /** One frame. `s` carries the flight for the instruments (see drawHud and
   *  drawDyn), and: q, the attitude; look { yaw, pitch }, the head, radians,
   *  right and up positive; vfov, degrees; sun, the direction to it in the
   *  world; shake, 0..1; lever, stickX, stickY for the hands. */
  function render(s) {
    // The head. It leans into a look to the side, sinks a little under g,
    // and rattles in the buffet.
    const hx = Math.sin(s.look.yaw) * 0.07, hy = -clamp((s.nz - 1) * 0.004, -0.012, 0.03);
    head.x += (hx - head.x) * 0.2; head.y += (hy - head.y) * 0.15;
    const sh = s.shake || 0;
    camera.position.set(head.x + (Math.random() - 0.5) * 0.006 * sh, head.y + (Math.random() - 0.5) * 0.006 * sh, 0);
    camera.rotation.set(s.look.pitch, -s.look.yaw, 0);
    if (camera.fov !== s.vfov) { camera.fov = s.vfov; camera.updateProjectionMatrix(); }

    // The sun, brought into the aircraft's axes.
    qc.set(-s.q[0], -s.q[1], -s.q[2], s.q[3]);
    v3.set(s.sun[0], s.sun[1], s.sun[2]).applyQuaternion(qc);
    sun.position.copy(v3).multiplyScalar(10); sun.target.position.set(0, 0, 0);
    sunSprite.position.copy(v3).multiplyScalar(140);
    // The light under the canopy is the sky's, and the sky is wherever up is.
    v3.set(0, 1, 0).applyQuaternion(qc);
    hemi.position.copy(v3);

    lastFpm = drawHud(hudCtx, s);
    hudTex.needsUpdate = true;
    if (mode === 'pit') {
      if ((frame & 1) === 0) { drawDyn(dynCtx, s, turn++); dynTex.needsUpdate = true; }
      throttle.position.z = -0.18 - 0.13 * s.lever;
      stick.rotation.z = -s.stickX * 0.2; stick.rotation.x = s.stickY * 0.16;
      gearH.position.set(-0.254, PANEL.H / 2 - (0.185 + 0.06 * (s.gearDown ? 1 : 0)), 0.002);
      gearH.userData.hub.material.emissive.setHex(s.gear > 0.01 && s.gear < 0.99 ? 0xff2a1a : 0x000000);
    }
    frame++;
    renderer.render(scene, camera);
  }

  setMode('pit');
  return {
    resize, setMode, render,
    getMode: () => mode,
    debug: () => ({ mode, frame, fpm: lastFpm, children: pit.children.length }),
    canvases: { hud: hudCanvas, dyn: dynCanvas }
  };
}
