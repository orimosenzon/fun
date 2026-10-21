// Shared 3D scaffolding: z-up scene, orbit controls, HTML labels, render on demand, theme-aware materials.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { CSS2DRenderer, CSS2DObject } from 'three/addons/renderers/CSS2DRenderer.js';
import { token, themeHooks } from '../plot.js';

export { THREE };
export const col = name => new THREE.Color(token(name).slice(0, 7));
export const V = (x, y, z) => new THREE.Vector3(x, y, z);

export class View3D {
  constructor(el, { cam = [7, -9, 6], target = [0, 0, 0], fov = 35 } = {}) {
    this.el = el;
    this.renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    el.appendChild(this.renderer.domElement);
    this.labels = new CSS2DRenderer();
    Object.assign(this.labels.domElement.style, { position: 'absolute', inset: '0', pointerEvents: 'none' });
    el.appendChild(this.labels.domElement);

    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(fov, 1, 0.05, 500);
    this.camera.up.set(0, 0, 1);
    this.camera.position.set(...cam);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.target.set(...target);
    this.controls.enableZoom = false;   // the wheel keeps scrolling the page
    this.controls.enablePan = false;
    this.controls.update();
    this.controls.addEventListener('change', () => this.request());

    const hemi = new THREE.HemisphereLight(0xffffff, 0x666666, 1.7);
    hemi.position.set(0, 0, 1);
    const sun = new THREE.DirectionalLight(0xffffff, 1.5);
    sun.position.set(4, -6, 10);
    this.scene.add(hemi, sun);

    this.mats = [];
    themeHooks.add(() => { this.recolor(); this.request(); });
    new ResizeObserver(() => this.resize()).observe(el);
  }

  // material whose color follows a CSS token, so dark mode just works
  mat(tok, { kind = 'standard', ...opts } = {}) {
    const Ctor = { standard: THREE.MeshStandardMaterial, basic: THREE.MeshBasicMaterial, line: THREE.LineBasicMaterial }[kind];
    const params = { color: col(tok), ...opts };
    if (kind === 'standard' && params.roughness === undefined) params.roughness = 0.75;
    const m = new Ctor(params);
    this.mats.push([m, tok]);
    return m;
  }

  recolor() {
    for (const [m, tok] of this.mats) m.color = col(tok);
    this.onTheme && this.onTheme();
  }

  resize() {
    const w = this.el.clientWidth, h = this.el.clientHeight;
    if (!w || !h) return;
    this.renderer.setSize(w, h);
    this.labels.setSize(w, h);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.request();
  }

  request() {
    if (this.pending) return;
    this.pending = true;
    requestAnimationFrame(() => {
      this.pending = false;
      this.scene.background = col('--plot-bg');
      this.renderer.render(this.scene, this.camera);
      this.labels.render(this.scene, this.camera);
    });
  }

  label(text, cls = '') {
    const d = document.createElement('div');
    d.className = 'lbl3 ' + cls;
    d.dir = 'auto';
    d.textContent = text;
    return new CSS2DObject(d);
  }

  // group that can be emptied and refilled each update
  layer() {
    const g = new THREE.Group();
    this.scene.add(g);
    g.reset = () => {
      for (const c of [...g.children]) {
        g.remove(c);
        c.traverse(o => {
          if (o.geometry) o.geometry.dispose();
          if (o.element) o.element.remove();
        });
      }
    };
    return g;
  }
}

/* ---------- geometry helpers ---------- */

// height field z = fn(x, y) over a rectangle, colored by height between two tokens
export function surface(fn, [x0, x1, y0, y1], n, material, colorAt) {
  const pos = [], colors = [], idx = [];
  for (let j = 0; j <= n; j++) {
    for (let i = 0; i <= n; i++) {
      const x = x0 + (x1 - x0) * i / n, y = y0 + (y1 - y0) * j / n, z = fn(x, y);
      pos.push(x, y, z);
      if (colorAt) { const c = colorAt(z); colors.push(c.r, c.g, c.b); }
    }
  }
  for (let j = 0; j < n; j++) {
    for (let i = 0; i < n; i++) {
      const a = j * (n + 1) + i, b = a + 1, c = a + n + 1, d = c + 1;
      idx.push(a, b, d, a, d, c);
    }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  if (colorAt) g.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
  g.setIndex(idx);
  g.computeVertexNormals();
  return new THREE.Mesh(g, material);
}

export function tube(points, radius, material, closed = false) {
  const curve = new THREE.CatmullRomCurve3(points.map(p => p.isVector3 ? p : V(...p)), closed);
  return new THREE.Mesh(new THREE.TubeGeometry(curve, Math.max(8, points.length * 2), radius, 8, closed), material);
}

export function segments(pairs, material) {
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(pairs.flat(), 3));
  return new THREE.LineSegments(g, material);
}

export function ball(p, r, material) {
  const m = new THREE.Mesh(new THREE.SphereGeometry(r, 20, 14), material);
  m.position.set(...(p.isVector3 ? p.toArray() : p));
  return m;
}

// arrow from p along v (true length), shaft radius r
export function arrow(p, v, material, r = 0.035) {
  const from = p.isVector3 ? p.clone() : V(...p), dir = v.isVector3 ? v.clone() : V(...v);
  const L = dir.length(), g = new THREE.Group();
  if (L < 1e-6) return g;
  const head = Math.min(L * 0.35, r * 7);
  const shaft = new THREE.Mesh(new THREE.CylinderGeometry(r, r, L - head, 10), material);
  shaft.position.y = (L - head) / 2;
  const cone = new THREE.Mesh(new THREE.ConeGeometry(r * 2.8, head, 14), material);
  cone.position.y = L - head / 2;
  g.add(shaft, cone);
  g.position.copy(from);
  g.quaternion.setFromUnitVectors(V(0, 1, 0), dir.normalize());
  return g;
}

// flat square (or disk) through p with normal nrm
export function patch(p, nrm, size, material, disk = false) {
  const geo = disk ? new THREE.CircleGeometry(size, 48) : new THREE.PlaneGeometry(size, size);
  const m = new THREE.Mesh(geo, material);
  m.position.set(...(p.isVector3 ? p.toArray() : p));
  m.quaternion.setFromUnitVectors(V(0, 0, 1), (nrm.isVector3 ? nrm.clone() : V(...nrm)).normalize());
  return m;
}

export function translucent(view, tok, opacity) {
  return view.mat(tok, { transparent: true, opacity, side: THREE.DoubleSide, depthWrite: false });
}

export function axesLines(view, [x0, x1, y0, y1, z0, z1], names = ['x', 'y', 'z']) {
  const g = new THREE.Group();
  const m = view.mat('--axis', { kind: 'line' });
  g.add(segments([[x0, 0, 0, x1, 0, 0], [0, y0, 0, 0, y1, 0], [0, 0, z0, 0, 0, z1]], m));
  [[x1, 0, 0], [0, y1, 0], [0, 0, z1]].forEach((p, i) => {
    const l = view.label(names[i], 'axis');
    l.position.set(...p);
    g.add(l);
  });
  return g;
}
