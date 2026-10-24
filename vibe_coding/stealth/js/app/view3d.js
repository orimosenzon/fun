// התצוגה התלת-ממדית: המטוס, הקצוות המפזרים, קרן המכ"ם ובועת ה-RCS
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

// מהצירים שלנו (x קדימה, y שמאלה, z למעלה) לצירים של three (y למעלה)
const T = (x, y, z) => new THREE.Vector3(x, z, -y);
const PART_COLORS = [0x3a4a66, 0x56709a, 0x7d6fb0, 0x5b8a9c, 0xb0653a];

export class View3D {
  constructor(el) {
    this.el = el;
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    this.renderer.setPixelRatio(Math.min(2, window.devicePixelRatio));
    this.renderer.setClearColor(0x070b14);
    el.appendChild(this.renderer.domElement);
    this.scene = new THREE.Scene();
    this.scene.fog = new THREE.Fog(0x070b14, 90, 260);
    this.camera = new THREE.PerspectiveCamera(40, 1, 0.1, 2000);
    this.camera.position.set(-24, 16, 26);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true; this.controls.target.set(0, 0, 0);
    this.scene.add(new THREE.HemisphereLight(0xa9c6ff, 0x1a1f2b, 0.9));
    const d = new THREE.DirectionalLight(0xffffff, 1.6); d.position.set(-20, 40, 15); this.scene.add(d);
    const d2 = new THREE.DirectionalLight(0x4fd1ff, 0.35); d2.position.set(20, -10, -30); this.scene.add(d2);
    const grid = new THREE.PolarGridHelper(60, 24, 6, 72, 0x1a2a4a, 0x14203a);
    grid.position.y = -4; this.scene.add(grid);
    this.grid = grid;
    this.group = new THREE.Group(); this.scene.add(this.group);
    this.beam = new THREE.Group(); this.scene.add(this.beam);
    this.bubbleObj = null; this.edgesObj = null; this.mesh = null;
    this.mode = { parts: true, heat: false, bubble: false, edges: true, beam: true };
    this.pulses = [];
    this._buildBeam();
    new ResizeObserver(() => this.resize()).observe(el);
    this.resize();
    this.t0 = performance.now();
    const loop = () => { requestAnimationFrame(loop); this.tick(); };
    loop();
  }
  resize() {
    const w = this.el.clientWidth, h = this.el.clientHeight;
    this.renderer.setSize(w, h, false);
    this.renderer.domElement.style.width = w + 'px'; this.renderer.domElement.style.height = h + 'px';
    this.camera.aspect = w / Math.max(1, h); this.camera.updateProjectionMatrix();
  }
  setModel(ac, P) {
    this.group.clear();
    const nt = P.nt, posA = new Float32Array(nt * 9), colA = new Float32Array(nt * 9);
    for (let t = 0; t < nt; t++) for (let k = 0; k < 3; k++) {
      const v = T(P.v[9 * t + 3 * k], P.v[9 * t + 3 * k + 1], P.v[9 * t + 3 * k + 2]);
      posA.set([v.x, v.y, v.z], 9 * t + 3 * k);
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(posA, 3));
    g.setAttribute('color', new THREE.BufferAttribute(colA, 3));
    g.computeVertexNormals();
    this.geom = g; this.P = P;
    this.mesh = new THREE.Mesh(g, new THREE.MeshStandardMaterial({ vertexColors: true, flatShading: true, metalness: 0.35, roughness: 0.55, side: THREE.DoubleSide }));
    this.group.add(this.mesh);
    // תא טייס (קישוט בלבד)
    const p = ac.p, cpx = p.L / 2 - 0.22 * p.L;
    const can = new THREE.Mesh(new THREE.SphereGeometry(1, 20, 12, 0, Math.PI * 2, 0, Math.PI / 2),
      new THREE.MeshStandardMaterial({ color: p.canopy === 'coated' ? 0xc79a3a : 0x6fb6d8, metalness: 0.8, roughness: 0.15, transparent: true, opacity: 0.85 }));
    can.scale.set(Math.min(2.2, p.L * 0.11), Math.min(0.75, p.fus.HU * 0.6), Math.min(0.55, p.fus.W * 0.45));
    can.position.copy(T(cpx, 0, p.fus.HU * 0.85));
    if (p.tail.type !== 'none' || p.wing.span < 30) this.group.add(can);
    // קצוות מפזרים
    const ep = new Float32Array(P.edges.length * 6);
    P.edges.forEach((e, i) => { const a = T(...e.a), b = T(...e.b); ep.set([a.x, a.y, a.z, b.x, b.y, b.z], 6 * i); });
    const eg = new THREE.BufferGeometry(); eg.setAttribute('position', new THREE.BufferAttribute(ep, 3));
    this.edgesObj = new THREE.LineSegments(eg, new THREE.LineBasicMaterial({ color: 0x7cf0c5, transparent: true, opacity: 0.75 }));
    this.edgesObj.visible = this.mode.edges;
    this.group.add(this.edgesObj);
    const span = Math.max(p.L, p.wing.span);
    this.scale = span;
    this.grid.scale.setScalar(Math.max(1, span / 40));
    this.grid.position.y = -Math.max(3, p.fus.HL + 2.5);
    this.colorParts();
    if (this.bubbleObj) { this.group.add(this.bubbleObj); }
    if (!this._framed || Math.abs(this._framed - span) > 8) { this.frame(span); this._framed = span; }
  }
  frame(span) {
    const d = span * 1.55 + 8;
    this.camera.position.set(-d * 0.62, d * 0.42, d * 0.68);
    this.controls.target.set(0, 0, 0);
  }
  colorParts() {
    if (!this.geom) return;
    const col = this.geom.attributes.color.array, c = new THREE.Color();
    for (let t = 0; t < this.P.nt; t++) {
      c.setHex(PART_COLORS[this.P.part[t]] || 0x888888);
      for (let k = 0; k < 3; k++) col.set([c.r, c.g, c.b], 9 * t + 3 * k);
    }
    this.geom.attributes.color.needsUpdate = true;
  }
  colorHeat(vals) {
    if (!this.geom || !vals || vals.length !== this.P.nt) return;
    const col = this.geom.attributes.color.array, c = new THREE.Color();
    let mx = 0; for (const v of vals) mx = Math.max(mx, v);
    const top = 10 * Math.log10(mx + 1e-20);
    for (let t = 0; t < this.P.nt; t++) {
      const db = 10 * Math.log10(vals[t] + 1e-20);
      const u = Math.max(0, Math.min(1, (db - (top - 40)) / 40));
      if (vals[t] <= 0) c.setRGB(0.12, 0.14, 0.2);
      else c.setHSL(0.66 - 0.66 * u, 0.9, 0.18 + 0.42 * u);
      for (let k = 0; k < 3; k++) col.set([c.r, c.g, c.b], 9 * t + 3 * k);
    }
    this.geom.attributes.color.needsUpdate = true;
  }
  setMode(k, v) {
    this.mode[k] = v;
    if (k === 'edges' && this.edgesObj) this.edgesObj.visible = v;
    if (k === 'beam') this.beam.visible = v;
    if (k === 'bubble' && this.bubbleObj) this.bubbleObj.visible = v;
  }
  // r: כיוון מהמטוס אל המכ"ם (בצירים שלנו)
  setRadar(r) {
    this.r = r;
    const dir = T(...r).normalize();
    const q = new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir);
    this.beam.quaternion.copy(q);
  }
  _buildBeam() {
    const len = 1;
    const cone = new THREE.Mesh(new THREE.ConeGeometry(0.16, 1, 32, 1, true),
      new THREE.MeshBasicMaterial({ color: 0xffd34d, transparent: true, opacity: 0.07, side: THREE.DoubleSide, depthWrite: false }));
    cone.position.y = 0.5 * len; cone.rotation.x = 0; this.beamCone = cone; this.beam.add(cone);
    const lg = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(0, 0, 0), new THREE.Vector3(0, 1, 0)]);
    this.beamLine = new THREE.Line(lg, new THREE.LineBasicMaterial({ color: 0xffd34d, transparent: true, opacity: 0.7 }));
    this.beam.add(this.beamLine);
    const dish = new THREE.Mesh(new THREE.SphereGeometry(0.6, 16, 8), new THREE.MeshBasicMaterial({ color: 0xffd34d }));
    dish.position.y = 1; this.beamDish = dish; this.beam.add(dish);
    for (let i = 0; i < 5; i++) {
      const ring = new THREE.Mesh(new THREE.TorusGeometry(1, 0.035, 6, 48), new THREE.MeshBasicMaterial({ color: 0xffd34d, transparent: true, opacity: 0.6 }));
      ring.rotation.x = Math.PI / 2; this.beam.add(ring); this.pulses.push(ring);
    }
  }
  setBubble(vals, na, ne) {
    if (this.bubbleObj) { this.group.remove(this.bubbleObj); this.bubbleObj.geometry.dispose(); this.bubbleObj = null; }
    if (!vals) return;
    let mx = -1e9; for (const v of vals) mx = Math.max(mx, 10 * Math.log10(v + 1e-20));
    const lo = mx - 45, R = this.scale * 0.95;
    const pos = [], col = [], idx = [], c = new THREE.Color();
    for (let j = 0; j < ne; j++) for (let i = 0; i <= na; i++) {
      const ii = i % na, v = vals[j * na + ii];
      const db = 10 * Math.log10(v + 1e-20), u = Math.max(0, Math.min(1, (db - lo) / (mx - lo)));
      const el = (-90 + 180 * j / (ne - 1)) * Math.PI / 180, az = (i * 360 / na) * Math.PI / 180;
      const rr = R * (0.04 + 0.96 * u * u);
      const p = T(rr * Math.cos(el) * Math.cos(az), rr * Math.cos(el) * Math.sin(az), rr * Math.sin(el));
      pos.push(p.x, p.y, p.z);
      c.setHSL(0.62 - 0.62 * u, 0.95, 0.35 + 0.25 * u); col.push(c.r, c.g, c.b);
    }
    for (let j = 0; j + 1 < ne; j++) for (let i = 0; i < na; i++) {
      const a = j * (na + 1) + i, b = a + 1, cc = a + na + 1, d = cc + 1;
      idx.push(a, b, d, a, d, cc);
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
    g.setAttribute('color', new THREE.Float32BufferAttribute(col, 3));
    g.setIndex(idx); g.computeVertexNormals();
    this.bubbleObj = new THREE.Mesh(g, new THREE.MeshBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.32, side: THREE.DoubleSide, depthWrite: false, wireframe: false }));
    const wire = new THREE.LineSegments(new THREE.WireframeGeometry(g), new THREE.LineBasicMaterial({ color: 0x4fd1ff, transparent: true, opacity: 0.08 }));
    this.bubbleObj.add(wire);
    this.bubbleObj.visible = this.mode.bubble;
    this.group.add(this.bubbleObj);
  }
  tick() {
    const t = (performance.now() - this.t0) / 1000;
    const L = (this.scale || 20) * 1.6;
    this.beamCone.scale.set(L * 0.9, L, L * 0.9); this.beamCone.position.y = L / 2;
    this.beamLine.scale.set(1, L, 1); this.beamDish.position.y = L;
    this.pulses.forEach((r, i) => {
      const u = ((t * 0.45 + i / this.pulses.length) % 1);
      const y = L * (1 - u); r.position.y = y;
      const rad = 0.144 * L * u + 0.3; r.scale.setScalar(rad);
      r.material.opacity = 0.55 * Math.sin(Math.PI * u);
    });
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }
}
