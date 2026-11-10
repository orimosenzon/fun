// Buildings, landing pads, and the check-point rings of the current event.
import * as THREE from '../../lib/three.module.min.js';
import { T, groundAt } from '../sim/terrain.js';
import { S } from '../sim/scenery.js';

function canvasTex(w, h, draw, repeat) {
  const c = document.createElement('canvas'); c.width = w; c.height = h;
  draw(c.getContext('2d'), w, h);
  const t = new THREE.CanvasTexture(c); t.colorSpace = THREE.SRGBColorSpace; t.anisotropy = 8;
  if (repeat) { t.wrapS = t.wrapT = THREE.RepeatWrapping; }
  return t;
}
function wallTex(kind) {
  return canvasTex(256, 256, (g, w, h) => {
    const wood = kind !== 'church';
    g.fillStyle = wood ? '#e9e2d3' : '#efece4'; g.fillRect(0, 0, w, h);
    if (wood) {
      g.fillStyle = '#5a3a22'; g.fillRect(0, 0, w, h * 0.56);
      for (let i = 0; i < 26; i++) { g.fillStyle = `rgba(${20 + (i * 37) % 40},${10 + (i * 17) % 20},0,0.25)`; g.fillRect(0, i * h * 0.56 / 26, w, 2); }
      g.fillStyle = '#3d2715'; g.fillRect(0, h * 0.53, w, 6);
    }
    // windows with shutters and flower boxes
    for (const [x, y] of wood ? [[40, 60], [150, 60], [40, 170], [150, 170]] : [[60, 50], [170, 50]]) {
      const ww = wood ? 44 : 30, wh = wood ? 46 : 110;
      g.fillStyle = '#1b2530'; g.fillRect(x, y, ww, wh);
      g.strokeStyle = '#f4f0e6'; g.lineWidth = 3; g.strokeRect(x, y, ww, wh); g.beginPath(); g.moveTo(x + ww / 2, y); g.lineTo(x + ww / 2, y + wh); g.stroke();
      if (wood) { g.fillStyle = '#2f5d3a'; g.fillRect(x - 12, y, 10, wh); g.fillRect(x + ww + 2, y, 10, wh); g.fillStyle = '#b3262b'; g.fillRect(x - 2, y + wh + 2, ww + 4, 6); }
    }
  });
}
function litTex() {       // emissive map: only the window panes
  return canvasTex(256, 256, (g, w, h) => {
    g.fillStyle = '#000'; g.fillRect(0, 0, w, h);
    g.fillStyle = '#ffb45a';
    for (const [x, y] of [[40, 60], [150, 60], [40, 170], [150, 170]]) g.fillRect(x + 3, y + 3, 38, 40);
  });
}
function padTex(label) {
  return canvasTex(512, 512, (g, w) => {
    const c = w / 2;
    g.clearRect(0, 0, w, w);
    g.fillStyle = '#3a3d40'; g.beginPath(); g.arc(c, c, c - 4, 0, 7); g.fill();
    for (let i = 0; i < 900; i++) { g.fillStyle = `rgba(${i % 2 ? 255 : 0},${i % 2 ? 255 : 0},${i % 2 ? 255 : 0},0.035)`; g.fillRect((i * 97) % w, (i * 53) % w, 5, 5); }
    g.strokeStyle = '#f2c230'; g.lineWidth = 16; g.beginPath(); g.arc(c, c, c - 26, 0, 7); g.stroke();
    g.strokeStyle = '#e8e8e8'; g.lineWidth = 8; g.beginPath(); g.arc(c, c, c * 0.52, 0, 7); g.stroke();
    g.fillStyle = '#f0f0f0'; g.font = 'bold 200px sans-serif'; g.textAlign = 'center'; g.textBaseline = 'middle'; g.fillText(label, c, c + 12);
  });
}

function gable(w, d, h, over = 0.7) {       // roof prism, ridge along z
  const x = w / 2 + over, z = d / 2 + over;
  const v = [-x, 0, -z, x, 0, -z, 0, h, -z, -x, 0, z, x, 0, z, 0, h, z];
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(v, 3));
  g.setIndex([0, 2, 1, 3, 4, 5, 0, 3, 5, 0, 5, 2, 1, 2, 5, 1, 5, 4, 0, 1, 4, 0, 4, 3]);
  const ng = g.toNonIndexed(); ng.computeVertexNormals();
  return ng;
}

export function buildProps(scene) {
  const group = new THREE.Group();
  const wood = new THREE.MeshStandardMaterial({ map: wallTex('chalet'), emissiveMap: litTex(), emissive: 0xffffff, emissiveIntensity: 0, roughness: 0.9 });
  const plaster = new THREE.MeshStandardMaterial({ map: wallTex('church'), roughness: 0.85 });
  const roofs = [0x4a2f22, 0x3a3432, 0x5a2a20].map((c) => new THREE.MeshStandardMaterial({ color: c, roughness: 0.8 }));
  const stone = new THREE.MeshStandardMaterial({ color: 0x8b8880, roughness: 0.95 });
  const metal = new THREE.MeshStandardMaterial({ color: 0x9aa3ab, roughness: 0.45, metalness: 0.7 });
  const glass = new THREE.MeshStandardMaterial({ color: 0x223344, roughness: 0.1, metalness: 0.6, emissive: 0x88ccff, emissiveIntensity: 0.3 });
  const mk = (geo, mat, x, y, z, parent) => { const m = new THREE.Mesh(geo, mat); m.position.set(x, y, z); m.castShadow = true; m.receiveShadow = true; parent.add(m); return m; };
  for (const p of S.props) {
    const b = new THREE.Group(); b.position.set(p.x, p.y - 0.4, p.z); b.rotation.y = p.yaw;
    if (p.type === 'chalet' || p.type === 'hut' || p.type === 'barn' || p.type === 'boathouse') {
      mk(new THREE.BoxGeometry(p.w, p.h, p.d), p.type === 'barn' || p.type === 'boathouse' ? new THREE.MeshStandardMaterial({ color: 0x4d3120, roughness: 0.95 }) : wood, 0, p.h / 2, 0, b);
      mk(new THREE.BoxGeometry(p.w + 0.5, 1.0, p.d + 0.5), stone, 0, 0.3, 0, b);
      mk(gable(p.w, p.d, p.h * 0.5, 0.9), roofs[(p.seed || 0) % 3], 0, p.h, 0, b);
      if (p.type !== 'barn') mk(new THREE.BoxGeometry(0.7, 1.6, 0.7), stone, p.w * 0.25, p.h * 1.42, p.d * 0.15, b);
    } else if (p.type === 'church') {
      mk(new THREE.BoxGeometry(p.w, p.h, p.d), plaster, 0, p.h / 2, 0, b);
      mk(gable(p.w, p.d, p.h * 0.55, 0.4), roofs[1], 0, p.h, 0, b);
      mk(new THREE.BoxGeometry(5, p.steeple * 0.7, 5), plaster, 0, p.steeple * 0.35, -p.d / 2 + 2.5, b);
      const sp = mk(new THREE.ConeGeometry(4.1, p.steeple * 0.42, 4), roofs[0], 0, p.steeple * 0.7 + p.steeple * 0.21, -p.d / 2 + 2.5, b); sp.rotation.y = Math.PI / 4;
    } else if (p.type === 'hangar') {
      const hg = new THREE.CylinderGeometry(p.d / 2, p.d / 2, p.w, 20, 1, false, 0, Math.PI); hg.rotateZ(Math.PI / 2); hg.rotateY(Math.PI / 2);
      const h = mk(hg, metal, 0, 0, 0, b); h.rotation.y = Math.PI / 2; h.scale.y = p.h / (p.d / 2);
      mk(new THREE.BoxGeometry(p.w, 0.6, p.d), stone, 0, 0.3, 0, b);
    } else if (p.type === 'tower') {
      mk(new THREE.BoxGeometry(3.4, p.h, 3.4), plaster, 0, p.h / 2, 0, b);
      mk(new THREE.CylinderGeometry(4.2, 3.4, 3.2, 8), glass, 0, p.h + 1.6, 0, b);
      mk(new THREE.CylinderGeometry(4.6, 4.6, 0.4, 8), metal, 0, p.h + 3.4, 0, b);
    }
    group.add(b);
  }
  // pads
  const padGeo = new THREE.CircleGeometry(1, 48); padGeo.rotateX(-Math.PI / 2);
  for (const [name, p] of Object.entries(T.meta.pads)) {
    const r = name === 'village' ? 16 : Math.min(p[3] * 0.62, 18);
    const m = new THREE.Mesh(padGeo, new THREE.MeshStandardMaterial({ map: padTex(name === 'base' ? 'F' : 'H'), transparent: true, roughness: 0.9, polygonOffset: true, polygonOffsetFactor: -4, polygonOffsetUnits: -4 }));
    m.scale.set(r, 1, r); m.position.set(p[0], p[1] + 0.06, p[2]); m.receiveShadow = true; m.renderOrder = 1;
    group.add(m);
  }
  // wooden pier
  const pier = T.meta.pads.pier;
  mk(new THREE.BoxGeometry(34, 0.5, 34), new THREE.MeshStandardMaterial({ color: 0x6b4a2e, roughness: 0.9 }), pier[0], pier[1] - 0.3, pier[2], group);
  scene.add(group);
  return { group, wood, setNight(k) { wood.emissiveIntensity = k * 2.2; } };
}

// ---------- check-points ----------
export class Gates {
  constructor(scene) {
    this.group = new THREE.Group(); scene.add(this.group);
    this.items = [];
    this.matNext = new THREE.MeshBasicMaterial({ color: new THREE.Color(3.0, 1.3, 0.2), toneMapped: false });
    this.matLater = new THREE.MeshBasicMaterial({ color: new THREE.Color(0.25, 1.1, 1.6), transparent: true, opacity: 0.55, toneMapped: false });
    this.matFinish = new THREE.MeshBasicMaterial({ color: new THREE.Color(2.6, 2.6, 2.6), toneMapped: false });
    this.matBeam = new THREE.MeshBasicMaterial({ color: new THREE.Color(1.6, 0.8, 0.15), transparent: true, opacity: 0.22, blending: THREE.AdditiveBlending, depthWrite: false, side: THREE.DoubleSide, toneMapped: false });
    this.matDisc = new THREE.MeshBasicMaterial({ color: new THREE.Color(1.2, 0.5, 0.08), transparent: true, opacity: 0.10, blending: THREE.AdditiveBlending, depthWrite: false, side: THREE.DoubleSide, toneMapped: false });
    this.beam = new THREE.Mesh(new THREE.CylinderGeometry(1, 1, 1, 12, 1, true), this.matBeam); this.beam.visible = false; scene.add(this.beam);
  }
  setEvent(build) {
    for (const it of this.items) { this.group.remove(it.obj); it.obj.traverse((o) => o.geometry && o.geometry.dispose()); }
    this.items = [];
    const up = new THREE.Vector3(0, 0, 1);
    build.real.forEach((c, i) => {
      const obj = new THREE.Group();
      obj.position.set(c.p[0], c.p[1], c.p[2]);
      let ring, disc = null;
      if (c.type === 'ring') {
        ring = new THREE.Mesh(new THREE.TorusGeometry(c.r, 0.3, 10, 64), this.matLater);
        obj.quaternion.setFromUnitVectors(up, new THREE.Vector3(c.n[0], c.n[1], c.n[2]));
        disc = new THREE.Mesh(new THREE.CircleGeometry(c.r - 0.4, 40), this.matDisc); obj.add(disc);
        // four inward ticks
        for (let k = 0; k < 4; k++) { const t = new THREE.Mesh(new THREE.BoxGeometry(0.35, 2.2, 0.35), this.matLater); const a = k * Math.PI / 2 + Math.PI / 4; t.position.set(Math.cos(a) * (c.r - 1.2), Math.sin(a) * (c.r - 1.2), 0); t.rotation.z = a - Math.PI / 2; obj.add(t); t.userData.tick = true; }
      } else {
        ring = new THREE.Mesh(new THREE.TorusGeometry(c.r, 0.3, 8, 56), this.matLater); ring.rotation.x = Math.PI / 2; ring.position.y = 0.4;
      }
      obj.add(ring);
      this.group.add(obj);
      this.items.push({ obj, ring, disc, c, i });
    });
  }
  update(game, t) {
    const cur = game.cp;
    this.beam.visible = false;
    for (const it of this.items) {
      const d = it.i - cur;
      it.obj.visible = d >= 0 && d <= 3 && game.state !== 'finished';
      if (!it.obj.visible) continue;
      const next = d === 0;
      const mat = next ? (it.c.finish || it.i === this.items.length - 1 ? this.matFinish : this.matNext) : this.matLater;
      it.obj.traverse((o) => { if (o.isMesh && o !== it.disc) o.material = mat; });
      if (it.disc) it.disc.visible = next;
      const s = next ? 1 + 0.03 * Math.sin(t * 6) : 1;
      it.ring.scale.setScalar(s);
      if (next) {
        // a light beam marks the next check-point from afar
        const h = it.c.type === 'land' ? 260 : 190;
        this.beam.visible = true; this.beam.scale.set(it.c.type === 'land' ? it.c.r : 1.2, h, it.c.type === 'land' ? it.c.r : 1.2);
        this.beam.position.set(it.c.p[0], it.c.p[1] + (it.c.type === 'land' ? h / 2 : it.c.r + h / 2), it.c.p[2]);
      }
    }
    this.matNext.color.setRGB(3.0 + 0.8 * Math.sin(t * 6), 1.3, 0.2);
  }
}
