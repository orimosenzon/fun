// mode_museum.js — לשונית "מוזיאון": כל הכלים זה לצד זה, בקנה מידה אמיתי, עם דמות אדם להשוואה
import * as THREE from 'three';
import { buildModel } from './models.js';
import { SPECS, MUSEUM_ROCKETS, MUSEUM_SATS } from './data.js';
import { $, $$, h } from './util.js';

const HALL = {
  rockets: { ids: MUSEUM_ROCKETS, gap: 16 },
  sats: { ids: MUSEUM_SATS, gap: 10 },
};
// כיוון התצוגה של כל לוויין במוזיאון (המודלים בנויים בכיוון של טיסה)
const SAT_POSE = {
  iss: o => { o.rotation.y = 0.5; },
  tiangong: o => { o.rotation.y = 0.3; },
  geo: o => { o.rotateY(Math.PI / 2 + 0.3); o.rotateZ(Math.PI / 2); },
  starlink: o => { o.rotation.x = Math.PI / 2.4; o.position.y = 8; },
  hubble: o => { o.position.y = 3.5; },
  shenzhou: o => { o.position.y = 0; },
  dragon: o => { o.position.y = 0; },
  apolloCSM: o => { o.position.y = 0; },
  apolloLM: o => { o.position.y = 0; },
  vostok: o => { o.position.y = 0; },
  sputnik: o => { o.position.y = 3; },
};

export const museumMode = {
  enter(world, panel) {
    this.world = world;
    this.hall = this.hall ?? 'rockets';
    if (!this.scene) this.build(world);
    this.scene.environment = world.scene.environment;
    this.saved = { pos: world.camera.position.clone(), tgt: world.controls.target.clone(), min: world.controls.minDistance, near: world.camera.near };
    world.controls.minDistance = 2;
    world.controls.maxDistance = 2000;
    world.camera.near = 0.05; world.camera.updateProjectionMatrix();
    world.setWarps([0], 0);
    world.timeDisplay = () => 'קנה מידה 1:1';
    panel.innerHTML = `
      <h2>מוזיאון</h2>
      <p class="lead">כל הכלים בגודלם האמיתי, זה לצד זה. הדמות הכתומה היא אדם בגובה 1.8 מטר. אפשר לסובב, להתקרב וללחוץ על כל כלי.</p>
      <div class="filters"><button class="chip" data-h="rockets">רקטות</button><button class="chip" data-h="sats">חלליות ולוויינים</button></div>
      <div class="presets" id="items"></div>
      <div id="spec"></div>
      <p class="note">המודלים נבנו מהמידות שפורסמו: גובה, קוטר, מספר המנועים ומיקומם, מבנה השלבים וחלוקת הצבעים. פרטים קטנים (צנרת, אנטנות קטנות, כיתובים) פושטו.</p>`;
    this.panel = panel;
    $('.filters', panel).onclick = e => { const b = e.target.closest('button'); if (b) { this.hall = b.dataset.h; this.showHall(); } };
    $('#items', panel).onclick = e => { const b = e.target.closest('button'); if (b) this.focus(b.dataset.id); };
    this.onClick = e => this.pick(e);
    this.onDown = e => { this.down = [e.clientX, e.clientY]; };
    world.renderer.domElement.addEventListener('pointerdown', this.onDown);
    world.renderer.domElement.addEventListener('pointerup', this.onClick);
    this.showHall();
  },

  build(world) {
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0b1222);
    scene.fog = new THREE.Fog(0x0b1222, 400, 1400);
    scene.add(new THREE.HemisphereLight(0xcfe0ff, 0x30343c, 1.4));
    const key = new THREE.DirectionalLight(0xffffff, 2.4); key.position.set(-120, 220, 160); scene.add(key);
    const rim = new THREE.DirectionalLight(0x88aaff, 0.8); rim.position.set(150, 80, -200); scene.add(rim);
    const floor = new THREE.Mesh(new THREE.CircleGeometry(1500, 64), new THREE.MeshStandardMaterial({ color: 0x1a2133, roughness: 0.95 }));
    floor.rotation.x = -Math.PI / 2; scene.add(floor);
    const grid = new THREE.GridHelper(1000, 100, 0x34405a, 0x222b3e); grid.position.y = 0.02; scene.add(grid);
    this.scene = scene;
    this.halls = {};
    this.items = {};
    for (const [name, cfg] of Object.entries(HALL)) {
      const g = new THREE.Group();
      let x = 0;
      const placed = [];
      for (const id of cfg.ids) {
        const o = buildModel(id);
        const holder = new THREE.Group(); holder.add(o);
        if (name === 'sats') SAT_POSE[id]?.(o);
        const box = new THREE.Box3().setFromObject(holder);
        const size = box.getSize(new THREE.Vector3());
        holder.position.x = x + size.x / 2 - (box.min.x + box.max.x) / 2;
        holder.position.z = -(box.min.z + box.max.z) / 2;
        if (name === 'sats') holder.position.y = -box.min.y + (id === 'iss' || id === 'tiangong' ? 0 : 0);
        x += size.x + cfg.gap;
        holder.userData.id = id;
        g.add(holder);
        placed.push(holder);
        this.items[id] = { holder, size, hall: name };
      }
      // דמות אדם בסוף השורה
      const man = buildModel('human'); man.position.x = x; g.add(man);
      g.position.x = -x / 2;
      g.userData.width = x;
      scene.add(g);
      this.halls[name] = g;
    }
  },

  showHall() {
    const w = this.world;
    $$('.filters .chip', this.panel).forEach(c => c.classList.toggle('on', c.dataset.h === this.hall));
    for (const [n, g] of Object.entries(this.halls)) g.visible = n === this.hall;
    const ids = HALL[this.hall].ids;
    $('#items', this.panel).innerHTML = ids.map(id => `<button class="chip" data-id="${id}">${SPECS[id].name}</button>`).join('');
    this.labelsOff();
    this.labels = ids.map((id, k) => {
      const it = this.items[id];
      const L = w.addLabel(SPECS[id].name, 'dim big');
      L.occlude = false;
      const p = new THREE.Vector3(); it.holder.getWorldPosition(p);
      const top = new THREE.Box3().setFromObject(it.holder).max.y;
      L.pos.set(p.x, top + 3 + (this.hall === 'sats' && k % 2 ? 9 : 0), p.z);
      return L;
    });
    const g = this.halls[this.hall];
    const width = g.userData.width;
    const tall = this.hall === 'rockets' ? 125 : 75;
    // הלוח מכסה את צד ימין של המסך, ולכן מזיזים את מרכז התמונה ימינה
    const shift = window.innerWidth > 760 ? width * (this.hall === 'rockets' ? 0.16 : 0.1) : 0;
    w.controls.target.set(shift, tall * 0.45, 0);
    w.camera.position.set(shift - width * 0.1, tall * 0.6, Math.max(width * (this.hall === 'rockets' ? 1.05 : 1.25), tall * 1.9));
    $('#spec', this.panel).innerHTML = '';
  },

  focus(id) {
    const w = this.world, it = this.items[id];
    if (it.hall !== this.hall) { this.hall = it.hall; this.showHall(); }
    const box = new THREE.Box3().setFromObject(it.holder);
    const c = box.getCenter(new THREE.Vector3()), s = box.getSize(new THREE.Vector3());
    const r = Math.max(s.x, s.y, s.z);
    w.controls.target.copy(c);
    w.camera.position.set(c.x - r * 0.35, c.y + r * 0.25, c.z + r * 1.25 + 3);
    const sp = SPECS[id];
    $('#spec', this.panel).innerHTML = `
      <h3>${sp.name}</h3>
      <p class="small">${sp.org}${sp.first ? ' · ' + sp.first : ''}</p>
      <dl class="facts">${sp.rows.map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join('')}</dl>
      <p>${sp.text}</p>`;
    $$('#items .chip', this.panel).forEach(b => b.classList.toggle('on', b.dataset.id === id));
  },

  pick(e) {
    if (e.button !== 0) return;
    const w = this.world;
    const r = w.renderer.domElement.getBoundingClientRect();
    const ndc = new THREE.Vector2(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
    if (this.down && Math.hypot(e.clientX - this.down[0], e.clientY - this.down[1]) > 5) return;
    const rc = new THREE.Raycaster();
    rc.setFromCamera(ndc, w.camera);
    const hits = rc.intersectObjects(this.halls[this.hall].children, true);
    for (const hit of hits) {
      let o = hit.object;
      while (o && !o.userData.id) o = o.parent;
      if (o) { this.focus(o.userData.id); return; }
    }
  },

  labelsOff() { (this.labels ?? []).forEach(L => this.world.removeLabel(L)); this.labels = []; },

  update(world) { world.camera.updateMatrixWorld(); },

  get camera() { return null; },

  exit(world) {
    this.labelsOff();
    world.renderer.domElement.removeEventListener('pointerup', this.onClick);
    world.renderer.domElement.removeEventListener('pointerdown', this.onDown);
    world.camera.position.copy(this.saved.pos);
    world.controls.target.copy(this.saved.tgt);
    world.controls.minDistance = this.saved.min;
    world.controls.maxDistance = 2e6;
    world.camera.near = this.saved.near; world.camera.updateProjectionMatrix();
    world.timeDisplay = null;
  },
};
