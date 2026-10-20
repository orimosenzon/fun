// קלט: מקלדת להליכה ולטלאופרציה של הידיים, עכבר לדחיפה ולבחירת יעד, ובקר משחק.
import * as THREE from 'three';
import { rotZYX } from '../ctrl/linalg.js';

const KEYS = {};
export class Input {
  constructor() {
    this.keys = KEYS;
    this.handMode = null;      // null | 'l' | 'r' | 'both'
    this.grip = { l: false, r: false };
    this.pushStrength = 22;    // ניוטון-שנייה (מתקף)
    this.speedMul = 1;
    addEventListener('keydown', (e) => {
      if (e.target.closest('input, select, textarea')) return;
      KEYS[e.code] = true;
      this.onKey?.(e);
    });
    addEventListener('keyup', (e) => { KEYS[e.code] = false; });
    addEventListener('blur', () => { for (const k in KEYS) KEYS[k] = false; });
  }
  bind(app, ctx) {
    this.app = app; this.ctx = ctx;
    const dom = ctx.renderer.domElement;
    ctx.rig.orbit.enablePan = false;
    this.ray = new THREE.Raycaster();
    // ג'ויסטיק למסכי מגע: למעלה/למטה הליכה, ימינה/שמאלה פנייה
    this.joy = [0, 0];
    if (matchMedia('(pointer: coarse)').matches) {
      const joy = document.getElementById('joy'), knob = joy.querySelector('.knob');
      joy.classList.remove('hidden');
      const move = (e) => {
        const r = joy.getBoundingClientRect(), R = r.width / 2;
        let dx = (e.clientX - r.left - R) / R, dy = (e.clientY - r.top - R) / R;
        const L = Math.hypot(dx, dy); if (L > 1) { dx /= L; dy /= L; }
        this.joy = [dx, dy];
        knob.style.transform = `translate(${dx * R * 0.6}px, ${dy * R * 0.6}px)`;
      };
      joy.addEventListener('pointerdown', (e) => { joy.setPointerCapture(e.pointerId); move(e); });
      joy.addEventListener('pointermove', (e) => { if (e.buttons || e.pressure) move(e); });
      const end = () => { this.joy = [0, 0]; knob.style.transform = ''; };
      joy.addEventListener('pointerup', end); joy.addEventListener('pointercancel', end);
    }
    // לחיצה ימנית על הרובוט = דחיפה. לחיצה שמאלית על הרצפה במצב "לך לנקודה" = יעד
    dom.addEventListener('contextmenu', (e) => e.preventDefault());
    let down = null;
    dom.addEventListener('pointerdown', (e) => { down = { x: e.clientX, y: e.clientY, b: e.button, t: performance.now() }; });
    dom.addEventListener('pointerup', (e) => {
      if (!down) return;
      const moved = Math.hypot(e.clientX - down.x, e.clientY - down.y) > 6;
      const b = down.b; down = null;
      if (moved) return;
      const hit = this.pick(e);
      if (!hit) return;
      if (b === 2 || (b === 0 && this.pushArmed)) { if (hit.robot) this.push(hit); this.pushArmed = false; ctx.hud.setPushArmed(false); }
      else if (b === 0 && hit.floor && app.tasks.wantsGoal) app.tasks.setGoal([hit.sim.x, hit.sim.y]);
    });
  }
  // הקרנת קרן מהעכבר: מחזיר נקודה בקואורדינטות הסימולציה, ואם פגענו ברובוט
  pick(e) {
    const { camera, renderer, simRoot } = this.ctx;
    const r = renderer.domElement.getBoundingClientRect();
    const ndc = new THREE.Vector2(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
    this.ray.setFromCamera(ndc, camera);
    const hits = this.ray.intersectObject(this.app.model.root, true);
    const inv = new THREE.Matrix4().copy(simRoot.matrixWorld).invert();
    if (hits.length) {
      const p = hits[0].point.clone().applyMatrix4(inv);
      // לאיזה גוף שייכת הרשת
      let o = hits[0].object, part = null;
      while (o && !part) { part = this.app.model.parts.find(pp => pp.obj === o); o = o.parent; }
      const dirW = this.ray.ray.direction.clone();
      const dir = dirW.transformDirection(inv);
      return { robot: true, sim: p, body: part ? part.id : this.app.robot.wbc.ids.torso, dir };
    }
    // רצפה: מישור y=0 בעולם של three
    const t = -this.ray.ray.origin.y / this.ray.ray.direction.y;
    if (t > 0) {
      const p = this.ray.ray.origin.clone().addScaledVector(this.ray.ray.direction, t).applyMatrix4(inv);
      return { floor: true, sim: p };
    }
    return null;
  }
  push(hit) {
    // דחיפה אופקית בכיוון המבט, 0.1 שנייה
    const d = new THREE.Vector3(hit.dir.x, hit.dir.y, 0).normalize();
    const J = this.pushStrength, T = 0.1, F = [d.x * J / T, d.y * J / T, 0];
    this.app.push = { body: hit.body, F, t: T };
    this.app.pushViz = { p: [hit.sim.x, hit.sim.y, hit.sim.z], F, t: 0.6 };
    this.ctx.hud.toast(`דחיפה: ${J} ניוטון·שנייה (${Math.round(J / T)} ניוטון למשך 0.1 ש')`, 1600);
  }
  pushFromCamera() {
    const { camera } = this.ctx;
    const r = this.app.robot, tid = r.wbc.ids.torso, d = this.app.world.data;
    const p = { x: d.xpos[3 * tid], y: d.xpos[3 * tid + 1], z: d.xpos[3 * tid + 2] + 0.15 };
    const dir3 = new THREE.Vector3(); camera.getWorldDirection(dir3);
    this.push({ sim: p, body: tid, dir: { x: dir3.x, y: -dir3.z, z: 0 } });
  }

  // בכל פריים: הליכה וטלאופרציה
  apply(app, dt) {
    const K = this.keys, r = app.robot;
    if (!r) return;
    if (app.tasks.active && !app.tasks.active.manualWalk) return;
    let vx = 0, vy = 0, wz = 0;
    if (K.KeyW || K.ArrowUp) vx += 1;
    if (K.KeyS || K.ArrowDown) vx -= 0.6;
    if (K.KeyA || K.ArrowLeft) wz += 1;
    if (K.KeyD || K.ArrowRight) wz -= 1;
    if (K.KeyQ) vy += 1;
    if (K.KeyE) vy -= 1;
    const fast = K.ShiftLeft || K.ShiftRight;
    if (this.joy && (this.joy[0] || this.joy[1])) { vx += -this.joy[1] * 1.3; wz += -this.joy[0]; }
    // בקר משחק
    const gp = navigator.getGamepads?.()[0];
    if (gp) {
      const dz = (v) => Math.abs(v) < 0.15 ? 0 : v;
      vx += -dz(gp.axes[1]); vy += -dz(gp.axes[0]) * 0.8; wz += -dz(gp.axes[2]);
    }
    const sp = fast ? 0.62 : 0.4;
    r.loco.setCommand(vx * sp, vy * 0.18, wz * 0.6);
    // טלאופרציה של הידיים
    if (this.handMode) {
      const sides = this.handMode === 'both' ? ['l', 'r'] : [this.handMode];
      const h = r.loco.heading, c = Math.cos(h), s = Math.sin(h), v = 0.35 * dt;
      let dx = 0, dy = 0, dz = 0;
      if (K.KeyI) dx += v; if (K.KeyK) dx -= v;
      if (K.KeyJ) dy += v; if (K.KeyL) dy -= v;
      if (K.KeyU) dz += v; if (K.KeyO) dz -= v;
      for (const sd of sides) {
        const T = r.handTargets[sd];
        if (!T) continue;
        const mirror = this.handMode === 'both' && sd === 'r' ? -1 : 1;
        T.pos[0] += c * dx - s * dy * mirror; T.pos[1] += s * dx + c * dy * mirror; T.pos[2] += dz;
        this.clampReach(r, sd, T.pos);
      }
    }
  }
  clampReach(r, side, p) {
    const sh = r.wbc.w.data.xpos, id = r.wbc.ids.torso;
    const h = r.loco.heading, sg = side === 'l' ? 1 : -1;
    const s = [sh[3 * id] - Math.sin(h) * sg * 0.2, sh[3 * id + 1] + Math.cos(h) * sg * 0.2, sh[3 * id + 2] + 0.38];
    const d = [p[0] - s[0], p[1] - s[1], p[2] - s[2]], L = Math.hypot(...d), max = 0.62;
    if (L > max) for (let i = 0; i < 3; i++) p[i] = s[i] + d[i] * max / L;
    p[2] = Math.max(0.15, p[2]);
  }
  // הפעלה/כיבוי של טלאופרציה: היעד מתחיל במקום הנוכחי של היד
  setHandMode(mode) {
    const r = this.app.robot;
    this.handMode = mode;
    for (const s of ['l', 'r']) {
      const on = mode === s || mode === 'both';
      if (on && !r.handTargets[s]) {
        const p = r.wbc.sitePos(r.wbc.ids[s + '_grip']);
        r.handTargets[s] = { pos: [p[0] + Math.cos(r.loco.heading) * 0.15, p[1] + Math.sin(r.loco.heading) * 0.15, p[2] + 0.12], w: 80 };
      }
      if (!on && r.handTargets[s] && !this.app.tasks.active) r.handTargets[s] = null;
    }
  }
  toggleGrip(side) {
    const sides = side === 'both' ? ['l', 'r'] : [side];
    for (const s of sides) {
      this.grip[s] = !this.grip[s];
      this.app.robot.hands.set(s, this.grip[s] ? (this.gripShape || 'power') : 'open', this.grip[s] ? 1 : 0, 2.5);
    }
  }
}
