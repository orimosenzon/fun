// מצלמות: עוקבת מאחור, מבט מתוך הרכב (עם תנועת ראש לפי כוחות ה-G), ומצלמה חופשית.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { V3 } from './math.js';
import { makeHit } from './collide.js';

export class CameraRig {
  constructor(camera, dom) {
    this.camera = camera;
    this.mode = 'chase';
    this.qCam = new THREE.Quaternion();
    this.pos = new THREE.Vector3();
    this.head = new THREE.Vector3();
    this.headVel = new THREE.Vector3();
    this.shake = 0;
    this.orbit = new OrbitControls(camera, dom);
    this.orbit.enabled = false;
    this.orbit.enableDamping = true;
    this.orbit.maxDistance = 120; this.orbit.minDistance = 3;
    this.init = false;
    this.hit = makeHit();
    this.dist = 6.4;
    this.lookYaw = 0;
  }
  setMode(m) {
    this.mode = m;
    this.orbit.enabled = m === 'orbit';
    if (m === 'orbit') { this.orbit.target.copy(this.lastCarPos || new THREE.Vector3()); }
  }
  cycle() {
    const order = ['chase', 'cockpit', 'orbit'];
    this.setMode(order[(order.indexOf(this.mode) + 1) % order.length]);
    return this.mode;
  }
  snap() { this.init = false; }

  update(car, dt, world, model) {
    const cam = this.camera;
    const cq = new THREE.Quaternion(car.q.x, car.q.y, car.q.z, car.q.w);
    const cpos = new THREE.Vector3(car.pos.x, car.pos.y, car.pos.z);
    this.lastCarPos = cpos;
    const speed = car.speed;

    if (this.mode === 'orbit') {
      // המטרה עוקבת אחרי הרכב
      const delta = cpos.clone().sub(this.orbit.target);
      this.orbit.target.add(delta); cam.position.add(delta);
      this.orbit.update();
      cam.fov += (55 - cam.fov) * 0.1; cam.updateProjectionMatrix();
      model.cockpit.visible = true;
      return;
    }

    if (this.mode === 'cockpit') {
      // הראש כמסה על קפיץ: תאוצה קדימה מושכת את הראש אחורה, וכו'. ~2 ס"מ ל-g
      const g = car.gSmooth; // במערכת הגוף, ביחידות g
      const target = new THREE.Vector3(-g.x * 0.018, -(g.y - 1) * 0.012, -g.z * 0.022);
      const k = 140, c = 18;
      const acc = target.clone().sub(this.head).multiplyScalar(k).addScaledVector(this.headVel, -c);
      this.headVel.addScaledVector(acc, dt); this.head.addScaledVector(this.headVel, dt);
      this.head.clampLength(0, 0.08);
      // רעד מהכביש ומהדחף
      const thrust = car.thrustTotal / 80000;
      const sh = Math.min(1, speed / 60) * 0.0025 + thrust * 0.006 + (car.fan.n * 0.0008);
      const shake = new THREE.Vector3((Math.random() - 0.5) * sh, (Math.random() - 0.5) * sh, 0);
      const eye = new THREE.Vector3(model.driverX, 0.47, -0.62).add(this.head).add(shake);
      cam.position.copy(eye.applyQuaternion(cq).add(cpos));
      // מבט קדימה, עם נטייה קלה של הראש בפניות
      const look = new THREE.Quaternion().setFromEuler(new THREE.Euler(-0.08 + g.z * 0.012, this.lookYaw, g.x * 0.02, 'YXZ'));
      cam.quaternion.copy(cq).multiply(new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), Math.PI)).multiply(look);
      const fov = 74 + Math.min(16, speed * 0.1);
      cam.fov += (fov - cam.fov) * Math.min(1, dt * 3); cam.near = 0.05; cam.updateProjectionMatrix();
      model.cockpit.visible = true;
      this.init = false;
      return;
    }

    // ---------- מצלמה עוקבת ----------
    cam.near = 0.1;
    // בטיסה: עוקבים רק אחרי כיוון התנועה (לא מסתובבים עם כל סלטה). על משטח: מסתובבים עם הרכב.
    const onSurface = car.contacts > 0 || car.airTime < 0.25;
    let targetQ;
    if (onSurface) targetQ = cq.clone();
    else {
      const v = new THREE.Vector3(car.vel.x, 0, car.vel.z);
      let yaw;
      if (v.lengthSq() > 4) yaw = Math.atan2(v.x, v.z);
      else { const f = new THREE.Vector3(0, 0, 1).applyQuaternion(cq); yaw = Math.atan2(f.x, f.z); }
      targetQ = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), yaw);
    }
    if (!this.init) { this.qCam.copy(targetQ); this.init = true; this.pos.set(0, 0, 0); }
    const rate = onSurface ? 4.5 : 3.2;
    this.qCam.slerp(targetQ, 1 - Math.exp(-rate * dt));
    const back = this.dist + Math.min(4, speed * 0.04);
    const offset = new THREE.Vector3(0, 1.75, -back).applyQuaternion(this.qCam);
    const target = cpos.clone().add(new THREE.Vector3(0, 0.7, 1.5).applyQuaternion(this.qCam));
    let desired = cpos.clone().add(offset);
    // לא להיכנס לתוך קירות: קרן מהרכב אל המצלמה
    const dir = desired.clone().sub(target); const L = dir.length(); dir.normalize();
    const o = new V3(target.x, target.y, target.z), d = new V3(dir.x, dir.y, dir.z);
    if (world.raycastAny(o, d, L, this.hit)) desired = target.clone().addScaledVector(dir, Math.max(1.2, this.hit.t - 0.4));
    // לא מתחת לקרקע
    desired.y = Math.max(desired.y, 0.35);
    cam.position.copy(desired);
    const up = new THREE.Vector3(0, 1, 0).applyQuaternion(this.qCam);
    cam.up.copy(up);
    cam.lookAt(target);
    const fov = 60 + Math.min(18, speed * 0.12);
    cam.fov += (fov - cam.fov) * Math.min(1, dt * 3); cam.updateProjectionMatrix();
    model.cockpit.visible = true;
  }
}
