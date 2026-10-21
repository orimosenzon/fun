// מצלמות: מסלול סביב הרובוט שעוקב אחריו (ברירת מחדל), מבט מהעיניים של הרובוט, ומצלמה חופשית.
// מצלמת המסלול (view) ממשיכה לעקוב גם במצב עיני הרובוט, כדי שחלון הפינה יוכל להציג מבט מבחוץ,
// וכדי שבחזרה מהעיניים המשתמש ימצא את הזווית שהשאיר.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

export class CameraRig {
  constructor(camera, dom) {
    this.camera = camera;
    this.view = camera.clone();
    this.mode = 'follow';
    this.orbit = new OrbitControls(this.view, dom);
    this.orbit.enableDamping = true;
    this.orbit.dampingFactor = 0.08;
    this.orbit.minDistance = 0.6; this.orbit.maxDistance = 14;
    this.orbit.maxPolarAngle = Math.PI * 0.495;
    this.orbit.target.set(0, 1.0, 0);
    this.view.position.set(2.2, 1.55, 2.6);
    this.last = new THREE.Vector3();
    this.init = false;
    this.onMode = null;
  }
  setMode(m) {
    if (!['follow', 'eyes', 'free'].includes(m)) return this.mode;
    this.mode = m; this.orbit.enabled = m !== 'eyes';
    this.onMode?.(m);
    return m;
  }
  cycle() {
    const order = ['follow', 'eyes', 'free'];
    return this.setMode(order[(order.indexOf(this.mode) + 1) % order.length]);
  }
  // robotPos: מיקום האגן בקואורדינטות three. eye: {pos, quat} של מצלמת העיניים
  update(robotPos, eye, dt) {
    if (this.mode !== 'free') {
      const target = new THREE.Vector3(robotPos.x, Math.max(0.75, robotPos.y + 0.05), robotPos.z);
      if (!this.init) { this.last.copy(target); this.init = true; }
      const delta = new THREE.Vector3().subVectors(target, this.last);
      // מזיזים גם את המצלמה וגם את המטרה, כך שהמשתמש שומר על הזווית שבחר
      this.view.position.add(delta);
      this.orbit.target.add(delta);
      this.last.copy(target);
    }
    this.orbit.update();
    const fov = this.mode === 'eyes' ? 70 : 45;
    if (this.camera.fov !== fov) { this.camera.fov = fov; this.camera.updateProjectionMatrix(); }
    if (this.mode === 'eyes' && eye) {
      this.camera.position.copy(eye.pos);
      this.camera.quaternion.copy(eye.quat);
    } else {
      this.camera.position.copy(this.view.position);
      this.camera.quaternion.copy(this.view.quaternion);
    }
  }
}
