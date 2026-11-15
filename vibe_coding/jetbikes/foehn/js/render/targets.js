// What the gun needs on screen: the tethered balloons of a shooting range, and the rounds in
// flight as bright streaks.
import * as THREE from '../../lib/three.module.min.js';
import { surfaceAt } from '../sim/terrain.js';

const X = new THREE.Vector3(1, 0, 0), tmp = new THREE.Vector3();

export class Targets {
  constructor(scene) {
    this.group = new THREE.Group(); scene.add(this.group);
    this.items = [];
    this.skin = new THREE.MeshStandardMaterial({ color: 0xe23a26, roughness: 0.35, metalness: 0, emissive: 0x4a0d05 });
    this.skinNext = new THREE.MeshStandardMaterial({ color: 0xffb21e, roughness: 0.35, metalness: 0, emissive: 0x5a3200 });
    this.rope = new THREE.MeshBasicMaterial({ color: 0x2a2a2a });
    this.ball = new THREE.SphereGeometry(1, 28, 20); this.knot = new THREE.ConeGeometry(0.18, 0.35, 10); this.line = new THREE.CylinderGeometry(0.035, 0.035, 1, 5);
    // rounds in flight
    this.tracers = [];
    const tg = new THREE.BoxGeometry(1, 0.16, 0.16), tm = new THREE.MeshBasicMaterial({ color: new THREE.Color(9, 5.5, 1.6), toneMapped: false, transparent: true, blending: THREE.AdditiveBlending, depthWrite: false });
    for (let i = 0; i < 8; i++) { const m = new THREE.Mesh(tg, tm); m.visible = false; m.frustumCulled = false; scene.add(m); this.tracers.push(m); }
  }
  setEvent(game) {
    for (const it of this.items) this.group.remove(it.obj);
    this.items = [];
    for (const c of game.cps0) {
      if (c.type !== 'target') continue;
      const obj = new THREE.Group(), R = c.r * 0.92, agl = c.p[1] - surfaceAt(c.p[0], c.p[2]);
      const ball = new THREE.Mesh(this.ball, this.skin); ball.scale.set(R, R * 1.12, R); ball.castShadow = true;
      const knot = new THREE.Mesh(this.knot, this.skin); knot.position.y = -R * 1.12 - 0.12;
      const rope = new THREE.Mesh(this.line, this.rope); rope.scale.y = Math.max(1, agl - R); rope.position.y = -R * 1.12 - Math.max(1, agl - R) / 2;
      obj.add(ball, knot, rope); obj.position.set(c.p[0], c.p[1], c.p[2]);
      this.group.add(obj); this.items.push({ obj, ball, knot, c });
    }
  }
  // alive(k): is balloon k still up?  next: the one the arrow points to.  shots: [{ p, v }]
  update(time, alive, next, shots) {
    for (const it of this.items) {
      it.obj.visible = alive(it.c.k);
      if (!it.obj.visible) continue;
      const isNext = it.c === next, m = isNext ? this.skinNext : this.skin;
      it.ball.material = m; it.knot.material = m;
      it.obj.rotation.z = 0.035 * Math.sin(time * 0.9 + it.c.k * 1.7); it.obj.rotation.x = 0.03 * Math.sin(time * 0.7 + it.c.k);   // the ball stays where the physics has it; it only nods
    }
    this.tracers.forEach((m, i) => {
      const s = shots[i]; m.visible = !!s; if (!s) return;
      const sp = Math.hypot(s.v[0], s.v[1], s.v[2]) || 1, L = Math.min(14, sp * 0.045);
      tmp.set(s.v[0] / sp, s.v[1] / sp, s.v[2] / sp);
      m.quaternion.setFromUnitVectors(X, tmp); m.scale.set(L, 1, 1);
      m.position.set(s.p[0] - tmp.x * L / 2, s.p[1] - tmp.y * L / 2, s.p[2] - tmp.z * L / 2);
    });
  }
}
