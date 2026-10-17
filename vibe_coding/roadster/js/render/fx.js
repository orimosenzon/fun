// אפקטים: סילוני גז קר עם יהלומי הלם (Mach diamonds), אדי עיבוי, אבק, עשן צמיגים וניצוצות.
import * as THREE from 'three';
import { softDot } from './textures.js';

// ---------- סילון ----------
// סילון אוויר קר הוא כמעט שקוף. מה שרואים הוא: יהלומי הלם (גלי דחיסה שמתחממים
// ומאירים מעט, ובעיקר משנים את מקדם השבירה) ואדי מים שמתעבים כי הגז מתקרר ל-−150°C.
const plumeVS = `
varying vec2 vUv;
varying float vFacing;
void main() {
  vUv = uv;
  vec4 mv = modelViewMatrix * vec4(position, 1.0);
  vec3 n = normalize(normalMatrix * normal);
  vFacing = abs(dot(n, normalize(-mv.xyz)));
  gl_Position = projectionMatrix * mv;
}`;
const plumeFS = `
uniform float uThrust;
uniform float uCell;
uniform float uLen;
uniform float uTime;
varying vec2 vUv;
varying float vFacing;
void main() {
  float along = vUv.y;            // 0 בנחיר, 1 בקצה
  float d = along * uLen;
  float core = pow(vFacing, 1.6);  // בהיר במרכז, שקוף בשוליים
  // תאי הלם: עוצמה מחזורית לאורך הסילון, דועכת עם המרחק
  float cells = pow(0.5 + 0.5 * cos(6.2831 * d / uCell), 6.0);
  float fade = exp(-along * 2.6);
  float flicker = 0.85 + 0.15 * sin(uTime * 90.0 + d * 13.0);
  float a = uThrust * fade * core * (0.18 + 0.82 * cells * smoothstep(0.0, 0.04, along)) * flicker;
  vec3 col = mix(vec3(0.75, 0.88, 1.0), vec3(1.0), cells * 0.6);
  gl_FragColor = vec4(col, a * 0.9);
}`;

export class Plumes {
  constructor(scene, model, car) {
    this.items = [];
    for (let i = 0; i < model.nozzles.length; i++) {
      const nz = model.nozzles[i];
      const geo = new THREE.CylinderGeometry(0.06, 0.02, 1, 16, 24, true);
      geo.translate(0, -0.5, 0); // מתחיל בנחיר ומתארך כלפי −y
      geo.rotateX(Math.PI);      // עכשיו מתארך ב-+y
      // UV של הגליל: v מ-0 (למטה) ל-1 (למעלה) → אחרי הסיבוב v=0 ליד הנחיר? מתקנים:
      const uv = geo.attributes.uv; for (let k = 0; k < uv.count; k++) uv.setY(k, 1 - uv.getY(k));
      const mat = new THREE.ShaderMaterial({
        vertexShader: plumeVS, fragmentShader: plumeFS, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, side: THREE.DoubleSide,
        uniforms: { uThrust: { value: 0 }, uCell: { value: 0.5 }, uLen: { value: 1 }, uTime: { value: 0 } },
      });
      const mesh = new THREE.Mesh(geo, mat);
      mesh.frustumCulled = false;
      // הסילון יושב בתוך הקבוצה של הנחיר (ציר +y = כיוון הפליטה)
      mesh.position.y = 0.06 * nz.size;
      nz.g.add(mesh);
      this.items.push({ mesh, mat, nz, len: 0 });
    }
  }
  update(car, dt, t) {
    for (let i = 0; i < this.items.length; i++) {
      const it = this.items[i], th = car.thr[i];
      const frac = th.nozzle ? Math.min(1, th.F / (th.F0 || (th.F0 = th.nozzle.perf(car.P.fillPressure, 293).F))) : 0;
      // אורך הליבה הנראית ~ שורש הדחף; רוחב לפי יחס הלחצים ביציאה
      const L = 0.3 + 3.4 * Math.sqrt(frac) * it.nz.size;
      it.len += (L - it.len) * Math.min(1, dt * 30);
      const pr = Math.max(1, th.pe / 101325);
      const w = 1 + Math.min(2.2, Math.sqrt(pr) * 0.45);
      it.mesh.scale.set(w * it.nz.size, it.len, w * it.nz.size);
      it.mesh.visible = th.u > 0.01;
      it.mat.uniforms.uThrust.value = Math.min(1, th.u * 1.3);
      it.mat.uniforms.uLen.value = it.len;
      // מרווח תאי ההלם ≈ 1.2·D·√(pe/pa) (קירוב מקובל לסילון תת-מורחב)
      it.mat.uniforms.uCell.value = Math.max(0.08, 1.2 * 0.045 * it.nz.size * Math.sqrt(pr) * 2.2);
      it.mat.uniforms.uTime.value = t;
    }
  }
}

// ---------- מערכת חלקיקים ----------
const partVS = `
attribute float aSize;
attribute float aAlpha;
attribute vec3 aColor;
varying float vAlpha;
varying vec3 vColor;
void main() {
  vAlpha = aAlpha; vColor = aColor;
  vec4 mv = modelViewMatrix * vec4(position, 1.0);
  gl_PointSize = aSize * (800.0 / -mv.z);
  gl_Position = projectionMatrix * mv;
}`;
const partFS = `
uniform sampler2D uTex;
varying float vAlpha;
varying vec3 vColor;
void main() {
  vec4 t = texture2D(uTex, gl_PointCoord);
  // הצללה פשוטה: החלק התחתון של כל ענן כהה יותר, כדי שיראו אותו גם על רקע בהיר
  float shade = mix(1.0, 0.72, gl_PointCoord.y);
  gl_FragColor = vec4(vColor * shade, t.a * vAlpha);
}`;

export class Particles {
  constructor(scene, max = 4000, additive = false) {
    this.max = max; this.n = 0;
    this.p = new Float32Array(max * 3); this.v = new Float32Array(max * 3);
    this.life = new Float32Array(max); this.age = new Float32Array(max);
    this.size0 = new Float32Array(max); this.grow = new Float32Array(max); this.a0 = new Float32Array(max);
    this.drag = new Float32Array(max); this.buoy = new Float32Array(max);
    this.col = new Float32Array(max * 3);
    const g = new THREE.BufferGeometry();
    this.posAttr = new THREE.BufferAttribute(new Float32Array(max * 3), 3).setUsage(THREE.DynamicDrawUsage);
    this.sizeAttr = new THREE.BufferAttribute(new Float32Array(max), 1).setUsage(THREE.DynamicDrawUsage);
    this.alphaAttr = new THREE.BufferAttribute(new Float32Array(max), 1).setUsage(THREE.DynamicDrawUsage);
    this.colAttr = new THREE.BufferAttribute(new Float32Array(max * 3), 3).setUsage(THREE.DynamicDrawUsage);
    g.setAttribute('position', this.posAttr); g.setAttribute('aSize', this.sizeAttr); g.setAttribute('aAlpha', this.alphaAttr); g.setAttribute('aColor', this.colAttr);
    this.mat = new THREE.ShaderMaterial({
      vertexShader: partVS, fragmentShader: partFS, transparent: true, depthWrite: false,
      blending: additive ? THREE.AdditiveBlending : THREE.NormalBlending,
      uniforms: { uTex: { value: softDot() } },
    });
    this.points = new THREE.Points(g, this.mat);
    this.points.frustumCulled = false;
    scene.add(this.points);
  }
  emit(x, y, z, vx, vy, vz, { life = 1.5, size = 0.6, grow = 2, alpha = 0.5, drag = 1.5, buoy = 0.3, color = [1, 1, 1] } = {}) {
    let i = this.n < this.max ? this.n++ : (Math.random() * this.max) | 0;
    this.p[i * 3] = x; this.p[i * 3 + 1] = y; this.p[i * 3 + 2] = z;
    this.v[i * 3] = vx; this.v[i * 3 + 1] = vy; this.v[i * 3 + 2] = vz;
    this.life[i] = life; this.age[i] = 0; this.size0[i] = size; this.grow[i] = grow; this.a0[i] = alpha;
    this.drag[i] = drag; this.buoy[i] = buoy;
    this.col[i * 3] = color[0]; this.col[i * 3 + 1] = color[1]; this.col[i * 3 + 2] = color[2];
  }
  update(dt) {
    const P = this.posAttr.array, S = this.sizeAttr.array, A = this.alphaAttr.array, C = this.colAttr.array;
    let w = 0;
    for (let i = 0; i < this.n; i++) {
      this.age[i] += dt;
      if (this.age[i] >= this.life[i]) continue;
      // דחיסה: מעבירים חלקיק חי למקום w
      if (w !== i) {
        for (const arr of [this.p, this.v, this.col]) { arr[w * 3] = arr[i * 3]; arr[w * 3 + 1] = arr[i * 3 + 1]; arr[w * 3 + 2] = arr[i * 3 + 2]; }
        for (const arr of [this.life, this.age, this.size0, this.grow, this.a0, this.drag, this.buoy]) arr[w] = arr[i];
      }
      const k = Math.exp(-this.drag[w] * dt);
      this.v[w * 3] *= k; this.v[w * 3 + 1] = this.v[w * 3 + 1] * k + this.buoy[w] * dt; this.v[w * 3 + 2] *= k;
      this.p[w * 3] += this.v[w * 3] * dt; this.p[w * 3 + 1] += this.v[w * 3 + 1] * dt; this.p[w * 3 + 2] += this.v[w * 3 + 2] * dt;
      if (this.p[w * 3 + 1] < 0.05 && this.buoy[w] < 0) { this.p[w * 3 + 1] = 0.05; this.v[w * 3 + 1] *= -0.3; }
      const t = this.age[w] / this.life[w];
      P[w * 3] = this.p[w * 3]; P[w * 3 + 1] = this.p[w * 3 + 1]; P[w * 3 + 2] = this.p[w * 3 + 2];
      S[w] = this.size0[w] * (1 + this.grow[w] * t);
      A[w] = this.a0[w] * (t < 0.1 ? t / 0.1 : 1 - (t - 0.1) / 0.9);
      C[w * 3] = this.col[w * 3]; C[w * 3 + 1] = this.col[w * 3 + 1]; C[w * 3 + 2] = this.col[w * 3 + 2];
      w++;
    }
    this.n = w;
    this.points.geometry.setDrawRange(0, w);
    this.posAttr.needsUpdate = true; this.sizeAttr.needsUpdate = true; this.alphaAttr.needsUpdate = true; this.colAttr.needsUpdate = true;
  }
}

// ---------- מנהל האפקטים ----------
export class FX {
  constructor(scene, model) {
    this.scene = scene; this.model = model;
    this.vapor = new Particles(scene, 3500);
    this.sparks = new Particles(scene, 600, true);
    this.plumes = new Plumes(scene, model);
    this.t = 0;
    this._v = new THREE.Vector3(); this._d = new THREE.Vector3(); this._q = new THREE.Quaternion();
  }
  update(car, dt, world) {
    this.t += dt;
    this.plumes.update(car, dt, this.t);
    const q = this._q.set(car.q.x, car.q.y, car.q.z, car.q.w);
    // אדי עיבוי מכל נחיר פעיל
    for (let i = 0; i < car.thr.length; i++) {
      const th = car.thr[i];
      if (th.u < 0.03) continue;
      const nz = this.model.nozzles[i];
      const p = this._v.copy(nz.pos).applyQuaternion(q).add(car.pos);
      const d = this._d.copy(nz.exit).applyQuaternion(q);
      const rate = th.mdot * 3.2 * dt; // חלקיקים לפריים לפי ספיקת המסה
      let n = Math.floor(rate) + (Math.random() < rate % 1 ? 1 : 0);
      for (let k = 0; k < n; k++) {
        const s = 12 + Math.random() * 30;
        const off = 0.5 + Math.random() * 1.5;
        this.vapor.emit(p.x + d.x * off, p.y + d.y * off, p.z + d.z * off,
          car.vel.x * 0.6 + d.x * s + (Math.random() - 0.5) * 4, car.vel.y * 0.6 + d.y * s + (Math.random() - 0.5) * 4, car.vel.z * 0.6 + d.z * s + (Math.random() - 0.5) * 4,
          { life: 1.4 + Math.random() * 1.6, size: 0.55, grow: 5, alpha: 0.5, drag: 2.2, buoy: 0.25, color: [0.93, 0.95, 0.98] });
      }
      // פגיעה בקרקע: ענן שמתפשט לצדדים
      if (d.y < -0.5) {
        const h = p.y;
        if (h < 6) {
          const m = Math.ceil(rate * (1 - h / 6) * 0.8);
          for (let k = 0; k < m; k++) {
            const a = Math.random() * Math.PI * 2, sp = 8 + Math.random() * 14;
            this.vapor.emit(p.x, 0.3, p.z, Math.cos(a) * sp, 0.6, Math.sin(a) * sp,
              { life: 2 + Math.random() * 1.5, size: 1.0, grow: 5, alpha: 0.45 * (1 - h / 6), drag: 1.6, buoy: 0.35, color: [0.9, 0.9, 0.88] });
          }
        }
      }
    }
    // עשן צמיגים
    for (const w of car.wheels) {
      if (!w.contact) continue;
      const slipSpeed = Math.abs(w.kappa) * Math.max(3, car.speed) + Math.abs(Math.tan(w.alpha)) * car.speed;
      if (slipSpeed > 6 && Math.random() < Math.min(0.6, (slipSpeed - 6) * 0.08)) {
        this.vapor.emit(w.cp.x, w.cp.y + 0.2, w.cp.z, (Math.random() - 0.5) * 2, 0.6, (Math.random() - 0.5) * 2,
          { life: 2.2, size: 0.5, grow: 5, alpha: Math.min(0.2, (slipSpeed - 6) * 0.02), drag: 1.2, buoy: 0.4, color: [0.78, 0.78, 0.78] });
      }
    }
    // אבק מהמאווררים בנסיעה איטית על קרקע
    if (car.fan.n > 0.6 && car.contacts > 0 && Math.random() < 0.25) {
      const p = this._v.set(0, -0.1, -2.2).applyQuaternion(q).add(car.pos);
      const d = this._d.set(0, 0, -1).applyQuaternion(q);
      this.vapor.emit(p.x, p.y, p.z, d.x * 9 + car.vel.x * 0.5, 0.5, d.z * 9 + car.vel.z * 0.5,
        { life: 1.2, size: 0.4, grow: 3, alpha: 0.12 * car.fan.n, drag: 2, buoy: 0.2, color: [0.75, 0.68, 0.55] });
    }
    // ניצוצות ופגיעות
    for (const e of car.events) {
      if (e.type === 'scrape') {
        for (let k = 0; k < 2; k++) this.sparks.emit(e.pos.x, e.pos.y, e.pos.z, car.vel.x * 0.7 + (Math.random() - 0.5) * 5, Math.random() * 4, car.vel.z * 0.7 + (Math.random() - 0.5) * 5,
          { life: 0.35 + Math.random() * 0.3, size: 0.08, grow: -0.6, alpha: 1, drag: 0.6, buoy: -9, color: [1, 0.75, 0.3] });
      } else if (e.type === 'impact') {
        for (let k = 0; k < Math.min(40, e.speed * 3); k++) this.vapor.emit(e.pos.x, e.pos.y, e.pos.z, (Math.random() - 0.5) * 6, Math.random() * 3, (Math.random() - 0.5) * 6,
          { life: 1.5, size: 0.6, grow: 3, alpha: 0.3, drag: 2, buoy: 0.3, color: [0.7, 0.66, 0.6] });
      }
    }
    this.vapor.update(dt); this.sparks.update(dt);
  }
}
