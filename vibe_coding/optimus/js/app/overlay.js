// שכבות מידע בתלת-ממד: מצולע התמיכה על הרצפה, נקודת ה-ZMP, היטל מרכז המסה, ה-DCM (נקודת הלכידה),
// הצעדים המתוכננים, וחיצי כוח התגובה של הרצפה בכל כף רגל. הכול במערכת הצירים של MuJoCo (z למעלה).
import * as THREE from 'three';
import { G } from '../robot/spec.js';

const Z = 0.004;
export const OVERLAY_COLORS = { zmp: 0xff4d3d, com: 0xffffff, dcm: 0x2f8cff, poly: 0x23c483, step: 0xffc23d, force: 0xffd23d };

function disc(r, color, opacity = 1) {
  const m = new THREE.Mesh(new THREE.CircleGeometry(r, 24), new THREE.MeshBasicMaterial({ color, transparent: true, opacity, depthTest: false }));
  m.renderOrder = 10;
  return m;
}
function ring(r, w, color) {
  const m = new THREE.Mesh(new THREE.RingGeometry(r - w, r, 32), new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.95, depthTest: false }));
  m.renderOrder = 11;
  return m;
}

export class Overlay {
  constructor(parent, app) {
    this.app = app;
    this.g = new THREE.Group();
    parent.add(this.g);
    this.visible = true;
    // מצולע תמיכה
    this.polyGeo = new THREE.BufferGeometry();
    this.polyFill = new THREE.Mesh(this.polyGeo, new THREE.MeshBasicMaterial({ color: OVERLAY_COLORS.poly, transparent: true, opacity: 0.18, depthTest: false, side: THREE.DoubleSide }));
    this.polyFill.renderOrder = 9;
    this.polyLine = new THREE.LineLoop(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ color: OVERLAY_COLORS.poly, depthTest: false }));
    this.polyLine.renderOrder = 10;
    this.g.add(this.polyFill, this.polyLine);
    this.zmp = disc(0.022, OVERLAY_COLORS.zmp);
    this.com = ring(0.03, 0.009, OVERLAY_COLORS.com);
    this.dcm = disc(0.018, OVERLAY_COLORS.dcm);
    this.dcmRing = ring(0.034, 0.006, OVERLAY_COLORS.dcm);
    this.g.add(this.zmp, this.com, this.dcm, this.dcmRing);
    // קו אנכי ממרכז המסה לרצפה
    this.comLine = new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3()]), new THREE.LineDashedMaterial({ color: 0xffffff, dashSize: 0.03, gapSize: 0.02, depthTest: false, transparent: true, opacity: 0.7 }));
    this.comLine.renderOrder = 10;
    this.comBall = new THREE.Mesh(new THREE.SphereGeometry(0.022, 16, 12), new THREE.MeshBasicMaterial({ color: 0xffffff, depthTest: false }));
    this.comBall.renderOrder = 12;
    this.g.add(this.comLine, this.comBall);
    // צעדים מתוכננים
    this.steps = [];
    for (let i = 0; i < 5; i++) {
      const shape = new THREE.Shape();
      const hx = G.footLen / 2, hy = G.footW / 2, r = 0.035;
      shape.moveTo(-hx + r, -hy); shape.lineTo(hx - r, -hy); shape.quadraticCurveTo(hx, -hy, hx, -hy + r); shape.lineTo(hx, hy - r);
      shape.quadraticCurveTo(hx, hy, hx - r, hy); shape.lineTo(-hx + r, hy); shape.quadraticCurveTo(-hx, hy, -hx, hy - r); shape.lineTo(-hx, -hy + r); shape.quadraticCurveTo(-hx, -hy, -hx + r, -hy);
      const pts = shape.getPoints(6);
      const line = new THREE.LineLoop(new THREE.BufferGeometry().setFromPoints(pts.map(p => new THREE.Vector3(p.x, p.y, 0))), new THREE.LineBasicMaterial({ color: OVERLAY_COLORS.step, transparent: true, opacity: 1 - i * 0.18, depthTest: false }));
      line.renderOrder = 10;
      this.g.add(line); this.steps.push(line);
    }
    // חיצי כוח
    this.arrows = ['l', 'r'].map(() => {
      const a = new THREE.ArrowHelper(new THREE.Vector3(0, 0, 1), new THREE.Vector3(), 0.3, OVERLAY_COLORS.force, 0.05, 0.03);
      a.line.material.depthTest = false; a.cone.material.depthTest = false; a.renderOrder = 13;
      this.g.add(a); return a;
    });
    // יעד ידיים
    this.handTargets = ['l', 'r'].map(() => {
      const m = new THREE.Mesh(new THREE.SphereGeometry(0.022, 14, 10), new THREE.MeshBasicMaterial({ color: 0x00e5ff, transparent: true, opacity: 0.75, depthTest: false }));
      m.renderOrder = 12; this.g.add(m); return m;
    });
    // יעד ניווט
    this.goal = ring(0.12, 0.02, 0xffc23d); this.g.add(this.goal); this.goal.visible = false;
    // חץ דחיפה
    this.pushArrow = new THREE.ArrowHelper(new THREE.Vector3(1, 0, 0), new THREE.Vector3(), 0.5, 0xff3b30, 0.12, 0.07);
    this.pushArrow.visible = false; this.g.add(this.pushArrow);
  }
  setVisible(v) { this.visible = v; }
  dispose() { this.g.parent?.remove(this.g); }

  update() {
    const app = this.app, r = app.robot, v = r.loco.viz;
    this.g.visible = true;
    const showInfo = this.visible;
    for (const o of [this.polyFill, this.polyLine, this.zmp, this.com, this.dcm, this.dcmRing, this.comLine, this.comBall, ...this.steps, ...this.arrows]) o.visible = showInfo;
    if (showInfo && v.poly) {
      const pts = v.poly.map(p => new THREE.Vector3(p[0], p[1], Z));
      // מספר הקודקודים משתנה (רגל אחת/שתיים), אז בונים גיאומטריה חדשה
      this.polyLine.geometry.dispose();
      this.polyLine.geometry = new THREE.BufferGeometry().setFromPoints(pts);
      const shape = new THREE.Shape(v.poly.map(p => new THREE.Vector2(p[0], p[1])));
      this.polyFill.geometry.dispose();
      this.polyFill.geometry = new THREE.ShapeGeometry(shape);
      this.polyFill.position.z = Z * 0.5;
      this.zmp.position.set(v.zmp[0], v.zmp[1], Z * 2);
      this.dcm.position.set(v.xi[0], v.xi[1], Z * 2.2);
      this.dcmRing.position.copy(this.dcm.position);
      const c = r.wbc.com();
      this.com.position.set(c[0], c[1], Z * 2.4);
      this.comBall.position.set(c[0], c[1], c[2]);
      this.comLine.geometry.setFromPoints([new THREE.Vector3(c[0], c[1], c[2]), new THREE.Vector3(c[0], c[1], Z)]);
      this.comLine.computeLineDistances();
      const plan = (v.plan || []).slice(0, this.steps.length);
      this.steps.forEach((s, i) => {
        const st = plan[i];
        s.visible = !!st && app.robot.loco.mode === 'walk';
        if (st) { s.position.set(st.p[0], st.p[1], (st.p[2] || 0) + Z * 3); s.rotation.z = st.yaw; }
      });
      const ws = r.wbc.out.wrenches || [];
      this.arrows.forEach((a, i) => {
        const w = ws[i];
        a.visible = !!w;
        if (!w) return;
        const F = new THREE.Vector3(w.w[0], w.w[1], w.w[2]);
        const mag = F.length();
        // מרכז הלחץ: Tx = py·Fz, Ty = −px·Fz במערכת של כף הרגל
        const R = w.R, Tl = [R[0] * w.w[3] + R[3] * w.w[4] + R[6] * w.w[5], R[1] * w.w[3] + R[4] * w.w[4] + R[7] * w.w[5]];
        const fz = Math.max(1, R[2] * w.w[0] + R[5] * w.w[1] + R[8] * w.w[2]);
        const px = -Tl[1] / fz, py = Tl[0] / fz;
        const o = new THREE.Vector3(w.p[0] + R[0] * px + R[1] * py, w.p[1] + R[3] * px + R[4] * py, w.p[2] + 0.002);
        a.position.copy(o);
        a.setDirection(F.clone().normalize());
        a.setLength(Math.max(0.05, mag / 1000), 0.05, 0.03);
      });
    }
    // יעדי ידיים
    ['l', 'r'].forEach((s, i) => {
      const h = r.handTargets[s];
      this.handTargets[i].visible = !!h && showInfo;
      if (h) this.handTargets[i].position.set(...h.pos);
    });
    const goal = app.tasks?.goal;
    this.goal.visible = !!goal;
    if (goal) this.goal.position.set(goal[0], goal[1], Z * 3);
    const pa = app.pushViz;
    this.pushArrow.visible = !!pa && pa.t > 0;
    if (pa && pa.t > 0) {
      pa.t -= 1 / 60;
      const dir = new THREE.Vector3(...pa.F).normalize();
      this.pushArrow.position.set(pa.p[0] - dir.x * 0.6, pa.p[1] - dir.y * 0.6, pa.p[2] - dir.z * 0.6);
      this.pushArrow.setDirection(dir); this.pushArrow.setLength(0.55, 0.13, 0.08);
    }
  }
}
