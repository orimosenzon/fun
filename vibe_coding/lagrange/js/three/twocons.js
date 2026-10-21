// 3D lab: maximize a linear f on the circle where a sphere (g1) meets a plane (g2).
// At the optimum ∇f lies in the plane spanned by ∇g1 and ∇g2: one multiplier per constraint.
import { View3D, THREE, V, tube, ball, arrow, patch, translucent, axesLines } from './view3d.js';
import { $, fmt, put, range } from '../ui.js';

const R = 2;
const N = V(1, 0.6, 1.6).normalize();   // plane normal
const D = 0.8;                           // plane: N·x = D

export function initTwoCons() {
  const view = new View3D($('#two-3d'), { cam: [5.4, -6.4, 3.8], target: [0, 0, 0.3] });
  const s = { th: 0.6, az: 40, el: 25 };
  const c0 = N.clone().multiplyScalar(D), rho = Math.sqrt(R * R - D * D);
  const U = V(0, 0, 1).cross(N).normalize(), W = N.clone().cross(U).normalize();
  const P = th => c0.clone().addScaledVector(U, rho * Math.cos(th)).addScaledVector(W, rho * Math.sin(th));
  const Tan = th => U.clone().multiplyScalar(-Math.sin(th)).addScaledVector(W, Math.cos(th));
  const objective = () => {
    const a = s.az * Math.PI / 180, e = s.el * Math.PI / 180;
    return V(Math.cos(e) * Math.cos(a), Math.cos(e) * Math.sin(a), Math.sin(e));
  };

  view.scene.add(axesLines(view, [-2.6, 2.6, -2.6, 2.6, -2.6, 2.6]));
  const sphere = new THREE.Mesh(new THREE.SphereGeometry(R, 48, 32), translucent(view, '--c-g', 0.1));
  const wire = new THREE.LineSegments(new THREE.WireframeGeometry(new THREE.SphereGeometry(R, 18, 12)), view.mat('--c-g', { kind: 'line', transparent: true, opacity: 0.08 }));
  const plane = patch(c0, N, 2.5, translucent(view, '--c-g', 0.12), true);
  const circle = [];
  for (let i = 0; i < 160; i++) circle.push(P(i / 160 * 2 * Math.PI));
  view.scene.add(sphere, wire, plane, tube(circle, 0.035, view.mat('--c-g'), true));
  const l1 = view.label('כדור (g₁)', 'g'); l1.position.set(-1.2, -1.2, 1.6);
  const l2 = view.label('מישור (g₂)', 'g'); l2.position.copy(c0.clone().addScaledVector(U, 2.7));
  view.scene.add(l1, l2);

  const dyn = view.layer();
  const mF = view.mat('--c-f'), mG = view.mat('--c-g'), mT = view.mat('--c-tan'), mInk = view.mat('--ink');
  const mNormal = translucent(view, '--c-lam', 0.22), mLevel = translucent(view, '--c-f', 0.12);

  function build() {
    dyn.reset();
    const p = P(s.th), t = Tan(s.th), a = objective();
    const g1 = p.clone().multiplyScalar(2), g2 = N.clone();
    // least squares a ≈ λ1 g1 + λ2 g2
    const A11 = g1.dot(g1), A12 = g1.dot(g2), A22 = g2.dot(g2), r1 = g1.dot(a), r2 = g2.dot(a), det = A11 * A22 - A12 * A12;
    const lam1 = (r1 * A22 - r2 * A12) / det, lam2 = (A11 * r2 - A12 * r1) / det;
    const tc = a.dot(t);

    dyn.add(patch(p, t, 1.05, mNormal, true));
    dyn.add(patch(p, a, 2.4, mLevel));
    dyn.add(arrow(p, g1.clone().normalize().multiplyScalar(0.95), mG, 0.03));
    dyn.add(arrow(p, g2.clone().multiplyScalar(0.95), mG, 0.03));
    dyn.add(arrow(p, a.clone().multiplyScalar(1.3), mF, 0.04));
    if (Math.abs(tc) > 0.02) dyn.add(arrow(p, t.clone().multiplyScalar(tc * 1.3), mT, 0.04));
    dyn.add(ball(p, 0.08, mInk));
    const lf = view.label('∇f', 'f'); lf.position.copy(p.clone().addScaledVector(a, 1.5));
    const lg1 = view.label('∇g₁', 'g'); lg1.position.copy(p.clone().addScaledVector(g1.clone().normalize(), 1.12));
    const lg2 = view.label('∇g₂', 'g'); lg2.position.copy(p.clone().addScaledVector(g2, 1.12));
    dyn.add(lf, lg1, lg2);
    view.request();

    put('two-f', fmt(a.dot(p), 3));
    put('two-l1', fmt(lam1, 3));
    put('two-l2', fmt(lam2, 3));
    put('two-tan', fmt(Math.abs(tc), 3));
    const st = $('#two-status');
    if (Math.abs(tc) < 0.01) {
      st.className = 'status ok';
      st.textContent = `נקודה קריטית: ∇f נמצא בתוך המישור הסגול, ולכן ∇f = λ₁∇g₁ + λ₂∇g₂ עם λ₁ = ${fmt(lam1, 3)} ו-λ₂ = ${fmt(lam2, 3)}.`;
    } else {
      st.className = 'status';
      st.textContent = 'החץ הכחול בולט מחוץ למישור הסגול. החלק הבולט (ירוק) מצביע לאורך המעגל, ולכן כאן עוד אפשר לשפר.';
    }
  }

  const crit = () => {
    const a = objective();
    return Math.atan2(a.dot(W), a.dot(U));   // maximum; minimum is +π
  };
  const thSlider = range('two-th', v => { s.th = v * Math.PI / 180; build(); }, v => v + '°');
  const jump = th => { s.th = ((th % (2 * Math.PI)) + 2 * Math.PI) % (2 * Math.PI); thSlider.set(Math.round(s.th * 180 / Math.PI)); build(); };
  $('#two-max').addEventListener('click', () => jump(crit()));
  $('#two-min').addEventListener('click', () => jump(crit() + Math.PI));
  range('two-az', v => { s.az = v; build(); }, v => v + '°');
  range('two-el', v => { s.el = v; build(); }, v => v + '°');
  let anim = null;
  $('#two-walk').addEventListener('click', () => {
    if (anim) { cancelAnimationFrame(anim); anim = null; return; }
    const step = () => { s.th = (s.th + 0.012) % (2 * Math.PI); thSlider.set(Math.round(s.th * 180 / Math.PI)); build(); anim = requestAnimationFrame(step); };
    step();
  });
  build();
}
