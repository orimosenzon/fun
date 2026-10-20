// לופט (loft): משטח חלק שנמתח דרך רצף של חתכים. כל חתך הוא סופר-אליפסה (מלבן מעוגל עד אליפסה),
// עם חצי-רוחב שונה מלפנים ומאחור, כדי לקבל צורות של שריר, חזה או שוק.
import * as THREE from 'three';

// ring: { z, a, ab?, b, bb?, n?, x?, y?, tilt?: [kx, ky] }  a = חצי-רוחב קדימה (x+), ab = אחורה, b = לצדדים (y+), bb = y−
// axis: 'z' (חתכים לאורך z) או 'x' (לאורך x, למשל כף רגל): אז a הוא חצי-הגובה (z) ו-b חצי-הרוחב (y)
// capBottom/capTop: מכסה על החתך הראשון/האחרון ברשימה
export function loft(rings, { seg = 40, capTop = true, capBottom = true, axis = 'z' } = {}) {
  const pos = [], idx = [];
  const nr = rings.length;
  for (let i = 0; i < nr; i++) {
    const R = rings[i], n = R.n ?? 2.4, e = 2 / n;
    for (let j = 0; j < seg; j++) {
      const t = (j / seg) * Math.PI * 2, c = Math.cos(t), s = Math.sin(t);
      const ca = Math.sign(c) * Math.abs(c) ** e, sa = Math.sign(s) * Math.abs(s) ** e;
      const ax = c >= 0 ? R.a : (R.ab ?? R.a), bx = s >= 0 ? R.b : (R.bb ?? R.b);
      const u = (R.x || 0) + ax * ca, v = (R.y || 0) + bx * sa;
      // tilt: חתך משופע (למשל פתח רגל גבוה בצד החיצוני). z משתנה עם הזווית
      const zz = R.z + (R.tilt ? R.tilt[0] * ca + R.tilt[1] * sa : 0);
      if (axis === 'z') pos.push(u, v, zz);
      else pos.push(R.z, v, u);   // axis x: R.z הוא המיקום לאורך x, u הוא גובה
    }
  }
  // כיוון הנורמלים תלוי בכיוון שבו החתכים מתקדמים: עולים או יורדים לאורך הציר
  const asc = rings[nr - 1].z >= rings[0].z;
  const outward = (axis === 'z') === asc;
  for (let i = 0; i < nr - 1; i++) for (let j = 0; j < seg; j++) {
    const a = i * seg + j, b = i * seg + (j + 1) % seg, c = (i + 1) * seg + j, d = (i + 1) * seg + (j + 1) % seg;
    if (outward) idx.push(a, b, d, a, d, c); else idx.push(a, d, b, a, c, d);
  }
  const cap = (ri, last) => {
    const R = rings[ri], ci = pos.length / 3;
    if (axis === 'z') pos.push(R.x || 0, R.y || 0, R.z); else pos.push(R.z, R.y || 0, R.x || 0);
    for (let j = 0; j < seg; j++) {
      const a = ri * seg + j, b = ri * seg + (j + 1) % seg;
      // המכסה של החתך האחרון פונה "קדימה" בכיוון ההתקדמות, והראשון אחורה
      const flip = outward === last;
      if (flip) idx.push(ci, a, b); else idx.push(ci, b, a);
    }
  };
  if (capBottom) cap(0, false);
  if (capTop) cap(nr - 1, true);
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  g.setIndex(idx);
  g.computeVertexNormals();
  return g;
}

// חתכים עגולים לאורך ציר z: [[z, r], ...]
export function tube(prof, opts) {
  return loft(prof.map(([z, r, r2]) => ({ z, a: r, b: r2 ?? r, n: 2 })), opts);
}

// קפסולה בין שתי נקודות (לאצבעות, צינורות)
export function capsuleBetween(p0, p1, r, mat) {
  const d = new THREE.Vector3().subVectors(p1, p0);
  const len = d.length();
  const g = new THREE.CapsuleGeometry(r, Math.max(1e-4, len), 6, 12);
  const m = new THREE.Mesh(g, mat);
  m.position.copy(p0).addScaledVector(d, 0.5);
  m.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), d.normalize());
  return m;
}
