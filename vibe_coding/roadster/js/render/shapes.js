// כלים לבניית מרכבים: פרופיל צד → אקסטרוזיה עם שיפוע → עיוות למבט על (התחדדות החרטום).
import * as THREE from 'three';

// pts: [[z, y], ...] בסדר נגד כיוון השעון, במערכת שבה z קדימה.
// מחזיר BufferGeometry שבו x = רוחב (מרוכז), y = גובה, z = אורך.
export function profileBody(pts, width, { bevel = 0.08, bevelSegs = 4, taper = null, roundTop = 0, curveSegs = 0 } = {}) {
  const shape = new THREE.Shape();
  shape.moveTo(pts[0][0], pts[0][1]);
  for (let i = 1; i < pts.length; i++) {
    const p = pts[i];
    if (p.length === 4) shape.quadraticCurveTo(p[2], p[3], p[0], p[1]);
    else shape.lineTo(p[0], p[1]);
  }
  shape.closePath();
  const depth = Math.max(0.01, width - bevel * 2);
  const geo = new THREE.ExtrudeGeometry(shape, {
    depth, bevelEnabled: bevel > 0, bevelThickness: bevel, bevelSize: bevel * 0.9, bevelSegments: bevelSegs, curveSegments: curveSegs || 12, steps: 1,
  });
  // הצורה נבנתה במישור (x=z, y=y) ומוחצנת לאורך z → מסובבים כך שהאורך יהיה z והרוחב x
  geo.translate(0, 0, -depth / 2);
  const pos = geo.attributes.position;
  for (let i = 0; i < pos.count; i++) {
    const zz = pos.getX(i), yy = pos.getY(i), xx = pos.getZ(i);
    let x = -xx; // היפוך ציר אחד שומר על כיוון הסיבוב של המשולשים (אחרת הנורמלים פונים פנימה)
    if (taper) {
      const t = taper(zz, yy); // מקדם רוחב 0..1
      x *= t;
    }
    let y = yy;
    if (roundTop > 0) {
      // מעגל את קצוות הגג: ככל שמתקרבים לצד, הגובה יורד מעט
      const r = Math.abs(xx) / (width / 2);
      y -= roundTop * Math.pow(r, 4) * Math.max(0, yy);
    }
    pos.setXYZ(i, x, y, zz);
  }
  geo.computeVertexNormals();
  return geo;
}

// גלגל: צמיג בפרופיל מעוגל (lathe) + חישוק
export function tireGeometry(radius, width, segs = 36) {
  const pts = [];
  const r0 = radius * 0.66, sw = width / 2;
  const prof = [
    [r0, -sw * 0.92], [radius - 0.04, -sw], [radius - 0.012, -sw * 0.86], [radius, -sw * 0.6], [radius, sw * 0.6],
    [radius - 0.012, sw * 0.86], [radius - 0.04, sw], [r0, sw * 0.92],
  ];
  for (const [r, y] of prof) pts.push(new THREE.Vector2(r, y));
  const g = new THREE.LatheGeometry(pts, segs);
  g.rotateZ(Math.PI / 2); // ציר הסיבוב לאורך x
  return g;
}

export function capsule(r, len, segs = 16) {
  const g = new THREE.CapsuleGeometry(r, len, 6, segs);
  return g;
}

// צינור לאורך נקודות
export function tube(points, r = 0.02, segs = 40) {
  const curve = new THREE.CatmullRomCurve3(points.map(p => new THREE.Vector3(...p)));
  return new THREE.TubeGeometry(curve, segs, r, 8, false);
}
