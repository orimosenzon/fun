'use strict';
/* גאומטריה, ציור SVG ואנימציה: משותף לכל הפרקים.
   קואורדינטות מסך של SVG: y יורד למטה. */

const RAD = Math.PI / 180, DEG = 180 / Math.PI;

const V = {
  add: (p, q) => [p[0] + q[0], p[1] + q[1]],
  sub: (p, q) => [p[0] - q[0], p[1] - q[1]],
  mul: (p, k) => [p[0] * k, p[1] * k],
  len: p => Math.hypot(p[0], p[1]),
  dist: (p, q) => Math.hypot(p[0] - q[0], p[1] - q[1]),
  dot: (p, q) => p[0] * q[0] + p[1] * q[1],
  cross: (p, q) => p[0] * q[1] - p[1] * q[0],
  norm: p => { const l = Math.hypot(p[0], p[1]) || 1; return [p[0] / l, p[1] / l]; },
  rot: (p, a) => { const c = Math.cos(a), s = Math.sin(a); return [p[0] * c - p[1] * s, p[0] * s + p[1] * c]; },
  lerp: (p, q, t) => [p[0] + (q[0] - p[0]) * t, p[1] + (q[1] - p[1]) * t],
  mid: (p, q) => [(p[0] + q[0]) / 2, (p[1] + q[1]) / 2],
  perp: p => [-p[1], p[0]],
  ang: p => Math.atan2(p[1], p[0]),
  unit: a => [Math.cos(a), Math.sin(a)],
};

const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
function normAng(a) { while (a <= -Math.PI) a += 2 * Math.PI; while (a > Math.PI) a -= 2 * Math.PI; return a; }

// הזווית ב-Q (במעלות) בין הקרניים QP ו-QR
function angleAt(P, Q, R) {
  const u = V.sub(P, Q), w = V.sub(R, Q);
  return Math.acos(clamp(V.dot(u, w) / (V.len(u) * V.len(w) || 1), -1, 1)) * DEG;
}
const orient = (A, B, C) => Math.sign(V.cross(V.sub(B, A), V.sub(C, A)));
function centroid(pts) { let x = 0, y = 0; for (const p of pts) { x += p[0]; y += p[1]; } return [x / pts.length, y / pts.length]; }

function circleCircle(c1, r1, c2, r2) {
  const d = V.dist(c1, c2);
  if (d < 1e-9 || d > r1 + r2 || d < Math.abs(r1 - r2)) return [];
  const a = (r1 * r1 - r2 * r2 + d * d) / (2 * d), h = Math.sqrt(Math.max(0, r1 * r1 - a * a));
  const e = V.mul(V.sub(c2, c1), 1 / d), p = V.add(c1, V.mul(e, a)), n = V.perp(e);
  return [V.add(p, V.mul(n, h)), V.sub(p, V.mul(n, h))];
}
// הישר o + t·d (d וקטור יחידה) מול מעגל: מחזיר את ערכי t ממוינים
function lineCircle(o, d, c, r) {
  const f = V.sub(o, c), b = V.dot(d, f), disc = b * b - (V.dot(f, f) - r * r);
  if (disc < 0) return [];
  const s = Math.sqrt(disc);
  return [-b - s, -b + s];
}
// p + t·d = q + u·e
function lineLine(p, d, q, e) {
  const den = V.cross(d, e);
  if (Math.abs(den) < 1e-9) return null;
  const w = V.sub(q, p);
  return { t: V.cross(w, e) / den, u: V.cross(w, d) / den };
}

const fmt = x => String(Math.round(x * 10) / 10);
const fmtDeg = x => Math.round(x) + '°';
const M = s => `<span class="m" dir="ltr">${s}</span>`;
const shuffle = a => { a = a.slice(); for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; } return a; };
const rf = (a, b) => a + Math.random() * (b - a);
const ri = (a, b) => Math.floor(rf(a, b + 1));

/* ---------- SVG ---------- */
const SVGNS = 'http://www.w3.org/2000/svg';
function S(tag, attrs, parent) {
  const e = document.createElementNS(SVGNS, tag);
  if (attrs) for (const k in attrs) e.setAttribute(k, attrs[k]);
  if (parent) parent.appendChild(e);
  return e;
}
function clearEl(e) { while (e.firstChild) e.removeChild(e.firstChild); }
function svgPt(svg, ev) {
  const p = svg.createSVGPoint(); p.x = ev.clientX; p.y = ev.clientY;
  const q = p.matrixTransform(svg.getScreenCTM().inverse());
  return [q.x, q.y];
}
// גרירה עם pointer events (עכבר ומגע). start יכול להחזיר false כדי לבטל.
function onDrag(svg, target, h) {
  target.addEventListener('pointerdown', ev => {
    if (ev.button > 0) return;
    const p = svgPt(svg, ev);
    if (h.start && h.start(p, ev) === false) return;
    ev.preventDefault();
    try { target.setPointerCapture(ev.pointerId); } catch (e) { /* ok */ }
    target.classList.add('dragging');
    const mv = e => { if (h.move) h.move(svgPt(svg, e), e); };
    const up = () => {
      target.removeEventListener('pointermove', mv);
      target.removeEventListener('pointerup', up);
      target.removeEventListener('pointercancel', up);
      target.classList.remove('dragging');
      if (h.end) h.end();
    };
    target.addEventListener('pointermove', mv);
    target.addEventListener('pointerup', up);
    target.addEventListener('pointercancel', up);
  });
}
const ptsAttr = pts => pts.map(p => p[0].toFixed(1) + ',' + p[1].toFixed(1)).join(' ');
const at = x => (typeof x === 'string' ? { class: x } : (x || {}));
const place = (el, p) => { el.setAttribute('cx', p[0]); el.setAttribute('cy', p[1]); };

const D = {
  seg(g, P, Q, a) { return S('line', Object.assign({ x1: P[0], y1: P[1], x2: Q[0], y2: Q[1] }, at(a)), g); },
  poly(g, pts, a) { return S('polygon', Object.assign({ points: ptsAttr(pts) }, at(a)), g); },
  dot(g, P, r, a) { return S('circle', Object.assign({ cx: P[0], cy: P[1], r: r || 4 }, at(a || 'dot')), g); },
  text(g, P, txt, a) {
    const t = S('text', Object.assign({ x: P[0], y: P[1], 'text-anchor': 'middle', 'dominant-baseline': 'central' }, at(a)), g);
    t.textContent = txt; return t;
  },
  // תווית של קודקוד, מוזזת החוצה ממרכז הצורה
  label(g, P, txt, center, off, a) { const d = V.norm(V.sub(P, center)); return D.text(g, V.add(P, V.mul(d, off || 18)), txt, a || 'lbl'); },
  // תווית של צלע, בצד שרחוק מהמרכז
  sideLabel(g, P, Q, center, txt, a, off) {
    const m = V.mid(P, Q); let n = V.norm(V.perp(V.sub(Q, P)));
    if (V.dot(n, V.sub(m, center)) < 0) n = V.mul(n, -1);
    return D.text(g, V.add(m, V.mul(n, off || 16)), txt, a || 'val');
  },
  // סימוני "צלעות שוות": n קווים קטנים באמצע הצלע
  ticks(g, P, Q, n, a) {
    const m = V.mid(P, Q), d = V.norm(V.sub(Q, P)), nn = V.perp(d);
    for (let i = 0; i < n; i++) {
      const c = V.add(m, V.mul(d, (i - (n - 1) / 2) * 6));
      D.seg(g, V.add(c, V.mul(nn, 8)), V.sub(c, V.mul(nn, 8)), a);
    }
  },
  // קשת זווית ב-Q בין QP ל-QR; n קשתות; fill = טריז צבוע
  arc(g, Q, P, R, n, r, a, fill) {
    const a1 = V.ang(V.sub(P, Q)), d = normAng(V.ang(V.sub(R, Q)) - a1), sweep = d > 0 ? 1 : 0;
    const pt = (rr, ang) => V.add(Q, V.mul(V.unit(ang), rr));
    if (fill) {
      const s = pt(r, a1), e = pt(r, a1 + d);
      S('path', Object.assign({ d: `M${Q[0]},${Q[1]} L${s[0]},${s[1]} A${r},${r} 0 0 ${sweep} ${e[0]},${e[1]} Z` }, at(fill)), g);
    }
    for (let i = 0; i < n; i++) {
      const rr = r + i * 5, s = pt(rr, a1), e = pt(rr, a1 + d);
      S('path', Object.assign({ d: `M${s[0]},${s[1]} A${rr},${rr} 0 0 ${sweep} ${e[0]},${e[1]}`, fill: 'none' }, at(a)), g);
    }
  },
  right(g, Q, P, R, size, a) {
    const u = V.mul(V.norm(V.sub(P, Q)), size), w = V.mul(V.norm(V.sub(R, Q)), size);
    const p1 = V.add(Q, u), p2 = V.add(V.add(Q, u), w), p3 = V.add(Q, w);
    S('path', Object.assign({ d: `M${p1[0]},${p1[1]} L${p2[0]},${p2[1]} L${p3[0]},${p3[1]}`, fill: 'none' }, at(a)), g);
  },
  angLabel(g, Q, P, R, r, txt, a) {
    const b = V.norm(V.add(V.norm(V.sub(P, Q)), V.norm(V.sub(R, Q))));
    return D.text(g, V.add(Q, V.mul(b, r)), txt, a || 'val');
  },
};

/* ---------- אנימציה ---------- */
const ease = t => (t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2);
function animate(ms, fn, done) {
  const t0 = performance.now();
  function step(now) {
    const t = Math.min(1, (now - t0) / ms);
    fn(t);
    if (t < 1) requestAnimationFrame(step); else if (done) done();
  }
  requestAnimationFrame(step);
}
// תנועה קשיחה ממשולש from למשולש to (חופפים, לפי סדר הקודקודים):
// קודם היפוך אם צריך (כמו להפוך קלף), ואחר כך סיבוב והזזה ביחד.
function rigidPlan(from, to) {
  const cf = centroid(from), ct = centroid(to);
  const flip = orient(from[0], from[1], from[2]) !== orient(to[0], to[1], to[2]);
  const loc = from.map(p => V.sub(p, cf));
  const locF = flip ? loc.map(p => [-p[0], p[1]]) : loc;
  const phi = normAng(V.ang(V.sub(to[0], ct)) - V.ang(locF[0]));
  const fn = t => {
    let sx = 1, r = t;
    if (flip) { if (t < 0.35) { sx = 1 - 2 * ease(t / 0.35); r = 0; } else { sx = -1; r = (t - 0.35) / 0.65; } }
    const e = ease(r), c = V.lerp(cf, ct, e), a = phi * e;
    return loc.map(p => V.add(c, V.rot([p[0] * sx, p[1]], a)));
  };
  fn.flip = flip;
  return fn;
}

function celebrate() {
  const box = document.createElement('div'); box.className = 'confetti';
  const colors = ['#2563eb', '#ea580c', '#16a34a', '#db2777', '#d97706', '#7c3aed'];
  for (let i = 0; i < 44; i++) {
    const s = document.createElement('i');
    s.style.left = Math.random() * 100 + 'vw';
    s.style.background = colors[i % colors.length];
    s.style.animationDelay = Math.random() * 0.35 + 's';
    s.style.animationDuration = (1.4 + Math.random() * 0.8) + 's';
    box.appendChild(s);
  }
  document.body.appendChild(box);
  setTimeout(() => box.remove(), 2600);
}
